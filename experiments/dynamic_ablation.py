"""Ablate the handcrafted Dynamic forecaster, alone and as a shortcut for Dual-LSTM.

Each variant is evaluated twice:
1) baseline alone: the exact causal Dynamic runner with selected components removed;
2) baseline + Dual-LSTM: baseline is ONLY an additive shortcut. The neural model
   sees raw net-load histories (recent 56 days + last 2 same weekdays) and calendar,
   predicts q0.8 of the shortcut residual, and is never given baseline features.

This directly tests which handcrafted components remain necessary once the learned
temporal residual model is present.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("mcm_code")
pkg.__path__ = [str(ROOT / "code")]
pkg.__package__ = "mcm_code"
sys.modules["mcm_code"] = pkg

from mcm_code.common import dataset as common
from mcm_code.common import physical
from mcm_code.result.problem2.solve_problem2_yesterday_dynamic import (
    FINAL_DYNAMIC_CONFIG,
    YESTERDAY_BASELINE_CONFIG,
    choose_candidate,
    run_config,
)

# Reuse the tested Dual-LSTM implementation from the preceding experiment.
spec = importlib.util.spec_from_file_location(
    "resnet_exp", ROOT / "experiments" / "lstm_resnet_ablation.py"
)
resnet = importlib.util.module_from_spec(spec)
sys.modules["resnet_exp"] = resnet
assert spec.loader is not None
spec.loader.exec_module(resnet)

SEEDS = (42, 2026, 3407)
TAU = 0.8
HIDDEN = 32
MAX_EPOCHS = 80
PATIENCE = 8
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
OUTPUT_START_DAY = 31

VARIANTS = (
    "full",
    "no_bias",
    "no_load_season",
    "no_pv_season",
    "no_selector_yesterday",
    "fixed70",
    "no_soc_state",
    "no_reserve",
    "stripped_forecast",
    "minimal_selector_fixed70",
    "minimal_yesterday_fixed70",
)

def variant_config(name):
    c = FINAL_DYNAMIC_CONFIG
    if name == "full":
        return c
    if name == "no_bias":
        return replace(c, name=name, bias_gain=0.0)
    if name == "no_load_season":
        return replace(c, name=name, load_seasonal_strength=0.0)
    if name == "no_pv_season":
        return replace(c, name=name, seasonal_strength=0.0)
    if name == "no_selector_yesterday":
        return replace(c, name=name, candidate_mode="yesterday_only")
    if name == "fixed70":
        return replace(
            c, name=name, delta_mode="fixed", default_delta=70.0,
            reserve_in_milp=True
        )
    if name == "no_soc_state":
        return replace(c, name=name, state_multiplier_strength=0.0)
    if name == "no_reserve":
        return replace(
            c, name=name, delta_mode="fixed", default_delta=0.0,
            reserve_in_milp=True
        )
    if name == "stripped_forecast":
        return replace(
            c, name=name, bias_gain=0.0, load_seasonal_strength=0.0,
            seasonal_strength=0.0
        )
    if name == "minimal_selector_fixed70":
        return replace(
            c, name=name, bias_gain=0.0, load_seasonal_strength=0.0,
            seasonal_strength=0.0, delta_mode="fixed", default_delta=70.0,
            reserve_in_milp=True
        )
    if name == "minimal_yesterday_fixed70":
        return replace(
            c, name=name, candidate_mode="yesterday_only", bias_gain=0.0,
            load_seasonal_strength=0.0, seasonal_strength=0.0,
            delta_mode="fixed", default_delta=70.0, reserve_in_milp=True
        )
    raise ValueError(name)


class ShortcutQ80(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recent_lstm = torch.nn.LSTM(144, HIDDEN, batch_first=True)
        self.weekly_lstm = torch.nn.LSTM(144, HIDDEN, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * HIDDEN + 9, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 144),
        )

    @staticmethod
    def encode(lstm, seqs):
        lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = lstm(packed)
        return h[-1]

    def forward(self, recent, weekly, cal):
        hr = self.encode(self.recent_lstm, recent)
        hw = self.encode(self.weekly_lstm, weekly)
        return self.head(torch.cat([hr, hw, cal], dim=1))


def pinball(pred, target):
    e = target - pred
    return torch.maximum(TAU * e, (TAU - 1.0) * e).mean()


def month_ranges(dates):
    out = []
    for m in range(2, 13):
        s = int(np.flatnonzero((dates.month == m) & (dates.day == 1))[0])
        e = int(np.flatnonzero((dates.month == m + 1) & (dates.day == 1))[0]) if m < 12 else len(dates)
        out.append((m, s, e))
    return out


def build_examples(days, net, residual, dates, nmean, nstd, rmean, rstd):
    rows = []
    for d0 in days:
        d = int(d0)
        ri = np.arange(max(0, d - 56), d, dtype=int)
        wd = dates[d].dayofweek
        wi = np.flatnonzero(dates[:d].dayofweek == wd)[-2:]
        if len(ri) == 0 or len(wi) == 0:
            continue
        recent = torch.tensor((net[ri] - nmean) / nstd, dtype=torch.float32)
        weekly = torch.tensor((net[wi] - nmean) / nstd, dtype=torch.float32)
        cal = torch.tensor(resnet.calendar_feature(d, dates), dtype=torch.float32)
        y = torch.tensor((residual[d] - rmean) / rstd, dtype=torch.float32)
        rows.append((recent, weekly, cal, y))
    return rows


def batchify(rows, idx):
    b = [rows[int(i)] for i in idx]
    return (
        [x[0] for x in b],
        [x[1] for x in b],
        torch.stack([x[2] for x in b]),
        torch.stack([x[3] for x in b]),
    )


def fit_month(net, residual, dates, start, seed):
    fit_end = start - 7
    train_days = np.arange(7, fit_end, dtype=int)
    val_days = np.arange(fit_end, start, dtype=int)

    nmean = net[:fit_end].mean(0)
    nstd = np.maximum(net[:fit_end].std(0), 1e-3)
    rmean = residual[:fit_end].mean(0)
    rstd = np.maximum(residual[:fit_end].std(0), 1e-3)

    tr = build_examples(train_days, net, residual, dates, nmean, nstd, rmean, rstd)
    va = build_examples(val_days, net, residual, dates, nmean, nstd, rmean, rstd)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ShortcutQ80()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + start)

    best = float("inf")
    state = None
    best_epoch = 0
    bad = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(tr), generator=gen)
        for pos in range(0, len(order), BATCH):
            ids = order[pos:pos+BATCH]
            recent, weekly, cal, y = batchify(tr, ids)
            opt.zero_grad(set_to_none=True)
            p = model(recent, weekly, cal)
            loss = pinball(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for pos in range(0, len(va), BATCH):
                ids = np.arange(pos, min(pos+BATCH, len(va)))
                recent, weekly, cal, y = batchify(va, ids)
                total += float(pinball(model(recent, weekly, cal), y)) * len(ids)
        vl = total / len(va)
        if vl < best - 1e-6:
            best = vl
            best_epoch = epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    assert state is not None
    model.load_state_dict(state)
    model.eval()
    return model, nmean, nstd, rmean, rstd, best_epoch, best


def predict_one(model, nmean, nstd, rmean, rstd, net, base, dates, d):
    ri = np.arange(max(0, d - 56), d, dtype=int)
    wi = np.flatnonzero(dates[:d].dayofweek == dates[d].dayofweek)[-2:]
    recent = [torch.tensor((net[ri] - nmean) / nstd, dtype=torch.float32)]
    weekly = [torch.tensor((net[wi] - nmean) / nstd, dtype=torch.float32)]
    cal = torch.tensor(resnet.calendar_feature(d, dates)[None], dtype=torch.float32)
    with torch.no_grad():
        z = model(recent, weekly, cal).cpu().numpy()[0]
    correction = rmean + rstd * np.clip(z, -5.0, 5.0)
    return np.clip(base[d] + correction, -2500.0, 3000.0)


def build_predictions(net, base, dates, seed):
    residual = net - base
    pred = np.full_like(net, np.nan)
    meta = []
    train_seconds = 0.0
    for month, s, e in month_ranges(dates):
        t0 = time.perf_counter()
        model, nmean, nstd, rmean, rstd, ep, vl = fit_month(
            net, residual, dates, s, seed
        )
        train_seconds += time.perf_counter() - t0
        for d in range(s, e):
            pred[d] = predict_one(
                model, nmean, nstd, rmean, rstd, net, base, dates, d
            )
        meta.append({"month": month, "best_epoch": ep, "val_loss": vl})
    return pred, meta, train_seconds


def simulate(pred, net, prices, initial_soc):
    soc = float(initial_soc)
    pc = ec = ek = 0.0
    days = 0
    solver = 0.0
    t0 = time.perf_counter()
    for d in range(OUTPUT_START_DAY, len(net)):
        plan = resnet.solve_net_day(pred[d], prices, soc)
        solver += float(plan.solver_seconds)
        gap = net[d] + plan.charge - plan.x - plan.discharge
        emergency = np.maximum(gap, 0.0)
        pc += float(np.dot(prices, plan.x))
        ec += float(np.dot(5.0 * prices, emergency))
        ek += float(emergency.sum())
        days += int(emergency.sum() > 1e-7)
        soc = float(plan.soc[-1])
    wall = time.perf_counter() - t0
    actual = net[OUTPUT_START_DAY:]
    p = pred[OUTPUT_START_DAY:]
    err = actual - p
    return {
        "forecast_mae_kwh": float(np.mean(np.abs(err))),
        "forecast_rmse_kwh": float(np.sqrt(np.mean(err**2))),
        "coverage": float(np.mean(actual <= p)),
        "underprediction_mean_kwh": float(np.mean(np.maximum(err, 0.0))),
        "planned_cost_yuan": pc,
        "emergency_purchase_kwh": ek,
        "emergency_cost_yuan": ec,
        "total_cost_yuan": pc + ec,
        "days_with_emergency": days,
        "final_soc_kwh": soc,
        "dispatch_wall_seconds": wall,
        "dispatch_solver_seconds": solver,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)

    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)
    cfg = variant_config(args.variant)

    # Exact reproduction gate for the frozen reference environment.
    y = run_config(data, cold_load, cold_pv, YESTERDAY_BASELINE_CONFIG, 20.0)
    expected_y = 22823747.29284119
    if abs(float(y.summary["total_cost_yuan"]) - expected_y) > 1e-4:
        raise RuntimeError("Yesterday reproduction gate failed")

    # Standalone ablated Dynamic.
    t0 = time.perf_counter()
    base_result = run_config(data, cold_load, cold_pv, cfg, 20.0)
    baseline_wall = time.perf_counter() - t0
    base_summary = {
        k: base_result.summary[k]
        for k in (
            "planned_cost_yuan","emergency_purchase_kwh","emergency_cost_yuan",
            "total_cost_yuan","days_with_emergency","mean_selected_delta_kwh_per_interval",
            "selected_yesterday_days","selected_weekday2_days","total_solver_seconds",
            "terminal_soc_last_kwh"
        )
    }
    base_summary["wall_seconds"] = baseline_wall
    print("BASELINE_JSON=" + json.dumps(base_summary), flush=True)

    net = np.asarray(data.load_kwh - data.pv_kwh, dtype=float)
    base = np.zeros_like(net)
    t0 = time.perf_counter()
    for d in range(len(net)):
        ch = choose_candidate(d, data, cold_load, cold_pv, cfg)
        base[d] = ch.load - ch.pv
    base_build_seconds = time.perf_counter() - t0

    # Fixed common Feb-1 SOC = 1200 kWh (also what full Dynamic reaches).
    common_soc = 1200.0
    rows = []
    for seed in SEEDS:
        print(f"START variant={args.variant} seed={seed}", flush=True)
        t0 = time.perf_counter()
        pred, meta, train_seconds = build_predictions(net, base, data.dates, seed)
        prediction_total_seconds = time.perf_counter() - t0
        met = simulate(pred, net, data.prices, common_soc)
        row = {
            "variant": args.variant,
            "seed": seed,
            **met,
            "train_seconds": train_seconds,
            "train_plus_prediction_seconds": prediction_total_seconds,
            "base_build_seconds": base_build_seconds,
        }
        rows.append(row)
        print("RESULT_JSON=" + json.dumps(row), flush=True)

    agg = {}
    for k in rows[0]:
        if k in {"variant","seed"}: continue
        vals = np.asarray([float(r[k]) for r in rows], dtype=float)
        agg[k+"_mean"] = float(vals.mean())
        agg[k+"_std"] = float(vals.std(ddof=1))
    full_cost = 15069931.173475182
    agg["lstm_cost_vs_frozen_dynamic_pct"] = 100.0 * (
        agg["total_cost_yuan_mean"] / full_cost - 1.0
    )
    base_summary["cost_vs_frozen_dynamic_pct"] = 100.0 * (
        float(base_summary["total_cost_yuan"]) / full_cost - 1.0
    )

    final = {
        "variant": args.variant,
        "config": cfg.__dict__,
        "baseline": base_summary,
        "dual_lstm": agg,
        "seed_results": rows,
        "protocol": {
            "recent_days": 56,
            "same_weekday_k": 2,
            "tau": 0.8,
            "hidden_per_branch": 32,
            "seeds": list(SEEDS),
            "period": ["2025-02-01","2025-12-31"],
            "days": 334,
            "baseline_role_in_lstm": "additive shortcut only; not fed to neural head",
        },
    }
    (ROOT / f"dynamic-ablation-{args.variant}.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FINAL_SUMMARY_JSON=" + json.dumps(final), flush=True)


if __name__ == "__main__":
    main()

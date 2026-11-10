"""Can Dual-LSTM replace the handcrafted Dynamic forecaster?

Seven architectures share the same dual-history backbone:
recent 56 days + last 2 same-weekdays, tau=0.8, hidden=32/branch,
monthly causal refit, last 7 prior days for validation, 3 seeds, same P2 MILP.

Architectures
-------------
dynamic_patch:
    Exact current best family: Dynamic residual histories -> dual LSTM;
    Dynamic baseline is also fed to the fusion head and added back.
dynamic_skip:
    Dynamic is ONLY a ResNet-style shortcut. LSTMs see raw net-load history;
    Dynamic is not fed to the head.
yesterday_skip:
    N[d-1] shortcut; no Dynamic anywhere.
week1_skip:
    N[d-7] shortcut; no Dynamic anywhere.
week2mean_skip:
    (N[d-7]+N[d-14])/2 shortcut; no Dynamic anywhere.
pure_dual:
    No shortcut; dual LSTM predicts the q0.8 net-load profile directly.
learned_gate:
    No Dynamic. A learned softmax gate combines N[d-1], N[d-7], N[d-14]
    as a shortcut, while the same dual-LSTM fusion learns a residual correction.

All independent variants use only raw historical net-load + calendar.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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
    run_day,
)

SEEDS = (42, 2026, 3407)
ARCHES = (
    "dynamic_patch",
    "dynamic_skip",
    "yesterday_skip",
    "week1_skip",
    "week2mean_skip",
    "pure_dual",
    "learned_gate",
)
TAU = 0.8
RECENT_DAYS = 56
WEEKLY_K = 2
HIDDEN = 32
MAX_EPOCHS = 80
PATIENCE = 8
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
OUTPUT_START_DAY = 31


def calendar_feature(day: int, dates: pd.DatetimeIndex) -> np.ndarray:
    ts = dates[day]
    doy = int(ts.dayofyear)
    wd = int(ts.dayofweek)
    out = np.zeros(9, dtype=np.float32)
    out[0] = math.sin(2 * math.pi * doy / 365.0)
    out[1] = math.cos(2 * math.pi * doy / 365.0)
    out[2 + wd] = 1.0
    return out


def recent_idx(day: int) -> np.ndarray:
    return np.arange(max(0, day - RECENT_DAYS), day, dtype=int)


def weekly_idx(day: int, dates: pd.DatetimeIndex) -> np.ndarray:
    wd = dates[day].dayofweek
    idx = np.flatnonzero(dates[:day].dayofweek == wd)
    return idx[-WEEKLY_K:]


def shortcut_profile(mode: str, day: int, net: np.ndarray, dynamic_base: np.ndarray) -> np.ndarray:
    if mode == "dynamic":
        return dynamic_base[day]
    if mode == "yesterday":
        return net[day - 1]
    if mode == "week1":
        return net[day - 7]
    if mode == "week2mean":
        return 0.5 * (net[day - 7] + net[day - 14])
    raise ValueError(mode)


def min_target_day(arch: str) -> int:
    if arch in {"week2mean_skip", "learned_gate"}:
        return 14
    if arch == "week1_skip":
        return 7
    if arch == "yesterday_skip":
        return 1
    return 0


def fixed_shortcut_mode(arch: str) -> str | None:
    return {
        "dynamic_patch": "dynamic",
        "dynamic_skip": "dynamic",
        "yesterday_skip": "yesterday",
        "week1_skip": "week1",
        "week2mean_skip": "week2mean",
    }.get(arch)


class DualModel(nn.Module):
    def __init__(self, *, include_base_feature: bool, learned_gate: bool):
        super().__init__()
        self.include_base_feature = include_base_feature
        self.learned_gate = learned_gate
        self.recent_lstm = nn.LSTM(144, HIDDEN, batch_first=True)
        self.weekly_lstm = nn.LSTM(144, HIDDEN, batch_first=True)
        fusion_dim = 2 * HIDDEN + 9 + (144 if include_base_feature else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 144),
        )
        self.gate = (
            nn.Sequential(
                nn.Linear(2 * HIDDEN + 9, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
            if learned_gate else None
        )

    @staticmethod
    def encode(lstm: nn.LSTM, seqs: list[torch.Tensor]) -> torch.Tensor:
        lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True)
        packed = pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = lstm(packed)
        return h[-1]

    def forward(self, recent, weekly, cal, base_feature=None, gate_candidates=None):
        hr = self.encode(self.recent_lstm, recent)
        hw = self.encode(self.weekly_lstm, weekly)
        core = torch.cat([hr, hw, cal], dim=1)
        z = core
        if self.include_base_feature:
            assert base_feature is not None
            z = torch.cat([z, base_feature], dim=1)
        correction = self.head(z)
        if self.learned_gate:
            assert gate_candidates is not None
            alpha = torch.softmax(self.gate(core), dim=1)
            # gate_candidates: batch x 3 x 144, already normalized to target scale.
            shortcut = torch.sum(alpha[:, :, None] * gate_candidates, dim=1)
            return shortcut + correction, alpha
        return correction, None


def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    e = target - pred
    return torch.maximum(TAU * e, (TAU - 1.0) * e).mean()


@dataclass
class Fitted:
    model: DualModel
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    net_mean: np.ndarray
    net_std: np.ndarray
    best_epoch: int
    val_loss: float


def target_series(
    arch: str,
    net: np.ndarray,
    dynamic_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target values and validity mask for model training."""
    n = len(net)
    if arch in {"pure_dual", "learned_gate"}:
        return net.copy(), np.ones(n, dtype=bool)
    mode = fixed_shortcut_mode(arch)
    out = np.full_like(net, np.nan)
    valid = np.zeros(n, dtype=bool)
    start = min_target_day(arch)
    for d in range(start, n):
        out[d] = net[d] - shortcut_profile(mode, d, net, dynamic_base)
        valid[d] = True
    return out, valid


def input_series(arch: str, net: np.ndarray, dynamic_base: np.ndarray) -> np.ndarray:
    # Only the historical "patch" control gets Dynamic residual histories.
    if arch == "dynamic_patch":
        return net - dynamic_base
    # Every proposed Dynamic-free / shortcut-only main model sees raw net-load history.
    return net


def build_examples(
    days,
    *,
    arch,
    inp,
    target,
    net,
    dynamic_base,
    dates,
    input_mean,
    input_std,
    target_mean,
    target_std,
    net_mean,
    net_std,
):
    rows = []
    for d0 in days:
        d = int(d0)
        ri = recent_idx(d)
        wi = weekly_idx(d, dates)
        if len(ri) == 0 or len(wi) == 0:
            continue
        recent = torch.tensor((inp[ri] - input_mean) / input_std, dtype=torch.float32)
        weekly = torch.tensor((inp[wi] - input_mean) / input_std, dtype=torch.float32)
        cal = torch.tensor(calendar_feature(d, dates), dtype=torch.float32)
        base_feature = None
        if arch == "dynamic_patch":
            base_feature = torch.tensor(
                (dynamic_base[d] - net_mean) / net_std, dtype=torch.float32
            )
        gate_candidates = None
        if arch == "learned_gate":
            raw = np.stack([net[d - 1], net[d - 7], net[d - 14]], axis=0)
            gate_candidates = torch.tensor(
                (raw - net_mean[None, :]) / net_std[None, :], dtype=torch.float32
            )
            y = torch.tensor((net[d] - net_mean) / net_std, dtype=torch.float32)
        else:
            y = torch.tensor((target[d] - target_mean) / target_std, dtype=torch.float32)
        rows.append((recent, weekly, cal, base_feature, gate_candidates, y, d))
    return rows


def batchify(rows, idx):
    batch = [rows[int(i)] for i in idx]
    recent = [x[0] for x in batch]
    weekly = [x[1] for x in batch]
    cal = torch.stack([x[2] for x in batch])
    if batch[0][3] is None:
        base = None
    else:
        base = torch.stack([x[3] for x in batch])
    if batch[0][4] is None:
        gates = None
    else:
        gates = torch.stack([x[4] for x in batch])
    y = torch.stack([x[5] for x in batch])
    return recent, weekly, cal, base, gates, y


def fit_month(
    *,
    arch,
    inp,
    target,
    valid_target,
    net,
    dynamic_base,
    dates,
    month_start,
    seed,
) -> Fitted:
    fit_end = month_start - 7
    first = max(7, min_target_day(arch))
    train_days = np.array(
        [d for d in range(first, fit_end) if valid_target[d]], dtype=int
    )
    val_days = np.array(
        [d for d in range(fit_end, month_start) if valid_target[d]], dtype=int
    )
    if len(train_days) < 3 or len(val_days) == 0:
        raise RuntimeError(
            f"too few examples arch={arch} month_start={month_start} "
            f"train={len(train_days)} val={len(val_days)}"
        )

    input_mean = inp[:fit_end].mean(axis=0)
    input_std = np.maximum(inp[:fit_end].std(axis=0), 1e-3)
    net_mean = net[:fit_end].mean(axis=0)
    net_std = np.maximum(net[:fit_end].std(axis=0), 1e-3)

    if arch in {"pure_dual", "learned_gate"}:
        target_mean = net_mean.copy()
        target_std = net_std.copy()
    else:
        target_hist = target[:fit_end][valid_target[:fit_end]]
        target_mean = target_hist.mean(axis=0)
        target_std = np.maximum(target_hist.std(axis=0), 1e-3)

    kwargs = dict(
        arch=arch,
        inp=inp,
        target=target,
        net=net,
        dynamic_base=dynamic_base,
        dates=dates,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        net_mean=net_mean,
        net_std=net_std,
    )
    tr = build_examples(train_days, **kwargs)
    va = build_examples(val_days, **kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DualModel(
        include_base_feature=(arch == "dynamic_patch"),
        learned_gate=(arch == "learned_gate"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + month_start)

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(tr), generator=gen)
        for pos in range(0, len(order), BATCH):
            idx = order[pos : pos + BATCH]
            r, w, c, b, g, y = batchify(tr, idx)
            opt.zero_grad(set_to_none=True)
            p, _ = model(r, w, c, b, g)
            loss = pinball(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for pos in range(0, len(va), BATCH):
                ids = np.arange(pos, min(pos + BATCH, len(va)))
                r, w, c, b, g, y = batchify(va, ids)
                p, _ = model(r, w, c, b, g)
                total += float(pinball(p, y)) * len(ids)
        vl = total / len(va)
        if vl < best - 1e-6:
            best = vl
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return Fitted(
        model=model,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        net_mean=net_mean,
        net_std=net_std,
        best_epoch=best_epoch,
        val_loss=best,
    )


def predict_one(
    fit: Fitted,
    *,
    arch,
    day,
    inp,
    target,
    net,
    dynamic_base,
    dates,
):
    ex = build_examples(
        np.array([day]),
        arch=arch,
        inp=inp,
        target=target,
        net=net,
        dynamic_base=dynamic_base,
        dates=dates,
        input_mean=fit.input_mean,
        input_std=fit.input_std,
        target_mean=fit.target_mean,
        target_std=fit.target_std,
        net_mean=fit.net_mean,
        net_std=fit.net_std,
    )
    r, w, c, b, g, _ = batchify(ex, [0])
    with torch.no_grad():
        z, alpha = fit.model(r, w, c, b, g)
    z = z.cpu().numpy()[0]
    alpha_np = None if alpha is None else alpha.cpu().numpy()[0]

    if arch == "learned_gate":
        # z is normalized final net-load profile.
        raw = fit.net_mean + fit.net_std * np.clip(z, -5.0, 5.0)
        return raw, alpha_np

    if arch == "pure_dual":
        raw = fit.target_mean + fit.target_std * np.clip(z, -5.0, 5.0)
        return raw, None

    mode = fixed_shortcut_mode(arch)
    correction = fit.target_mean + fit.target_std * np.clip(z, -5.0, 5.0)
    raw = shortcut_profile(mode, day, net, dynamic_base) + correction
    return raw, None


def month_ranges(dates):
    out = []
    for m in range(2, 13):
        s = int(np.flatnonzero((dates.month == m) & (dates.day == 1))[0])
        e = (
            int(np.flatnonzero((dates.month == m + 1) & (dates.day == 1))[0])
            if m < 12
            else len(dates)
        )
        out.append((m, s, e))
    return out


def build_predictions(arch, seed, net, dynamic_base, dates):
    inp = input_series(arch, net, dynamic_base)
    target, valid = target_series(arch, net, dynamic_base)
    pred = np.full_like(net, np.nan)
    gate_weights = []
    meta = []
    for month, s, e in month_ranges(dates):
        fit = fit_month(
            arch=arch,
            inp=inp,
            target=target,
            valid_target=valid,
            net=net,
            dynamic_base=dynamic_base,
            dates=dates,
            month_start=s,
            seed=seed,
        )
        month_gate = []
        for d in range(s, e):
            raw, alpha = predict_one(
                fit,
                arch=arch,
                day=d,
                inp=inp,
                target=target,
                net=net,
                dynamic_base=dynamic_base,
                dates=dates,
            )
            pred[d] = np.clip(raw, -2500.0, 3000.0)
            if alpha is not None:
                month_gate.append(alpha)
        meta.append(
            {
                "month": month,
                "best_epoch": fit.best_epoch,
                "val_loss": fit.val_loss,
            }
        )
        if month_gate:
            gate_weights.extend(month_gate)
    gate_mean = None
    if gate_weights:
        gate_mean = np.asarray(gate_weights).mean(axis=0).tolist()
    return pred, meta, gate_mean


def solve_net_day(raw_net, prices, soc):
    load = np.maximum(raw_net, 0.0)
    pv = np.maximum(-raw_net, 0.0)
    try:
        return physical.solve_physical_day(load, pv, prices, soc, 3.0)
    except RuntimeError:
        return physical.solve_physical_day(load, pv, prices, soc, 20.0)


def simulate_prediction(pred, net, prices, initial_soc):
    soc = float(initial_soc)
    pc = ec = ek = 0.0
    days = 0
    daily = []
    for d in range(OUTPUT_START_DAY, len(net)):
        plan = solve_net_day(pred[d], prices, soc)
        gap = net[d] + plan.charge - plan.x - plan.discharge
        emergency = np.maximum(gap, 0.0)
        p = float(np.dot(prices, plan.x))
        e = float(np.dot(5.0 * prices, emergency))
        pc += p
        ec += e
        ek += float(emergency.sum())
        days += int(emergency.sum() > 1e-7)
        daily.append(p + e)
        soc = float(plan.soc[-1])
    actual = net[OUTPUT_START_DAY:]
    forecast = pred[OUTPUT_START_DAY:]
    err = actual - forecast
    return {
        "forecast_mae_kwh": float(np.mean(np.abs(err))),
        "forecast_rmse_kwh": float(np.sqrt(np.mean(err ** 2))),
        "coverage": float(np.mean(actual <= forecast)),
        "underprediction_mean_kwh": float(np.mean(np.maximum(err, 0.0))),
        "planned_cost_yuan": pc,
        "emergency_purchase_kwh": ek,
        "emergency_cost_yuan": ec,
        "total_cost_yuan": pc + ec,
        "days_with_emergency": days,
        "final_soc_kwh": soc,
        "max_daily_cost_yuan": float(max(daily)),
    }


def simulate_dynamic_fair(data, cold_load, cold_pv, initial_soc):
    soc = float(initial_soc)
    pc = ec = ek = 0.0
    days = 0
    for d in range(OUTPUT_START_DAY, len(data.dates)):
        out = run_day(
            data,
            cold_load,
            cold_pv,
            d,
            soc,
            FINAL_DYNAMIC_CONFIG,
            20.0,
        )
        rec = out.record
        pc += float(rec["planned_cost_yuan"])
        ec += float(rec["actual_emergency_cost_yuan"])
        ek += float(rec["actual_emergency_kwh"])
        days += int(float(rec["actual_emergency_kwh"]) > 1e-7)
        soc = out.terminal_soc
    return {
        "planned_cost_yuan": pc,
        "emergency_purchase_kwh": ek,
        "emergency_cost_yuan": ec,
        "total_cost_yuan": pc + ec,
        "days_with_emergency": days,
        "final_soc_kwh": soc,
    }


def static_shortcut_predictions(mode, net, dynamic_base):
    p = np.full_like(net, np.nan)
    for d in range(OUTPUT_START_DAY, len(net)):
        p[d] = shortcut_profile(mode, d, net, dynamic_base)
    return p


def aggregate(rows):
    out = {}
    for k in rows[0]:
        if k in {"arch", "seed", "gate_mean"}:
            continue
        vals = []
        ok = True
        for r in rows:
            v = r[k]
            if not isinstance(v, (int, float)):
                ok = False
                break
            vals.append(float(v))
        if not ok:
            continue
        a = np.asarray(vals)
        out[k + "_mean"] = float(a.mean())
        out[k + "_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out[k + "_min"] = float(a.min())
        out[k + "_max"] = float(a.max())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    args = parser.parse_args()
    arch = args.arch
    torch.set_num_threads(2)

    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)

    # Reproduction gate.
    y = run_config(data, cold_load, cold_pv, YESTERDAY_BASELINE_CONFIG, 20.0)
    dyn = run_config(data, cold_load, cold_pv, FINAL_DYNAMIC_CONFIG, 20.0)
    expected_y = 22823747.29284119
    expected_d = 15069856.025188675
    expected_e = 318742.2694089983
    gate = {
        "yesterday_total": float(y.summary["total_cost_yuan"]),
        "dynamic_total": float(dyn.summary["total_cost_yuan"]),
        "dynamic_emergency": float(dyn.summary["emergency_purchase_kwh"]),
    }
    gate["yesterday_abs_diff"] = abs(gate["yesterday_total"] - expected_y)
    gate["dynamic_total_rel_diff"] = abs(gate["dynamic_total"] - expected_d) / expected_d
    gate["dynamic_emergency_rel_diff"] = abs(gate["dynamic_emergency"] - expected_e) / expected_e
    print("REPRO_JSON=" + json.dumps(gate), flush=True)
    if (
        gate["yesterday_abs_diff"] > 1e-4
        or gate["dynamic_total_rel_diff"] > 1e-4
        or gate["dynamic_emergency_rel_diff"] > 1e-4
    ):
        raise RuntimeError("reproduction gate failed")

    net = np.asarray(data.load_kwh - data.pv_kwh, dtype=float)
    dynamic_base = np.zeros_like(net)
    for d in range(len(net)):
        ch = choose_candidate(d, data, cold_load, cold_pv, FINAL_DYNAMIC_CONFIG)
        dynamic_base[d] = ch.load - ch.pv

    # Common Feb-1 SOC comes from the very simple Yesterday baseline, not Dynamic.
    common_soc = float(y.daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"])
    dynamic_soc = float(dyn.daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"])
    fair_dyn = simulate_dynamic_fair(data, cold_load, cold_pv, common_soc)
    baseline = {
        "published_dynamic": {k: dyn.summary[k] for k in (
            "planned_cost_yuan",
            "emergency_purchase_kwh",
            "emergency_cost_yuan",
            "total_cost_yuan",
            "days_with_emergency",
        )},
        "fair_dynamic_common_feb1_soc": fair_dyn,
        "common_feb1_soc_from_yesterday": common_soc,
        "dynamic_feb1_soc": dynamic_soc,
    }
    print("BASELINE_JSON=" + json.dumps(baseline), flush=True)

    # Simple shortcut-only controls: no LSTM, no Dynamic.
    simple = {}
    for mode in ("yesterday", "week1", "week2mean"):
        sp = static_shortcut_predictions(mode, net, dynamic_base)
        simple[mode] = simulate_prediction(sp, net, data.prices, common_soc)
    print("SIMPLE_BASELINES_JSON=" + json.dumps(simple), flush=True)

    results = []
    for seed in SEEDS:
        print(f"START arch={arch} seed={seed}", flush=True)
        pred, meta, gate_mean = build_predictions(
            arch, seed, net, dynamic_base, data.dates
        )
        metrics = simulate_prediction(pred, net, data.prices, common_soc)
        row = {
            "arch": arch,
            "seed": seed,
            **metrics,
            "gate_mean": gate_mean,
        }
        results.append(row)
        print(
            "TRAIN_META_JSON="
            + json.dumps({"arch": arch, "seed": seed, "months": meta}),
            flush=True,
        )
        print("RESULT_JSON=" + json.dumps(row), flush=True)

    agg = aggregate(results)
    ref = fair_dyn["total_cost_yuan"]
    agg["cost_vs_fair_dynamic_pct"] = 100.0 * (
        agg["total_cost_yuan_mean"] / ref - 1.0
    )
    agg["emergency_vs_fair_dynamic_pct"] = 100.0 * (
        agg["emergency_purchase_kwh_mean"]
        / fair_dyn["emergency_purchase_kwh"]
        - 1.0
    )
    if arch == "learned_gate":
        gm = np.asarray([r["gate_mean"] for r in results], dtype=float)
        agg["gate_mean_over_seeds"] = gm.mean(axis=0).tolist()
        agg["gate_std_over_seeds"] = gm.std(axis=0, ddof=1).tolist()

    final = {
        "protocol": {
            "arch": arch,
            "seeds": list(SEEDS),
            "recent_days": RECENT_DAYS,
            "same_weekday_k": WEEKLY_K,
            "tau": TAU,
            "hidden_per_branch": HIDDEN,
            "monthly_refit": True,
            "validation_days": 7,
            "period": ["2025-02-01", "2025-12-31"],
            "days": 334,
            "common_soc_source": "Yesterday baseline Jan replay; no Dynamic dependency",
        },
        "reproduction": gate,
        "baseline": baseline,
        "simple_shortcut_baselines": simple,
        "seed_results": results,
        "aggregate": agg,
    }
    out = ROOT / f"resnet-ablation-{arch}.json"
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FINAL_SUMMARY_JSON="
        + json.dumps({"arch": arch, "aggregate": agg}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

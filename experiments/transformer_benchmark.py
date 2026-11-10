"""Transformer-family benchmark against the current Dynamic-free Dual-LSTM.

Fixed across all learned models
--------------------------------
- 2025-02-01..2025-12-31, 334-day causal replay
- monthly refit, last 7 prior days as validation
- 3 seeds: 42, 2026, 3407
- q0.8 pinball loss
- 7-day causal scalar reserve selected from {0,10,...,140}
- same MILP and Yesterday-derived Feb-1 SOC
- raw historical net load only; no Dynamic feature is used by T1-T5

Architectures
-------------
lstm_control:
    Current Dynamic-free control: recent LSTM32 + weekly LSTM32 +
    learned periodic shortcut + q0.8 residual.
t1:
    Recent daily-token Transformer + weekly LSTM32.
t2:
    Recent daily-token Transformer + weekly MLP.
t3:
    Recent daily-token Transformer + weekly Transformer.
t4:
    Patch Transformer over the recent continuous 10-min history (24 slots/patch)
    + explicit weekly MLP feature.
t5:
    Pure recent daily-token Transformer, direct q0.8 net-load prediction,
    no learned periodic shortcut and no weekly branch.

T1-T4 keep the same learned periodic shortcut over
N[d-1], N[d-7], N[d-14]. T5 intentionally removes that structured shortcut.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("mcm_code")
pkg.__path__ = [str(ROOT / "code")]
pkg.__package__ = "mcm_code"
sys.modules["mcm_code"] = pkg

from mcm_code.common import dataset as common
from mcm_code.common import physical
from mcm_code.result.problem2.solve_problem2_yesterday_dynamic import (
    YESTERDAY_BASELINE_CONFIG,
    run_config,
)

spec = importlib.util.spec_from_file_location(
    "resnet_exp", ROOT / "experiments" / "lstm_resnet_ablation.py"
)
resnet = importlib.util.module_from_spec(spec)
sys.modules["resnet_exp"] = resnet
assert spec.loader is not None
spec.loader.exec_module(resnet)

ARCHES = ("lstm_control", "t1", "t2", "t3", "t4", "t5")
SEEDS = (42, 2026, 3407)
TAU = 0.8
RECENT_DAYS = 56
WEEKLY_K = 2
D_MODEL = 32
NHEAD = 4
RECENT_LAYERS = 2
WEEKLY_TRANSFORMER_LAYERS = 2
FF_DIM = 64
DROPOUT = 0.10
PATCH_LEN = 24
MAX_EPOCHS = 80
PATIENCE = 8
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
OUTPUT_START_DAY = 31
DELTAS = tuple(float(x) for x in range(0, 141, 10))
DYNAMIC_REF_COST = 15069931.173475182


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


def positional_encoding(length: int, dim: int, device, dtype):
    pos = torch.arange(length, device=device, dtype=dtype)[:, None]
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def left_pad(seqs: list[torch.Tensor]):
    max_len = max(int(x.shape[0]) for x in seqs)
    dim = int(seqs[0].shape[1])
    padded = torch.zeros(
        len(seqs), max_len, dim, dtype=seqs[0].dtype, device=seqs[0].device
    )
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool, device=seqs[0].device)
    for i, x in enumerate(seqs):
        n = int(x.shape[0])
        padded[i, max_len - n :] = x
        mask[i, max_len - n :] = False
    return padded, mask


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, layers: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, seqs: list[torch.Tensor]) -> torch.Tensor:
        x, mask = left_pad(seqs)
        x = self.proj(x)
        pe = positional_encoding(x.shape[1], D_MODEL, x.device, x.dtype)
        x = x + pe[None, :, :]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x[:, -1, :])


class WeeklyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(144, D_MODEL, batch_first=True)

    def forward(self, seqs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(seqs, dim=0)
        _, (h, _) = self.lstm(x)
        return h[-1]


class WeeklyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WEEKLY_K * 144, 64),
            nn.ReLU(),
            nn.Linear(64, D_MODEL),
        )

    def forward(self, seqs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(seqs, dim=0).reshape(len(seqs), WEEKLY_K * 144)
        return self.net(x)


class ForecastModel(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch
        if arch == "t4":
            self.recent = TransformerEncoder(PATCH_LEN, RECENT_LAYERS)
        else:
            self.recent = TransformerEncoder(144, RECENT_LAYERS)

        if arch == "t1":
            self.weekly = WeeklyLSTM()
        elif arch in {"t2", "t4"}:
            self.weekly = WeeklyMLP()
        elif arch == "t3":
            self.weekly = TransformerEncoder(144, WEEKLY_TRANSFORMER_LAYERS)
        else:
            self.weekly = None

        core_dim = D_MODEL + (D_MODEL if self.weekly is not None else 0) + 9
        self.head = nn.Sequential(
            nn.Linear(core_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 144),
        )
        self.gate = None
        if arch != "t5":
            self.gate = nn.Sequential(
                nn.Linear(core_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )

    def forward(self, recent, weekly, cal, gate_candidates):
        hr = self.recent(recent)
        parts = [hr]
        if self.weekly is not None:
            assert weekly is not None
            parts.append(self.weekly(weekly))
        parts.append(cal)
        core = torch.cat(parts, dim=1)
        correction_or_direct = self.head(core)
        if self.gate is None:
            return correction_or_direct, None
        assert gate_candidates is not None
        alpha = torch.softmax(self.gate(core), dim=1)
        shortcut = torch.sum(alpha[:, :, None] * gate_candidates, dim=1)
        return shortcut + correction_or_direct, alpha


def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    e = target - pred
    return torch.maximum(TAU * e, (TAU - 1.0) * e).mean()


@dataclass
class Fitted:
    model: ForecastModel
    net_mean: np.ndarray
    net_std: np.ndarray
    best_epoch: int
    val_loss: float
    param_count: int


def make_example(
    day: int,
    arch: str,
    net: np.ndarray,
    dates: pd.DatetimeIndex,
    net_mean: np.ndarray,
    net_std: np.ndarray,
):
    ri = recent_idx(day)
    wi = weekly_idx(day, dates)
    if len(ri) == 0 or len(wi) < WEEKLY_K or day < 14:
        return None

    recent_norm = (net[ri] - net_mean) / net_std
    if arch == "t4":
        flat = recent_norm.reshape(-1)
        assert len(flat) % PATCH_LEN == 0
        recent = torch.tensor(
            flat.reshape(-1, PATCH_LEN), dtype=torch.float32
        )
    else:
        recent = torch.tensor(recent_norm, dtype=torch.float32)

    weekly = None
    if arch in {"t1", "t2", "t3", "t4"}:
        weekly = torch.tensor(
            (net[wi] - net_mean) / net_std, dtype=torch.float32
        )

    cal = torch.tensor(calendar_feature(day, dates), dtype=torch.float32)
    gates = None
    if arch != "t5":
        candidates = np.stack(
            [net[day - 1], net[day - 7], net[day - 14]], axis=0
        )
        gates = torch.tensor(
            (candidates - net_mean[None, :]) / net_std[None, :],
            dtype=torch.float32,
        )
    y = torch.tensor((net[day] - net_mean) / net_std, dtype=torch.float32)
    return recent, weekly, cal, gates, y, day


def build_examples(days, arch, net, dates, net_mean, net_std):
    rows = []
    for d in days:
        ex = make_example(int(d), arch, net, dates, net_mean, net_std)
        if ex is not None:
            rows.append(ex)
    return rows


def batchify(rows, idx):
    batch = [rows[int(i)] for i in idx]
    recent = [x[0] for x in batch]
    weekly = None
    if batch[0][1] is not None:
        weekly = [x[1] for x in batch]
    cal = torch.stack([x[2] for x in batch])
    gates = None
    if batch[0][3] is not None:
        gates = torch.stack([x[3] for x in batch])
    y = torch.stack([x[4] for x in batch])
    return recent, weekly, cal, gates, y


def fit_month(
    *,
    arch,
    net,
    dates,
    month_start,
    seed,
) -> Fitted:
    fit_end = month_start - 7
    train_days = np.arange(14, fit_end, dtype=int)
    val_days = np.arange(fit_end, month_start, dtype=int)
    if len(train_days) < 3 or len(val_days) == 0:
        raise RuntimeError(
            f"too few examples arch={arch} month_start={month_start} "
            f"train={len(train_days)} val={len(val_days)}"
        )

    net_mean = net[:fit_end].mean(axis=0)
    net_std = np.maximum(net[:fit_end].std(axis=0), 1e-3)
    tr = build_examples(train_days, arch, net, dates, net_mean, net_std)
    va = build_examples(val_days, arch, net, dates, net_mean, net_std)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ForecastModel(arch)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            ids = order[pos : pos + BATCH]
            recent, weekly, cal, gates, y = batchify(tr, ids)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(recent, weekly, cal, gates)
            loss = pinball(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for pos in range(0, len(va), BATCH):
                ids = np.arange(pos, min(pos + BATCH, len(va)))
                recent, weekly, cal, gates, y = batchify(va, ids)
                pred, _ = model(recent, weekly, cal, gates)
                total += float(pinball(pred, y)) * len(ids)
        val_loss = total / len(va)
        if val_loss < best - 1e-6:
            best = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
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
        net_mean=net_mean,
        net_std=net_std,
        best_epoch=best_epoch,
        val_loss=best,
        param_count=param_count,
    )


def predict_one(fit: Fitted, arch, day, net, dates):
    ex = make_example(day, arch, net, dates, fit.net_mean, fit.net_std)
    assert ex is not None
    recent, weekly, cal, gates, _, _ = ex
    weekly_batch = None if weekly is None else [weekly]
    gates_batch = None if gates is None else gates[None, :, :]
    with torch.no_grad():
        z, alpha = fit.model(
            [recent], weekly_batch, cal[None, :], gates_batch
        )
    zn = np.clip(z.cpu().numpy()[0], -5.0, 5.0)
    raw = fit.net_mean + fit.net_std * zn
    alpha_np = None if alpha is None else alpha.cpu().numpy()[0]
    return raw, alpha_np


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


def build_transformer_predictions(arch, seed, net, dates):
    pred = np.full_like(net, np.nan)
    meta = []
    gate_weights = []
    params = None
    for month, s, e in month_ranges(dates):
        fit = fit_month(
            arch=arch,
            net=net,
            dates=dates,
            month_start=s,
            seed=seed,
        )
        params = fit.param_count
        month_gate = []
        for d in range(s, e):
            raw, alpha = predict_one(fit, arch, d, net, dates)
            pred[d] = np.clip(raw, -2500.0, 3000.0)
            if alpha is not None:
                month_gate.append(alpha)
        meta.append(
            {
                "month": month,
                "best_epoch": fit.best_epoch,
                "val_loss": fit.val_loss,
                "param_count": fit.param_count,
            }
        )
        if month_gate:
            gate_weights.extend(month_gate)
    gate_mean = None
    if gate_weights:
        gate_mean = np.asarray(gate_weights).mean(axis=0).tolist()
    return pred, meta, gate_mean, int(params)


def select_dynamic7_delta(errors, prices):
    if not errors:
        return 70.0
    arr = np.asarray(errors[-7:], dtype=float)
    pos = np.maximum(arr, 0.0)
    normal_unit_cost = float(np.sum(prices))
    best_delta = None
    best_score = float("inf")
    for delta in DELTAS:
        reserve_cost = float(delta) * normal_unit_cost
        emergency_each = np.sum(
            np.maximum(pos - float(delta), 0.0)
            * (5.0 * prices[None, :]),
            axis=1,
        )
        score = reserve_cost + float(np.mean(emergency_each))
        if score < best_score:
            best_score = score
            best_delta = float(delta)
    return best_delta


def solve_shifted(pred_day, delta, prices, soc):
    risk = pred_day + float(delta)
    load = np.maximum(risk, 0.0)
    pv = np.maximum(-risk, 0.0)
    try:
        return physical.solve_physical_day(load, pv, prices, soc, 3.0)
    except RuntimeError:
        return physical.solve_physical_day(load, pv, prices, soc, 20.0)


def simulate(pred, net, prices, initial_soc):
    soc = float(initial_soc)
    planned_cost = 0.0
    emergency_cost = 0.0
    emergency_kwh = 0.0
    emergency_days = 0
    raw_errors = []
    deltas = []
    solver_seconds = 0.0
    t0 = time.perf_counter()

    for d in range(OUTPUT_START_DAY, len(net)):
        delta = select_dynamic7_delta(raw_errors, prices)
        plan = solve_shifted(pred[d], delta, prices, soc)
        solver_seconds += float(plan.solver_seconds)

        gap = net[d] + plan.charge - plan.x - plan.discharge
        emergency = np.maximum(gap, 0.0)
        pc = float(np.dot(prices, plan.x))
        ec = float(np.dot(5.0 * prices, emergency))
        planned_cost += pc
        emergency_cost += ec
        emergency_kwh += float(emergency.sum())
        emergency_days += int(emergency.sum() > 1e-7)
        soc = float(plan.soc[-1])
        deltas.append(delta)
        raw_errors.append(net[d] - pred[d])

    wall = time.perf_counter() - t0
    actual = net[OUTPUT_START_DAY:]
    forecast = pred[OUTPUT_START_DAY:]
    err = actual - forecast
    return {
        "forecast_mae_kwh": float(np.mean(np.abs(err))),
        "forecast_rmse_kwh": float(np.sqrt(np.mean(err ** 2))),
        "raw_coverage": float(np.mean(actual <= forecast)),
        "raw_underprediction_mean_kwh": float(
            np.mean(np.maximum(err, 0.0))
        ),
        "planned_cost_yuan": planned_cost,
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": planned_cost + emergency_cost,
        "days_with_emergency": emergency_days,
        "final_soc_kwh": soc,
        "mean_delta_kwh": float(np.mean(deltas)),
        "median_delta_kwh": float(np.median(deltas)),
        "dispatch_wall_seconds": wall,
        "dispatch_solver_seconds": solver_seconds,
    }


def aggregate(rows):
    out = {}
    for k in rows[0]:
        if k in {"arch", "seed", "gate_mean", "months"}:
            continue
        vals = [r[k] for r in rows]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        arr = np.asarray(vals, dtype=float)
        out[k + "_mean"] = float(arr.mean())
        out[k + "_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[k + "_min"] = float(arr.min())
        out[k + "_max"] = float(arr.max())
    return out


def arch_description(arch):
    return {
        "lstm_control": "Recent LSTM32 + Weekly LSTM32 + learned periodic gate",
        "t1": "Recent daily-token Transformer + Weekly LSTM32 + learned periodic gate",
        "t2": "Recent daily-token Transformer + Weekly MLP + learned periodic gate",
        "t3": "Recent daily-token Transformer + Weekly Transformer + learned periodic gate",
        "t4": "24-slot Patch Transformer + explicit Weekly MLP + learned periodic gate",
        "t5": "Pure recent daily-token Transformer direct q0.8 forecast; no shortcut/weekly branch",
    }[arch]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, required=True)
    args = parser.parse_args()
    arch = args.arch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)

    y = run_config(
        data, cold_load, cold_pv, YESTERDAY_BASELINE_CONFIG, 20.0
    )
    expected_y = 22823747.29284119
    y_total = float(y.summary["total_cost_yuan"])
    if abs(y_total - expected_y) > 1e-4:
        raise RuntimeError(
            f"Yesterday reproduction gate failed: {y_total} vs {expected_y}"
        )
    initial_soc = float(
        y.daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"]
    )

    net = np.asarray(data.load_kwh - data.pv_kwh, dtype=float)
    dynamic_placeholder = np.zeros_like(net)

    rows = []
    training = []
    for seed in SEEDS:
        print(f"START arch={arch} seed={seed}", flush=True)
        t0 = time.perf_counter()
        if arch == "lstm_control":
            pred, meta, gate_mean = resnet.build_predictions(
                "learned_gate",
                seed,
                net,
                dynamic_placeholder,
                data.dates,
            )
            param_count = sum(
                p.numel()
                for p in resnet.DualModel(
                    include_base_feature=False, learned_gate=True
                ).parameters()
                if p.requires_grad
            )
        else:
            pred, meta, gate_mean, param_count = (
                build_transformer_predictions(
                    arch, seed, net, data.dates
                )
            )
        train_seconds = time.perf_counter() - t0
        metrics = simulate(
            pred, net, data.prices, initial_soc
        )
        row = {
            "arch": arch,
            "seed": seed,
            "param_count": int(param_count),
            "train_predict_seconds": train_seconds,
            **metrics,
            "gate_mean": gate_mean,
        }
        rows.append(row)
        training.append(
            {
                "seed": seed,
                "train_predict_seconds": train_seconds,
                "param_count": int(param_count),
                "gate_mean": gate_mean,
                "months": meta,
            }
        )
        print("RESULT_JSON=" + json.dumps(row), flush=True)

    agg = aggregate(rows)
    agg["cost_vs_dynamic_pct"] = 100.0 * (
        agg["total_cost_yuan_mean"] / DYNAMIC_REF_COST - 1.0
    )
    if any(r["gate_mean"] is not None for r in rows):
        gm = np.asarray(
            [r["gate_mean"] for r in rows if r["gate_mean"] is not None],
            dtype=float,
        )
        agg["gate_mean_over_seeds"] = gm.mean(axis=0).tolist()
        agg["gate_std_over_seeds"] = (
            gm.std(axis=0, ddof=1).tolist() if len(gm) > 1 else [0.0] * 3
        )

    final = {
        "arch": arch,
        "description": arch_description(arch),
        "protocol": {
            "seeds": list(SEEDS),
            "tau": TAU,
            "recent_days": RECENT_DAYS,
            "same_weekday_k": WEEKLY_K,
            "monthly_refit": True,
            "validation_days": 7,
            "period": ["2025-02-01", "2025-12-31"],
            "days": 334,
            "reserve": "dynamic7_uniform, causal errors only from days < d",
            "reserve_grid": list(DELTAS),
            "common_soc_source": "Yesterday baseline Jan replay",
            "transformer": {
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "recent_layers": RECENT_LAYERS,
                "weekly_transformer_layers": WEEKLY_TRANSFORMER_LAYERS,
                "ff_dim": FF_DIM,
                "dropout": DROPOUT,
                "patch_len_slots_t4": PATCH_LEN,
            },
            "training": {
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "batch": BATCH,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            },
        },
        "yesterday_reproduction_total_yuan": y_total,
        "dynamic_reference_total_yuan": DYNAMIC_REF_COST,
        "seed_results": rows,
        "training": training,
        "aggregate": agg,
    }

    out = ROOT / f"transformer-experiment-{arch}.json"
    out.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "FINAL_SUMMARY_JSON="
        + json.dumps(
            {
                "arch": arch,
                "description": arch_description(arch),
                "aggregate": agg,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

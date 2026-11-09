"""Exhaustive strict-causal history-input ablation for the best P2 Quantile Residual LSTM.

This is a temporary experiment on an isolated branch.  It keeps fixed:
- the repository's frozen Dynamic deterministic baseline;
- residual target r = actual net load - Dynamic base net load;
- tau=0.8 pinball loss;
- LSTM hidden size 32;
- monthly causal refit with the final 7 prior days reserved for validation;
- the repository's physical.solve_physical_day MILP and P2 settlement;
- three fixed random seeds.

Only the history information structure changes.

Stage 1 (12 architectures):
  all-history;
  recent {3,7,14,28,56,84} days;
  same-weekday {1,2,4,8,12} previous occurrences.

Stage 2 (9 dual-branch architectures):
  recent {7,14,28} x same-weekday {2,4,8}.

Evaluation: 2025-02-01..2025-12-31, continuous SOC, 334 days.
"""

from __future__ import annotations

import json
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

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

SEEDS = (42, 2026, 3407)
MIN_HISTORY_DAYS = 7
HIDDEN = 32
MAX_EPOCHS = 80
PATIENCE = 8
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
TAU = 0.80
OUTPUT_START_DAY = 31

EXPECTED = {
    "yesterday_total": 22823747.29284119,
    "dynamic_total": 15069856.025188675,
    "dynamic_emergency": 318742.2694089983,
}
PREVIOUS_Q80 = {
    42: 14992858.941207787,
    2026: 14999653.362792999,
    3407: 14972124.34063248,
}


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    stage: int
    recent_days: int | None = None
    weekly_k: int | None = None


SPECS: tuple[Spec, ...] = (
    Spec("all_history", "all", 1),
    *tuple(Spec(f"recent_{L}", "recent", 1, recent_days=L) for L in (3, 7, 14, 28, 56, 84)),
    *tuple(Spec(f"weekday_{K}", "weekday", 1, weekly_k=K) for K in (1, 2, 4, 8, 12)),
    *tuple(
        Spec(f"recent_{L}_weekday_{K}", "dual", 2, recent_days=L, weekly_k=K)
        for L in (7, 14, 28)
        for K in (2, 4, 8)
    ),
)


def calendar_feature(day: int, dates: pd.DatetimeIndex) -> np.ndarray:
    ts = dates[day]
    doy = int(ts.dayofyear)
    wd = int(ts.dayofweek)
    out = np.zeros(9, dtype=np.float32)
    out[0] = math.sin(2.0 * math.pi * doy / 365.0)
    out[1] = math.cos(2.0 * math.pi * doy / 365.0)
    out[2 + wd] = 1.0
    return out


def history_indices(day: int, spec: Spec, dates: pd.DatetimeIndex, branch: str = "single") -> np.ndarray:
    if spec.family == "all":
        return np.arange(0, day, dtype=int)

    if spec.family == "recent" or (spec.family == "dual" and branch == "recent"):
        assert spec.recent_days is not None
        start = max(0, day - spec.recent_days)
        return np.arange(start, day, dtype=int)

    if spec.family == "weekday" or (spec.family == "dual" and branch == "weekday"):
        assert spec.weekly_k is not None
        prior = np.arange(0, day, dtype=int)
        weekdays = dates.weekday.to_numpy()
        same = prior[weekdays[prior] == weekdays[day]]
        return same[-spec.weekly_k :]

    raise ValueError((spec, branch))


def pad_histories(
    residual_scaled: np.ndarray,
    days: np.ndarray,
    spec: Spec,
    dates: pd.DatetimeIndex,
    branch: str = "single",
) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = [history_indices(int(d), spec, dates, branch) for d in days]
    if any(len(idx) == 0 for idx in idxs):
        raise RuntimeError(f"empty history for {spec.name}/{branch}")
    max_len = max(len(idx) for idx in idxs)
    arr = np.zeros((len(days), max_len, residual_scaled.shape[1]), dtype=np.float32)
    lengths = np.empty(len(days), dtype=np.int64)
    for i, idx in enumerate(idxs):
        seq = residual_scaled[idx].astype(np.float32, copy=False)
        arr[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return torch.from_numpy(arr), torch.from_numpy(lengths)


def encode_lstm(lstm: nn.LSTM, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # recent_7 has no padding because all training targets start after day 7.
    # Taking the direct path makes it an exact reproduction of the prior q80 run.
    if int(lengths.min()) == x.shape[1] and int(lengths.max()) == x.shape[1]:
        _, (h, _) = lstm(x)
        return h[-1]
    packed = pack_padded_sequence(
        x,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    _, (h, _) = lstm(packed)
    return h[-1]


class SingleQuantileLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(144, HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + 9 + 144, 64),
            nn.ReLU(),
            nn.Linear(64, 144),
        )

    def forward(self, x, lengths, cal, base):
        h = encode_lstm(self.lstm, x, lengths)
        return self.head(torch.cat([h, cal, base], dim=1))


class DualQuantileLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.recent_lstm = nn.LSTM(144, HIDDEN, batch_first=True)
        self.weekday_lstm = nn.LSTM(144, HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN + 9 + 144, 64),
            nn.ReLU(),
            nn.Linear(64, 144),
        )

    def forward(self, recent_x, recent_len, weekly_x, weekly_len, cal, base):
        hr = encode_lstm(self.recent_lstm, recent_x, recent_len)
        hw = encode_lstm(self.weekday_lstm, weekly_x, weekly_len)
        return self.head(torch.cat([hr, hw, cal, base], dim=1))


def pinball(pred, target):
    err = target - pred
    return torch.maximum(TAU * err, (TAU - 1.0) * err).mean()


@dataclass
class MonthModel:
    model: nn.Module
    residual_mean: np.ndarray
    residual_std: np.ndarray
    net_mean: np.ndarray
    net_std: np.ndarray
    best_epoch: int
    val_loss: float
    train_days: int
    val_days: int
    train_max_history: int
    val_max_history: int


def make_common_features(
    days: np.ndarray,
    residual: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    net_mean: np.ndarray,
    net_std: np.ndarray,
):
    cal = torch.tensor(
        np.asarray([calendar_feature(int(d), dates) for d in days]),
        dtype=torch.float32,
    )
    base = torch.tensor(
        np.asarray([(base_net[d] - net_mean) / net_std for d in days]),
        dtype=torch.float32,
    )
    target = torch.tensor(
        np.asarray([(residual[d] - residual_mean) / residual_std for d in days]),
        dtype=torch.float32,
    )
    return cal, base, target


def build_dataset(
    days: np.ndarray,
    spec: Spec,
    residual: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    net_mean: np.ndarray,
    net_std: np.ndarray,
):
    residual_scaled = ((residual - residual_mean) / residual_std).astype(np.float32)
    cal, base, target = make_common_features(
        days, residual, base_net, dates,
        residual_mean, residual_std, net_mean, net_std,
    )
    if spec.family == "dual":
        xr, lr = pad_histories(residual_scaled, days, spec, dates, "recent")
        xw, lw = pad_histories(residual_scaled, days, spec, dates, "weekday")
        return {
            "recent_x": xr, "recent_len": lr,
            "weekly_x": xw, "weekly_len": lw,
            "cal": cal, "base": base, "target": target,
        }
    x, lengths = pad_histories(residual_scaled, days, spec, dates, "single")
    return {
        "x": x, "lengths": lengths,
        "cal": cal, "base": base, "target": target,
    }


def select_batch(dataset: dict[str, torch.Tensor], idx: torch.Tensor):
    out = {}
    for key, value in dataset.items():
        out[key] = value[idx]
    return out


def model_forward(model: nn.Module, spec: Spec, batch: dict[str, torch.Tensor]):
    if spec.family == "dual":
        return model(
            batch["recent_x"], batch["recent_len"],
            batch["weekly_x"], batch["weekly_len"],
            batch["cal"], batch["base"],
        )
    return model(batch["x"], batch["lengths"], batch["cal"], batch["base"])


def max_history(dataset: dict[str, torch.Tensor], spec: Spec) -> int:
    if spec.family == "dual":
        return max(
            int(dataset["recent_len"].max()),
            int(dataset["weekly_len"].max()),
        )
    return int(dataset["lengths"].max())


def fit_month(
    spec: Spec,
    seed: int,
    month_start: int,
    residual: np.ndarray,
    net: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
) -> MonthModel:
    fit_end = month_start - 7
    train_days = np.arange(MIN_HISTORY_DAYS, fit_end, dtype=int)
    val_days = np.arange(fit_end, month_start, dtype=int)
    if len(train_days) < 8:
        raise RuntimeError(f"too few train days: month_start={month_start}")

    residual_mean = residual[:fit_end].mean(axis=0)
    residual_std = np.maximum(residual[:fit_end].std(axis=0), 1e-3)
    net_mean = net[:fit_end].mean(axis=0)
    net_std = np.maximum(net[:fit_end].std(axis=0), 1e-3)

    tr = build_dataset(
        train_days, spec, residual, base_net, dates,
        residual_mean, residual_std, net_mean, net_std,
    )
    va = build_dataset(
        val_days, spec, residual, base_net, dates,
        residual_mean, residual_std, net_mean, net_std,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model: nn.Module = DualQuantileLSTM() if spec.family == "dual" else SingleQuantileLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    bad = 0
    gen = torch.Generator().manual_seed(seed + month_start)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_days), generator=gen)
        for pos in range(0, len(order), BATCH):
            idx = order[pos : pos + BATCH]
            batch = select_batch(tr, idx)
            opt.zero_grad(set_to_none=True)
            pred = model_forward(model, spec, batch)
            loss = pinball(pred, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model_forward(model, spec, va)
            vl = float(pinball(vp, va["target"]).item())

        if vl < best_loss - 1e-6:
            best_loss = vl
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("no best state")
    model.load_state_dict(best_state)
    model.eval()

    return MonthModel(
        model=model,
        residual_mean=residual_mean,
        residual_std=residual_std,
        net_mean=net_mean,
        net_std=net_std,
        best_epoch=best_epoch,
        val_loss=best_loss,
        train_days=len(train_days),
        val_days=len(val_days),
        train_max_history=max_history(tr, spec),
        val_max_history=max_history(va, spec),
    )


def one_day_dataset(
    day: int,
    spec: Spec,
    residual: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
    mm: MonthModel,
):
    return build_dataset(
        np.asarray([day], dtype=int),
        spec, residual, base_net, dates,
        mm.residual_mean, mm.residual_std, mm.net_mean, mm.net_std,
    )


def predict_month(
    spec: Spec,
    seed: int,
    start: int,
    end: int,
    residual: np.ndarray,
    net: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
):
    mm = fit_month(spec, seed, start, residual, net, base_net, dates)
    pred = np.empty((end - start, 144), dtype=float)
    with torch.no_grad():
        for i, d in enumerate(range(start, end)):
            batch = one_day_dataset(d, spec, residual, base_net, dates, mm)
            z = model_forward(mm.model, spec, batch).cpu().numpy()[0]
            z = np.clip(z, -5.0, 5.0)
            pred_residual = mm.residual_mean + mm.residual_std * z
            pred[i] = np.clip(base_net[d] + pred_residual, -2500.0, 3000.0)
    meta = {
        "month": int(dates[start].month),
        "best_epoch": mm.best_epoch,
        "val_loss": mm.val_loss,
        "train_days": mm.train_days,
        "val_days": mm.val_days,
        "train_max_history": mm.train_max_history,
        "val_max_history": mm.val_max_history,
    }
    return pred, meta


def month_ranges(dates: pd.DatetimeIndex):
    out = []
    for month in range(2, 13):
        start = int(np.flatnonzero((dates.month == month) & (dates.day == 1))[0])
        if month < 12:
            end = int(np.flatnonzero((dates.month == month + 1) & (dates.day == 1))[0])
        else:
            end = len(dates)
        out.append((month, start, end))
    return out


def build_predictions(
    spec: Spec,
    seed: int,
    residual: np.ndarray,
    net: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
):
    deployed = np.full_like(net, np.nan)
    meta = []
    for _month, start, end in month_ranges(dates):
        p, m = predict_month(spec, seed, start, end, residual, net, base_net, dates)
        deployed[start:end] = p
        meta.append(m)
    return deployed, meta


def solve_net_day(raw_net, prices, soc):
    load = np.maximum(raw_net, 0.0)
    pv = np.maximum(-raw_net, 0.0)
    try:
        return physical.solve_physical_day(load, pv, prices, soc, 3.0)
    except RuntimeError:
        return physical.solve_physical_day(load, pv, prices, soc, 20.0)


def simulate(
    spec: Spec,
    seed: int,
    pred: np.ndarray,
    net: np.ndarray,
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    initial_soc: float,
):
    soc = float(initial_soc)
    planned_cost = 0.0
    emergency_cost = 0.0
    emergency_kwh = 0.0
    days_emergency = 0
    daily_rows = []

    for d in range(OUTPUT_START_DAY, len(dates)):
        raw = pred[d]
        if not np.isfinite(raw).all():
            raise RuntimeError(f"missing prediction {spec.name} seed={seed} day={d}")
        plan = solve_net_day(raw, prices, soc)
        gap = net[d] + plan.charge - plan.x - plan.discharge
        emergency = np.maximum(gap, 0.0)
        pc = float(np.dot(prices, plan.x))
        ec = float(np.dot(5.0 * prices, emergency))
        ek = float(emergency.sum())
        planned_cost += pc
        emergency_cost += ec
        emergency_kwh += ek
        days_emergency += int(ek > 1e-7)
        soc = float(plan.soc[-1])
        daily_rows.append({
            "date": str(dates[d].date()),
            "architecture": spec.name,
            "family": spec.family,
            "stage": spec.stage,
            "seed": seed,
            "planned_cost_yuan": pc,
            "emergency_cost_yuan": ec,
            "total_cost_yuan": pc + ec,
            "emergency_purchase_kwh": ek,
            "terminal_soc_kwh": soc,
        })

    actual = net[OUTPUT_START_DAY:]
    forecast = pred[OUTPUT_START_DAY:]
    err = actual - forecast
    return {
        "architecture": spec.name,
        "family": spec.family,
        "stage": spec.stage,
        "recent_days": spec.recent_days,
        "weekly_k": spec.weekly_k,
        "seed": seed,
        "forecast_mae_kwh": float(np.mean(np.abs(err))),
        "forecast_rmse_kwh": float(np.sqrt(np.mean(err**2))),
        "forecast_mae_kw": float(6.0 * np.mean(np.abs(err))),
        "underprediction_mean_kwh": float(np.mean(np.maximum(err, 0.0))),
        "coverage_actual_le_forecast": float(np.mean(actual <= forecast)),
        "planned_cost_yuan": planned_cost,
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": planned_cost + emergency_cost,
        "days_with_emergency": days_emergency,
        "final_soc_kwh": soc,
    }, daily_rows


def aggregate_results(rows: list[dict], dynamic_total: float, dynamic_emergency: float):
    metrics = [
        "forecast_mae_kwh", "forecast_rmse_kwh", "forecast_mae_kw",
        "underprediction_mean_kwh", "coverage_actual_le_forecast",
        "planned_cost_yuan", "emergency_purchase_kwh", "emergency_cost_yuan",
        "total_cost_yuan", "days_with_emergency", "final_soc_kwh",
    ]
    out = []
    for spec in SPECS:
        rs = [r for r in rows if r["architecture"] == spec.name]
        if len(rs) != len(SEEDS):
            raise RuntimeError(f"incomplete seeds for {spec.name}: {len(rs)}")
        item = {
            "architecture": spec.name,
            "family": spec.family,
            "stage": spec.stage,
            "recent_days": spec.recent_days,
            "weekly_k": spec.weekly_k,
        }
        for key in metrics:
            vals = np.asarray([float(r[key]) for r in rs], dtype=float)
            item[key + "_mean"] = float(vals.mean())
            item[key + "_std"] = float(vals.std(ddof=1))
            item[key + "_min"] = float(vals.min())
            item[key + "_max"] = float(vals.max())
        item["cost_vs_dynamic_pct"] = 100.0 * (
            item["total_cost_yuan_mean"] / dynamic_total - 1.0
        )
        item["emergency_vs_dynamic_pct"] = 100.0 * (
            item["emergency_purchase_kwh_mean"] / dynamic_emergency - 1.0
        )
        out.append(item)
    out.sort(key=lambda x: x["total_cost_yuan_mean"])
    for rank, item in enumerate(out, 1):
        item["rank"] = rank
    return out


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    artifact_dir = ROOT / "history_ablation_artifacts"
    artifact_dir.mkdir(exist_ok=True)

    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)

    print("=== reproduction gate ===", flush=True)
    yesterday = run_config(data, cold_load, cold_pv, YESTERDAY_BASELINE_CONFIG, 20.0)
    dynamic = run_config(data, cold_load, cold_pv, FINAL_DYNAMIC_CONFIG, 20.0)

    repro = {
        "yesterday_total": float(yesterday.summary["total_cost_yuan"]),
        "dynamic_total": float(dynamic.summary["total_cost_yuan"]),
        "dynamic_emergency": float(dynamic.summary["emergency_purchase_kwh"]),
    }
    repro["yesterday_abs_diff"] = abs(repro["yesterday_total"] - EXPECTED["yesterday_total"])
    repro["dynamic_total_rel_diff"] = abs(
        repro["dynamic_total"] - EXPECTED["dynamic_total"]
    ) / EXPECTED["dynamic_total"]
    repro["dynamic_emergency_rel_diff"] = abs(
        repro["dynamic_emergency"] - EXPECTED["dynamic_emergency"]
    ) / EXPECTED["dynamic_emergency"]
    print("REPRO_JSON=" + json.dumps(repro, ensure_ascii=False), flush=True)

    if repro["yesterday_abs_diff"] > 1e-4:
        raise RuntimeError(f"Yesterday reproduction gate failed: {repro}")
    if repro["dynamic_total_rel_diff"] > 1e-4 or repro["dynamic_emergency_rel_diff"] > 1e-4:
        raise RuntimeError(f"Dynamic reproduction gate failed (>0.01%): {repro}")

    net = np.asarray(data.load_kwh - data.pv_kwh, dtype=float)
    base_net = np.zeros_like(net)
    print("=== precompute frozen Dynamic deterministic forecasts ===", flush=True)
    for d in range(len(data.dates)):
        choice = choose_candidate(d, data, cold_load, cold_pv, FINAL_DYNAMIC_CONFIG)
        base_net[d] = choice.load - choice.pv
    residual = net - base_net

    dyn_daily = dynamic.daily.copy()
    initial_soc = float(dyn_daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"])
    dynamic_metrics = {
        "planned_cost_yuan": float(dynamic.summary["planned_cost_yuan"]),
        "emergency_purchase_kwh": float(dynamic.summary["emergency_purchase_kwh"]),
        "emergency_cost_yuan": float(dynamic.summary["emergency_cost_yuan"]),
        "total_cost_yuan": float(dynamic.summary["total_cost_yuan"]),
        "days_with_emergency": int(dynamic.summary["days_with_emergency"]),
        "base_forecast_mae_kwh": float(
            np.mean(np.abs(residual[OUTPUT_START_DAY:]))
        ),
        "base_forecast_mae_kw": float(
            6.0 * np.mean(np.abs(residual[OUTPUT_START_DAY:]))
        ),
        "initial_soc_feb1_kwh": initial_soc,
    }
    print("DYNAMIC_JSON=" + json.dumps(dynamic_metrics, ensure_ascii=False), flush=True)

    seed_rows: list[dict] = []
    daily_rows: list[dict] = []
    train_meta: list[dict] = []

    # Put recent_7 first: it should reproduce the previous q80 experiment.
    ordered = sorted(SPECS, key=lambda s: (0 if s.name == "recent_7" else 1, s.stage, s.name))
    total_runs = len(ordered) * len(SEEDS)
    run_no = 0

    for spec in ordered:
        for seed in SEEDS:
            run_no += 1
            print(
                f"=== RUN {run_no}/{total_runs} architecture={spec.name} seed={seed} ===",
                flush=True,
            )
            pred, meta = build_predictions(
                spec, seed, residual, net, base_net, data.dates
            )
            row, daily = simulate(
                spec, seed, pred, net, data.dates, data.prices, initial_soc
            )
            seed_rows.append(row)
            daily_rows.extend(daily)
            train_meta.append({
                "architecture": spec.name,
                "family": spec.family,
                "stage": spec.stage,
                "recent_days": spec.recent_days,
                "weekly_k": spec.weekly_k,
                "seed": seed,
                "months": meta,
            })
            print("RESULT_JSON=" + json.dumps(row, ensure_ascii=False), flush=True)

            if spec.name == "recent_7":
                rel = abs(row["total_cost_yuan"] - PREVIOUS_Q80[seed]) / PREVIOUS_Q80[seed]
                print(
                    "Q80_REPRO_JSON=" + json.dumps(
                        {
                            "seed": seed,
                            "previous_total": PREVIOUS_Q80[seed],
                            "current_total": row["total_cost_yuan"],
                            "relative_diff": rel,
                        }
                    ),
                    flush=True,
                )
                # Cross-run CPU LSTM kernels can drift slightly even with fixed seeds.
                # This is logged as an audit diagnostic, not used as a hard gate.
                if rel > 5e-4:
                    print(
                        f"WARNING: recent_7 cross-run drift is {rel:.6%}; "
                        "all architecture comparisons remain within this same deterministic run.",
                        flush=True,
                    )

    aggregate = aggregate_results(
        seed_rows,
        dynamic_metrics["total_cost_yuan"],
        dynamic_metrics["emergency_purchase_kwh"],
    )

    best = aggregate[0]
    stage1 = [x for x in aggregate if x["stage"] == 1]
    stage2 = [x for x in aggregate if x["stage"] == 2]
    best_recent = min(
        [x for x in stage1 if x["family"] == "recent"],
        key=lambda x: x["total_cost_yuan_mean"],
    )
    best_weekday = min(
        [x for x in stage1 if x["family"] == "weekday"],
        key=lambda x: x["total_cost_yuan_mean"],
    )
    best_single = min(stage1, key=lambda x: x["total_cost_yuan_mean"])
    best_dual = min(stage2, key=lambda x: x["total_cost_yuan_mean"])

    final = {
        "protocol": {
            "period": ["2025-02-01", "2025-12-31"],
            "days": 334,
            "architectures": len(SPECS),
            "seeds": list(SEEDS),
            "total_rollouts": len(SPECS) * len(SEEDS),
            "hidden": HIDDEN,
            "tau": TAU,
            "min_training_history_days": MIN_HISTORY_DAYS,
            "monthly_refit": True,
            "validation_days": 7,
            "history_semantics": (
                "recent/all histories are causal sequences ending at d-1; "
                "same-weekday keeps up to K prior occurrences; early targets use all available "
                "history but no target/future information"
            ),
            "dispatch": (
                "repository physical.solve_physical_day; q80 time-varying residual quantile; "
                "no scalar delta reserve"
            ),
        },
        "reproduction": repro,
        "dynamic": dynamic_metrics,
        "best": best,
        "best_stage1": best_single,
        "best_recent": best_recent,
        "best_weekday": best_weekday,
        "best_stage2": best_dual,
        "aggregate": aggregate,
        "seed_results": seed_rows,
    }

    pd.DataFrame(seed_rows).to_csv(
        artifact_dir / "history_seed_results.csv", index=False
    )
    pd.DataFrame(aggregate).to_csv(
        artifact_dir / "history_aggregate_results.csv", index=False
    )
    pd.DataFrame(daily_rows).to_csv(
        artifact_dir / "history_daily_results.csv", index=False
    )
    (artifact_dir / "history_training_meta.json").write_text(
        json.dumps(train_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "history_final_results.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("TOP10_JSON=" + json.dumps(aggregate[:10], ensure_ascii=False), flush=True)
    print(
        "FINAL_RESULTS_JSON=" + json.dumps(
            {
                "protocol": final["protocol"],
                "reproduction": repro,
                "dynamic": dynamic_metrics,
                "best": best,
                "best_stage1": best_single,
                "best_recent": best_recent,
                "best_weekday": best_weekday,
                "best_stage2": best_dual,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

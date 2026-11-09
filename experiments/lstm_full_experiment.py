"""Strict-causal LSTM experiment for CUMCM 2026 C / P2.

Temporary experiment only. It imports the repository's frozen Dynamic policy and
physical MILP, verifies the published baselines, then compares:
1) current Dynamic;
2) pure daily-profile LSTM + the same scalar reserve principle;
3) Dynamic-base + residual LSTM + recalibrated scalar reserve;
4) Dynamic-base + tau=0.8 quantile residual LSTM (time-varying risk forecast).

All neural models:
- use a 7-day sequence of 144-slot daily profiles;
- are refit from scratch at each month boundary;
- train only on days before the final 7-day validation/calibration window;
- use 3 fixed seeds;
- are evaluated on 2025-02-01..2025-12-31 with one continuous SOC.
"""

from __future__ import annotations

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
HISTORY_DAYS = 7
HIDDEN = 32
MAX_EPOCHS = 80
PATIENCE = 8
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
TAU = 0.80
OUTPUT_START_DAY = 31
DELTA_GRID = np.arange(0.0, 141.0, 10.0)
DECAY_DAYS = 5.0
SOC_TARGET = (common.SOC_MIN + common.SOC_MAX) / 2.0


def calendar_feature(day: int, dates: pd.DatetimeIndex) -> np.ndarray:
    ts = dates[day]
    doy = int(ts.dayofyear)
    wd = int(ts.dayofweek)
    out = np.zeros(9, dtype=np.float32)
    out[0] = math.sin(2 * math.pi * doy / 365.0)
    out[1] = math.cos(2 * math.pi * doy / 365.0)
    out[2 + wd] = 1.0
    return out


class DailyLSTM(nn.Module):
    def __init__(self, use_base: bool):
        super().__init__()
        self.use_base = use_base
        self.lstm = nn.LSTM(144, HIDDEN, batch_first=True)
        in_dim = HIDDEN + 9 + (144 if use_base else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 144),
        )

    def forward(self, x, cal, base=None):
        _, (h, _) = self.lstm(x)
        z = [h[-1], cal]
        if self.use_base:
            assert base is not None
            z.append(base)
        return self.head(torch.cat(z, dim=1))


def pinball(pred, target, tau=TAU):
    err = target - pred
    return torch.maximum(tau * err, (tau - 1.0) * err).mean()


@dataclass
class MonthModel:
    model: DailyLSTM
    source_mean: np.ndarray
    source_std: np.ndarray
    net_mean: np.ndarray
    net_std: np.ndarray
    train_days: int
    val_days: int
    best_epoch: int
    val_loss: float


def examples(
    source: np.ndarray,
    target: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
    days: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    net_mean: np.ndarray,
    net_std: np.ndarray,
    use_base: bool,
):
    xs, cs, bs, ys = [], [], [], []
    for d in days:
        xs.append((source[d-HISTORY_DAYS:d] - source_mean) / source_std)
        cs.append(calendar_feature(int(d), dates))
        if use_base:
            bs.append((base_net[d] - net_mean) / net_std)
        ys.append((target[d] - source_mean) / source_std)
    X = torch.tensor(np.asarray(xs), dtype=torch.float32)
    C = torch.tensor(np.asarray(cs), dtype=torch.float32)
    Y = torch.tensor(np.asarray(ys), dtype=torch.float32)
    B = torch.tensor(np.asarray(bs), dtype=torch.float32) if use_base else None
    return X, C, B, Y


def fit_month(
    *,
    source: np.ndarray,
    target: np.ndarray,
    base_net: np.ndarray,
    net: np.ndarray,
    dates: pd.DatetimeIndex,
    month_start: int,
    seed: int,
    use_base: bool,
    quantile: bool,
) -> MonthModel:
    fit_end = month_start - 7
    train_days = np.arange(HISTORY_DAYS, fit_end, dtype=int)
    val_days = np.arange(fit_end, month_start, dtype=int)
    if len(train_days) < 8:
        raise RuntimeError(f"too few train days at month start {month_start}: {len(train_days)}")

    # Scalers are fitted only on the training prefix, never on validation/current month.
    source_mean = source[:fit_end].mean(axis=0)
    source_std = np.maximum(source[:fit_end].std(axis=0), 1e-3)
    net_mean = net[:fit_end].mean(axis=0)
    net_std = np.maximum(net[:fit_end].std(axis=0), 1e-3)

    Xtr, Ctr, Btr, Ytr = examples(
        source, target, base_net, dates, train_days,
        source_mean, source_std, net_mean, net_std, use_base,
    )
    Xva, Cva, Bva, Yva = examples(
        source, target, base_net, dates, val_days,
        source_mean, source_std, net_mean, net_std, use_base,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DailyLSTM(use_base=use_base)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    bad = 0
    gen = torch.Generator().manual_seed(seed + month_start)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(Xtr), generator=gen)
        for pos in range(0, len(order), BATCH):
            idx = order[pos:pos+BATCH]
            opt.zero_grad(set_to_none=True)
            pred = model(Xtr[idx], Ctr[idx], None if Btr is None else Btr[idx])
            loss = pinball(pred, Ytr[idx]) if quantile else torch.mean((pred - Ytr[idx]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xva, Cva, Bva)
            vl = pinball(vp, Yva).item() if quantile else torch.mean((vp - Yva) ** 2).item()
        if vl < best_loss - 1e-6:
            best_loss = vl
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
    return MonthModel(
        model=model,
        source_mean=source_mean,
        source_std=source_std,
        net_mean=net_mean,
        net_std=net_std,
        train_days=len(train_days),
        val_days=len(val_days),
        best_epoch=best_epoch,
        val_loss=best_loss,
    )


def predict_day(
    mm: MonthModel,
    source: np.ndarray,
    base_net: np.ndarray,
    dates: pd.DatetimeIndex,
    day: int,
    use_base: bool,
) -> np.ndarray:
    x = ((source[day-HISTORY_DAYS:day] - mm.source_mean) / mm.source_std)[None]
    c = calendar_feature(day, dates)[None]
    b = ((base_net[day] - mm.net_mean) / mm.net_std)[None] if use_base else None
    with torch.no_grad():
        z = mm.model(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(c, dtype=torch.float32),
            None if b is None else torch.tensor(b, dtype=torch.float32),
        ).cpu().numpy()[0]
    z = np.clip(z, -5.0, 5.0)
    return mm.source_mean + mm.source_std * z


def month_ranges(dates: pd.DatetimeIndex):
    out = []
    for m in range(2, 13):
        start = int(np.flatnonzero((dates.month == m) & (dates.day == 1))[0])
        if m < 12:
            end = int(np.flatnonzero((dates.month == m + 1) & (dates.day == 1))[0])
        else:
            end = len(dates)
        out.append((m, start, end))
    return out


def build_predictions(kind, seed, net, base_net, dates):
    if kind == "pure":
        source = net
        target = net
        use_base = False
        quantile = False
    elif kind == "residual":
        source = net - base_net
        target = source
        use_base = True
        quantile = False
    elif kind == "q80":
        source = net - base_net
        target = source
        use_base = True
        quantile = True
    else:
        raise ValueError(kind)

    deployed = np.full_like(net, np.nan)
    calibration = {}
    train_meta = []
    for month, start, end in month_ranges(dates):
        mm = fit_month(
            source=source,
            target=target,
            base_net=base_net,
            net=net,
            dates=dates,
            month_start=start,
            seed=seed,
            use_base=use_base,
            quantile=quantile,
        )
        cal = {}
        for d in range(start - 7, end):
            out = predict_day(mm, source, base_net, dates, d, use_base)
            raw = out if kind == "pure" else base_net[d] + out
            # Generous physical sanity clip; the observed range is far inside this.
            raw = np.clip(raw, -2500.0, 3000.0)
            if d < start:
                cal[d] = raw
            else:
                deployed[d] = raw
        calibration[month] = cal
        train_meta.append({
            "month": month,
            "train_days": mm.train_days,
            "val_days": mm.val_days,
            "best_epoch": mm.best_epoch,
            "val_loss": mm.val_loss,
        })
    return deployed, calibration, train_meta


def state_multiplier(initial_soc: float) -> float:
    normalized = (SOC_TARGET - float(initial_soc)) / (common.SOC_MAX - common.SOC_MIN)
    return float(np.clip(1.0 + 0.8 * normalized, 0.4, 1.6))


def reserve_from_recent(recent_errors, initial_soc, prices):
    if not recent_errors:
        return 70.0
    arr = np.asarray(recent_errors[-7:], dtype=float)
    n = len(arr)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = np.exp(-ages / DECAY_DAYS)
    w /= w.sum()
    positive = np.maximum(arr, 0.0)
    sf = state_multiplier(initial_soc)
    normal_unit_cost = float(np.sum(prices))
    best_d, best_s = None, float("inf")
    for delta in DELTA_GRID:
        reserve_cost = float(delta) * normal_unit_cost
        emerg_each_day = np.sum(
            np.maximum(positive - float(delta), 0.0) * (5.0 * prices[None, :]),
            axis=1,
        )
        score = reserve_cost + sf * float(np.dot(w, emerg_each_day))
        if score < best_s:
            best_s, best_d = score, float(delta)
    return best_d


def solve_net_day(raw_net, delta, prices, soc):
    risk_net = raw_net + float(delta)
    load = np.maximum(risk_net, 0.0)
    pv = np.maximum(-risk_net, 0.0)
    try:
        return physical.solve_physical_day(load, pv, prices, soc, 3.0)
    except RuntimeError:
        return physical.solve_physical_day(load, pv, prices, soc, 20.0)


def simulate_neural(
    *,
    kind,
    pred,
    calibration,
    net,
    dates,
    prices,
    initial_soc,
):
    soc = float(initial_soc)
    planned_cost = 0.0
    emergency_cost = 0.0
    emergency_kwh = 0.0
    days_emergency = 0
    deltas = []
    daily_cost = []
    month_to_range = {m:(s,e) for m,s,e in month_ranges(dates)}

    for month, start, end in month_ranges(dates):
        # Calibrate the reserve with the new month's model on its untouched prior week.
        recent = [net[d] - calibration[month][d] for d in range(start - 7, start)]
        for d in range(start, end):
            raw = pred[d]
            if not np.isfinite(raw).all():
                raise RuntimeError(f"missing prediction {kind} day {d}")
            delta = 0.0 if kind == "q80" else reserve_from_recent(recent, soc, prices)
            plan = solve_net_day(raw, delta, prices, soc)
            x = plan.x
            charge = plan.charge
            discharge = plan.discharge
            gap = net[d] + charge - x - discharge
            emergency = np.maximum(gap, 0.0)
            pc = float(np.dot(prices, x))
            ec = float(np.dot(5.0 * prices, emergency))
            planned_cost += pc
            emergency_cost += ec
            ek = float(emergency.sum())
            emergency_kwh += ek
            days_emergency += int(ek > 1e-7)
            daily_cost.append(pc + ec)
            deltas.append(delta)
            soc = float(plan.soc[-1])
            recent.append(net[d] - raw)
            if len(recent) > 7:
                recent.pop(0)

    raw_eval = pred[OUTPUT_START_DAY:]
    actual_eval = net[OUTPUT_START_DAY:]
    err = actual_eval - raw_eval
    return {
        "forecast_mae_kwh": float(np.mean(np.abs(err))),
        "forecast_rmse_kwh": float(np.sqrt(np.mean(err ** 2))),
        "forecast_mae_kw": float(6.0 * np.mean(np.abs(err))),
        "underprediction_mean_kwh": float(np.mean(np.maximum(err, 0.0))),
        "coverage_actual_le_forecast": float(np.mean(actual_eval <= raw_eval)),
        "planned_cost_yuan": planned_cost,
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": planned_cost + emergency_cost,
        "days_with_emergency": days_emergency,
        "mean_delta_kwh": float(np.mean(deltas)),
        "final_soc_kwh": soc,
        "max_daily_cost_yuan": float(np.max(daily_cost)),
    }


def main():
    torch.set_num_threads(2)
    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)

    expected = {
        "yesterday_total": 22823747.29284119,
        "dynamic_total": 15069856.025188675,
        "dynamic_emergency": 318742.2694089983,
    }

    print("=== reproduction gate: frozen repo baselines ===", flush=True)
    y = run_config(data, cold_load, cold_pv, YESTERDAY_BASELINE_CONFIG, 20.0)
    dyn = run_config(data, cold_load, cold_pv, FINAL_DYNAMIC_CONFIG, 20.0)
    repro = {
        "yesterday_total": float(y.summary["total_cost_yuan"]),
        "dynamic_total": float(dyn.summary["total_cost_yuan"]),
        "dynamic_emergency": float(dyn.summary["emergency_purchase_kwh"]),
    }
    repro["max_abs_diff"] = max(abs(repro[k] - expected[k]) for k in expected)
    print("REPRO_JSON=" + json.dumps(repro, ensure_ascii=False), flush=True)
    if repro["max_abs_diff"] > 1e-4:
        raise RuntimeError(f"reproduction gate failed: {repro}")

    net = np.asarray(data.load_kwh - data.pv_kwh, dtype=float)
    base_net = np.zeros_like(net)
    print("=== precomputing frozen Dynamic base forecasts ===", flush=True)
    for d in range(len(data.dates)):
        choice = choose_candidate(d, data, cold_load, cold_pv, FINAL_DYNAMIC_CONFIG)
        base_net[d] = choice.load - choice.pv

    # Current Dynamic forecast diagnostics, pre-reserve and with its deployed scalar reserve.
    dyn_daily = dyn.daily.copy()
    dyn_delta = dyn_daily["delta_kwh_per_interval"].to_numpy(float)
    dyn_raw_err = net[OUTPUT_START_DAY:] - base_net[OUTPUT_START_DAY:]
    dyn_risk_err = net[OUTPUT_START_DAY:] - (
        base_net[OUTPUT_START_DAY:] + dyn_delta[OUTPUT_START_DAY:, None]
    )
    dynamic_metrics = {
        **{k: dyn.summary[k] for k in (
            "planned_cost_yuan", "emergency_purchase_kwh", "emergency_cost_yuan",
            "total_cost_yuan", "days_with_emergency", "terminal_soc_last_kwh"
        )},
        "base_forecast_mae_kwh": float(np.mean(np.abs(dyn_raw_err))),
        "base_forecast_mae_kw": float(6*np.mean(np.abs(dyn_raw_err))),
        "risk_adjusted_mae_kwh": float(np.mean(np.abs(dyn_risk_err))),
        "mean_delta_kwh": float(np.mean(dyn_delta[OUTPUT_START_DAY:])),
    }
    initial_soc = float(dyn_daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"])
    print("DYNAMIC_JSON=" + json.dumps(dynamic_metrics, ensure_ascii=False), flush=True)
    print(f"COMMON_FEB1_SOC={initial_soc:.12f}", flush=True)

    results = []
    training = []
    for kind in ("pure", "residual", "q80"):
        for seed in SEEDS:
            print(f"=== train+forecast {kind} seed={seed} ===", flush=True)
            pred, calibration, meta = build_predictions(kind, seed, net, base_net, data.dates)
            print(
                "TRAIN_META_JSON=" + json.dumps(
                    {"kind":kind, "seed":seed, "months":meta}, ensure_ascii=False
                ),
                flush=True,
            )
            print(f"=== dispatch {kind} seed={seed} ===", flush=True)
            metrics = simulate_neural(
                kind=kind,
                pred=pred,
                calibration=calibration,
                net=net,
                dates=data.dates,
                prices=data.prices,
                initial_soc=initial_soc,
            )
            row = {"model": kind, "seed": seed, **metrics}
            results.append(row)
            print("RESULT_JSON=" + json.dumps(row, ensure_ascii=False), flush=True)

    # Aggregate across the 3 independently trained seeds.
    aggregate = {}
    for kind in ("pure", "residual", "q80"):
        rows = [r for r in results if r["model"] == kind]
        metrics = {}
        for key in rows[0]:
            if key in {"model", "seed"}:
                continue
            vals = np.asarray([r[key] for r in rows], dtype=float)
            metrics[key + "_mean"] = float(vals.mean())
            metrics[key + "_std"] = float(vals.std(ddof=1))
            metrics[key + "_min"] = float(vals.min())
            metrics[key + "_max"] = float(vals.max())
        aggregate[kind] = metrics

    final = {
        "protocol": {
            "period": ["2025-02-01", "2025-12-31"],
            "days": 334,
            "seeds": list(SEEDS),
            "history_days": HISTORY_DAYS,
            "hidden": HIDDEN,
            "monthly_refit": True,
            "validation_days": 7,
            "pure_and_residual_reserve": "same 0..140 by 10 scalar cost proxy, recalibrated on each model residual",
            "quantile_tau": TAU,
            "q80_reserve": "time-varying quantile prediction replaces scalar reserve",
            "initial_soc": "common frozen-Dynamic Jan warmup SOC at 2025-02-01",
            "dispatch": "repository physical.solve_physical_day, identical fixed-price P2 settlement",
        },
        "reproduction": repro,
        "dynamic": dynamic_metrics,
        "seed_results": results,
        "aggregate": aggregate,
    }
    print("FINAL_RESULTS_JSON=" + json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

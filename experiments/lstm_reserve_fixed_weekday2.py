"""Explicit safety-margin experiment for Dual-LSTM forecasts.

Purpose
-------
The preceding q0.8 models produce a risk-shifted profile but use no explicit
post-forecast scalar reserve. This experiment separates those two concepts by
adding an explicit nonnegative margin delta (kWh / 10-min interval) to every
predicted slot before the same P2 MILP.

Architectures:
  dynamic_skip, learned_gate, yesterday_skip, week1_skip, week2mean_skip, pure_dual

Policies:
  fixed delta in {0,10,...,140}
  dynamic7_uniform  : daily delta selected causally from previous 7 forecast errors
  dynamic28_uniform : same over previous 28 days
  dynamic7_exp_soc  : production-style 7-day exponentially weighted selector
                      with carried-SOC multiplier, applied to LSTM errors

All dynamic selectors use only errors from days strictly before d. No target-day
actual is used. They choose delta from the same 0..140 grid by a cost proxy:
normal reserve purchase cost + 5x emergency-cost proxy.
"""
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace
import json
import sys
import time
import types
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

spec = importlib.util.spec_from_file_location(
    "resnet_exp", ROOT / "experiments" / "lstm_resnet_ablation.py"
)
resnet = importlib.util.module_from_spec(spec)
sys.modules["resnet_exp"] = resnet
assert spec.loader is not None
spec.loader.exec_module(resnet)

ARCHES = (
    "dynamic_skip",
    "learned_gate",
    "yesterday_skip",
    "week1_skip",
    "week2mean_skip",
    "pure_dual",
)
SEEDS = (42, 2026, 3407)
DELTAS = tuple(float(x) for x in range(0, 141, 10))
OUTPUT_START_DAY = 31
SOC_TARGET = 6000.0
SOC_MIN = 1200.0
SOC_MAX = 10800.0


def solve_shifted(pred_day, delta, prices, soc):
    risk = pred_day + float(delta)
    load = np.maximum(risk, 0.0)
    pv = np.maximum(-risk, 0.0)
    try:
        return physical.solve_physical_day(load, pv, prices, soc, 3.0)
    except RuntimeError:
        return physical.solve_physical_day(load, pv, prices, soc, 20.0)


def state_multiplier(soc):
    z = (SOC_TARGET - float(soc)) / (SOC_MAX - SOC_MIN)
    return float(np.clip(1.0 + 0.8 * z, 0.4, 1.6))


def select_dynamic_delta(errors, prices, soc, mode):
    if not errors:
        return 70.0
    if mode == "dynamic7_uniform":
        arr = np.asarray(errors[-7:], dtype=float)
        weights = np.ones(len(arr), dtype=float) / len(arr)
        sf = 1.0
    elif mode == "dynamic28_uniform":
        arr = np.asarray(errors[-28:], dtype=float)
        weights = np.ones(len(arr), dtype=float) / len(arr)
        sf = 1.0
    elif mode == "dynamic7_exp_soc":
        arr = np.asarray(errors[-7:], dtype=float)
        ages = np.arange(len(arr)-1, -1, -1, dtype=float)
        weights = np.exp(-ages / 5.0)
        weights /= weights.sum()
        sf = state_multiplier(soc)
    else:
        raise ValueError(mode)

    pos = np.maximum(arr, 0.0)
    normal_unit_cost = float(np.sum(prices))
    best_delta = None
    best_score = float("inf")
    for delta in DELTAS:
        reserve_cost = float(delta) * normal_unit_cost
        emergency_each = np.sum(
            np.maximum(pos - float(delta), 0.0) * (5.0 * prices[None, :]),
            axis=1,
        )
        score = reserve_cost + sf * float(np.dot(weights, emergency_each))
        if score < best_score:
            best_score = score
            best_delta = float(delta)
    return best_delta


def simulate(pred, net, prices, initial_soc, policy):
    soc = float(initial_soc)
    pc = ec = ek = 0.0
    emergency_days = 0
    deltas = []
    raw_errors = []
    solver_seconds = 0.0
    t0 = time.perf_counter()

    for d in range(OUTPUT_START_DAY, len(net)):
        if policy.startswith("fixed_"):
            delta = float(policy.split("_", 1)[1])
        else:
            delta = select_dynamic_delta(raw_errors, prices, soc, policy)

        plan = solve_shifted(pred[d], delta, prices, soc)
        solver_seconds += float(plan.solver_seconds)
        gap = net[d] + plan.charge - plan.x - plan.discharge
        emergency = np.maximum(gap, 0.0)

        p = float(np.dot(prices, plan.x))
        e = float(np.dot(5.0 * prices, emergency))
        pc += p
        ec += e
        ek += float(emergency.sum())
        emergency_days += int(emergency.sum() > 1e-7)
        soc = float(plan.soc[-1])
        deltas.append(delta)

        # Reserve calibration always observes the underlying unshifted forecast error.
        raw_errors.append(net[d] - pred[d])

    wall = time.perf_counter() - t0
    return {
        "planned_cost_yuan": pc,
        "emergency_purchase_kwh": ek,
        "emergency_cost_yuan": ec,
        "total_cost_yuan": pc + ec,
        "days_with_emergency": emergency_days,
        "final_soc_kwh": soc,
        "mean_delta_kwh": float(np.mean(deltas)),
        "median_delta_kwh": float(np.median(deltas)),
        "delta_0_days": int(np.sum(np.asarray(deltas) == 0.0)),
        "delta_140_days": int(np.sum(np.asarray(deltas) == 140.0)),
        "dispatch_wall_seconds": wall,
        "dispatch_solver_seconds": solver_seconds,
    }


def aggregate(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["policy"], []).append(r)
    out = []
    for policy, rs in grouped.items():
        a = {"policy": policy, "n": len(rs)}
        for k in rs[0]:
            if k in {"arch", "seed", "policy"}:
                continue
            vals = np.asarray([float(x[k]) for x in rs], dtype=float)
            a[k + "_mean"] = float(vals.mean())
            a[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            a[k + "_min"] = float(vals.min())
            a[k + "_max"] = float(vals.max())
        out.append(a)
    out.sort(key=lambda x: x["total_cost_yuan_mean"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("dynamic_skip",), default="dynamic_skip", nargs="?")
    args = parser.parse_args()
    arch = args.arch
    torch.set_num_threads(2)

    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)

    # Reproduction + common SOC.
    y = run_config(data, cold_load, cold_pv, YESTERDAY_BASELINE_CONFIG, 20.0)
    if abs(float(y.summary["total_cost_yuan"]) - 22823747.29284119) > 1e-4:
        raise RuntimeError("Yesterday reproduction gate failed")
    initial_soc = float(y.daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"])

    net = np.asarray(data.load_kwh - data.pv_kwh, dtype=float)
    dynamic_base = np.zeros_like(net)
    fixed_weekday2 = replace(
        FINAL_DYNAMIC_CONFIG,
        name="fixed_weekday2_shortcut",
        candidate_mode="weekday2_only",
    )
    for d in range(len(net)):
        ch = choose_candidate(d, data, cold_load, cold_pv, fixed_weekday2)
        dynamic_base[d] = ch.load - ch.pv

    rows = []
    train_meta = []
    policies = [f"fixed_{int(x)}" for x in DELTAS] + [
        "dynamic7_uniform",
        "dynamic28_uniform",
        "dynamic7_exp_soc",
    ]

    for seed in SEEDS:
        print(f"TRAIN arch={arch} seed={seed}", flush=True)
        t0 = time.perf_counter()
        pred, meta, gate_mean = resnet.build_predictions(
            arch, seed, net, dynamic_base, data.dates
        )
        train_seconds = time.perf_counter() - t0
        train_meta.append({
            "seed": seed,
            "train_seconds": train_seconds,
            "gate_mean": gate_mean,
            "months": meta,
        })

        actual = net[OUTPUT_START_DAY:]
        raw = pred[OUTPUT_START_DAY:]
        err = actual - raw
        forecast_stats = {
            "forecast_mae_kwh": float(np.mean(np.abs(err))),
            "forecast_rmse_kwh": float(np.sqrt(np.mean(err ** 2))),
            "raw_coverage": float(np.mean(actual <= raw)),
            "raw_underprediction_mean_kwh": float(np.mean(np.maximum(err, 0.0))),
        }

        for policy in policies:
            met = simulate(pred, net, data.prices, initial_soc, policy)
            row = {
                "arch": arch,
                "seed": seed,
                "policy": policy,
                **forecast_stats,
                **met,
            }
            rows.append(row)
            print("RESULT_JSON=" + json.dumps(row), flush=True)

    agg = aggregate(rows)
    frozen_dynamic = 15069931.173475182
    for a in agg:
        a["cost_vs_dynamic_pct"] = 100.0 * (
            a["total_cost_yuan_mean"] / frozen_dynamic - 1.0
        )

    final = {
        "arch": "fixed_weekday2_skip",
        "protocol": {
            "seeds": list(SEEDS),
            "tau": 0.8,
            "recent_days": 56,
            "same_weekday_k": 2,
            "fixed_delta_grid": list(DELTAS),
            "dynamic_policies": [
                "dynamic7_uniform",
                "dynamic28_uniform",
                "dynamic7_exp_soc",
            ],
            "period": ["2025-02-01", "2025-12-31"],
            "days": 334,
            "dynamic_selector_causality": "only unshifted forecast errors from days < d",
        },
        "training": train_meta,
        "seed_policy_results": rows,
        "aggregate": agg,
        "best": agg[0],
    }

    (ROOT / "reserve-experiment-fixed_weekday2_skip.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "FINAL_SUMMARY_JSON="
        + json.dumps(
            {
                "arch": "fixed_weekday2_skip",
                "best": agg[0],
                "top5": agg[:5],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""问题三 baseline：Dynamic 框架 + 附件3的 0:00 光伏预报，不做日内调整。

相对问题二 Dynamic 的唯一求解侧改动：
  MILP 的光伏输入改为附件 3 当日 0:00 预报（连线+中点插值），
  不再使用 Dynamic 候选里的历史光伏代理。

负载预测、候选选择、δ 选择仍完整复用 Dynamic
（walk-forward 净负荷 MAE 仍用历史实际光伏，仅用于选负载与 δ）。

费用（不调整 ⇒ x_act = x_plan）：
  总费用 = Σ c·x + Σ 5c·e   （与问题二结算同构；口径 A 下无偏差项）

输出：写入 results/problem3/audit/baseline/，**不**覆盖正式 result3.xlsx。
"""

from __future__ import annotations
from mcm_code.common.paths import project_root, results_root

import argparse
import json

import numpy as np
import pandas as pd

ROOT = project_root()

from ..problem2.solve_problem2_yesterday_dynamic import (
    paper_config,
    choose_candidate,
    choose_delta,
)
from ...common import dataset as common
from ...common import physical
from .problem3_data import load_problem3_data

N = common.INTERVALS_PER_DAY
SOC_MIN = common.SOC_MIN
SOC_MAX = common.SOC_MAX
SOC_INITIAL = common.SOC_INITIAL
OUTPUT_START = common.OUTPUT_START

OUT_DIR = results_root() / "problem3" / "audit" / "baseline"


def run_baseline(
    time_limit: float = 40.0,
    max_days: int | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    p3 = load_problem3_data(ROOT)
    data = common.from_arrays(p3.dates, p3.prices, p3.load_kwh, p3.pv_kwh)
    cold_load, cold_pv = p3.cold_load, p3.cold_pv
    # 附件3 0:00 预报电量 (365, 144)
    pv_f0 = p3.forecast_kwh[:, 0, :]
    config = paper_config()

    days = len(data.dates) if max_days is None else min(max_days, len(data.dates))
    records: list[dict] = []
    xs: list[np.ndarray] = []
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    es: list[np.ndarray] = []
    socs: list[np.ndarray] = []
    initial_soc = float(SOC_INITIAL)
    total_solver = 0.0
    bad_soc = bad_sim = bad_bal = 0

    for day in range(days):
        choice = choose_candidate(day, data, cold_load, cold_pv, config)
        delta_choice = choose_delta(
            day, choice.candidate, initial_soc, data, cold_load, cold_pv, config
        )
        delta = float(delta_choice.delta)
        solve_load = choice.load + delta if config.reserve_in_milp else choice.load
        # P3 口径：优化用 0:00 官方预报，不用 choice.pv
        solve_pv = pv_f0[day]

        plan = physical.solve_physical_day(
            solve_load, solve_pv, data.prices, initial_soc, time_limit
        )
        total_solver += plan.solver_seconds
        bad_soc += int(
            plan.soc_residual_max > 1e-5
            or np.min(plan.soc) < SOC_MIN - 1e-5
            or np.max(plan.soc) > SOC_MAX + 1e-5
        )
        bad_sim += int(plan.simultaneous_count)

        actual = physical.settle_actual(
            data.load_kwh[day],
            data.pv_kwh[day],
            plan,
            0.0 if config.reserve_in_milp else delta,
        )
        balance = np.asarray(actual["balance"], dtype=float)
        bad_bal += int(float(np.max(np.abs(balance))) > 1e-5)
        emergency = np.asarray(actual["emergency"], dtype=float)
        x = plan.x if config.reserve_in_milp else plan.x + delta
        planned_cost = float(np.dot(data.prices, x))
        emergency_cost = float(np.dot(5.0 * data.prices, emergency))

        records.append(
            {
                "date": str(data.dates[day].date()),
                "day_index": day,
                "selected_candidate": choice.candidate,
                "delta_kwh_per_interval": delta,
                "initial_soc_kwh": initial_soc,
                "terminal_soc_kwh": float(plan.soc[-1]),
                "forecast_load_kwh": float(solve_load.sum()),
                "forecast_pv_att3_0am_kwh": float(solve_pv.sum()),
                "forecast_pv_dynamic_proxy_kwh": float(choice.pv.sum()),
                "planned_purchase_kwh": float(x.sum()),
                "planned_cost_yuan": planned_cost,
                "actual_emergency_kwh": float(emergency.sum()),
                "actual_emergency_cost_yuan": emergency_cost,
                "pv_curtailment_kwh": float(np.asarray(actual["pv_curtailment"]).sum()),
                "grid_spill_kwh": float(np.asarray(actual["grid_spill"]).sum()),
                "total_cost_yuan": planned_cost + emergency_cost,
                "solver_seconds": plan.solver_seconds,
            }
        )
        xs.append(x.copy())
        us.append(plan.charge.copy())
        vs.append(plan.discharge.copy())
        es.append(emergency.copy())
        socs.append(plan.soc.copy())

        initial_soc = float(plan.soc[-1])
        if (day + 1) % 30 == 0 or day == 0:
            print(
                f"[P3-baseline] day {day + 1}/{days} {data.dates[day].date()} "
                f"cost={planned_cost + emergency_cost:.1f} "
                f"emerg={emergency.sum():.1f} soc_end={initial_soc:.1f}",
                flush=True,
            )

    daily = pd.DataFrame(records)
    output = daily[pd.to_datetime(daily["date"]) >= OUTPUT_START].copy()
    summary = {
        "name": "problem3_baseline_no_adjust",
        "description": (
            "Dynamic load+delta; PV = 附件3 0:00 forecast (linear midpoint); "
            "no 6/12/18 adjust; cost = c·x + 5c·e"
        ),
        "config": config.name,
        "days_solved": int(days),
        "output_days": int(len(output)),
        "planned_purchase_kwh": float(output["planned_purchase_kwh"].sum()),
        "planned_cost_yuan": float(output["planned_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(output["actual_emergency_kwh"].sum()),
        "emergency_cost_yuan": float(output["actual_emergency_cost_yuan"].sum()),
        "total_cost_yuan": float(output["total_cost_yuan"].sum()),
        "mean_daily_cost_yuan": float(output["total_cost_yuan"].mean()),
        "mean_delta_kwh": float(output["delta_kwh_per_interval"].mean()),
        "pv_curtailment_kwh": float(output["pv_curtailment_kwh"].sum()),
        "grid_spill_kwh": float(output["grid_spill_kwh"].sum()),
        "total_solver_seconds": float(total_solver),
        "bad_soc_days": int(bad_soc),
        "simultaneous_intervals": int(bad_sim),
        "bad_balance_rows": int(bad_bal),
        "note": "非正式交付；未写入 result3.xlsx",
    }

    arrays = {
        "dates": np.array([r["date"] for r in records]),
        "x": np.stack(xs),
        "charge": np.stack(us),
        "discharge": np.stack(vs),
        "emergency": np.stack(es),
        "soc": np.stack(socs),
    }
    return daily, summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=40.0)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily, summary, arrays = run_baseline(args.time_limit, args.max_days)

    daily_path = OUT_DIR / "problem3_baseline_daily.csv"
    summary_path = OUT_DIR / "problem3_baseline_stats.json"
    npz_path = OUT_DIR / "problem3_baseline_daily.npz"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(npz_path, **arrays)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"daily -> {daily_path}", flush=True)
    print(f"stats -> {summary_path}", flush=True)
    print(f"npz   -> {npz_path}", flush=True)


if __name__ == "__main__":
    main()

"""Problem 4-3 causal runners: A, B0, bstar ablation, and bfreeze main."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from ...common import dataset as p2_common
from ..problem2.solve_problem2_yesterday_dynamic import (
    paper_config,
    choose_candidate,
    choose_delta,
)
from ...common import physical
from ...common.physical import rebuild_soc, settle_contract
from ..problem3.problem3_adjust_milp import solve_stage
from .export_result4 import export_result4_3
from .p4_common import (
    AUDIT_DIR,
    FORMAL_DIR,
    ISSUE_HOURS,
    OUTPUT_START,
    SOC_INITIAL,
    SOC_MAX,
    SOC_MIN,
)
from .p4_data import load_bundle, issue_start_index
from .p4_price import forecast_day, refresh_intraday
from .p4_settle import settle_p3


def run_p4_3(
    *,
    mode: str = "bstar",
    price_method: str = "wd2",
    time_limit: float = 40.0,
    max_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    mode:
      - a: no intraday adjust
      - b0: PV rolling, yesterday/held price, freeze storage, no underbuy, hold-gate
      - bfreeze: PV rolling, WD2(+refresh), freeze storage (C8 fallback main)
      - bstar: PV rolling, price refresh, unfreeze storage (ablation; worse on this data)
    """
    mode = mode.lower()
    if mode not in {"a", "b0", "bfreeze", "bstar"}:
        raise ValueError(mode)

    bundle = load_bundle()
    days = len(bundle.dates) if max_days is None else max_days
    config = paper_config()
    # B0 uses yesterday price and never refreshes (口径对照).
    effective_price_method = "yest" if mode == "b0" else price_method
    allow_underbuy = False
    hold_gate = True
    refresh_price = mode in {"bstar", "bfreeze"}
    freeze_on_adjust = mode in {"b0", "bfreeze"}
    do_adjust = mode in {"b0", "bfreeze", "bstar"}

    initial_soc = float(SOC_INITIAL)
    records: list[dict[str, Any]] = []
    x_plans, x_acts, us, vs, es, socs = [], [], [], [], [], []
    plan_refs, totals = [], []

    for day in range(days):
        price_raw, price_src = forecast_day(
            effective_price_method,
            bundle.prices_att4,
            bundle.prices_att1,
            bundle.dates,
            day,
        )
        p2 = p2_common.from_arrays(bundle.dates, price_raw, bundle.load_kwh, bundle.pv_kwh)
        choice = choose_candidate(
            day, p2, bundle.cold_load, bundle.cold_pv, config
        )
        delta_choice = choose_delta(
            day, choice.candidate, initial_soc, p2, bundle.cold_load, bundle.cold_pv, config
        )
        delta = float(delta_choice.delta)
        load0 = choice.load + delta if config.reserve_in_milp else choice.load.copy()
        pv0 = bundle.forecast_kwh[day, 0, :]
        plan = physical.solve_physical_day(
            load0, pv0, price_raw, initial_soc, time_limit
        )
        solver_s = float(plan.solver_seconds)
        x_plan = plan.x.copy()
        x_act = x_plan.copy()
        charge = plan.charge.copy()
        discharge = plan.discharge.copy()
        soc = rebuild_soc(initial_soc, charge, discharge)
        accepted = 0

        if do_adjust:
            for issue_hour in (6, 12, 18):
                t0 = issue_start_index(issue_hour)
                issue_idx = ISSUE_HOURS.index(issue_hour)
                pv_tau = bundle.forecast_kwh[day, issue_idx, :]
                if refresh_price:
                    price_tau = refresh_intraday(
                        price_raw, bundle.prices_att4[day], t0
                    )
                else:
                    price_tau = price_raw
                stage = solve_stage(
                    t_start=t0,
                    load_forecast=load0,
                    pv_forecast=pv_tau,
                    prices=price_tau,
                    initial_soc=float(soc[t0]),
                    x_plan=x_plan,
                    locked_x=x_act,
                    locked_charge=charge,
                    locked_discharge=discharge,
                    time_limit=time_limit,
                    freeze_storage=freeze_on_adjust,
                )
                solver_s += float(stage.solver_seconds)
                hold = settle_contract(
                    load0, pv_tau, x_act, charge, discharge, price_tau, x_plan
                )
                trial = settle_contract(
                    load0,
                    pv_tau,
                    stage.x,
                    stage.charge,
                    stage.discharge,
                    price_tau,
                    x_plan,
                )
                accept = (not hold_gate) or (
                    float(trial["cost_total"]) + 1e-6 < float(hold["cost_total"])
                )
                if accept:
                    x_act[t0:] = stage.x[t0:]
                    charge[t0:] = stage.charge[t0:]
                    discharge[t0:] = stage.discharge[t0:]
                    soc = rebuild_soc(initial_soc, charge, discharge)
                    accepted += 1

        actual_p = bundle.prices_att4[day]
        settled = settle_p3(
            bundle.load_kwh[day],
            bundle.pv_kwh[day],
            x_plan,
            x_act,
            charge,
            discharge,
            actual_p,
        )
        emergency = np.asarray(settled["emergency"], dtype=float)
        records.append(
            {
                "strategy": f"p4_3_{mode}_{effective_price_method}",
                "date": str(bundle.dates[day].date()),
                "day_index": day,
                "mode": mode,
                "price_method": effective_price_method,
                "price_forecast_source": price_src,
                "selected_candidate": choice.candidate,
                "delta_kwh_per_interval": delta,
                "price_forecast_mae": float(np.mean(np.abs(price_raw - actual_p))),
                "adjustment_accept_count": accepted,
                "initial_soc_kwh": float(soc[0]),
                "terminal_soc_kwh": float(soc[-1]),
                "planned_purchase_kwh": float(x_plan.sum()),
                "actual_purchase_kwh": float(x_act.sum()),
                "plan_ref_cost_yuan": float(settled["plan_ref_cost"]),
                "contract_cost_yuan": float(settled["contract_cost"]),
                "emergency_kwh": float(emergency.sum()),
                "emergency_cost_yuan": float(settled["emergency_cost"]),
                "total_cost_yuan": float(settled["total_cost"]),
                "solver_seconds": solver_s,
                "soc_ok": bool(
                    np.min(soc) >= SOC_MIN - 1e-5 and np.max(soc) <= SOC_MAX + 1e-5
                ),
                "balance_max_abs": float(np.max(np.abs(settled["balance"]))),
            }
        )
        x_plans.append(x_plan)
        x_acts.append(x_act)
        us.append(charge)
        vs.append(discharge)
        es.append(emergency)
        socs.append(soc)
        plan_refs.append(float(settled["plan_ref_cost"]))
        totals.append(float(settled["total_cost"]))
        initial_soc = float(soc[-1])
        if (day + 1) % 30 == 0 or day == 0:
            print(
                f"[P4-3/{mode}] day {day + 1}/{days} {bundle.dates[day].date()} "
                f"cost={settled['total_cost']:.1f} emerg={emergency.sum():.1f} "
                f"accept={accepted}",
                flush=True,
            )

    daily = pd.DataFrame(records)
    output = daily[pd.to_datetime(daily["date"]) >= OUTPUT_START]
    summary = {
        "name": f"problem4_p4_3_{mode}_{effective_price_method}",
        "mode": mode,
        "price_method": effective_price_method,
        "refresh_price_intraday": refresh_price,
        "freeze_storage_on_adjust": freeze_on_adjust,
        "allow_underbuy": allow_underbuy,
        "hold_gate": hold_gate,
        "config": config.name,
        "days_solved": int(days),
        "output_days": int(len(output)),
        "planned_purchase_kwh": float(output["planned_purchase_kwh"].sum()),
        "actual_purchase_kwh": float(output["actual_purchase_kwh"].sum()),
        "contract_cost_yuan": float(output["contract_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(output["emergency_kwh"].sum()),
        "emergency_cost_yuan": float(output["emergency_cost_yuan"].sum()),
        "total_cost_yuan": float(output["total_cost_yuan"].sum()),
        "mean_daily_cost_yuan": float(output["total_cost_yuan"].mean()),
        "accepted_adjustments": int(output["adjustment_accept_count"].sum()),
        "mean_price_forecast_mae": float(output["price_forecast_mae"].mean()),
        "terminal_soc_last_kwh": float(daily["terminal_soc_kwh"].iloc[-1]),
        "soc_bad_days": int((~daily["soc_ok"]).sum()),
        "total_solver_seconds": float(daily["solver_seconds"].sum()),
        "口径": {
            "C5": "contract A with actual price",
            "C8": (
                "no intraday adjustment (A)"
                if not do_adjust
                else (
                    "freeze storage on adjustment (formal bfreeze/b0)"
                    if freeze_on_adjust
                    else "unfreeze remaining storage on adjustment (bstar ablation)"
                )
            ),
            "C9": "no underbuy",
            "C10": "PV-only load",
            "C11": "hold-gate on",
        },
    }
    arrays = {
        "dates": np.array([r["date"] for r in records]),
        "x_plan": np.stack(x_plans),
        "x_act": np.stack(x_acts),
        "charge": np.stack(us),
        "discharge": np.stack(vs),
        "emergency": np.stack(es),
        "soc": np.stack(socs),
        "plan_ref_cost": np.asarray(plan_refs, dtype=float),
        "cost_total": np.asarray(totals, dtype=float),
    }
    return daily, summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default="bfreeze", choices=["a", "b0", "bfreeze", "bstar"]
    )
    parser.add_argument(
        "--price-method", default="wd2", choices=["wd2", "yest", "att1"]
    )
    parser.add_argument("--time-limit", type=float, default=40.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    out_dir = AUDIT_DIR / "p4_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily, summary, arrays = run_p4_3(
        mode=args.mode,
        price_method=args.price_method,
        time_limit=args.time_limit,
        max_days=args.max_days,
    )
    tag = f"{args.mode}_{args.price_method if args.mode != 'b0' else 'yest'}"
    if not args.no_write:
        daily.to_csv(
            out_dir / f"p4_3_{tag}_daily.csv", index=False, encoding="utf-8-sig"
        )
        (out_dir / f"p4_3_{tag}_stats.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            out_dir / f"p4_3_{tag}_arrays.npz",
            **{k: v for k, v in arrays.items() if k != "dates"},
            dates=arrays["dates"],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.export:
        if args.max_days is not None:
            raise SystemExit("full year required for export")
        path = export_result4_3(arrays, FORMAL_DIR / "result4-3.xlsx")
        print(f"result4-3 -> {path}", flush=True)


if __name__ == "__main__":
    main()

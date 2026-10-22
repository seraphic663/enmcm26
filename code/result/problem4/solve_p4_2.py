"""Problem 4-2 causal runners (Att1 / Yest / WD2)."""

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
from ...common.physical import rebuild_soc
from .export_result4 import export_result4_2
from .p4_common import (
    AUDIT_DIR,
    FORMAL_DIR,
    OUTPUT_START,
    SOC_INITIAL,
    SOC_MAX,
    SOC_MIN,
)
from .p4_data import load_bundle
from .p4_price import forecast_day
from .p4_settle import settle_p2


def run_p4_2(
    *,
    price_method: str = "wd2",
    time_limit: float = 40.0,
    max_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    bundle = load_bundle()
    days = len(bundle.dates) if max_days is None else max_days
    config = paper_config()
    initial_soc = float(SOC_INITIAL)
    records: list[dict[str, Any]] = []
    xs, us, vs, es, socs = [], [], [], [], []
    planned_costs: list[float] = []

    for day in range(days):
        price_fc, price_src = forecast_day(
            price_method,
            bundle.prices_att4,
            bundle.prices_att1,
            bundle.dates,
            day,
        )
        p2 = p2_common.from_arrays(bundle.dates, price_fc, bundle.load_kwh, bundle.pv_kwh)
        choice = choose_candidate(
            day, p2, bundle.cold_load, bundle.cold_pv, config
        )
        delta_choice = choose_delta(
            day, choice.candidate, initial_soc, p2, bundle.cold_load, bundle.cold_pv, config
        )
        delta = float(delta_choice.delta)
        load_fc = choice.load + delta if config.reserve_in_milp else choice.load.copy()
        pv_fc = np.asarray(choice.pv, dtype=float)
        plan = physical.solve_physical_day(
            load_fc, pv_fc, price_fc, initial_soc, time_limit
        )
        x = plan.x.copy()
        charge = plan.charge.copy()
        discharge = plan.discharge.copy()
        soc = rebuild_soc(initial_soc, charge, discharge)
        actual_p = bundle.prices_att4[day]
        settled = settle_p2(
            bundle.load_kwh[day],
            bundle.pv_kwh[day],
            x,
            charge,
            discharge,
            actual_p,
        )
        emergency = np.asarray(settled["emergency"], dtype=float)
        records.append(
            {
                "strategy": f"p4_2_{price_method}",
                "date": str(bundle.dates[day].date()),
                "day_index": day,
                "price_method": price_method,
                "price_forecast_source": price_src,
                "selected_candidate": choice.candidate,
                "delta_kwh_per_interval": delta,
                "price_forecast_mae": float(np.mean(np.abs(price_fc - actual_p))),
                "initial_soc_kwh": float(soc[0]),
                "terminal_soc_kwh": float(soc[-1]),
                "planned_purchase_kwh": float(x.sum()),
                "planned_cost_yuan": float(settled["planned_cost"]),
                "emergency_kwh": float(emergency.sum()),
                "emergency_cost_yuan": float(settled["emergency_cost"]),
                "total_cost_yuan": float(settled["total_cost"]),
                "pv_curtailment_kwh": float(settled["pv_curtailment"].sum()),
                "grid_spill_kwh": float(settled["grid_spill"].sum()),
                "solver_seconds": float(plan.solver_seconds),
                "soc_ok": bool(
                    np.min(soc) >= SOC_MIN - 1e-5 and np.max(soc) <= SOC_MAX + 1e-5
                ),
                "balance_max_abs": float(np.max(np.abs(settled["balance"]))),
            }
        )
        xs.append(x)
        us.append(charge)
        vs.append(discharge)
        es.append(emergency)
        socs.append(soc)
        planned_costs.append(float(settled["planned_cost"]))
        initial_soc = float(soc[-1])
        if (day + 1) % 30 == 0 or day == 0:
            print(
                f"[P4-2/{price_method}] day {day + 1}/{days} "
                f"{bundle.dates[day].date()} cost={settled['total_cost']:.1f} "
                f"emerg={emergency.sum():.1f}",
                flush=True,
            )

    daily = pd.DataFrame(records)
    output = daily[pd.to_datetime(daily["date"]) >= OUTPUT_START]
    summary = {
        "name": f"problem4_p4_2_{price_method}",
        "price_method": price_method,
        "config": config.name,
        "days_solved": int(days),
        "output_days": int(len(output)),
        "planned_purchase_kwh": float(output["planned_purchase_kwh"].sum()),
        "planned_cost_yuan": float(output["planned_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(output["emergency_kwh"].sum()),
        "emergency_cost_yuan": float(output["emergency_cost_yuan"].sum()),
        "total_cost_yuan": float(output["total_cost_yuan"].sum()),
        "mean_daily_cost_yuan": float(output["total_cost_yuan"].mean()),
        "mean_price_forecast_mae": float(output["price_forecast_mae"].mean()),
        "mean_delta_kwh": float(output["delta_kwh_per_interval"].mean()),
        "terminal_soc_last_kwh": float(daily["terminal_soc_kwh"].iloc[-1]),
        "soc_bad_days": int((~daily["soc_ok"]).sum()),
        "total_solver_seconds": float(daily["solver_seconds"].sum()),
        "口径": {
            "optimize_price": "causal forecast (C3)",
            "settle_price": "attachment4 actual (C4)",
            "load_pv": "Dynamic",
        },
    }
    arrays = {
        "dates": np.array([r["date"] for r in records]),
        "x": np.stack(xs),
        "charge": np.stack(us),
        "discharge": np.stack(vs),
        "emergency": np.stack(es),
        "soc": np.stack(socs),
        "planned_cost": np.asarray(planned_costs, dtype=float),
    }
    return daily, summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-method",
        default="wd2",
        choices=["wd2", "yest", "att1"],
    )
    parser.add_argument("--time-limit", type=float, default=40.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--export", action="store_true", help="Write result4-2.xlsx")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    out_dir = AUDIT_DIR / "p4_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily, summary, arrays = run_p4_2(
        price_method=args.price_method,
        time_limit=args.time_limit,
        max_days=args.max_days,
    )
    tag = args.price_method
    if not args.no_write:
        daily.to_csv(
            out_dir / f"p4_2_{tag}_daily.csv", index=False, encoding="utf-8-sig"
        )
        (out_dir / f"p4_2_{tag}_stats.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            out_dir / f"p4_2_{tag}_arrays.npz",
            **{k: v for k, v in arrays.items() if k != "dates"},
            dates=arrays["dates"],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.export:
        if args.max_days is not None:
            raise SystemExit("full year required for export")
        path = export_result4_2(arrays, FORMAL_DIR / "result4-2.xlsx")
        print(f"result4-2 -> {path}", flush=True)


if __name__ == "__main__":
    main()

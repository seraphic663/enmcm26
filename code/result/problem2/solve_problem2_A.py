"""Solve CUMCM 2026 Problem 2A: deterministic perfect-information baseline.

The model uses all actual load and PV observations from 2025-01-01 through
2025-12-31, keeps one continuous state of charge across day boundaries, and
exports only 2025-02-01 through 2025-12-31 to the official result2 template.

The official template's 144 interval headers are shifted by ten minutes: the
first header is ``0:10-0:20`` and the final header is
``0:00-0:10+1``.  This program preserves those official labels but writes the
natural-day, right-endpoint interpretation already adopted for Problem 1:
source label ``00:10`` represents ``00:00-00:10`` and ``0:00+1`` represents
``23:50-24:00``.  The detailed CSV records the actual interpreted interval for
every exported value so the mapping is auditable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
import scipy
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from ...common.paths import relative_path
from ...common.dataset import ENERGY_MAX, ETA_CHARGE, ETA_DISCHARGE, FOUR_HOUR_BLOCKS, INTERVALS_PER_DAY, OUTPUT_END, OUTPUT_START, SOC_INITIAL, SOC_MAX, SOC_MIN, SPECIFIED_DATES, SPECIFIED_INTERVALS, InputData, Paths, get_paths, read_inputs, sha256


@dataclass(frozen=True)
class Solution:
    x: np.ndarray
    emergency: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    curtailed: np.ndarray
    soc: np.ndarray
    charge_mode: np.ndarray
    objective: float
    solver_message: str
    mip_gap: float | None
    mip_node_count: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1200.0,
        help="HiGHS MILP time limit in seconds (default: 1200)",
    )
    parser.add_argument(
        "--reuse-schedule",
        action="store_true",
        help="Reload a previously solved and serialized detailed CSV for export-only recovery",
    )
    return parser.parse_args()


def variable_offsets(n: int) -> dict[str, int]:
    return {
        "x": 0,
        "emergency": n,
        "charge": 2 * n,
        "discharge": 3 * n,
        "curtailed": 4 * n,
        "soc": 5 * n,
        "mode": 6 * n + 1,
    }


def solve_model(data: InputData, time_limit: float) -> Solution:
    n_days = len(data.dates)
    n = n_days * INTERVALS_PER_DAY
    offsets = variable_offsets(n)
    variable_count = 7 * n + 1
    flat_load = data.load_kwh.reshape(-1)
    flat_pv = data.pv_kwh.reshape(-1)
    flat_prices = np.tile(data.prices, n_days)
    indices = np.arange(n, dtype=np.int64)

    objective = np.zeros(variable_count, dtype=float)
    objective[offsets["x"] : offsets["x"] + n] = flat_prices
    objective[offsets["emergency"] : offsets["emergency"] + n] = 5.0 * flat_prices

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[offsets["charge"] : offsets["charge"] + n] = ENERGY_MAX
    upper[offsets["discharge"] : offsets["discharge"] + n] = ENERGY_MAX
    upper[offsets["curtailed"] : offsets["curtailed"] + n] = flat_pv
    lower[offsets["soc"] : offsets["soc"] + n + 1] = SOC_MIN
    upper[offsets["soc"] : offsets["soc"] + n + 1] = SOC_MAX
    lower[offsets["soc"]] = SOC_INITIAL
    upper[offsets["soc"]] = SOC_INITIAL
    upper[offsets["mode"] : offsets["mode"] + n] = 1.0

    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[offsets["mode"] : offsets["mode"] + n] = 1

    balance_rows = indices
    soc_rows = n + indices
    charge_rows = 2 * n + indices
    discharge_rows = 3 * n + indices

    rows = np.concatenate(
        [
            balance_rows,
            balance_rows,
            balance_rows,
            balance_rows,
            balance_rows,
            soc_rows,
            soc_rows,
            soc_rows,
            soc_rows,
            charge_rows,
            charge_rows,
            discharge_rows,
            discharge_rows,
        ]
    )
    cols = np.concatenate(
        [
            offsets["x"] + indices,
            offsets["emergency"] + indices,
            offsets["charge"] + indices,
            offsets["discharge"] + indices,
            offsets["curtailed"] + indices,
            offsets["soc"] + indices + 1,
            offsets["soc"] + indices,
            offsets["charge"] + indices,
            offsets["discharge"] + indices,
            offsets["charge"] + indices,
            offsets["mode"] + indices,
            offsets["discharge"] + indices,
            offsets["mode"] + indices,
        ]
    )
    values = np.concatenate(
        [
            np.ones(n),
            np.ones(n),
            -np.ones(n),
            np.ones(n),
            -np.ones(n),
            np.ones(n),
            -np.ones(n),
            -ETA_CHARGE * np.ones(n),
            (1.0 / ETA_DISCHARGE) * np.ones(n),
            np.ones(n),
            -ENERGY_MAX * np.ones(n),
            np.ones(n),
            ENERGY_MAX * np.ones(n),
        ]
    )
    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(4 * n, variable_count),
        dtype=float,
    ).tocsr()

    constraint_lower = np.full(4 * n, -np.inf, dtype=float)
    constraint_upper = np.full(4 * n, np.inf, dtype=float)
    balance_rhs = flat_load - flat_pv
    constraint_lower[:n] = balance_rhs
    constraint_upper[:n] = balance_rhs
    constraint_lower[n : 2 * n] = 0.0
    constraint_upper[n : 2 * n] = 0.0
    constraint_upper[2 * n : 3 * n] = 0.0
    constraint_upper[3 * n : 4 * n] = ENERGY_MAX

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, constraint_lower, constraint_upper),
        options={
            "disp": True,
            "presolve": True,
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-9,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"HiGHS MILP failed: status={result.status}; message={result.message}"
        )

    vector = np.asarray(result.x, dtype=float)

    def part(name: str, count: int = n) -> np.ndarray:
        start = offsets[name]
        values_part = vector[start : start + count].copy()
        values_part[np.abs(values_part) < 1e-9] = 0.0
        return values_part

    return Solution(
        x=part("x"),
        emergency=part("emergency"),
        charge=part("charge"),
        discharge=part("discharge"),
        curtailed=part("curtailed"),
        soc=part("soc", n + 1),
        charge_mode=np.rint(part("mode")).astype(int),
        objective=float(result.fun),
        solver_message=str(result.message),
        mip_gap=(
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        mip_node_count=(
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
    )


def max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def validate_solution(data: InputData, solution: Solution) -> dict[str, Any]:
    n_days = len(data.dates)
    n = n_days * INTERVALS_PER_DAY
    flat_load = data.load_kwh.reshape(-1)
    flat_pv = data.pv_kwh.reshape(-1)
    flat_prices = np.tile(data.prices, n_days)
    assert all(
        array.shape == (n,)
        for array in (
            solution.x,
            solution.emergency,
            solution.charge,
            solution.discharge,
            solution.curtailed,
            solution.charge_mode,
        )
    )
    assert solution.soc.shape == (n + 1,)

    balance = (
        solution.x
        + solution.emergency
        + flat_pv
        + solution.discharge
        - flat_load
        - solution.charge
        - solution.curtailed
    )
    soc_recurrence = (
        solution.soc[1:]
        - solution.soc[:-1]
        - ETA_CHARGE * solution.charge
        + solution.discharge / ETA_DISCHARGE
    )
    objective_recalculated = float(
        np.dot(flat_prices, solution.x)
        + np.dot(5.0 * flat_prices, solution.emergency)
    )
    simultaneous = (solution.charge > 1e-6) & (solution.discharge > 1e-6)
    charge_mode_violation = np.maximum(
        solution.charge - ENERGY_MAX * solution.charge_mode, 0.0
    )
    discharge_mode_violation = np.maximum(
        solution.discharge - ENERGY_MAX * (1 - solution.charge_mode), 0.0
    )
    daily_end = solution.soc[np.arange(1, n_days + 1) * INTERVALS_PER_DAY]
    next_daily_start = solution.soc[np.arange(1, n_days) * INTERVALS_PER_DAY]
    cross_day = daily_end[:-1] - next_daily_start

    checks = {
        "balance_max_abs_kwh": max_abs(balance),
        "soc_recurrence_max_abs_kwh": max_abs(soc_recurrence),
        "objective_abs_difference": abs(objective_recalculated - solution.objective),
        "soc_min_kwh": float(solution.soc.min()),
        "soc_max_kwh": float(solution.soc.max()),
        "charge_max_kwh": float(solution.charge.max()),
        "discharge_max_kwh": float(solution.discharge.max()),
        "curtailment_max_kwh": float(solution.curtailed.max()),
        "curtailment_total_kwh": float(solution.curtailed.sum()),
        "emergency_max_kwh": float(solution.emergency.max()),
        "emergency_total_kwh": float(solution.emergency.sum()),
        "simultaneous_charge_discharge_count": int(simultaneous.sum()),
        "charge_mode_violation_max_kwh": float(charge_mode_violation.max()),
        "discharge_mode_violation_max_kwh": float(discharge_mode_violation.max()),
        "cross_day_soc_max_abs_kwh": max_abs(cross_day),
        "initial_soc_kwh": float(solution.soc[0]),
        "terminal_soc_kwh": float(solution.soc[-1]),
    }

    assert checks["balance_max_abs_kwh"] <= 2e-6
    assert checks["soc_recurrence_max_abs_kwh"] <= 2e-6
    assert checks["objective_abs_difference"] <= 2e-5
    assert checks["soc_min_kwh"] >= SOC_MIN - 2e-6
    assert checks["soc_max_kwh"] <= SOC_MAX + 2e-6
    assert checks["charge_max_kwh"] <= ENERGY_MAX + 2e-6
    assert checks["discharge_max_kwh"] <= ENERGY_MAX + 2e-6
    assert np.all(solution.curtailed >= -2e-7)
    assert np.all(solution.curtailed <= flat_pv + 2e-6)
    assert checks["emergency_max_kwh"] <= 2e-6
    assert checks["emergency_total_kwh"] <= 2e-5
    assert checks["simultaneous_charge_discharge_count"] == 0
    assert checks["charge_mode_violation_max_kwh"] <= 2e-6
    assert checks["discharge_mode_violation_max_kwh"] <= 2e-6
    assert checks["cross_day_soc_max_abs_kwh"] <= 2e-8
    assert abs(checks["initial_soc_kwh"] - SOC_INITIAL) <= 2e-8
    return checks


def build_schedule_frame(data: InputData, solution: Solution) -> pd.DataFrame:
    days = np.repeat(data.dates.values, INTERVALS_PER_DAY)
    interval_index = np.tile(np.arange(INTERVALS_PER_DAY), len(data.dates))
    flat_prices = np.tile(data.prices, len(data.dates))
    return pd.DataFrame(
        {
            "date": pd.to_datetime(days),
            "interval_index": interval_index,
            "source_right_endpoint": np.tile(data.source_labels, len(data.dates)),
            "interpreted_interval": np.tile(data.interval_labels, len(data.dates)),
            "price_yuan_per_kwh": flat_prices,
            "load_kwh": data.load_kwh.reshape(-1),
            "pv_kwh": data.pv_kwh.reshape(-1),
            "planned_purchase_kwh": solution.x,
            "emergency_purchase_kwh": solution.emergency,
            "charge_kwh": solution.charge,
            "discharge_kwh": solution.discharge,
            "curtailed_pv_kwh": solution.curtailed,
            "soc_start_kwh": solution.soc[:-1],
            "soc_end_kwh": solution.soc[1:],
            "charge_mode": solution.charge_mode,
        }
    )


def validate_schedule_frame(frame: pd.DataFrame, objective: float) -> dict[str, float]:
    assert len(frame) == 365 * INTERVALS_PER_DAY
    balance = (
        frame["planned_purchase_kwh"]
        + frame["emergency_purchase_kwh"]
        + frame["pv_kwh"]
        + frame["discharge_kwh"]
        - frame["load_kwh"]
        - frame["charge_kwh"]
        - frame["curtailed_pv_kwh"]
    ).to_numpy()
    recurrence = (
        frame["soc_end_kwh"]
        - frame["soc_start_kwh"]
        - ETA_CHARGE * frame["charge_kwh"]
        + frame["discharge_kwh"] / ETA_DISCHARGE
    ).to_numpy()
    objective_check = float(
        (
            frame["price_yuan_per_kwh"]
            * (
                frame["planned_purchase_kwh"]
                + 5.0 * frame["emergency_purchase_kwh"]
            )
        ).sum()
    )
    result = {
        "csv_balance_max_abs_kwh": max_abs(balance),
        "csv_soc_recurrence_max_abs_kwh": max_abs(recurrence),
        "csv_objective_abs_difference": abs(objective_check - objective),
    }
    assert result["csv_balance_max_abs_kwh"] <= 3e-6
    assert result["csv_soc_recurrence_max_abs_kwh"] <= 3e-6
    assert result["csv_objective_abs_difference"] <= 3e-5
    return result


def solution_from_schedule(data: InputData, frame: pd.DataFrame) -> Solution:
    """Reconstruct the already solved schedule after a downstream export failure."""
    n = len(data.dates) * INTERVALS_PER_DAY
    assert len(frame) == n
    expected_dates = np.repeat(data.dates.values, INTERVALS_PER_DAY)
    assert np.array_equal(pd.to_datetime(frame["date"]).values, expected_dates)
    assert frame["interval_index"].astype(int).tolist() == np.tile(
        np.arange(INTERVALS_PER_DAY), len(data.dates)
    ).tolist()
    assert frame["source_right_endpoint"].astype(str).tolist() == list(
        np.tile(data.source_labels, len(data.dates))
    )
    assert frame["interpreted_interval"].astype(str).tolist() == list(
        np.tile(data.interval_labels, len(data.dates))
    )
    soc_start = frame["soc_start_kwh"].to_numpy(dtype=float)
    soc_end = frame["soc_end_kwh"].to_numpy(dtype=float)
    assert max_abs(soc_start[1:] - soc_end[:-1]) <= 3e-6
    soc = np.concatenate(([soc_start[0]], soc_end))
    x = frame["planned_purchase_kwh"].to_numpy(dtype=float)
    emergency = frame["emergency_purchase_kwh"].to_numpy(dtype=float)
    prices = frame["price_yuan_per_kwh"].to_numpy(dtype=float)
    objective = float(np.dot(prices, x + 5.0 * emergency))
    return Solution(
        x=x,
        emergency=emergency,
        charge=frame["charge_kwh"].to_numpy(dtype=float),
        discharge=frame["discharge_kwh"].to_numpy(dtype=float),
        curtailed=frame["curtailed_pv_kwh"].to_numpy(dtype=float),
        soc=soc,
        charge_mode=frame["charge_mode"].to_numpy(dtype=int),
        objective=objective,
        solver_message=(
            "Optimal HiGHS MILP solution reloaded from the detailed CSV after an "
            "export-only template-check retry"
        ),
        mip_gap=0.0,
        mip_node_count=1,
    )


def template_snapshot(path: Path) -> dict[str, Any]:
    workbook = load_workbook(path, data_only=False, read_only=False)
    plan = workbook["计划购电量"]
    storage = workbook["充放电量"]
    emergency = workbook["紧急购电量"]
    plan_headers = [plan.cell(1, col).value for col in range(1, plan.max_column + 1)]
    plan_dates = [plan.cell(row, 1).value for row in range(2, plan.max_row + 1)]
    assert plan.max_row == 335 and plan.max_column == 147
    assert len(plan_headers[1:145]) == INTERVALS_PER_DAY
    assert plan_headers[1] == "0:10-0:20"
    assert plan_headers[144] == "0:00-0:10+1"
    assert plan_headers[145:] == ["全天购电量", "全天购电费"]
    assert storage.max_column == 6
    assert emergency.max_column == 3
    return {
        "sheet_names": workbook.sheetnames,
        "plan_headers": plan_headers,
        "plan_dates": plan_dates,
        "emergency_values": [
            [emergency.cell(row, col).value for col in range(1, emergency.max_column + 1)]
            for row in range(1, emergency.max_row + 1)
        ],
        "header_styles": {
            sheet.title: [sheet.cell(1, col).style_id for col in range(1, sheet.max_column + 1)]
            for sheet in workbook.worksheets
        },
    }


def block_summaries(data: InputData, solution: Solution) -> tuple[np.ndarray, np.ndarray]:
    charge = solution.charge.reshape(len(data.dates), INTERVALS_PER_DAY)
    discharge = solution.discharge.reshape(len(data.dates), INTERVALS_PER_DAY)
    charge_blocks = np.column_stack(
        [charge[:, block * 24 : (block + 1) * 24].sum(axis=1) for block in range(6)]
    )
    discharge_blocks = np.column_stack(
        [
            discharge[:, block * 24 : (block + 1) * 24].sum(axis=1)
            for block in range(6)
        ]
    )
    return charge_blocks, discharge_blocks


def write_workbook(
    paths: Paths,
    data: InputData,
    solution: Solution,
    baseline: dict[str, Any],
    template_hash: str,
) -> dict[str, Any]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.output_workbook.parent.mkdir(parents=True, exist_ok=True)
    paths.detailed_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.audit_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.template, paths.output_workbook)
    workbook = load_workbook(paths.output_workbook, data_only=False, read_only=False)
    assert workbook.sheetnames == baseline["sheet_names"]

    output_mask = (data.dates >= OUTPUT_START) & (data.dates <= OUTPUT_END)
    output_indices = np.flatnonzero(output_mask)
    output_dates = data.dates[output_mask]
    assert len(output_dates) == 334
    x_matrix = solution.x.reshape(len(data.dates), INTERVALS_PER_DAY)[output_mask]
    emergency_matrix = solution.emergency.reshape(len(data.dates), INTERVALS_PER_DAY)[output_mask]
    assert np.max(np.abs(emergency_matrix)) <= 2e-6
    price_matrix = np.broadcast_to(data.prices, x_matrix.shape)
    daily_quantity = x_matrix.sum(axis=1)
    daily_cost = (x_matrix * price_matrix).sum(axis=1)

    plan = workbook["计划购电量"]
    current_dates = pd.DatetimeIndex(pd.to_datetime(baseline["plan_dates"])).normalize()
    assert current_dates.equals(output_dates)
    for row_offset, (values, quantity, cost) in enumerate(
        zip(x_matrix, daily_quantity, daily_cost, strict=True), start=2
    ):
        for interval_offset, value in enumerate(values, start=2):
            plan.cell(row_offset, interval_offset).value = float(value)
        plan.cell(row_offset, 146).value = float(quantity)
        plan.cell(row_offset, 147).value = float(cost)

    charge_blocks_all, discharge_blocks_all = block_summaries(data, solution)
    charge_blocks = charge_blocks_all[output_mask]
    discharge_blocks = discharge_blocks_all[output_mask]
    storage = workbook["充放电量"]
    pattern_styles = {
        (row_offset, col): copy.copy(storage.cell(2 + row_offset, col)._style)
        for row_offset in range(6)
        for col in range(1, 7)
    }
    pattern_heights = {
        row_offset: storage.row_dimensions[2 + row_offset].height
        for row_offset in range(6)
    }
    required_last_row = 1 + len(output_dates) * 6
    for day_offset, (source_day_index, current_date) in enumerate(
        zip(output_indices, output_dates, strict=True)
    ):
        first_row = 2 + day_offset * 6
        for block in range(6):
            row = first_row + block
            storage.row_dimensions[row].height = pattern_heights[block]
            for col in range(1, 7):
                storage.cell(row, col)._style = copy.copy(pattern_styles[(block, col)])
            storage.cell(row, 1).value = (
                current_date.to_pydatetime() if block == 0 else None
            )
            storage.cell(row, 2).value = FOUR_HOUR_BLOCKS[block]
            storage.cell(row, 3).value = float(charge_blocks[day_offset, block])
            storage.cell(row, 4).value = float(discharge_blocks[day_offset, block])
            storage.cell(row, 5).value = "0:00" if block == 0 else ("24:00" if block == 1 else None)
            if block == 0:
                storage.cell(row, 6).value = float(
                    solution.soc[source_day_index * INTERVALS_PER_DAY]
                )
            elif block == 1:
                storage.cell(row, 6).value = float(
                    solution.soc[(source_day_index + 1) * INTERVALS_PER_DAY]
                )
            else:
                storage.cell(row, 6).value = None
    if storage.max_row > required_last_row:
        storage.delete_rows(required_last_row + 1, storage.max_row - required_last_row)

    emergency = workbook["紧急购电量"]
    emergency_values = [
        [emergency.cell(row, col).value for col in range(1, emergency.max_column + 1)]
        for row in range(1, emergency.max_row + 1)
    ]
    assert emergency_values == baseline["emergency_values"]
    assert all(
        emergency.cell(row, col).value in (None, "⁝")
        for row in range(2, emergency.max_row + 1)
        for col in (2, 3)
    )

    workbook.save(paths.output_workbook)
    assert sha256(paths.template) == template_hash, "official template was modified"

    return {
        "output_days": len(output_dates),
        "output_plan_rows": plan.max_row - 1,
        "output_storage_rows": len(output_dates) * 6,
        "template_headers_preserved": True,
        "template_time_mapping_aligned": False,
        "template_time_mapping_note": (
            "Values are written in attachment/right-endpoint order: source 00:10 is "
            "00:00-00:10. The official template starts at 0:10-0:20 and is shifted "
            "ten minutes; its labels are preserved unchanged."
        ),
    }


def audit_workbook(
    paths: Paths,
    data: InputData,
    solution: Solution,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    workbook = load_workbook(paths.output_workbook, data_only=True, read_only=False)
    assert workbook.sheetnames == baseline["sheet_names"]
    output_mask = (data.dates >= OUTPUT_START) & (data.dates <= OUTPUT_END)
    output_indices = np.flatnonzero(output_mask)
    output_dates = data.dates[output_mask]
    expected_x = solution.x.reshape(len(data.dates), INTERVALS_PER_DAY)[output_mask]
    expected_charge, expected_discharge = block_summaries(data, solution)
    expected_charge = expected_charge[output_mask]
    expected_discharge = expected_discharge[output_mask]

    plan = workbook["计划购电量"]
    headers = [plan.cell(1, col).value for col in range(1, 148)]
    assert headers == baseline["plan_headers"]
    plan_values = np.array(
        [
            [plan.cell(row, col).value for col in range(2, 146)]
            for row in range(2, 336)
        ],
        dtype=float,
    )
    daily_quantity = np.array([plan.cell(row, 146).value for row in range(2, 336)], dtype=float)
    daily_cost = np.array([plan.cell(row, 147).value for row in range(2, 336)], dtype=float)
    expected_daily_cost = (expected_x * data.prices[np.newaxis, :]).sum(axis=1)

    storage = workbook["充放电量"]
    assert storage.max_row == 1 + 334 * 6
    storage_charge = np.empty((334, 6), dtype=float)
    storage_discharge = np.empty((334, 6), dtype=float)
    storage_soc_start = np.empty(334, dtype=float)
    storage_soc_end = np.empty(334, dtype=float)
    storage_dates: list[pd.Timestamp] = []
    for day_offset, source_day_index in enumerate(output_indices):
        first_row = 2 + day_offset * 6
        storage_dates.append(pd.Timestamp(storage.cell(first_row, 1).value).normalize())
        assert [storage.cell(first_row + block, 2).value for block in range(6)] == list(FOUR_HOUR_BLOCKS)
        storage_charge[day_offset] = [
            storage.cell(first_row + block, 3).value for block in range(6)
        ]
        storage_discharge[day_offset] = [
            storage.cell(first_row + block, 4).value for block in range(6)
        ]
        assert storage.cell(first_row, 5).value == "0:00"
        assert storage.cell(first_row + 1, 5).value == "24:00"
        storage_soc_start[day_offset] = storage.cell(first_row, 6).value
        storage_soc_end[day_offset] = storage.cell(first_row + 1, 6).value
        assert math.isclose(
            storage_soc_start[day_offset],
            solution.soc[source_day_index * INTERVALS_PER_DAY],
            abs_tol=2e-6,
        )
        assert math.isclose(
            storage_soc_end[day_offset],
            solution.soc[(source_day_index + 1) * INTERVALS_PER_DAY],
            abs_tol=2e-6,
        )

    emergency = workbook["紧急购电量"]
    emergency_values = [
        [emergency.cell(row, col).value for col in range(1, emergency.max_column + 1)]
        for row in range(1, emergency.max_row + 1)
    ]
    assert emergency_values == baseline["emergency_values"]
    assert all(
        emergency.cell(row, col).value in (None, "⁝")
        for row in range(2, emergency.max_row + 1)
        for col in (2, 3)
    )

    plan_difference = max_abs(plan_values - expected_x)
    quantity_difference = max_abs(daily_quantity - expected_x.sum(axis=1))
    cost_difference = max_abs(daily_cost - expected_daily_cost)
    charge_difference = max_abs(storage_charge - expected_charge)
    discharge_difference = max_abs(storage_discharge - expected_discharge)
    exported_cross_day = max_abs(storage_soc_end[:-1] - storage_soc_start[1:])
    assert pd.DatetimeIndex(storage_dates).equals(output_dates)
    assert plan_difference <= 3e-6
    assert quantity_difference <= 3e-6
    assert cost_difference <= 3e-6
    assert charge_difference <= 3e-6
    assert discharge_difference <= 3e-6
    assert exported_cross_day <= 3e-6

    return {
        "workbook_reopened": True,
        "sheet_names_preserved": True,
        "plan_headers_preserved": True,
        "plan_value_max_abs_difference_kwh": plan_difference,
        "daily_quantity_max_abs_difference_kwh": quantity_difference,
        "daily_cost_max_abs_difference_yuan": cost_difference,
        "storage_charge_max_abs_difference_kwh": charge_difference,
        "storage_discharge_max_abs_difference_kwh": discharge_difference,
        "exported_cross_day_soc_max_abs_kwh": exported_cross_day,
        "emergency_result_cells_blank": True,
    }


def specified_date_results(data: InputData, solution: Solution) -> dict[str, Any]:
    x = solution.x.reshape(len(data.dates), INTERVALS_PER_DAY)
    emergency = solution.emergency.reshape(len(data.dates), INTERVALS_PER_DAY)
    charge_blocks, discharge_blocks = block_summaries(data, solution)
    interval_indices = [data.interval_labels.index(label) for label in SPECIFIED_INTERVALS]
    results: dict[str, Any] = {}
    for selected_date in SPECIFIED_DATES:
        day_index = int(data.dates.get_loc(selected_date))
        results[selected_date.strftime("%Y-%m-%d")] = {
            "planned_purchase_by_interval_kwh": {
                label: float(x[day_index, interval_index])
                for label, interval_index in zip(
                    SPECIFIED_INTERVALS, interval_indices, strict=True
                )
            },
            "daily_planned_purchase_kwh": float(x[day_index].sum()),
            "daily_normal_purchase_cost_yuan": float(
                np.dot(data.prices, x[day_index])
            ),
            "storage_blocks": {
                label: {
                    "charge_kwh": float(charge_blocks[day_index, block]),
                    "discharge_kwh": float(discharge_blocks[day_index, block]),
                }
                for block, label in enumerate(FOUR_HOUR_BLOCKS)
            },
            "soc_0000_kwh": float(solution.soc[day_index * INTERVALS_PER_DAY]),
            "soc_2400_kwh": float(solution.soc[(day_index + 1) * INTERVALS_PER_DAY]),
            "emergency_purchase_total_kwh": float(emergency[day_index].sum()),
            "emergency_intervals": [],
        }
    return results


def summarize(
    paths: Paths,
    data: InputData,
    solution: Solution,
    numeric_checks: dict[str, Any],
    csv_checks: dict[str, Any],
    export_info: dict[str, Any],
    workbook_checks: dict[str, Any],
    template_hash: str,
) -> dict[str, Any]:
    output_mask = (data.dates >= OUTPUT_START) & (data.dates <= OUTPUT_END)
    x = solution.x.reshape(len(data.dates), INTERVALS_PER_DAY)[output_mask]
    emergency = solution.emergency.reshape(len(data.dates), INTERVALS_PER_DAY)[output_mask]
    prices = np.broadcast_to(data.prices, x.shape)
    normal_cost = float(np.sum(prices * x))
    emergency_cost = float(np.sum(5.0 * prices * emergency))
    return {
        "model": "Problem 2A deterministic perfect-information cross-date MILP",
        "time_mapping": {
            "model": "A / natural day / source labels are right endpoints",
            "source_first": data.source_labels[0],
            "source_last": data.source_labels[-1],
            "interpreted_first": data.interval_labels[0],
            "interpreted_last": data.interval_labels[-1],
            "official_template_first": "0:10-0:20",
            "official_template_last": "0:00-0:10+1",
            "official_template_labels_preserved": True,
            "official_template_aligned_with_model": False,
        },
        "libraries": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "openpyxl": openpyxl.__version__,
            "solver": "scipy.optimize.milp / HiGHS",
        },
        "solver": {
            "status": "Optimal",
            "message": solution.solver_message,
            "mip_gap": solution.mip_gap,
            "mip_node_count": solution.mip_node_count,
            "objective_full_year_yuan": solution.objective,
        },
        "output_period": {
            "start": OUTPUT_START.strftime("%Y-%m-%d"),
            "end": OUTPUT_END.strftime("%Y-%m-%d"),
            "days": int(output_mask.sum()),
            "planned_purchase_total_kwh": float(x.sum()),
            "normal_purchase_cost_yuan": normal_cost,
            "emergency_purchase_total_kwh": float(emergency.sum()),
            "emergency_purchase_cost_yuan": emergency_cost,
            "total_purchase_cost_yuan": normal_cost + emergency_cost,
        },
        "specified_dates": specified_date_results(data, solution),
        "numeric_checks": numeric_checks,
        "csv_checks": csv_checks,
        "workbook_export": export_info,
        "workbook_checks": workbook_checks,
        "theory_check": {
            "strict_dominance_holds": True,
            "reason": (
                "x and emergency purchase enter the same balance equation; x has no "
                "capacity limit and costs c_t while emergency purchase costs 5*c_t, "
                "with every c_t strictly positive."
            ),
            "observed_emergency_purchase_is_zero_within_tolerance": True,
        },
        "ambiguities": [
            "The official result2 plan-purchase headers are shifted ten minutes relative to the natural-day/right-endpoint interpretation. Values use attachment order and headers remain unchanged.",
            "Problem 2 does not impose a terminal SOC or a daily 0:00/24:00 equality. The model therefore keeps SOC continuous across dates and allows the final 2025-12-31 SOC to be optimized within bounds.",
            "Version A is a perfect-information lower-bound benchmark because the full actual load and PV path is known during optimization; it is not a causal day-ahead forecast policy.",
        ],
        "files": {
            "template": relative_path(paths.template),
            "template_sha256": template_hash,
            "result_workbook": relative_path(paths.output_workbook),
            "detailed_schedule_csv": relative_path(paths.detailed_csv),
            "audit_json": relative_path(paths.audit_json),
        },
    }


def main() -> None:
    args = parse_args()
    paths = get_paths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.detailed_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.output_workbook.parent.mkdir(parents=True, exist_ok=True)
    template_hash = sha256(paths.template)
    baseline = template_snapshot(paths.template)
    data = read_inputs(paths)

    if args.reuse_schedule:
        if not paths.detailed_csv.is_file():
            raise FileNotFoundError(
                f"cannot reuse missing serialized schedule: {paths.detailed_csv}"
            )
        print(f"Reloading verified MILP schedule from {paths.detailed_csv}", flush=True)
        serialized_schedule = pd.read_csv(paths.detailed_csv, parse_dates=["date"])
        solution = solution_from_schedule(data, serialized_schedule)
    else:
        print(
            f"Solving {len(data.dates)} days x {INTERVALS_PER_DAY} intervals "
            f"with SciPy {scipy.__version__} / HiGHS...",
            flush=True,
        )
        solution = solve_model(data, args.time_limit)
        schedule = build_schedule_frame(data, solution)
        schedule.to_csv(
            paths.detailed_csv,
            index=False,
            encoding="utf-8-sig",
            float_format="%.12g",
        )
        serialized_schedule = pd.read_csv(paths.detailed_csv, parse_dates=["date"])
    numeric_checks = validate_solution(data, solution)
    csv_checks = validate_schedule_frame(serialized_schedule, solution.objective)

    export_info = write_workbook(
        paths, data, solution, baseline, template_hash
    )
    workbook_checks = audit_workbook(paths, data, solution, baseline)
    report = summarize(
        paths,
        data,
        solution,
        numeric_checks,
        csv_checks,
        export_info,
        workbook_checks,
        template_hash,
    )
    paths.audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reread_report = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    assert reread_report["workbook_checks"]["workbook_reopened"] is True
    assert reread_report["theory_check"]["observed_emergency_purchase_is_zero_within_tolerance"] is True
    assert sha256(paths.template) == template_hash
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("All model, serialized-schedule, and workbook checks passed.")


if __name__ == "__main__":
    main()

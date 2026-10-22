"""Shared physical single-day model used by P2, P3 and P4.

The deterministic A model keeps its own full-horizon MILP because its
variable layout and Excel audit are different.  This module owns the smaller
single-day binary MILP and the actual-data settlement used by Dynamic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from . import dataset as common


N = common.INTERVALS_PER_DAY
DT = common.DELTA_HOURS
SOC_MIN = common.SOC_MIN
SOC_MAX = common.SOC_MAX
ENERGY_MAX = common.ENERGY_MAX
ETA_C = common.ETA_CHARGE
ETA_D = common.ETA_DISCHARGE


@dataclass(frozen=True)
class DayPlan:
    x: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    objective: float
    solver_seconds: float
    simultaneous_count: int
    soc_residual_max: float


def read_cold_start(paths: common.Paths) -> tuple[np.ndarray, np.ndarray]:
    """Read the typical-day profile used only on 2025-01-01."""

    frame = pd.read_excel(paths.attachment1, sheet_name="Sheet1")
    if frame.shape != (N, 4):
        raise ValueError(f"cold-start profile shape mismatch: {frame.shape}")
    load = pd.to_numeric(frame.iloc[:, 2], errors="raise").to_numpy(float) * DT
    pv = pd.to_numeric(frame.iloc[:, 3], errors="raise").to_numpy(float) * DT
    if np.any(load < 0.0) or np.any(pv < 0.0):
        raise ValueError("cold-start profile contains negative values")
    return load, pv


def add_entry(
    rows: list[int], cols: list[int], values: list[float], row: int, col: int, value: float
) -> None:
    rows.append(row)
    cols.append(col)
    values.append(value)


def solve_physical_day(
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    prices: np.ndarray,
    initial_soc: float,
    time_limit: float,
) -> DayPlan:
    """Solve one physical single-scenario day-ahead MILP."""

    if load_forecast.shape != (N,) or pv_forecast.shape != (N,):
        raise ValueError("forecast profile must have 144 intervals")
    offsets = {
        "x": 0,
        "charge": N,
        "discharge": 2 * N,
        "soc": 3 * N,
        "mode": 4 * N,
        "emergency": 5 * N,
        "curtailment": 6 * N,
    }
    variable_count = 7 * N
    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[offsets["charge"] : offsets["charge"] + N] = ENERGY_MAX
    upper[offsets["discharge"] : offsets["discharge"] + N] = ENERGY_MAX
    lower[offsets["soc"] : offsets["soc"] + N] = SOC_MIN
    upper[offsets["soc"] : offsets["soc"] + N] = SOC_MAX
    upper[offsets["mode"] : offsets["mode"] + N] = 1.0
    upper[offsets["curtailment"] : offsets["curtailment"] + N] = pv_forecast

    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[offsets["mode"] : offsets["mode"] + N] = 1
    objective = np.zeros(variable_count, dtype=float)
    objective[offsets["x"] : offsets["x"] + N] = prices
    objective[offsets["emergency"] : offsets["emergency"] + N] = 5.0 * prices

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower_rows: list[float] = []
    upper_rows: list[float] = []
    row = 0

    # x + PV + discharge + emergency = load + charge + PV curtailment.
    for t in range(N):
        add_entry(rows, cols, values, row, offsets["x"] + t, 1.0)
        add_entry(rows, cols, values, row, offsets["charge"] + t, -1.0)
        add_entry(rows, cols, values, row, offsets["discharge"] + t, 1.0)
        add_entry(rows, cols, values, row, offsets["emergency"] + t, 1.0)
        add_entry(rows, cols, values, row, offsets["curtailment"] + t, -1.0)
        rhs = float(load_forecast[t] - pv_forecast[t])
        lower_rows.append(rhs)
        upper_rows.append(rhs)
        row += 1

    # E_{t+1} = E_t + eta_c*charge - discharge/eta_d, E0 is fixed.
    for t in range(N):
        add_entry(rows, cols, values, row, offsets["soc"] + t, 1.0)
        if t > 0:
            add_entry(rows, cols, values, row, offsets["soc"] + t - 1, -1.0)
        add_entry(rows, cols, values, row, offsets["charge"] + t, -ETA_C)
        add_entry(rows, cols, values, row, offsets["discharge"] + t, 1.0 / ETA_D)
        lower_rows.append(float(initial_soc) if t == 0 else 0.0)
        upper_rows.append(float(initial_soc) if t == 0 else 0.0)
        row += 1

    # Binary mode: a slot permits charge or discharge, not both.
    for t in range(N):
        add_entry(rows, cols, values, row, offsets["charge"] + t, 1.0)
        add_entry(rows, cols, values, row, offsets["mode"] + t, -ENERGY_MAX)
        lower_rows.append(-np.inf)
        upper_rows.append(0.0)
        row += 1
    for t in range(N):
        add_entry(rows, cols, values, row, offsets["discharge"] + t, 1.0)
        add_entry(rows, cols, values, row, offsets["mode"] + t, ENERGY_MAX)
        lower_rows.append(-np.inf)
        upper_rows.append(ENERGY_MAX)
        row += 1

    matrix = coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
        shape=(row, variable_count),
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix, np.asarray(lower_rows), np.asarray(upper_rows)
        ),
        options={
            "disp": False,
            "presolve": True,
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-8,
        },
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None:
        raise RuntimeError(f"Dynamic MILP failed: {result.status} {result.message}")

    vector = np.asarray(result.x, dtype=float)
    x = vector[offsets["x"] : offsets["x"] + N].copy()
    charge = vector[offsets["charge"] : offsets["charge"] + N].copy()
    discharge = vector[offsets["discharge"] : offsets["discharge"] + N].copy()
    soc = np.concatenate(
        [np.array([float(initial_soc)]), vector[offsets["soc"] : offsets["soc"] + N]]
    )
    soc_residual = soc[1:] - soc[:-1] - ETA_C * charge + discharge / ETA_D
    return DayPlan(
        x=x,
        charge=charge,
        discharge=discharge,
        soc=soc,
        objective=float(result.fun),
        solver_seconds=elapsed,
        simultaneous_count=int(((charge > 1e-6) & (discharge > 1e-6)).sum()),
        soc_residual_max=float(np.max(np.abs(soc_residual))),
    )


def settle_actual(
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    plan: DayPlan,
    redundancy: float,
) -> dict[str, float | np.ndarray]:
    """Settle actual data and separate PV curtailment from grid spill."""

    return settle_flows(
        actual_load, actual_pv, plan.x + float(redundancy), plan.charge, plan.discharge
    )


def settle_flows(
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    x: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
) -> dict[str, np.ndarray]:
    """Settle energy balance without adding a problem-specific price policy."""
    gap = actual_load + charge - x - actual_pv - discharge
    emergency = np.maximum(gap, 0.0)
    surplus = np.maximum(-gap, 0.0)
    pv_curtailment = np.minimum(surplus, actual_pv)
    grid_spill = np.maximum(surplus - actual_pv, 0.0)
    balance = x + actual_pv + discharge + emergency - actual_load - charge - surplus
    return {
        "emergency": emergency,
        "surplus": surplus,
        "pv_curtailment": pv_curtailment,
        "grid_spill": grid_spill,
        "balance": balance,
    }


def settle_contract(
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    x: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    prices: np.ndarray,
    x_plan: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Settle the retained P3/P4 contract with the 0.5-price deviation charge."""
    flows = settle_flows(actual_load, actual_pv, x, charge, discharge)
    cost_contract, cost_emerg = contract_costs(x, x_plan, prices, flows["emergency"])
    return {
        "emergency": flows["emergency"],
        "surplus": flows["surplus"],
        "cost_contract": cost_contract,
        "cost_emerg": cost_emerg,
        "cost_total": cost_contract + cost_emerg,
        "cost_plan_ref": float(np.sum(prices * x_plan)),
    }


def contract_costs(
    x: np.ndarray, x_plan: np.ndarray, prices: np.ndarray, emergency: np.ndarray
) -> tuple[float, float]:
    """Return contract and emergency costs with the retained calculation order."""
    delta = x - x_plan
    contract = prices * x + 0.5 * prices * np.abs(delta)
    return float(np.sum(contract)), float(np.sum(5.0 * prices * emergency))


def rebuild_soc(
    initial_soc: float, charge: np.ndarray, discharge: np.ndarray
) -> np.ndarray:
    soc = np.zeros(N + 1, dtype=float)
    soc[0] = float(initial_soc)
    for t in range(N):
        soc[t + 1] = soc[t] + 0.9 * charge[t] - discharge[t] / 0.9
    return soc

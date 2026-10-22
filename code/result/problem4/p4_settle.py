"""Settlement and SOC helpers for Problem 4."""

from __future__ import annotations

import numpy as np

from ...common.physical import contract_costs, settle_flows


def settle_p2(
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    x: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    actual_prices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    flows = settle_flows(actual_load, actual_pv, x, charge, discharge)
    planned_cost = float(np.dot(actual_prices, x))
    emergency_cost = float(np.dot(5.0 * actual_prices, flows["emergency"]))
    return {
        **flows,
        "planned_cost": planned_cost,
        "emergency_cost": emergency_cost,
        "total_cost": planned_cost + emergency_cost,
    }


def settle_p3(
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    x_plan: np.ndarray,
    x_act: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    actual_prices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    flows = settle_flows(actual_load, actual_pv, x_act, charge, discharge)
    contract_cost, emergency_cost = contract_costs(
        x_act, x_plan, actual_prices, flows["emergency"]
    )
    deviation = x_act - x_plan
    plan_ref = float(np.dot(actual_prices, x_plan))
    return {
        **flows,
        "deviation": deviation,
        "contract_cost": contract_cost,
        "emergency_cost": emergency_cost,
        "plan_ref_cost": plan_ref,
        "total_cost": contract_cost + emergency_cost,
    }

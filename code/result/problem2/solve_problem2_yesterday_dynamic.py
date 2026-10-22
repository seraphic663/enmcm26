"""Production causal Yesterday-MILP-Dynamic runner for Problem 2.

Contains the two paper baselines and the frozen Dynamic production policy.
P3/P4 reuse its causal forecast and reserve selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import protected_output_path, relative_path
from ...common import dataset as common
from ...common import physical


N = common.INTERVALS_PER_DAY
SOC_MIN = common.SOC_MIN
SOC_MAX = common.SOC_MAX
SOC_INITIAL = common.SOC_INITIAL
OUTPUT_START = common.OUTPUT_START
SOC_TARGET = (SOC_MIN + SOC_MAX) / 2.0

BASELINE_DELTA_GRID = (0.0, 50.0, 70.0, 100.0)
FINAL_DELTA_GRID = tuple(float(value) for value in range(0, 141, 10))


@dataclass(frozen=True)
class DynamicConfig:
    """Parameters for the three retained paper strategies."""

    name: str
    description: str
    candidate_mode: str = "selector"
    same_type_history_window: int = 2
    seasonal_strength: float = 0.0
    seasonal_daylight_base_hours: float = 13.3
    seasonal_daylight_amplitude_hours: float = 1.5
    seasonal_daylight_phase_day: float = 80.0
    pv_active_relative_threshold: float = 1e-5
    pv_active_absolute_threshold: float = 1e-8
    load_seasonal_strength: float = 0.0
    load_seasonal_min_history: int = 28
    load_seasonal_factor_min: float = 0.70
    load_seasonal_factor_max: float = 1.30
    bias_gain: float = 0.0
    residual_decay_days: float = 5.0
    residual_clip_kwh: float = 250.0
    delta_mode: str = "dynamic_state_cost"
    default_delta: float = 70.0
    delta_grid: tuple[float, ...] = BASELINE_DELTA_GRID
    delta_emergency_price_multiplier: float = 5.0
    selector_window: int = 7
    bias_window: int = 7
    reserve_in_milp: bool = False
    state_multiplier_strength: float = 0.8
    state_multiplier_min: float = 0.4
    state_multiplier_max: float = 1.6
    soc_target_kwh: float = SOC_TARGET


# Report baseline 1: yesterday's actual curve is used directly as the causal
# day-ahead forecast; no post-solve redundancy is added.
YESTERDAY_BASELINE_CONFIG = DynamicConfig(
    name="yesterday_baseline",
    description=(
        "natural causal baseline: yesterday actual profile fed directly "
        "into the physical single-day MILP"
    ),
    candidate_mode="yesterday_only",
    same_type_history_window=2,
    seasonal_strength=0.0,
    load_seasonal_strength=0.0,
    load_seasonal_min_history=28,
    bias_gain=0.0,
    residual_decay_days=5.0,
    residual_clip_kwh=250.0,
    delta_mode="fixed",
    default_delta=0.0,
    delta_grid=BASELINE_DELTA_GRID,
    delta_emergency_price_multiplier=5.0,
    selector_window=7,
    bias_window=7,
    reserve_in_milp=False,
    state_multiplier_strength=0.8,
    state_multiplier_min=0.4,
    state_multiplier_max=1.6,
    soc_target_kwh=SOC_TARGET,
)

# Report baseline 2: the same yesterday forecast, with a fixed 70 kWh per
# interval added after the base MILP solve as the historical comparison.
YESTERDAY_PLUS70_CONFIG = DynamicConfig(
    name="yesterday_baseline_plus70",
    description=(
        "yesterday-only MILP baseline with a fixed 70 kWh per 10-minute "
        "slot added after the base solve"
    ),
    candidate_mode="yesterday_only",
    same_type_history_window=2,
    seasonal_strength=0.0,
    load_seasonal_strength=0.0,
    load_seasonal_min_history=28,
    bias_gain=0.0,
    residual_decay_days=5.0,
    residual_clip_kwh=250.0,
    delta_mode="fixed",
    default_delta=70.0,
    delta_grid=BASELINE_DELTA_GRID,
    delta_emergency_price_multiplier=5.0,
    selector_window=7,
    bias_window=7,
    reserve_in_milp=False,
    state_multiplier_strength=0.8,
    state_multiplier_min=0.4,
    state_multiplier_max=1.6,
    soc_target_kwh=SOC_TARGET,
)

# Frozen production Dynamic configuration.  The values are deliberately
# written out rather than obtained from a search or a runtime config registry.
FINAL_DYNAMIC_CONFIG = DynamicConfig(
    name="dynamic_full_seasonal_bias75_dense_delta",
    description=(
        "causal date/weekday seasonal load-level correction, 25% seasonal "
        "PV shape, 75% residual correction, and reserve-aware MILP"
    ),
    candidate_mode="selector",
    same_type_history_window=2,
    seasonal_strength=0.25,
    seasonal_daylight_base_hours=13.3,
    seasonal_daylight_amplitude_hours=1.5,
    seasonal_daylight_phase_day=80.0,
    pv_active_relative_threshold=1e-5,
    pv_active_absolute_threshold=1e-8,
    load_seasonal_strength=0.5,
    load_seasonal_min_history=28,
    load_seasonal_factor_min=0.70,
    load_seasonal_factor_max=1.30,
    bias_gain=0.75,
    residual_decay_days=5.0,
    residual_clip_kwh=250.0,
    delta_mode="dynamic_state_cost",
    default_delta=70.0,
    delta_grid=FINAL_DELTA_GRID,
    delta_emergency_price_multiplier=5.0,
    selector_window=7,
    bias_window=7,
    reserve_in_milp=True,
    state_multiplier_strength=0.8,
    state_multiplier_min=0.4,
    state_multiplier_max=1.6,
    soc_target_kwh=SOC_TARGET,
)

PRODUCTION_CONFIGS = (
    YESTERDAY_BASELINE_CONFIG,
    YESTERDAY_PLUS70_CONFIG,
    FINAL_DYNAMIC_CONFIG,
)


@dataclass(frozen=True)
class ForecastChoice:
    candidate: str
    load: np.ndarray
    pv: np.ndarray
    selected_score: float
    candidate_scores: dict[str, float]


@dataclass(frozen=True)
class DeltaChoice:
    delta: float
    scores: dict[float, float]
    state_multiplier: float
    residual_count: int


@dataclass(frozen=True)
class DayOutcome:
    arrays: dict[str, np.ndarray]
    record: dict[str, float | int | str]
    terminal_soc: float
    bad_soc: int
    bad_simultaneous: int
    bad_balance: int


@dataclass(frozen=True)
class DynamicResult:
    config: DynamicConfig
    summary: dict[str, float | int | str]
    daily: pd.DataFrame
    arrays: dict[str, np.ndarray]


def raw_candidate_profile(
    day: int,
    candidate: str,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    same_type_history_window: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return a strictly causal raw candidate profile."""

    if day == 0:
        return cold_load.copy(), cold_pv.copy(), (-1,)

    prior = np.arange(day, dtype=int)
    if candidate == "yesterday":
        source = int(day - 1)
        return data.load_kwh[source].copy(), data.pv_kwh[source].copy(), (source,)

    if candidate == "weekday2":
        weekdays = data.dates.weekday.to_numpy()
        history_window = max(1, int(same_type_history_window))
        selected = prior[weekdays[prior] == weekdays[day]][-history_window:]
        if len(selected) == 0:
            selected = prior[-min(history_window, len(prior)) :]
        return (
            data.load_kwh[selected].mean(axis=0).copy(),
            data.pv_kwh[selected].mean(axis=0).copy(),
            tuple(int(item) for item in selected),
        )

    raise ValueError(f"unsupported candidate: {candidate}")


def seasonal_daylight_hours(
    day: int,
    dates: pd.DatetimeIndex,
    base_hours: float = 13.3,
    amplitude_hours: float = 1.5,
    phase_day: float = 80.0,
) -> float:
    """Return a calendar-only daylight-window prior."""

    doy = int(dates[day].dayofyear)
    return float(
        base_hours
        + amplitude_hours * np.sin(2.0 * np.pi * (doy - phase_day) / 365.0)
    )


def calendar_pv_shape(
    day: int,
    dates: pd.DatetimeIndex,
    cold_pv: np.ndarray,
    *,
    daylight_base_hours: float = 13.3,
    daylight_amplitude_hours: float = 1.5,
    daylight_phase_day: float = 80.0,
    active_relative_threshold: float = 1e-5,
    active_absolute_threshold: float = 1e-8,
) -> np.ndarray:
    """Stretch the typical-day PV shape to the date's daylight window."""

    typical = np.asarray(cold_pv, dtype=float)
    active = np.flatnonzero(
        typical
        > max(active_absolute_threshold, float(typical.max()) * active_relative_threshold)
    )
    if len(active) < 2 or float(typical.sum()) <= 0.0:
        return typical.copy()

    daylight_hours = seasonal_daylight_hours(
        day,
        dates,
        daylight_base_hours,
        daylight_amplitude_hours,
        daylight_phase_day,
    )
    center = 12.0
    target_start = center - daylight_hours / 2.0
    target_end = center + daylight_hours / 2.0
    interval_hours = (np.arange(N, dtype=float) + 0.5) / 6.0
    target_hours = np.linspace(target_start, target_end, len(active))
    shape = np.interp(
        interval_hours,
        target_hours,
        typical[active],
        left=0.0,
        right=0.0,
    )
    shape = np.maximum(shape, 0.0)
    if float(shape.sum()) <= 0.0:
        return typical.copy()
    return shape / float(shape.sum())


def apply_seasonal_pv_reference(
    day: int,
    candidate_pv: np.ndarray,
    data: common.InputData,
    cold_pv: np.ndarray,
    strength: float,
    config: DynamicConfig | None = None,
) -> np.ndarray:
    """Blend a date-shaped PV profile while preserving candidate PV energy."""

    if strength <= 0.0:
        return candidate_pv.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    candidate_total = float(np.sum(candidate_pv))
    if candidate_total <= 1e-9:
        return candidate_pv.copy()
    if config is None:
        seasonal_shape = calendar_pv_shape(day, data.dates, cold_pv)
    else:
        seasonal_shape = calendar_pv_shape(
            day,
            data.dates,
            cold_pv,
            daylight_base_hours=config.seasonal_daylight_base_hours,
            daylight_amplitude_hours=config.seasonal_daylight_amplitude_hours,
            daylight_phase_day=config.seasonal_daylight_phase_day,
            active_relative_threshold=config.pv_active_relative_threshold,
            active_absolute_threshold=config.pv_active_absolute_threshold,
        )
    seasonal_profile = seasonal_shape * candidate_total
    return (1.0 - strength) * candidate_pv + strength * seasonal_profile


def causal_seasonal_load_total(
    day: int, data: common.InputData, min_history: int
) -> float | None:
    """Predict target-day total load from prior weekday and date harmonics."""

    if day < max(min_history, 12):
        return None
    prior = np.arange(day, dtype=int)
    weekdays = data.dates.weekday.to_numpy()
    day_of_year = data.dates.dayofyear.to_numpy()
    theta = 2.0 * np.pi * (day_of_year[prior] - 1.0) / 365.0
    # Intercept + six weekday dummies + annual and semiannual harmonics.
    # Leaving out weekday 6 avoids exact collinearity with the intercept.
    design = np.column_stack(
        [
            np.ones(day, dtype=float),
            *[(weekdays[prior] == weekday).astype(float) for weekday in range(6)],
            np.sin(theta),
            np.cos(theta),
            np.sin(2.0 * theta),
            np.cos(2.0 * theta),
        ]
    )
    target_theta = 2.0 * np.pi * (day_of_year[day] - 1.0) / 365.0
    target = np.asarray(
        [
            1.0,
            *[float(weekdays[day] == weekday) for weekday in range(6)],
            np.sin(target_theta),
            np.cos(target_theta),
            np.sin(2.0 * target_theta),
            np.cos(2.0 * target_theta),
        ],
        dtype=float,
    )
    coefficients = np.linalg.lstsq(
        design, data.load_kwh[prior].sum(axis=1), rcond=None
    )[0]
    prediction = float(target @ coefficients)
    return max(prediction, 0.0)


def apply_seasonal_load_reference(
    day: int,
    candidate_load: np.ndarray,
    data: common.InputData,
    strength: float,
    min_history: int,
    factor_min: float = 0.70,
    factor_max: float = 1.30,
) -> np.ndarray:
    """Apply a causal date/weekday level correction without changing shape."""

    if strength <= 0.0:
        return candidate_load.copy()
    predicted_total = causal_seasonal_load_total(day, data, min_history)
    candidate_total = float(np.sum(candidate_load))
    if predicted_total is None or candidate_total <= 1e-9:
        return candidate_load.copy()
    target_factor = float(
        np.clip(predicted_total / candidate_total, factor_min, factor_max)
    )
    factor = 1.0 + float(np.clip(strength, 0.0, 1.0)) * (target_factor - 1.0)
    return np.maximum(candidate_load * factor, 0.0)


def raw_candidate_with_season(
    day: int,
    candidate: str,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    load, pv, sources = raw_candidate_profile(
        day,
        candidate,
        data,
        cold_load,
        cold_pv,
        config.same_type_history_window,
    )
    load = apply_seasonal_load_reference(
        day,
        load,
        data,
        config.load_seasonal_strength,
        config.load_seasonal_min_history,
        config.load_seasonal_factor_min,
        config.load_seasonal_factor_max,
    )
    pv = apply_seasonal_pv_reference(
        day, pv, data, cold_pv, config.seasonal_strength, config
    )
    return load, pv, sources


def residual_history(
    target_day: int,
    candidate: str,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal residual profiles and their source day indices."""

    start = max(0, target_day - window)
    residuals: list[np.ndarray] = []
    source_days: list[int] = []
    for validation_day in range(start, target_day):
        forecast_load, forecast_pv, _ = raw_candidate_with_season(
            validation_day, candidate, data, cold_load, cold_pv, config
        )
        forecast_net = forecast_load - forecast_pv
        actual_net = data.load_kwh[validation_day] - data.pv_kwh[validation_day]
        residuals.append(actual_net - forecast_net)
        source_days.append(validation_day)
    if not residuals:
        return np.empty((0, N), dtype=float), np.empty(0, dtype=int)
    return np.vstack(residuals), np.asarray(source_days, dtype=int)


def robust_bias_profile(
    target_day: int,
    candidate: str,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
) -> np.ndarray:
    """Use a recency-weighted, clipped mean of causal net-load residuals."""

    residuals, source_days = residual_history(
        target_day,
        candidate,
        data,
        cold_load,
        cold_pv,
        config,
        config.bias_window,
    )
    if len(source_days) == 0 or config.bias_gain == 0.0:
        return np.zeros(N, dtype=float)

    ages = target_day - 1 - source_days.astype(float)
    weights = np.exp(-ages / config.residual_decay_days)
    weights /= float(weights.sum())
    # Clip each validation residual so one cloudy/high-load day does not turn
    # the correction into an accidental large fixed reserve.
    clipped = np.clip(
        residuals, -config.residual_clip_kwh, config.residual_clip_kwh
    )
    return np.sum(clipped * weights[:, None], axis=0)


def candidate_forecast(
    target_day: int,
    candidate: str,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a causal candidate forecast with its net-load bias correction."""

    load, pv, _ = raw_candidate_with_season(
        target_day, candidate, data, cold_load, cold_pv, config
    )
    bias = robust_bias_profile(
        target_day, candidate, data, cold_load, cold_pv, config
    )
    if config.bias_gain:
        load = np.maximum(load + config.bias_gain * bias, 0.0)
    return load, pv


def choose_candidate(
    day: int,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
) -> ForecastChoice:
    """Select the lower recent walk-forward net-load MAE candidate."""

    if config.candidate_mode == "yesterday_only":
        candidates = ("yesterday",)
    elif config.candidate_mode == "selector":
        candidates = ("yesterday", "weekday2")
    else:
        raise ValueError(f"unsupported candidate mode: {config.candidate_mode}")
    scores: dict[str, float] = {}
    for candidate in candidates:
        validation_days = range(max(0, day - config.selector_window), day)
        errors: list[float] = []
        for validation_day in validation_days:
            load, pv = candidate_forecast(
                validation_day, candidate, data, cold_load, cold_pv, config
            )
            predicted_net = load - pv
            actual_net = data.load_kwh[validation_day] - data.pv_kwh[validation_day]
            errors.append(float(np.mean(np.abs(actual_net - predicted_net))))
        scores[candidate] = float(np.mean(errors)) if errors else float("inf")

    # A tie goes to weekday2 once it exists because it suppresses one-day
    # weather noise without changing the information boundary.
    if len(candidates) == 1:
        selected = candidates[0]
    elif scores["weekday2"] <= scores["yesterday"]:
        selected = "weekday2"
    else:
        selected = "yesterday"
    load, pv = candidate_forecast(
        day, selected, data, cold_load, cold_pv, config
    )
    return ForecastChoice(
        candidate=selected,
        load=load,
        pv=pv,
        selected_score=scores[selected],
        candidate_scores=scores,
    )


def state_multiplier(
    initial_soc: float, config: DynamicConfig
) -> float:
    """Increase the reserve value when the carried SOC is depleted."""

    target = float(config.soc_target_kwh)
    strength = float(config.state_multiplier_strength)
    minimum = float(config.state_multiplier_min)
    maximum = float(config.state_multiplier_max)
    normalized = (target - float(initial_soc)) / (SOC_MAX - SOC_MIN)
    return float(np.clip(1.0 + strength * normalized, minimum, maximum))


def choose_delta(
    day: int,
    candidate: str,
    initial_soc: float,
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
) -> DeltaChoice:
    """Select reserve from recent causal residual-cost proxies plus SOC state."""

    if config.delta_mode == "fixed":
        return DeltaChoice(
            delta=float(config.default_delta),
            scores={float(config.default_delta): 0.0},
            state_multiplier=state_multiplier(initial_soc, config),
            residual_count=0,
        )

    residuals, source_days = residual_history(
        day,
        candidate,
        data,
        cold_load,
        cold_pv,
        config,
        config.bias_window,
    )
    state_factor = state_multiplier(initial_soc, config)
    if len(source_days) == 0:
        default = float(config.default_delta)
        return DeltaChoice(
            delta=default,
            scores={float(value): float("nan") for value in config.delta_grid},
            state_multiplier=state_factor,
            residual_count=0,
        )

    ages = day - 1 - source_days.astype(float)
    weights = np.exp(-ages / config.residual_decay_days)
    weights /= float(weights.sum())
    # A positive residual means the actual net load exceeded the candidate.
    residuals = residuals - config.bias_gain * np.vstack(
        [
            robust_bias_profile(
                item, candidate, data, cold_load, cold_pv, config
            )
            for item in source_days
        ]
    )
    residuals = np.maximum(residuals, 0.0)
    scores: dict[float, float] = {}
    normal_unit_cost = float(np.sum(data.prices))
    for delta in config.delta_grid:
        reserve_cost = float(delta) * normal_unit_cost
        emergency_proxy = np.sum(
            np.maximum(residuals - float(delta), 0.0)
            * (
                config.delta_emergency_price_multiplier
                * data.prices[None, :]
            ),
            axis=1,
        )
        score = reserve_cost + state_factor * float(np.dot(weights, emergency_proxy))
        scores[float(delta)] = float(score)
    selected = min(scores, key=scores.get)
    return DeltaChoice(
        delta=float(selected),
        scores=scores,
        state_multiplier=state_factor,
        residual_count=len(source_days),
    )


def run_day(
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    day: int,
    initial_soc: float,
    config: DynamicConfig,
    time_limit: float,
) -> DayOutcome:
    """Execute one causal day and return its record plus carried state."""

    choice = choose_candidate(day, data, cold_load, cold_pv, config)
    delta_choice = choose_delta(
        day,
        choice.candidate,
        initial_soc,
        data,
        cold_load,
        cold_pv,
        config,
    )
    delta = delta_choice.delta
    solve_load = choice.load + delta if config.reserve_in_milp else choice.load
    plan = physical.solve_physical_day(
        solve_load,
        choice.pv,
        data.prices,
        initial_soc,
        time_limit,
    )

    bad_soc = int(
        plan.soc_residual_max > 1e-5
        or np.min(plan.soc) < SOC_MIN - 1e-5
        or np.max(plan.soc) > SOC_MAX + 1e-5
    )
    actual = physical.settle_actual(
        data.load_kwh[day],
        data.pv_kwh[day],
        plan,
        0.0 if config.reserve_in_milp else delta,
    )
    balance = np.asarray(actual["balance"], dtype=float)
    balance_max = float(np.max(np.abs(balance)))
    bad_balance = int(balance_max > 1e-5)
    emergency = np.asarray(actual["emergency"], dtype=float)
    surplus = np.asarray(actual["surplus"], dtype=float)
    pv_curtailment = np.asarray(actual["pv_curtailment"], dtype=float)
    grid_spill = np.asarray(actual["grid_spill"], dtype=float)
    x = plan.x if config.reserve_in_milp else plan.x + delta
    planned_cost = float(np.dot(data.prices, x))
    emergency_cost = float(np.dot(5.0 * data.prices, emergency))
    source_yesterday = day - 1 if day > 0 else -1
    record: dict[str, float | int | str] = {
        "variant": config.name,
        "date": str(data.dates[day].date()),
        "day_index": day,
        "selected_candidate": choice.candidate,
        "candidate_yesterday_score_mae_kwh": choice.candidate_scores.get(
            "yesterday", np.nan
        ),
        "candidate_weekday2_score_mae_kwh": choice.candidate_scores.get(
            "weekday2", np.nan
        ),
        "selected_candidate_score_mae_kwh": choice.selected_score,
        "source_yesterday_day": source_yesterday,
        "delta_kwh_per_interval": delta,
        "delta_selection_state_multiplier": delta_choice.state_multiplier,
        "delta_selection_residual_days": delta_choice.residual_count,
        "delta_score_0": delta_choice.scores.get(0.0, np.nan),
        "delta_score_50": delta_choice.scores.get(50.0, np.nan),
        "delta_score_70": delta_choice.scores.get(70.0, np.nan),
        "delta_score_100": delta_choice.scores.get(100.0, np.nan),
        "initial_soc_kwh": initial_soc,
        "terminal_soc_kwh": float(plan.soc[-1]),
        "forecast_load_kwh": float(choice.load.sum()),
        "forecast_pv_kwh": float(choice.pv.sum()),
        "planned_purchase_kwh": float(x.sum()),
        "planned_cost_yuan": planned_cost,
        "actual_emergency_kwh": float(emergency.sum()),
        "actual_emergency_cost_yuan": emergency_cost,
        "actual_surplus_kwh": float(surplus.sum()),
        "pv_curtailment_kwh": float(pv_curtailment.sum()),
        "grid_spill_kwh": float(grid_spill.sum()),
        "total_cost_yuan": planned_cost + emergency_cost,
        "base_milp_objective_yuan": plan.objective,
        "base_simultaneous_count": plan.simultaneous_count,
        "base_soc_residual_max_kwh": plan.soc_residual_max,
        "balance_max_abs_kwh": balance_max,
        "solver_seconds": plan.solver_seconds,
    }
    return DayOutcome(
        arrays={"x":x.copy(),"charge":plan.charge.copy(),"discharge":plan.discharge.copy(),"soc":plan.soc.copy(),"emergency":emergency.copy()},
        record=record,
        terminal_soc=float(plan.soc[-1]),
        bad_soc=bad_soc,
        bad_simultaneous=plan.simultaneous_count,
        bad_balance=bad_balance,
    )


def run_config(
    data: common.InputData,
    cold_load: np.ndarray,
    cold_pv: np.ndarray,
    config: DynamicConfig,
    time_limit: float,
    max_days: int | None = None,
) -> DynamicResult:
    """Run one configuration over the continuous 2025 replay."""

    days = len(data.dates) if max_days is None else min(max_days, len(data.dates))
    records: list[dict[str, float | int | str]] = []
    saved: list[dict[str, np.ndarray]] = []
    initial_soc = float(SOC_INITIAL)
    total_solver_seconds = 0.0
    bad_soc = 0
    bad_simultaneous = 0
    bad_balance = 0

    for day in range(days):
        outcome = run_day(
            data,
            cold_load,
            cold_pv,
            day,
            initial_soc,
            config,
            time_limit,
        )
        records.append(outcome.record)
        saved.append(outcome.arrays)
        initial_soc = outcome.terminal_soc
        total_solver_seconds += float(outcome.record["solver_seconds"])
        bad_soc += outcome.bad_soc
        bad_simultaneous += outcome.bad_simultaneous
        bad_balance += outcome.bad_balance

    daily = pd.DataFrame(records)
    output = daily[pd.to_datetime(daily["date"]) >= OUTPUT_START].copy()
    if output.empty:
        raise ValueError("no output-period rows; increase max_days")
    candidate_counts = output["selected_candidate"].value_counts().to_dict()
    delta_counts = output["delta_kwh_per_interval"].value_counts().to_dict()
    summary: dict[str, float | int | str] = {
        "variant": config.name,
        "description": config.description,
        "days_solved": int(days),
        "output_days": int(len(output)),
        "planned_purchase_kwh": float(output["planned_purchase_kwh"].sum()),
        "planned_cost_yuan": float(output["planned_cost_yuan"].sum()),
        "emergency_purchase_kwh": float(output["actual_emergency_kwh"].sum()),
        "emergency_cost_yuan": float(
            output["actual_emergency_cost_yuan"].sum()
        ),
        "total_cost_yuan": float(output["total_cost_yuan"].sum()),
        "emergency_cost_share": float(
            output["actual_emergency_cost_yuan"].sum()
            / output["total_cost_yuan"].sum()
        ),
        "days_with_emergency": int(
            (output["actual_emergency_kwh"] > 1e-7).sum()
        ),
        "max_daily_emergency_kwh": float(output["actual_emergency_kwh"].max()),
        "pv_curtailment_kwh": float(output["pv_curtailment_kwh"].sum()),
        "grid_spill_kwh": float(output["grid_spill_kwh"].sum()),
        "mean_selected_delta_kwh_per_interval": float(
            output["delta_kwh_per_interval"].mean()
        ),
        "mean_selected_candidate_mae_kwh": float(
            output["selected_candidate_score_mae_kwh"].mean()
        ),
        "selected_yesterday_days": int(candidate_counts.get("yesterday", 0)),
        "selected_weekday2_days": int(candidate_counts.get("weekday2", 0)),
        "delta_0_days": int(delta_counts.get(0.0, 0)),
        "delta_50_days": int(delta_counts.get(50.0, 0)),
        "delta_70_days": int(delta_counts.get(70.0, 0)),
        "delta_100_days": int(delta_counts.get(100.0, 0)),
        "selected_delta_counts_json": json.dumps(
            {str(float(key)): int(value) for key, value in sorted(delta_counts.items())},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "terminal_soc_last_kwh": float(output.iloc[-1]["terminal_soc_kwh"]),
        "total_solver_seconds": float(total_solver_seconds),
        "bad_soc_days": int(bad_soc),
        "simultaneous_charge_discharge_intervals": int(bad_simultaneous),
        "bad_balance_rows": int(bad_balance),
    }
    arrays = {key: np.stack([day[key] for day in saved]) for key in saved[0]}
    arrays["dates"] = data.dates[:days].strftime("%Y-%m-%d").to_numpy(dtype=str)
    return DynamicResult(config=config, summary=summary, daily=daily, arrays=arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.results_root() / "problem2" / "formal" / "dynamic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = common.get_paths()
    data = common.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)
    configs = PRODUCTION_CONFIGS

    output_dir = protected_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float | int | str]] = []
    daily_frames: list[pd.DataFrame] = []
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] {config.name}: {config.description}",
            flush=True,
        )
        result = run_config(
            data,
            cold_load,
            cold_pv,
            config,
            args.time_limit,
            args.max_days,
        )
        summaries.append(result.summary)
        daily_frames.append(result.daily)
        if config == FINAL_DYNAMIC_CONFIG:
            np.savez_compressed(output_dir / "dynamic_intervals.npz", **result.arrays)
            result.daily.to_csv(output_dir / "dynamic_daily.csv", index=False)
        print(
            f"  total={result.summary['total_cost_yuan']:.2f} "
            f"emergency={result.summary['emergency_purchase_kwh']:.2f} "
            f"grid_spill={result.summary['grid_spill_kwh']:.2f} "
            f"mean_delta={result.summary['mean_selected_delta_kwh_per_interval']:.2f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(summaries).sort_values("total_cost_yuan")
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    summary_path = output_dir / "problem2_yesterday_dynamic.csv"
    daily_path = output_dir / "problem2_yesterday_dynamic_daily.csv"
    metadata_path = output_dir / "problem2_yesterday_dynamic.json"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily_frame.to_csv(daily_path, index=False, encoding="utf-8-sig")
    metadata = {
        "model": "causal Yesterday-MILP-Dynamic production runner",
        "information_boundary": (
            "day d uses only dates strictly earlier than d; current SOC is the "
            "carried state at 00:00; no target-day actual enters the day-ahead solve"
        ),
        "candidate_profiles": {
            "yesterday": "actual profile at d-1",
            "weekday2": "mean of the most recent prior same-weekday profiles",
        },
        "selection": "unweighted interval net-load MAE over the immediately prior walk-forward days",
        "state": "SOC-dependent reserve value multiplier plus causal recent net-load residual proxy",
        "seasonal_reference": (
            "date-only smooth daylight-window prior blended into candidate PV shape; "
            "candidate total PV energy is preserved"
        ),
        "load_seasonal_reference": (
            "optional causal expanding least-squares correction using prior weekday "
            "dummies and annual/semiannual date harmonics"
        ),
        "settlement": (
            "PV curtailment is min(actual surplus, actual PV); grid spill is "
            "max(actual surplus - actual PV, 0)"
        ),
        "configs": [config.__dict__ for config in configs],
        "files": {
            "summary_csv": relative_path(summary_path),
            "daily_csv": relative_path(daily_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(f"summary: {summary_path}")
    print(f"daily: {daily_path}")
    print(f"metadata: {metadata_path}")


def paper_config() -> DynamicConfig:
    """Return the fixed cross-problem policy after checking its identity."""
    if FINAL_DYNAMIC_CONFIG.name != "dynamic_full_seasonal_bias75_dense_delta":
        raise ValueError(f"unexpected Dynamic config: {FINAL_DYNAMIC_CONFIG.name}")
    return FINAL_DYNAMIC_CONFIG


if __name__ == "__main__":
    main()

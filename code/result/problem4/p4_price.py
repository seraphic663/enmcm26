"""Causal price forecasts for Problem 4 (口径 C1/C2/C7)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_att1(prices_att1: np.ndarray) -> np.ndarray:
    return np.asarray(prices_att1, dtype=float).copy()


def forecast_yesterday(
    prices_att4: np.ndarray,
    prices_att1: np.ndarray,
    day: int,
) -> tuple[np.ndarray, str]:
    if day <= 0:
        return forecast_att1(prices_att1), "att1_cold_start"
    return prices_att4[day - 1].copy(), f"yesterday_day{day - 1}"


def forecast_wd2(
    prices_att4: np.ndarray,
    prices_att1: np.ndarray,
    dates: pd.DatetimeIndex,
    day: int,
    history_window: int = 2,
) -> tuple[np.ndarray, str]:
    """Mean of up to ``history_window`` prior same-weekday price curves."""

    if day <= 0:
        return forecast_att1(prices_att1), "att1_cold_start"

    target_wd = int(dates[day].dayofweek)
    refs: list[int] = []
    for h in range(day - 1, -1, -1):
        if int(dates[h].dayofweek) == target_wd:
            refs.append(h)
            if len(refs) >= history_window:
                break
    if refs:
        pred = np.mean(prices_att4[refs, :], axis=0)
        return pred.astype(float), f"wd2_refs={refs}"
    return forecast_yesterday(prices_att4, prices_att1, day)


def forecast_day(
    method: str,
    prices_att4: np.ndarray,
    prices_att1: np.ndarray,
    dates: pd.DatetimeIndex,
    day: int,
) -> tuple[np.ndarray, str]:
    method = method.lower()
    if method == "att1":
        return forecast_att1(prices_att1), "att1_fixed"
    if method == "yest":
        return forecast_yesterday(prices_att4, prices_att1, day)
    if method == "wd2":
        return forecast_wd2(prices_att4, prices_att1, dates, day)
    raise ValueError(f"unknown price method: {method}")


def refresh_intraday(
    raw_forecast: np.ndarray,
    actual_prices: np.ndarray,
    t_start: int,
    *,
    scale_lo: float = 0.5,
    scale_hi: float = 2.0,
    min_prefix_mean: float = 1e-4,
) -> np.ndarray:
    """Prefix = realized actual; suffix = scaled raw forecast (口径 C2)."""

    out = np.asarray(raw_forecast, dtype=float).copy()
    actual = np.asarray(actual_prices, dtype=float)
    if t_start <= 0:
        return out
    out[:t_start] = actual[:t_start]
    morn_act = float(actual[:t_start].mean())
    morn_fc = float(raw_forecast[:t_start].mean())
    if morn_act < min_prefix_mean or morn_fc < min_prefix_mean:
        out[t_start:] = raw_forecast[t_start:]
        return out
    scale = float(np.clip(morn_act / (morn_fc + 1e-12), scale_lo, scale_hi))
    out[t_start:] = raw_forecast[t_start:] * scale
    # Keep strictly positive for MILP weights.
    out = np.maximum(out, 1e-6)
    return out

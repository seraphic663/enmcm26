"""Load Attachment 1–4 inputs for the merged Problem 4 implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .p4_common import N, PROJECT_DIR

# Read-only imports from the Problem 3 package.
from ..problem3.problem3_data import (
    issue_start_index,
    load_problem3_data,
)


@dataclass(frozen=True)
class P4Bundle:
    dates: pd.DatetimeIndex
    prices_att4: np.ndarray  # (365, 144)
    prices_att1: np.ndarray  # (144,)
    load_kwh: np.ndarray
    pv_kwh: np.ndarray
    cold_load: np.ndarray
    cold_pv: np.ndarray
    forecast_kwh: np.ndarray  # (365, 4, 144)


def load_bundle(project_root: Path | None = None) -> P4Bundle:
    root = Path(project_root) if project_root else PROJECT_DIR
    p3 = load_problem3_data(root)


    att4_path = root / "附件" / "附件4.xlsx"
    price_frame = pd.read_excel(att4_path, sheet_name="Sheet1")
    if price_frame.shape != (len(p3.dates), N + 1):
        raise ValueError(f"附件4 shape={price_frame.shape}")
    price_dates = pd.DatetimeIndex(
        pd.to_datetime(price_frame.iloc[:, 0], errors="raise")
    ).normalize()
    if not price_dates.equals(p3.dates):
        raise ValueError("附件4 日期与附件2 不一致")
    prices_att4 = price_frame.iloc[:, 1:].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(prices_att4).all() or np.any(prices_att4 <= 0.0):
        raise ValueError("附件4 含非正或非有限电价")


    return P4Bundle(
        dates=p3.dates,
        prices_att4=prices_att4,
        prices_att1=np.asarray(p3.prices, dtype=float),
        load_kwh=np.asarray(p3.load_kwh, dtype=float),
        pv_kwh=np.asarray(p3.pv_kwh, dtype=float),
        cold_load=np.asarray(p3.cold_load, dtype=float),
        cold_pv=np.asarray(p3.cold_pv, dtype=float),
        forecast_kwh=np.asarray(p3.forecast_kwh, dtype=float),
    )


__all__ = [
    "P4Bundle",
    "load_bundle",
    "issue_start_index",
]

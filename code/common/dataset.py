"""Shared input validation, physical constants and time labels for P2-P4."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import project_root, results_root


INTERVALS_PER_DAY = 144
DELTA_HOURS = 1.0 / 6.0
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INITIAL = 6000.0
POWER_MAX_KW = 5000.0
ENERGY_MAX = POWER_MAX_KW * DELTA_HOURS
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
OUTPUT_START = pd.Timestamp("2025-02-01")
OUTPUT_END = pd.Timestamp("2025-12-31")
SPECIFIED_DATES = (
    pd.Timestamp("2025-03-20"),
    pd.Timestamp("2025-06-21"),
    pd.Timestamp("2025-09-23"),
    pd.Timestamp("2025-12-21"),
)
SPECIFIED_INTERVALS = (
    "10:00-10:10",
    "12:00-12:10",
    "14:00-14:10",
    "16:00-16:10",
    "18:00-18:10",
    "20:00-20:10",
)
FOUR_HOUR_BLOCKS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)


@dataclass(frozen=True)
class Paths:
    problem_dir: Path
    attachment1: Path
    attachment2: Path
    template: Path
    output_dir: Path
    output_workbook: Path
    detailed_csv: Path
    audit_json: Path


@dataclass(frozen=True)
class InputData:
    dates: pd.DatetimeIndex
    source_labels: tuple[str, ...]
    interval_labels: tuple[str, ...]
    prices: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray


def get_paths(data_root: Path | None = None) -> Paths:
    problem_dir = Path(data_root) if data_root is not None else project_root()
    output_dir = results_root() / "problem2"
    return Paths(
        problem_dir=problem_dir,
        attachment1=problem_dir / "附件" / "附件1.xlsx",
        attachment2=problem_dir / "附件" / "附件2.xlsx",
        template=problem_dir / "附件" / "附件5" / "result2.xlsx",
        output_dir=output_dir,
        output_workbook=output_dir / "formal" / "result2-A.xlsx",
        detailed_csv=output_dir / "audit" / "problem2_A_schedule.csv",
        audit_json=output_dir / "audit" / "problem2_A_audit.json",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_time_label(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}"
    text = str(value).strip()
    if text in {"0:00+1", "00:00+1"}:
        return "0:00+1"
    if text.endswith(":00") and text.count(":") == 2:
        text = text[:-3]
    parts = text.split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    return text


def clock_label(total_minutes: int) -> str:
    if total_minutes == 1440:
        return "0:00+1"
    if not 0 <= total_minutes < 1440:
        raise ValueError(f"minute outside natural day: {total_minutes}")
    return f"{total_minutes // 60}:{total_minutes % 60:02d}"


def interval_label(index: int) -> str:
    start = index * 10
    end = (index + 1) * 10
    return f"{clock_label(start)}-{clock_label(end)}"


def read_inputs(paths: Paths) -> InputData:
    for required in (paths.attachment1, paths.attachment2):
        if not required.is_file():
            raise FileNotFoundError(required)

    price_frame = pd.read_excel(paths.attachment1, sheet_name="Sheet1")
    assert price_frame.shape == (INTERVALS_PER_DAY, 4), price_frame.shape
    price_labels = tuple(normalize_time_label(value) for value in price_frame.iloc[:, 0])
    prices = pd.to_numeric(price_frame.iloc[:, 1], errors="raise").to_numpy(dtype=float)

    load_frame = pd.read_excel(paths.attachment2, sheet_name="小区负载")
    pv_frame = pd.read_excel(paths.attachment2, sheet_name="光伏发电实际功率")
    assert load_frame.shape == pv_frame.shape == (365, INTERVALS_PER_DAY + 1)

    dates = pd.DatetimeIndex(pd.to_datetime(load_frame.iloc[:, 0], errors="raise")).normalize()
    pv_dates = pd.DatetimeIndex(pd.to_datetime(pv_frame.iloc[:, 0], errors="raise")).normalize()
    assert dates.equals(pv_dates)
    assert dates[0] == pd.Timestamp("2025-01-01")
    assert dates[-1] == pd.Timestamp("2025-12-31")
    assert dates.is_monotonic_increasing and dates.is_unique
    assert np.all(np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1)

    load_labels = tuple(normalize_time_label(value) for value in load_frame.columns[1:])
    pv_labels = tuple(normalize_time_label(value) for value in pv_frame.columns[1:])
    assert load_labels == pv_labels == price_labels
    assert price_labels[0] == "00:10" and price_labels[-1] == "0:00+1"

    load_kw = load_frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    pv_kw = pv_frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    assert np.isfinite(load_kw).all() and np.isfinite(pv_kw).all()
    assert np.isfinite(prices).all() and np.all(prices > 0.0)
    assert np.all(load_kw >= 0.0) and np.all(pv_kw >= 0.0)

    natural_labels = tuple(interval_label(index) for index in range(INTERVALS_PER_DAY))
    return InputData(
        dates=dates,
        source_labels=price_labels,
        interval_labels=natural_labels,
        prices=prices,
        load_kwh=load_kw * DELTA_HOURS,
        pv_kwh=pv_kw * DELTA_HOURS,
    )


def from_arrays(dates, prices, load_kwh, pv_kwh) -> InputData:
    source = []
    for i in range(INTERVALS_PER_DAY):
        end = (i + 1) * 10
        source.append("0:00+1" if end == 1440 else f"{end // 60:02d}:{end % 60:02d}")
    labels = tuple(interval_label(i) for i in range(INTERVALS_PER_DAY))
    return InputData(
        dates=dates,
        source_labels=tuple(source),
        interval_labels=labels,
        prices=np.asarray(prices, dtype=float),
        load_kwh=np.asarray(load_kwh, dtype=float),
        pv_kwh=np.asarray(pv_kwh, dtype=float),
    )

"""问题三数据读入：电价、实际负载/光伏、冷启动、附件3预报立方体。"""

from __future__ import annotations
from ...common.paths import project_root
from ...common import dataset, physical

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = project_root()
N = 144
DT = 1.0 / 6.0
ISSUE_HOURS = (0, 6, 12, 18)


@dataclass(frozen=True)
class Problem3Data:
    dates: pd.DatetimeIndex
    prices: np.ndarray  # (144,)
    load_kwh: np.ndarray  # (365, 144)
    pv_kwh: np.ndarray  # (365, 144)
    cold_load: np.ndarray  # (144,)
    cold_pv: np.ndarray  # (144,)
    # forecast_kwh[date_idx, issue_idx, t] ; issue_idx 0..3 -> 0/6/12/18
    forecast_kwh: np.ndarray  # (365, 4, 144)


def interpolate_issue_to_day(
    issue_hour: int,
    horizon_powers_kw: np.ndarray,
) -> np.ndarray:
    """整点预报 → 144 个 10min 电量 (kWh)。

    节点在 issue_hour+h (h=1..24)；区间中点线性插值。
    发布时刻到第一个节点之间（中点 < τ+1）：用**时效1的功率**外推，
    不得再取 0（否则 12:00 调整会把正午光伏抹成 0）。
    """
    if horizon_powers_kw.shape != (24,):
        raise ValueError("需要长度 24 的时效功率")
    if issue_hour not in ISSUE_HOURS:
        raise ValueError(f"unsupported forecast issue hour: {issue_hour}")
    node_hours = np.arange(1, 25, dtype=float) + float(issue_hour)
    node_kw = np.asarray(horizon_powers_kw, dtype=float)
    first_node = float(node_hours[0])
    first_kw = float(node_kw[0])

    out_kwh = np.zeros(N, dtype=float)
    for t in range(N):
        mid_hour = (t * 10 + 5) / 60.0
        if mid_hour < first_node:
            power_kw = first_kw
        else:
            power_kw = float(np.interp(mid_hour, node_hours, node_kw))
        out_kwh[t] = max(power_kw, 0.0) * DT
    return out_kwh


def _build_forecast_cube_from_frame(
    fc: pd.DataFrame, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Build the forecast cube from the validated official attachment rows."""
    need = {"日期", "预报时刻", "预报时效（小时）", "光伏发电预报功率（千瓦）"}
    if not need.issubset(fc.columns):
        raise KeyError(f"预报表缺列: {fc.columns.tolist()}")

    fc = fc.copy()
    fc["日期"] = pd.to_datetime(fc["日期"], errors="raise").dt.normalize()
    fc["预报时刻"] = fc["预报时刻"].astype(str).str.strip()
    cube = np.zeros((len(dates), len(ISSUE_HOURS), N), dtype=float)
    date_to_idx = {d.normalize(): i for i, d in enumerate(dates)}
    seen: set[tuple[pd.Timestamp, int]] = set()

    for (day, issue_label), group in fc.groupby(
        ["日期", "预报时刻"], sort=False
    ):
        day = pd.Timestamp(day).normalize()
        if day not in date_to_idx:
            continue
        issue_text = str(issue_label).strip()
        if issue_text in {"0:00", "00:00"}:
            issue_hour = 0
        elif issue_text in {"6:00", "06:00"}:
            issue_hour = 6
        elif issue_text in {"12:00"}:
            issue_hour = 12
        elif issue_text in {"18:00"}:
            issue_hour = 18
        else:
            raise ValueError(f"未知预报时刻: {issue_text!r}")

        issue_idx = ISSUE_HOURS.index(issue_hour)
        g = group.sort_values("预报时效（小时）")
        if len(g) != 24:
            raise ValueError(f"{day.date()} {issue_text} 时效条数={len(g)}")
        horizons = g["预报时效（小时）"].to_numpy(dtype=int)
        if not np.array_equal(horizons, np.arange(1, 25)):
            raise ValueError(f"{day.date()} {issue_text} 时效不连续: {horizons[:5]}...")
        powers = g["光伏发电预报功率（千瓦）"].to_numpy(dtype=float)
        if not np.isfinite(powers).all() or np.any(powers < 0.0):
            raise ValueError(f"{day.date()} {issue_text} 含非法预报功率")
        cube[date_to_idx[day], issue_idx, :] = interpolate_issue_to_day(issue_hour, powers)
        seen.add((day, issue_hour))

    expected = {
        (pd.Timestamp(day).normalize(), issue_hour)
        for day in dates
        for issue_hour in ISSUE_HOURS
    }
    missing = sorted(expected - seen)
    if missing:
        preview = ", ".join(f"{day.date()} {hour}:00" for day, hour in missing[:5])
        raise ValueError(f"附件3缺少预报组（共{len(missing)}组）: {preview}")

    return cube


def _tidy_attachment3(path: Path) -> pd.DataFrame:
    """Read the live raw 附件3 workbook without requiring a temp/src helper."""
    frame = pd.read_excel(path)
    need = {"日期", "预报时刻"}
    if not need.issubset(frame.columns):
        raise KeyError(f"附件3列不符合预期: {frame.columns.tolist()}")

    forecast_cols = [
        column
        for column in frame.columns
        if str(column).startswith("预报") and column != "预报时刻"
    ]
    if len(forecast_cols) != 24:
        raise ValueError(f"附件3应有24个预报时效列，实际{len(forecast_cols)}列")

    frame = frame.copy()
    frame["日期"] = frame["日期"].ffill()
    if frame["日期"].isna().any():
        raise ValueError("附件3日期补全后仍有缺失")
    frame["日期"] = pd.to_datetime(frame["日期"], errors="raise").dt.normalize()
    frame["预报时刻"] = frame["预报时刻"].astype(str).str.strip()

    long = frame.melt(
        id_vars=["日期", "预报时刻"],
        value_vars=forecast_cols,
        var_name="时效列",
        value_name="光伏发电预报功率（千瓦）",
    )
    long["预报时效（小时）"] = (
        long["时效列"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    )
    long["光伏发电预报功率（千瓦）"] = pd.to_numeric(
        long["光伏发电预报功率（千瓦）"], errors="raise"
    )
    return long[
        ["日期", "预报时刻", "预报时效（小时）", "光伏发电预报功率（千瓦）"]
    ]


def load_problem3_data(root: Path | None = None) -> Problem3Data:
    root = Path(root) if root else PROJECT_DIR
    paths = dataset.get_paths(root)
    base = dataset.read_inputs(paths)
    cold_load, cold_pv = physical.read_cold_start(paths)
    raw_forecast = _tidy_attachment3(root / "附件" / "附件3.xlsx")
    cube = _build_forecast_cube_from_frame(raw_forecast, base.dates)
    return Problem3Data(
        dates=base.dates,
        prices=base.prices,
        load_kwh=base.load_kwh,
        pv_kwh=base.pv_kwh,
        cold_load=cold_load,
        cold_pv=cold_pv,
        forecast_kwh=cube,
    )


def issue_start_index(issue_hour: int) -> int:
    """0-based 起始时段下标：0/6/12/18 → 0/36/72/108。"""
    if issue_hour not in ISSUE_HOURS:
        raise ValueError(issue_hour)
    return issue_hour * 6

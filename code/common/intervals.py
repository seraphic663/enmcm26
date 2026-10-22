"""Shared interval labels and contiguous emergency-purchase groups."""
import numpy as np
import pandas as pd
N = 144

def slot_end_minutes() -> np.ndarray:
    """Return the 10-minute interval end labels used by the templates."""
    return np.arange(10, 1441, 10, dtype=int)

def _fmt_min(m: int) -> str:
    if m == 1440:
        return "0:00+1"
    if m == 0:
        return "0:00"
    h, mi = divmod(m, 60)
    return f"{h}:{mi:02d}" if mi else f"{h}:00"

def emergency_intervals(e_act: np.ndarray, threshold: float) -> list[tuple[str, float]]:
    ends = slot_end_minutes()
    out: list[tuple[str, float]] = []
    i = 0
    while i < N:
        if e_act[i] <= threshold:
            i += 1
            continue
        j = i
        s = 0.0
        while j < N and e_act[j] > threshold:
            s += float(e_act[j])
            j += 1
        out.append(
            (f"{_fmt_min(int(ends[i]) - 10)}-{_fmt_min(int(ends[j - 1]))}", s)
        )
        i = j
    return out



BLOCKS_4H = tuple((f"{i*4}:00-{(i+1)*4}:00", i*240, (i+1)*240) for i in range(6))

def aggregate_blocks_4h(series_10min: np.ndarray | list[float]) -> dict[str, float]:
    """Aggregate 144 ten-minute energy values into six four-hour blocks."""
    values = np.asarray(series_10min, dtype=float).reshape(-1)
    if values.size != N:
        raise ValueError(f"需要 {N} 个10分钟点，收到 {values.size}")
    ends = slot_end_minutes()
    return {
        name: float(values[(ends > lo) & (ends <= hi)].sum())
        for name, lo, hi in BLOCKS_4H
    }

def date_rows(ws) -> dict[pd.Timestamp, int]:
    # Preserve the submitted official headers; values follow attachment order.
    mapping: dict[pd.Timestamp, int] = {}
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, 1).value
        if v is not None:
            mapping[pd.Timestamp(v).normalize()] = r
    return mapping


def write_storage_sheet(wb, dates, out_idx, charge, discharge, soc) -> None:
    if "充放电量" in wb.sheetnames:
        del wb["充放电量"]
    ws = wb.create_sheet("充放电量", 2 if "调整购电量" in wb.sheetnames else 1)
    for c, h in enumerate(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"], 1):
        ws.cell(1, c, h)
    row = 2
    for i in out_idx:
        ch = aggregate_blocks_4h(charge[i])
        dis = aggregate_blocks_4h(discharge[i])
        s = soc[i]
        for k, (name, _lo, _hi) in enumerate(BLOCKS_4H):
            if k == 0:
                ws.cell(
                    row,
                    1,
                    dates[i].to_pydatetime().replace(hour=0, minute=0, second=0),
                )
                ws.cell(row, 5, "0:00")
                ws.cell(row, 6, float(s[0]))
            elif k == 1:
                ws.cell(row, 5, "24:00")
                ws.cell(row, 6, float(s[-1]))
            ws.cell(row, 2, name)
            ws.cell(row, 3, float(ch[name]))
            ws.cell(row, 4, float(dis[name]))
            row += 1


def write_emergency_sheet(wb, dates, out_idx, emergency, sheet_index: int, threshold: float) -> None:
    if "紧急购电量" in wb.sheetnames:
        del wb["紧急购电量"]
    ws = wb.create_sheet("紧急购电量", sheet_index)
    ws.cell(1, 1, "日期")
    ws.cell(1, 2, "紧急购电时间段")
    ws.cell(1, 3, "紧急购电量")
    row = 2
    for i in out_idx:
        intervals = emergency_intervals(emergency[i], threshold)
        if not intervals:
            continue
        first = True
        for lab, amt in intervals:
            if first:
                ws.cell(
                    row,
                    1,
                    dates[i].to_pydatetime().replace(hour=0, minute=0, second=0),
                )
                first = False
            ws.cell(row, 2, lab)
            ws.cell(row, 3, float(amt))
            row += 1


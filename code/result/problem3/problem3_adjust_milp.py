"""问题三/四日内调整 MILP：6·12·18 调整剩余时段，禁止少购。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

N = 144
SOC_MIN = 1200.0
SOC_MAX = 10800.0
ENERGY_MAX = 5000.0 / 6.0
ETA_C = 0.9
ETA_D = 0.9


@dataclass(frozen=True)
class StageSolution:
    x: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    solver_seconds: float


def _add(
    rows: list[int],
    cols: list[int],
    values: list[float],
    row: int,
    col: int,
    value: float,
) -> None:
    rows.append(row)
    cols.append(col)
    values.append(value)


def solve_stage(
    *,
    t_start: int,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    prices: np.ndarray,
    initial_soc: float,
    x_plan: np.ndarray,
    locked_x: np.ndarray,
    locked_charge: np.ndarray,
    locked_discharge: np.ndarray,
    time_limit: float = 30.0,
    freeze_storage: bool = False,
) -> StageSolution:
    """求解从 t_start 到日终的剩余时段。

    - t_start>0：相对固定 x_plan 计 0.5c(|d+|+|d-|)，已执行段由 locked_* 固定。
    - 禁止少购（d-=0），只允许相对计划加购，避免乐观预报砍计划。
    - freeze_storage=True：剩余时段充放电钉死为 locked_u/v（通常即 0:00 计划），只调购电。
    """
    if not (0 < t_start < N):
        raise ValueError(f"t_start={t_start}")
    for name, arr in (
        ("load", load_forecast),
        ("pv", pv_forecast),
        ("prices", prices),
    ):
        if np.asarray(arr).shape != (N,):
            raise ValueError(f"{name} shape must be (144,)")

    rem = N - t_start
    if np.asarray(x_plan).shape != (N,):
        raise ValueError("x_plan shape must be (144,)")

    # 保留已验证的变量布局和约束顺序，d_minus 固定为0。
    off = {
        "x": 0,
        "u": rem,
        "v": 2 * rem,
        "soc": 3 * rem,
        "z": 4 * rem,
        "e": 5 * rem,
        "w": 6 * rem,
    }
    n_base = 7 * rem
    off["dm"] = n_base
    off["dp"] = n_base + rem
    n_var = n_base + 2 * rem

    lower = np.zeros(n_var, dtype=float)
    upper = np.full(n_var, np.inf, dtype=float)
    upper[off["u"] : off["u"] + rem] = ENERGY_MAX
    upper[off["v"] : off["v"] + rem] = ENERGY_MAX
    lower[off["soc"] : off["soc"] + rem] = SOC_MIN
    upper[off["soc"] : off["soc"] + rem] = SOC_MAX
    upper[off["z"] : off["z"] + rem] = 1.0
    for k in range(rem):
        t = t_start + k
        upper[off["w"] + k] = float(pv_forecast[t])
    upper[off["dm"] : off["dm"] + rem] = 0.0
    if freeze_storage:
        for k in range(rem):
            t = t_start + k
            u0 = float(locked_charge[t])
            v0 = float(locked_discharge[t])
            lower[off["u"] + k] = u0
            upper[off["u"] + k] = u0
            lower[off["v"] + k] = v0
            upper[off["v"] + k] = v0

    integrality = np.zeros(n_var, dtype=np.uint8)
    integrality[off["z"] : off["z"] + rem] = 1

    obj = np.zeros(n_var, dtype=float)
    for k in range(rem):
        t = t_start + k
        c = float(prices[t])
        obj[off["x"] + k] = c
        obj[off["e"] + k] = 5.0 * c
        obj[off["dm"] + k] = 0.5 * c
        obj[off["dp"] + k] = 0.5 * c

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lo_rows: list[float] = []
    hi_rows: list[float] = []
    row = 0

    # 电量平衡
    for k in range(rem):
        t = t_start + k
        _add(rows, cols, values, row, off["x"] + k, 1.0)
        _add(rows, cols, values, row, off["u"] + k, -1.0)
        _add(rows, cols, values, row, off["v"] + k, 1.0)
        _add(rows, cols, values, row, off["e"] + k, 1.0)
        _add(rows, cols, values, row, off["w"] + k, -1.0)
        rhs = float(load_forecast[t] - pv_forecast[t])
        lo_rows.append(rhs)
        hi_rows.append(rhs)
        row += 1

    # SOC: E_k = E_{k-1} + 0.9 u - v/0.9 ；k=0 时 E_{-1}=initial_soc
    for k in range(rem):
        _add(rows, cols, values, row, off["soc"] + k, 1.0)
        if k > 0:
            _add(rows, cols, values, row, off["soc"] + k - 1, -1.0)
        _add(rows, cols, values, row, off["u"] + k, -ETA_C)
        _add(rows, cols, values, row, off["v"] + k, 1.0 / ETA_D)
        rhs0 = float(initial_soc) if k == 0 else 0.0
        lo_rows.append(rhs0)
        hi_rows.append(rhs0)
        row += 1

    # 充放电互斥
    for k in range(rem):
        _add(rows, cols, values, row, off["u"] + k, 1.0)
        _add(rows, cols, values, row, off["z"] + k, -ENERGY_MAX)
        lo_rows.append(-np.inf)
        hi_rows.append(0.0)
        row += 1
    for k in range(rem):
        _add(rows, cols, values, row, off["v"] + k, 1.0)
        _add(rows, cols, values, row, off["z"] + k, ENERGY_MAX)
        lo_rows.append(-np.inf)
        hi_rows.append(ENERGY_MAX)
        row += 1

    # x = x_plan - d- + d+
    for k in range(rem):
        t = t_start + k
        _add(rows, cols, values, row, off["x"] + k, 1.0)
        _add(rows, cols, values, row, off["dm"] + k, 1.0)
        _add(rows, cols, values, row, off["dp"] + k, -1.0)
        rhs = float(x_plan[t])
        lo_rows.append(rhs)
        hi_rows.append(rhs)
        row += 1

    matrix = coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
        shape=(row, n_var),
    ).tocsr()

    started = time.perf_counter()
    result = milp(
        c=obj,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix, np.asarray(lo_rows), np.asarray(hi_rows)
        ),
        options={
            "disp": False,
            "presolve": True,
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-7,
        },
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None:
        raise RuntimeError(
            f"问题三 MILP 失败 t_start={t_start}: {result.status} {result.message}"
        )

    vec = np.asarray(result.x, dtype=float)
    x_rem = vec[off["x"] : off["x"] + rem]
    u_rem = vec[off["u"] : off["u"] + rem]
    v_rem = vec[off["v"] : off["v"] + rem]

    x = np.zeros(N, dtype=float)
    charge = np.zeros(N, dtype=float)
    discharge = np.zeros(N, dtype=float)

    x[:t_start] = locked_x[:t_start]
    charge[:t_start] = locked_charge[:t_start]
    discharge[:t_start] = locked_discharge[:t_start]

    x[t_start:] = x_rem
    charge[t_start:] = u_rem
    discharge[t_start:] = v_rem

    return StageSolution(
        x=x,
        charge=charge,
        discharge=discharge,
        solver_seconds=elapsed,
    )

from collections.abc import Callable
from typing import overload

import numpy as np

FLOAT_TYPE = np.float32
INT_TYPE = np.int32
DEFAULT_N_LEVELS = 20
DEFAULT_N_ISSUE_LEVELS = min(10, max(3, DEFAULT_N_LEVELS // 2))
MAX_CARDINALITY = 100_000
DEFAULT_N_OUTCOMES = 1000
DEFAULT_UTIL_LEVELS = 20
DEFAULT_N_OFFERS = 4
DEFAULT_MIN_ENCODABLE = 0.05
DEFAULT_N_DIMS = 5
Policy = Callable[[np.ndarray | dict[str, np.ndarray]], np.ndarray]
"""A policy takes input as a numpy array or a mapping to such arrays and generates actions as a numpy array"""
SAOPolicy = Policy
"""A policy for the SAOMechanism is just a normal policy"""


@overload
def to_float(
    x: int,
    n: int | np.ndarray,
    start: int | np.ndarray = 0,
    mn: float = 0,
    mx: float = 1,
) -> float: ...


@overload
def to_float(
    x: np.ndarray,
    n: np.ndarray | int,
    start: int | np.ndarray = 0,
    mn: float = 0,
    mx: float = 1,
) -> np.ndarray: ...


def to_float(
    x: int | np.ndarray,
    n: int | np.ndarray,
    start: int | np.ndarray = 0,
    mn: float = 0,
    mx: float = 1,
) -> float | np.ndarray:
    """Converts x to a float between mn and mx assuming it is an int between 0 and n - 1"""
    if isinstance(x, np.ndarray):
        x = x.astype(INT_TYPE)
    else:
        x = int(x)
    if isinstance(n, np.ndarray):
        n = n.astype(INT_TYPE)
    else:
        n = int(n)
    y = (x - start + 0.5) / n
    z = y * (mx - mn) + mn
    return z


@overload
def from_float(
    x: float, n: int, start: int = 0, mn: float = 0, mx: float = 1
) -> int: ...


@overload
def from_float(
    x: np.ndarray, n: np.ndarray | int, start: int = 0, mn: float = 0, mx: float = 1
) -> np.ndarray: ...


def from_float(
    x: float | np.ndarray,
    n: int | np.ndarray,
    start: int = 0,
    mn: float = 0,
    mx: float = 1,
) -> int | np.ndarray:
    """Converts x to an integer between 0 and n-1 so that each integer is mapped to an equal interval between mn and mx"""
    y = (x - mn) / (mx - mn)
    if isinstance(y, np.ndarray):
        return (y * n + start).astype(INT_TYPE)
    return int(y * n) + start

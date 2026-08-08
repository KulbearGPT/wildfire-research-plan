"""Deterministic reference predictors for wildfire targets."""

from numbers import Integral
from typing import Literal

import numpy as np


def no_fire(shape: tuple[int, int]) -> np.ndarray:
    """Return a fresh all-zero prediction for a two-dimensional target."""
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension <= 0
               for dimension in shape)
    ):
        raise ValueError("shape must contain two positive dimensions")
    return np.zeros(shape, dtype=np.uint8)


def persistence(history: np.ndarray, mode: Literal["latest", "all"]) -> np.ndarray:
    """Predict fire from the latest frame or the union of all history frames."""
    if history.ndim != 3 or history.shape[0] == 0:
        raise ValueError("history must be a non-empty three-dimensional array")
    if not np.all((history == 0) | (history == 1)):
        raise ValueError("history must contain only binary values")
    if mode == "latest":
        return history[-1].copy()
    if mode == "all":
        return np.any(history, axis=0).astype(np.uint8)
    raise ValueError("mode must be 'latest' or 'all'")

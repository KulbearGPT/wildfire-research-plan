from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np


ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]
REPAIR_VERSION = "1"


def _finite_positive_integers(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.any(np.isinf(array)):
        raise ValueError("invalid active-fire encoding: infinity is not allowed")
    finite = array[np.isfinite(array)]
    if np.any(finite < 0) or np.any(finite != np.floor(finite)):
        raise ValueError("invalid active-fire encoding: values must be nonnegative integers")
    return finite[finite > 0]


def classify_active_fire_encoding(values: np.ndarray) -> ActiveFireEncoding:
    positive = _finite_positive_integers(values)
    if positive.size == 0:
        return "no_positive_values"
    hour_mask = positive <= 23
    if np.all(hour_mask):
        return "hour"
    if np.any(hour_mask):
        raise ValueError("invalid active-fire encoding: mixed hour and HHMM values")
    hours = np.floor_divide(positive, 100)
    minutes = np.mod(positive, 100)
    if np.any(hours > 23) or np.any(minutes > 59):
        raise ValueError("invalid active-fire encoding: invalid HHMM value")
    return "hhmm"


def normalize_active_fire(
    values: np.ndarray,
) -> tuple[np.ndarray, ActiveFireEncoding]:
    encoding = classify_active_fire_encoding(values)
    normalized = np.nan_to_num(np.asarray(values), nan=0.0).copy()
    if encoding == "hhmm":
        normalized = np.floor_divide(normalized, 100)
    if np.any(normalized < 0) or np.any(normalized > 23):
        raise ValueError("invalid active-fire encoding after normalization")
    return normalized, encoding


def source_fingerprint(paths: Sequence[Path], source_root: Path) -> str:
    root = Path(source_root).resolve()
    digest = sha256()
    relatives = sorted(Path(item).resolve().relative_to(root) for item in paths)
    for relative_path in relatives:
        relative = relative_path.as_posix()
        stat = (root / relative_path).stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()

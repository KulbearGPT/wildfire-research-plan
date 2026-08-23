"""Deterministic raw-space controlled-missingness transformations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath

import numpy as np

from .matrix import CORRUPTIONS


SCHEMA_VERSION = 1
RAW_FEATURE_COUNT = 23
ACTIVE_FIRE_FEATURE = 22
DYNAMIC_NON_FIRE_FEATURES = tuple(range(12)) + (15,) + tuple(range(17, 22))


@dataclass(frozen=True)
class CorruptionResult:
    """One corrupted copy and optional spatial-mask audit evidence."""

    values: np.ndarray
    spatial_mask: np.ndarray | None
    schema_version: int
    key_digest: str


def _validate_identity(event_relative_path: str, target_date: str) -> None:
    path = PurePosixPath(event_relative_path)
    if (
        not event_relative_path
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("event_relative_path must be a normalized relative path")
    try:
        parsed = date.fromisoformat(target_date)
    except (TypeError, ValueError) as error:
        raise ValueError("target_date must be an ISO calendar date") from error
    if parsed.isoformat() != target_date:
        raise ValueError("target_date must use YYYY-MM-DD form")


def stable_key_digest(
    scenario_id: str,
    matrix_seed: int,
    event_relative_path: str,
    target_date: str,
) -> str:
    """Hash only the frozen schema and sample identity fields."""

    _validate_identity(event_relative_path, target_date)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "matrix_seed": matrix_seed,
        "event_relative_path": event_relative_path,
        "target_date": target_date,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def structured_block_mask(
    height: int,
    width: int,
    fraction: float,
    *,
    key_digest: str,
) -> np.ndarray:
    """Place one deterministic near-rectangular block with exact pixel area."""

    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("spatial dimensions must be positive integers")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("mask fraction must be within (0, 1]")
    if len(key_digest) != 64:
        raise ValueError("key_digest must be a SHA-256 hexadecimal digest")
    try:
        key = bytes.fromhex(key_digest)
    except ValueError as error:
        raise ValueError("key_digest must be hexadecimal") from error

    target = max(1, min(height * width, round(height * width * fraction)))
    block_height = max(1, min(height, round(math.sqrt(target * height / width))))
    block_width = math.ceil(target / block_height)
    if block_width > width:
        block_height = math.ceil(target / width)
        block_width = math.ceil(target / block_height)
    block_height = min(block_height, height)
    block_width = min(block_width, width)

    top_slots = height - block_height + 1
    left_slots = width - block_width + 1
    top = int.from_bytes(key[:8], "big") % top_slots
    left = int.from_bytes(key[8:16], "big") % left_slots

    mask = np.zeros((height, width), dtype=bool)
    remaining = target
    for row in range(block_height):
        take = min(block_width, remaining)
        mask[top + row, left : left + take] = True
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise AssertionError("structured block construction did not reach target area")
    return mask


def _validate_raw(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != 4
        or array.shape[1] != RAW_FEATURE_COUNT
        or array.shape[0] <= 0
        or array.shape[2] <= 0
        or array.shape[3] <= 0
    ):
        raise ValueError("raw input must have shape (T, 23, H, W)")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("raw input must use a floating dtype")
    return array


def apply_corruption(
    values: np.ndarray,
    scenario_id: str,
    *,
    event_relative_path: str,
    target_date: str,
    stale_active_fire: np.ndarray | None = None,
) -> CorruptionResult:
    """Apply one declared M00--M07 condition without mutating its input."""

    try:
        scenario = CORRUPTIONS[scenario_id]
    except KeyError as error:
        raise ValueError(f"unknown controlled-missingness scenario: {scenario_id}") from error
    source = _validate_raw(values)
    digest = stable_key_digest(
        scenario_id,
        scenario.matrix_seed,
        event_relative_path,
        target_date,
    )
    result = source.copy()
    spatial_mask: np.ndarray | None = None

    if scenario_id == "M01":
        result[:, ACTIVE_FIRE_FEATURE] = 0.0
    elif scenario_id == "M02":
        if stale_active_fire is None:
            raise ValueError("M02 requires stale_active_fire")
        stale = np.asarray(stale_active_fire)
        expected = (source.shape[0], source.shape[2], source.shape[3])
        if stale.shape != expected:
            raise ValueError(
                f"stale_active_fire shape must equal {expected}, got {stale.shape}"
            )
        if not np.isfinite(stale).all():
            raise ValueError("stale_active_fire must be finite")
        result[:, ACTIVE_FIRE_FEATURE] = stale
    elif scenario_id in {"M03", "M04", "M05"}:
        result[:, scenario.feature_indices] = np.nan
    elif scenario_id in {"M06", "M07"}:
        fraction = 0.25 if scenario_id == "M06" else 0.50
        spatial_mask = structured_block_mask(
            source.shape[2], source.shape[3], fraction, key_digest=digest
        )
        for feature in DYNAMIC_NON_FIRE_FEATURES:
            result[:, feature, spatial_mask] = np.nan
        result[:, ACTIVE_FIRE_FEATURE, spatial_mask] = 0.0

    return CorruptionResult(result, spatial_mask, SCHEMA_VERSION, digest)

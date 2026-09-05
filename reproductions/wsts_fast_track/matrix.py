"""Controlled missingness scenarios used by retained T=1 evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True)
class CorruptionSpec:
    scenario_id: str
    condition: str
    family: str
    feature_indices: tuple[int, ...]
    severity: str
    matrix_seed: int
    implementation_state: Literal["implemented"]


_DYNAMIC_FEATURES = tuple(range(12)) + (15,) + tuple(range(17, 23))

CORRUPTIONS: Final[dict[str, CorruptionSpec]] = {
    "M00": CorruptionSpec("M00", "clean reference", "clean", (), "none", 0, "implemented"),
    "M01": CorruptionSpec(
        "M01", "active-fire history absent", "fire_history", (22,), "all", 0, "implemented"
    ),
    "M02": CorruptionSpec(
        "M02", "active-fire history stale", "fire_history", (22,), "days=1", 0, "implemented"
    ),
    "M03": CorruptionSpec(
        "M03",
        "observed weather absent",
        "modality",
        tuple(range(5, 12)),
        "full",
        0,
        "implemented",
    ),
    "M04": CorruptionSpec(
        "M04",
        "forecast weather absent",
        "modality",
        tuple(range(17, 22)),
        "full",
        0,
        "implemented",
    ),
    "M05": CorruptionSpec(
        "M05",
        "observed and forecast weather absent",
        "modality",
        tuple(range(5, 12)) + tuple(range(17, 22)),
        "full",
        0,
        "implemented",
    ),
    "M06": CorruptionSpec(
        "M06",
        "structured spatial blocks",
        "spatial",
        _DYNAMIC_FEATURES,
        "area=0.25",
        0,
        "implemented",
    ),
    "M07": CorruptionSpec(
        "M07",
        "structured spatial blocks",
        "spatial",
        _DYNAMIC_FEATURES,
        "area=0.50",
        0,
        "implemented",
    ),
}

"""Typed clean-run and controlled-missingness matrices for WSTS+."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal


RunStage = Literal["screening", "promotion", "replication"]
LaunchState = Literal["active", "promotable", "gated"]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    experiment_id: str
    stage: RunStage
    seed: int
    max_steps: int
    prerequisites: tuple[str, ...]
    launch_state: LaunchState


@dataclass(frozen=True)
class CorruptionSpec:
    scenario_id: str
    condition: str
    family: str
    feature_indices: tuple[int, ...]
    severity: str
    matrix_seed: int
    implementation_state: Literal["declared"]


_SCREENING_PREREQUISITES = ("C00-S0-3K", "C02-S0-3K")
_PROMOTION_PREREQUISITES = ("C00-S0-10K", "C02-S0-10K")

CLEAN_RUNS: Final[dict[str, RunSpec]] = {
    "C00-S0-3K": RunSpec(
        "C00-S0-3K", "C00", "screening", 0, 3_000, (), "active"
    ),
    "C02-S0-3K": RunSpec(
        "C02-S0-3K", "C02", "screening", 0, 3_000, (), "active"
    ),
    "C00-S0-10K": RunSpec(
        "C00-S0-10K",
        "C00",
        "promotion",
        0,
        10_000,
        _SCREENING_PREREQUISITES,
        "promotable",
    ),
    "C02-S0-10K": RunSpec(
        "C02-S0-10K",
        "C02",
        "promotion",
        0,
        10_000,
        _SCREENING_PREREQUISITES,
        "promotable",
    ),
    "C00-S1-10K": RunSpec(
        "C00-S1-10K",
        "C00",
        "replication",
        1,
        10_000,
        _PROMOTION_PREREQUISITES,
        "gated",
    ),
    "C00-S2-10K": RunSpec(
        "C00-S2-10K",
        "C00",
        "replication",
        2,
        10_000,
        _PROMOTION_PREREQUISITES,
        "gated",
    ),
    "C02-S1-10K": RunSpec(
        "C02-S1-10K",
        "C02",
        "replication",
        1,
        10_000,
        _PROMOTION_PREREQUISITES,
        "gated",
    ),
    "C02-S2-10K": RunSpec(
        "C02-S2-10K",
        "C02",
        "replication",
        2,
        10_000,
        _PROMOTION_PREREQUISITES,
        "gated",
    ),
}

_DYNAMIC_FEATURES = tuple(range(12)) + (15,) + tuple(range(17, 23))

CORRUPTIONS: Final[dict[str, CorruptionSpec]] = {
    "M00": CorruptionSpec("M00", "clean reference", "clean", (), "none", 0, "declared"),
    "M01": CorruptionSpec(
        "M01", "active-fire history absent", "fire_history", (22,), "all", 0, "declared"
    ),
    "M02": CorruptionSpec(
        "M02", "active-fire history stale", "fire_history", (22,), "days=1", 0, "declared"
    ),
    "M03": CorruptionSpec(
        "M03",
        "observed weather absent",
        "modality",
        tuple(range(5, 12)),
        "full",
        0,
        "declared",
    ),
    "M04": CorruptionSpec(
        "M04",
        "forecast weather absent",
        "modality",
        tuple(range(17, 22)),
        "full",
        0,
        "declared",
    ),
    "M05": CorruptionSpec(
        "M05",
        "observed and forecast weather absent",
        "modality",
        tuple(range(5, 12)) + tuple(range(17, 22)),
        "full",
        0,
        "declared",
    ),
    "M06": CorruptionSpec(
        "M06",
        "structured spatial blocks",
        "spatial",
        _DYNAMIC_FEATURES,
        "area=0.25",
        0,
        "declared",
    ),
    "M07": CorruptionSpec(
        "M07",
        "structured spatial blocks",
        "spatial",
        _DYNAMIC_FEATURES,
        "area=0.50",
        0,
        "declared",
    ),
}


def run_spec(run_id: str) -> RunSpec:
    """Return one declared clean-run specification."""

    try:
        return CLEAN_RUNS[run_id]
    except KeyError as error:
        raise ValueError(f"unknown fast-track run: {run_id}") from error


def matrix_payload() -> dict[str, object]:
    """Return the canonical versioned experiment-matrix payload."""

    return {
        "schema_version": 1,
        "clean_runs": [asdict(item) for item in CLEAN_RUNS.values()],
        "corruptions": [asdict(item) for item in CORRUPTIONS.values()],
    }

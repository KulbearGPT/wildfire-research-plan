"""Corrected-index, from-scratch baseline definitions for the active track."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from .corruption_training import (
    install_training_fire_and_block_dropout,
    install_training_fire_dropout,
)


TrainingPolicy = Literal["clean", "fire", "fire-block"]


@dataclass(frozen=True)
class CorrectedBaselineSpec:
    """One minimal corrected-index baseline screen."""

    baseline_id: str
    experiment_id: Literal["C00"]
    training_policy: TrainingPolicy
    seed: int = 0
    max_steps: int = 3_000


CORRECTED_BASELINES: Final[dict[str, CorrectedBaselineSpec]] = {
    "B0": CorrectedBaselineSpec("B0", "C00", "clean"),
    "B2": CorrectedBaselineSpec("B2", "C00", "fire"),
    "B3": CorrectedBaselineSpec("B3", "C00", "fire-block"),
}


def corrected_baseline_spec(baseline_id: str) -> CorrectedBaselineSpec:
    """Return one registered corrected-index baseline."""

    try:
        return CORRECTED_BASELINES[baseline_id]
    except KeyError as error:
        raise ValueError(f"unknown corrected baseline: {baseline_id}") from error


def resolve_dataset_index(dataset: Any, target_id: int) -> tuple[int, str, int]:
    """Resolve the first matching year/fire without the upstream loop leak."""

    length = len(dataset)
    if target_id < 0:
        target_id += length
    if target_id < 0 or target_id >= length:
        raise RuntimeError(
            f"Tried to access item {target_id}, but maximum index is {length - 1}."
        )
    remaining = target_id
    per_fire = getattr(dataset, "datapoints_per_fire", None)
    if not isinstance(per_fire, Mapping):
        raise ValueError("dataset must expose per-fire datapoint counts")
    for year, fires in per_fire.items():
        for fire_name, count_value in fires.items():
            count = int(count_value)
            if remaining < count:
                return int(year), str(fire_name), remaining
            remaining -= count
    raise RuntimeError(f"dataset index did not resolve: {target_id}")


def install_corrected_baseline(upstream_root: Path | str, baseline_id: str) -> None:
    """Install first-match resolution and only the selected train corruption."""

    spec = corrected_baseline_spec(baseline_id)
    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index

    if spec.training_policy == "fire":
        install_training_fire_dropout(upstream)
    elif spec.training_policy == "fire-block":
        install_training_fire_and_block_dropout(upstream)

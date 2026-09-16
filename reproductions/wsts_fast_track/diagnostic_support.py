"""Exact archived diagnostic helpers without unrelated P-series dependencies.

Functions originate at 4b843ad; only checkpoint path relocation is additive.
"""
from __future__ import annotations
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from .runtime import TRAIN_YEARS
from .corruption_training import FIRE_DROPOUT_PROBABILITY
from .diagnostic_artifacts import resolve_artifact
PROTOTYPE_ID = "P00-FireDrop-C00"

def balanced_year_sampling_weights(dataset: Any) -> torch.DoubleTensor:
    """Return per-item weights giving every training year total mass one."""

    per_fire = getattr(dataset, "datapoints_per_fire", None)
    if not isinstance(per_fire, Mapping) or set(per_fire) != set(TRAIN_YEARS):
        raise ValueError("dataset must expose exactly the five training years")
    weights: list[float] = []
    for year in TRAIN_YEARS:
        fires = per_fire[year]
        if not isinstance(fires, Mapping):
            raise ValueError("per-year datapoints must be a mapping")
        count = sum(int(value) for value in fires.values())
        if count <= 0:
            raise ValueError(f"training year has no samples: {year}")
        weights.extend([1.0 / count] * count)
    if len(weights) != len(dataset):
        raise ValueError("sampling weights differ from dataset length")
    return torch.tensor(weights, dtype=torch.double)

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

def _checkpoint_from_record(
    path: Path,
    *,
    prototype_id: str,
    training_policy: Mapping[str, object],
    artifact_map: Path | None = None,
) -> Path:
    record = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("status") != "pass"
        or record.get("prototype_id") != prototype_id
        or record.get("experiment") != "C00"
        or record.get("seed") != 0
        or record.get("max_steps") != 10_000
        or record.get("training_policy") != dict(training_policy)
        or record.get("validation_years") != [2021]
        or record.get("test_enabled") is not False
    ):
        raise ValueError(f"invalid frozen prototype record: {prototype_id}")
    return resolve_artifact(str(record.get("checkpoint", "")), artifact_map=artifact_map, relative_to=path.parent)

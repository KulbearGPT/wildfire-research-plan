"""Minimal year-by-corruption GroupDRO primitives for P09."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .entrypoint import TRAIN_YEARS
from .prototype import (
    FIRE_DROPOUT_PROBABILITY,
    apply_training_fire_and_block_dropout,
)


PROTOTYPE_ID = "P09-YearCorruptionGroupDRO"
ERM_PROTOTYPE_ID = "P10-YearBalancedERM"
CORRECTED_ALPHA_PROTOTYPE_ID = "P12-CorrectedFocalAlphaERM"
FIRE_ONLY_PROTOTYPE_ID = "P13-CorrectedIndexFireDropERM"
BLOCK_STATES = (0, 1, 2)
GROUP_COUNT = len(TRAIN_YEARS) * len(BLOCK_STATES)
TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-4
GROUP_DRO_STEP_SIZE = 0.1


def training_block_states(*, fire_only: bool) -> tuple[int, ...]:
    """Return the registered block states for a training corruption policy."""

    return (0,) if fire_only else BLOCK_STATES


def p13_expert_policy(scenario_id: str) -> str:
    """Select the frozen expert used by the P13 diagnostic router."""

    policy = {"M00": "default", "M01": "fire", "M06": "block", "M07": "block"}
    try:
        return policy[scenario_id]
    except KeyError as error:
        raise ValueError(f"unsupported P13 scenario: {scenario_id}") from error


def corrected_focal_alpha(positive_weight: float) -> float:
    """Convert a raw positive-class ratio to torchvision focal alpha."""

    value = float(positive_weight)
    if value <= 0.0:
        raise ValueError("focal positive weight must be positive")
    if value >= 1.0:
        value /= 1.0 + value
    return value


def erm_objective(per_sample_losses: torch.Tensor) -> torch.Tensor:
    """Return the ordinary empirical mean over sampled examples."""

    if per_sample_losses.ndim != 1 or per_sample_losses.numel() == 0:
        raise ValueError("ERM losses must be a non-empty one-dimensional tensor")
    return per_sample_losses.mean()


def environment_group(year: int, block_state: int) -> int:
    """Map a training year and clean/25%/50% state to one of 15 groups."""

    if year not in TRAIN_YEARS:
        raise ValueError(f"environment year must be one of {TRAIN_YEARS}")
    if block_state not in BLOCK_STATES:
        raise ValueError("block state must be 0, 1, or 2")
    return TRAIN_YEARS.index(year) * len(BLOCK_STATES) + block_state


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


def group_dro_objective(
    per_sample_losses: torch.Tensor,
    group_ids: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    step_size: float = GROUP_DRO_STEP_SIZE,
) -> tuple[torch.Tensor, dict[int, float], torch.Tensor]:
    """Update exponentiated group weights and return active normalized loss."""

    if per_sample_losses.ndim != 1 or group_ids.shape != per_sample_losses.shape:
        raise ValueError("losses and group IDs must be matching one-dimensional tensors")
    if log_weights.shape != (GROUP_COUNT,) or log_weights.requires_grad:
        raise ValueError("log weights must be a fixed 15-element tensor")
    if not 0.0 < step_size <= 1.0:
        raise ValueError("GroupDRO step size must be within (0, 1]")
    if group_ids.numel() == 0 or int(group_ids.min()) < 0 or int(group_ids.max()) >= GROUP_COUNT:
        raise ValueError("group IDs must be within the 15 registered groups")

    active_groups = tuple(int(group) for group in torch.unique(group_ids).tolist())
    means = {
        group: per_sample_losses[group_ids == group].mean()
        for group in active_groups
    }
    with torch.no_grad():
        for group, loss in means.items():
            log_weights[group].add_(step_size * loss.detach())
        log_weights.sub_(log_weights.max())
    weights = torch.softmax(log_weights, dim=0)
    active_mass = torch.stack([weights[group] for group in active_groups]).sum()
    objective = sum(
        weights[group] / active_mass * loss for group, loss in means.items()
    )
    reported = {group: float(loss.detach().cpu()) for group, loss in means.items()}
    return objective, reported, weights.detach().clone()


def install_training_environment_groups(
    upstream_root: Path, *, fire_only: bool = False
) -> None:
    """Patch training samples with uniform block state and a 15-group ID."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_load_imgs = dataset_class.load_imgs
    original_getitem = dataset_class.__getitem__
    block_states = training_block_states(fire_only=fire_only)

    def load_imgs_with_environment(
        self: Any,
        found_fire_year: int,
        found_fire_name: str,
        in_fire_index: int,
    ):
        loaded = original_load_imgs(
            self, found_fire_year, found_fire_name, in_fire_index
        )
        if getattr(self, "is_train", False) is not True or found_fire_year not in TRAIN_YEARS:
            self._environment_group_id = None
            return loaded
        block_state = (
            block_states[0]
            if len(block_states) == 1
            else int(np.random.randint(0, len(block_states)))
        )
        self._environment_group_id = environment_group(
            int(found_fire_year), block_state
        )
        result, _ = apply_training_fire_and_block_dropout(
            loaded,
            is_train=True,
            fire_probability=FIRE_DROPOUT_PROBABILITY,
            block_probability=0.0 if block_state == 0 else 1.0,
            fire_random_value=float(np.random.random()),
            block_random_value=0.0,
            fraction_random_value=0.0 if block_state == 1 else 0.75,
            key_digest=np.random.bytes(32).hex(),
        )
        return result

    def getitem_with_environment(self: Any, index: int):
        sample = original_getitem(self, index)
        group_id = getattr(self, "_environment_group_id", None)
        if group_id is None:
            return sample
        x, target = sample
        return x, target, group_id

    dataset_class.load_imgs = load_imgs_with_environment
    dataset_class.__getitem__ = getitem_with_environment
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index

"""FireDrop and BlockDrop augmentations retained by the B2/B3 baselines."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .missingness import (
    ACTIVE_FIRE_FEATURE,
    DYNAMIC_NON_FIRE_FEATURES,
    RAW_FEATURE_COUNT,
    structured_block_mask,
)


FIRE_DROPOUT_PROBABILITY = 0.3
BLOCK_DROPOUT_PROBABILITY = 0.3


def apply_training_fire_dropout(
    loaded_imgs: tuple[Any, ...],
    *,
    is_train: bool,
    probability: float,
    random_value: float,
) -> tuple[Any, ...]:
    """Zero active-fire history for one sampled training item."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("fire dropout probability must be within [0, 1]")
    if not 0.0 <= random_value < 1.0:
        raise ValueError("random value must be within [0, 1)")
    if not is_train or random_value >= probability:
        return loaded_imgs
    if len(loaded_imgs) not in {2, 3}:
        raise ValueError("upstream image loader must return x/y with optional doys")
    x = np.asarray(loaded_imgs[0])
    if x.ndim != 4 or x.shape[1] != RAW_FEATURE_COUNT:
        raise ValueError("raw training input must have shape (T, 23, H, W)")
    dropped = x.copy()
    dropped[:, ACTIVE_FIRE_FEATURE] = 0.0
    return (dropped, *loaded_imgs[1:])


def apply_training_fire_and_block_dropout(
    loaded_imgs: tuple[Any, ...],
    *,
    is_train: bool,
    fire_probability: float,
    block_probability: float,
    fire_random_value: float,
    block_random_value: float,
    fraction_random_value: float,
    key_digest: str,
) -> tuple[tuple[Any, ...], np.ndarray | None]:
    """Apply independent FireDrop and 25%/50% dynamic-input BlockDrop."""

    result = apply_training_fire_dropout(
        loaded_imgs,
        is_train=is_train,
        probability=fire_probability,
        random_value=fire_random_value,
    )
    if not 0.0 <= block_probability <= 1.0:
        raise ValueError("block dropout probability must be within [0, 1]")
    if not 0.0 <= block_random_value < 1.0:
        raise ValueError("block random value must be within [0, 1)")
    if not 0.0 <= fraction_random_value < 1.0:
        raise ValueError("fraction random value must be within [0, 1)")
    if not is_train or block_random_value >= block_probability:
        return result, None

    x = np.asarray(result[0])
    if x.ndim != 4 or x.shape[1] != RAW_FEATURE_COUNT:
        raise ValueError("raw training input must have shape (T, 23, H, W)")
    fraction = 0.25 if fraction_random_value < 0.5 else 0.50
    mask = structured_block_mask(
        x.shape[2], x.shape[3], fraction, key_digest=key_digest
    )
    blocked = x.copy()
    for feature in DYNAMIC_NON_FIRE_FEATURES:
        blocked[:, feature, mask] = np.nan
    blocked[:, ACTIVE_FIRE_FEATURE, mask] = 0.0
    return (blocked, *result[1:]), mask


def install_training_fire_dropout(
    upstream_root: Path,
    *,
    probability: float = FIRE_DROPOUT_PROBABILITY,
) -> None:
    """Patch the pinned upstream raw loader for training samples only."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_load_imgs = dataset_class.load_imgs

    def load_imgs_with_fire_dropout(self: Any, *args: Any, **kwargs: Any):
        loaded = original_load_imgs(self, *args, **kwargs)
        return apply_training_fire_dropout(
            loaded,
            is_train=getattr(self, "is_train", False) is True,
            probability=probability,
            random_value=float(np.random.random()),
        )

    dataset_class.load_imgs = load_imgs_with_fire_dropout


def install_training_fire_and_block_dropout(
    upstream_root: Path,
    *,
    fire_probability: float = FIRE_DROPOUT_PROBABILITY,
    block_probability: float = BLOCK_DROPOUT_PROBABILITY,
) -> None:
    """Patch the pinned raw loader for the B3 mixed training augmentation."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_load_imgs = dataset_class.load_imgs

    def load_imgs_with_fire_and_block_dropout(
        self: Any, *args: Any, **kwargs: Any
    ):
        loaded = original_load_imgs(self, *args, **kwargs)
        result, _ = apply_training_fire_and_block_dropout(
            loaded,
            is_train=getattr(self, "is_train", False) is True,
            fire_probability=fire_probability,
            block_probability=block_probability,
            fire_random_value=float(np.random.random()),
            block_random_value=float(np.random.random()),
            fraction_random_value=float(np.random.random()),
            key_digest=np.random.bytes(32).hex(),
        )
        return result

    dataset_class.load_imgs = load_imgs_with_fire_and_block_dropout

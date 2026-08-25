"""Minimal train-time corruption used by the P00 rapid prototype."""

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


PROTOTYPE_ID = "P00-FireDrop-C00"
RELIABILITY_PROTOTYPE_ID = "P01-FireDropMask-C00"
BLOCK_PROTOTYPE_ID = "P02-FireBlockDrop-C00"
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


def apply_training_fire_dropout_with_validity(
    loaded_imgs: tuple[Any, ...],
    *,
    is_train: bool,
    probability: float,
    random_value: float,
) -> tuple[tuple[Any, ...], float]:
    """Apply FireDrop and return whether active-fire history remains valid."""

    dropped = is_train and random_value < probability
    result = apply_training_fire_dropout(
        loaded_imgs,
        is_train=is_train,
        probability=probability,
        random_value=random_value,
    )
    return result, 0.0 if dropped else 1.0


def append_fire_validity_channel(x: Any, validity: float) -> Any:
    """Append one constant active-fire-validity map to a processed sample."""

    if len(x.shape) != 4:
        raise ValueError("processed input must have shape (T, C, H, W)")
    shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    if isinstance(x, np.ndarray):
        channel = np.full(shape, validity, dtype=x.dtype)
        return np.concatenate((x, channel), axis=1)

    import torch

    channel = x.new_full(shape, validity)
    return torch.cat((x, channel), dim=1)


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


def install_training_fire_dropout_with_validity(
    upstream_root: Path,
    *,
    probability: float = FIRE_DROPOUT_PROBABILITY,
) -> None:
    """Patch P01 FireDrop plus a synchronized postprocessed validity channel."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_load_imgs = dataset_class.load_imgs
    original_preprocess = dataset_class.preprocess_and_augment
    original_get_n_features = dataset_class.get_n_features

    def load_imgs_with_validity(self: Any, *args: Any, **kwargs: Any):
        loaded = original_load_imgs(self, *args, **kwargs)
        result, validity = apply_training_fire_dropout_with_validity(
            loaded,
            is_train=getattr(self, "is_train", False) is True,
            probability=probability,
            random_value=float(np.random.random()),
        )
        self._active_fire_validity = validity
        return result

    def preprocess_with_validity(self: Any, *args: Any, **kwargs: Any):
        x, y = original_preprocess(self, *args, **kwargs)
        validity = float(getattr(self, "_active_fire_validity", 1.0))
        return append_fire_validity_channel(x, validity), y

    def get_n_features_with_validity(*args: Any, **kwargs: Any) -> int:
        return int(original_get_n_features(*args, **kwargs)) + 1

    dataset_class.load_imgs = load_imgs_with_validity
    dataset_class.preprocess_and_augment = preprocess_with_validity
    dataset_class.get_n_features = staticmethod(get_n_features_with_validity)


def install_training_fire_and_block_dropout(
    upstream_root: Path,
    *,
    fire_probability: float = FIRE_DROPOUT_PROBABILITY,
    block_probability: float = BLOCK_DROPOUT_PROBABILITY,
) -> None:
    """Patch the pinned raw loader for the P02 mixed training augmentation."""

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

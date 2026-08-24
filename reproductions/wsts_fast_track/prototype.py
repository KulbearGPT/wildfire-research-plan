"""Minimal train-time corruption used by the P00 rapid prototype."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .missingness import ACTIVE_FIRE_FEATURE, RAW_FEATURE_COUNT


PROTOTYPE_ID = "P00-FireDrop-C00"
FIRE_DROPOUT_PROBABILITY = 0.3


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

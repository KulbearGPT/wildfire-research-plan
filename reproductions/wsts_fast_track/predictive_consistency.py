"""Paired clean/corrupt data and loss primitives for D1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .prototype import apply_training_fire_and_block_dropout


def bernoulli_kl_from_logits(
    clean_logits: torch.Tensor, corrupt_logits: torch.Tensor
) -> torch.Tensor:
    """Return KL(clean || corrupt) with the clean prediction stop-gradiented."""

    if clean_logits.shape != corrupt_logits.shape or clean_logits.numel() == 0:
        raise ValueError("clean and corrupt logits must have the same nonempty shape")
    teacher = clean_logits.detach()
    probability = torch.sigmoid(teacher)
    per_pixel = probability * (
        F.logsigmoid(teacher) - F.logsigmoid(corrupt_logits)
    ) + (1.0 - probability) * (
        F.logsigmoid(-teacher) - F.logsigmoid(-corrupt_logits)
    )
    return per_pixel.mean()


class CleanCorruptPairDataset:
    """Return one clean/corrupt pair under the exact same augmentation."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        fire_probability: float,
        block_probability: float,
    ) -> None:
        if getattr(base_dataset, "is_train", None) is not True:
            raise ValueError("paired corruption requires an upstream training dataset")
        if not 0.0 <= fire_probability <= 1.0:
            raise ValueError("fire probability must be within [0, 1]")
        if not 0.0 <= block_probability <= 1.0:
            raise ValueError("block probability must be within [0, 1]")
        if getattr(base_dataset, "return_doy", None) is not False:
            raise ValueError("paired corruption does not support day-of-year output")
        self.base = base_dataset
        self.fire_probability = fire_probability
        self.block_probability = block_probability

    def __len__(self) -> int:
        return len(self.base)

    def _select_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.base.remove_duplicate_features and self.base.n_leading_observations > 1:
            return self.base.flatten_and_remove_duplicate_features_(x)
        if self.base.features_to_keep is not None:
            if x.ndim != 4:
                raise ValueError("feature selection requires four-dimensional input")
            return x[:, self.base.features_to_keep, ...]
        return x

    def __getitem__(self, index: int):
        year, fire_name, in_fire_index = (
            self.base.find_image_index_from_dataset_index(index)
        )
        loaded = self.base.load_imgs(year, fire_name, in_fire_index)
        if len(loaded) != 2:
            raise ValueError("paired corruption requires upstream x/y samples")
        clean_x = np.asarray(loaded[0]).copy()
        target = np.asarray(loaded[1]).copy()
        (corrupt_x, corrupt_target), _ = apply_training_fire_and_block_dropout(
            (clean_x, target),
            is_train=True,
            fire_probability=self.fire_probability,
            block_probability=self.block_probability,
            fire_random_value=float(np.random.random()),
            block_random_value=float(np.random.random()),
            fraction_random_value=float(np.random.random()),
            key_digest=np.random.bytes(32).hex(),
        )

        augmentation_state = np.random.get_state()
        processed_clean, clean_target = self.base.preprocess_and_augment(
            clean_x, target
        )
        np.random.set_state(augmentation_state)
        processed_corrupt, corrupt_processed_target = (
            self.base.preprocess_and_augment(corrupt_x, corrupt_target)
        )
        if not torch.equal(clean_target, corrupt_processed_target):
            raise RuntimeError("paired augmentation produced different targets")
        return (
            self._select_features(processed_clean),
            self._select_features(processed_corrupt),
            clean_target,
        )

"""Paired clean/corrupt data and loss primitives for D1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .missingness import structured_block_mask


PROCESSED_FEATURE_COUNT = 40
PROCESSED_ACTIVE_FIRE_VALUE = 38
PROCESSED_ACTIVE_FIRE_BINARY = 39
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))


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
        active_fire_missing_value: float,
        return_reliability: bool = False,
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
        self.active_fire_missing_value = float(active_fire_missing_value)
        self.return_reliability = bool(return_reliability)

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
        processed_clean, clean_target = self.base[index]
        if (
            processed_clean.ndim != 4
            or processed_clean.shape[1] != PROCESSED_FEATURE_COUNT
        ):
            raise ValueError("paired processed input must have shape (T, 40, H, W)")
        processed_corrupt = processed_clean.clone()
        fire_missing = processed_clean.new_zeros(
            (processed_clean.shape[0], 1, *processed_clean.shape[-2:])
        )
        block_missing = torch.zeros_like(fire_missing)
        if float(np.random.random()) < self.fire_probability:
            processed_corrupt[:, PROCESSED_ACTIVE_FIRE_VALUE] = (
                self.active_fire_missing_value
            )
            processed_corrupt[:, PROCESSED_ACTIVE_FIRE_BINARY] = 0.0
            fire_missing.fill_(1.0)
        if float(np.random.random()) < self.block_probability:
            fraction = 0.25 if float(np.random.random()) < 0.5 else 0.50
            mask = torch.as_tensor(
                structured_block_mask(
                    processed_clean.shape[-2],
                    processed_clean.shape[-1],
                    fraction,
                    key_digest=np.random.bytes(32).hex(),
                ),
                dtype=torch.bool,
            )
            block_missing[:, 0] = mask
            processed_corrupt[:, PROCESSED_DYNAMIC_NON_FIRE] = (
                processed_corrupt[:, PROCESSED_DYNAMIC_NON_FIRE].masked_fill(
                    mask[None, None], 0.0
                )
            )
            processed_corrupt[:, PROCESSED_ACTIVE_FIRE_VALUE] = (
                processed_corrupt[:, PROCESSED_ACTIVE_FIRE_VALUE].masked_fill(
                    mask[None], self.active_fire_missing_value
                )
            )
            processed_corrupt[:, PROCESSED_ACTIVE_FIRE_BINARY] = (
                processed_corrupt[:, PROCESSED_ACTIVE_FIRE_BINARY].masked_fill(
                    mask[None], 0.0
                )
            )
        result = (
            self._select_features(processed_clean),
            self._select_features(processed_corrupt),
            clean_target,
        )
        if self.return_reliability:
            return (*result, fire_missing, block_missing)
        return result

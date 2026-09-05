"""Matched processed-space corruption shared by D2-STD and D12-SARP."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .corruption_training import (
    BLOCK_DROPOUT_PROBABILITY,
    FIRE_DROPOUT_PROBABILITY,
)
from .missingness import structured_block_mask


PROCESSED_FEATURE_COUNT = 40
PROCESSED_ACTIVE_FIRE_VALUE = 38
PROCESSED_ACTIVE_FIRE_BINARY = 39
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))


def apply_processed_reliability_corruption(
    x: torch.Tensor,
    *,
    fire_drop: bool,
    block_fraction: float,
    key_digest: str,
    active_fire_missing_value: float,
) -> torch.Tensor:
    """Apply aligned FireDrop/BlockDrop and append binary invalidity."""

    if x.ndim != 4 or x.shape[1] != PROCESSED_FEATURE_COUNT:
        raise ValueError("processed C00 input must have shape (T, 40, H, W)")
    if block_fraction not in {0.0, 0.25, 0.5}:
        raise ValueError("block fraction must be 0, 0.25, or 0.5")
    result = x.clone()
    if fire_drop:
        result[:, PROCESSED_ACTIVE_FIRE_VALUE] = active_fire_missing_value
        result[:, PROCESSED_ACTIVE_FIRE_BINARY] = 0.0

    invalid = torch.zeros(x.shape[-2:], dtype=torch.bool, device=x.device)
    if block_fraction:
        numpy_mask = structured_block_mask(
            x.shape[-2], x.shape[-1], block_fraction, key_digest=key_digest
        )
        invalid = torch.as_tensor(numpy_mask, dtype=torch.bool, device=x.device)
        expanded = invalid[None, None]
        dynamic = result[:, PROCESSED_DYNAMIC_NON_FIRE]
        result[:, PROCESSED_DYNAMIC_NON_FIRE] = dynamic.masked_fill(expanded, 0.0)
        fire_value = result[:, PROCESSED_ACTIVE_FIRE_VALUE]
        result[:, PROCESSED_ACTIVE_FIRE_VALUE] = fire_value.masked_fill(
            invalid[None], active_fire_missing_value
        )
        fire_binary = result[:, PROCESSED_ACTIVE_FIRE_BINARY]
        result[:, PROCESSED_ACTIVE_FIRE_BINARY] = fire_binary.masked_fill(
            invalid[None], 0.0
        )
    invalid_channel = invalid.to(dtype=result.dtype)[None, None].expand(
        result.shape[0], 1, -1, -1
    )
    return torch.cat((result, invalid_channel), dim=1)


class ProcessedReliabilityDataset:
    """Add matched processed-space corruption and an invalidity map."""

    def __init__(self, base_dataset: Any, active_fire_missing_value: float) -> None:
        if getattr(base_dataset, "is_train", None) is not True:
            raise ValueError("reliability training requires a training dataset")
        self.base = base_dataset
        self.active_fire_missing_value = active_fire_missing_value

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, target = self.base[index]
        fire_drop = bool(np.random.random() < FIRE_DROPOUT_PROBABILITY)
        block_drop = bool(np.random.random() < BLOCK_DROPOUT_PROBABILITY)
        block_fraction = 0.25 if np.random.random() < 0.5 else 0.5
        if not block_drop:
            block_fraction = 0.0
        return (
            apply_processed_reliability_corruption(
                x,
                fire_drop=fire_drop,
                block_fraction=block_fraction,
                key_digest=np.random.bytes(32).hex(),
                active_fire_missing_value=self.active_fire_missing_value,
            ),
            target,
        )

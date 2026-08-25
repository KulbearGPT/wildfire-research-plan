"""P04 single-forward spatial residual gate on frozen P00 features."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .missingness import structured_block_mask


PROTOTYPE_ID = "P04-FrozenP00-SpatialResidualGate"
SPATIAL_PROTOTYPE_ID = "P05-FrozenP00-SpatialResidualGate3x3"
GATE_TRAINING_STEPS = 1_000
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))


def apply_masked_residual(
    default_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    missing_mask: torch.Tensor,
) -> torch.Tensor:
    """Add a learned correction only at pixels declared spatially missing."""

    if (
        default_logits.shape != residual_logits.shape
        or missing_mask.shape != default_logits.shape
    ):
        raise ValueError("default logits, residual, and mask must have identical shapes")
    return default_logits + residual_logits * missing_mask.to(default_logits.dtype)


def apply_processed_block_dropout(
    processed: torch.Tensor,
    *,
    fraction: float,
    key_digest: str,
    normalized_active_fire_zero: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M06/M07-equivalent corruption after synchronized crop/rotation."""

    if processed.ndim != 4 or processed.shape[1] != 40:
        raise ValueError("processed C00 input must have shape (T, 40, H, W)")
    mask_array = structured_block_mask(
        processed.shape[-2],
        processed.shape[-1],
        fraction,
        key_digest=key_digest,
    )
    mask = torch.as_tensor(mask_array, dtype=torch.bool, device=processed.device)
    blocked = processed.clone()
    for channel in PROCESSED_DYNAMIC_NON_FIRE:
        blocked[:, channel, mask] = 0.0
    blocked[:, 38, mask] = normalized_active_fire_zero
    blocked[:, 39, mask] = 0.0
    mask_channel = mask.to(processed.dtype)[None, None].expand(
        processed.shape[0], 1, -1, -1
    )
    return torch.cat((blocked, mask_channel), dim=1), mask


def install_training_processed_block_dropout(upstream_root: Path) -> None:
    """Patch C00 preprocessing to emit always-blocked inputs plus routing mask."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_preprocess = dataset_class.preprocess_and_augment

    def preprocess_with_block(self: Any, *args: Any, **kwargs: Any):
        processed, target = original_preprocess(self, *args, **kwargs)
        if getattr(self, "is_train", False) is not True:
            mask = processed.new_zeros(
                (processed.shape[0], 1, processed.shape[-2], processed.shape[-1])
            )
            return torch.cat((processed, mask), dim=1), target
        fraction = 0.25 if float(np.random.random()) < 0.5 else 0.50
        normalized_zero = float(-self.means[0, 22, 0, 0] / self.stds[0, 22, 0, 0])
        routed, _ = apply_processed_block_dropout(
            processed,
            fraction=fraction,
            key_digest=np.random.bytes(32).hex(),
            normalized_active_fire_zero=normalized_zero,
        )
        return routed, target

    dataset_class.preprocess_and_augment = preprocess_with_block


class FrozenSpatialResidualGate(torch.nn.Module):
    """Add a minimal spatial correction to frozen P00 decoder features."""

    def __init__(
        self,
        default_model: torch.nn.Module,
        *,
        residual_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if residual_kernel_size not in {1, 3}:
            raise ValueError("residual kernel size must be 1 or 3")
        self.default_model = default_model
        self.default_model.requires_grad_(False)
        self.default_model.eval()
        self.residual_head = torch.nn.Conv2d(
            16,
            1,
            kernel_size=residual_kernel_size,
            padding=residual_kernel_size // 2,
        )
        torch.nn.init.zeros_(self.residual_head.weight)
        torch.nn.init.zeros_(self.residual_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.default_model.eval()
        return self

    def trainable_parameters(self):
        return self.residual_head.parameters()

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        if (
            routed_input.ndim != 5
            or routed_input.shape[1] != 1
            or routed_input.shape[2] != 41
        ):
            raise ValueError("gate input must have shape (B, 1, 41, H, W)")
        features = routed_input[:, 0, :40]
        missing_mask = routed_input[:, 0, 40:41].bool()
        with torch.no_grad():
            encoded = self.default_model.model.encoder(features)
            decoder_features = self.default_model.model.decoder(*encoded)
            default_logits = self.default_model.model.segmentation_head(
                decoder_features
            )
        residual_logits = self.residual_head(decoder_features.detach())
        return apply_masked_residual(
            default_logits, residual_logits, missing_mask
        )

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.default_model.compute_loss(logits, target)

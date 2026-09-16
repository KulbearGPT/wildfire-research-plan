"""Severity-adaptive reliability prompting for the C02 T=5 model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .contract import MULTI_FEATURES
from .missingness import structured_block_mask
from .corruption_training import BLOCK_DROPOUT_PROBABILITY, FIRE_DROPOUT_PROBABILITY
from .reliability_prompt_pyramid import (
    PROMPT_POOLING,
    SEVERITY_THRESHOLD,
    apply_reliability_prompts,
)
from .reliability_token_conv import InputReliabilityTokenConv2d
from .train_reliability_normalized import PROCESSED_DYNAMIC_NON_FIRE


TEMPORAL_STEPS = 5
TEMPORAL_FEATURE_COUNT = len(MULTI_FEATURES)
TEMPORAL_ACTIVE_FIRE_VALUE = MULTI_FEATURES.index(38)
TEMPORAL_ACTIVE_FIRE_BINARY = MULTI_FEATURES.index(39)
TEMPORAL_DYNAMIC_NON_FIRE = tuple(
    index
    for index, original in enumerate(MULTI_FEATURES)
    if original in PROCESSED_DYNAMIC_NON_FIRE
)


def apply_processed_temporal_reliability_corruption(
    x: torch.Tensor,
    *,
    fire_drop: bool,
    block_fraction: float,
    key_digest: str,
    active_fire_missing_value: float,
) -> torch.Tensor:
    """Apply D2's processed corruption to the frozen C02 feature selection."""

    if x.ndim != 4 or x.shape[:2] != (
        TEMPORAL_STEPS,
        TEMPORAL_FEATURE_COUNT,
    ):
        raise ValueError("processed C02 input must have shape (5, 33, H, W)")
    if block_fraction not in {0.0, 0.25, 0.5}:
        raise ValueError("block fraction must be 0, 0.25, or 0.5")
    result = x.clone()
    if fire_drop:
        result[:, TEMPORAL_ACTIVE_FIRE_VALUE] = active_fire_missing_value
        result[:, TEMPORAL_ACTIVE_FIRE_BINARY] = 0.0

    invalid = torch.zeros(x.shape[-2:], dtype=torch.bool, device=x.device)
    if block_fraction:
        invalid = torch.as_tensor(
            structured_block_mask(
                x.shape[-2], x.shape[-1], block_fraction, key_digest=key_digest
            ),
            dtype=torch.bool,
            device=x.device,
        )
        result[:, TEMPORAL_DYNAMIC_NON_FIRE] = result[
            :, TEMPORAL_DYNAMIC_NON_FIRE
        ].masked_fill(invalid[None, None], 0.0)
        result[:, TEMPORAL_ACTIVE_FIRE_VALUE] = result[
            :, TEMPORAL_ACTIVE_FIRE_VALUE
        ].masked_fill(invalid[None], active_fire_missing_value)
        result[:, TEMPORAL_ACTIVE_FIRE_BINARY] = result[
            :, TEMPORAL_ACTIVE_FIRE_BINARY
        ].masked_fill(invalid[None], 0.0)
    invalid_channel = invalid.to(dtype=result.dtype)[None, None].expand(
        result.shape[0], 1, -1, -1
    )
    return torch.cat((result, invalid_channel), dim=1)


class ProcessedTemporalReliabilityDataset:
    """Add matched single-view corruption and invalidity to C02 samples."""

    def __init__(self, base_dataset: Any, active_fire_missing_value: float) -> None:
        if getattr(base_dataset, "is_train", None) is not True:
            raise ValueError("T5 reliability training requires an upstream train set")
        if getattr(base_dataset, "n_leading_observations", None) != TEMPORAL_STEPS:
            raise ValueError("T5 reliability training requires five time steps")
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
            apply_processed_temporal_reliability_corruption(
                x,
                fire_drop=fire_drop,
                block_fraction=block_fraction,
                key_digest=np.random.bytes(32).hex(),
                active_fire_missing_value=self.active_fire_missing_value,
            ),
            target,
        )


class TemporalSeverityAdaptiveReliabilityPrompting(torch.nn.Module):
    """Route spatial reliability prompts before unchanged temporal fusion."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        base_model.model.encoder.conv1 = InputReliabilityTokenConv2d(
            base_model.model.encoder.conv1
        )
        self.base_model = base_model
        self.prompt_channels = tuple(base_model.model.encoder.out_channels[1:])
        self.prompt_tokens = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(channels)) for channels in self.prompt_channels]
        )
        self.prompt_parameter_count = sum(
            token.numel() for token in self.prompt_tokens
        ) + self.input_token.invalid_token.numel()
        self.prompt_pooling = PROMPT_POOLING

    @property
    def input_token(self) -> InputReliabilityTokenConv2d:
        return self.base_model.model.encoder.conv1

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1:3] != (
            TEMPORAL_STEPS,
            TEMPORAL_FEATURE_COUNT + 1,
        ):
            raise ValueError("T5 SARP input must have shape [B,5,34,H,W]")
        features = packed[:, :, :TEMPORAL_FEATURE_COUNT]
        invalidity = packed[:, :, TEMPORAL_FEATURE_COUNT:].clamp(0.0, 1.0)
        severity = invalidity.mean(dim=(1, 2, 3, 4), keepdim=True)
        severe = severity > SEVERITY_THRESHOLD
        mild = (severity > 0.0) & ~severe
        input_invalidity = invalidity * severe.to(invalidity.dtype)
        deep_invalidity = invalidity * mild.to(invalidity.dtype)

        per_scale: list[list[torch.Tensor]] = [
            [] for _ in self.base_model.model.encoder.out_channels
        ]
        for step in range(TEMPORAL_STEPS):
            encoded = tuple(
                self.base_model.model.encoder(
                    torch.cat(
                        (features[:, step], input_invalidity[:, step]), dim=1
                    )
                )
            )
            if tuple(item.shape[1] for item in encoded[1:]) != self.prompt_channels:
                raise ValueError("runtime encoder channels differ from T5 SARP")
            prompted = apply_reliability_prompts(
                encoded[1:], deep_invalidity[:, step], self.prompt_tokens
            )
            per_scale[0].append(features[:, step])
            for scale, item in enumerate(prompted, start=1):
                per_scale[scale].append(item)

        batch = packed.shape[0]
        positions = torch.arange(
            TEMPORAL_STEPS, device=packed.device
        ).unsqueeze(0).repeat(batch, 1)
        last_stage = torch.stack(per_scale[-1], dim=1)
        aggregated_last, attention = self.base_model.ltae(
            last_stage, batch_positions=positions
        )
        aggregated_skips = []
        for scale in range(1, len(per_scale) - 1):
            stage = torch.stack(per_scale[scale], dim=1)
            aggregated_skips.append(
                self.base_model.temporal_aggregator(stage, attn_mask=attention)
            )
        decoder_features = [per_scale[0][0], *aggregated_skips, aggregated_last]
        decoded = self.base_model.model.decoder(*decoder_features)
        return self.base_model.model.segmentation_head(decoded)

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.base_model.compute_loss(logits, target)

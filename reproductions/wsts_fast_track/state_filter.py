"""Recurrent fire-state filter with observation-consistent updates."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .latent_state_common import (
    ACTIVE_BINARY,
    FrozenP00BeliefModel,
    PROCESSED_FEATURES,
    STATE_CONTEXT,
    observation_consistent_state,
)


class _ConvBlock(torch.nn.Module):
    """Two GroupNorm/SiLU convolutions at one U-Net resolution."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(8, output_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(8, output_channels),
            torch.nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _RecurrentUpdater(torch.nn.Module):
    """Three-level U-Net that maps context, state, and reliability to a logit."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_16 = _ConvBlock(35, 16)
        self.encoder_32 = _ConvBlock(16, 32)
        self.encoder_64 = _ConvBlock(32, 64)
        self.bottleneck = _ConvBlock(64, 96)
        self.decoder_64 = _ConvBlock(96 + 64, 64)
        self.decoder_32 = _ConvBlock(64 + 32, 32)
        self.decoder_16 = _ConvBlock(32 + 16, 16)
        self.output = torch.nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _upsample(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level_16 = self.encoder_16(inputs)
        level_32 = self.encoder_32(functional.max_pool2d(level_16, kernel_size=2))
        level_64 = self.encoder_64(functional.max_pool2d(level_32, kernel_size=2))
        bottleneck = self.bottleneck(functional.max_pool2d(level_64, kernel_size=2))

        up_64 = self.decoder_64(
            torch.cat((self._upsample(bottleneck, level_64), level_64), dim=1)
        )
        up_32 = self.decoder_32(
            torch.cat((self._upsample(up_64, level_32), level_32), dim=1)
        )
        up_16 = self.decoder_16(
            torch.cat((self._upsample(up_32, level_16), level_16), dim=1)
        )
        return self.output(up_16)


class RecurrentBeliefFilter(FrozenP00BeliefModel):
    """Infer a fire belief state by unrolling one shared spatial updater."""

    def __init__(self, default_model: torch.nn.Module, history: int) -> None:
        super().__init__(default_model, history)
        self.updater = _RecurrentUpdater()

    def trainable_parameters(self):
        """Return only the learned recurrent updater's parameters."""

        return self.updater.parameters()

    def infer_state_logits(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        if (
            features.ndim != 5
            or features.shape[1] != self.history
            or features.shape[2] != PROCESSED_FEATURES
            or features.shape[-2] <= 0
            or features.shape[-1] <= 0
        ):
            raise ValueError("features must have shape [B,history,40,H,W]")
        if (
            reliability.ndim != 5
            or reliability.shape != (
                features.shape[0], self.history, 1, features.shape[-2], features.shape[-1]
            )
        ):
            raise ValueError("reliability must have shape [B,history,1,H,W]")
        if reliability.device != features.device:
            raise ValueError("features and reliability must share a device")

        state = features.new_zeros((features.shape[0], 1, *features.shape[-2:]))
        proposal = state
        for step in range(self.history):
            proposal = self.updater(
                torch.cat(
                    (
                        features[:, step, STATE_CONTEXT],
                        state,
                        reliability[:, step],
                    ),
                    dim=1,
                )
            )
            state = observation_consistent_state(
                features[:, step, ACTIVE_BINARY : ACTIVE_BINARY + 1],
                proposal,
                reliability[:, step],
            )
        return proposal

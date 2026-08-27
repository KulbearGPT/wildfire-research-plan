"""Reconstruction-only fire-state baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .latent_state_common import FrozenP00BeliefModel


class _ConvBlock(torch.nn.Module):
    """Two GroupNorm/SiLU convolutional layers at one U-Net resolution."""

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


class TinyMaskUNet(torch.nn.Module):
    """Three-level U-Net that estimates one current active-fire logit."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.encoder_1 = _ConvBlock(input_channels, 16)
        self.encoder_2 = _ConvBlock(16, 32)
        self.encoder_3 = _ConvBlock(32, 64)
        self.bottleneck = _ConvBlock(64, 96)
        self.decoder_3 = _ConvBlock(96 + 64, 64)
        self.decoder_2 = _ConvBlock(64 + 32, 32)
        self.decoder_1 = _ConvBlock(32 + 16, 16)
        self.output = torch.nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _upsample(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip_1 = self.encoder_1(inputs)
        skip_2 = self.encoder_2(functional.max_pool2d(skip_1, kernel_size=2))
        skip_3 = self.encoder_3(functional.max_pool2d(skip_2, kernel_size=2))
        latent = self.bottleneck(functional.max_pool2d(skip_3, kernel_size=2))
        decoded_3 = self.decoder_3(
            torch.cat((self._upsample(latent, skip_3), skip_3), dim=1)
        )
        decoded_2 = self.decoder_2(
            torch.cat((self._upsample(decoded_3, skip_2), skip_2), dim=1)
        )
        decoded_1 = self.decoder_1(
            torch.cat((self._upsample(decoded_2, skip_1), skip_1), dim=1)
        )
        return self.output(decoded_1)


class ReconstructionFirstBeliefState(FrozenP00BeliefModel):
    """Reconstruct a state before passing it to the frozen P00 forecaster."""

    def __init__(self, default_model: torch.nn.Module, history: int) -> None:
        super().__init__(default_model, history)
        self.reconstructor = TinyMaskUNet(input_channels=36 * history)

    def infer_state_logits(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        state_inputs = torch.cat(
            (features[:, :, :33], features[:, :, 38:40], reliability), dim=2
        ).flatten(1, 2)
        return self.reconstructor(state_inputs)

    def trainable_parameters(self):
        """Return the reconstruction module parameters and exclude frozen P00."""

        return self.reconstructor.parameters()

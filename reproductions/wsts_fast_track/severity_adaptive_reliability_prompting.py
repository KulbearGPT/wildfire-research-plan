"""Severity-adaptive spatial reliability prompts for the retained D12 method."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


PROMPT_CHANNELS = (64, 64, 128, 256, 512)
PROMPT_PARAMETER_COUNT = sum(PROMPT_CHANNELS) + PROMPT_CHANNELS[0]
PROMPT_POOLING = "adaptive-area-average"
INPUT_PROMPT = "local-invalid-coverage-token"
SEVERITY_THRESHOLD = 0.375


class InputReliabilityTokenConv2d(nn.Module):
    """Preserve the source convolution and add one learned invalid-region token."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        if conv.dilation != (1, 1):
            raise ValueError("reliability token requires unit convolution dilation")
        self.conv = conv
        self.invalid_token = nn.Parameter(torch.zeros(conv.out_channels))
        self.in_channels = conv.in_channels + 1
        self.out_channels = conv.out_channels

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 4 or packed.shape[1] != self.in_channels:
            raise ValueError(
                "packed input must contain features plus one invalidity map"
            )
        features = packed[:, :-1]
        invalid = packed[:, -1:].to(dtype=features.dtype)
        base = self.conv(features)
        coverage = F.avg_pool2d(
            invalid,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            count_include_pad=False,
        )
        return base + coverage * self.invalid_token[None, :, None, None]


def apply_reliability_prompts(
    features: Sequence[torch.Tensor],
    invalidity: torch.Tensor,
    tokens: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Add channel tokens scaled by invalid-cell coverage at each resolution."""

    if (
        invalidity.ndim != 4
        or invalidity.shape[1] != 1
        or len(features) == 0
        or len(features) != len(tokens)
    ):
        raise ValueError("prompt features, invalidity, and tokens are incompatible")
    prompted: list[torch.Tensor] = []
    for feature, token in zip(features, tokens, strict=True):
        if (
            feature.ndim != 4
            or feature.shape[0] != invalidity.shape[0]
            or token.ndim != 1
            or token.shape[0] != feature.shape[1]
        ):
            raise ValueError("one reliability prompt scale is incompatible")
        coverage = F.adaptive_avg_pool2d(invalidity, feature.shape[-2:])
        prompted.append(
            feature + coverage.to(feature.dtype) * token[None, :, None, None]
        )
    return tuple(prompted)


class SeverityAdaptiveReliabilityPrompting(nn.Module):
    """Route mild blocks to deep prompts and severe blocks to the input token."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        base_model.model.encoder.conv1 = InputReliabilityTokenConv2d(
            base_model.model.encoder.conv1
        )
        self.base_model = base_model
        encoder_channels = tuple(self.base_model.model.encoder.out_channels)
        if len(encoder_channels) < 2:
            raise ValueError("D12 requires a multi-scale encoder")
        self.prompt_channels = encoder_channels[1:]
        self.prompt_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels)) for channels in self.prompt_channels]
        )
        self.prompt_parameter_count = self.input_token.invalid_token.numel() + sum(
            parameter.numel() for parameter in self.prompt_tokens
        )

    @property
    def input_token(self) -> InputReliabilityTokenConv2d:
        return self.base_model.model.encoder.conv1

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != 1 or packed.shape[2] != 41:
            raise ValueError("D12 input must have shape [B,1,41,H,W]")
        features = packed[:, 0, :40]
        invalidity = packed[:, 0, 40:41].clamp(0.0, 1.0)
        severity = invalidity.mean(dim=(2, 3), keepdim=True)
        severe = severity > SEVERITY_THRESHOLD
        mild = (severity > 0.0) & ~severe
        input_invalidity = invalidity * severe.to(invalidity.dtype)
        deep_invalidity = invalidity * mild.to(invalidity.dtype)
        encoded = tuple(
            self.base_model.model.encoder(
                torch.cat((features, input_invalidity), dim=1)
            )
        )
        if tuple(feature.shape[1] for feature in encoded[1:]) != self.prompt_channels:
            raise ValueError("runtime encoder channels differ from D12 initialization")
        prompted = apply_reliability_prompts(
            encoded[1:], deep_invalidity, self.prompt_tokens
        )
        decoder_features = self.base_model.model.decoder(encoded[0], *prompted)
        return self.base_model.model.segmentation_head(decoder_features)

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.base_model.compute_loss(logits, target)

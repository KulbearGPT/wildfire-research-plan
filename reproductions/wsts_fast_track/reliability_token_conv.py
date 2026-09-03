"""Minimal invalid-region token for the D4 first-convolution prototype."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class InputReliabilityTokenConv2d(nn.Module):
    """Add one learned output token in proportion to local invalid coverage."""

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

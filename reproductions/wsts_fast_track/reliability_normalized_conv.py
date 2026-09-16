"""Mask-normalized convolution primitive for the D2 prototype."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair


class ReliabilityNormalizedConv2d(nn.Module):
    """Apply a Conv2d using only valid pixels and local renormalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_conv(cls, source: nn.Conv2d) -> "ReliabilityNormalizedConv2d":
        """Clone one ordinary convolution's contract and parameters."""

        result = cls(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
        )
        with torch.no_grad():
            result.weight.copy_(source.weight)
            if result.bias is not None and source.bias is not None:
                result.bias.copy_(source.bias)
        return result

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("input channels differ from normalized convolution")
        if valid.shape != (x.shape[0], 1, x.shape[2], x.shape[3]):
            raise ValueError("validity must have shape (B, 1, H, W)")
        if not torch.is_floating_point(valid):
            raise ValueError("validity must be floating point")

        validity = valid.to(dtype=x.dtype)
        raw = F.conv2d(
            x * validity,
            self.weight,
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        count_kernel = torch.ones(
            (1, 1, *self.kernel_size), dtype=x.dtype, device=x.device
        )
        valid_count = F.conv2d(
            validity,
            count_kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        available_count = F.conv2d(
            torch.ones_like(validity),
            count_kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        has_valid = valid_count > 0
        normalized = raw * (available_count / valid_count.clamp_min(1.0))
        if self.bias is not None:
            normalized = normalized + self.bias[None, :, None, None]
        return normalized * has_valid.to(dtype=normalized.dtype)


class InputReliabilityNormalizedConv2d(nn.Module):
    """Consume features plus one final invalidity channel inside an encoder."""

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        self.normalized = ReliabilityNormalizedConv2d.from_conv(source)
        self.in_channels = source.in_channels + 1
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        if combined.ndim != 4 or combined.shape[1] != self.in_channels:
            raise ValueError("combined encoder input must end with invalidity")
        features = combined[:, :-1]
        invalidity = combined[:, -1:].clamp(0.0, 1.0)
        return self.normalized(features, 1.0 - invalidity)

from __future__ import annotations

import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track.reliability_normalized_conv import (
    ReliabilityNormalizedConv2d,
)


def _layer() -> ReliabilityNormalizedConv2d:
    source = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        source.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 18)
        source.bias.fill_(0.25)
    return ReliabilityNormalizedConv2d.from_conv(source)


def test_all_valid_mask_matches_the_original_convolution() -> None:
    layer = _layer()
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) / 10
    valid = torch.ones(1, 1, 4, 4)

    actual = layer(x, valid)
    expected = F.conv2d(
        x,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )

    torch.testing.assert_close(actual, expected)


def test_invalid_placeholder_values_cannot_change_the_output() -> None:
    layer = _layer()
    x = torch.ones(1, 2, 5, 5)
    changed = x.clone()
    changed[:, :, 1:4, 1:4] = 10_000.0
    valid = torch.ones(1, 1, 5, 5)
    valid[:, :, 1:4, 1:4] = 0.0

    first = layer(x, valid)
    second = layer(changed, valid)

    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


def test_reliability_normalized_convolution_has_finite_gradients() -> None:
    layer = _layer()
    x = torch.randn(2, 2, 4, 4, requires_grad=True)
    valid = torch.ones(2, 1, 4, 4)
    valid[:, :, :2, :2] = 0.0

    output = layer(x, valid)
    assert output.shape == (2, 1, 4, 4)
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()

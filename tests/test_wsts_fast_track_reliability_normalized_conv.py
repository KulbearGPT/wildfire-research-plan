from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track.evaluate_reliability_normalized import (
    validate_d2_checkpoint,
)
from reproductions.wsts_fast_track.reliability_normalized_conv import (
    InputReliabilityNormalizedConv2d,
    ReliabilityNormalizedConv2d,
)
from reproductions.wsts_fast_track.train_reliability_normalized import (
    PROCESSED_ACTIVE_FIRE_BINARY,
    PROCESSED_ACTIVE_FIRE_VALUE,
    apply_processed_reliability_corruption,
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


def test_combined_input_wrapper_uses_the_last_channel_as_invalidity() -> None:
    source = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
    source.weight.data.fill_(1.0)
    layer = InputReliabilityNormalizedConv2d(source)
    features = torch.ones(1, 2, 4, 4)
    no_invalidity = torch.zeros(1, 1, 4, 4)

    actual = layer(torch.cat((features, no_invalidity), dim=1))
    expected = source(features)

    torch.testing.assert_close(actual, expected)
    assert layer.in_channels == 3


def test_processed_corruption_appends_aligned_invalidity_channel() -> None:
    x = torch.ones(1, 40, 4, 4)
    result = apply_processed_reliability_corruption(
        x,
        fire_drop=False,
        block_fraction=0.5,
        key_digest="00" * 32,
        active_fire_missing_value=-0.25,
    )

    assert result.shape == (1, 41, 4, 4)
    invalid = result[0, 40].bool()
    assert int(invalid.sum()) == 8
    assert torch.all(result[:, PROCESSED_ACTIVE_FIRE_VALUE, invalid] == -0.25)
    assert torch.all(result[:, PROCESSED_ACTIVE_FIRE_BINARY, invalid] == 0.0)
    assert torch.all(result[:, 12:15] == 1.0)
    assert torch.all(result[:, 40, ~invalid] == 0.0)


def test_d2_checkpoint_requires_one_of_the_matched_variants() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D2-RNC",
        "matched_pair": "D2",
        "variant": "rnc",
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "processed_space_matched_corruption": True,
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }
    assert validate_d2_checkpoint(payload) == "rnc"

    payload["variant"] = "standard"
    with pytest.raises(ValueError, match="D2 checkpoint"):
        validate_d2_checkpoint(payload)


def test_d2_runner_rejects_unknown_variant_before_cluster_setup(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_normalized_on_nibi.sh"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    record = tmp_path / "b3.json"
    record.write_text("{}\n", encoding="utf-8")
    invalid = subprocess.run(
        [str(runner), str(record), "other"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "variant must be standard or rnc" in invalid.stderr
    text = runner.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "train_reliability_normalized" in text
    assert "evaluate_reliability_normalized" in text

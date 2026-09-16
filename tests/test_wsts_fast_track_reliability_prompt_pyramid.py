from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track.evaluate_reliability_prompt_pyramid import (
    validate_rpp_checkpoint,
)
from reproductions.wsts_fast_track.reliability_prompt_pyramid import (
    ReliabilityPromptPyramid,
    apply_reliability_prompts,
)


def test_scale_matched_prompts_use_area_coverage() -> None:
    features = (
        torch.zeros(1, 2, 4, 4),
        torch.zeros(1, 3, 2, 2),
    )
    invalid = torch.zeros(1, 1, 4, 4)
    invalid[:, :, :2, :2] = 1.0
    tokens = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0, 3.0]),
    )

    prompted = apply_reliability_prompts(features, invalid, tokens)

    expected_first = invalid * tokens[0][None, :, None, None]
    expected_second = F.adaptive_avg_pool2d(invalid, (2, 2)) * tokens[1][
        None, :, None, None
    ]
    torch.testing.assert_close(prompted[0], expected_first)
    torch.testing.assert_close(prompted[1], expected_second)


class _Encoder(torch.nn.Module):
    out_channels = (40, 2, 4)

    def forward(self, x: torch.Tensor):
        return (
            x,
            x[:, :2],
            F.avg_pool2d(x[:, :4], kernel_size=2),
        )


class _Decoder(torch.nn.Module):
    def forward(self, _input, shallow: torch.Tensor, deep: torch.Tensor):
        shallow_four = torch.cat((shallow, shallow), dim=1)
        return F.interpolate(deep, scale_factor=2, mode="nearest") + shallow_four


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.segmentation_head = torch.nn.Conv2d(4, 1, 1)


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.model.encoder(x[:, 0])
        return self.model.segmentation_head(self.model.decoder(*encoded))

    @staticmethod
    def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()


def test_zero_prompts_and_valid_inputs_are_exact_base_paths() -> None:
    base = _Base()
    model = ReliabilityPromptPyramid(base)
    features = torch.randn(2, 1, 40, 4, 4)
    valid = torch.cat((features, torch.zeros(2, 1, 1, 4, 4)), dim=2)

    torch.testing.assert_close(model(valid), base(features))
    with torch.no_grad():
        for token in model.prompt_tokens:
            token.fill_(3.0)
    torch.testing.assert_close(model(valid), base(features))
    assert model.prompt_parameter_count == 6


def test_invalid_support_updates_prompts_and_backbone() -> None:
    base = _Base()
    model = ReliabilityPromptPyramid(base)
    features = torch.randn(2, 1, 40, 4, 4)
    invalid = torch.zeros(2, 1, 1, 4, 4)
    invalid[:, :, :, :2, :2] = 1.0
    packed = torch.cat((features, invalid), dim=2)

    model(packed).sum().backward()

    assert all(token.grad is not None for token in model.prompt_tokens)
    assert any(torch.count_nonzero(token.grad) > 0 for token in model.prompt_tokens)
    assert base.model.segmentation_head.weight.grad is not None


def test_rpp_checkpoint_requires_frozen_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D10-RPP",
        "matched_pair": "D10",
        "base_control": "D2-STD",
        "closest_ablation": "D4-TOKEN",
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "variant": "prompt-pyramid",
        "processed_space_matched_corruption": True,
        "prompt_channels": [64, 64, 128, 256, 512],
        "prompt_parameter_count": 1_024,
        "prompt_pooling": "adaptive-area-average",
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_rpp_checkpoint(payload) == "D10-RPP"

    payload["prompt_parameter_count"] = 64
    with pytest.raises(ValueError, match="RPP checkpoint"):
        validate_rpp_checkpoint(payload)

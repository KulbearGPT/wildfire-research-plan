from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track.evaluate_reliability_prompt_pyramid import (
    validate_crpp_checkpoint,
)
from reproductions.wsts_fast_track.reliability_prompt_pyramid import (
    CompleteReliabilityPromptPyramid,
)


class _Encoder(torch.nn.Module):
    out_channels = (40, 2, 4)

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(40, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        shallow = self.conv1(x)
        deep = F.avg_pool2d(torch.cat((shallow, shallow), dim=1), kernel_size=2)
        return x, shallow, deep


class _Decoder(torch.nn.Module):
    def forward(self, _input, shallow: torch.Tensor, deep: torch.Tensor):
        return F.interpolate(deep, scale_factor=2, mode="nearest") + torch.cat(
            (shallow, shallow), dim=1
        )


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


def test_complete_pyramid_is_exact_on_valid_input_for_any_tokens() -> None:
    base = _Base()
    reference = copy.deepcopy(base)
    model = CompleteReliabilityPromptPyramid(base)
    with torch.no_grad():
        model.input_token.invalid_token.fill_(2.0)
        for token in model.prompt_tokens:
            token.fill_(3.0)
    features = torch.randn(2, 1, 40, 4, 4)
    valid = torch.cat((features, torch.zeros(2, 1, 1, 4, 4)), dim=2)

    torch.testing.assert_close(model(valid), reference(features))
    assert model.prompt_parameter_count == 8


def test_complete_pyramid_updates_input_and_hierarchical_tokens() -> None:
    model = CompleteReliabilityPromptPyramid(_Base())
    features = torch.randn(2, 1, 40, 4, 4)
    invalid = torch.zeros(2, 1, 1, 4, 4)
    invalid[:, :, :, :2, :2] = 1.0

    model(torch.cat((features, invalid), dim=2)).sum().backward()

    assert model.input_token.invalid_token.grad is not None
    assert torch.count_nonzero(model.input_token.invalid_token.grad) > 0
    assert all(token.grad is not None for token in model.prompt_tokens)
    assert all(torch.count_nonzero(token.grad) > 0 for token in model.prompt_tokens)


def test_crpp_checkpoint_requires_complete_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D11-CRPP",
        "matched_pair": "D11",
        "base_control": "D2-STD",
        "closest_ablations": ["D4-TOKEN", "D10-RPP"],
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "variant": "complete-prompt-pyramid",
        "processed_space_matched_corruption": True,
        "prompt_channels": [64, 64, 128, 256, 512],
        "prompt_parameter_count": 1_088,
        "prompt_pooling": "adaptive-area-average",
        "input_prompt": "local-invalid-coverage-token",
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_crpp_checkpoint(payload) == "D11-CRPP"

    payload["prompt_parameter_count"] = 1_024
    with pytest.raises(ValueError, match="CRPP checkpoint"):
        validate_crpp_checkpoint(payload)


def test_crpp_runner_has_valid_shell_contract(tmp_path: Path) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_complete_reliability_prompt_pyramid_on_nibi.sh"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    record = tmp_path / "b3.json"
    record.write_text("{}\n", encoding="utf-8")

    invalid = subprocess.run(
        [str(runner), str(record), "extra"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert invalid.returncode == 2
    assert "usage:" in invalid.stderr


def test_generic_heldout_runner_accepts_crpp_before_year_validation(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_evaluation_on_nibi.sh"
    )
    checkpoint = tmp_path / "crpp.pt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [str(runner), "crpp", str(checkpoint), "2021", "D11-CRPP"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "year must be 2022 or 2023" in result.stderr

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from reproductions.wsts_fast_track.counterfactual_reliability_adapter import (
    CounterfactualReliabilityAdapter,
    pack_reliability_input,
)
from reproductions.wsts_fast_track.evaluate_counterfactual_reliability_adapter import (
    validate_ffca_checkpoint,
)


class _Encoder(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return (x[:, :16],)


class _Decoder(torch.nn.Module):
    def forward(self, feature: torch.Tensor):
        return feature


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.segmentation_head = torch.nn.Conv2d(16, 1, 1)


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()

    @staticmethod
    def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()


def _base_logits(base: _Base, features: torch.Tensor) -> torch.Tensor:
    encoded = base.model.encoder(features[:, 0])
    return base.model.segmentation_head(base.model.decoder(*encoded))


def test_block_scope_has_frozen_capacity_and_fire_only_is_exact_bypass() -> None:
    base = _Base()
    model = CounterfactualReliabilityAdapter(base, adapter_scope="block")
    with torch.no_grad():
        model.adapter[-1].weight.fill_(1.0)
        model.adapter[-1].bias.fill_(1.0)
    features = torch.randn(2, 1, 40, 5, 5)
    fire = torch.ones(2, 1, 1, 5, 5)
    block = torch.zeros_like(fire)

    actual = model(pack_reliability_input(features, fire, block))

    torch.testing.assert_close(actual, _base_logits(base, features))
    assert model.adapter_parameter_count == 2_625


def test_block_scope_applies_global_residual_and_updates_both_paths() -> None:
    base = _Base()
    model = CounterfactualReliabilityAdapter(base, adapter_scope="block")
    features = torch.randn(2, 1, 40, 5, 5)
    fire = torch.zeros(2, 1, 1, 5, 5)
    block = torch.zeros_like(fire)
    block[:, :, :, :2, :2] = 1.0

    model(pack_reliability_input(features, fire, block)).sum().backward()

    assert model.adapter[-1].weight.grad is not None
    assert torch.count_nonzero(model.adapter[-1].weight.grad) > 0
    assert base.model.segmentation_head.weight.grad is not None


def test_ffca_checkpoint_requires_block_scope_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D8-FFCA",
        "matched_pair": "D8",
        "base_control": "D1-ERM",
        "closest_ablations": ["D5-CIWC", "D7-CRA", "D4-TOKEN", "P04-P06"],
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "lambda_ciwc": 0.1,
        "adapter_scope": "block",
        "adapter_parameter_count": 2_625,
        "reliability_maps": ["fire-drop", "block-drop"],
        "hyper_parameters": {"n_channels": 40},
        "base_state_dict": {"weight": torch.tensor(1.0)},
        "adapter_state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_ffca_checkpoint(payload) == "D8-FFCA"

    payload["adapter_scope"] = "all"
    with pytest.raises(ValueError, match="FFCA checkpoint"):
        validate_ffca_checkpoint(payload)


def test_ffca_runner_has_valid_shell_contract(tmp_path: Path) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_failure_factorized_adapter_on_nibi.sh"
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


def test_generic_heldout_runner_accepts_ffca_before_year_validation(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_evaluation_on_nibi.sh"
    )
    checkpoint = tmp_path / "ffca.pt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [str(runner), "ffca", str(checkpoint), "2021", "D8-FFCA"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "year must be 2022 or 2023" in result.stderr

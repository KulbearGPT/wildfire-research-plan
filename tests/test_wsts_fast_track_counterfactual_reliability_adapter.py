from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from reproductions.wsts_fast_track.counterfactual_impact_consistency import (
    counterfactual_impact_kl_from_logits,
)
from reproductions.wsts_fast_track.counterfactual_reliability_adapter import (
    CounterfactualReliabilityAdapter,
    TwoRegimeReliabilityDataset,
    pack_reliability_input,
)
from reproductions.wsts_fast_track.evaluate_counterfactual_reliability_adapter import (
    validate_cra_checkpoint,
)
from reproductions.wsts_fast_track.predictive_consistency import (
    CleanCorruptPairDataset,
)
from reproductions.wsts_fast_track.train_counterfactual_reliability_adapter import (
    cra_training_objective,
)


class _FakeBaseDataset:
    is_train = True
    remove_duplicate_features = False
    features_to_keep = None
    return_doy = False
    n_leading_observations = 1

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return torch.ones((1, 40, 8, 8)), torch.zeros((8, 8))


def test_paired_reliability_maps_preserve_the_original_random_stream() -> None:
    ordinary = CleanCorruptPairDataset(
        _FakeBaseDataset(),
        fire_probability=0.3,
        block_probability=0.3,
        active_fire_missing_value=-2.0,
    )
    reliability = CleanCorruptPairDataset(
        _FakeBaseDataset(),
        fire_probability=0.3,
        block_probability=0.3,
        active_fire_missing_value=-2.0,
        return_reliability=True,
    )

    np.random.seed(17)
    clean_a, corrupt_a, target_a = ordinary[0]
    np.random.seed(17)
    clean_b, corrupt_b, target_b, fire_missing, block_missing = reliability[0]

    torch.testing.assert_close(clean_a, clean_b)
    torch.testing.assert_close(corrupt_a, corrupt_b)
    torch.testing.assert_close(target_a, target_b)
    assert fire_missing.shape == (1, 1, 8, 8)
    assert block_missing.shape == (1, 1, 8, 8)


def test_fire_and_block_reliability_are_distinct() -> None:
    dataset = CleanCorruptPairDataset(
        _FakeBaseDataset(),
        fire_probability=1.0,
        block_probability=0.0,
        active_fire_missing_value=-2.0,
        return_reliability=True,
    )

    _clean, _corrupt, _target, fire_missing, block_missing = dataset[0]

    assert torch.all(fire_missing == 1)
    assert torch.count_nonzero(block_missing) == 0


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


def test_zero_initialized_adapter_is_identity_and_clean_bypasses() -> None:
    base = _Base()
    model = CounterfactualReliabilityAdapter(base)
    features = torch.randn(2, 1, 40, 5, 5)
    zeros = torch.zeros(2, 1, 1, 5, 5)
    packed = pack_reliability_input(features, zeros, zeros)

    actual = model(packed)
    encoded = base.model.encoder(features[:, 0])
    expected = base.model.segmentation_head(base.model.decoder(*encoded))

    torch.testing.assert_close(actual, expected)
    assert model.adapter_parameter_count == 2_769


def test_missing_path_updates_adapter_and_joint_base() -> None:
    base = _Base()
    model = CounterfactualReliabilityAdapter(base)
    features = torch.randn(2, 1, 40, 5, 5)
    fire = torch.ones(2, 1, 1, 5, 5)
    block = torch.zeros_like(fire)
    packed = pack_reliability_input(features, fire, block)

    model(packed).sum().backward()

    assert model.adapter[-1].weight.grad is not None
    assert torch.count_nonzero(model.adapter[-1].weight.grad) > 0
    assert base.model.segmentation_head.weight.grad is not None


def test_cra_objective_is_d5_loss_on_adapted_logits() -> None:
    clean = torch.tensor([[[2.0, 0.0]]])
    adapted = torch.tensor([[[0.0, 2.0]]])
    target = torch.zeros_like(clean)
    objective, parts = cra_training_objective(_Base(), clean, adapted, target)
    supervised = 2.0
    ciwc = float(counterfactual_impact_kl_from_logits(clean, adapted))

    assert parts == pytest.approx({"supervised": supervised, "ciwc": ciwc})
    assert float(objective) == pytest.approx(supervised + 0.1 * ciwc)


class _Controlled:
    def __init__(self, block: torch.Tensor) -> None:
        self.block = block

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        features = torch.zeros((1, 40, 4, 4))
        routed = torch.cat((features, self.block), dim=1)
        return routed, torch.zeros((4, 4))


def test_evaluation_wrapper_encodes_failure_type_and_support() -> None:
    block = torch.zeros((1, 1, 4, 4))
    block[:, :, :2, :2] = 1

    fire_x, _ = TwoRegimeReliabilityDataset(
        _Controlled(torch.zeros_like(block)), "M01"
    )[0]
    block_x, _ = TwoRegimeReliabilityDataset(_Controlled(block), "M06")[0]

    assert fire_x.shape == (1, 42, 4, 4)
    assert torch.all(fire_x[:, 40] == 1)
    assert torch.count_nonzero(fire_x[:, 41]) == 0
    assert torch.count_nonzero(block_x[:, 40]) == 0
    torch.testing.assert_close(block_x[:, 41:42], block)


def test_cra_checkpoint_requires_frozen_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D7-CRA",
        "matched_pair": "D7",
        "base_control": "D1-ERM",
        "closest_ablations": ["D5-CIWC", "D4-TOKEN", "P04-P06"],
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "lambda_ciwc": 0.1,
        "adapter_parameter_count": 2_769,
        "reliability_maps": ["fire-drop", "block-drop"],
        "hyper_parameters": {"n_channels": 40},
        "base_state_dict": {"weight": torch.tensor(1.0)},
        "adapter_state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_cra_checkpoint(payload) == "D7-CRA"

    payload["adapter_parameter_count"] = 145
    with pytest.raises(ValueError, match="CRA checkpoint"):
        validate_cra_checkpoint(payload)


def test_cra_runner_has_valid_shell_contract(tmp_path: Path) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_counterfactual_reliability_adapter_on_nibi.sh"
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


def test_generic_heldout_runner_accepts_cra_kind_before_year_validation(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_evaluation_on_nibi.sh"
    )
    checkpoint = tmp_path / "cra.pt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [str(runner), "cra", str(checkpoint), "2021", "D7-CRA"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "year must be 2022 or 2023" in result.stderr

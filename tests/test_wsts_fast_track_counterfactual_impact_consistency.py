from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from reproductions.wsts_fast_track.counterfactual_impact_consistency import (
    counterfactual_impact_kl_from_logits,
)
from reproductions.wsts_fast_track.predictive_consistency import (
    bernoulli_kl_from_logits,
)
from reproductions.wsts_fast_track.evaluate_counterfactual_impact_consistency import (
    validate_ciwc_checkpoint,
)
from reproductions.wsts_fast_track.train_counterfactual_impact_consistency import (
    ciwc_training_objective,
)


def test_ciwc_focuses_disagreement_and_stops_clean_gradient() -> None:
    clean = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    corrupt = torch.tensor([[[-4.0, 0.0], [0.0, 0.0]]], requires_grad=True)

    focused = counterfactual_impact_kl_from_logits(clean, corrupt)
    global_kl = bernoulli_kl_from_logits(clean, corrupt)

    assert float(focused.detach()) > float(global_kl.detach())
    focused.backward()
    assert clean.grad is None
    assert corrupt.grad is not None
    assert torch.count_nonzero(corrupt.grad) == 1


def test_ciwc_zero_impact_is_exactly_zero() -> None:
    logits = torch.tensor([[[0.0, 1.0]]])

    loss = counterfactual_impact_kl_from_logits(logits, logits)

    assert float(loss) == 0.0


def test_ciwc_objective_adds_only_the_fixed_impact_term() -> None:
    class SquaredLoss:
        @staticmethod
        def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (logits - target).square().mean()

    clean = torch.tensor([[[1.0, 0.0]]])
    corrupt = torch.tensor([[[-1.0, 0.0]]])
    target = torch.zeros_like(clean)
    objective, parts = ciwc_training_objective(
        SquaredLoss(), clean, corrupt, target
    )
    expected_supervised = 0.5
    expected_consistency = float(
        counterfactual_impact_kl_from_logits(clean, corrupt)
    )

    assert parts["supervised"] == pytest.approx(expected_supervised)
    assert parts["consistency"] == pytest.approx(expected_consistency)
    assert float(objective) == pytest.approx(
        expected_supervised + 0.1 * expected_consistency
    )


def test_ciwc_checkpoint_requires_the_fixed_method_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D5-CIWC",
        "matched_pair": "D5",
        "base_control": "D1-ERM",
        "closest_ablation": "D1-KL",
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "lambda_ciwc": 0.1,
        "impact_weighting": "per-sample-absolute-probability-change",
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_ciwc_checkpoint(payload) == "D5-CIWC"

    payload["lambda_ciwc"] = 0.3
    with pytest.raises(ValueError, match="CIWC checkpoint"):
        validate_ciwc_checkpoint(payload)


def test_ciwc_runner_rejects_extra_arguments_before_cluster_setup(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_counterfactual_impact_consistency_on_nibi.sh"
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


def test_generic_heldout_runner_accepts_ciwc_kind_before_year_validation(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_evaluation_on_nibi.sh"
    )
    checkpoint = tmp_path / "ciwc.pt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [str(runner), "ciwc", str(checkpoint), "2021", "D5-CIWC"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "year must be 2022 or 2023" in result.stderr

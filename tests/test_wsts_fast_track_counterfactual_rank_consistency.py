from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from reproductions.wsts_fast_track.counterfactual_impact_consistency import (
    counterfactual_impact_kl_from_logits,
)
from reproductions.wsts_fast_track.counterfactual_rank_consistency import (
    normalized_spatial_js_from_logits,
)
from reproductions.wsts_fast_track.evaluate_counterfactual_rank_consistency import (
    validate_circ_checkpoint,
)
from reproductions.wsts_fast_track.train_counterfactual_rank_consistency import (
    circ_training_objective,
)


def test_spatial_rank_loss_is_zero_for_identical_ordering() -> None:
    logits = torch.tensor([[[2.0, 0.0], [-1.0, 1.0]]])

    loss = normalized_spatial_js_from_logits(logits, logits)

    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_spatial_rank_loss_detects_reversal_and_stops_clean_gradient() -> None:
    clean = torch.tensor([[[4.0, 0.0]]], requires_grad=True)
    corrupt = torch.tensor([[[0.0, 4.0]]], requires_grad=True)

    loss = normalized_spatial_js_from_logits(clean, corrupt)

    assert 0.0 < float(loss.detach()) <= 1.0
    loss.backward()
    assert clean.grad is None
    assert corrupt.grad is not None
    assert torch.count_nonzero(corrupt.grad) == corrupt.numel()


def test_spatial_rank_loss_ignores_uniform_logit_offset() -> None:
    clean = torch.tensor([[[2.0, 0.0, -1.0]]])
    corrupt = clean + 9.0

    loss = normalized_spatial_js_from_logits(clean, corrupt)

    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_circ_objective_uses_fixed_ciwc_and_rank_weights() -> None:
    class SquaredLoss:
        @staticmethod
        def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (logits - target).square().mean()

    clean = torch.tensor([[[2.0, 0.0]]])
    corrupt = torch.tensor([[[0.0, 2.0]]])
    target = torch.zeros_like(clean)
    objective, parts = circ_training_objective(
        SquaredLoss(), clean, corrupt, target
    )
    expected_supervised = 2.0
    expected_ciwc = float(
        counterfactual_impact_kl_from_logits(clean, corrupt)
    )
    expected_rank = float(normalized_spatial_js_from_logits(clean, corrupt))

    assert parts == pytest.approx(
        {
            "supervised": expected_supervised,
            "ciwc": expected_ciwc,
            "rank": expected_rank,
        }
    )
    assert float(objective) == pytest.approx(
        expected_supervised + 0.1 * expected_ciwc + 0.01 * expected_rank
    )


def test_circ_checkpoint_requires_frozen_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D6-CIRC",
        "matched_pair": "D6",
        "base_control": "D1-ERM",
        "closest_ablations": ["D1-KL", "D5-CIWC"],
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "lambda_ciwc": 0.1,
        "lambda_rank": 0.01,
        "rank_temperature": 1.0,
        "rank_formulation": "normalized-spatial-jensen-shannon",
        "impact_weighting": "per-sample-absolute-probability-change",
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_circ_checkpoint(payload) == "D6-CIRC"

    payload["lambda_rank"] = 0.1
    with pytest.raises(ValueError, match="CIRC checkpoint"):
        validate_circ_checkpoint(payload)


def test_circ_runner_rejects_extra_arguments_before_cluster_setup(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_counterfactual_rank_consistency_on_nibi.sh"
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


def test_generic_heldout_runner_accepts_circ_kind_before_year_validation(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_evaluation_on_nibi.sh"
    )
    checkpoint = tmp_path / "circ.pt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [str(runner), "circ", str(checkpoint), "2021", "D6-CIRC"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "year must be 2022 or 2023" in result.stderr

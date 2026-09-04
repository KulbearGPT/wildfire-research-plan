from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from reproductions.wsts_fast_track.counterfactual_error_pair_ranking import (
    counterfactual_error_pair_ranking_from_logits,
)
from reproductions.wsts_fast_track.counterfactual_impact_consistency import (
    counterfactual_impact_kl_from_logits,
)
from reproductions.wsts_fast_track.evaluate_counterfactual_error_pair_ranking import (
    validate_cepr_checkpoint,
)
from reproductions.wsts_fast_track.train_counterfactual_error_pair_ranking import (
    cepr_training_objective,
)


def test_cepr_selects_counterfactual_errors_and_only_updates_corrupt_pair() -> None:
    clean = torch.tensor([[2.0, 0.0, -2.0, 0.0]], requires_grad=True)
    corrupt = torch.tensor([[0.0, -1.0, 0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[1, 1, 0, 0]])
    corrupted_sample = torch.tensor([True])

    loss = counterfactual_error_pair_ranking_from_logits(
        clean, corrupt, target, corrupted_sample, top_k=1
    )

    assert float(loss.detach()) == pytest.approx(1.0)
    loss.backward()
    assert clean.grad is None
    assert corrupt.grad is not None
    assert torch.count_nonzero(corrupt.grad) == 2
    assert corrupt.grad[0, 0] < 0
    assert corrupt.grad[0, 2] > 0


def test_cepr_is_differentiable_zero_without_valid_pairs() -> None:
    clean = torch.tensor([[1.0, -1.0]], requires_grad=True)
    corrupt = torch.tensor([[0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1, 0]])

    loss = counterfactual_error_pair_ranking_from_logits(
        clean, corrupt, target, torch.tensor([False]), top_k=64
    )

    assert float(loss.detach()) == 0.0
    loss.backward()
    assert clean.grad is None
    assert torch.count_nonzero(corrupt.grad) == 0


def test_cepr_objective_adds_fixed_impact_and_pair_terms() -> None:
    class SquaredLoss:
        @staticmethod
        def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (logits - target).square().mean()

    clean = torch.tensor([[2.0, 0.0, -2.0, 0.0]])
    corrupt = torch.tensor([[0.0, -1.0, 0.0, 1.0]])
    target = torch.tensor([[1, 1, 0, 0]])
    corrupted = torch.tensor([True])
    objective, parts = cepr_training_objective(
        SquaredLoss(), clean, corrupt, target, corrupted
    )
    supervised = 1.5
    ciwc = float(counterfactual_impact_kl_from_logits(clean, corrupt))
    pair = float(
        counterfactual_error_pair_ranking_from_logits(
            clean, corrupt, target, corrupted, top_k=64
        )
    )

    assert parts == pytest.approx(
        {"supervised": supervised, "ciwc": ciwc, "pair": pair}
    )
    assert float(objective) == pytest.approx(
        supervised + 0.1 * ciwc + 0.01 * pair
    )


def test_cepr_checkpoint_requires_frozen_contract() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D9-CEPR",
        "matched_pair": "D9",
        "base_control": "D1-ERM",
        "closest_ablations": ["D5-CIWC", "D6-CIRC"],
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "lambda_ciwc": 0.1,
        "lambda_pair": 0.01,
        "pair_top_k": 64,
        "pair_formulation": "counterfactual-error-softplus",
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }

    assert validate_cepr_checkpoint(payload) == "D9-CEPR"

    payload["pair_top_k"] = 128
    with pytest.raises(ValueError, match="CEPR checkpoint"):
        validate_cepr_checkpoint(payload)


def test_cepr_runner_has_valid_shell_contract(tmp_path: Path) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_counterfactual_error_pair_ranking_on_nibi.sh"
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


def test_generic_heldout_runner_accepts_cepr_before_year_validation(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_reliability_evaluation_on_nibi.sh"
    )
    checkpoint = tmp_path / "cepr.pt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [str(runner), "cepr", str(checkpoint), "2021", "D9-CEPR"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "year must be 2022 or 2023" in result.stderr

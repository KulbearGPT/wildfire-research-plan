from __future__ import annotations

import pytest
import torch

from reproductions.wsts_fast_track.counterfactual_impact_consistency import (
    counterfactual_impact_kl_from_logits,
)
from reproductions.wsts_fast_track.predictive_consistency import (
    bernoulli_kl_from_logits,
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

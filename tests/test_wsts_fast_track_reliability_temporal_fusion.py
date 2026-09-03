from __future__ import annotations

import torch

from reproductions.wsts_fast_track.reliability_temporal_fusion import (
    masked_temporal_softmax,
)


def test_all_valid_temporal_softmax_matches_ordinary_softmax() -> None:
    logits = torch.tensor(
        [[[[[0.0]], [[1.0]], [[2.0]]]], [[[[2.0]], [[1.0]], [[0.0]]]]]
    )
    valid = torch.ones(1, 3, 1, 1, dtype=torch.bool)

    actual = masked_temporal_softmax(logits, valid)

    torch.testing.assert_close(actual, torch.softmax(logits, dim=2))


def test_invalid_times_receive_zero_mass_and_remaining_mass_is_normalized() -> None:
    logits = torch.tensor([[[[[0.0]], [[100.0]], [[2.0]]]]])
    valid = torch.tensor([[[[True]], [[False]], [[True]]]])

    result = masked_temporal_softmax(logits, valid)

    assert result[0, 0, 1, 0, 0] == 0.0
    torch.testing.assert_close(result.sum(dim=2), torch.ones(1, 1, 1, 1))
    assert result[0, 0, 2, 0, 0] > result[0, 0, 0, 0, 0]


def test_single_or_no_valid_time_remains_finite() -> None:
    logits = torch.randn(2, 1, 3, 2, 2)
    one_valid = torch.zeros(1, 3, 2, 2, dtype=torch.bool)
    one_valid[:, 1] = True
    result = masked_temporal_softmax(logits, one_valid)
    assert torch.isfinite(result).all()
    assert torch.all(result[:, :, 1] == 1.0)

    no_valid = torch.zeros_like(one_valid)
    empty = masked_temporal_softmax(logits, no_valid)
    assert torch.isfinite(empty).all()
    assert torch.count_nonzero(empty) == 0

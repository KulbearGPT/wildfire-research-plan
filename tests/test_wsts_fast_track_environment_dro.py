from __future__ import annotations

import torch


class _TinyYearDataset:
    datapoints_per_fire = {
        2016: {"a": 2},
        2017: {"b": 4},
        2018: {"c": 1},
        2019: {"d": 5},
        2020: {"e": 3},
    }

    def __len__(self) -> int:
        return 15


def test_environment_groups_and_balanced_year_sampling_are_exact() -> None:
    from reproductions.wsts_fast_track.environment_dro import (
        balanced_year_sampling_weights,
        environment_group,
        resolve_dataset_index,
    )

    assert environment_group(2016, 0) == 0
    assert environment_group(2018, 1) == 7
    assert environment_group(2020, 2) == 14

    weights = balanced_year_sampling_weights(_TinyYearDataset())
    offsets = (0, 2, 6, 7, 12, 15)
    masses = [weights[start:end].sum() for start, end in zip(offsets, offsets[1:])]
    torch.testing.assert_close(torch.stack(masses), torch.ones(5, dtype=torch.double))
    assert resolve_dataset_index(_TinyYearDataset(), 0) == (2016, "a", 0)
    assert resolve_dataset_index(_TinyYearDataset(), 2) == (2017, "b", 0)
    assert resolve_dataset_index(_TinyYearDataset(), 14) == (2020, "e", 2)
    assert resolve_dataset_index(_TinyYearDataset(), -1) == (2020, "e", 2)


def test_group_dro_upweights_harder_group_and_preserves_gradients() -> None:
    from reproductions.wsts_fast_track.environment_dro import group_dro_objective

    losses = torch.tensor([1.0, 1.0, 4.0, 4.0], requires_grad=True)
    group_ids = torch.tensor([0, 0, 1, 1])
    log_weights = torch.zeros(15)

    objective, group_losses, weights = group_dro_objective(
        losses, group_ids, log_weights, step_size=0.1
    )

    assert group_losses == {0: 1.0, 1: 4.0}
    assert float(weights[1]) > float(weights[0])
    assert 2.5 < float(objective.detach()) < 4.0
    objective.backward()
    assert torch.all(losses.grad > 0)


def test_erm_objective_is_plain_per_sample_mean() -> None:
    from reproductions.wsts_fast_track.environment_dro import erm_objective

    losses = torch.tensor([1.0, 1.0, 4.0, 4.0], requires_grad=True)

    objective = erm_objective(losses)

    assert float(objective.detach()) == 2.5
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.full((4,), 0.25))


def test_corrected_focal_alpha_weights_the_positive_class() -> None:
    from reproductions.wsts_fast_track.environment_dro import corrected_focal_alpha

    assert corrected_focal_alpha(0.9) == 0.9
    raw_weight = 761.0785701324623
    assert corrected_focal_alpha(raw_weight) == raw_weight / (1.0 + raw_weight)

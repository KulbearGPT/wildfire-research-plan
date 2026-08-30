from __future__ import annotations

import numpy as np
import pytest
import torch


def test_balanced_cohort_selects_equal_target_states_per_year_and_unique_events() -> None:
    from reproductions.wsts_fast_track.censored_training import (
        select_balanced_candidates,
    )

    candidates = []
    for year in (2016, 2017):
        for suffix, positive in (
            ("p0", 4),
            ("p1", 3),
            ("p2", 2),
            ("n0", 0),
            ("n1", 0),
            ("n2", 0),
        ):
            candidates.append(
                {
                    "year": year,
                    "event": f"fire_{year}_{suffix}",
                    "target_day": f"{year}-07-01",
                    "target_positive_pixels": positive,
                }
            )

    selected = select_balanced_candidates(
        candidates, years=(2016, 2017), per_year=4
    )

    assert len(selected) == 8
    assert len({record["event"] for record in selected}) == 8
    for year in (2016, 2017):
        year_records = [record for record in selected if record["year"] == year]
        assert len(year_records) == 4
        assert sum(record["target_positive_pixels"] > 0 for record in year_records) == 2


def test_aligned_preprocessing_keeps_target_reliability_geometry_together() -> None:
    from reproductions.wsts_fast_track.censored_training import preprocess_aligned

    class Base:
        is_train = True
        is_pad = False
        crop_side_length = 2
        indices_of_degree_features: list[int] = []
        one_hot_matrix = torch.eye(17)
        means = torch.zeros((1, 23, 1, 1))
        stds = torch.ones((1, 23, 1, 1))

        def standardize_features(self, values: torch.Tensor) -> torch.Tensor:
            return (values - self.means) / self.stds

    features = np.zeros((1, 23, 4, 4), dtype=np.float32)
    features[:, 16] = 1.0
    target = np.zeros((4, 4), dtype=np.float32)
    target[0, 0] = 1.0
    reliability = target.astype(np.uint8)
    np.random.seed(7)

    processed, processed_target, processed_reliability = preprocess_aligned(
        Base(), features, target, reliability
    )

    assert processed.shape == (1, 40, 2, 2)
    assert processed_target.shape == processed_reliability.shape == (2, 2)
    assert torch.equal(processed_target.bool(), processed_reliability.bool())
    assert int(processed_target.sum()) == 1


def test_censored_cohort_loss_excludes_only_unknown_zero_pixels() -> None:
    from reproductions.wsts_fast_track.censored_training import cohort_loss

    probabilities = torch.tensor([[0.9, 0.2, 0.8]], dtype=torch.float64)
    logits = torch.logit(probabilities)
    target = torch.tensor([[1, 0, 0]])
    reliability = torch.tensor([[0, 1, 0]], dtype=torch.uint8)

    censored = cohort_loss(logits, target, reliability, censored=True)
    standard = cohort_loss(logits, target, reliability, censored=False)

    assert float(censored) == pytest.approx(0.00498967, abs=1e-7)
    assert float(standard) > float(censored)

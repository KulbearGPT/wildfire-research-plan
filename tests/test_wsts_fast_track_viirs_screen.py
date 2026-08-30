from __future__ import annotations

import numpy as np
import torch


def test_first_event_indices_select_the_common_t1_t5_population() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_reliability import (
        first_event_indices,
    )

    class Base:
        rows = (
            (2021, "fire_a", 0),
            (2021, "fire_a", 1),
            (2021, "fire_b", 0),
            (2021, "fire_b", 1),
        )

        def __len__(self) -> int:
            return len(self.rows)

        def find_image_index_from_dataset_index(self, index: int):
            return self.rows[index]

    assert first_event_indices(Base(), ("fire_b", "fire_a")) == {
        "fire_a": 0,
        "fire_b": 2,
    }


def test_natural_reliability_replaces_only_the_packed_validity_crop() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_reliability import (
        replace_t1_reliability,
    )

    packed = torch.randn((1, 41, 4, 4))
    packed[:, 40] = 1.0
    original_features = packed[:, :40].clone()
    full = np.zeros((6, 6), dtype=np.uint8)
    full[1:5, 1:5] = np.eye(4, dtype=np.uint8)

    replaced = replace_t1_reliability(packed, full)

    assert torch.equal(replaced[:, :40], original_features)
    assert torch.equal(replaced[0, 40], torch.eye(4))

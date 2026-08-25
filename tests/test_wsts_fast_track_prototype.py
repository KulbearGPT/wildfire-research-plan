from __future__ import annotations

import numpy as np
import pytest

from reproductions.wsts_fast_track import prototype


def test_fire_dropout_changes_only_training_active_fire_history() -> None:
    x = np.ones((2, 23, 3, 3), dtype=np.float32)
    y = np.ones((3, 3), dtype=np.float32)

    train_x, train_y = prototype.apply_training_fire_dropout(
        (x, y),
        is_train=True,
        probability=0.3,
        random_value=0.2,
    )
    validation_x, validation_y = prototype.apply_training_fire_dropout(
        (x, y),
        is_train=False,
        probability=0.3,
        random_value=0.2,
    )

    assert np.all(train_x[:, 22] == 0.0)
    assert np.all(train_x[:, :22] == 1.0)
    assert np.array_equal(train_y, y)
    assert np.array_equal(validation_x, x)
    assert np.array_equal(validation_y, y)
    assert np.all(x == 1.0)


def test_fire_dropout_reports_matching_active_fire_validity() -> None:
    x = np.ones((1, 23, 2, 2), dtype=np.float32)
    y = np.ones((2, 2), dtype=np.float32)

    dropped, dropped_validity = prototype.apply_training_fire_dropout_with_validity(
        (x, y),
        is_train=True,
        probability=0.3,
        random_value=0.2,
    )
    clean, clean_validity = prototype.apply_training_fire_dropout_with_validity(
        (x, y),
        is_train=True,
        probability=0.3,
        random_value=0.4,
    )

    assert np.all(dropped[0][:, 22] == 0.0)
    assert dropped_validity == 0.0
    assert np.array_equal(clean[0], x)
    assert clean_validity == 1.0

    processed = np.ones((1, 40, 2, 2), dtype=np.float32)
    with_validity = prototype.append_fire_validity_channel(
        processed, dropped_validity
    )
    assert with_validity.shape == (1, 41, 2, 2)
    assert np.all(with_validity[:, :40] == 1.0)
    assert np.all(with_validity[:, 40] == 0.0)


def test_block_dropout_masks_only_training_dynamic_inputs() -> None:
    x = np.ones((1, 23, 4, 4), dtype=np.float32)
    y = np.ones((4, 4), dtype=np.float32)

    (train_x, train_y), mask = prototype.apply_training_fire_and_block_dropout(
        (x, y),
        is_train=True,
        fire_probability=0.3,
        block_probability=0.3,
        fire_random_value=0.8,
        block_random_value=0.2,
        fraction_random_value=0.8,
        key_digest="00" * 32,
    )
    (validation_x, validation_y), validation_mask = (
        prototype.apply_training_fire_and_block_dropout(
            (x, y),
            is_train=False,
            fire_probability=0.3,
            block_probability=0.3,
            fire_random_value=0.2,
            block_random_value=0.2,
            fraction_random_value=0.8,
            key_digest="00" * 32,
        )
    )

    assert mask is not None
    assert int(mask.sum()) == 8
    dynamic_non_fire = tuple(range(12)) + (15,) + tuple(range(17, 22))
    assert np.isnan(train_x[:, dynamic_non_fire][:, :, mask]).all()
    assert np.count_nonzero(train_x[:, 22][:, mask]) == 0
    np.testing.assert_array_equal(train_x[:, (12, 13, 14, 16)], 1.0)
    np.testing.assert_array_equal(train_x[:, :, ~mask], 1.0)
    np.testing.assert_array_equal(train_y, y)
    np.testing.assert_array_equal(validation_x, x)
    np.testing.assert_array_equal(validation_y, y)
    assert validation_mask is None


def test_prototype_heldout_year_requires_explicit_authorization() -> None:
    from reproductions.wsts_fast_track import evaluate_prototype

    assert evaluate_prototype.evaluation_boundary(
        2021, heldout_authorized=False
    ) == ("prototype-validation", False)

    with pytest.raises(ValueError, match="authorization"):
        evaluate_prototype.evaluation_boundary(2022, heldout_authorized=False)

    assert evaluate_prototype.evaluation_boundary(
        2023, heldout_authorized=True
    ) == ("prototype-formal", True)

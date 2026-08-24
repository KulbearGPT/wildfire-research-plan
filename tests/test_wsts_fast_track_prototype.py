from __future__ import annotations

import numpy as np

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

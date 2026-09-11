import numpy as np

from reproductions.cross_history.reliability_masks import footprint_mask


def test_footprint_mask_is_deterministic_with_exact_target_area() -> None:
    reliability = np.linspace(0.0, 1.0, 7 * 9, dtype=np.float32).reshape(7, 9)

    first = footprint_mask(reliability, 11, 13, 0.25, key_digest="ab" * 32)
    repeated = footprint_mask(reliability, 11, 13, 0.25, key_digest="ab" * 32)

    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.bool_
    assert first.shape == (11, 13)
    assert int(first.sum()) == round(11 * 13 * 0.25)

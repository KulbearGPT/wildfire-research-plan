from __future__ import annotations

import numpy as np
import pytest

from reproductions.wsts_fast_track import matrix, missingness


def _raw_sample(time: int = 5, height: int = 8, width: int = 10) -> np.ndarray:
    values = np.arange(time * 23 * height * width, dtype=np.float32)
    return values.reshape(time, 23, height, width) + 1


def _apply(values: np.ndarray, scenario_id: str, **kwargs: object):
    return missingness.apply_corruption(
        values,
        scenario_id,
        event_relative_path="2021/fire-a.hdf5",
        target_date="2021-08-14",
        **kwargs,
    )


def test_m00_is_an_equal_non_aliasing_copy() -> None:
    source = _raw_sample()

    result = _apply(source, "M00")

    np.testing.assert_array_equal(result.values, source)
    assert not np.shares_memory(result.values, source)
    assert result.spatial_mask is None
    assert result.schema_version == 1


def test_m01_removes_only_raw_active_fire_history() -> None:
    source = _raw_sample()

    result = _apply(source, "M01")

    assert np.count_nonzero(result.values[:, 22]) == 0
    np.testing.assert_array_equal(result.values[:, :22], source[:, :22])
    np.testing.assert_array_equal(source, _raw_sample())


def test_m02_replaces_fire_history_with_exact_preceding_sequence() -> None:
    source = _raw_sample()
    stale = np.arange(5 * 8 * 10, dtype=np.float32).reshape(5, 8, 10)

    result = _apply(source, "M02", stale_active_fire=stale)

    np.testing.assert_array_equal(result.values[:, 22], stale)
    np.testing.assert_array_equal(result.values[:, :22], source[:, :22])


@pytest.mark.parametrize(
    ("stale", "message"),
    [
        (None, "stale_active_fire"),
        (np.zeros((4, 8, 10), dtype=np.float32), "shape"),
        (np.full((5, 8, 10), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_m02_requires_matching_finite_preceding_history(
    stale: np.ndarray | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _apply(_raw_sample(), "M02", stale_active_fire=stale)


@pytest.mark.parametrize(
    ("scenario_id", "missing_channels"),
    [
        ("M03", tuple(range(5, 12))),
        ("M04", tuple(range(17, 22))),
        ("M05", tuple(range(5, 12)) + tuple(range(17, 22))),
    ],
)
def test_modality_scenarios_mean_impute_only_declared_raw_channels(
    scenario_id: str, missing_channels: tuple[int, ...]
) -> None:
    source = _raw_sample()

    result = _apply(source, scenario_id)

    assert np.isnan(result.values[:, missing_channels]).all()
    retained = tuple(channel for channel in range(23) if channel not in missing_channels)
    np.testing.assert_array_equal(result.values[:, retained], source[:, retained])


@pytest.mark.parametrize(("scenario_id", "fraction"), [("M06", 0.25), ("M07", 0.50)])
def test_spatial_scenarios_mask_exact_area_and_all_dynamic_features(
    scenario_id: str, fraction: float
) -> None:
    source = _raw_sample(height=9, width=11)

    result = _apply(source, scenario_id)

    assert result.spatial_mask is not None
    expected_pixels = round(9 * 11 * fraction)
    assert int(result.spatial_mask.sum()) == expected_pixels
    dynamic_non_fire = tuple(range(12)) + (15,) + tuple(range(17, 22))
    assert np.isnan(result.values[:, dynamic_non_fire][:, :, result.spatial_mask]).all()
    assert np.count_nonzero(result.values[:, 22][:, result.spatial_mask]) == 0
    static = (12, 13, 14, 16)
    np.testing.assert_array_equal(result.values[:, static], source[:, static])
    outside = ~result.spatial_mask
    np.testing.assert_array_equal(result.values[:, :, outside], source[:, :, outside])


def test_spatial_mask_is_stable_and_key_sensitive() -> None:
    source = _raw_sample(height=20, width=20)
    first = _apply(source, "M06").spatial_mask
    repeated = _apply(source, "M06").spatial_mask
    other_event = missingness.apply_corruption(
        source,
        "M06",
        event_relative_path="2021/fire-b.hdf5",
        target_date="2021-08-14",
    ).spatial_mask

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other_event)


def test_hash_identity_rejects_paths_or_dates_outside_frozen_schema() -> None:
    source = _raw_sample()
    with pytest.raises(ValueError, match="relative"):
        missingness.apply_corruption(
            source,
            "M06",
            event_relative_path="/2021/fire.hdf5",
            target_date="2021-08-14",
        )
    with pytest.raises(ValueError, match="target_date"):
        missingness.apply_corruption(
            source,
            "M06",
            event_relative_path="2021/fire.hdf5",
            target_date="not-a-date",
        )


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((23, 8, 8), dtype=np.float32),
        np.zeros((1, 22, 8, 8), dtype=np.float32),
        np.zeros((1, 23, 0, 8), dtype=np.float32),
    ],
)
def test_corruption_rejects_noncanonical_raw_shape(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="T, 23, H, W"):
        _apply(values, "M00")


def test_matrix_marks_every_scenario_implemented() -> None:
    assert all(
        scenario.implementation_state == "implemented"
        for scenario in matrix.CORRUPTIONS.values()
    )

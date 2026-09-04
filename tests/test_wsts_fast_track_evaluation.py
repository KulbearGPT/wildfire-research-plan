from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from reproductions.wsts_fast_track import contract, evaluation


class FakeDataset:
    def __init__(self, path: Path, *, history: int, year: int = 2021) -> None:
        self.data_dir = str(path.parents[1])
        self.included_fire_years = [year]
        self.stats_years = [2016, 2017, 2018, 2019, 2020]
        self.n_leading_observations = history
        self.n_leading_observations_test_adjustment = 6
        self.skip_initial_samples = 6 - history
        self.load_from_hdf5 = True
        self.is_train = False
        self.crop_side_length = 8
        self.remove_duplicate_features = False
        self.features_to_keep = None
        self.return_doy = False
        self.imgs_per_fire = {year: {path.stem: [str(path)]}}
        self.datapoints_per_fire = {year: {path.stem: 10 - 6}}
        self.length = 4
        self.preprocessed_x: np.ndarray | None = None

    def __len__(self) -> int:
        return self.length

    def find_image_index_from_dataset_index(self, index: int):
        year = self.included_fire_years[0]
        return year, next(iter(self.imgs_per_fire[year])), index

    def preprocess_and_augment(self, x: np.ndarray, y: np.ndarray):
        assert self.is_train is False
        self.preprocessed_x = x.copy()
        return torch.from_numpy(x.copy()), torch.from_numpy((y > 0).astype(np.int64))


def _event(
    tmp_path: Path, *, year: int = 2021, height: int = 8, width: int = 8
) -> Path:
    path = tmp_path / "data" / str(year) / "fire-a.hdf5"
    path.parent.mkdir(parents=True)
    values = np.zeros((10, 23, height, width), dtype=np.float32)
    for day_index in range(10):
        for feature in range(23):
            values[day_index, feature] = day_index * 100 + feature
    start = date(year, 8, 1)
    dates = [(start + timedelta(days=index)).isoformat() for index in range(10)]
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=values)
        dataset.attrs["img_dates"] = dates
    return path


@pytest.mark.parametrize("history", [1, 5])
def test_controlled_dataset_uses_same_target_population_and_dates(
    tmp_path: Path, history: int
) -> None:
    base = FakeDataset(_event(tmp_path), history=history)
    dataset = evaluation.ControlledMissingnessDataset(base, "M00")

    descriptor = dataset.describe(0)

    assert len(dataset) == 4
    assert descriptor.event_relative_path == "2021/fire-a.hdf5"
    assert descriptor.raw_start_index == 6 - history
    assert descriptor.target_index == 6
    assert descriptor.target_date == "2021-08-07"
    x, y = dataset[0]
    assert x.shape == (history, 23, 8, 8)
    assert y.shape == (8, 8)


@pytest.mark.parametrize(
    ("history", "expected_stale_days"),
    [(1, [4]), (5, [0, 1, 2, 3, 4])],
)
def test_m02_loads_the_immediately_preceding_fire_sequence(
    tmp_path: Path, history: int, expected_stale_days: list[int]
) -> None:
    base = FakeDataset(_event(tmp_path), history=history)
    dataset = evaluation.ControlledMissingnessDataset(base, "M02")

    dataset[0]

    assert base.preprocessed_x is not None
    for time_index, day_index in enumerate(expected_stale_days):
        np.testing.assert_array_equal(
            base.preprocessed_x[time_index, 22], day_index * 100 + 22
        )


def test_dataset_applies_corruption_before_base_preprocessing(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path), history=5)
    dataset = evaluation.ControlledMissingnessDataset(base, "M03")

    dataset[0]

    assert base.preprocessed_x is not None
    assert np.isnan(base.preprocessed_x[:, 5:12]).all()


def test_spatial_fraction_is_exact_on_the_model_visible_center_crop(
    tmp_path: Path,
) -> None:
    base = FakeDataset(_event(tmp_path, height=12, width=10), history=5)
    dataset = evaluation.ControlledMissingnessDataset(base, "M06")

    corruption = dataset.corruption_for_index(0)
    x, _ = dataset[0]

    assert corruption.spatial_mask is not None
    assert corruption.spatial_mask.shape == (8, 8)
    assert int(corruption.spatial_mask.sum()) == 16
    assert x.shape[-2:] == (8, 8)


def test_c02_routing_mask_is_appended_after_feature_selection(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path), history=5)
    base.features_to_keep = contract.MULTI_FEATURES

    def preprocess_c02(x: np.ndarray, y: np.ndarray):
        expanded = torch.zeros((5, 40, 8, 8), dtype=torch.float32)
        expanded[:, :23] = torch.from_numpy(x.copy())
        return expanded, torch.from_numpy((y > 0).astype(np.int64))

    base.preprocess_and_augment = preprocess_c02
    dataset = evaluation.ControlledMissingnessDataset(
        base, "M06", routing_mask_channel=True
    )

    x, _ = dataset[0]

    assert x.shape == (5, len(contract.MULTI_FEATURES) + 1, 8, 8)
    assert torch.equal(x[:, -1], x[:1, -1].expand(5, -1, -1))
    assert int(x[0, -1].sum()) == 16


@pytest.mark.parametrize(
    ("height", "width"),
    [(14, 12), (8, 8), (6, 10), (10, 6)],
)
def test_hdf5_center_crop_matches_materialized_crop_without_full_read(
    tmp_path: Path, height: int, width: int,
) -> None:
    path = _event(tmp_path, height=height, width=width)
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    spatial = rows * 100 + columns
    with h5py.File(path, "r+") as handle:
        data = handle["data"]
        for day in range(data.shape[0]):
            for feature in range(data.shape[1]):
                data[day, feature] = day * 100_000 + feature * 1_000 + spatial

    selections = {
        "raw": (slice(1, 6), slice(None)),
        "target": (6, -1),
        "stale": (slice(0, 5), -1),
    }
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        for selection in selections.values():
            expected = evaluation._center_crop_last_two(
                np.asarray(data[selection], dtype=np.float32), 8
            )
            actual = evaluation._read_center_crop(data, selection, 8)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("train", "is_train=false"),
        ("year", "2021-only"),
        ("adjustment", "adjustment"),
        ("hdf5", "HDF5"),
    ],
)
def test_wrapper_rejects_non_engineering_dataset_contract(
    tmp_path: Path, mutation: str, message: str
) -> None:
    base = FakeDataset(_event(tmp_path), history=5)
    if mutation == "train":
        base.is_train = True
    elif mutation == "year":
        base.included_fire_years = [2022]
    elif mutation == "adjustment":
        base.n_leading_observations_test_adjustment = 5
    else:
        base.load_from_hdf5 = False

    with pytest.raises(ValueError, match=message):
        evaluation.ControlledMissingnessDataset(base, "M00")


def test_heldout_wrapper_requires_explicit_formal_authorization(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path, year=2022), history=5, year=2022)

    with pytest.raises(ValueError, match="held-out authorization"):
        evaluation.ControlledMissingnessDataset(
            base, "M00", evaluation_year=2022
        )

    dataset = evaluation.ControlledMissingnessDataset(
        base, "M00", evaluation_year=2022, heldout_authorized=True
    )
    assert dataset.describe(0).target_date == "2022-08-07"


@pytest.mark.parametrize("experiment_id", ["C00", "C02"])
def test_engineering_kwargs_freeze_2021_is_train_false_and_adjustment_six(
    tmp_path: Path, experiment_id: str
) -> None:
    spec = contract.experiment_spec(experiment_id)

    kwargs = evaluation.engineering_dataset_kwargs(spec, tmp_path)

    assert kwargs["data_dir"] == str(tmp_path.resolve())
    assert kwargs["included_fire_years"] == [2021]
    assert kwargs["stats_years"] == [2016, 2017, 2018, 2019, 2020]
    assert kwargs["is_train"] is False
    assert kwargs["load_from_hdf5"] is True
    assert kwargs["n_leading_observations_test_adjustment"] == 6
    assert kwargs["n_leading_observations"] == spec.n_leading_observations

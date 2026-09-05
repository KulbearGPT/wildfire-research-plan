from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from reproductions.wsts_fast_track import evaluation


class FakeDataset:
    def __init__(self, path: Path, *, year: int = 2021) -> None:
        self.data_dir = str(path.parents[1])
        self.included_fire_years = [year]
        self.stats_years = [2016, 2017, 2018, 2019, 2020]
        self.n_leading_observations = 1
        self.n_leading_observations_test_adjustment = 6
        self.skip_initial_samples = 5
        self.load_from_hdf5 = True
        self.is_train = False
        self.crop_side_length = 8
        self.remove_duplicate_features = False
        self.features_to_keep = None
        self.return_doy = False
        self.imgs_per_fire = {year: {path.stem: [str(path)]}}
        self.datapoints_per_fire = {year: {path.stem: 4}}
        self.preprocessed_x: np.ndarray | None = None

    def __len__(self) -> int:
        return 4

    def find_image_index_from_dataset_index(self, index: int):
        year = self.included_fire_years[0]
        return year, next(iter(self.imgs_per_fire[year])), index

    def preprocess_and_augment(self, x: np.ndarray, y: np.ndarray):
        self.preprocessed_x = x.copy()
        return torch.from_numpy(x.copy()), torch.from_numpy((y > 0).astype(np.int64))


def _event(tmp_path: Path, *, year: int = 2021) -> Path:
    path = tmp_path / "data" / str(year) / "fire-a.hdf5"
    path.parent.mkdir(parents=True)
    values = np.zeros((10, 23, 8, 8), dtype=np.float32)
    for day_index in range(10):
        for feature in range(23):
            values[day_index, feature] = day_index * 100 + feature
    start = date(year, 8, 1)
    dates = [(start + timedelta(days=index)).isoformat() for index in range(10)]
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=values)
        dataset.attrs["img_dates"] = dates
    return path


def test_controlled_t1_dataset_uses_the_frozen_target_population(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path))
    dataset = evaluation.ControlledMissingnessDataset(base, "M00")

    descriptor = dataset.describe(0)
    assert descriptor.raw_start_index == 5
    assert descriptor.target_index == 6
    assert descriptor.target_date == "2021-08-07"
    x, y = dataset[0]
    assert x.shape == (1, 23, 8, 8)
    assert y.shape == (8, 8)


def test_corruption_is_applied_before_preprocessing(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path))
    dataset = evaluation.ControlledMissingnessDataset(base, "M06")

    corruption = dataset.corruption_for_index(0)
    dataset[0]
    assert corruption.spatial_mask is not None
    assert int(corruption.spatial_mask.sum()) == 16
    assert base.preprocessed_x is not None
    assert np.isnan(base.preprocessed_x[:, :12, corruption.spatial_mask]).all()


def test_t5_history_is_outside_the_retained_contract(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path))
    base.n_leading_observations = 5
    base.skip_initial_samples = 1
    with pytest.raises(ValueError, match="requires T=1"):
        evaluation.ControlledMissingnessDataset(base, "M00")


def test_heldout_year_requires_explicit_authorization(tmp_path: Path) -> None:
    base = FakeDataset(_event(tmp_path, year=2022), year=2022)
    with pytest.raises(ValueError, match="held-out authorization"):
        evaluation.ControlledMissingnessDataset(base, "M00", evaluation_year=2022)

    dataset = evaluation.ControlledMissingnessDataset(
        base, "M00", evaluation_year=2022, heldout_authorized=True
    )
    assert dataset.describe(0).target_date == "2022-08-07"

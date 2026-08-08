from pathlib import Path

import h5py
import numpy as np
import pytest

from wildfire_phase0.inventory import inspect_hdf5, inventory_dataset


def _write_event(path: Path, *, byte_dates: bool = False) -> None:
    path.parent.mkdir(parents=True)
    values = np.zeros((3, 23, 8, 8), dtype=np.float32)
    values[0, 2, 0, 0] = np.nan
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values)
        data.attrs["year"] = 2021
        data.attrs["fire_name"] = "demo_fire"
        data.attrs["img_dates"] = (
            np.array([b"2021-08-01", b"2021-08-02", b"2021-08-03"])
            if byte_dates
            else ["2021-08-01", "2021-08-02", "2021-08-03"]
        )
        data.attrs["lnglat"] = [-120.5, 54.1]


def test_inspect_hdf5_reads_official_layout(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "demo_fire.hdf5"
    _write_event(path)
    item = inspect_hdf5(path, tmp_path)
    assert (item.year, item.fire_name, item.n_days, item.n_channels) == (2021, "demo_fire", 3, 23)
    assert item.nan_fraction == 1 / (3 * 23 * 8 * 8)


def test_inventory_dataset_is_deterministic(tmp_path: Path) -> None:
    _write_event(tmp_path / "2021" / "demo_fire.hdf5")
    assert [x.fire_name for x in inventory_dataset(tmp_path)] == ["demo_fire"]


def test_inspect_hdf5_rejects_missing_data_dataset(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "demo_fire.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w"):
        pass

    with pytest.raises(ValueError, match="data"):
        inspect_hdf5(path, tmp_path)


def test_inspect_hdf5_rejects_metadata_path_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "2020" / "demo_fire.hdf5"
    _write_event(path)

    with pytest.raises(ValueError, match="year"):
        inspect_hdf5(path, tmp_path)


def test_inspect_hdf5_decodes_byte_string_dates(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "demo_fire.hdf5"
    _write_event(path, byte_dates=True)

    assert inspect_hdf5(path, tmp_path).dates == (
        "2021-08-01",
        "2021-08-02",
        "2021-08-03",
    )


def test_inventory_dataset_rejects_empty_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no event files"):
        inventory_dataset(tmp_path)


def test_inventory_dataset_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inventory_dataset(tmp_path / "missing")

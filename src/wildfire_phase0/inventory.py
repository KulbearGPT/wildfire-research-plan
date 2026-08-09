"""Read-only inventory adapter for official WSTS/WSTS+ HDF5 event files."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wildfire_phase0.schema import EventInventory, validate_inventory_row


_REQUIRED_ATTRIBUTES = ("year", "fire_name", "img_dates", "lnglat")


def _decode_utf8(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_dates(value: Any) -> tuple[str, ...]:
    return tuple(_decode_utf8(item) for item in np.asarray(value).reshape(-1))


def inspect_hdf5(path: Path, data_root: Path) -> EventInventory:
    """Read metadata and missingness for one official WSTS/WSTS+ event file."""
    path = Path(path)
    data_root = Path(data_root)
    try:
        relative_path = path.resolve().relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError("path must be inside data_root") from error

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError("missing data dataset")
        data = handle["data"]
        if not isinstance(data, h5py.Dataset):
            raise ValueError("data object must be an HDF5 dataset")
        if len(data.shape) != 4:
            raise ValueError("data shape must be (days, 23, height, width)")
        if any(dimension <= 0 for dimension in data.shape):
            raise ValueError("data shape dimensions must be strictly positive")
        if data.shape[1] != 23:
            raise ValueError("data shape must be (days, 23, height, width)")
        missing_attributes = [name for name in _REQUIRED_ATTRIBUTES if name not in data.attrs]
        if missing_attributes:
            raise ValueError(f"missing required attribute: {missing_attributes[0]}")

        year = int(data.attrs["year"])
        fire_name = _decode_utf8(data.attrs["fire_name"])
        dates = _decode_dates(data.attrs["img_dates"])
        folder_year = int(path.parent.name)
        if folder_year != year:
            raise ValueError("folder year must match attribute year")
        if path.stem != fire_name:
            raise ValueError("filename stem must match fire_name")

        n_days, n_channels, height, width = data.shape
        nan_count = 0
        target_days = 0
        zero_target_days = 0
        positive_target_pixels = 0
        active_fire_min_positive: float | None = None
        active_fire_max_positive: float | None = None
        for day_index in range(n_days):
            day = np.asarray(data[day_index])
            nan_count += int(np.isnan(day).sum())
            active = day[22]
            if np.any(np.isinf(active)):
                raise ValueError("stored active-fire values must be finite or NaN")
            finite_active = active[np.isfinite(active)]
            if np.any(finite_active < 0) or np.any(
                finite_active != np.floor(finite_active)
            ):
                raise ValueError("stored active-fire values must be nonnegative integer hours")
            if np.any(finite_active > 23):
                raise ValueError("stored active-fire values must be within 0-23 hours")
            positive_active = finite_active[finite_active > 0]
            if positive_active.size:
                day_min = float(np.min(positive_active))
                day_max = float(np.max(positive_active))
                active_fire_min_positive = (
                    day_min
                    if active_fire_min_positive is None
                    else min(active_fire_min_positive, day_min)
                )
                active_fire_max_positive = (
                    day_max
                    if active_fire_max_positive is None
                    else max(active_fire_max_positive, day_max)
                )
            if day_index > 0:
                positives = int(np.count_nonzero(finite_active > 0))
                target_days += 1
                zero_target_days += int(positives == 0)
                positive_target_pixels += positives
        nan_fraction = nan_count / (n_days * n_channels * height * width)

    return validate_inventory_row(
        {
            "year": year,
            "fire_name": fire_name,
            "path": relative_path,
            "n_days": n_days,
            "n_channels": n_channels,
            "height": height,
            "width": width,
            "dates": dates,
            "nan_fraction": nan_fraction,
            "target_days": target_days,
            "zero_target_days": zero_target_days,
            "positive_target_pixels": positive_target_pixels,
            "active_fire_min_positive": active_fire_min_positive,
            "active_fire_max_positive": active_fire_max_positive,
        }
    )


def inventory_dataset(data_root: Path) -> list[EventInventory]:
    """Inspect all event files immediately below numeric year directories."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    event_paths = sorted(
        (
            path
            for year_directory in data_root.iterdir()
            if year_directory.is_dir() and year_directory.name.isdigit()
            for path in year_directory.glob("*.hdf5")
            if path.is_file()
        ),
        key=lambda path: (path.parent.name, path.name),
    )
    if not event_paths:
        raise ValueError("no event files found")

    return sorted(
        (inspect_hdf5(path, data_root) for path in event_paths),
        key=lambda item: (item.year, item.fire_name),
    )

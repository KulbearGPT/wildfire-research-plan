"""Read-only, streaming evaluation of fixed wildfire rule baselines."""

from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from wildfire_phase0.metrics import (
    BinaryScoreCounts,
    binary_average_precision,
    binary_score_counts,
)
from wildfire_phase0.path_safety import canonical_root, require_contained_path


_BASELINES = ("no_fire", "persistence_latest")
_EVENT_COLUMNS = [
    "event_id",
    "year",
    "fire_name",
    "path",
    "split",
    "baseline",
    "target_days",
    "zero_target_days",
    "positive_target_pixels",
    "total_pixels",
    "tp",
    "fp",
    "fn",
    "tn",
    "event_ap",
    "event_ap_defined",
    "zero_target_pixels",
    "zero_target_predicted_positive_pixels",
]
_MANIFEST_COLUMNS = ("event_id", "year", "fire_name", "path", "split")
_REQUIRED_ATTRIBUTES = ("year", "fire_name", "img_dates", "lnglat")
_SPLITS = {"train", "validation", "test"}


def _decode_utf8(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _validate_dates(value: Any, n_days: int) -> None:
    dates = tuple(_decode_utf8(item) for item in np.asarray(value).reshape(-1))
    if len(dates) != n_days:
        raise ValueError("dates count must equal n_days")
    try:
        parsed = tuple(date.fromisoformat(value) for value in dates)
    except ValueError as error:
        raise ValueError("image dates must use ISO format") from error
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly increasing")


def _active_mask(values: object) -> np.ndarray:
    active = np.asarray(values)
    if np.any(np.isinf(active)):
        raise ValueError("stored active-fire values must be finite or NaN")
    finite = active[np.isfinite(active)]
    if np.any(finite < 0) or np.any(finite != np.floor(finite)):
        raise ValueError("stored active-fire values must be nonnegative integer hours")
    if np.any(finite > 23):
        raise ValueError("stored active-fire values must be within 0-23 hours")
    return active > 0


def evaluate_rule_event(path: Path, data_root: Path, split: str) -> pd.DataFrame:
    """Evaluate no-fire and latest-mask persistence for one event at T=1."""
    if split not in _SPLITS:
        raise ValueError("split must be one of train, validation, or test")

    root = canonical_root(Path(data_root), "data root", must_exist=True)
    event_path = require_contained_path(root, Path(path), "data root")
    relative_path = event_path.relative_to(root).as_posix()

    with h5py.File(event_path, "r") as handle:
        if "data" not in handle:
            raise ValueError("missing data dataset")
        data = handle["data"]
        if not isinstance(data, h5py.Dataset):
            raise ValueError("data object must be an HDF5 dataset")
        if len(data.shape) != 4 or any(dimension <= 0 for dimension in data.shape):
            raise ValueError("data shape must be (days, 23, height, width)")
        if data.shape[1] != 23:
            raise ValueError("data shape must be (days, 23, height, width)")
        n_days, _, _, _ = data.shape
        if n_days < 2:
            raise ValueError("event must contain at least 2 days")

        missing_attributes = [
            name for name in _REQUIRED_ATTRIBUTES if name not in data.attrs
        ]
        if missing_attributes:
            raise ValueError(f"missing required attribute: {missing_attributes[0]}")

        year = int(data.attrs["year"])
        fire_name = _decode_utf8(data.attrs["fire_name"])
        try:
            folder_year = int(event_path.parent.name)
        except ValueError as error:
            raise ValueError("folder year must be numeric") from error
        if folder_year != year:
            raise ValueError("folder year must match attribute year")
        if event_path.stem != fire_name:
            raise ValueError("filename stem must match fire_name")
        _validate_dates(data.attrs["img_dates"], n_days)

        counts = {baseline: BinaryScoreCounts() for baseline in _BASELINES}
        zero_target_predicted = {baseline: 0 for baseline in _BASELINES}
        target_days = n_days - 1
        zero_target_days = 0
        positive_target_pixels = 0
        zero_target_pixels = 0

        previous = _active_mask(data[0, 22])
        for target_index in range(1, n_days):
            target = _active_mask(data[target_index, 22])
            predictions = {
                "no_fire": np.zeros_like(target, dtype=np.uint8),
                "persistence_latest": previous,
            }
            positives = int(np.count_nonzero(target))
            positive_target_pixels += positives
            if positives == 0:
                zero_target_days += 1
                zero_target_pixels += int(target.size)
                for baseline, prediction in predictions.items():
                    zero_target_predicted[baseline] += int(
                        np.count_nonzero(prediction)
                    )
            for baseline, prediction in predictions.items():
                counts[baseline] = counts[baseline] + binary_score_counts(
                    target, prediction
                )
            previous = target

    event_id = f"{year}:{fire_name}"
    records = []
    for baseline in _BASELINES:
        baseline_counts = counts[baseline]
        records.append(
            {
                "event_id": event_id,
                "year": year,
                "fire_name": fire_name,
                "path": relative_path,
                "split": split,
                "baseline": baseline,
                "target_days": target_days,
                "zero_target_days": zero_target_days,
                "positive_target_pixels": positive_target_pixels,
                "total_pixels": baseline_counts.total,
                "tp": baseline_counts.tp,
                "fp": baseline_counts.fp,
                "fn": baseline_counts.fn,
                "tn": baseline_counts.tn,
                "event_ap": binary_average_precision(baseline_counts),
                "event_ap_defined": baseline_counts.positives > 0,
                "zero_target_pixels": zero_target_pixels,
                "zero_target_predicted_positive_pixels": zero_target_predicted[
                    baseline
                ],
            }
        )
    return pd.DataFrame(records, columns=_EVENT_COLUMNS)


def _event_paths(data_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for year_directory in data_root.iterdir()
            if year_directory.is_dir() and year_directory.name.isdigit()
            for path in year_directory.glob("*.hdf5")
            if path.is_file()
        ),
        key=lambda path: (path.parent.name, path.name),
    )


def _frozen_split(year: int) -> str:
    if 2016 <= year <= 2020:
        return "train"
    if year == 2021:
        return "validation"
    if 2022 <= year <= 2023:
        return "test"
    raise ValueError("event year must be within the 2016-2023 frozen split")


def _validate_manifest(
    data_root: Path, split_manifest: pd.DataFrame
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    missing_columns = set(_MANIFEST_COLUMNS).difference(split_manifest.columns)
    if missing_columns:
        raise ValueError(
            f"split manifest missing required columns: {sorted(missing_columns)}"
        )
    if split_manifest[list(_MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError("split manifest values must not be missing")
    duplicate_identity = (
        split_manifest.duplicated(subset=["event_id"]).any()
        or split_manifest.duplicated(subset=["path"]).any()
        or split_manifest.duplicated(subset=["year", "fire_name"]).any()
    )
    if duplicate_identity:
        raise ValueError("split manifest contains duplicate event entries")

    event_paths = _event_paths(data_root)
    paths_by_relative = {
        path.relative_to(data_root).as_posix(): path for path in event_paths
    }
    records = split_manifest.to_dict("records")
    manifest_paths = {row["path"] for row in records}
    if manifest_paths != set(paths_by_relative):
        raise ValueError("split manifest paths must exactly match dataset event paths")
    return records, paths_by_relative


def evaluate_rule_dataset(
    data_root: Path, split_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate every event joined exactly to its supplied frozen split row."""
    root = canonical_root(Path(data_root), "data root", must_exist=True)
    manifest_records, paths_by_relative = _validate_manifest(root, split_manifest)

    event_frames = []
    for manifest_row in manifest_records:
        relative_path = manifest_row["path"]
        event_frame = evaluate_rule_event(
            paths_by_relative[relative_path], root, manifest_row["split"]
        )
        observed = event_frame.iloc[0]
        expected_metadata = {
            "path": relative_path,
            "year": manifest_row["year"],
            "fire_name": manifest_row["fire_name"],
            "event_id": manifest_row["event_id"],
        }
        if any(observed[column] != expected for column, expected in expected_metadata.items()):
            raise ValueError(
                "split manifest path, year, fire_name, and event_id must exactly match events"
            )
        if manifest_row["split"] != _frozen_split(int(observed["year"])):
            raise ValueError("split manifest does not follow the frozen split")
        event_frames.append(event_frame)

    if not event_frames:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    result = pd.concat(event_frames, ignore_index=True)
    return result.sort_values(
        ["split", "year", "fire_name", "baseline"], kind="stable"
    ).reset_index(drop=True)

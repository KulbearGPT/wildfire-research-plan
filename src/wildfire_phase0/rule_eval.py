"""Read-only, streaming evaluation of fixed wildfire rule baselines."""

from datetime import date
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
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
_SUMMARY_COLUMNS = [
    "split",
    "baseline",
    "events",
    "event_ap_defined",
    "event_ap_undefined",
    "event_macro_ap",
    "pooled_ap",
    "positive_prevalence",
    "target_days",
    "zero_target_days",
    "zero_target_day_far",
    "positive_target_pixels",
    "total_pixels",
]
_EVENT_COUNT_COLUMNS = (
    "target_days",
    "zero_target_days",
    "positive_target_pixels",
    "total_pixels",
    "tp",
    "fp",
    "fn",
    "tn",
    "zero_target_pixels",
    "zero_target_predicted_positive_pixels",
)
_MANIFEST_COLUMNS = ("event_id", "year", "fire_name", "path", "split")
_REQUIRED_ATTRIBUTES = ("year", "fire_name", "img_dates", "lnglat")
_SPLITS = {"train", "validation", "test"}
# Covers decimal CSV round-tripping while remaining strict for event-level AP.
_EVENT_AP_ABS_TOLERANCE = 1e-12


def _decode_utf8_attribute(value: Any, name: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} attribute must contain UTF-8 text") from error
    if isinstance(value, str):
        return value
    raise ValueError(f"{name} attribute must contain UTF-8 text")


def _validate_dates(value: Any, n_days: int) -> None:
    dates = tuple(
        _decode_utf8_attribute(item, "img_dates")
        for item in np.asarray(value).reshape(-1)
    )
    if len(dates) != n_days:
        raise ValueError("dates count must equal n_days")
    try:
        parsed = tuple(date.fromisoformat(value) for value in dates)
    except ValueError as error:
        raise ValueError("image dates must use ISO format") from error
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly increasing")
    if any((right - left).days != 1 for left, right in zip(parsed, parsed[1:])):
        raise ValueError("image dates must be consecutive calendar days")


def _validate_lnglat(value: Any) -> None:
    coordinates = np.asarray(value)
    numeric = np.issubdtype(coordinates.dtype, np.integer) or np.issubdtype(
        coordinates.dtype, np.floating
    )
    if coordinates.shape != (2,) or not numeric:
        raise ValueError("lnglat attribute must be a numeric pair")
    if np.any(np.isinf(coordinates)):
        raise ValueError("lnglat attribute must not contain infinities")


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

        stored_year = data.attrs["year"]
        if isinstance(stored_year, (bool, np.bool_)) or not isinstance(
            stored_year, Integral
        ):
            raise ValueError("year attribute must be a non-boolean integer scalar")
        year = int(stored_year)
        fire_name = _decode_utf8_attribute(data.attrs["fire_name"], "fire_name")
        try:
            folder_year = int(event_path.parent.name)
        except ValueError as error:
            raise ValueError("folder year must be numeric") from error
        if folder_year != year:
            raise ValueError("folder year must match attribute year")
        if event_path.stem != fire_name:
            raise ValueError("filename stem must match fire_name")
        _validate_dates(data.attrs["img_dates"], n_days)
        _validate_lnglat(data.attrs["lnglat"])

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
    event_paths = []
    for year_entry in data_root.iterdir():
        if not year_entry.name.isdigit():
            continue
        year_directory = require_contained_path(data_root, year_entry, "data root")
        if not year_directory.is_dir():
            continue
        for candidate_entry in year_directory.glob("*.hdf5"):
            candidate = require_contained_path(
                data_root, candidate_entry, "data root"
            )
            if candidate.is_file():
                event_paths.append(candidate)
    return sorted(event_paths, key=lambda path: (path.parent.name, path.name))


def _frozen_split(year: int) -> str:
    if 2016 <= year <= 2020:
        return "train"
    if year == 2021:
        return "validation"
    if 2022 <= year <= 2023:
        return "test"
    raise ValueError("event year must be within the 2016-2023 frozen split")


def _validate_manifest_path(value: str) -> None:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("manifest path must be a canonical POSIX relative path")


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
    records = split_manifest.to_dict("records")
    for row in records:
        manifest_year = row["year"]
        if isinstance(manifest_year, (bool, np.bool_)) or not isinstance(
            manifest_year, Integral
        ):
            raise ValueError("manifest year must be a non-boolean integer scalar")
        for column in ("event_id", "fire_name", "path", "split"):
            if not isinstance(row[column], str):
                raise ValueError(f"manifest {column} must be a string")
        _validate_manifest_path(row["path"])

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


def _validate_event_metrics_for_summary(event_metrics: pd.DataFrame) -> None:
    missing_columns = set(_EVENT_COLUMNS).difference(event_metrics.columns)
    if missing_columns:
        raise ValueError(
            f"event metrics missing required columns: {sorted(missing_columns)}"
        )
    if event_metrics.empty:
        raise ValueError("event metrics must contain at least one row")
    identity_columns = ["event_id", "split", "baseline"]
    if event_metrics[identity_columns].isna().any().any():
        raise ValueError(
            "event metrics event_id, split, and baseline values must not be missing"
        )
    if not event_metrics["split"].isin(_SPLITS).all():
        raise ValueError(
            "event metrics split values must be train, validation, or test"
        )
    if not event_metrics["baseline"].isin(_BASELINES).all():
        raise ValueError(
            "event metrics baseline values must be no_fire or persistence_latest"
        )
    if event_metrics.duplicated(subset=["event_id", "baseline"]).any():
        raise ValueError("event metrics must contain one row per event and baseline")
    if not all(
        isinstance(value, (bool, np.bool_))
        for value in event_metrics["event_ap_defined"]
    ):
        raise ValueError("event_ap_defined values must be boolean")
    if not all(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and value >= 0
        for column in _EVENT_COUNT_COLUMNS
        for value in event_metrics[column]
    ):
        raise ValueError(
            "event metric count fields must contain non-negative integers"
        )
    if not (
        event_metrics["tp"]
        + event_metrics["fp"]
        + event_metrics["fn"]
        + event_metrics["tn"]
        == event_metrics["total_pixels"]
    ).all():
        raise ValueError("tp + fp + fn + tn must equal total_pixels")
    if not (
        event_metrics["tp"] + event_metrics["fn"]
        == event_metrics["positive_target_pixels"]
    ).all():
        raise ValueError("tp + fn must equal positive_target_pixels")
    if (
        (event_metrics["target_days"] == 0)
        | (event_metrics["total_pixels"] == 0)
    ).any():
        raise ValueError("target_days and total_pixels must be positive")
    if (event_metrics["zero_target_days"] > event_metrics["target_days"]).any():
        raise ValueError("zero_target_days must not exceed target_days")
    if (event_metrics["zero_target_pixels"] > event_metrics["total_pixels"]).any():
        raise ValueError("zero_target_pixels must not exceed total_pixels")
    if not (
        event_metrics["zero_target_pixels"] * event_metrics["target_days"]
        == event_metrics["total_pixels"] * event_metrics["zero_target_days"]
    ).all():
        raise ValueError(
            "zero_target_pixels must match zero_target_days and target_days"
        )
    if (
        event_metrics["zero_target_predicted_positive_pixels"]
        > event_metrics["zero_target_pixels"]
    ).any():
        raise ValueError(
            "zero_target_predicted_positive_pixels must not exceed zero_target_pixels"
        )
    if (
        event_metrics["zero_target_predicted_positive_pixels"]
        > event_metrics["fp"]
    ).any():
        raise ValueError(
            "zero_target_predicted_positive_pixels must not exceed fp"
        )
    if not all(
        bool(defined) == (positive_pixels > 0)
        for defined, positive_pixels in zip(
            event_metrics["event_ap_defined"],
            event_metrics["positive_target_pixels"],
        )
    ):
        raise ValueError(
            "event_ap_defined must equal whether positive_target_pixels is positive"
        )
    positive_target_pixels = event_metrics["positive_target_pixels"]
    if (
        (
            positive_target_pixels
            < event_metrics["target_days"] - event_metrics["zero_target_days"]
        )
        | (
            positive_target_pixels
            > event_metrics["total_pixels"] - event_metrics["zero_target_pixels"]
        )
    ).any():
        raise ValueError(
            "positive_target_pixels must satisfy target_days - zero_target_days <= "
            "positive_target_pixels <= total_pixels - zero_target_pixels"
        )
    for defined, value, tp, fp, fn, tn in zip(
        event_metrics["event_ap_defined"],
        event_metrics["event_ap"],
        event_metrics["tp"],
        event_metrics["fp"],
        event_metrics["fn"],
        event_metrics["tn"],
    ):
        finite = (
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and bool(np.isfinite(value))
        )
        missing = value is None or value is pd.NA or (
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and bool(np.isnan(value))
        )
        if not ((bool(defined) and finite) or (not bool(defined) and missing)):
            raise ValueError(
                "event_ap must be finite exactly when event_ap_defined is true"
            )
        if bool(defined):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("defined event_ap values must be within [0, 1]")
            expected = binary_average_precision(
                BinaryScoreCounts(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
            )
            if not np.isclose(
                float(value), expected, rtol=0.0, atol=_EVENT_AP_ABS_TOLERANCE
            ):
                raise ValueError(
                    "defined event_ap must match AP recomputed from confusion counts"
                )


def summarize_rule_metrics(event_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event metrics for each frozen split and fixed rule baseline."""
    _validate_event_metrics_for_summary(event_metrics)

    records = []
    grouped = event_metrics.groupby(["split", "baseline"], sort=True)
    for (split, baseline), group in grouped:
        defined = group["event_ap_defined"]
        counts = BinaryScoreCounts(
            tp=int(group["tp"].sum()),
            fp=int(group["fp"].sum()),
            fn=int(group["fn"].sum()),
            tn=int(group["tn"].sum()),
        )
        total_pixels = int(group["total_pixels"].sum())
        positive_target_pixels = int(group["positive_target_pixels"].sum())
        zero_target_pixels = int(group["zero_target_pixels"].sum())
        zero_target_predicted = int(
            group["zero_target_predicted_positive_pixels"].sum()
        )
        defined_count = int(defined.sum())
        records.append(
            {
                "split": split,
                "baseline": baseline,
                "events": len(group),
                "event_ap_defined": defined_count,
                "event_ap_undefined": len(group) - defined_count,
                "event_macro_ap": (
                    float(group.loc[defined, "event_ap"].mean())
                    if defined_count
                    else float("nan")
                ),
                "pooled_ap": binary_average_precision(counts),
                "positive_prevalence": (
                    positive_target_pixels / total_pixels
                    if total_pixels
                    else float("nan")
                ),
                "target_days": int(group["target_days"].sum()),
                "zero_target_days": int(group["zero_target_days"].sum()),
                "zero_target_day_far": (
                    zero_target_predicted / zero_target_pixels
                    if zero_target_pixels
                    else float("nan")
                ),
                "positive_target_pixels": positive_target_pixels,
                "total_pixels": total_pixels,
            }
        )
    return pd.DataFrame(records, columns=_SUMMARY_COLUMNS)


def _format_summary_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "undefined"
    if column in {
        "events",
        "event_ap_defined",
        "event_ap_undefined",
        "target_days",
        "zero_target_days",
        "positive_target_pixels",
        "total_pixels",
    }:
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def render_rule_report(summary: pd.DataFrame) -> str:
    """Render a deterministic Markdown report for fixed rule evaluation."""
    missing_columns = set(_SUMMARY_COLUMNS).difference(summary.columns)
    if missing_columns:
        raise ValueError(f"rule summary missing required columns: {sorted(missing_columns)}")
    ordered = summary.sort_values(["split", "baseline"], kind="stable")
    header = "| " + " | ".join(_SUMMARY_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in _SUMMARY_COLUMNS) + " |"
    rows = [
        "| "
        + " | ".join(
            _format_summary_value(column, value)
            for column, value in zip(_SUMMARY_COLUMNS, values)
        )
        + " |"
        for values in ordered[_SUMMARY_COLUMNS].itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "# Fixed Rule Baseline Evaluation",
            "",
            "This report evaluates fixed T=1 rules with next-day target semantics.",
            "The frozen split definition is 2016--2020 train, 2021 validation, "
            "and 2022--2023 test.",
            "No threshold or model tuning was performed.",
            "Event-macro AP includes only defined events; event_ap_undefined "
            "records each undefined event AP from zero-positive events.",
            "Warning: Raw AP is prevalence-dependent; compare it together with "
            "positive_prevalence.",
            "",
            header,
            separator,
            *rows,
            "",
        ]
    )

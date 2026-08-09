from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile


_REPAIR_ATTRIBUTES = (
    "active_fire_source_encoding",
    "active_fire_stored_encoding",
    "active_fire_repair_version",
    "active_fire_source_fingerprint",
)


@dataclass(frozen=True)
class ExpectedYear:
    files: int
    target_days: int
    zero_target_days: int
    positive_target_pixels: int


@dataclass(frozen=True)
class VerifiedYear:
    files: int
    target_days: int
    zero_target_days: int
    positive_target_pixels: int


def _decoded_strings(values: object) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.atleast_1d(values).tolist()
    )


def _raw_channels(path: Path, height: int, width: int) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.shape == (height, width, 23):
        return np.moveaxis(image, -1, 0)
    if image.shape == (23, height, width):
        return image
    raise ValueError(
        f"raw comparison shape must be HWC or CHW with 23 channels: {path}"
    )


def _normalize_raw_active(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.any(np.isinf(array)):
        raise ValueError("raw comparison active-fire values contain infinity")
    finite = array[np.isfinite(array)]
    if np.any(finite < 0) or np.any(finite != np.floor(finite)):
        raise ValueError(
            "raw comparison active-fire values must be nonnegative integers"
        )
    positive = finite[finite > 0]
    normalized = np.nan_to_num(array, nan=0.0).copy()
    if positive.size:
        hour_mask = positive <= 23
        if np.any(hour_mask) and not np.all(hour_mask):
            raise ValueError("raw comparison contains mixed hour and HHMM values")
        if not np.any(hour_mask):
            hours = np.floor_divide(positive, 100)
            minutes = np.mod(positive, 100)
            if np.any(hours > 23) or np.any(minutes > 59):
                raise ValueError("raw comparison contains an invalid HHMM value")
            normalized = np.floor_divide(normalized, 100)
    if np.any(normalized < 0) or np.any(normalized > 23):
        raise ValueError("raw comparison normalized values must be within 0-23")
    return normalized


def _require_repair_attributes(data: h5py.Dataset, path: Path) -> None:
    missing = [name for name in _REPAIR_ATTRIBUTES if name not in data.attrs]
    if missing:
        raise ValueError(f"missing repair attribute in {path}: {', '.join(missing)}")
    if str(data.attrs["active_fire_source_encoding"]) not in {
        "hour",
        "hhmm",
        "no_positive_values",
    }:
        raise ValueError(f"invalid active-fire source repair attribute in {path}")
    if str(data.attrs["active_fire_stored_encoding"]) != "hour":
        raise ValueError(f"invalid stored-encoding repair attribute in {path}")
    if str(data.attrs["active_fire_repair_version"]) != "1":
        raise ValueError(f"invalid repair version attribute in {path}")
    if not str(data.attrs["active_fire_source_fingerprint"]):
        raise ValueError(f"empty source-fingerprint repair attribute in {path}")


def _require_raw_sample(
    path: Path,
    staged_day: np.ndarray,
    height: int,
    width: int,
) -> None:
    raw = _raw_channels(path, height, width)
    if not np.array_equal(raw[:22], staged_day[:22], equal_nan=True):
        raise ValueError(f"raw comparison failed for non-active channels: {path}")
    normalized_active = _normalize_raw_active(raw[22])
    if not np.array_equal(normalized_active, staged_day[22], equal_nan=True):
        raise ValueError(f"raw comparison failed for active-fire channel: {path}")


def _verify_event(
    path: Path,
    source_tiff_root: Path,
    year: int,
    compare_raw: bool,
) -> VerifiedYear:
    event_dir = source_tiff_root / str(year) / path.stem
    tiff_paths = tuple(sorted(event_dir.glob("*.tif"), key=lambda item: item.name))
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if not isinstance(data, h5py.Dataset):
            raise ValueError(f"data object must be an HDF5 dataset: {path}")
        if len(data.shape) != 4:
            raise ValueError(f"data must have a 4D shape with 23 channels: {path}")
        if data.shape[1] != 23:
            raise ValueError(f"data must have 23 channels: {path}")
        if any(dimension <= 0 for dimension in data.shape):
            raise ValueError(f"data dimensions must be positive: {path}")
        if data.compression != "lzf" or not data.shuffle:
            raise ValueError(f"data compression must be LZF with shuffle: {path}")
        _require_repair_attributes(data, path)
        if int(data.attrs.get("year", -1)) != year:
            raise ValueError(f"dataset year does not match its path: {path}")
        if str(data.attrs.get("fire_name", "")) != path.stem:
            raise ValueError(f"dataset fire name does not match its path: {path}")

        dates = _decoded_strings(data.attrs.get("img_dates", ()))
        tiff_stems = tuple(item.stem for item in tiff_paths)
        if dates != tiff_stems:
            raise ValueError(f"dataset dates do not match TIFF stems: {path}")
        if data.shape[0] != len(dates):
            raise ValueError(f"dataset day count does not match TIFF stems: {path}")

        active = np.asarray(data[:, 22])
        if (
            np.any(~np.isfinite(active))
            or np.any(active < 0)
            or np.any(active > 23)
            or np.any(active != np.floor(active))
        ):
            raise ValueError(
                f"active-fire values must be finite integer hours in 0-23: {path}"
            )
        targets = active[1:]
        positive_by_day = np.count_nonzero(targets > 0, axis=(1, 2))
        summary = VerifiedYear(
            files=1,
            target_days=int(targets.shape[0]),
            zero_target_days=int(np.count_nonzero(positive_by_day == 0)),
            positive_target_pixels=int(np.count_nonzero(targets > 0)),
        )

        if compare_raw:
            _, _, height, width = data.shape
            for day in sorted({0, data.shape[0] - 1}):
                _require_raw_sample(
                    tiff_paths[day], np.asarray(data[day]), height, width
                )
    return summary


def _require_expected_totals(
    year: int, actual: VerifiedYear, expected: ExpectedYear
) -> None:
    comparisons = (
        ("files", actual.files, expected.files),
        ("target days", actual.target_days, expected.target_days),
        ("zero target days", actual.zero_target_days, expected.zero_target_days),
        (
            "positive target pixels",
            actual.positive_target_pixels,
            expected.positive_target_pixels,
        ),
    )
    for label, actual_value, expected_value in comparisons:
        if actual_value != expected_value:
            raise ValueError(
                f"year {year} {label}: expected {expected_value}, got {actual_value}"
            )


def verify_active_fire_dataset(
    hdf5_root: Path,
    source_tiff_root: Path,
    expected: Mapping[int, ExpectedYear],
) -> dict[int, VerifiedYear]:
    hdf5_root = Path(hdf5_root)
    source_tiff_root = Path(source_tiff_root)
    verified: dict[int, VerifiedYear] = {}
    for year in sorted(expected):
        paths = tuple(
            sorted(
                (hdf5_root / str(year)).glob("*.hdf5"),
                key=lambda item: item.name,
            )
        )
        if len(paths) != expected[year].files:
            raise ValueError(
                f"year {year} files: expected {expected[year].files}, got {len(paths)}"
            )
        totals = VerifiedYear(0, 0, 0, 0)
        raw_sample_paths = set(paths[:1] + paths[-1:])
        for path in paths:
            event = _verify_event(
                path, source_tiff_root, year, path in raw_sample_paths
            )
            totals = VerifiedYear(
                files=totals.files + event.files,
                target_days=totals.target_days + event.target_days,
                zero_target_days=totals.zero_target_days + event.zero_target_days,
                positive_target_pixels=(
                    totals.positive_target_pixels + event.positive_target_pixels
                ),
            )
        _require_expected_totals(year, totals, expected[year])
        verified[year] = totals
    return verified


def _parse_expectations(values: Sequence[str]) -> dict[int, ExpectedYear]:
    expected: dict[int, ExpectedYear] = {}
    for value in values:
        parts = value.split(":")
        if len(parts) != 5:
            raise ValueError(
                "expect value must be YEAR:FILES:TARGET_DAYS:ZERO_DAYS:POSITIVE_PIXELS"
            )
        try:
            year, files, target_days, zero_days, positive_pixels = map(int, parts)
        except ValueError as error:
            raise ValueError(f"expect value must contain integers: {value}") from error
        if year in expected:
            raise ValueError(f"expect year must be unique: {year}")
        if min(files, target_days, zero_days, positive_pixels) < 0:
            raise ValueError(f"expect counts must be nonnegative: {value}")
        expected[year] = ExpectedYear(
            files, target_days, zero_days, positive_pixels
        )
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify staged active-fire repairs."
    )
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--source-tiff-root", type=Path, required=True)
    parser.add_argument("--expect", action="append", required=True)
    arguments = parser.parse_args(argv)
    expected = _parse_expectations(arguments.expect)
    summary = verify_active_fire_dataset(
        arguments.hdf5_root, arguments.source_tiff_root, expected
    )
    print(
        json.dumps(
            {str(year): asdict(summary[year]) for year in sorted(summary)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

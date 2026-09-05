"""Compute train-only WSTS+ normalization statistics from event HDF5 files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np

from .runtime import TRAIN_YEARS


FEATURE_COUNT = 23
ACTIVE_FIRE_INDEX = 22


class FeatureAccumulator:
    """Accumulate finite per-feature population moments in float64."""

    def __init__(self) -> None:
        self.counts = np.zeros(FEATURE_COUNT, dtype=np.int64)
        self.totals = np.zeros(FEATURE_COUNT, dtype=np.int64)
        self.sums = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self.sum_squares = np.zeros(FEATURE_COUNT, dtype=np.float64)

    def update(self, data: np.ndarray) -> None:
        if data.ndim != 4 or data.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"expected HDF5 data shape (days, 23, height, width), got {data.shape}"
            )
        pixels_per_feature = data.shape[0] * data.shape[2] * data.shape[3]
        self.totals += pixels_per_feature
        for feature in range(FEATURE_COUNT):
            values = np.asarray(data[:, feature], dtype=np.float64).reshape(-1)
            valid = np.isfinite(values)
            if feature == ACTIVE_FIRE_INDEX:
                valid &= values > 0
            selected = values[valid]
            self.counts[feature] += selected.size
            self.sums[feature] += selected.sum(dtype=np.float64)
            self.sum_squares[feature] += np.square(selected).sum(dtype=np.float64)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (self.counts == 0).any():
            missing_features = np.flatnonzero(self.counts == 0).tolist()
            raise ValueError(f"no valid observations for features {missing_features}")
        means = self.sums / self.counts
        variances = self.sum_squares / self.counts - np.square(means)
        variances = np.maximum(variances, 0.0)
        stds = np.sqrt(variances)
        missing_values = 1.0 - self.counts / self.totals
        return (
            means.astype(np.float32),
            stds.astype(np.float32),
            missing_values.astype(np.float32),
        )


def compute_paths(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream one or more event files into canonical feature statistics."""

    accumulator = FeatureAccumulator()
    found = False
    for path in paths:
        found = True
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"HDF5 event has no data dataset: {path}")
            dataset = handle["data"]
            if dataset.ndim != 4 or dataset.shape[1] != FEATURE_COUNT:
                raise ValueError(f"invalid data shape in {path}: {dataset.shape}")
            for start in range(0, dataset.shape[0], 4):
                accumulator.update(dataset[start : start + 4])
    if not found:
        raise ValueError("no HDF5 paths were supplied")
    return accumulator.finalize()


def training_paths(data_root: Path) -> list[Path]:
    """Return all direct HDF5 events in the frozen training years."""

    data_root = data_root.resolve()
    paths: list[Path] = []
    for year in TRAIN_YEARS:
        year_root = data_root / str(year)
        if not year_root.is_dir():
            raise ValueError(f"training year directory is missing: {year_root}")
        paths.extend(sorted(year_root.glob("*.hdf5")))
    if not paths:
        raise ValueError(f"no training HDF5 events found under {data_root}")
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite training statistics: {output}")
    paths = training_paths(args.data_root)
    means, stds, missing_values = compute_paths(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        means=means,
        stds=stds,
        missing_values=missing_values,
    )
    print(
        json.dumps(
            {
                "event_files": len(paths),
                "output": str(output),
                "train_years": list(TRAIN_YEARS),
                "positive_fire_rate": float(1.0 - missing_values[-1]),
                "positive_class_weight": float(1.0 / (1.0 - missing_values[-1])),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

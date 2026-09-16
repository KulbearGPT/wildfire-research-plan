"""Runtime contract shared by retained corrected T=1 experiments."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

TRAIN_YEARS = (2016, 2017, 2018, 2019, 2020)
VALIDATION_YEARS = (2021,)


def frozen_fit_split(
    _data_fold_id: int,
    _additional_data: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Return train/validation years while aliasing unused test to validation."""

    return list(TRAIN_YEARS), list(VALIDATION_YEARS), list(VALIDATION_YEARS)


def validate_forwarded_arguments(arguments: Sequence[str]) -> None:
    """Reject caller attempts to bypass the literal split or test lock."""

    protected_prefixes = (
        "--do_test=",
        "--data.data_fold_id=",
        "--data.additional_data=",
    )
    for argument in arguments:
        if argument.startswith(protected_prefixes):
            raise ValueError(
                f"argument is controlled by the fast-track contract: {argument}"
            )


def load_training_stats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load canonical train-only normalization statistics from an NPZ file."""

    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"training statistics file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"means", "stds", "missing_values"}
        if set(payload.files) != required:
            raise ValueError(
                f"training statistics require keys {sorted(required)}, "
                f"got {sorted(payload.files)}"
            )
        means = np.asarray(payload["means"], dtype=np.float32)
        stds = np.asarray(payload["stds"], dtype=np.float32)
        missing_values = np.asarray(payload["missing_values"], dtype=np.float32)
    if any(array.shape != (23,) for array in (means, stds, missing_values)):
        raise ValueError("training statistics must contain exactly 23 features")
    if not all(np.isfinite(array).all() for array in (means, stds, missing_values)):
        raise ValueError("training statistics must be finite")
    if (stds <= 0).any():
        raise ValueError("training standard deviations must be positive")
    if ((missing_values < 0) | (missing_values > 1)).any():
        raise ValueError("training missing-value rates must be within [0, 1]")
    return means, stds, missing_values


def _install_runtime_contract(
    upstream_root: Path,
    experiment_id: str,
    stats: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    if experiment_id not in {"C00", "C02"}:
        raise ValueError("retained runtime supports C00 and C02 experiments")
    upstream_src = upstream_root / "src"
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(upstream_src))

    datamodule_module = importlib.import_module("dataloader.FireSpreadDataModule")
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    utils_module = importlib.import_module("dataloader.utils")
    datamodule_module.FireSpreadDataModule.split_fires = staticmethod(
        frozen_fit_split
    )

    means, stds, missing_values = stats

    def canonical_training_stats(
        training_years: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tuple(training_years) != TRAIN_YEARS:
            raise ValueError(
                f"fast-track statistics require train years {TRAIN_YEARS}, "
                f"got {tuple(training_years)}"
            )
        result_means = means.copy()
        result_stds = stds.copy()
        degree_and_landcover = [7, 13, 19, 16]
        result_means[degree_and_landcover] = 0
        result_stds[degree_and_landcover] = 1
        return result_means, result_stds, missing_values.copy()

    utils_module.get_means_stds_missing_values = canonical_training_stats
    dataset_module.get_means_stds_missing_values = canonical_training_stats

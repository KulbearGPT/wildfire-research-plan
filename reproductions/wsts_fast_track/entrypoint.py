"""Run one approved fast-track experiment through the pinned upstream CLI."""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .contract import experiment_spec, upstream_arguments, validate_inventory


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

    if experiment_id == "C02":
        models_module = importlib.import_module("models")
        temporal_module = importlib.import_module("models.SMPTempModel")
        models_module.SMPTempModel = temporal_module.SMPTempModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--experiment", choices=("C00", "C02"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    upstream_root = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    train_path = upstream_root / "src" / "train.py"
    if not train_path.is_file():
        raise ValueError(f"pinned upstream train.py is missing: {train_path}")
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    spec = experiment_spec(args.experiment)
    _install_runtime_contract(upstream_root, spec.experiment_id, stats)

    work_root = run_root / "work"
    work_root.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(work_root)
    sys.argv = [
        str(train_path),
        *upstream_arguments(spec, upstream_root, data_root, run_root),
    ]
    runpy.run_path(str(train_path), run_name="__main__")

    import torch

    print(
        f"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

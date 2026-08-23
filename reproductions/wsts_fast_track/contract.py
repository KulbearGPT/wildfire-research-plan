"""Literal contract for the first two Nibi WSTS+ screening experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final


EXPECTED_YEAR_COUNTS: Final[dict[int, int]] = {
    2016: 92,
    2017: 110,
    2018: 176,
    2019: 74,
    2020: 201,
    2021: 156,
    2022: 122,
    2023: 68,
}

MULTI_FEATURES: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    38,
    39,
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_name: str
    data_config: str
    model_class_override: str | None
    n_leading_observations: int
    features_to_keep: tuple[int, ...] | None
    remove_duplicate_features: bool


_EXPERIMENTS: Final[dict[str, ExperimentSpec]] = {
    "C00": ExperimentSpec(
        experiment_id="C00",
        model_name="Res18-UNet",
        data_config="cfgs/data_monotemporal_full_features.yaml",
        model_class_override=None,
        n_leading_observations=1,
        features_to_keep=None,
        remove_duplicate_features=True,
    ),
    "C02": ExperimentSpec(
        experiment_id="C02",
        model_name="Res18-UTAE",
        data_config="cfgs/data_multitemporal_multi_features.yaml",
        model_class_override="models.SMPTempModel",
        n_leading_observations=5,
        features_to_keep=MULTI_FEATURES,
        remove_duplicate_features=False,
    ),
}


def experiment_spec(experiment_id: str) -> ExperimentSpec:
    """Return one approved literal experiment specification."""

    try:
        return _EXPERIMENTS[experiment_id]
    except KeyError as error:
        raise ValueError(f"unknown fast-track experiment: {experiment_id}") from error


def _bool_argument(value: bool) -> str:
    return "true" if value else "false"


def upstream_arguments(
    spec: ExperimentSpec,
    upstream_root: Path,
    data_root: Path,
    run_root: Path,
) -> list[str]:
    """Render the exact LightningCLI arguments for one screening experiment."""

    upstream_root = upstream_root.resolve()
    data_root = data_root.resolve()
    run_root = run_root.resolve()
    features = (
        "null"
        if spec.features_to_keep is None
        else json.dumps(spec.features_to_keep, separators=(",", ":"))
    )
    arguments = [
        f"--config={upstream_root / 'cfgs/unet/res18_monotemporal.yaml'}",
        f"--trainer={upstream_root / 'cfgs/trainer_single_gpu.yaml'}",
        f"--data={upstream_root / spec.data_config}",
        f"--data.data_dir={data_root}",
        "--data.additional_data=true",
        "--data.data_fold_id=0",
        f"--data.features_to_keep={features}",
        f"--data.n_leading_observations={spec.n_leading_observations}",
        "--data.n_leading_observations_test_adjustment=5",
        "--data.return_doy=false",
        "--data.remove_duplicate_features="
        f"{_bool_argument(spec.remove_duplicate_features)}",
        "--data.batch_size=64",
        "--data.num_workers=8",
        "--trainer.max_steps=3000",
        "--trainer.num_sanity_val_steps=0",
        f"--trainer.default_root_dir={run_root / 'work'}",
        "--trainer.logger.init_args.log_model=false",
        "--do_train=true",
        "--do_validate=true",
        "--do_test=false",
        "--do_predict=false",
    ]
    if spec.model_class_override is not None:
        arguments.append(f"--model.class_path={spec.model_class_override}")
    return arguments


def validate_inventory(root: Path) -> dict[str, object]:
    """Validate only the direct eight-year 999-event HDF5 layout."""

    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"WSTS+ data root is not a directory: {root}")
    year_names = {path.name for path in root.iterdir() if path.is_dir()}
    expected_names = {str(year) for year in EXPECTED_YEAR_COUNTS}
    if year_names != expected_names:
        raise ValueError(
            f"WSTS+ year directories differ: expected {sorted(expected_names)}, "
            f"got {sorted(year_names)}"
        )

    counts: dict[int, int] = {}
    total_bytes = 0
    for year, expected_count in EXPECTED_YEAR_COUNTS.items():
        year_root = root / str(year)
        nested_directories = [path for path in year_root.iterdir() if path.is_dir()]
        if nested_directories:
            raise ValueError(f"nested directory is not allowed: {nested_directories[0]}")
        direct_files = sorted(year_root.glob("*.hdf5"))
        recursive_files = sorted(year_root.rglob("*.hdf5"))
        if direct_files != recursive_files:
            raise ValueError(f"nested HDF5 file is not allowed under year {year}")
        counts[year] = len(direct_files)
        if counts[year] != expected_count:
            raise ValueError(
                f"year {year} expected {expected_count} HDF5 files, got {counts[year]}"
            )
        total_bytes += sum(path.stat().st_size for path in direct_files)

    return {
        "file_count": sum(counts.values()),
        "total_bytes": total_bytes,
        "years": counts,
    }

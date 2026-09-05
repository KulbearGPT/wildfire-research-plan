from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reproductions.wsts_fast_track import contract, matrix, runtime


def test_retained_contract_contains_only_t1_c00() -> None:
    spec = contract.experiment_spec("C00")
    assert spec.n_leading_observations == 1
    assert spec.features_to_keep is None
    with pytest.raises(ValueError, match="unknown fast-track experiment"):
        contract.experiment_spec("C02")


def test_c00_renders_the_retained_training_command(tmp_path: Path) -> None:
    upstream = (tmp_path / "upstream").resolve()
    data = (tmp_path / "data").resolve()
    run = (tmp_path / "run").resolve()

    arguments = contract.upstream_arguments(
        contract.experiment_spec("C00"), upstream, data, run, seed=0
    )

    assert "--seed_everything=0" in arguments
    assert "--trainer.max_steps=3000" in arguments
    assert "--data.n_leading_observations=1" in arguments
    assert "--data.features_to_keep=null" in arguments
    assert "--do_test=false" in arguments
    assert not any("SMPTempModel" in argument for argument in arguments)


def test_controlled_scenarios_remain_the_frozen_m00_m07_set() -> None:
    assert tuple(matrix.CORRUPTIONS) == tuple(f"M{i:02d}" for i in range(8))
    assert matrix.CORRUPTIONS["M01"].feature_indices == (22,)
    assert matrix.CORRUPTIONS["M06"].severity == "area=0.25"
    assert matrix.CORRUPTIONS["M07"].severity == "area=0.50"


def test_runtime_uses_frozen_years_and_rejects_split_overrides() -> None:
    assert runtime.frozen_fit_split(0, False) == (
        [2016, 2017, 2018, 2019, 2020],
        [2021],
        [2021],
    )
    with pytest.raises(ValueError, match="fast-track contract"):
        runtime.validate_forwarded_arguments(["--do_test=true"])


def test_training_stats_require_23_finite_features(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    np.savez(
        valid,
        means=np.arange(23, dtype=np.float32),
        stds=np.ones(23, dtype=np.float32),
        missing_values=np.linspace(0.0, 1.0, 23, dtype=np.float32),
    )
    means, stds, missing = runtime.load_training_stats(valid)
    assert means.shape == stds.shape == missing.shape == (23,)

    invalid = tmp_path / "invalid.npz"
    np.savez(
        invalid,
        means=np.zeros(22, dtype=np.float32),
        stds=np.ones(22, dtype=np.float32),
        missing_values=np.zeros(22, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="23 features"):
        runtime.load_training_stats(invalid)

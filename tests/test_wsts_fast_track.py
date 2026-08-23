from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import h5py

from reproductions.wsts_fast_track import compute_stats, contract, entrypoint, matrix


EXPECTED_YEAR_COUNTS = {
    2016: 92,
    2017: 110,
    2018: 176,
    2019: 74,
    2020: 201,
    2021: 156,
    2022: 122,
    2023: 68,
}


def test_follow_on_matrix_has_exact_clean_runs() -> None:
    assert tuple(matrix.CLEAN_RUNS) == (
        "C00-S0-3K",
        "C02-S0-3K",
        "C00-S0-10K",
        "C02-S0-10K",
        "C00-S1-10K",
        "C00-S2-10K",
        "C02-S1-10K",
        "C02-S2-10K",
    )
    promoted = matrix.run_spec("C00-S0-10K")
    assert promoted.experiment_id == "C00"
    assert (promoted.seed, promoted.max_steps) == (0, 10_000)
    assert promoted.prerequisites == ("C00-S0-3K", "C02-S0-3K")
    assert promoted.launch_state == "promotable"
    assert matrix.run_spec("C02-S1-10K").launch_state == "gated"


def test_follow_on_matrix_has_exact_corruptions() -> None:
    assert tuple(matrix.CORRUPTIONS) == tuple(f"M{i:02d}" for i in range(8))
    assert matrix.CORRUPTIONS["M01"].feature_indices == (22,)
    assert matrix.CORRUPTIONS["M03"].feature_indices == tuple(range(5, 12))
    assert matrix.CORRUPTIONS["M04"].feature_indices == tuple(range(17, 22))
    assert matrix.CORRUPTIONS["M06"].severity == "area=0.25"
    assert matrix.CORRUPTIONS["M07"].severity == "area=0.50"
    assert all(
        scenario.implementation_state == "implemented"
        for scenario in matrix.CORRUPTIONS.values()
    )


def test_c00_and_c02_render_the_approved_screening_commands(tmp_path: Path) -> None:
    upstream_root = tmp_path / "upstream"
    data_root = tmp_path / "data"
    c00 = contract.upstream_arguments(
        contract.experiment_spec("C00"),
        upstream_root,
        data_root,
        tmp_path / "c00",
    )
    c02 = contract.upstream_arguments(
        contract.experiment_spec("C02"),
        upstream_root,
        data_root,
        tmp_path / "c02",
    )

    shared = {
        "--trainer.max_steps=3000",
        "--trainer.num_sanity_val_steps=0",
        "--data.batch_size=64",
        "--data.num_workers=8",
        "--do_train=true",
        "--do_validate=true",
        "--do_test=false",
        "--do_predict=false",
    }
    assert shared <= set(c00)
    assert shared <= set(c02)
    assert "--data.n_leading_observations=1" in c00
    assert "--data.features_to_keep=null" in c00
    assert "--model.class_path=models.SMPTempModel" not in c00
    assert "--data.n_leading_observations=5" in c02
    assert (
        "--data.features_to_keep="
        "[0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,38,39]"
    ) in c02
    assert "--model.class_path=models.SMPTempModel" in c02


def test_default_screening_command_remains_exact(tmp_path: Path) -> None:
    upstream_root = (tmp_path / "upstream").resolve()
    data_root = (tmp_path / "data").resolve()
    run_root = (tmp_path / "run").resolve()

    arguments = contract.upstream_arguments(
        contract.experiment_spec("C00"), upstream_root, data_root, run_root
    )

    assert arguments == [
        f"--config={upstream_root / 'cfgs/unet/res18_monotemporal.yaml'}",
        f"--trainer={upstream_root / 'cfgs/trainer_single_gpu.yaml'}",
        f"--data={upstream_root / 'cfgs/data_monotemporal_full_features.yaml'}",
        f"--data.data_dir={data_root}",
        "--data.additional_data=true",
        "--data.data_fold_id=0",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.n_leading_observations_test_adjustment=5",
        "--data.return_doy=false",
        "--data.remove_duplicate_features=true",
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


def test_declared_promotion_renders_seed_and_10k_without_test() -> None:
    spec, max_steps, seed = entrypoint.resolve_run(None, "C02-S0-10K")
    arguments = contract.upstream_arguments(
        spec,
        Path("/upstream"),
        Path("/data"),
        Path("/run"),
        max_steps=max_steps,
        seed=seed,
    )

    assert "--seed_everything=0" in arguments
    assert "--trainer.max_steps=10000" in arguments
    assert "--do_test=false" in arguments
    assert "--model.class_path=models.SMPTempModel" in arguments


@pytest.mark.parametrize(
    ("run_id", "expected_seed"),
    [
        ("C00-S1-10K", 1),
        ("C00-S2-10K", 2),
        ("C02-S1-10K", 1),
        ("C02-S2-10K", 2),
    ],
)
def test_entrypoint_resolves_declared_replication_seed(
    run_id: str, expected_seed: int
) -> None:
    _, max_steps, seed = entrypoint.resolve_run(None, run_id)

    assert max_steps == 10_000
    assert seed == expected_seed


@pytest.mark.parametrize(
    ("experiment_id", "run_id"), [(None, None), ("C00", "C00-S0-10K")]
)
def test_entrypoint_requires_exactly_one_run_selector(
    experiment_id: str | None, run_id: str | None
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        entrypoint.resolve_run(experiment_id, run_id)


def _write_inventory(root: Path) -> None:
    for year, count in EXPECTED_YEAR_COUNTS.items():
        year_root = root / str(year)
        year_root.mkdir(parents=True)
        for index in range(count):
            (year_root / f"event-{index:03d}.hdf5").touch()


def test_inventory_accepts_only_the_active_fixed_999_event_layout(
    tmp_path: Path,
) -> None:
    _write_inventory(tmp_path)

    summary = contract.validate_inventory(tmp_path)

    assert summary["file_count"] == 999
    assert summary["years"] == EXPECTED_YEAR_COUNTS


@pytest.mark.parametrize("mutation", ["missing", "extra-year", "nested"])
def test_inventory_rejects_layout_mutations(tmp_path: Path, mutation: str) -> None:
    _write_inventory(tmp_path)
    if mutation == "missing":
        (tmp_path / "2016" / "event-000.hdf5").unlink()
    elif mutation == "extra-year":
        extra = tmp_path / "2024"
        extra.mkdir()
        (extra / "event.hdf5").touch()
    else:
        nested = tmp_path / "2016" / "nested"
        nested.mkdir()
        (nested / "event.hdf5").touch()

    with pytest.raises(ValueError):
        contract.validate_inventory(tmp_path)


def test_entrypoint_uses_frozen_fit_years_and_withholds_test() -> None:
    assert entrypoint.frozen_fit_split(0, False) == (
        [2016, 2017, 2018, 2019, 2020],
        [2021],
        [2021],
    )


@pytest.mark.parametrize(
    "argument",
    [
        "--do_test=true",
        "--data.data_fold_id=2",
        "--data.additional_data=false",
    ],
)
def test_entrypoint_rejects_test_or_split_overrides(argument: str) -> None:
    with pytest.raises(ValueError, match="fast-track contract"):
        entrypoint.validate_forwarded_arguments([argument])


def test_training_stats_require_23_finite_features(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    np.savez(
        valid,
        means=np.arange(23, dtype=np.float32),
        stds=np.ones(23, dtype=np.float32),
        missing_values=np.linspace(0.0, 1.0, 23, dtype=np.float32),
    )
    means, stds, missing = entrypoint.load_training_stats(valid)
    assert means.shape == stds.shape == missing.shape == (23,)

    invalid = tmp_path / "invalid.npz"
    np.savez(
        invalid,
        means=np.zeros(22, dtype=np.float32),
        stds=np.ones(22, dtype=np.float32),
        missing_values=np.zeros(22, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="23 features"):
        entrypoint.load_training_stats(invalid)


def test_stats_treat_zero_active_fire_as_absent_not_as_detection_hour(
    tmp_path: Path,
) -> None:
    path = tmp_path / "event.hdf5"
    data = np.zeros((1, 23, 1, 4), dtype=np.float32)
    for feature in range(22):
        data[:, feature] = feature + 1
    data[0, 0, 0, 3] = np.nan
    data[0, 22, 0] = [0.0, 6.0, 12.0, 0.0]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)

    means, stds, missing = compute_stats.compute_paths([path])

    assert means[22] == pytest.approx(9.0)
    assert stds[22] == pytest.approx(3.0)
    assert missing[22] == pytest.approx(0.5)
    assert means[0] == pytest.approx(1.0)
    assert missing[0] == pytest.approx(0.25)

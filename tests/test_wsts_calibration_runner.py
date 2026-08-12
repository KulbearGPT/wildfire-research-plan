import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = (
    REPOSITORY_ROOT
    / "reproductions"
    / "wsts_res18_unet_t1"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import run_calibration as calibration_runner  # noqa: E402
from run_calibration import (  # noqa: E402
    acquire_launch_lock,
    assert_source_inventories_identical,
    build_official_command,
    build_source_inventory,
    create_unique_run_directory,
    measure_wall_seconds_from_markers,
    next_gpu_sample_deadline,
    parse_epoch_boundaries,
    measure_validation_seconds,
    observe_process,
    parse_progress,
    parse_peak_allocated,
    summarize_gpu_samples,
    summarize_epoch_aware_timing,
    summarize_timing,
    validate_effective_command,
    validate_effective_runtime_config,
    validate_import_only_evidence,
    validate_pretraining_failure_recovery,
    validate_runtime_safety_patch,
    validate_smoke_evidence,
    validate_success_evidence,
    validate_train_validation_smoke_evidence,
    validate_worker_failure_recovery,
    _write_calibration_summary_csv,
    _write_step_timing_csv,
)


def _event(seconds: float, text: str, stream: str = "stdout") -> str:
    return json.dumps(
        {"seconds": seconds, "stream": stream, "text": text},
        sort_keys=True,
        separators=(",", ":"),
    )


def _progress_events(final_step: int = 500) -> str:
    return "\n".join(
        [
            _event(2.0, "\rEpoch 0:   0%|          | 0/400 [00:00<?, ?it/s]"),
            _event(2.4, "\rEpoch 0:   0%|          | 1/400 [00:00<02:39, train_loss_step=0.25]"),
            _event(161.6, "\rEpoch 0: 100%|##########| 400/400 [02:39<00:00, train_loss_epoch=0.20]"),
            _event(162.0, "\rEpoch 1:   0%|          | 0/400 [00:00<?, ?it/s]"),
            _event(201.6, f"\rEpoch 1:  25%|##5       | {final_step - 400}/400 [00:39<01:59, train_loss_step=0.19]"),
        ]
    )


def _expected_command(tmp_path: Path) -> list[str]:
    upstream_root = (tmp_path / "WildfireSpreadTS").resolve()
    return build_official_command(
        python_executable=(tmp_path / "environment" / "python.exe").resolve(),
        entrypoint=(SCRIPTS_DIR / "official_entrypoint.py").resolve(),
        upstream_root=upstream_root,
        data_root=Path(r"D:\WildFire Project\data\hdf5"),
        run_directory=(tmp_path / "artifacts" / "run-001").resolve(),
    )


def test_parse_progress_collapses_duplicate_carriage_return_updates() -> None:
    text = "\n".join(
        [
            _event(10.0, "\rEpoch 0:   0%|          | 0/2 [00:00<?, ?it/s]"),
            _event(10.5, "\rEpoch 0:  50%|#####     | 1/2 [00:00<00:00, train_loss_step=0.3]"),
            _event(10.7, "\rEpoch 0:  50%|#####     | 1/2 [00:00<00:00, train_loss_step=0.2]"),
            _event(11.0, "\rEpoch 0: 100%|##########| 2/2 [00:01<00:00, train_loss_epoch=0.2]"),
            _event(12.0, "\rValidation DataLoader 0: 100%|##########| 1/1 [00:00<00:00]"),
            _event(12.2, "\rEpoch 1:   0%|          | 0/2 [00:00<?, ?it/s]"),
            _event(12.7, "\rEpoch 1:  50%|#####     | 1/2 [00:00<00:00, train_loss_step=0.1]"),
        ]
    )

    assert parse_progress(text) == [
        (0, 10.0),
        (1, 10.5),
        (2, 11.0),
        (3, 12.7),
    ]


def test_parse_progress_ignores_completed_epoch_teardown_reset() -> None:
    text = "\n".join(
        [
            _event(10.0, "\rEpoch 0: 100%|##########| 2/2 [00:01<00:00]"),
            _event(11.0, "\rEpoch 0:   0%|          | 0/2 [00:00<?, ?it/s]"),
            _event(12.0, "\rEpoch 1:   0%|          | 0/2 [00:00<?, ?it/s]"),
            _event(13.0, "\rEpoch 1:  50%|#####     | 1/2 [00:01<00:01]"),
        ]
    )

    assert parse_progress(text) == [(2, 10.0), (3, 13.0)]


@pytest.mark.parametrize(
    "text",
    [
        "\n".join(
            [
                _event(1.0, "\rEpoch 0:  50%|#####| 1/2 [00:00<00:00]"),
                _event(2.0, "\rEpoch 0:   0%|     | 0/2 [00:00<?, ?it/s]"),
            ]
        ),
        "\n".join(
            [
                _event(1.0, "\rEpoch 1:  50%|#####| 1/2 [00:00<00:00]"),
                _event(2.0, "\rEpoch 0: 100%|#####| 2/2 [00:01<00:00]"),
            ]
        ),
    ],
)
def test_parse_progress_rejects_optimizer_or_epoch_regression(text: str) -> None:
    with pytest.raises(ValueError, match="regress"):
        parse_progress(text)


@pytest.mark.parametrize("seconds", [math.nan, math.inf, -math.inf])
def test_parse_progress_rejects_non_finite_timestamps(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        parse_progress(_event(seconds, "\rEpoch 0: 0%| | 0/2 [00:00<?, ?it/s]"))


def test_summarize_timing_excludes_steps_zero_through_forty_nine() -> None:
    summary = summarize_timing(
        [(0, 0.0), (49, 4.9), (50, 5.2), (51, 5.6), (52, 6.0)],
        wall_seconds=100.0,
        startup_seconds=10.0,
        validation_seconds=2.0,
    )

    assert summary["median_step_seconds"] == pytest.approx(0.4)
    assert summary["instantaneous_samples_per_second"] == pytest.approx(160.0)
    assert summary["compute_only_10000_seconds"] == pytest.approx(4000.0)
    assert "extrapolated_10000_total_seconds" not in summary
    assert summary["wall_seconds"] == 100.0
    assert summary["startup_seconds"] == 10.0
    assert summary["validation_seconds"] == 2.0


def test_epoch_aware_summary_models_full_cycles_partial_steps_and_wall_linear() -> None:
    summary = summarize_epoch_aware_timing(
        epoch_boundaries=[
            (0, 1, 100.0, 121),
            (1, 122, 302.0, 121),
            (2, 243, 506.0, 121),
            (3, 364, 709.0, 121),
        ],
        startup_seconds=100.0,
        median_step_seconds=0.1,
        p25_step_seconds=0.08,
        p75_step_seconds=0.12,
        wall_seconds=1000.0,
    )

    assert summary["epoch_cycle_seconds"] == [202.0, 204.0, 203.0]
    assert summary["projected_full_epoch_cycles"] == 82
    assert summary["projected_partial_steps"] == 78
    assert summary["epoch_aware_10000_central_seconds"] == pytest.approx(
        100.0 + 82 * 203.0 + 78 * 0.1
    )
    assert summary["naive_wall_linear_10000_seconds"] == pytest.approx(20_000.0)
    assert summary["end_to_end_samples_per_second"] == pytest.approx(32.0)


def test_parse_epoch_boundaries_uses_first_completed_step_of_each_epoch() -> None:
    events = "\n".join(
        [
            _event(1.0, "Epoch 0: 0%| | 0/2"),
            _event(2.0, "Epoch 0: 50%| | 1/2"),
            _event(3.0, "Epoch 0: 50%| | 1/2, train_loss_step=0.1"),
            _event(4.0, "Epoch 0: 100%| | 2/2"),
            _event(5.0, "Epoch 0: 0%| | 0/2"),
            _event(6.0, "Epoch 1: 0%| | 0/2"),
            _event(7.0, "Epoch 1: 50%| | 1/2"),
        ]
    )

    assert parse_epoch_boundaries(events) == [(0, 1, 2.0, 2), (1, 3, 7.0, 2)]


def test_measure_wall_seconds_uses_started_utc_and_exit_marker_mtime() -> None:
    assert measure_wall_seconds_from_markers(
        "1970-01-01T00:00:01.250000+00:00", 2_750_000_000
    ) == pytest.approx(1.5)


def test_calibration_csv_is_one_summary_row_and_steps_have_separate_file(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "calibration.csv"
    step_path = tmp_path / "step-timing.csv"
    _write_calibration_summary_csv(
        summary_path,
        {"optimizer_steps": 500, "wall_seconds": 1000.0, "note": "timing only"},
    )
    _write_step_timing_csv(step_path, [(0, 1.0), (1, 1.1), (2, 1.2)])

    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with step_path.open(encoding="utf-8", newline="") as handle:
        step_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 1
    assert summary_rows[0]["optimizer_steps"] == "500"
    assert len(step_rows) == 3


@pytest.mark.parametrize(
    ("samples", "wall_seconds", "startup_seconds", "validation_seconds"),
    [
        ([(0, 0.0), (1, math.nan), (50, 1.0)], 2.0, 0.0, 0.0),
        ([(0, 0.0), (1, 1.0), (50, 0.5)], 2.0, 0.0, 0.0),
        ([(0, 0.0), (50, 1.0)], math.inf, 0.0, 0.0),
        ([(0, 0.0), (50, 1.0)], 2.0, -1.0, 0.0),
        ([(0, 0.0), (50, 1.0)], 2.0, 0.0, math.nan),
    ],
)
def test_summarize_timing_rejects_regressions_and_non_finite_values(
    samples: list[tuple[int, float]],
    wall_seconds: float,
    startup_seconds: float,
    validation_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="finite|regress|non-negative"):
        summarize_timing(
            samples,
            wall_seconds=wall_seconds,
            startup_seconds=startup_seconds,
            validation_seconds=validation_seconds,
        )


def test_build_official_command_uses_only_the_three_configs_and_allowlist(
    tmp_path: Path,
) -> None:
    command = _expected_command(tmp_path)

    validate_effective_command(command, command)
    assert any(argument.endswith("cfgs/unet/res18_monotemporal.yaml") for argument in command)
    assert any(argument.endswith("cfgs/trainer_single_gpu.yaml") for argument in command)
    assert any(argument.endswith("cfgs/data_monotemporal_full_features.yaml") for argument in command)
    assert "--data.data_fold_id=2" in command
    assert "--data.features_to_keep=null" in command
    assert "--data.n_leading_observations=1" in command
    assert "--data.remove_duplicate_features=true" in command
    assert "--data.num_workers=64" in command
    assert "--trainer.max_steps=500" in command
    assert "--do_test=false" in command
    assert not any("do_predict" in argument or "do_validate" in argument for argument in command)


def test_worker_recovery_command_changes_only_num_workers_and_output_directory(
    tmp_path: Path,
) -> None:
    original = _expected_command(tmp_path)
    recovered = build_official_command(
        python_executable=(tmp_path / "environment" / "python.exe").resolve(),
        entrypoint=(SCRIPTS_DIR / "official_entrypoint.py").resolve(),
        upstream_root=(tmp_path / "WildfireSpreadTS").resolve(),
        data_root=Path(r"D:\WildFire Project\data\hdf5"),
        run_directory=(tmp_path / "artifacts" / "run-002").resolve(),
        num_workers=8,
    )

    normalized_original = [
        "--trainer.default_root_dir=<run>"
        if argument.startswith("--trainer.default_root_dir=")
        else argument
        for argument in original
    ]
    normalized_recovered = [
        "--trainer.default_root_dir=<run>"
        if argument.startswith("--trainer.default_root_dir=")
        else "--data.num_workers=64"
        if argument == "--data.num_workers=8"
        else argument
        for argument in recovered
    ]
    assert normalized_recovered == normalized_original
    assert "--data.num_workers=8" in recovered


def test_effective_runtime_config_requires_official_dynamic_fold2_positive_weight(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n"
        "  class_path: models.SMPModel\n"
        "  init_args:\n"
        "    pos_class_weight: 608.4653828020165\n",
        encoding="utf-8",
    )

    assert validate_effective_runtime_config(config) == {
        "effective_pos_class_weight": 608.4653828020165,
        "source_yaml_pos_class_weight": 236.0,
        "official_dynamic_override": True,
    }
    config.write_text(
        config.read_text(encoding="utf-8").replace("608.4653828020165", "236"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive-class weight"):
        validate_effective_runtime_config(config)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("--do_test=false", "--do_test=true"),
        ("--data.data_fold_id=2", "--data.data_fold_id=1"),
        ("--data.features_to_keep=null", "--data.features_to_keep=[0,1]"),
        ("--trainer.max_steps=500", "--trainer.max_steps=501"),
        ("--data.n_leading_observations=1", "--data.n_leading_observations=5"),
        ("--data.num_workers=64", "--optimizer.init_args.lr=0.01"),
        ("--data.num_workers=64", "predict"),
        ("--data.num_workers=64", "test"),
    ],
)
def test_validate_effective_command_rejects_actions_or_scientific_changes(
    tmp_path: Path, old: str, new: str
) -> None:
    expected = _expected_command(tmp_path)
    changed = [new if argument == old else argument for argument in expected]

    with pytest.raises(ValueError, match="allowlisted|effective command"):
        validate_effective_command(changed, expected)


@pytest.mark.parametrize(
    ("events", "output", "exit_code", "peak_bytes", "message"),
    [
        (_progress_events(499), "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=0.19", 0, 1024, "exactly 500"),
        (_progress_events(), "train_loss_step=0.19", 0, 1024, "stop evidence"),
        (_progress_events(), "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=nan", 0, 1024, "finite"),
        (_progress_events(), "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=0.19", 1, 1024, "exit code"),
        (_progress_events(), "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=0.19\nTesting DataLoader 0", 0, 1024, "test/predict"),
        (_progress_events(), "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=0.19\nCUDA out of memory", 0, 1024, "OOM"),
        (_progress_events(), "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=0.19", 0, math.inf, "peak")
    ],
)
def test_validate_success_evidence_rejects_incomplete_or_unsafe_run(
    events: str,
    output: str,
    exit_code: int,
    peak_bytes: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_success_evidence(events, output, exit_code, peak_bytes)


def test_validate_success_evidence_accepts_exactly_step_500() -> None:
    samples = validate_success_evidence(
        _progress_events(),
        "`Trainer.fit` stopped: `max_steps=500` reached.\ntrain_loss_step=0.19",
        0,
        7_654_321,
    )

    assert samples[-1][0] == 500


def test_source_inventory_detects_size_or_mtime_changes(tmp_path: Path) -> None:
    for year in (2018, 2019, 2020, 2021):
        year_dir = tmp_path / str(year)
        year_dir.mkdir()
        (year_dir / "fire.hdf5").write_bytes(b"original")
    before = build_source_inventory(tmp_path)

    (tmp_path / "2020" / "fire.hdf5").write_bytes(b"changed-size")
    after = build_source_inventory(tmp_path)

    with pytest.raises(ValueError, match="source data changed"):
        assert_source_inventories_identical(before, after)


def test_launch_lock_is_atomic_and_refuses_a_second_attempt(tmp_path: Path) -> None:
    lock_path = tmp_path / "calibration-launch.lock.json"
    payload = {"run_directory": "run-001", "command_sha256": "abc123"}

    acquire_launch_lock(lock_path, payload)

    assert json.loads(lock_path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        acquire_launch_lock(lock_path, payload)


def test_official_entrypoint_executes_unchanged_train_in_same_process(
    tmp_path: Path,
) -> None:
    upstream_root = tmp_path / "WildfireSpreadTS"
    upstream_src = upstream_root / "src"
    upstream_models = upstream_src / "models"
    upstream_models.mkdir(parents=True)
    (upstream_src / "__init__.py").write_text("", encoding="utf-8")
    (upstream_models / "__init__.py").write_text("from .required import VALUE\n", encoding="utf-8")
    (upstream_models / "required.py").write_text("VALUE = 7\n", encoding="utf-8")
    (upstream_src / "train.py").write_text(
        "import json, os, sys\n"
        "from models import VALUE\n"
        "print('TRAIN_PROCESS=' + json.dumps({'pid': os.getpid(), 'path': sys.path, 'value': VALUE}))\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS_DIR / "official_entrypoint.py"),
            "--upstream-root",
            str(upstream_root),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(timeout=30)

    assert process.returncode == 0, stderr
    process_line = next(line for line in stdout.splitlines() if line.startswith("TRAIN_PROCESS="))
    observed = json.loads(process_line.removeprefix("TRAIN_PROCESS="))
    assert observed["pid"] == process.pid
    assert observed["value"] == 7
    assert str(upstream_src.resolve()) in observed["path"]
    assert str(upstream_root.resolve()) not in observed["path"]
    assert "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=" in stdout


def test_runtime_patch_removes_only_seven_unused_exports_and_keeps_res18_imports() -> None:
    patch_path = (
        REPOSITORY_ROOT
        / "reproductions"
        / "wsts_res18_unet_t1"
        / "patches"
        / "res18_import_scope.patch"
    )
    upstream_root = SCRIPTS_DIR.parent / ".local" / "WildfireSpreadTS"

    metadata = validate_runtime_safety_patch(patch_path, upstream_root)

    assert metadata["affected_files"] == ["src/models/__init__.py"]
    assert metadata["deleted_lines"] == [
        "from .UTAELightning import UTAELightning",
        "from .SwinUnetLightning import SwinUnetLightning",
        "from .SwinUnetTempLightning import SwinUnetTempLightning",
        "from .UTAELightningDumb import UTAELightningDumb",
        "from .TransUnetLightning import TransUnetLightning",
        "from .SMPTempModel import SMPTempModel ",
        "from .SegFormerLightning import SegFormerLightning",
    ]
    assert metadata["added_lines"] == []
    assert metadata["required_exports_unchanged"] == [
        "from .BaseModel import BaseModel",
        "from .ConvLSTMLightning import ConvLSTMLightning",
        "from .LogisticRegression import LogisticRegression",
        "from .SMPModel import SMPModel",
    ]


@pytest.mark.parametrize(
    ("exit_code", "stdout", "stderr"),
    [
        (1, "", "ModuleNotFoundError: missing architecture"),
        (0, "Epoch 0: 1/10", ""),
        (0, "`Trainer.fit` stopped: `max_steps=500` reached.", ""),
        (0, "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1", ""),
    ],
)
def test_import_only_gate_rejects_failure_or_training_evidence(
    exit_code: int, stdout: str, stderr: str
) -> None:
    with pytest.raises(ValueError, match="import-only"):
        validate_import_only_evidence(exit_code, stdout, stderr)


def test_import_only_gate_accepts_clean_help_output() -> None:
    validate_import_only_evidence(0, "usage: train.py [options]\n", "pkg_resources warning\n")


def test_create_unique_run_directory_never_reuses_a_name(tmp_path: Path) -> None:
    first = create_unique_run_directory(tmp_path)
    second = create_unique_run_directory(tmp_path)

    assert first.parent == tmp_path.resolve()
    assert second.parent == tmp_path.resolve()
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_validate_smoke_evidence_requires_the_passing_64_worker_training_gate() -> None:
    smoke = {
        "status": "pass",
        "purpose": "fold-2 real-data loader smoke only; no training performed",
        "upstream_commit": "ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad",
        "weights_revision": "acf70a37394849f4ec8d108a51d6f4325a554d0a",
        "fold_mapping": {"train": [2018, 2020], "validation": [2019], "test": [2021]},
        "inventory": {"2018": 176, "2019": 74, "2020": 201, "2021": 156, "total": 607},
        "batch_size": 64,
        "num_workers": 64,
        "temporal_steps": 1,
        "model_channels": 40,
        "crop_side_length": 128,
        "test_loader_called": False,
        "test_sample_loaded": False,
        "training_started": False,
    }

    validate_smoke_evidence(smoke)
    with pytest.raises(ValueError, match="smoke"):
        validate_smoke_evidence({**smoke, "num_workers": 8})
    with pytest.raises(ValueError, match="smoke"):
        validate_smoke_evidence({**smoke, "test_loader_called": True})
    with pytest.raises(ValueError, match="smoke"):
        validate_smoke_evidence({**smoke, "purpose": "scientific test"})


def test_validate_train_validation_smoke_requires_workers8_both_batches_and_ram() -> None:
    boundary = {
        "loader_input_shape": [64, 1, 40, 128, 128],
        "loader_target_shape": [64, 128, 128],
        "model_input_shape": [64, 40, 128, 128],
        "model_target_shape": [64, 1, 128, 128],
        "temporal_steps": 1,
        "model_channels": 40,
        "crop_side_length": 128,
        "inputs_finite": True,
        "targets_binary": True,
        "next_day_target_distinct": True,
    }
    smoke = {
        "status": "pass",
        "purpose": "fold-2 real-data train+validation loader smoke only; no training performed",
        "upstream_commit": "ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad",
        "weights_revision": "acf70a37394849f4ec8d108a51d6f4325a554d0a",
        "fold_mapping": {"train": [2018, 2020], "validation": [2019], "test": [2021]},
        "inventory": {"2018": 176, "2019": 74, "2020": 201, "2021": 156, "total": 607},
        "batch_size": 64,
        "num_workers": 8,
        "train": boundary,
        "validation": boundary,
        "validation_loader_called": True,
        "validation_sample_loaded": True,
        "test_loader_called": False,
        "test_sample_loaded": False,
        "training_started": False,
        "ram": {
            "ram_total_bytes": 64_000,
            "ram_available_before_bytes": 40_000,
            "ram_available_after_bytes": 30_000,
            "ram_minimum_available_bytes": 10_000,
            "ram_peak_used_bytes": 54_000,
            "ram_sample_count": 3,
        },
    }

    validate_train_validation_smoke_evidence(smoke)
    with pytest.raises(ValueError, match="train.validation smoke"):
        validate_train_validation_smoke_evidence({**smoke, "num_workers": 64})
    with pytest.raises(ValueError, match="train.validation smoke"):
        validate_train_validation_smoke_evidence(
            {**smoke, "validation_sample_loaded": False}
        )


def test_measure_validation_seconds_uses_only_complete_non_sanity_intervals() -> None:
    events = "\n".join(
        [
            _event(1.0, "\rSanity Checking DataLoader 0: 0%| | 0/1"),
            _event(2.0, "\rSanity Checking DataLoader 0: 100%|#| 1/1"),
            _event(10.0, "\rEpoch 0: 100%|#| 400/400"),
            _event(12.0, "\rValidation DataLoader 0: 0%| | 0/2"),
            _event(13.0, "\rValidation DataLoader 0: 50%|#| 1/2"),
            _event(14.0, "\rValidation DataLoader 0: 100%|#| 2/2"),
            _event(15.0, "\rEpoch 1: 0%| | 0/400"),
        ]
    )

    assert measure_validation_seconds(events) == (2.0, True)
    assert measure_validation_seconds(_event(13.0, "\rValidation DataLoader 0: 50%|#| 1/2")) == (0.0, False)


@pytest.mark.parametrize(
    "output",
    [
        "no sentinel",
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1\nWSTS_OBSERVER_PEAK_ALLOCATED_BYTES=2",
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=nan",
    ],
)
def test_parse_peak_allocated_rejects_missing_duplicate_or_non_finite_sentinel(
    output: str,
) -> None:
    with pytest.raises(ValueError, match="sentinel"):
        parse_peak_allocated(output)


def test_parse_peak_allocated_accepts_one_positive_integer_sentinel() -> None:
    assert parse_peak_allocated("log\nWSTS_OBSERVER_PEAK_ALLOCATED_BYTES=7654321\n") == 7_654_321


def test_summarize_gpu_samples_reports_gpu_and_child_process_peaks() -> None:
    rows = [
        {"observer_seconds": "0", "memory_used_mib": "1000", "utilization_gpu_percent": "10", "child_memory_mib": "900"},
        {"observer_seconds": "1", "memory_used_mib": "2500", "utilization_gpu_percent": "90", "child_memory_mib": "2300"},
        {"observer_seconds": "2", "memory_used_mib": "2000", "utilization_gpu_percent": "50", "child_memory_mib": "1900"},
    ]

    assert summarize_gpu_samples(rows) == {
        "gpu_sample_count": 3.0,
        "peak_gpu_used_mib": 2500.0,
        "peak_child_process_mib": 2300.0,
        "child_process_memory_available": True,
        "gpu_interval_count": 2,
        "gpu_interval_mean_seconds": 1.0,
        "gpu_interval_median_seconds": 1.0,
        "gpu_observed_effective_hz": 1.0,
        "gpu_peak_is_observed_sample_max": True,
        "gpu_peak_may_miss_between_sample_transients": True,
        "gpu_utilization_min_percent": 10.0,
        "gpu_utilization_median_percent": 50.0,
        "gpu_utilization_max_percent": 90.0,
    }


def test_summarize_gpu_samples_marks_zero_child_attribution_unavailable() -> None:
    summary = summarize_gpu_samples(
        [
            {"observer_seconds": "0", "memory_used_mib": "1000", "utilization_gpu_percent": "5", "child_memory_mib": "0"},
            {"observer_seconds": "1", "memory_used_mib": "2000", "utilization_gpu_percent": "10", "child_memory_mib": "0"},
        ]
    )

    assert summary["peak_child_process_mib"] is None
    assert summary["child_process_memory_available"] is False


def test_absolute_gpu_deadline_does_not_accumulate_query_duration() -> None:
    assert next_gpu_sample_deadline(100.0, 100.2) == pytest.approx(101.0)
    assert next_gpu_sample_deadline(101.0, 101.4) == pytest.approx(102.0)
    assert next_gpu_sample_deadline(102.0, 104.2) == pytest.approx(105.0)


def test_nvidia_sampler_parses_real_seven_column_gpu_row_and_child_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        [
            subprocess.CompletedProcess(
                [], 0, "0, NVIDIA GeForce RTX 3090, 24576, 3322, 21005, 8, 55.81\n", ""
            ),
            subprocess.CompletedProcess([], 0, "1234, 2300\n9999, 100\n", ""),
        ]
    )

    monkeypatch.setattr(calibration_runner, "_run_checked", lambda command: next(results))

    sample = dict(calibration_runner._sample_nvidia_smi(1234, 1.25))
    assert str(sample.pop("timestamp_utc")).endswith("+00:00")
    assert sample == {
        "observer_seconds": "1.250000000",
        "gpu_index": "0",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "memory_total_mib": "24576",
        "memory_used_mib": "3322",
        "memory_free_mib": "21005",
        "utilization_gpu_percent": "8",
        "power_draw_w": "55.81",
        "child_memory_mib": "2300.000",
    }


@pytest.mark.parametrize(
    ("memory_free_mib", "accepted"),
    [(19_999, False), (20_000, True)],
)
def test_gpu_preflight_default_preserves_twenty_gibibyte_scientific_gate(
    monkeypatch: pytest.MonkeyPatch,
    memory_free_mib: int,
    accepted: bool,
) -> None:
    monkeypatch.setattr(
        calibration_runner,
        "_run_checked",
        lambda _command: subprocess.CompletedProcess(
            [],
            0,
            f"0, NVIDIA GeForce RTX 3090, 24576, {memory_free_mib}, 0, 610.74\n",
            "",
        ),
    )

    if not accepted:
        with pytest.raises(ValueError, match="requires at least 20000 MiB"):
            calibration_runner._gpu_preflight()
        return

    assert calibration_runner._gpu_preflight()["memory_free_mib"] == 20_000.0


def test_observe_process_preserves_raw_streams_and_records_pid_without_retry(
    tmp_path: Path,
) -> None:
    child_script = tmp_path / "child.py"
    attempts = tmp_path / "attempts.txt"
    child_script.write_text(
        "import pathlib, sys\n"
        f"p = pathlib.Path({str(attempts)!r})\n"
        "p.write_text((p.read_text() if p.exists() else '') + 'launch\\n')\n"
        "sys.stdout.buffer.write(b'before\\rEpoch 0: 50%|#| 1/2\\r')\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'problem\\n')\n"
        "sys.stderr.buffer.flush()\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    result = observe_process(
        [sys.executable, str(child_script)],
        cwd=tmp_path,
        environment=dict(os.environ),
        run_directory=run_directory,
        gpu_sampler=None,
    )

    assert result["exit_code"] == 7
    assert result["pid"] > 0
    assert attempts.read_text(encoding="utf-8") == "launch\n"
    assert (run_directory / "stdout.log").read_bytes() == b"before\rEpoch 0: 50%|#| 1/2\r"
    assert (run_directory / "stderr.log").read_bytes() == b"problem\n"
    assert (run_directory / "exit-code.txt").read_text(encoding="utf-8") == "7\n"
    started = json.loads((run_directory / "started.json").read_text(encoding="utf-8"))
    assert started["pid"] == result["pid"]
    events = (run_directory / "stream-events.jsonl").read_text(encoding="utf-8")
    assert "Epoch 0: 50%" in events
    assert "problem" in events


def test_recovery_gate_accepts_only_linked_failure_before_trainer_or_optimizer(
    tmp_path: Path,
) -> None:
    failed_run = (tmp_path / "failed-run").resolve()
    failed_run.mkdir()
    command = ["fixed-python", "official_entrypoint.py", "--do_test=false"]
    (failed_run / "started.json").write_text(
        json.dumps({"command": command, "pid": 999_999, "started_utc": "2026-08-09T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (failed_run / "exit-code.txt").write_text("1\n", encoding="utf-8")
    (failed_run / "stdout.log").write_bytes(b"")
    stderr = "ModuleNotFoundError:\nNo module named 'src'\n"
    (failed_run / "stderr.log").write_text(stderr, encoding="utf-8")
    (failed_run / "stream-events.jsonl").write_text(_event(1.0, stderr, "stderr") + "\n", encoding="utf-8")
    (failed_run / "failure.json").write_text(
        json.dumps({"status": "fail", "training_retry_performed": False}), encoding="utf-8"
    )
    launch_lock = tmp_path / "global-launch.lock.json"
    launch_lock.write_text(
        json.dumps({"run_directory": str(failed_run), "command": command, "attempt": 1}),
        encoding="utf-8",
    )

    evidence = validate_pretraining_failure_recovery(failed_run, launch_lock)

    assert evidence["failed_child_pid"] == 999_999
    assert evidence["failed_exit_code"] == 1
    assert evidence["optimizer_steps_observed"] == 0
    assert evidence["pretraining_failure_recovery"] == 1

    (failed_run / "stdout.log").write_text("Epoch 0: 1/10", encoding="utf-8")
    with pytest.raises(ValueError, match="training progress|optimizer"):
        validate_pretraining_failure_recovery(failed_run, launch_lock)


def test_worker_recovery_gate_requires_linked_allocator_failure_before_optimizer(
    tmp_path: Path,
) -> None:
    import_failed_run = (tmp_path / "import-failed-run").resolve()
    import_failed_run.mkdir()
    corrected_run = (tmp_path / "workers64-failed-run").resolve()
    corrected_run.mkdir()
    command = ["fixed-python", "official_entrypoint.py", "--data.num_workers=64"]
    (corrected_run / "started.json").write_text(
        json.dumps({"command": command, "pid": 999_998}), encoding="utf-8"
    )
    (corrected_run / "exit-code.txt").write_text("1\n", encoding="utf-8")
    (corrected_run / "stdout.log").write_text(
        "Sanity Checking DataLoader 0: 0it [00:00, ?it/s]", encoding="utf-8"
    )
    stderr = (
        "RuntimeError: Caught RuntimeError in DataLoader worker process 0.\n"
        "RuntimeError: [enforce fail at alloc_cpu.cpp:81] "
        "DefaultCPUAllocator: not enough memory: you tried to allocate 1572864 bytes.\n"
    )
    (corrected_run / "stderr.log").write_text(stderr, encoding="utf-8")
    (corrected_run / "stream-events.jsonl").write_text(
        _event(1.0, "Sanity Checking DataLoader 0: 0it")
        + "\n"
        + _event(2.0, stderr, "stderr")
        + "\n",
        encoding="utf-8",
    )
    (corrected_run / "failure.json").write_text(
        json.dumps({"status": "fail", "training_retry_performed": False}),
        encoding="utf-8",
    )
    recovery_lock = tmp_path / "pretraining-recovery.lock.json"
    recovery_lock.write_text(
        json.dumps(
            {
                "run_directory": str(corrected_run),
                "failed_run_directory": str(import_failed_run),
                "command": command,
                "attempt": 2,
                "pretraining_failure_recovery": 1,
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_worker_failure_recovery(corrected_run, recovery_lock)

    assert evidence["failed_child_pid"] == 999_998
    assert evidence["optimizer_steps_observed"] == 0
    assert evidence["worker_recovery"] == 1
    assert evidence["lineage_import_failed_run"] == str(import_failed_run)

    (corrected_run / "stdout.log").write_text(
        "Sanity Checking DataLoader 0: 0it\nEpoch 0: 1/400", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="optimizer|training progress"):
        validate_worker_failure_recovery(corrected_run, recovery_lock)

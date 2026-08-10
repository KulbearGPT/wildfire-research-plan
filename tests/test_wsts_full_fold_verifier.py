import json
import sys
import hashlib
import os
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_full_fold as verifier  # noqa: E402
from verify_full_fold import verify_recorded_command, verify_result_summary  # noqa: E402


def test_independent_full_verifier_requires_exact_command_and_hash() -> None:
    command = ["python.exe", "entrypoint.py", "--trainer.max_steps=10000", "--do_test=true"]
    payloads = {
        "launch": {"command": command, "command_sha256": "abc"},
        "run_lock": {"command": command, "command_sha256": "abc"},
        "preflight": {"command": command, "command_sha256": "abc"},
        "started": {"command": command, "command_sha256": "abc"},
        "effective": {"command": command, "command_sha256": "abc"},
        "result": {"command_sha256": "abc"},
    }

    verify_recorded_command(payloads, command, "abc")
    del payloads["started"]["command_sha256"]
    with pytest.raises(ValueError, match="started.command_sha256"):
        verify_recorded_command(payloads, command, "abc")


def test_independent_full_verifier_requires_run_lock_and_typed_lineage() -> None:
    command = ["python.exe", "entrypoint.py", "--trainer.max_steps=10000", "--do_test=true"]
    payloads = {
        "launch": {"command": command, "command_sha256": "abc"},
        "run_lock": {"command": command, "command_sha256": "abc"},
        "preflight": {"command": command, "command_sha256": "abc"},
        "started": {"command": command, "command_sha256": "abc"},
        "effective": {"command": command, "command_sha256": "abc"},
        "result": {"command_sha256": "abc"},
    }

    without_run_lock = dict(payloads)
    del without_run_lock["run_lock"]
    with pytest.raises(ValueError) as error:
        verify_recorded_command(without_run_lock, command, "abc")
    assert str(error.value) == "run_lock marker is missing"

    payloads["started"]["command"] = "not-a-list"
    with pytest.raises(ValueError) as error:
        verify_recorded_command(payloads, command, "abc")
    assert str(error.value) == "started.command must be a list of strings"


def test_expected_full_command_is_built_independently_from_artifacts(tmp_path: Path) -> None:
    builder = getattr(verifier, "build_expected_full_command", None)
    assert builder is not None, "independent full command builder is required"
    run = (tmp_path / "fold2-full-run").resolve()
    derived = (tmp_path / "WildfireSpreadTS-res18-runtime").resolve()

    command = builder(run, derived)

    assert command == [
        r"D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe",
        str((SCRIPTS_DIR / "official_entrypoint.py").resolve()),
        "--upstream-root",
        str(derived),
        f"--config={(derived / 'cfgs' / 'unet' / 'res18_monotemporal.yaml').as_posix()}",
        f"--trainer={(derived / 'cfgs' / 'trainer_single_gpu.yaml').as_posix()}",
        f"--data={(derived / 'cfgs' / 'data_monotemporal_full_features.yaml').as_posix()}",
        r"--data.data_dir=D:\WildFire Project\data\hdf5",
        "--data.data_fold_id=2",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        "--trainer.max_steps=10000",
        f"--trainer.default_root_dir={run}",
        "--do_test=true",
    ]


def test_full_provenance_copy_validation_rejects_self_consistent_tamper(
    tmp_path: Path,
) -> None:
    validate = getattr(verifier, "validate_provenance_copies", None)
    assert validate is not None, "full provenance-copy validator is required"
    source = tmp_path / "authoritative.yaml"
    source.write_bytes(b"official: true\n")
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    copied = provenance / "authoritative.yaml"
    copied.write_bytes(source.read_bytes())
    copied_diff = provenance / "derived-runtime.diff"
    copied_diff.write_text("diff --git a/file b/file\n", encoding="utf-8")
    recorded = {
        "authoritative.yaml": hashlib.sha256(copied.read_bytes()).hexdigest(),
        "derived-runtime.diff": hashlib.sha256(copied_diff.read_bytes()).hexdigest(),
    }

    validate(
        provenance,
        recorded,
        {"authoritative.yaml": source},
        "diff --git a/file b/file\n",
    )
    copied.write_bytes(b"tampered: true\n")
    recorded["authoritative.yaml"] = hashlib.sha256(copied.read_bytes()).hexdigest()
    with pytest.raises(ValueError) as error:
        validate(
            provenance,
            recorded,
            {"authoritative.yaml": source},
            "diff --git a/file b/file\n",
        )
    assert str(error.value) == "provenance copy differs from authoritative source: authoritative.yaml"


def test_full_provenance_copy_validation_pins_launch_time_controller_hash(
    tmp_path: Path,
) -> None:
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    controller = provenance / "run_full_fold.py"
    controller.write_bytes(b"launch-time-controller\n")
    diff = provenance / "derived-runtime.diff"
    diff.write_text("diff --git a/file b/file\n", encoding="utf-8")
    controller_sha = hashlib.sha256(controller.read_bytes()).hexdigest()
    recorded = {
        "run_full_fold.py": controller_sha,
        "derived-runtime.diff": hashlib.sha256(diff.read_bytes()).hexdigest(),
    }

    verifier.validate_provenance_copies(
        provenance,
        recorded,
        {},
        "diff --git a/file b/file\n",
        launch_time_hashes={"run_full_fold.py": controller_sha},
    )
    controller.write_bytes(b"tampered-controller\n")
    recorded["run_full_fold.py"] = hashlib.sha256(controller.read_bytes()).hexdigest()
    with pytest.raises(ValueError) as error:
        verifier.validate_provenance_copies(
            provenance,
            recorded,
            {},
            "diff --git a/file b/file\n",
            launch_time_hashes={"run_full_fold.py": controller_sha},
        )
    assert str(error.value) == "launch-time provenance SHA-256 mismatch: run_full_fold.py"


def test_full_provenance_paths_cannot_be_ignored(tmp_path: Path) -> None:
    validate = getattr(verifier, "validate_full_provenance_paths", None)
    assert validate is not None, "full provenance path validator is required"
    reproduction = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1"
    original = reproduction / ".local" / "WildfireSpreadTS"
    derived = reproduction / ".local" / "WildfireSpreadTS-res18-runtime"
    patch = reproduction / "patches" / "res18_import_scope.patch"

    validate(original, derived, patch)
    with pytest.raises(ValueError) as error:
        validate(tmp_path / "ignored-original", derived, patch)
    assert str(error.value) == "original upstream path is not the pinned reproduction checkout"


def test_independent_full_verifier_rejects_metric_or_checkpoint_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"weights")
    raw = {
        "optimizer_steps": 10_000,
        "test_metrics": {"test_AP": 0.5712, "test_f1": 0.4},
        "best_checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
    }
    recorded = json.loads(json.dumps(raw))

    verify_result_summary(raw, recorded)
    recorded["test_metrics"]["test_AP"] = 0.6
    with pytest.raises(ValueError, match="full result summary"):
        verify_result_summary(raw, recorded)


def test_independent_metric_parser_accepts_observed_windows_cr_table() -> None:
    output = (
        "Test metric             DataLoader 0\r\r\n"
        "         test_AP            0.5546640157699585\r\r\n"
        "         test_f1            0.4988675117492676\r\r\n"
        "        test_iou            0.3323274552822113\r\r\n"
        "        test_loss          0.0064315893687307835\r\r\n"
        "     test_precision           0.7076455950737\r\r\n"
        "       test_recall          0.3852163851261139\r\r\n"
    )

    assert verifier._parse_metrics(output) == {
        "test_AP": pytest.approx(0.5546640157699585),
        "test_f1": pytest.approx(0.4988675117492676),
        "test_iou": pytest.approx(0.3323274552822113),
        "test_loss": pytest.approx(0.0064315893687307835),
        "test_precision": pytest.approx(0.7076455950737),
        "test_recall": pytest.approx(0.3852163851261139),
    }


def test_independent_verifier_requires_observer_recovery_lineage(tmp_path: Path) -> None:
    run = tmp_path / "fold2-full-terminal"
    run.mkdir()
    recorded = {
        "command_sha256": "command-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "optimizer_steps": 10_000,
        "test_metrics": {"test_AP": 0.5546640157699585},
        "raw_evidence_seal_sha256": "seal-sha",
    }
    (run / "full-result.json").write_text(
        json.dumps(recorded, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    failure = {
        "status": "fail",
        "error": "ValueError: test_AP is missing from the Lightning test table",
        "training_retry_performed": False,
    }
    (run / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
    failure_sha = hashlib.sha256((run / "failure.json").read_bytes()).hexdigest()
    result_sha = hashlib.sha256((run / "full-result.json").read_bytes()).hexdigest()
    finalized = "2026-08-10T13:22:08+00:00"
    recovery = {
        "status": "pass",
        "recovery_type": "observer_parser_finalize_existing",
        "finalized_utc": finalized,
        "previous_error": failure["error"],
        "failure_sha256": failure_sha,
        "full_result_sha256": result_sha,
        "command_sha256": "command-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "optimizer_steps": 10_000,
        "test_AP": 0.5546640157699585,
        "previous_failure_preserved": True,
        "scientific_child_relaunched": False,
        "offline_refinalization": True,
        "retrospective_raw_integrity_claim": False,
        "raw_evidence_seal_sha256": "seal-sha",
        "raw_evidence_before_equals_after": True,
        "raw_evidence_scope": "current offline re-finalization; not retrospective proof",
    }
    (run / "observer-recovery.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )
    completed = {
        "status": "pass",
        "exit_code": 0,
        "pid": 22944,
        "completed_utc": finalized,
        "finalized_existing": True,
        "observer_recovery": True,
    }
    (run / "completed.json").write_text(json.dumps(completed), encoding="utf-8")

    assert verifier.verify_completion_lineage(run, recorded, {"pid": 22944}) == {
        "completed_marker_verified": True,
        "observer_recovery_verified": True,
    }
    del recovery["failure_sha256"]
    (run / "observer-recovery.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )
    with pytest.raises(ValueError) as error:
        verifier.verify_completion_lineage(run, recorded, {"pid": 22944})
    assert str(error.value) == "observer recovery failure SHA-256 mismatch"


def _full_progress_events() -> str:
    rows = [
        json.dumps({"seconds": 1.0, "stream": "stdout", "text": "Epoch 0: 0%| | 0/121"})
    ]
    rows.extend(
        json.dumps(
            {
                "seconds": float(epoch + 2),
                "stream": "stdout",
                "text": f"Epoch {epoch}: 100%| | 121/121",
            }
        )
        for epoch in range(82)
    )
    rows.append(
        json.dumps(
            {"seconds": 100.0, "stream": "stdout", "text": "Epoch 82: 64%| | 78/121"}
        )
    )
    return "\n".join(rows)


def test_full_verifier_reconstructs_steps_wall_and_gpu_from_raw() -> None:
    assert verifier.reconstruct_optimizer_steps(_full_progress_events()) == 10_000
    rows = [
        {
            "observer_seconds": "0.0",
            "memory_used_mib": "100",
            "utilization_gpu_percent": "10",
            "child_memory_mib": "0",
        },
        {
            "observer_seconds": "1.0",
            "memory_used_mib": "110",
            "utilization_gpu_percent": "50",
            "child_memory_mib": "0",
        },
        {
            "observer_seconds": "3.0",
            "memory_used_mib": "105",
            "utilization_gpu_percent": "20",
            "child_memory_mib": "0",
        },
    ]

    summary = verifier.reconstruct_observer_statistics(
        "2026-08-10T00:00:00+00:00", 1_786_320_010_000_000_000, rows
    )

    assert summary["wall_seconds"] == pytest.approx(10.0)
    assert summary["wall_hours"] == pytest.approx(10.0 / 3600)
    assert summary["gpu_sample_count"] == 3.0
    assert summary["gpu_interval_count"] == 2
    assert summary["gpu_interval_mean_seconds"] == pytest.approx(1.5)
    assert summary["gpu_interval_median_seconds"] == pytest.approx(1.5)
    assert summary["gpu_observed_effective_hz"] == pytest.approx(2 / 3)
    assert summary["peak_gpu_used_mib"] == 110.0
    assert summary["gpu_utilization_median_percent"] == 20.0


@pytest.mark.parametrize(
    "metrics",
    [
        {
            "test_AP": 1.1,
            "test_f1": 0.4,
            "test_iou": 0.3,
            "test_loss": 0.01,
            "test_precision": 0.5,
            "test_recall": 0.5,
        },
        {
            "test_AP": 0.5,
            "test_f1": 0.4,
            "test_iou": 0.3,
            "test_loss": -0.01,
            "test_precision": 0.5,
            "test_recall": 0.5,
        },
        {
            "test_AP": 0.5,
            "test_f1": 0.4,
            "test_iou": 0.3,
            "test_loss": 0.01,
            "test_precision": 0.5,
        },
    ],
)
def test_full_verifier_requires_six_finite_legal_metrics(metrics: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="six legal test metrics"):
        verifier.validate_test_metrics(metrics)


def test_full_verifier_reads_checkpoint_step_and_displayed_validation_ap() -> None:
    metadata = verifier.validate_checkpoint_metadata(
        {"epoch": 79, "global_step": 9680, "pytorch-lightning_version": "2.0.1"}
    )
    events = (
        '{"seconds":1,"stream":"stdout","text":"Epoch 79: 100%| | 121/121, '
        'val_avg_precision=0.326\\r"}\n'
    )

    assert metadata == {
        "best_epoch": 79,
        "best_checkpoint_global_step": 9680,
        "checkpoint_lightning_version": "2.0.1",
    }
    assert verifier.reconstruct_displayed_validation_ap(events, 79) == pytest.approx(0.326)


def _write_full_raw_set(run: Path) -> Path:
    names = (
        "stdout.log",
        "stderr.log",
        "stream-events.jsonl",
        "gpu.csv",
        "exit-code.txt",
        "started.json",
        "config.yaml",
        "source-data-pre.json",
        "source-data-post.json",
        "launch.lock.json",
        "preflight.json",
        "effective-command.json",
        "failure.json",
    )
    for index, name in enumerate(names):
        (run / name).write_bytes(f"raw-{index}\n".encode())
    checkpoint = run / "model" / "checkpoints" / "best-epoch=79-val_avg_precision=0.33.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    return checkpoint


def test_full_raw_seal_is_independently_verified_and_detects_tamper(tmp_path: Path) -> None:
    run = tmp_path / "fold2-full-terminal"
    run.mkdir()
    _write_full_raw_set(run)
    manifest = verifier.capture_full_raw_manifest(run)
    seal = {
        "status": "pass",
        "seal_scope": "current offline re-finalization; not retrospective proof",
        "scientific_child_relaunched": False,
        "before": manifest,
        "after": manifest,
    }

    verified = verifier.verify_full_raw_seal(run, seal)

    assert verified["raw_evidence_seal_verified"] is True
    (run / "stdout.log").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="raw evidence manifest mismatch"):
        verifier.verify_full_raw_seal(run, seal)

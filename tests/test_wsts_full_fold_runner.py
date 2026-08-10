import json
import sys
import hashlib
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_full_fold as runner  # noqa: E402
from run_full_fold import (  # noqa: E402
    build_full_command,
    parse_test_metrics,
    validate_full_command,
    validate_full_success,
)


def _event(seconds: float, text: str) -> str:
    return json.dumps(
        {"seconds": seconds, "stream": "stdout", "text": text},
        sort_keys=True,
        separators=(",", ":"),
    )


def _command(tmp_path: Path) -> list[str]:
    return build_full_command(
        (tmp_path / "run-001").resolve(),
        python_executable=(tmp_path / "environment" / "python.exe").resolve(),
        entrypoint=(SCRIPTS_DIR / "official_entrypoint.py").resolve(),
        upstream_root=(tmp_path / "WildfireSpreadTS-runtime").resolve(),
        data_root=Path(r"D:\WildFire Project\data\hdf5"),
    )


def _ten_thousand_step_events() -> str:
    rows = [_event(1.0, "Epoch 0: 0%| | 0/121")]
    rows.extend(
        _event(float(epoch + 2), f"Epoch {epoch}: 100%| | 121/121")
        for epoch in range(82)
    )
    rows.append(_event(100.0, "Epoch 82: 64%| | 78/121"))
    return "\n".join(rows)


def _write_fixed_full_raw_set(run: Path) -> Path:
    contents = {
        "stdout.log": b"stdout\n",
        "stderr.log": b"stderr\n",
        "stream-events.jsonl": b"events\n",
        "gpu.csv": b"gpu\n",
        "exit-code.txt": b"0\n",
        "started.json": json.dumps({"pid": 22944}).encode(),
        "config.yaml": b"config\n",
        "source-data-pre.json": b"{}\n",
        "source-data-post.json": b"{}\n",
        "launch.lock.json": b"{}\n",
        "preflight.json": b"{}\n",
        "effective-command.json": b"{}\n",
        "failure.json": json.dumps(
            {
                "status": "fail",
                "error": "ValueError: test_AP is missing from the Lightning test table",
                "training_retry_performed": False,
            }
        ).encode(),
    }
    for name, content in contents.items():
        (run / name).write_bytes(content)
    checkpoint = run / "model" / "checkpoints" / "best-epoch=79-val_avg_precision=0.33.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    return checkpoint


def test_full_command_is_exact_official_fold2_10000_step_test_protocol(
    tmp_path: Path,
) -> None:
    command = _command(tmp_path)

    validate_full_command(command, (tmp_path / "run-001").resolve())
    assert command == [
        str((tmp_path / "environment" / "python.exe").resolve()),
        str((SCRIPTS_DIR / "official_entrypoint.py").resolve()),
        "--upstream-root",
        str((tmp_path / "WildfireSpreadTS-runtime").resolve()),
        f"--config={((tmp_path / 'WildfireSpreadTS-runtime' / 'cfgs' / 'unet' / 'res18_monotemporal.yaml').resolve()).as_posix()}",
        f"--trainer={((tmp_path / 'WildfireSpreadTS-runtime' / 'cfgs' / 'trainer_single_gpu.yaml').resolve()).as_posix()}",
        f"--data={((tmp_path / 'WildfireSpreadTS-runtime' / 'cfgs' / 'data_monotemporal_full_features.yaml').resolve()).as_posix()}",
        r"--data.data_dir=D:\WildFire Project\data\hdf5",
        "--data.data_fold_id=2",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        "--trainer.max_steps=10000",
        f"--trainer.default_root_dir={str((tmp_path / 'run-001').resolve())}",
        "--do_test=true",
    ]
    assert not any("do_predict" in item or "do_validate" in item for item in command)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("--trainer.max_steps=10000", "--trainer.max_steps=9999"),
        ("--do_test=true", "--do_test=false"),
        ("--data.num_workers=8", "--data.num_workers=64"),
        ("--data.features_to_keep=null", "--data.features_to_keep=[0,1]"),
    ],
)
def test_full_command_rejects_scientific_or_compatibility_drift(
    tmp_path: Path, old: str, new: str
) -> None:
    command = _command(tmp_path)
    changed = [new if item == old else item for item in command]

    with pytest.raises(ValueError, match="exact full-fold command"):
        validate_full_command(changed, (tmp_path / "run-001").resolve())


def test_test_metric_parser_accepts_lightning_table_and_rejects_missing_ap() -> None:
    output = (
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃        Test metric        ┃       DataLoader 0        ┃\n"
        "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
        "│          test_AP          │    0.5712345838546753     │\n"
        "│          test_f1          │    0.4010000228881836     │\n"
        "│          test_iou         │    0.2509999871253967     │\n"
        "│        test_loss          │    0.0031000000890344     │\n"
        "│      test_precision       │    0.4100000262260437     │\n"
        "│       test_recall         │    0.3920000195503235     │\n"
    )

    assert parse_test_metrics(output) == {
        "test_AP": pytest.approx(0.5712345838546753),
        "test_f1": pytest.approx(0.4010000228881836),
        "test_iou": pytest.approx(0.2509999871253967),
        "test_loss": pytest.approx(0.0031000000890344),
        "test_precision": pytest.approx(0.4100000262260437),
        "test_recall": pytest.approx(0.3920000195503235),
    }
    with pytest.raises(ValueError, match="all six test metrics"):
        parse_test_metrics("Testing DataLoader 0: 100%|##########| 1/1")


def test_test_metric_parser_accepts_observed_windows_cr_table_without_borders() -> None:
    output = (
        "Testing DataLoader 0: 100%|##########| 3337/3337\r\n"
        "Test metric             DataLoader 0\r\r\n"
        "         test_AP            0.5546640157699585\r\r\n"
        "         test_f1            0.4988675117492676\r\r\n"
        "        test_iou            0.3323274552822113\r\r\n"
        "        test_loss          0.0064315893687307835\r\r\n"
        "     test_precision           0.7076455950737\r\r\n"
        "       test_recall          0.3852163851261139\r\r\n"
    )

    assert parse_test_metrics(output) == {
        "test_AP": pytest.approx(0.5546640157699585),
        "test_f1": pytest.approx(0.4988675117492676),
        "test_iou": pytest.approx(0.3323274552822113),
        "test_loss": pytest.approx(0.0064315893687307835),
        "test_precision": pytest.approx(0.7076455950737),
        "test_recall": pytest.approx(0.3852163851261139),
    }


def test_full_success_requires_exact_step_best_checkpoint_and_completed_test(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best-epoch=42-val_avg_precision=0.55.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    events = _ten_thousand_step_events() + "\n" + _event(
        110.0, "Testing DataLoader 0: 100%| | 156/156"
    )
    output = (
        "`Trainer.fit` stopped: `max_steps=10000` reached.\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5712 │\n"
        "│ test_f1 │ 0.4 │\n"
        "        test_iou            0.3\n"
        "        test_loss           0.01\n"
        "        test_precision      0.5\n"
        "        test_recall         0.5\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1826138624\n"
    )

    result = validate_full_success(events, output, 0, [checkpoint])

    assert result["optimizer_steps"] == 10_000
    assert result["test_metrics"]["test_AP"] == pytest.approx(0.5712)
    assert result["best_checkpoint"] == str(checkpoint.resolve())


@pytest.mark.parametrize(
    ("output", "exit_code", "checkpoint_count", "message"),
    [
        ("Testing DataLoader 0: 100%| | 1/1\n│ test_AP │ 0.5 │", 0, 1, "max_steps"),
        ("`Trainer.fit` stopped: `max_steps=10000` reached.\n│ test_AP │ 0.5 │", 0, 1, "test completion"),
        ("`Trainer.fit` stopped: `max_steps=10000` reached.\nTesting DataLoader 0: 100%| | 1/1\n│ test_AP │ 0.5 │", 1, 1, "exit code"),
        ("`Trainer.fit` stopped: `max_steps=10000` reached.\nTesting DataLoader 0: 100%| | 1/1\n│ test_AP │ 0.5 │\nPredicting DataLoader", 0, 1, "predict"),
        ("`Trainer.fit` stopped: `max_steps=10000` reached.\nTesting DataLoader 0: 100%| | 1/1\n│ test_AP │ 0.5 │", 0, 2, "checkpoint"),
    ],
)
def test_full_success_fails_closed_on_incomplete_or_extra_actions(
    tmp_path: Path,
    output: str,
    exit_code: int,
    checkpoint_count: int,
    message: str,
) -> None:
    checkpoints = []
    for index in range(checkpoint_count):
        checkpoint = tmp_path / f"best-{index}.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        checkpoints.append(checkpoint)
    events = _ten_thousand_step_events()

    with pytest.raises(ValueError, match=message):
        validate_full_success(events, output, exit_code, checkpoints)


def test_finalize_existing_recovers_only_observer_parser_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "wsts-res18-t1-full"
    run = artifact_root / "fold2-full-terminal"
    run.mkdir(parents=True)
    checkpoint = _write_fixed_full_raw_set(run)
    failure_path = run / "failure.json"
    failure_before = failure_path.read_bytes()
    result = {
        "status": "pass",
        "optimizer_steps": 10_000,
        "command_sha256": "command-sha",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "test_metrics": {"test_AP": 0.5546640157699585},
    }
    monkeypatch.setattr(runner, "FULL_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(runner, "parse_full_result", lambda _: result)

    assert runner.finalize_existing_run(run) == result
    assert failure_path.read_bytes() == failure_before
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    recovery = json.loads((run / "observer-recovery.json").read_text(encoding="utf-8"))
    assert completed["status"] == "pass"
    assert completed["exit_code"] == 0
    assert completed["pid"] == 22944
    assert completed["finalized_existing"] is True
    assert completed["observer_recovery"] is True
    assert recovery["previous_error"] == (
        "ValueError: test_AP is missing from the Lightning test table"
    )
    assert recovery["scientific_child_relaunched"] is False
    assert recovery["optimizer_steps"] == 10_000
    assert recovery["test_AP"] == pytest.approx(0.5546640157699585)
    assert recovery["failure_sha256"] == runner._sha256(failure_path)
    assert recovery["full_result_sha256"] == runner._sha256(run / "full-result.json")
    seal = json.loads((run / "raw-evidence-seal.json").read_text(encoding="utf-8"))
    assert seal["before"] == seal["after"]
    assert seal["seal_scope"] == "current offline re-finalization; not retrospective proof"
    assert seal["scientific_child_relaunched"] is False
    assert result["raw_evidence_seal_sha256"] == runner._sha256(
        run / "raw-evidence-seal.json"
    )


def test_finalize_existing_reseals_completed_run_without_changing_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "wsts-res18-t1-full"
    run = artifact_root / "fold2-full-terminal"
    run.mkdir(parents=True)
    checkpoint = _write_fixed_full_raw_set(run)
    result = {
        "status": "pass",
        "optimizer_steps": 10_000,
        "command_sha256": "command-sha",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "test_metrics": {"test_AP": 0.5546640157699585},
    }
    (run / "full-result.json").write_text(json.dumps(result), encoding="utf-8")
    (run / "observer-recovery.json").write_text(
        json.dumps({"status": "pass", "scientific_child_relaunched": False}),
        encoding="utf-8",
    )
    (run / "completed.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "exit_code": 0,
                "pid": 22944,
                "finalized_existing": True,
                "observer_recovery": True,
            }
        ),
        encoding="utf-8",
    )
    raw_before = runner.capture_full_raw_manifest(run)
    monkeypatch.setattr(runner, "FULL_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(runner, "parse_full_result", lambda _: dict(result))

    finalized = runner.finalize_existing_run(run)

    assert runner.capture_full_raw_manifest(run) == raw_before
    assert finalized["optimizer_steps"] == 10_000
    recovery = json.loads((run / "observer-recovery.json").read_text(encoding="utf-8"))
    assert recovery["offline_refinalization"] is True
    assert recovery["retrospective_raw_integrity_claim"] is False


def test_checkpoint_metadata_and_displayed_ap_are_evidence_labeled() -> None:
    metadata = runner.validate_checkpoint_metadata(
        {"epoch": 79, "global_step": 9680, "pytorch-lightning_version": "2.0.1"}
    )
    events = _event(
        10.0,
        "Epoch 79: 100%| | 121/121, val_avg_precision=0.326, val_f1=0.317",
    )

    assert metadata["best_checkpoint_global_step"] == 9680
    assert runner.parse_displayed_validation_ap(events, 79) == pytest.approx(0.326)


def test_finalize_existing_rejects_non_parser_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "wsts-res18-t1-full"
    run = artifact_root / "fold2-full-terminal"
    run.mkdir(parents=True)
    (run / "failure.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "error": "RuntimeError: scientific child failed",
                "training_retry_performed": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "exit-code.txt").write_text("0\n", encoding="utf-8")
    (run / "started.json").write_text(json.dumps({"pid": 22944}), encoding="utf-8")
    monkeypatch.setattr(runner, "FULL_ARTIFACTS_ROOT", artifact_root)

    with pytest.raises(ValueError, match="exact observer parser failure"):
        runner.finalize_existing_run(run)

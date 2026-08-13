import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_released_weight import (  # noqa: E402
    _strict_load_preflight,
    build_weight_command,
    summarize_filename_aps,
    validate_download,
    validate_weight_command,
    validate_weight_manifest,
    validate_weight_output,
)
import evaluate_released_weight as weight_controller  # noqa: E402
import released_weight_contract as contract  # noqa: E402
import run_calibration as calibration_runner  # noqa: E402
from verify_released_weight import validate_weight_result  # noqa: E402
import verify_released_weight as weight_verifier  # noqa: E402


FILENAMES = [
    "fold0_testAP0.528.pth",
    "fold1_testAP0.426.pth",
    "fold2_testAP0.571.pth",
    "fold3_testAP0.307.pth",
    "fold4_testAP0.483.pth",
    "fold5_testAP0.322.pth",
    "fold6_testAP0.577.pth",
    "fold7_testAP0.474.pth",
    "fold8_testAP0.478.pth",
    "fold9_testAP0.471.pth",
    "fold10_testAP0.324.pth",
    "fold11_testAP0.474.pth",
]


@pytest.mark.parametrize(
    ("memory_free_mib", "accepted"),
    [(15_999, False), (16_000, True)],
)
def test_weight_preflight_uses_empirical_test_only_gpu_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_free_mib: int,
    accepted: bool,
) -> None:
    spec = weight_controller._fold2_spec()
    run = tmp_path / "weight-preflight"
    run.mkdir()
    weight = tmp_path / "weights" / spec.filename
    smoke = tmp_path / "worker-smoke.json"
    smoke.write_text("{}\n", encoding="utf-8")
    inventory = {
        "root": str(tmp_path / "data"),
        "file_count": 607,
        "total_bytes": 24_242_259_023,
        "files": [],
    }

    monkeypatch.setattr(weight_controller, "weight_cache_path", lambda _spec: weight)
    monkeypatch.setattr(
        weight_controller,
        "global_weight_lock_path",
        lambda _spec: tmp_path / "weight-evaluation.lock.json",
    )
    monkeypatch.setattr(weight_controller, "validate_download", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        weight_controller,
        "_run_checked",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(weight_controller, "WORKER_SMOKE_PATH", smoke)
    monkeypatch.setattr(
        weight_controller,
        "validate_train_validation_smoke_evidence",
        lambda _smoke: None,
    )
    monkeypatch.setattr(weight_controller, "verify_inventory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(weight_controller, "build_source_inventory", lambda _root: inventory)
    monkeypatch.setattr(
        weight_controller,
        "verify_upstream",
        lambda *_args, **_kwargs: calibration_runner.EXPECTED_CODE_COMMIT,
    )
    monkeypatch.setattr(
        weight_controller,
        "_prepare_derived_runtime",
        lambda: {"git_diff_exact_match": True},
    )
    monkeypatch.setattr(
        weight_controller,
        "_runtime_preflight",
        lambda: {"cuda_available": True},
    )
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
    monkeypatch.setattr(
        weight_controller,
        "_strict_load_preflight",
        lambda _command: {"status": "pass", "loaded_tensor_count": 182},
    )

    def reject_scientific_child(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("released-weight preflight must not start a scientific child")

    monkeypatch.setattr(weight_controller, "observe_process", reject_scientific_child)

    if not accepted:
        with pytest.raises(ValueError, match="requires at least 16000 MiB"):
            weight_controller._preflight(
                run,
                spec=spec,
                require_full_complete=False,
            )
        return

    preflight, source_inventory = weight_controller._preflight(
        run,
        spec=spec,
        require_full_complete=False,
    )
    assert preflight["gpu"]["memory_free_mib"] == 16_000.0
    assert preflight["strict_load_preflight"] == {
        "status": "pass",
        "loaded_tensor_count": 182,
    }
    assert "--data.num_workers=8" in preflight["command"]
    assert "--do_train=false" in preflight["command"]
    assert source_inventory == inventory


def test_strict_load_preflight_propagates_disabled_wandb_environment(
    tmp_path: Path,
) -> None:
    child = tmp_path / "strict_child.py"
    child.write_text(
        "import os\n"
        "if os.environ.get('WANDB_MODE') != 'disabled':\n"
        "    raise SystemExit(7)\n"
        "print('WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=1')\n",
        encoding="utf-8",
    )

    result = _strict_load_preflight(
        [sys.executable, str(child), "--upstream-root", "unused", "--weights-path", "unused"]
    )

    assert result["status"] == "pass"
    assert result["test_loader_invoked"] is False


def _manifest() -> list[dict[str, object]]:
    items = []
    for name in FILENAMES:
        item = {
            "type": "file",
            "path": f"trained_model_weights/Res18Unet_T1/All/{name}",
            "size": 57_889_221,
            "lfs": {"oid": "0" * 64},
        }
        if name == "fold2_testAP0.571.pth":
            item["lfs"] = {
                "oid": "e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec"
            }
        items.append(item)
    return items


def test_manifest_selects_exact_pinned_fold2_weight_and_aggregate() -> None:
    selected = validate_weight_manifest(_manifest())

    assert selected["path"] == (
        "trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth"
    )
    assert selected["size"] == 57_889_221
    assert selected["lfs_sha256"] == (
        "e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec"
    )
    aggregate = summarize_filename_aps(FILENAMES)
    assert aggregate["count"] == 12
    assert aggregate["mean"] == 0.45291666666666663
    assert aggregate["population_std"] == 0.08827179460179917
    assert selected["filename_manifest"]["filenames"] == FILENAMES
    assert selected["filename_manifest"]["aggregate"] == aggregate
    assert selected["filename_manifest"]["paper_table_provenance"] is False


def test_manifest_rejects_missing_duplicate_or_wrong_fold2_hash() -> None:
    with pytest.raises(ValueError, match="exactly twelve"):
        validate_weight_manifest(_manifest()[:-1])
    duplicate = _manifest()
    duplicate[-1]["path"] = duplicate[0]["path"]
    with pytest.raises(ValueError, match="unique folds"):
        validate_weight_manifest(duplicate)
    wrong_hash = _manifest()
    wrong_hash[2]["lfs"] = {"oid": "f" * 64}
    with pytest.raises(ValueError, match="Fold 2 weight"):
        validate_weight_manifest(wrong_hash)


def test_download_validation_checks_exact_size_and_sha(tmp_path: Path) -> None:
    weight = tmp_path / "fold2.pth"
    weight.write_bytes(b"official-weight")
    digest = hashlib.sha256(b"official-weight").hexdigest()

    validate_download(weight, expected_size=15, expected_sha256=digest)
    with pytest.raises(ValueError, match="size"):
        validate_download(weight, expected_size=16, expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_download(weight, expected_size=15, expected_sha256="0" * 64)


def test_weight_command_is_test_only_on_fold2_all_t1(tmp_path: Path) -> None:
    spec = contract.spec_for_fold(
        contract.load_pinned_manifest(
            REPOSITORY_ROOT
            / "reproductions"
            / "wsts_res18_unet_t1"
            / "official_weights_manifest.json"
        ),
        2,
    )
    command = build_weight_command(
        spec=spec,
        run_directory=(tmp_path / "run").resolve(),
        weight_path=(tmp_path / "fold2.pth").resolve(),
        python_executable=(tmp_path / "env" / "python.exe").resolve(),
        entrypoint=(SCRIPTS_DIR / "official_weight_entrypoint.py").resolve(),
        upstream_root=(tmp_path / "runtime").resolve(),
        data_root=Path(r"D:\WildFire Project\data\hdf5"),
    )

    assert "--data.data_fold_id=2" in command
    assert "--data.features_to_keep=null" in command
    assert "--data.n_leading_observations=1" in command
    assert "--data.num_workers=8" in command
    assert "--do_train=false" in command
    assert "--do_test=false" in command
    assert not any("max_steps" in item or "do_predict" in item or "do_validate" in item for item in command)
    validate_weight_command(
        command,
        (tmp_path / "run").resolve(),
        (tmp_path / "fold2.pth").resolve(),
        spec,
    )

    changed = ["--do_train=true" if item == "--do_train=false" else item for item in command]
    with pytest.raises(ValueError, match="exact released-weight command"):
        validate_weight_command(
            changed,
            (tmp_path / "run").resolve(),
            (tmp_path / "fold2.pth").resolve(),
            spec,
        )


def test_weight_output_requires_strict_load_and_test_without_fit_or_validation() -> None:
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5708 │\n"
        "│ test_f1 │ 0.4 │\n"
        "        test_iou            0.3\n"
        "        test_loss           0.01\n"
        "        test_precision      0.5\n"
        "        test_recall         0.5\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1700000000\n"
    )

    result = validate_weight_output(output, 0)

    assert result["strict_load"] is True
    assert result["test_metrics"]["test_AP"] == pytest.approx(0.5708)
    assert result["peak_allocated_bytes"] == 1_700_000_000


@pytest.mark.parametrize(
    ("extra", "exit_code", "message"),
    [
        ("", 1, "exit code"),
        ("Epoch 0: 1/121", 0, "training"),
        ("Validation DataLoader 0: 1/22", 0, "validation"),
        ("Predicting DataLoader 0: 1/1", 0, "predict"),
    ],
)
def test_weight_output_fails_closed_on_non_test_actions(
    extra: str, exit_code: int, message: str
) -> None:
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5708 │\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1700000000\n"
        + extra
    )
    with pytest.raises(ValueError, match=message):
        validate_weight_output(output, exit_code)


def test_released_weight_result_requires_strict_load_test_only_and_ap_delta() -> None:
    result = validate_weight_result(
        {
            "exit_code": 0,
            "strict_load": True,
            "train_invoked": False,
            "validation_invoked": False,
            "predict_invoked": False,
            "test_metrics": {"test_AP": 0.5708, "test_f1": 0.4},
            "filename_ap": 0.571,
        }
    )

    assert result["test_AP"] == pytest.approx(0.5708)
    assert result["filename_ap_absolute_difference"] == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="strict"):
        validate_weight_result(
            {
                "exit_code": 0,
                "strict_load": False,
                "train_invoked": False,
                "validation_invoked": False,
                "predict_invoked": False,
                "test_metrics": {"test_AP": 0.5708},
                "filename_ap": 0.571,
            }
        )


def test_weight_launch_dependency_requires_completed_and_independently_verified_full_run(
    tmp_path: Path,
) -> None:
    validate = getattr(weight_controller, "validate_full_run_dependency", None)
    assert validate is not None, "full-run dependency validator is required"
    run = (tmp_path / "fold2-full-run").resolve()
    run.mkdir()
    lock = tmp_path / "fold2-full-launch.lock.json"
    lock.write_text(json.dumps({"run_directory": str(run)}), encoding="utf-8")
    (run / "completed.json").write_text(
        json.dumps({"status": "pass", "exit_code": 0}), encoding="utf-8"
    )
    (run / "full-result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "optimizer_steps": 10_000,
                "command_sha256": "full-command",
                "checkpoint_sha256": "best-checkpoint",
            }
        ),
        encoding="utf-8",
    )
    independent = run / "independent-verification.json"
    independent.write_text(
        json.dumps(
            {
                "status": "pass",
                "optimizer_steps": 10_000,
                "command_sha256": "full-command",
                "checkpoint_sha256": "best-checkpoint",
            }
        ),
        encoding="utf-8",
    )

    result = validate(lock)
    assert result["independent_status"] == "pass"
    independent.unlink()
    with pytest.raises(ValueError) as error:
        validate(lock)
    assert str(error.value) == "full Fold-2 independent verification is missing"


def test_independent_weight_verifier_builds_exact_command_and_requires_all_lineage(
    tmp_path: Path,
) -> None:
    builder = getattr(weight_verifier, "build_expected_weight_command", None)
    validate = getattr(weight_verifier, "verify_weight_command_lineage", None)
    assert builder is not None, "independent weight command builder is required"
    assert validate is not None, "independent weight lineage validator is required"
    run = (tmp_path / "weight-run").resolve()
    derived = (tmp_path / "WildfireSpreadTS-res18-runtime").resolve()
    weight = (tmp_path / "fold2_testAP0.571.pth").resolve()
    command = builder(run, derived, weight)
    command_hash = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {"command": command, "command_sha256": command_hash}
    payloads = {
        "launch": dict(payload),
        "run_lock": dict(payload),
        "preflight": dict(payload),
        "started": dict(payload),
        "effective": dict(payload),
        "result": {"command_sha256": command_hash},
    }

    validate(payloads, command, command_hash)
    del payloads["run_lock"]
    with pytest.raises(ValueError) as error:
        validate(payloads, command, command_hash)
    assert str(error.value) == "run_lock marker is missing"


def test_independent_weight_verifier_reconstructs_raw_strict_test_evidence() -> None:
    reconstruct = getattr(weight_verifier, "reconstruct_weight_evidence", None)
    assert reconstruct is not None, "independent raw weight evidence parser is required"
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5708 │\n"
        "│ test_f1 │ 0.4 │\n"
        "        test_iou            0.3\n"
        "        test_loss           0.01\n"
        "        test_precision      0.5\n"
        "        test_recall         0.5\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1700000000\n"
    )

    assert reconstruct(output, 0) == {
        "exit_code": 0,
        "strict_load": True,
        "loaded_tensor_count": 123,
        "train_invoked": False,
        "validation_invoked": False,
        "predict_invoked": False,
        "test_metrics": {
            "test_AP": pytest.approx(0.5708),
            "test_f1": pytest.approx(0.4),
            "test_iou": pytest.approx(0.3),
            "test_loss": pytest.approx(0.01),
            "test_precision": pytest.approx(0.5),
            "test_recall": pytest.approx(0.5),
        },
        "peak_allocated_bytes": 1_700_000_000,
    }


def test_independent_weight_verifier_parses_windows_borderless_lightning_table() -> None:
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 3337/3337\n"
        "────────────────────────────────────────────────────────────────\n"
        "       Test metric             DataLoader 0\n"
        "\n"
        "         test_AP            0.5709022879600525\n"
        "\n"
        "         test_f1             0.433401882648468\n"
        "\n"
        "        test_iou            0.27665162086486816\n"
        "\n"
        "        test_loss          0.005918989889323711\n"
        "\n"
        "     test_precision         0.7646416425704956\n"
        "\n"
        "       test_recall          0.3024023771286011\n"
        "────────────────────────────────────────────────────────────────\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=12050196992\n"
    )

    result = weight_verifier.reconstruct_weight_evidence(output, 0)

    assert result["test_metrics"] == {
        "test_AP": pytest.approx(0.5709022879600525),
        "test_f1": pytest.approx(0.433401882648468),
        "test_iou": pytest.approx(0.27665162086486816),
        "test_loss": pytest.approx(0.005918989889323711),
        "test_precision": pytest.approx(0.7646416425704956),
        "test_recall": pytest.approx(0.3024023771286011),
    }


def test_weight_verifier_independently_derives_filename_manifest_aggregate() -> None:
    evidence = weight_verifier.derive_filename_manifest_evidence()

    assert evidence["filenames"] == FILENAMES
    assert evidence["aggregate"] == {
        "count": 12,
        "mean": 0.45291666666666663,
        "population_std": 0.08827179460179917,
    }
    assert evidence["paper_table_provenance"] is False


def test_weight_verifier_reconstructs_wall_gpu_and_preserves_raw_manifest(
    tmp_path: Path,
) -> None:
    run = tmp_path / "fold2-weight-terminal"
    run.mkdir()
    raw_names = (
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
        "official-test-pr-curve-data.npz",
    )
    for index, name in enumerate(raw_names):
        (run / name).write_bytes(f"raw-{index}\n".encode())
    weight = tmp_path / "fold2_testAP0.571.pth"
    weight.write_bytes(b"weight")
    before = weight_verifier.capture_weight_raw_manifest(run, weight)
    rows = [
        {
            "observer_seconds": "0.0",
            "memory_used_mib": "100",
            "utilization_gpu_percent": "10",
            "child_memory_mib": "0",
        },
        {
            "observer_seconds": "1.0",
            "memory_used_mib": "120",
            "utilization_gpu_percent": "50",
            "child_memory_mib": "0",
        },
        {
            "observer_seconds": "2.0",
            "memory_used_mib": "110",
            "utilization_gpu_percent": "30",
            "child_memory_mib": "0",
        },
    ]

    observer = weight_verifier.reconstruct_observer_statistics(
        "2026-08-10T00:00:00+00:00", 1_786_320_010_000_000_000, rows
    )
    after = weight_verifier.capture_weight_raw_manifest(run, weight)

    assert before == after
    assert observer["wall_seconds"] == pytest.approx(10.0)
    assert observer["gpu_sample_count"] == 3.0
    assert observer["gpu_observed_effective_hz"] == pytest.approx(1.0)
    assert observer["peak_gpu_used_mib"] == 120.0
    (run / "stdout.log").write_bytes(b"tampered")
    assert weight_verifier.capture_weight_raw_manifest(run, weight) != before


@pytest.mark.parametrize(
    "metrics",
    [
        {
            "test_AP": float("nan"),
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
def test_weight_verifier_requires_six_finite_legal_metrics(
    metrics: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="six legal test metrics"):
        weight_verifier.validate_test_metrics(metrics)


def test_weight_finalize_existing_adds_offline_manifest_without_rewriting_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "wsts-res18-t1-official-weight"
    run = artifact_root / "fold2-weight-terminal"
    run.mkdir(parents=True)
    spec = weight_controller._fold2_spec()
    weight = weight_controller.weight_cache_path(spec)
    command = weight_controller.build_weight_command(
        spec=spec, run_directory=run, weight_path=weight
    )
    command_hash = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    launch = {
        "command": command,
        "command_sha256": command_hash,
        "run_directory": str(run.resolve()),
        "test_only": True,
        "single_launch_no_retry": True,
    }
    (artifact_root / "fold2-weight-evaluation.lock.json").write_text(
        json.dumps(launch), encoding="utf-8"
    )
    (run / "launch.lock.json").write_text(json.dumps(launch), encoding="utf-8")
    preflight = run / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                **launch,
                "status": "pass",
                "weight_path": str(weight.resolve()),
                "weight_sha256": spec.sha256,
                "filename_ap": spec.filename_ap,
            }
        ),
        encoding="utf-8",
    )
    preflight_before = preflight.read_bytes()
    (run / "completed.json").write_text(
        json.dumps({"status": "pass", "exit_code": 0, "pid": 37668}),
        encoding="utf-8",
    )
    (run / "exit-code.txt").write_text("0\n", encoding="utf-8")
    (run / "started.json").write_text(
        json.dumps(
            {"pid": 37668, "command": command, "command_sha256": command_hash}
        ),
        encoding="utf-8",
    )
    (run / "effective-command.json").write_text(
        json.dumps({"command": command, "command_sha256": command_hash}),
        encoding="utf-8",
    )
    result = {
        "status": "pass",
        "exit_code": 0,
        "strict_load": True,
        "train_invoked": False,
        "validation_invoked": False,
        "predict_invoked": False,
        "test_metrics": {
            "test_AP": 0.5709,
            "test_f1": 0.4,
            "test_iou": 0.3,
            "test_loss": 0.01,
            "test_precision": 0.5,
            "test_recall": 0.5,
        },
        "filename_manifest": weight_controller.build_filename_manifest_evidence(),
    }
    monkeypatch.setattr(weight_controller, "WEIGHT_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(
        weight_controller, "parse_weight_run", lambda _run, **_kwargs: dict(result)
    )
    monkeypatch.setattr(
        weight_controller,
        "collect_official_test_output",
        lambda _run, *, recovery, spec, recovery_authorization: {
            "collection_mode": "offline-finalize-existing"
        }
        if recovery
        else {},
    )

    expected = {
        **result,
        "command_sha256": command_hash,
        "child_pid": 37668,
    }
    finalized = weight_controller.finalize_existing_weight(
        run, recovery_authorization=tmp_path / "authorization.json"
    )

    assert finalized == expected
    assert preflight.read_bytes() == preflight_before
    augmentation = json.loads(
        (run / "offline-preflight-augmentation.json").read_text(encoding="utf-8")
    )
    assert augmentation["filename_manifest"] == result["filename_manifest"]
    assert augmentation["scientific_child_relaunched"] is False
    assert augmentation["launch_preflight_sha256"] == hashlib.sha256(
        preflight_before
    ).hexdigest()


def test_weight_finalize_existing_cli_requires_external_recovery_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def finalize(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "pass"}

    monkeypatch.setattr(weight_controller, "finalize_existing_weight", finalize)

    with pytest.raises(ValueError, match="recovery authorization"):
        weight_controller.main(
            ["--fold-id", "0", "--finalize-existing", str(tmp_path / "run")]
        )
    assert called is False


def test_weight_verifier_requires_preserved_first_parser_failure(tmp_path: Path) -> None:
    run = tmp_path / "fold2-weight-terminal"
    run.mkdir()
    marker = {
        "status": "fail",
        "failure_stage": "independent_verifier_postprocess",
        "error": "ValueError: released-weight finite test metrics are missing",
        "scientific_child_exit_code": 0,
        "scientific_child_relaunched": False,
        "retrospective_documentation": True,
    }
    (run / "independent-verifier-first-failure.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )

    assert weight_verifier.verify_first_verifier_failure(run) == {
        "first_verifier_failure_preserved": True
    }
    marker["scientific_child_relaunched"] = True
    (run / "independent-verifier-first-failure.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="first verifier failure evidence mismatch"):
        weight_verifier.verify_first_verifier_failure(run)


def test_weight_provenance_classifies_controller_as_launch_time_copy() -> None:
    authoritative, launch_time = weight_verifier.build_weight_provenance_contract()

    assert "evaluate_released_weight.py" not in authoritative
    assert launch_time == {
        "official_weight_entrypoint.py": (
            "ea024f5982e1e3d34ddb5cdc53354d837dd00b755102249635db46f5cfeb3b04"
        ),
        "evaluate_released_weight.py": (
            "de60464477f74d324bd2fadb451526814bc729fa71d1eeb167f53340626e080b"
        )
    }


def test_weight_verifier_accepts_only_explicit_offline_full_dependency_augmentation(
    tmp_path: Path,
) -> None:
    run = tmp_path / "fold2-weight-terminal"
    run.mkdir()
    launch_recorded = {"result_sha256": "old", "independent_sha256": "old-independent"}
    current = {"result_sha256": "new", "independent_sha256": "new-independent"}
    marker = {
        "status": "pass",
        "augmentation_scope": "offline current full-run dependency hashes only",
        "launch_recorded_dependency": launch_recorded,
        "current_dependency": current,
        "scientific_child_relaunched": False,
    }
    (run / "offline-full-dependency-augmentation.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )

    weight_verifier.validate_full_dependency_augmentation(
        run, launch_recorded, current
    )
    marker["scientific_child_relaunched"] = True
    (run / "offline-full-dependency-augmentation.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="full dependency augmentation mismatch"):
        weight_verifier.validate_full_dependency_augmentation(
            run, launch_recorded, current
        )

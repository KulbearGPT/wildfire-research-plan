import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1"
SCRIPTS_DIR = REPRODUCTION_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_released_weight as controller  # noqa: E402
import released_weight_contract as contract  # noqa: E402
import verify_released_weight as verifier  # noqa: E402


@pytest.fixture(scope="module")
def specs() -> tuple[contract.WeightSpec, ...]:
    return contract.load_pinned_manifest(
        REPRODUCTION_ROOT / "official_weights_manifest.json"
    )


@pytest.mark.parametrize("fold_id", [0, 5, 11])
def test_command_is_strict_test_only_for_every_fold(
    tmp_path: Path, specs: tuple[contract.WeightSpec, ...], fold_id: int
) -> None:
    spec = contract.spec_for_fold(specs, fold_id)
    command = controller.build_weight_command(
        spec=spec,
        run_directory=tmp_path / f"fold{fold_id}",
        weight_path=tmp_path / spec.filename,
    )

    assert command.count(f"--data.data_fold_id={fold_id}") == 1
    assert "--do_train=false" in command
    assert "--do_test=false" in command
    assert not any("predict" in argument or "validate" in argument for argument in command)


def test_commands_differ_only_by_fold_weight_and_run(
    tmp_path: Path, specs: tuple[contract.WeightSpec, ...]
) -> None:
    commands = {}
    for fold_id in (0, 5, 11):
        spec = contract.spec_for_fold(specs, fold_id)
        commands[fold_id] = controller.build_weight_command(
            spec=spec,
            run_directory=tmp_path / f"fold{fold_id}-weight-run",
            weight_path=tmp_path / spec.filename,
        )

    normalized = []
    for fold_id, command in commands.items():
        spec = contract.spec_for_fold(specs, fold_id)
        normalized.append(
            [
                argument
                .replace(str((tmp_path / spec.filename).resolve()), "WEIGHT")
                .replace(str((tmp_path / f"fold{fold_id}-weight-run").resolve()), "RUN")
                .replace(f"--data.data_fold_id={fold_id}", "--data.data_fold_id=FOLD")
                for argument in command
            ]
        )
    assert normalized[0] == normalized[1] == normalized[2]


@pytest.mark.parametrize("fold_id", [0, 5, 11])
def test_fold_claims_capture_exact_split_and_weight_metadata(
    tmp_path: Path, specs: tuple[contract.WeightSpec, ...], fold_id: int
) -> None:
    spec = contract.spec_for_fold(specs, fold_id)
    weight = (tmp_path / spec.filename).resolve()

    claims = controller.build_fold_claims(spec=spec, weight_path=weight)

    assert claims == {
        "fold_id": fold_id,
        "train_years": list(spec.train_years),
        "validation_year": spec.validation_year,
        "test_year": spec.test_year,
        "weight_path": str(weight),
        "weight_filename": spec.filename,
        "weight_size": spec.size,
        "weight_sha256": spec.sha256,
        "filename_ap": spec.filename_ap,
    }
    verifier.verify_fold_claims(claims, spec=spec, weight_path=weight)


def test_verifier_rejects_cross_fold_preflight_and_result_claims(
    tmp_path: Path, specs: tuple[contract.WeightSpec, ...]
) -> None:
    fold0 = contract.spec_for_fold(specs, 0)
    fold5 = contract.spec_for_fold(specs, 5)
    weight0 = (tmp_path / fold0.filename).resolve()
    claims0 = controller.build_fold_claims(spec=fold0, weight_path=weight0)

    with pytest.raises(ValueError, match="fold claims"):
        verifier.verify_fold_claims(claims0, spec=fold5, weight_path=weight0)
    with pytest.raises(ValueError, match="fold claims"):
        verifier.verify_fold_claims(
            claims0, spec=fold0, weight_path=(tmp_path / fold5.filename).resolve()
        )


@pytest.mark.parametrize("fold_id", [0, 5, 11])
def test_fold_cache_lock_and_run_names_are_fold_specific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    fold_id: int,
) -> None:
    spec = contract.spec_for_fold(specs, fold_id)
    artifact_root = tmp_path / "artifacts"
    cache_root = tmp_path / "weights"
    monkeypatch.setattr(controller, "WEIGHT_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(controller, "WEIGHT_CACHE_ROOT", cache_root)

    run = controller._create_run_directory(spec)

    assert controller.weight_cache_path(spec) == cache_root / spec.filename
    assert controller.global_weight_lock_path(spec) == (
        artifact_root / f"fold{fold_id}-weight-evaluation.lock.json"
    )
    assert run.parent == artifact_root.resolve()
    assert run.name.startswith(f"fold{fold_id}-weight-")


def test_cross_fold_run_weight_and_lock_aliases_fail_closed(
    tmp_path: Path, specs: tuple[contract.WeightSpec, ...]
) -> None:
    fold0 = contract.spec_for_fold(specs, 0)
    fold5 = contract.spec_for_fold(specs, 5)
    artifact_root = tmp_path / "artifacts"
    run0 = artifact_root / "fold0-weight-existing"
    weight0 = tmp_path / fold0.filename
    lock0 = artifact_root / "fold0-weight-evaluation.lock.json"

    controller.validate_fold_paths(
        spec=fold0,
        run_directory=run0,
        weight_path=weight0,
        lock_path=lock0,
        artifacts_root=artifact_root,
    )
    for alias in (
        {"run_directory": artifact_root / "fold5-weight-existing"},
        {"weight_path": tmp_path / fold5.filename},
        {"lock_path": artifact_root / "fold5-weight-evaluation.lock.json"},
    ):
        arguments = {
            "spec": fold0,
            "run_directory": run0,
            "weight_path": weight0,
            "lock_path": lock0,
            "artifacts_root": artifact_root,
            **alias,
        }
        with pytest.raises(ValueError, match="fold path identity"):
            controller.validate_fold_paths(**arguments)


def test_expected_verifier_command_is_selected_from_spec(
    tmp_path: Path, specs: tuple[contract.WeightSpec, ...]
) -> None:
    for fold_id in (0, 5, 11):
        spec = contract.spec_for_fold(specs, fold_id)
        command = verifier.build_expected_weight_command(
            tmp_path / f"fold{fold_id}-weight-run",
            tmp_path / "derived",
            tmp_path / spec.filename,
            spec,
        )
        assert command.count(f"--data.data_fold_id={fold_id}") == 1


def test_cli_defaults_to_fold2_and_accepts_only_four_actions() -> None:
    arguments = controller.parse_arguments(["--fetch-only"])
    assert arguments.fold_id == 2
    assert controller.parse_arguments(["--fold-id", "11", "--launch"]).fold_id == 11
    with pytest.raises(SystemExit):
        controller.parse_arguments(["--fold-id", "12", "--fetch-only"])
    with pytest.raises(SystemExit):
        controller.parse_arguments(["--pin-manifest", "manifest.json"])


def test_fetch_weight_uses_exact_spec_and_default_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    spec = contract.spec_for_fold(specs, 11)
    cache_root = tmp_path / "weights"
    monkeypatch.setattr(controller, "WEIGHT_CACHE_ROOT", cache_root)
    monkeypatch.setattr(controller, "query_manifest", lambda: [])
    target = cache_root / spec.filename
    target.parent.mkdir(parents=True)
    target.write_bytes(b"official")
    local_spec = replace(
        spec,
        size=8,
        sha256=hashlib.sha256(b"official").hexdigest(),
    )

    result = controller.fetch_weight(local_spec)

    assert result["fold_id"] == 11
    assert result["filename"] == spec.filename
    assert result["filename_ap"] == spec.filename_ap
    assert result["sha256"] == local_spec.sha256
    assert result["local_path"] == str(target.resolve())
    assert result["download_reused"] is True


def test_retrospective_markers_are_optional_but_validated_when_present(
    tmp_path: Path,
) -> None:
    run = tmp_path / "fold0-weight-clean"
    run.mkdir()

    assert verifier.verify_retrospective_markers(run) == {
        "first_verifier_failure_preserved": False,
        "offline_full_dependency_augmentation_present": False,
    }
    (run / "independent-verifier-first-failure.json").write_text(
        json.dumps({"scientific_child_relaunched": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="first verifier failure"):
        verifier.verify_retrospective_markers(run)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finalize_existing_rejects_cross_fold_saved_lineage_before_writing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    fold0 = contract.spec_for_fold(specs, 0)
    fold5 = contract.spec_for_fold(specs, 5)
    artifact_root = tmp_path / "artifacts"
    cache_root = tmp_path / "weights"
    run = artifact_root / "fold5-weight-copied"
    run.mkdir(parents=True)
    command = controller.build_weight_command(
        spec=fold0,
        run_directory=run,
        weight_path=cache_root / fold0.filename,
    )
    command_hash = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    launch = {
        "command": command,
        "command_sha256": command_hash,
        "run_directory": str(run.resolve()),
        "test_only": True,
        "single_launch_no_retry": True,
    }
    _write_json(artifact_root / "fold5-weight-evaluation.lock.json", launch)
    _write_json(run / "launch.lock.json", launch)
    _write_json(
        run / "preflight.json",
        {
            **launch,
            **controller.build_fold_claims(
                spec=fold0, weight_path=cache_root / fold0.filename
            ),
        },
    )
    _write_json(
        run / "started.json",
        {"command": command, "command_sha256": command_hash, "pid": 7},
    )
    _write_json(
        run / "effective-command.json",
        {"command": command, "command_sha256": command_hash},
    )
    _write_json(run / "completed.json", {"status": "pass", "exit_code": 0, "pid": 7})
    (run / "exit-code.txt").write_text("0\n", encoding="utf-8")
    monkeypatch.setattr(controller, "WEIGHT_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(controller, "WEIGHT_CACHE_ROOT", cache_root)
    monkeypatch.setattr(controller, "validate_download", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        controller,
        "parse_weight_run",
        lambda *_args, **_kwargs: {"filename_manifest": controller.build_filename_manifest_evidence()},
    )

    with pytest.raises(ValueError, match="saved launch lineage"):
        controller.finalize_existing_weight(run, fold5)
    assert not (run / "weight-result.json").exists()


def test_verify_run_rejects_noncanonical_caller_spec_before_artifact_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    canonical = contract.spec_for_fold(specs, 5)
    tampered = replace(canonical, sha256="f" * 64)
    root = tmp_path / "artifacts"
    reproduction = tmp_path / "reproduction"
    run = root / "fold5-weight-run"
    weight = reproduction / ".local" / "released-weights" / tampered.filename
    run.mkdir(parents=True)
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"tampered")
    monkeypatch.setattr(verifier, "EXPECTED_WEIGHT_ROOT", root)
    monkeypatch.setattr(verifier, "REPRODUCTION_ROOT", reproduction)
    monkeypatch.setattr(verifier, "validate_local_weight", lambda *_args: None)

    with pytest.raises(ValueError, match="canonical pinned manifest spec"):
        verifier.verify_run(run, weight, tampered)


def test_non_fold2_provenance_rejects_legacy_controller_hash(
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    fold5 = contract.spec_for_fold(specs, 5)

    authoritative, launch_time = verifier.build_weight_provenance_contract(fold5)

    assert authoritative["evaluate_released_weight.py"] == Path(
        verifier.__file__
    ).with_name("evaluate_released_weight.py")
    assert "evaluate_released_weight.py" not in launch_time


def test_verifier_cli_rejects_output_outside_run_without_rewriting_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "fold2-weight-run"
    run.mkdir()
    output = tmp_path / "other-run" / "independent-verification.json"
    output.parent.mkdir()
    output.write_text("sentinel\n", encoding="utf-8")
    called = False

    def fake_verify(*_args) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "pass"}

    monkeypatch.setattr(verifier, "verify_run", fake_verify)

    with pytest.raises(ValueError, match="output must be inside the selected run"):
        verifier.main(
            [
                "--fold-id",
                "2",
                "--run-directory",
                str(run),
                "--weight-path",
                str(tmp_path / "fold2_testAP0.571.pth"),
                "--output",
                str(output),
            ]
        )
    assert called is False
    assert output.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        (
            "offline-preflight-augmentation.json",
            {
                "status": "pass",
                "augmentation_scope": "offline filename-manifest provenance only",
                "launch_preflight_sha256": "wrong-run",
                "filename_manifest": {},
                "scientific_child_relaunched": False,
            },
            "preflight augmentation",
        ),
        (
            "offline-full-dependency-augmentation.json",
            {
                "status": "pass",
                "augmentation_scope": "offline current full-run dependency hashes only",
                "launch_recorded_dependency": {"run_directory": "wrong-run"},
                "current_dependency": {"run_directory": "wrong-run"},
                "scientific_child_relaunched": False,
            },
            "full dependency augmentation",
        ),
        (
            "offline-finalization.json",
            {
                "status": "pass",
                "finalization_scope": "existing scientific output only",
                "weight_result_sha256": "wrong-result",
                "scientific_child_relaunched": False,
            },
            "offline finalization",
        ),
    ],
)
def test_present_offline_markers_fail_closed_on_cross_run_payloads(
    tmp_path: Path, filename: str, payload: dict[str, object], message: str
) -> None:
    run = tmp_path / "fold2-weight-run"
    run.mkdir()
    preflight = run / "preflight.json"
    result = run / "weight-result.json"
    _write_json(preflight, {"full_run_dependency": {"run_directory": "launch-run"}})
    _write_json(result, {"status": "pass"})
    _write_json(run / filename, payload)

    with pytest.raises(ValueError, match=message):
        verifier.verify_retrospective_markers(
            run,
            filename_manifest=verifier.derive_filename_manifest_evidence(),
            launch_recorded_dependency={"run_directory": "launch-run"},
            current_dependency={"run_directory": "current-run"},
        )


def test_public_verify_run_rejects_tampered_present_finalization_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    spec = contract.spec_for_fold(specs, 5)
    root = tmp_path / "artifacts"
    reproduction = tmp_path / "reproduction"
    derived = tmp_path / "derived"
    run = root / "fold5-weight-run"
    weight = reproduction / ".local" / "released-weights" / spec.filename
    run.mkdir(parents=True)
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    command = verifier.build_expected_weight_command(run, derived, weight, spec)
    command_hash = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    launch = {"command": command, "command_sha256": command_hash}
    claims = {
        "fold_id": spec.fold_id,
        "train_years": list(spec.train_years),
        "validation_year": spec.validation_year,
        "test_year": spec.test_year,
        "weight_path": str(weight.resolve()),
        "weight_filename": spec.filename,
        "weight_size": spec.size,
        "weight_sha256": spec.sha256,
        "filename_ap": spec.filename_ap,
    }
    filename_manifest = verifier.derive_filename_manifest_evidence()
    current_dependency = {"run_directory": "current"}
    _write_json(root / "fold5-weight-evaluation.lock.json", launch)
    _write_json(run / "launch.lock.json", launch)
    _write_json(
        run / "preflight.json",
        {
            **launch,
            **claims,
            "filename_manifest": filename_manifest,
            "full_run_dependency": current_dependency,
        },
    )
    _write_json(run / "started.json", {**launch, "pid": 7})
    _write_json(run / "effective-command.json", launch)
    _write_json(run / "weight-result.json", {**claims, "command_sha256": command_hash})
    _write_json(run / "completed.json", {"status": "pass", "exit_code": 0, "pid": 7})
    _write_json(
        run / "offline-finalization.json",
        {
            "status": "pass",
            "finalization_scope": "existing scientific output only",
            "weight_result_sha256": "cross-run-result",
            "scientific_child_relaunched": False,
        },
    )
    monkeypatch.setattr(verifier, "EXPECTED_WEIGHT_ROOT", root)
    monkeypatch.setattr(verifier, "REPRODUCTION_ROOT", reproduction)
    monkeypatch.setattr(verifier, "EXPECTED_DERIVED_UPSTREAM", derived)
    monkeypatch.setattr(verifier, "validate_local_weight", lambda *_args: None)
    monkeypatch.setattr(
        verifier,
        "capture_weight_raw_manifest",
        lambda *_args: {"schema_version": 1, "entry_count": 0, "entries": []},
    )
    monkeypatch.setattr(
        verifier,
        "_verify_weight_runtime_provenance",
        lambda *_args: {"provenance_copies_verified": True},
    )
    monkeypatch.setattr(
        verifier, "_verify_full_dependency", lambda *_args: current_dependency
    )

    with pytest.raises(ValueError, match="offline finalization"):
        verifier.verify_run(run, weight, spec)

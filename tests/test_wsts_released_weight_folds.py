import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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


def test_recovery_path_security_rejects_hardlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    sealed = root / "sealed.json"
    sealed.write_text("sealed\n", encoding="utf-8")
    hardlink = root / "hardlink.json"
    os.link(sealed, hardlink)

    with pytest.raises(ValueError, match="hardlink"):
        controller.require_sealed_regular_file(
            sealed, checked_root=root, description="sealed test file"
        )


def test_recovery_path_security_rejects_dangling_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    dangling = root / "dangling-target"
    try:
        os.symlink(root / "missing", dangling)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="aliased|reparse"):
        controller.require_absent_recovery_path(
            dangling, checked_root=root, description="absent target"
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction capability test")
def test_recovery_path_security_rejects_windows_junction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    actual = tmp_path / "actual"
    actual.mkdir()
    junction = root / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(actual)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("junction creation unavailable")
    sealed = actual / "sealed.json"
    sealed.write_text("sealed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="aliased|reparse"):
        controller.require_sealed_regular_file(
            junction / "sealed.json",
            checked_root=root,
            description="junction-backed file",
        )


def _write_output_collection_lineage(
    run: Path, derived: Path, *, pid: int = 7
) -> tuple[list[str], str]:
    command = ["fixed-python", "official-entrypoint", "--test-only"]
    command_hash = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(
        run / "started.json",
        {
            "command": command,
            "command_sha256": command_hash,
            "pid": pid,
            "started_utc": "2026-08-12T05:00:00+00:00",
        },
    )
    _write_json(
        run / "effective-command.json",
        {
            "command": command,
            "command_sha256": command_hash,
            "cwd": str(derived.resolve()),
            "provenance_sha256": {
                "evaluate_released_weight.py": "a" * 64,
                "official_weight_entrypoint.py": "b" * 64,
            },
        },
    )
    _write_json(run / "completed.json", {"status": "pass", "exit_code": 0, "pid": pid})
    _write_json(
        run / "preflight.json",
        {
            "runtime_patch": {
                "derived_root": str(derived.resolve()),
                "git_diff_exact_match": True,
                "scientific_code_touched": False,
            }
        },
    )
    (run / "exit-code.txt").write_text("0\n", encoding="utf-8")
    for name in (
        "stdout.log",
        "stderr.log",
        "stream-events.jsonl",
        "gpu.csv",
        "config.yaml",
        "source-data-pre.json",
        "source-data-post.json",
        "launch.lock.json",
    ):
        (run / name).write_text(f"sealed {name}\n", encoding="utf-8")
    exit_ns = int(datetime(2026, 8, 12, 5, 2, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(run / "exit-code.txt", ns=(exit_ns, exit_ns))
    return command, command_hash


def _fake_external_recovery_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    authorization = tmp_path / "external-recovery-authorization.json"
    authorization.write_text('{"schema_version":1}\n', encoding="utf-8")
    monkeypatch.setattr(
        controller,
        "validate_external_recovery_authorization",
        lambda *_args, **_kwargs: {"schema_version": 1},
    )
    return authorization


@pytest.mark.parametrize("tamper", ["valid", "resume-bytes", "reviewed-bytes"])
def test_evaluator_reconstructs_resume_and_reviewed_authorization_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    tamper: str,
) -> None:
    campaign_root = tmp_path / "campaigns"
    campaign_directory = campaign_root / "campaign"
    run = tmp_path / "run"
    derived = tmp_path / "derived"
    campaign_directory.mkdir(parents=True)
    run.mkdir()
    derived.mkdir()
    source = derived / controller.OFFICIAL_TEST_OUTPUT_SOURCE_NAME
    source.write_bytes(b"exact-source")
    target = run / controller.OFFICIAL_TEST_OUTPUT_TARGET_NAME
    for path in (
        campaign_directory / "schedule.json",
        campaign_root / "official-weight-12fold-campaign.lock.json",
        campaign_directory / "fold0.lock.json",
        campaign_directory / "failure.json",
    ):
        path.write_text(f"sealed {path.name}\n", encoding="utf-8")
    reviewed_path = campaign_directory / "fold0-recovery-reviewed-authorization.json"
    reviewed_payload = {"status": "APPROVED", "reviewed_commit": "f" * 40}
    reviewed_path.write_text(json.dumps(reviewed_payload) + "\n", encoding="utf-8")
    source_stat = source.stat()
    source_identity = {
        "path": str(source.absolute()),
        "bytes": source_stat.st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ctime_ns": source_stat.st_ctime_ns,
        "mtime_ns": source_stat.st_mtime_ns,
    }
    reviewed = {
        "seal": {
            "path": str(reviewed_path.absolute()),
            "bytes_hex": reviewed_path.read_bytes().hex(),
            "size": reviewed_path.stat().st_size,
            "sha256": hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
        },
        "payload": reviewed_payload,
    }
    resume_path = campaign_directory / "resume.lock.json"
    resume_payload = {
        "schema_version": 2,
        "campaign_id": campaign_directory.name,
        "campaign_directory": str(campaign_directory.absolute()),
        "resume_scope": "Fold 0 offline finalization then exact remaining schedule",
        "fold0_run_directory": str(run.absolute()),
        "fold0_scientific_child_relaunched": False,
        "automatic_retry_performed": False,
        "recovery_code_commit": "f" * 40,
        "original_failure": controller._file_seal(campaign_directory / "failure.json"),
        "source_output": source_identity,
        "target_output": str(target.absolute()),
        "target_output_absent_before_resume": True,
        "reviewed_authorization": reviewed,
    }
    resume_path.write_text(json.dumps(resume_payload) + "\n", encoding="utf-8")
    resume = {
        "path": str(resume_path.absolute()),
        "bytes_hex": resume_path.read_bytes().hex(),
        "size": resume_path.stat().st_size,
        "sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
    }
    token = {
        "schema_version": 1,
        "status": "authorized",
        "authorized_action": "one-time Fold 0 output recovery without scientific relaunch",
        "campaign_id": campaign_directory.name,
        "campaign_directory": str(campaign_directory.absolute()),
        "fold_id": 0,
        "run_directory": str(run.absolute()),
        "schedule": controller._file_seal(campaign_directory / "schedule.json"),
        "root_lock": controller._file_seal(
            campaign_root / "official-weight-12fold-campaign.lock.json"
        ),
        "fold0_lock": controller._file_seal(campaign_directory / "fold0.lock.json"),
        "original_failure": controller._file_seal(campaign_directory / "failure.json"),
        "source_output": source_identity,
        "target_output": {"path": str(target.absolute()), "absent": True},
        "resume_lock": resume,
        "reviewed_authorization": reviewed,
    }
    token_path = campaign_directory / controller.EXTERNAL_RECOVERY_AUTHORIZATION_NAME
    token_path.write_text(json.dumps(token) + "\n", encoding="utf-8")
    monkeypatch.setattr(controller, "CAMPAIGN_ARTIFACTS_ROOT", campaign_root)
    monkeypatch.setattr(controller, "RECOVERY_CAMPAIGN_ID", campaign_directory.name)
    monkeypatch.setattr(
        controller,
        "validate_reviewed_authorization_manifest",
        lambda *_args, **_kwargs: dict(reviewed_payload),
        raising=False,
    )
    if tamper == "resume-bytes":
        resume_path.write_text('{"changed":true}\n', encoding="utf-8")
    elif tamper == "reviewed-bytes":
        reviewed_path.write_text('{"changed":true}\n', encoding="utf-8")

    if tamper == "valid":
        assert controller.validate_external_recovery_authorization(
            token_path,
            run=run,
            spec=contract.spec_for_fold(specs, 0),
            source=source,
            target=target,
            source_candidate={
                "bytes": source_stat.st_size,
                "sha256": source_identity["sha256"],
                "ctime_ns": source_stat.st_ctime_ns,
                "mtime_ns": source_stat.st_mtime_ns,
            },
        ) == token
    else:
        with pytest.raises(ValueError, match="authorization"):
            controller.validate_external_recovery_authorization(
                token_path,
                run=run,
                spec=contract.spec_for_fold(specs, 0),
                source=source,
                target=target,
                source_candidate={
                    "bytes": source_stat.st_size,
                    "sha256": source_identity["sha256"],
                    "ctime_ns": source_stat.st_ctime_ns,
                    "mtime_ns": source_stat.st_mtime_ns,
                },
            )


def test_live_output_collection_refuses_orphan_then_atomically_attributes_exact_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    orphan = derived / "test_pr_curve_data.npz"
    orphan.write_bytes(b"orphan")

    with pytest.raises(ValueError, match="preexisting official test output"):
        controller.write_official_test_output_pre_inventory(run)

    orphan.unlink()
    controller.write_official_test_output_pre_inventory(run)
    _, command_hash = _write_output_collection_lineage(run, derived)
    source = derived / "test_pr_curve_data.npz"
    source.write_bytes(b"new-fold-output")
    source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(source, ns=(source_ns, source_ns))

    provenance = controller.collect_official_test_output(run, recovery=False)

    target = run / "official-test-pr-curve-data.npz"
    assert not source.exists()
    assert target.read_bytes() == b"new-fold-output"
    assert provenance["collection_mode"] == "live-postprocess"
    assert provenance["command_sha256"] == command_hash
    assert provenance["child_pid"] == 7
    assert provenance["atomic_move"] is True
    assert provenance["pre_inventory"]["candidates"] == []
    assert len(provenance["post_inventory"]["candidates"]) == 1


def test_recovery_output_collection_requires_exact_lineage_and_one_time_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    _, command_hash = _write_output_collection_lineage(run, derived, pid=18568)
    source = derived / "test_pr_curve_data.npz"
    source.write_bytes(b"preserved-fold0")
    source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(source, ns=(source_ns, source_ns))

    fold0 = contract.spec_for_fold(specs, 0)
    external = _fake_external_recovery_authorization(tmp_path, monkeypatch)
    provenance = controller.collect_official_test_output(
        run, recovery=True, spec=fold0, recovery_authorization=external
    )

    authorization = json.loads(
        (run / "official-test-output-recovery-authorization.json").read_text(
            encoding="utf-8"
        )
    )
    assert authorization["authorized_action"] == (
        "one-time atomic adoption of the existing completed child output only"
    )
    assert authorization["scientific_child_relaunched"] is False
    assert authorization["child_pid"] == 18568
    assert authorization["command_sha256"] == command_hash
    assert authorization["fold_id"] == 0
    assert authorization["weight"]["weight_sha256"] == fold0.sha256
    assert authorization["raw_evidence_manifest"]["entry_count"] == 13
    assert len(authorization["raw_evidence_manifest_sha256"]) == 64
    assert authorization["launch_provenance_sha256"] == {
        "evaluate_released_weight.py": "a" * 64,
        "official_weight_entrypoint.py": "b" * 64,
    }
    assert provenance["collection_mode"] == "offline-finalize-existing"
    assert provenance["recovery_authorization_sha256"] == controller._sha256(
        run / "official-test-output-recovery-authorization.json"
    )
    with pytest.raises((FileExistsError, ValueError), match="already|one-time"):
        controller.collect_official_test_output(
            run, recovery=True, spec=fold0, recovery_authorization=external
        )


def test_output_collection_rejects_ambiguous_candidate_and_target_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    controller.write_official_test_output_pre_inventory(run)
    _write_output_collection_lineage(run, derived)
    (derived / "test_pr_curve_data.npz").write_bytes(b"exact")
    (derived / "test_pr_curve_data-copy.npz").write_bytes(b"ambiguous")
    for path in derived.glob("test_pr_curve_data*.npz"):
        source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        os.utime(path, ns=(source_ns, source_ns))
    with pytest.raises(ValueError, match="exactly one newly created"):
        controller.collect_official_test_output(run, recovery=False)

    (derived / "test_pr_curve_data-copy.npz").unlink()
    (run / "official-test-pr-curve-data.npz").write_bytes(b"sentinel")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        controller.collect_official_test_output(run, recovery=False)
    assert (run / "official-test-pr-curve-data.npz").read_bytes() == b"sentinel"


def test_independent_verifier_validates_output_collection_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    monkeypatch.setattr(verifier, "EXPECTED_DERIVED_UPSTREAM", derived)
    controller.write_official_test_output_pre_inventory(run)
    _write_output_collection_lineage(run, derived)
    source = derived / "test_pr_curve_data.npz"
    source.write_bytes(b"independently-verified")
    source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(source, ns=(source_ns, source_ns))
    controller.collect_official_test_output(run, recovery=False)

    evidence = verifier.verify_official_test_output_provenance(run, fold_id=5)

    assert evidence["official_test_output_provenance_verified"] is True
    assert evidence["official_test_output_collection_mode"] == "live-postprocess"
    marker = run / "official-test-output-provenance.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["child_pid"] = 999
    _write_json(marker, payload)
    with pytest.raises(ValueError, match="output provenance"):
        verifier.verify_official_test_output_provenance(run, fold_id=5)


@pytest.mark.parametrize(
    "tamper",
    [
        "top-extra",
        "schema-float",
        "collected-invalid",
        "pre-extra",
        "post-extra",
        "candidate-extra",
        "candidate-bytes-float",
        "candidate-ctime-bool",
        "candidate-path",
    ],
)
def test_schema_v2_provenance_requires_exact_nested_schema_and_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    monkeypatch.setattr(verifier, "EXPECTED_DERIVED_UPSTREAM", derived)
    controller.write_official_test_output_pre_inventory(run)
    _write_output_collection_lineage(run, derived)
    source = derived / "test_pr_curve_data.npz"
    source.write_bytes(b"exact-schema-v2")
    source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(source, ns=(source_ns, source_ns))
    controller.collect_official_test_output(run, recovery=False)
    marker_path = run / "official-test-output-provenance.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    candidate = marker["post_inventory"]["candidates"][0]
    if tamper == "top-extra":
        marker["unreviewed"] = True
    elif tamper == "schema-float":
        marker["schema_version"] = 2.0
    elif tamper == "collected-invalid":
        marker["collected_utc"] = "not-a-time"
    elif tamper == "pre-extra":
        marker["pre_inventory"]["unreviewed"] = True
    elif tamper == "post-extra":
        marker["post_inventory"]["unreviewed"] = True
    elif tamper == "candidate-extra":
        candidate["unreviewed"] = True
    elif tamper == "candidate-bytes-float":
        candidate["bytes"] = float(candidate["bytes"])
    elif tamper == "candidate-ctime-bool":
        candidate["ctime_ns"] = True
    else:
        candidate["path"] = str(tmp_path / "another.npz")
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="provenance.*schema|inventory"):
        verifier.verify_official_test_output_provenance(run, fold_id=5)


@pytest.mark.parametrize("tamper", ["extra", "bytes-float", "moved-invalid"])
def test_legacy_fold2_provenance_requires_exact_schema_types_and_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(verifier, "EXPECTED_DERIVED_UPSTREAM", derived)
    _, command_hash = _write_output_collection_lineage(run, derived, pid=37668)
    target = run / "official-test-pr-curve-data.npz"
    target.write_bytes(b"legacy-fold2")
    marker_path = run / "official-test-output-provenance.json"
    marker = {
        "bytes": target.stat().st_size,
        "child_pid": 37668,
        "command_sha256": command_hash,
        "moved_utc": "2026-08-10T13:44:23.2709753Z",
        "reason": (
            "official test emitted evaluation output in the derived runtime cwd; "
            "preserved unchanged inside the immutable run evidence directory so "
            "source-integrity verification can require the exact authorized patch only"
        ),
        "sha256": verifier._sha256(target),
        "source": str(derived.resolve() / "test_pr_curve_data.npz"),
        "status": "preserved-official-test-output",
        "target": str(target.resolve()),
    }
    if tamper == "extra":
        marker["unreviewed"] = True
    elif tamper == "bytes-float":
        marker["bytes"] = float(marker["bytes"])
    else:
        marker["moved_utc"] = "not-a-time"
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="legacy contract"):
        verifier.verify_official_test_output_provenance(run, fold_id=2)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-10T13:44:23.2Z",
        "2026-08-10T13:44:23.270975Z",
        "2026-08-10T13:44:23.2709753Z",
        "2026-08-10T13:44:23.270975300Z",
        "2026-08-10T13:44:23.2709753+00:00",
    ],
)
def test_iso8601_validator_accepts_one_to_nine_fractional_digits(
    timestamp: str,
) -> None:
    assert verifier._is_aware_iso8601(timestamp) is True


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-10T13:44:23.2709753000Z",
        "2026-08-10T13:44:23.2709753",
        "2026-08-10T13:44:23.2709753-00:00",
        "2026-08-10T07:44:23.2709753-06:00",
        "2026-08-10T14:44:23.2709753+01:00",
        "2026-02-30T13:44:23.2709753Z",
        "not-a-time",
    ],
)
def test_iso8601_validator_rejects_invalid_or_unaware_timestamps(
    timestamp: str,
) -> None:
    assert verifier._is_aware_iso8601(timestamp) is False


def test_iso8601_validator_accepts_seven_fractional_digits_in_fixed_runtime() -> None:
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(SCRIPTS_DIR)!r});"
        "import verify_released_weight as verifier;"
        "print(verifier._is_aware_iso8601('2026-08-10T13:44:23.2709753Z'))"
    )
    result = subprocess.run(
        [str(controller.ENVIRONMENT_PYTHON), "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_independent_verifier_requires_exact_live_pre_inventory_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    monkeypatch.setattr(verifier, "EXPECTED_DERIVED_UPSTREAM", derived)
    controller.write_official_test_output_pre_inventory(run)
    _write_output_collection_lineage(run, derived)
    source = derived / "test_pr_curve_data.npz"
    source.write_bytes(b"live-pre-inventory")
    source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(source, ns=(source_ns, source_ns))
    controller.collect_official_test_output(run, recovery=False)
    pre_path = run / "official-test-output-cwd-pre.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre["status"] = "tampered"
    _write_json(pre_path, pre)
    marker_path = run / "official-test-output-provenance.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["pre_inventory"]["pre_inventory_sha256"] = verifier._sha256(pre_path)
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="live pre-inventory"):
        verifier.verify_official_test_output_provenance(run, fold_id=5)


@pytest.mark.parametrize("tamper", ["status", "extra-field", "raw", "provenance"])
def test_independent_verifier_requires_exact_recovery_authorization_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    tamper: str,
) -> None:
    derived = tmp_path / "derived"
    run = tmp_path / "run"
    derived.mkdir()
    run.mkdir()
    monkeypatch.setattr(controller, "DERIVED_UPSTREAM_ROOT", derived)
    monkeypatch.setattr(verifier, "EXPECTED_DERIVED_UPSTREAM", derived)
    _write_output_collection_lineage(run, derived, pid=18568)
    source = derived / "test_pr_curve_data.npz"
    source.write_bytes(b"recovery-authorization")
    source_ns = int(datetime(2026, 8, 12, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    os.utime(source, ns=(source_ns, source_ns))
    external = _fake_external_recovery_authorization(tmp_path, monkeypatch)
    controller.collect_official_test_output(
        run,
        recovery=True,
        spec=contract.spec_for_fold(specs, 0),
        recovery_authorization=external,
    )
    assert verifier.verify_official_test_output_provenance(run, fold_id=0)[
        "official_test_output_provenance_verified"
    ] is True
    authorization_path = run / "official-test-output-recovery-authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if tamper == "status":
        authorization["status"] = "tampered"
    elif tamper == "extra-field":
        authorization["unreviewed"] = True
    elif tamper == "raw":
        (run / "stdout.log").write_text("changed raw bytes\n", encoding="utf-8")
    else:
        effective_path = run / "effective-command.json"
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
        effective["provenance_sha256"]["evaluate_released_weight.py"] = "c" * 64
        _write_json(effective_path, effective)
    _write_json(authorization_path, authorization)
    marker_path = run / "official-test-output-provenance.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["recovery_authorization_sha256"] = verifier._sha256(authorization_path)
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="recovery authorization"):
        verifier.verify_official_test_output_provenance(run, fold_id=0)


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


def test_fold0_provenance_accepts_only_exact_historical_controller_copy(
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    fold0 = contract.spec_for_fold(specs, 0)

    authoritative, launch_time = verifier.build_weight_provenance_contract(fold0)

    assert authoritative["official_weight_entrypoint.py"] == Path(
        verifier.__file__
    ).with_name("official_weight_entrypoint.py")
    assert "evaluate_released_weight.py" not in authoritative
    assert launch_time == {
        "evaluate_released_weight.py": (
            "84db331675557425c1af7f98173bb623ae08088476a66cc14d5d0c0d05958b23"
        )
    }


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

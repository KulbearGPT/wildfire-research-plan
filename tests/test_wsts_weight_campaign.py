import sys
import json
import hashlib
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import released_weight_contract as contract  # noqa: E402
import run_weight_campaign as campaign  # noqa: E402
import released_weight_path_security as path_security  # noqa: E402
import verify_released_weight as fold_verifier  # noqa: E402


@pytest.fixture(scope="module")
def specs() -> tuple[contract.WeightSpec, ...]:
    return contract.load_pinned_manifest(
        REPOSITORY_ROOT
        / "reproductions"
        / "wsts_res18_unet_t1"
        / "official_weights_manifest.json"
    )


def test_schedule_is_exact_and_adopts_only_fold2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    adopted = (tmp_path / "fold2-weight-existing").resolve()
    monkeypatch.setattr(campaign, "ADOPTED_FOLD2_RUN", adopted)

    schedule = campaign.build_campaign_schedule(specs, {2: adopted})

    assert tuple(item.spec.fold_id for item in schedule) == tuple(range(12))
    assert tuple(item.mode for item in schedule) == (
        "launch", "launch", "adopt", "launch", "launch", "launch",
        "launch", "launch", "launch", "launch", "launch", "launch",
    )
    assert schedule[2].run_directory == adopted
    assert all(item.run_directory is None for item in schedule if item.spec.fold_id != 2)


@pytest.mark.parametrize("tamper", ["valid", "wrong-commit", "extra", "file-sha", "untracked-python"])
def test_external_reviewed_authorization_requires_exact_approved_clean_commit(
    tmp_path: Path, tamper: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=repository, check=True)
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    scripts = repository / "scripts"
    scripts.mkdir()
    reviewed_paths = []
    for name in (
        "run_weight_campaign.py",
        "evaluate_released_weight.py",
        "verify_released_weight.py",
        "released_weight_path_security.py",
    ):
        path = scripts / name
        path.write_text(f"# reviewed {name}\n", encoding="utf-8")
        reviewed_paths.append(path.absolute())
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "reviewed"], cwd=repository, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True
    ).stdout.strip()
    campaign_directory = repository / "artifacts" / "campaign"
    run = repository / "artifacts" / "run"
    campaign_directory.mkdir(parents=True)
    run.mkdir()
    approval_path = campaign_directory / "fold0-recovery-reviewed-authorization.json"
    payload = {
        "schema_version": 1,
        "status": "APPROVED",
        "approval_scope": "exact Fold 0 offline recovery code after independent review",
        "campaign_id": campaign_directory.name,
        "campaign_directory": str(campaign_directory.absolute()),
        "fold_id": 0,
        "run_directory": str(run.absolute()),
        "reviewed_commit": commit,
        "reviewed_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in reviewed_paths
        ],
    }
    if tamper == "wrong-commit":
        payload["reviewed_commit"] = "f" * 40
    elif tamper == "extra":
        payload["self_approved"] = True
    elif tamper == "file-sha":
        payload["reviewed_files"][0]["sha256"] = "0" * 64
    approval_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if tamper == "untracked-python":
        (scripts / "injected.py").write_text("raise SystemExit(9)\n", encoding="utf-8")

    if tamper == "valid":
        assert path_security.validate_reviewed_authorization_manifest(
            approval_path,
            expected_path=approval_path,
            repository_root=repository,
            campaign_directory=campaign_directory,
            run_directory=run,
            reviewed_paths=reviewed_paths,
        ) == payload
    else:
        with pytest.raises(ValueError, match="reviewed authorization"):
            path_security.validate_reviewed_authorization_manifest(
                approval_path,
                expected_path=approval_path,
                repository_root=repository,
                campaign_directory=campaign_directory,
                run_directory=run,
                reviewed_paths=reviewed_paths,
            )


def test_resume_cli_requires_explicit_reviewed_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def resume(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "pass"}

    monkeypatch.setattr(campaign, "resume_existing_campaign", resume)

    with pytest.raises(ValueError, match="reviewed authorization"):
        campaign.main(["--resume-existing", str(tmp_path / "campaign")])

    assert called is False


@pytest.mark.parametrize("adopted", [{}, {1: Path("wrong")}, {2: Path("a"), 3: Path("b")}])
def test_schedule_rejects_any_adoption_other_than_exact_fold2(
    specs: tuple[contract.WeightSpec, ...], adopted: dict[int, Path]
) -> None:
    with pytest.raises(ValueError, match="exactly Fold 2"):
        campaign.build_campaign_schedule(specs, adopted)


def test_qualify_fold2_requires_generic_pass_before_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    spec = contract.spec_for_fold(specs, 2)
    run = (tmp_path / "fold2-weight-sealed").resolve()
    run.mkdir()
    independent = run / "independent-verification.json"
    independent.write_text('{"status":"pass"}\n', encoding="utf-8")
    monkeypatch.setattr(campaign, "ADOPTED_FOLD2_RUN", run)
    monkeypatch.setattr(campaign, "weight_cache_path", lambda _spec: tmp_path / spec.filename)
    def cli(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"status":"fail","fold_id":2}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 0, '{"status":"fail","fold_id":2}\n', ""
        )

    monkeypatch.setattr(campaign.subprocess, "run", cli)

    with pytest.raises(ValueError, match="generic independent verifier did not pass"):
        campaign.qualify_existing_fold2(run, spec)


def _configure_fake_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    verifier_failure: int | None = None,
    child_failure: int | None = None,
) -> tuple[Path, list[int], list[int], list[list[str]]]:
    root = tmp_path / "campaigns"
    campaign_directory = root / "official-weight-12fold-test"
    fold_runs = tmp_path / "fold-runs"
    adopted = fold_runs / "fold2-weight-adopted"
    adopted.mkdir(parents=True)
    adopted_payload = {
        "fold_id": 2,
        "raw_evidence_manifest": {"schema_version": 1, "entries": []},
        "raw_evidence_unchanged_during_verification": True,
        "status": "pass",
    }
    (adopted / "independent-verification.json").write_text(
        json.dumps(adopted_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    scientific_folds: list[int] = []
    qualified_folds: list[int] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(campaign, "CAMPAIGN_ARTIFACTS_ROOT", root)
    monkeypatch.setattr(campaign, "WEIGHT_ARTIFACTS_ROOT", fold_runs)
    monkeypatch.setattr(campaign, "ADOPTED_FOLD2_RUN", adopted.resolve())

    def cli(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        fold_id = int(command[command.index("--fold-id") + 1])
        if Path(command[1]).name == "evaluate_released_weight.py":
            if "--fetch-only" in command:
                payload = {"status": "pass", "fold_id": fold_id}
            elif "--launch" in command:
                scientific_folds.append(fold_id)
                if fold_id == child_failure:
                    return subprocess.CompletedProcess(command, 9, "", "synthetic child failure")
                run = (fold_runs / f"fold{fold_id}-weight-launched").resolve()
                run.mkdir(parents=True)
                lock = fold_runs / f"fold{fold_id}-weight-evaluation.lock.json"
                lock.write_text(json.dumps({"run_directory": str(run)}) + "\n", encoding="utf-8")
                payload = {"status": "pass", "fold_id": fold_id, "run_directory": str(run)}
            else:
                payload = {"status": "pass", "fold_id": fold_id}
        else:
            run = Path(command[command.index("--run-directory") + 1]).resolve()
            output = Path(command[command.index("--output") + 1])
            if fold_id == 2:
                qualified_folds.append(fold_id)
            payload = (
                {"status": "fail", "fold_id": fold_id}
                if fold_id == verifier_failure
                else adopted_payload
                if fold_id == 2
                else {"status": "pass", "fold_id": fold_id}
            )
            output.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            "",
        )

    monkeypatch.setattr(campaign.subprocess, "run", cli)
    return campaign_directory, scientific_folds, qualified_folds, commands


def _authorize_fake_resume_incident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign_directory: Path,
) -> Path:
    failure = campaign_directory / "failure.json"
    evaluator_lock = json.loads(
        (
            campaign.WEIGHT_ARTIFACTS_ROOT
            / "fold0-weight-evaluation.lock.json"
        ).read_text(encoding="utf-8")
    )
    fold0_run = Path(evaluator_lock["run_directory"]).resolve()
    source = tmp_path / "derived" / "test_pr_curve_data.npz"
    source.parent.mkdir()
    source.write_bytes(b"exact-fold0-output")
    source_stat = source.stat()
    monkeypatch.setattr(
        campaign, "RECOVERY_CAMPAIGN_ID", campaign_directory.name, raising=False
    )
    monkeypatch.setattr(
        campaign, "RECOVERY_FOLD0_RUN_NAME", fold0_run.name, raising=False
    )
    monkeypatch.setattr(
        campaign, "RECOVERY_FAILURE_BYTES", failure.stat().st_size, raising=False
    )
    monkeypatch.setattr(
        campaign,
        "RECOVERY_FAILURE_SHA256",
        hashlib.sha256(failure.read_bytes()).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(campaign, "RECOVERY_OUTPUT_SOURCE", source, raising=False)
    monkeypatch.setattr(
        campaign, "RECOVERY_OUTPUT_BYTES", source_stat.st_size, raising=False
    )
    monkeypatch.setattr(
        campaign,
        "RECOVERY_OUTPUT_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        campaign, "RECOVERY_OUTPUT_CTIME_NS", source_stat.st_ctime_ns, raising=False
    )
    monkeypatch.setattr(
        campaign, "RECOVERY_OUTPUT_MTIME_NS", source_stat.st_mtime_ns, raising=False
    )
    monkeypatch.setattr(
        campaign, "_recovery_code_commit", lambda: "f" * 40, raising=False
    )
    reviewed_path = campaign_directory / campaign.RECOVERY_REVIEWED_AUTHORIZATION_NAME
    reviewed_payload = {
        "schema_version": 1,
        "status": "APPROVED",
        "approval_scope": "exact Fold 0 offline recovery code after independent review",
        "campaign_id": campaign_directory.name,
        "campaign_directory": str(campaign_directory.absolute()),
        "fold_id": 0,
        "run_directory": str(fold0_run.absolute()),
        "reviewed_commit": "f" * 40,
        "reviewed_files": [],
    }
    reviewed_path.write_text(json.dumps(reviewed_payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "_validate_reviewed_authorization",
        lambda *_args, **_kwargs: dict(reviewed_payload),
        raising=False,
    )
    return source


def test_campaign_launches_eleven_folds_sequentially_and_records_no_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, qualified_folds, commands = _configure_fake_campaign(
        tmp_path, monkeypatch
    )
    fold2_independent = campaign.ADOPTED_FOLD2_RUN / "independent-verification.json"
    fold2_bytes_before = fold2_independent.read_bytes()
    fold2_sha_before = campaign.sha256_file(fold2_independent)

    result = campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert qualified_folds == [2]
    assert fold2_independent.read_bytes() == fold2_bytes_before
    assert campaign.sha256_file(fold2_independent) == fold2_sha_before
    launch_commands = [command for command in commands if "--launch" in command]
    verifier_commands = [
        command for command in commands if Path(command[1]).name == "verify_released_weight.py"
    ]
    assert [int(command[command.index("--fold-id") + 1]) for command in launch_commands] == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert [int(command[command.index("--fold-id") + 1]) for command in verifier_commands] == [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert all(Path(command[0]).resolve() == campaign.FIXED_ENVIRONMENT_PYTHON.resolve() for command in commands)
    assert result == {"status": "pass", "fold_count": 12, "next_fold": None}
    schedule = json.loads((campaign_directory / "schedule.json").read_text(encoding="utf-8"))
    assert [row["fold_id"] for row in schedule["folds"]] == list(range(12))
    assert schedule["folds"][2]["mode"] == "adopt"
    for fold_id in range(12):
        state = json.loads(
            (campaign_directory / f"fold{fold_id}.json").read_text(encoding="utf-8")
        )
        assert set(state) == {
            "fold_id", "independent_verification_sha256", "mode", "run_directory", "status"
        }


def test_campaign_stops_before_next_fold_when_verifier_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=4
    )

    with pytest.raises(ValueError, match="fold 4 independent verification failed"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3, 4]
    assert not (campaign_directory / "fold5.json").exists()
    failure = json.loads((campaign_directory / "failure.json").read_text(encoding="utf-8"))
    assert failure["fold_id"] == 4
    assert failure["automatic_retry_performed"] is False


def test_campaign_stops_before_verifier_and_next_fold_when_child_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, child_failure=3
    )

    with pytest.raises(ValueError, match="synthetic child failure"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3]
    assert not (campaign_directory / "fold3.json").exists()
    assert not (campaign_directory / "fold4.lock.json").exists()


@pytest.mark.parametrize("lock_kind", ["global", "campaign", "fold"])
def test_existing_immutable_lock_refuses_campaign_before_any_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_kind: str
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch
    )
    campaign_directory.mkdir(parents=True)
    if lock_kind == "global":
        lock = campaign.CAMPAIGN_ARTIFACTS_ROOT / campaign.GLOBAL_LOCK_NAME
    elif lock_kind == "campaign":
        lock = campaign_directory / "campaign.lock.json"
    else:
        lock = campaign_directory / "fold0.lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="lock already exists"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == []
    assert lock.read_text(encoding="utf-8") == "sentinel\n"


def test_second_campaign_call_cannot_duplicate_a_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch
    )
    campaign.run_campaign(campaign_directory)

    with pytest.raises(FileExistsError, match="lock already exists"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]


@pytest.mark.parametrize(
    ("failure_stage", "expected_stage", "root_owned"),
    [
        ("schedule", "write_schedule", False),
        ("root_lock", "acquire_root_lock", False),
        ("qualify", "qualify_fold2", True),
        ("prepare", "prepare_weight", True),
        ("fold_lock", "acquire_fold_lock", True),
    ],
)
def test_transaction_faults_leave_owned_locks_and_exact_failure_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_stage: str,
    root_owned: bool,
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path,
        monkeypatch,
        verifier_failure=2 if failure_stage == "qualify" else None,
    )
    root_lock = campaign.CAMPAIGN_ARTIFACTS_ROOT / campaign.GLOBAL_LOCK_NAME
    if failure_stage == "root_lock":
        root_lock.parent.mkdir(parents=True, exist_ok=True)
        root_lock.write_text("sentinel\n", encoding="utf-8")

    original_acquire = campaign._acquire_immutable_json
    if failure_stage in {"schedule", "fold_lock"}:
        target_name = "schedule.json" if failure_stage == "schedule" else "fold0.lock.json"

        def fail_acquire(path: Path, payload: object) -> None:
            if path.name == target_name:
                raise OSError(f"synthetic {failure_stage} failure")
            original_acquire(path, payload)

        monkeypatch.setattr(campaign, "_acquire_immutable_json", fail_acquire)

    if failure_stage == "prepare":
        original_cli = campaign.subprocess.run

        def fail_prepare(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if "--fetch-only" in command:
                return subprocess.CompletedProcess(command, 8, "", "synthetic prepare failure")
            return original_cli(command, **kwargs)

        monkeypatch.setattr(campaign.subprocess, "run", fail_prepare)

    with pytest.raises((FileExistsError, OSError, ValueError)):
        campaign.run_campaign(campaign_directory)

    failure = json.loads(
        (campaign_directory / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["stage"] == expected_stage
    assert failure["root_lock_owned_by_campaign"] is root_owned
    assert failure["automatic_retry_performed"] is False
    assert failure["partial_aggregate_written"] is False
    assert scientific_folds == []
    assert not (campaign_directory / "fold1.lock.json").exists()
    if root_owned:
        root_payload = json.loads(root_lock.read_text(encoding="utf-8"))
        assert root_payload["campaign_id"] == campaign_directory.name
        assert root_payload["campaign_directory"] == str(campaign_directory.resolve())
    else:
        assert not root_lock.exists() or root_lock.read_text(encoding="utf-8") == "sentinel\n"


def test_finalize_existing_fold_never_launches_a_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    spec = contract.spec_for_fold(specs, 5)
    run = tmp_path / "fold5-weight-existing"
    run.mkdir()
    authorization = tmp_path / "external-recovery-authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    def cli(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert "--launch" not in command
        payload = {"status": "pass", "fold_id": 5}
        if Path(command[1]).name == "verify_released_weight.py":
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, json.dumps(payload) + "\n", "")

    monkeypatch.setattr(campaign.subprocess, "run", cli)

    result = campaign.finalize_existing_fold(run, spec, authorization)

    assert result["status"] == "pass"
    assert result["fold_id"] == 5
    assert [Path(command[1]).name for command in commands] == [
        "evaluate_released_weight.py",
        "verify_released_weight.py",
    ]
    evaluator = commands[0]
    assert evaluator[evaluator.index("--recovery-authorization") + 1] == str(
        authorization.resolve()
    )


def test_generic_finalize_main_preserves_task2_interface_without_recovery_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    called: list[tuple[Path, int, Path | None]] = []

    def finalize(
        run: Path,
        spec: contract.WeightSpec,
        authorization: Path | None = None,
    ) -> dict[str, object]:
        called.append((run, spec.fold_id, authorization))
        return {"status": "pass", "fold_id": spec.fold_id}

    monkeypatch.setattr(campaign, "finalize_existing_fold", finalize)
    monkeypatch.setattr(campaign, "load_pinned_manifest", lambda _path: specs)

    assert campaign.main(
        ["--fold-id", "5", "--finalize-existing", str(tmp_path / "run")]
    ) == 0
    assert called == [(tmp_path / "run", 5, None)]


def test_generic_finalize_subprocess_omits_recovery_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    run = tmp_path / "fold5-weight-existing"
    run.mkdir()
    commands: list[list[str]] = []

    def cli(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        payload = {"status": "pass", "fold_id": 5}
        if Path(command[1]).name == "verify_released_weight.py":
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, json.dumps(payload) + "\n", "")

    monkeypatch.setattr(campaign.subprocess, "run", cli)

    result = campaign.finalize_existing_fold(run, contract.spec_for_fold(specs, 5))

    assert result["status"] == "pass"
    assert "--recovery-authorization" not in commands[0]


def test_campaign_main_rejects_fold0_finalize_without_external_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    called = False

    def finalize(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "pass"}

    monkeypatch.setattr(campaign, "finalize_existing_fold", finalize)
    monkeypatch.setattr(campaign, "load_pinned_manifest", lambda _path: specs)

    with pytest.raises(ValueError, match="Fold 0 recovery"):
        campaign.main(
            ["--fold-id", "0", "--finalize-existing", str(tmp_path / "run")]
        )

    assert called is False


def test_resume_existing_recovers_fold0_without_relaunch_and_continues_same_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=0
    )
    with pytest.raises(ValueError, match="fold 0 independent verification failed"):
        campaign.run_campaign(campaign_directory)
    source = _authorize_fake_resume_incident(tmp_path, monkeypatch, campaign_directory)
    assert scientific_folds == [0]
    original_failure = (campaign_directory / "failure.json").read_bytes()
    fold0_lock = json.loads(
        (campaign.WEIGHT_ARTIFACTS_ROOT / "fold0-weight-evaluation.lock.json").read_text(
            encoding="utf-8"
        )
    )
    fold0_run = Path(fold0_lock["run_directory"]).resolve()
    finalized: list[tuple[int, Path]] = []
    resumed_launches: list[int] = []

    def record(run: Path, spec: contract.WeightSpec) -> dict[str, object]:
        run.mkdir(parents=True, exist_ok=True)
        independent = run / "independent-verification.json"
        independent.write_text(
            json.dumps({"status": "pass", "fold_id": spec.fold_id}) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "pass",
            "fold_id": spec.fold_id,
            "run_directory": str(run.resolve()),
            "independent_verification_sha256": campaign.sha256_file(independent),
        }

    def finalize(
        run: Path, spec: contract.WeightSpec, authorization: Path
    ) -> dict[str, object]:
        assert authorization.is_file()
        finalized.append((spec.fold_id, authorization.resolve()))
        return record(run.resolve(), spec)

    def launch(spec: contract.WeightSpec) -> Path:
        resumed_launches.append(spec.fold_id)
        return (
            campaign.WEIGHT_ARTIFACTS_ROOT
            / f"fold{spec.fold_id}-weight-resumed"
        ).resolve()

    monkeypatch.setattr(campaign, "finalize_existing_fold", finalize)
    monkeypatch.setattr(campaign, "prepare_weight_cli", lambda _spec: {"status": "pass"})
    monkeypatch.setattr(campaign, "launch_one_fold_cli", launch)
    monkeypatch.setattr(campaign, "verify_one_fold_cli", record)
    monkeypatch.setattr(
        campaign, "_validate_recovered_fold0_result", lambda *_args: None
    )
    monkeypatch.setattr(
        campaign,
        "qualify_existing_fold2",
        lambda run, spec: record(run.resolve(), spec),
    )

    result = campaign.resume_existing_campaign(campaign_directory)

    assert len(finalized) == 1
    assert finalized[0][0] == 0
    authorization_path = finalized[0][1]
    assert authorization_path == (
        campaign_directory / "fold0-recovery-authorization.json"
    ).resolve()
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["campaign_id"] == campaign_directory.name
    assert authorization["run_directory"] == str(fold0_run)
    assert authorization["schedule"]["sha256"] == campaign.sha256_file(
        campaign_directory / "schedule.json"
    )
    assert authorization["original_failure"]["sha256"] == campaign.sha256_file(
        campaign_directory / "failure.json"
    )
    reviewed_path = campaign_directory / campaign.RECOVERY_REVIEWED_AUTHORIZATION_NAME
    assert authorization["reviewed_authorization"]["payload"]["status"] == "APPROVED"
    assert authorization["reviewed_authorization"]["seal"] == campaign._exact_file_seal(
        reviewed_path
    )
    assert resumed_launches == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert scientific_folds == [0]
    assert result == {"status": "pass", "fold_count": 12, "next_fold": None}
    assert (campaign_directory / "failure.json").read_bytes() == original_failure
    fold0_state = json.loads(
        (campaign_directory / "fold0.json").read_text(encoding="utf-8")
    )
    assert fold0_state["mode"] == "launch"
    assert fold0_state["recovered_via_finalize_existing"] is True
    recovery = json.loads(
        (campaign_directory / "resume-completed.json").read_text(encoding="utf-8")
    )
    assert recovery["fold0_scientific_child_relaunched"] is False
    assert recovery["original_failure_sha256"] == campaign.sha256_file(
        campaign_directory / "failure.json"
    )
    resume_lock = json.loads(
        (campaign_directory / "resume.lock.json").read_text(encoding="utf-8")
    )
    assert resume_lock["schema_version"] == 2
    assert resume_lock["recovery_code_commit"] == "f" * 40
    assert resume_lock["source_output"] == {
        "path": str(source.resolve()),
        "bytes": len(b"exact-fold0-output"),
        "sha256": hashlib.sha256(b"exact-fold0-output").hexdigest(),
        "ctime_ns": source.stat().st_ctime_ns,
        "mtime_ns": source.stat().st_mtime_ns,
    }
    with pytest.raises(FileExistsError, match="resume lock already exists"):
        campaign.resume_existing_campaign(campaign_directory)


@pytest.mark.parametrize("failure_mode", ["interrupted-after-first-lock", "resume-lock-race"])
def test_resume_never_leaves_a_consumable_token_without_winning_resume_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    campaign_directory, _, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=0
    )
    with pytest.raises(ValueError):
        campaign.run_campaign(campaign_directory)
    _authorize_fake_resume_incident(tmp_path, monkeypatch, campaign_directory)
    original_acquire = campaign._acquire_immutable_json
    acquired: list[str] = []

    def acquire(path: Path, payload: object) -> None:
        acquired.append(path.name)
        if failure_mode == "resume-lock-race":
            path.write_text('{"competitor":true}\n', encoding="utf-8")
            original_acquire(path, payload)
            return
        original_acquire(path, payload)
        raise RuntimeError("synthetic interruption after first immutable lock")

    monkeypatch.setattr(campaign, "_acquire_immutable_json", acquire)

    with pytest.raises((FileExistsError, RuntimeError)):
        campaign.resume_existing_campaign(campaign_directory)

    assert acquired == ["resume.lock.json"]
    assert not (
        campaign_directory / campaign.EXTERNAL_RECOVERY_AUTHORIZATION_NAME
    ).exists()


def test_external_recovery_token_exactly_seals_preexisting_resume_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, _, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=0
    )
    with pytest.raises(ValueError):
        campaign.run_campaign(campaign_directory)
    _authorize_fake_resume_incident(tmp_path, monkeypatch, campaign_directory)
    captured: dict[str, object] = {}

    def stop_before_finalize(
        _run: Path, _spec: contract.WeightSpec, authorization: Path
    ) -> dict[str, object]:
        captured.update(json.loads(authorization.read_text(encoding="utf-8")))
        raise RuntimeError("stop after token creation")

    monkeypatch.setattr(campaign, "finalize_existing_fold", stop_before_finalize)

    with pytest.raises(RuntimeError, match="stop after token"):
        campaign.resume_existing_campaign(campaign_directory)

    resume_path = campaign_directory / "resume.lock.json"
    resume_bytes = resume_path.read_bytes()
    assert captured["resume_lock"] == {
        "path": str(resume_path.absolute()),
        "bytes_hex": resume_bytes.hex(),
        "size": len(resume_bytes),
        "sha256": hashlib.sha256(resume_bytes).hexdigest(),
    }


@pytest.mark.parametrize("tamper_at", ["before-token", "after-token", "before-evaluator"])
def test_resume_revalidates_incident_around_token_and_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_at: str,
) -> None:
    campaign_directory, _, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=0
    )
    with pytest.raises(ValueError):
        campaign.run_campaign(campaign_directory)
    source = _authorize_fake_resume_incident(tmp_path, monkeypatch, campaign_directory)
    original_assert = campaign._assert_recovery_incident_state
    checks = 0
    finalized = False

    def assert_incident(*args: object) -> None:
        nonlocal checks
        checks += 1
        target_check = {"before-token": 1, "after-token": 2, "before-evaluator": 3}[tamper_at]
        if checks == target_check:
            source.write_bytes(b"changed between recovery boundaries")
        original_assert(*args)

    def finalize(*_args: object) -> dict[str, object]:
        nonlocal finalized
        finalized = True
        return {}

    monkeypatch.setattr(campaign, "_assert_recovery_incident_state", assert_incident)
    monkeypatch.setattr(campaign, "finalize_existing_fold", finalize)

    with pytest.raises(ValueError, match="source seal"):
        campaign.resume_existing_campaign(campaign_directory)

    assert checks == {"before-token": 1, "after-token": 2, "before-evaluator": 3}[tamper_at]
    assert finalized is False
    if tamper_at == "before-token":
        assert not (
            campaign_directory / campaign.EXTERNAL_RECOVERY_AUTHORIZATION_NAME
        ).exists()


@pytest.mark.parametrize("tamper_at", ["before-fold1", "before-fold3", "before-complete"])
def test_resume_revalidates_original_failure_at_every_scientific_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_at: str,
) -> None:
    campaign_directory, _, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=0
    )
    with pytest.raises(ValueError):
        campaign.run_campaign(campaign_directory)
    _authorize_fake_resume_incident(tmp_path, monkeypatch, campaign_directory)
    failure = campaign_directory / "failure.json"
    fold0_lock = json.loads(
        (
            campaign.WEIGHT_ARTIFACTS_ROOT / "fold0-weight-evaluation.lock.json"
        ).read_text(encoding="utf-8")
    )
    fold0_run = Path(fold0_lock["run_directory"]).resolve()
    launched: list[int] = []

    def record(run: Path, spec: contract.WeightSpec) -> dict[str, object]:
        run.mkdir(parents=True, exist_ok=True)
        independent = run / "independent-verification.json"
        independent.write_text(
            json.dumps({"status": "pass", "fold_id": spec.fold_id}) + "\n",
            encoding="utf-8",
        )
        if tamper_at == "before-fold3" and spec.fold_id == 1:
            failure.write_bytes(b"tampered after Fold 1\n")
        if tamper_at == "before-complete" and spec.fold_id == 11:
            failure.write_bytes(b"tampered after Fold 11\n")
        return {
            "status": "pass",
            "fold_id": spec.fold_id,
            "run_directory": str(run.resolve()),
            "independent_verification_sha256": campaign.sha256_file(independent),
        }

    def prepare(spec: contract.WeightSpec) -> dict[str, object]:
        if tamper_at == "before-fold1" and spec.fold_id == 11:
            failure.write_bytes(b"tampered before Fold 1\n")
        return {"status": "pass"}

    def launch(spec: contract.WeightSpec) -> Path:
        launched.append(spec.fold_id)
        return (campaign.WEIGHT_ARTIFACTS_ROOT / f"fold{spec.fold_id}-resumed").resolve()

    monkeypatch.setattr(
        campaign,
        "finalize_existing_fold",
        lambda _run, spec, _authorization: record(fold0_run, spec),
    )
    monkeypatch.setattr(campaign, "_validate_recovered_fold0_result", lambda *_args: None)
    monkeypatch.setattr(campaign, "prepare_weight_cli", prepare)
    monkeypatch.setattr(campaign, "launch_one_fold_cli", launch)
    monkeypatch.setattr(campaign, "verify_one_fold_cli", record)
    monkeypatch.setattr(
        campaign,
        "qualify_existing_fold2",
        lambda run, spec: record(run.resolve(), spec),
    )

    with pytest.raises(ValueError, match="original failure"):
        campaign.resume_existing_campaign(campaign_directory)

    if tamper_at == "before-fold1":
        assert launched == []
    elif tamper_at == "before-fold3":
        assert launched == [1]
    else:
        assert launched == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        assert not (campaign_directory / "completed.json").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "schedule",
        "root-lock",
        "future-state",
        "failure-bytes",
        "source",
        "campaign-id",
        "run-id",
        "approval-tamper",
    ],
)
def test_resume_existing_fails_closed_before_finalize_on_tampered_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    campaign_directory, scientific_folds, _, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=0
    )
    with pytest.raises(ValueError):
        campaign.run_campaign(campaign_directory)
    source = _authorize_fake_resume_incident(tmp_path, monkeypatch, campaign_directory)
    called = False

    def finalize(*_args: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(campaign, "finalize_existing_fold", finalize)
    if tamper == "schedule":
        payload = json.loads((campaign_directory / "schedule.json").read_text(encoding="utf-8"))
        payload["folds"][1]["mode"] = "adopt"
        (campaign_directory / "schedule.json").write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "root-lock":
        root_lock = campaign.CAMPAIGN_ARTIFACTS_ROOT / campaign.GLOBAL_LOCK_NAME
        payload = json.loads(root_lock.read_text(encoding="utf-8"))
        payload["campaign_id"] = "different-campaign"
        root_lock.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "future-state":
        (campaign_directory / "fold1.json").write_text("{}\n", encoding="utf-8")
    elif tamper == "failure-bytes":
        failure = campaign_directory / "failure.json"
        failure.write_text(
            json.dumps(json.loads(failure.read_text(encoding="utf-8")), indent=2),
            encoding="utf-8",
        )
    elif tamper == "source":
        source.write_bytes(b"tampered-output")
    elif tamper == "campaign-id":
        monkeypatch.setattr(campaign, "RECOVERY_CAMPAIGN_ID", "other-campaign")
    elif tamper == "run-id":
        monkeypatch.setattr(campaign, "RECOVERY_FOLD0_RUN_NAME", "other-fold0-run")
    else:
        reviewed = campaign_directory / campaign.RECOVERY_REVIEWED_AUTHORIZATION_NAME
        reviewed.write_text('{"status":"SELF-APPROVED"}\n', encoding="utf-8")
        monkeypatch.setattr(
            campaign,
            "_validate_reviewed_authorization",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("reviewed authorization manifest mismatch")
            ),
        )

    with pytest.raises(ValueError, match="resume|future fold state|reviewed authorization"):
        campaign.resume_existing_campaign(campaign_directory)
    assert called is False
    assert scientific_folds == [0]


@pytest.mark.parametrize(
    "tamper", ["valid", "extra-field", "raw", "provenance", "record"]
)
def test_recovered_fold0_result_requires_exact_schema_and_byte_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    tamper: str,
) -> None:
    run = (tmp_path / "fold0-weight-recovered").resolve()
    run.mkdir()
    canonical = contract.spec_for_fold(specs, 0)
    weight = tmp_path / canonical.filename
    weight.write_bytes(b"sealed weight")
    spec = contract.WeightSpec(
        fold_id=canonical.fold_id,
        filename=canonical.filename,
        hub_path=canonical.hub_path,
        size=weight.stat().st_size,
        sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        filename_ap=canonical.filename_ap,
        train_years=canonical.train_years,
        validation_year=canonical.validation_year,
        test_year=canonical.test_year,
    )
    monkeypatch.setattr(campaign, "weight_cache_path", lambda _spec: weight)
    raw_entries = []
    for relative in campaign.RECOVERED_FOLD0_RAW_RELATIVE_PATHS:
        path = run / relative
        path.write_bytes(f"sealed {relative}".encode())
        raw_entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    raw_entries.append(
        {
            "path": "released-weight::" + str(weight.resolve()),
            "bytes": weight.stat().st_size,
            "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        }
    )
    raw_manifest = {
        "schema_version": 1,
        "entry_count": len(raw_entries),
        "entries": raw_entries,
    }
    payload = {}
    for key, expected_type in campaign.RECOVERED_FOLD0_INDEPENDENT_TYPES.items():
        if expected_type is str:
            payload[key] = "evidence"
        elif expected_type is bool:
            payload[key] = True
        elif expected_type is int:
            payload[key] = 1
        elif expected_type is float:
            payload[key] = 1.0
        elif expected_type is dict:
            payload[key] = {}
        elif expected_type is list:
            payload[key] = []
        else:
            payload[key] = None
    target = run / "official-test-pr-curve-data.npz"
    provenance = run / "official-test-output-provenance.json"
    provenance.write_bytes(b"sealed provenance marker")
    payload.update(
        {
            "status": "pass",
            "fold_id": 0,
            "weight_sha256": spec.sha256,
            "command_sha256": "a" * 64,
            "original_upstream_commit": "b" * 40,
            "derived_upstream_commit": "b" * 40,
            "patch_sha256": "c" * 64,
            "raw_evidence_manifest": raw_manifest,
            "raw_evidence_manifest_sha256": hashlib.sha256(
                json.dumps(raw_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "raw_evidence_unchanged_during_verification": True,
            "official_test_output_provenance_verified": True,
            "official_test_output_collection_mode": "offline-finalize-existing",
            "official_test_output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "official_test_output_provenance_sha256": hashlib.sha256(
                provenance.read_bytes()
            ).hexdigest(),
            "independent_implementation": True,
        }
    )
    independent = run / "independent-verification.json"
    independent.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    record = {
        "status": "pass",
        "fold_id": 0,
        "run_directory": str(run),
        "independent_verification_sha256": campaign.sha256_file(independent),
    }
    if tamper == "valid":
        campaign._validate_recovered_fold0_result(run, spec, record)
        return
    if tamper == "extra-field":
        payload["unreviewed"] = True
        independent.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        record["independent_verification_sha256"] = campaign.sha256_file(independent)
    elif tamper == "raw":
        (run / "stdout.log").write_bytes(b"changed")
    elif tamper == "provenance":
        provenance.write_bytes(b"changed provenance")
    else:
        record["unreviewed"] = True

    with pytest.raises(ValueError, match="recovered Fold 0"):
        campaign._validate_recovered_fold0_result(run, spec, record)


def test_resume_cli_is_mutually_exclusive_and_requires_campaign_only() -> None:
    parsed = campaign.parse_arguments(
        ["--resume-existing", "official-weight-12fold-existing"]
    )
    assert parsed.resume_existing == Path("official-weight-12fold-existing")
    with pytest.raises(SystemExit):
        campaign.parse_arguments(
            [
                "--launch",
                "--resume-existing",
                "official-weight-12fold-existing",
            ]
        )


def test_generic_verifier_cli_preserves_previous_output_when_atomic_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "fold2-weight-existing"
    run.mkdir()
    output = run / "independent-verification.json"
    output.write_bytes(b"sentinel-verifier-bytes\n")
    monkeypatch.setattr(
        fold_verifier,
        "verify_run",
        lambda *_args: {"status": "pass", "fold_id": 2},
    )

    def fail_commit(_source: Path, _target: Path) -> None:
        raise OSError("synthetic verifier atomic commit failure")

    monkeypatch.setattr(fold_verifier.os, "replace", fail_commit)
    with pytest.raises(OSError, match="atomic commit failure"):
        fold_verifier.main(
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

    assert output.read_bytes() == b"sentinel-verifier-bytes\n"

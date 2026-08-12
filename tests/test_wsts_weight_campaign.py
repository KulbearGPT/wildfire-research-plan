import sys
import json
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import released_weight_contract as contract  # noqa: E402
import run_weight_campaign as campaign  # noqa: E402
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

    result = campaign.finalize_existing_fold(run, spec)

    assert result["status"] == "pass"
    assert result["fold_id"] == 5
    assert [Path(command[1]).name for command in commands] == [
        "evaluate_released_weight.py",
        "verify_released_weight.py",
    ]


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

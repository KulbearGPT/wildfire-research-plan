"""Run the pinned twelve-fold official-weight evaluation exactly once, sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluate_released_weight import weight_cache_path
from released_weight_contract import (
    WeightSpec,
    load_pinned_manifest,
    spec_for_fold,
)
from released_weight_path_security import (
    require_absent_recovery_path,
    require_pairwise_distinct_files,
    require_sealed_directory,
    require_sealed_regular_file,
)
from run_calibration import ENVIRONMENT_PYTHON as FIXED_ENVIRONMENT_PYTHON


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST_PATH = REPRODUCTION_ROOT / "official_weights_manifest.json"
WEIGHT_ARTIFACTS_ROOT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1-official-weight"
)
CAMPAIGN_ARTIFACTS_ROOT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1-official-weight-12fold"
)
ADOPTED_FOLD2_RUN = (
    WEIGHT_ARTIFACTS_ROOT
    / "fold2-weight-20260810T133336Z-2e071197"
).resolve()
GLOBAL_LOCK_NAME = "official-weight-12fold-campaign.lock.json"
EVALUATOR_CLI = Path(__file__).with_name("evaluate_released_weight.py").absolute()
VERIFIER_CLI = Path(__file__).with_name("verify_released_weight.py").absolute()
PATH_SECURITY_MODULE = Path(__file__).with_name("released_weight_path_security.py").absolute()
RECOVERY_CAMPAIGN_ID = "official-weight-12fold-20260812T052555Z"
RECOVERY_FOLD0_RUN_NAME = "fold0-weight-20260812T053209Z-e7c067f5"
RECOVERY_FAILURE_BYTES = 539
RECOVERY_FAILURE_SHA256 = (
    "29edd7b31ba9d1be8ea01cf23957c9c74474e8319d28b5f715277e9172288a85"
)
RECOVERY_OUTPUT_SOURCE = (
    REPRODUCTION_ROOT
    / ".local"
    / "WildfireSpreadTS-res18-runtime"
    / "test_pr_curve_data.npz"
).absolute()
RECOVERY_OUTPUT_BYTES = 1_976
RECOVERY_OUTPUT_SHA256 = (
    "2bd718e71a22e3723e2a307c2ebdbf9fda5d18e30d3da6b61d9e611f49f671fb"
)
RECOVERY_OUTPUT_CTIME_NS = 1_786_513_307_458_344_000
RECOVERY_OUTPUT_MTIME_NS = 1_786_513_307_459_353_300
RECOVERY_OUTPUT_TARGET_NAME = "official-test-pr-curve-data.npz"
RECOVERY_AUTHORIZATION_NAME = "official-test-output-recovery-authorization.json"
RECOVERY_PROVENANCE_NAME = "official-test-output-provenance.json"
EXTERNAL_RECOVERY_AUTHORIZATION_NAME = "fold0-recovery-authorization.json"
RECOVERED_FOLD0_RAW_RELATIVE_PATHS = (
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
    RECOVERY_OUTPUT_TARGET_NAME,
)
RECOVERED_FOLD0_INDEPENDENT_TYPES = {
    "status": str,
    "fold_id": int,
    "command_sha256": str,
    "weight_sha256": str,
    "loaded_tensor_count": int,
    "test_metrics": dict,
    "filename_manifest": dict,
    "filename_aggregate": dict,
    "wall_seconds": float,
    "gpu_sample_count": float,
    "gpu_interval_count": int,
    "gpu_interval_mean_seconds": float,
    "gpu_interval_median_seconds": float,
    "gpu_observed_effective_hz": float,
    "gpu_peak_is_observed_sample_max": bool,
    "gpu_peak_may_miss_between_sample_transients": bool,
    "peak_gpu_used_mib": float,
    "peak_child_process_mib": (float, type(None)),
    "child_process_memory_available": bool,
    "gpu_utilization_min_percent": float,
    "gpu_utilization_median_percent": float,
    "gpu_utilization_max_percent": float,
    "raw_evidence_manifest": dict,
    "raw_evidence_manifest_sha256": str,
    "raw_evidence_unchanged_during_verification": bool,
    "data_inventory_live_verified": bool,
    "data_file_count": int,
    "data_total_bytes": int,
    "full_run_dependency": dict,
    "test_AP": float,
    "filename_ap": float,
    "filename_ap_absolute_difference": float,
    "original_upstream_commit": str,
    "original_upstream_clean": bool,
    "derived_upstream_commit": str,
    "derived_status": list,
    "patch_sha256": str,
    "provenance_copies_verified": bool,
    "first_verifier_failure_preserved": bool,
    "offline_full_dependency_augmentation_present": bool,
    "official_test_output_provenance_verified": bool,
    "official_test_output_collection_mode": str,
    "official_test_output_sha256": str,
    "official_test_output_provenance_sha256": str,
    "independent_implementation": bool,
}


@dataclass(frozen=True, slots=True)
class CampaignFold:
    spec: WeightSpec
    mode: str
    run_directory: Path | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_seal(path: Path) -> dict[str, object]:
    target = require_sealed_regular_file(
        path, checked_root=path.parent, description="campaign recovery sealed file"
    )
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _acquire_immutable_json(path: Path, payload: object) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"campaign lock already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _canonical_specs(specs: Sequence[WeightSpec]) -> tuple[WeightSpec, ...]:
    canonical = load_pinned_manifest(PINNED_MANIFEST_PATH)
    ordered = tuple(sorted(specs, key=lambda spec: spec.fold_id))
    if ordered != canonical:
        raise ValueError("campaign specs differ from the canonical pinned manifest")
    return ordered


def build_campaign_schedule(
    specs: Sequence[WeightSpec], adopted: Mapping[int, Path]
) -> tuple[CampaignFold, ...]:
    """Return the one legal schedule: folds 0..11, adopting only sealed Fold 2."""
    ordered = _canonical_specs(specs)
    if set(adopted) != {2}:
        raise ValueError("campaign must adopt exactly Fold 2")
    adopted_fold2 = Path(adopted[2]).resolve()
    if adopted_fold2 != ADOPTED_FOLD2_RUN.resolve():
        raise ValueError("campaign must use the fixed sealed Fold 2 run")
    return tuple(
        CampaignFold(
            spec=spec,
            mode="adopt" if spec.fold_id == 2 else "launch",
            run_directory=adopted_fold2 if spec.fold_id == 2 else None,
        )
        for spec in ordered
    )


def _require_generic_pass(result: Mapping[str, object], fold_id: int) -> None:
    if result.get("status") != "pass" or result.get("fold_id") != fold_id:
        raise ValueError(
            f"fold {fold_id} generic independent verifier did not pass"
        )


def _require_campaign_record(result: Mapping[str, object], fold_id: int) -> None:
    if (
        result.get("status") != "pass"
        or result.get("fold_id") != fold_id
        or not isinstance(result.get("run_directory"), str)
        or not isinstance(result.get("independent_verification_sha256"), str)
    ):
        raise ValueError(f"fold {fold_id} independent verification did not pass")


def _validate_recovered_fold0_result(
    run_directory: Path,
    spec: WeightSpec,
    record: Mapping[str, object],
) -> None:
    run = run_directory.resolve()
    independent_path = run / "independent-verification.json"
    independent_sha256 = sha256_file(independent_path)
    expected_record = {
        "status": "pass",
        "fold_id": 0,
        "run_directory": str(run),
        "independent_verification_sha256": independent_sha256,
    }
    if type(record) is not dict or dict(record) != expected_record:
        raise ValueError("recovered Fold 0 campaign record mismatch")
    payload = json.loads(independent_path.read_text(encoding="utf-8"))
    if type(payload) is not dict or set(payload) != set(
        RECOVERED_FOLD0_INDEPENDENT_TYPES
    ):
        raise ValueError("recovered Fold 0 independent result schema mismatch")
    for field, expected_type in RECOVERED_FOLD0_INDEPENDENT_TYPES.items():
        actual_type = type(payload[field])
        allowed = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        if actual_type not in allowed:
            raise ValueError(
                f"recovered Fold 0 independent result type mismatch: {field}"
            )
    target = run / RECOVERY_OUTPUT_TARGET_NAME
    provenance = run / RECOVERY_PROVENANCE_NAME
    if (
        payload["status"] != "pass"
        or payload["fold_id"] != 0
        or payload["weight_sha256"] != spec.sha256
        or payload["raw_evidence_unchanged_during_verification"] is not True
        or payload["official_test_output_provenance_verified"] is not True
        or payload["official_test_output_collection_mode"]
        != "offline-finalize-existing"
        or payload["independent_implementation"] is not True
        or not target.is_file()
        or payload["official_test_output_sha256"] != sha256_file(target)
        or not provenance.is_file()
        or payload["official_test_output_provenance_sha256"]
        != sha256_file(provenance)
    ):
        raise ValueError("recovered Fold 0 independent result identity mismatch")
    raw = payload["raw_evidence_manifest"]
    if (
        type(raw) is not dict
        or set(raw) != {"schema_version", "entry_count", "entries"}
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or type(raw["entry_count"]) is not int
        or type(raw["entries"]) is not list
        or raw["entry_count"] != len(raw["entries"])
    ):
        raise ValueError("recovered Fold 0 raw evidence manifest schema mismatch")
    expected_paths = [
        *RECOVERED_FOLD0_RAW_RELATIVE_PATHS,
        "released-weight::" + str(weight_cache_path(spec).resolve()),
    ]
    if [entry.get("path") for entry in raw["entries"] if type(entry) is dict] != expected_paths:
        raise ValueError("recovered Fold 0 raw evidence paths mismatch")
    for entry in raw["entries"]:
        if (
            type(entry) is not dict
            or set(entry) != {"path", "bytes", "sha256"}
            or type(entry["path"]) is not str
            or type(entry["bytes"]) is not int
            or type(entry["sha256"]) is not str
        ):
            raise ValueError("recovered Fold 0 raw evidence entry schema mismatch")
        if entry["path"].startswith("released-weight::"):
            path = weight_cache_path(spec).resolve()
        else:
            path = run / entry["path"]
        if (
            not path.is_file()
            or entry["bytes"] != path.stat().st_size
            or entry["sha256"] != sha256_file(path)
        ):
            raise ValueError("recovered Fold 0 raw evidence bytes mismatch")
    raw_sha256 = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload["raw_evidence_manifest_sha256"] != raw_sha256:
        raise ValueError("recovered Fold 0 raw evidence manifest hash mismatch")


def _verification_record(
    run_directory: Path,
    spec: WeightSpec,
    verified: Mapping[str, object],
) -> dict[str, object]:
    _require_generic_pass(verified, spec.fold_id)
    independent_path = run_directory.resolve() / "independent-verification.json"
    return {
        "status": "pass",
        "fold_id": spec.fold_id,
        "run_directory": str(run_directory.resolve()),
        "independent_verification_sha256": sha256_file(independent_path),
    }


def _terminal_json(command: Sequence[str]) -> dict[str, object]:
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(
            f"official fold CLI exited {completed.returncode}: {detail}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("official fold CLI terminal JSON is missing")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError("official fold CLI terminal JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError("official fold CLI terminal JSON must be an object")
    return dict(payload)


def prepare_weight_cli(spec: WeightSpec) -> dict[str, object]:
    result = _terminal_json(
        [
            str(FIXED_ENVIRONMENT_PYTHON.resolve()),
            str(EVALUATOR_CLI),
            "--fold-id",
            str(spec.fold_id),
            "--fetch-only",
        ]
    )
    if result.get("fold_id") != spec.fold_id:
        raise ValueError(f"fold {spec.fold_id} weight preparation mismatch")
    return result


def launch_one_fold_cli(spec: WeightSpec) -> Path:
    result = _terminal_json(
        [
            str(FIXED_ENVIRONMENT_PYTHON.resolve()),
            str(EVALUATOR_CLI),
            "--fold-id",
            str(spec.fold_id),
            "--launch",
        ]
    )
    lock_path = (
        WEIGHT_ARTIFACTS_ROOT / f"fold{spec.fold_id}-weight-evaluation.lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError(f"fold {spec.fold_id} controller lock is invalid")
    run = Path(str(lock.get("run_directory", ""))).resolve()
    if (
        result.get("status") != "pass"
        or result.get("fold_id") != spec.fold_id
        or result.get("run_directory") != str(run)
        or run.parent != WEIGHT_ARTIFACTS_ROOT.resolve()
        or not run.name.startswith(f"fold{spec.fold_id}-weight-")
    ):
        raise ValueError(f"fold {spec.fold_id} launch CLI result mismatch")
    return run


def verify_one_fold_cli(run_directory: Path, spec: WeightSpec) -> dict[str, object]:
    run = run_directory.resolve()
    output = run / "independent-verification.json"
    result = _terminal_json(
        [
            str(FIXED_ENVIRONMENT_PYTHON.resolve()),
            str(VERIFIER_CLI),
            "--fold-id",
            str(spec.fold_id),
            "--run-directory",
            str(run),
            "--weight-path",
            str(weight_cache_path(spec).resolve()),
            "--output",
            str(output),
        ]
    )
    _require_generic_pass(result, spec.fold_id)
    recorded = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(recorded, Mapping) or dict(recorded) != result:
        raise ValueError(f"fold {spec.fold_id} verifier CLI output mismatch")
    return _verification_record(run, spec, result)


def qualify_existing_fold2(
    run_directory: Path, spec: WeightSpec
) -> dict[str, object]:
    """Re-verify Fold 2 while preserving raw evidence and verifier bytes exactly."""
    run = run_directory.resolve()
    if spec.fold_id != 2 or run != ADOPTED_FOLD2_RUN.resolve():
        raise ValueError("campaign may adopt only the fixed sealed Fold 2 run")
    independent_path = run / "independent-verification.json"
    before_bytes = independent_path.read_bytes()
    before_sha256 = hashlib.sha256(before_bytes).hexdigest()
    before = json.loads(before_bytes)
    if not isinstance(before, Mapping):
        raise ValueError("sealed Fold 2 verifier artifact must be an object")
    record = verify_one_fold_cli(run, spec)
    after_bytes = independent_path.read_bytes()
    after_sha256 = hashlib.sha256(after_bytes).hexdigest()
    after = json.loads(after_bytes)
    if (
        before_bytes != after_bytes
        or before_sha256 != after_sha256
        or before != after
        or before.get("raw_evidence_manifest") != after.get("raw_evidence_manifest")
        or after.get("raw_evidence_unchanged_during_verification") is not True
    ):
        raise ValueError(
            "sealed Fold 2 verifier bytes or raw evidence seal changed during qualification"
        )
    return record


def verify_one_fold(run_directory: Path, spec: WeightSpec) -> dict[str, object]:
    """Run the generic independent verifier once and publish its fold-local result."""
    return verify_one_fold_cli(run_directory, spec)


def finalize_existing_fold(
    run_directory: Path, spec: WeightSpec, recovery_authorization: Path
) -> dict[str, object]:
    """Finalize preserved output and verify it, without calling the launch path."""
    finalized = _terminal_json(
        [
            str(FIXED_ENVIRONMENT_PYTHON.resolve()),
            str(EVALUATOR_CLI),
            "--fold-id",
            str(spec.fold_id),
            "--finalize-existing",
            str(run_directory.resolve()),
            "--recovery-authorization",
            str(recovery_authorization.resolve()),
        ]
    )
    if finalized.get("status") != "pass" or finalized.get("fold_id") != spec.fold_id:
        raise ValueError(f"fold {spec.fold_id} finalization did not pass")
    return verify_one_fold_cli(run_directory, spec)


def _validate_campaign_directory(campaign_directory: Path) -> Path:
    campaign = campaign_directory.absolute()
    root = CAMPAIGN_ARTIFACTS_ROOT.absolute()
    if campaign.parent != root or not campaign.name.startswith(
        "official-weight-12fold-"
    ):
        raise ValueError("campaign directory is outside the fixed artifact root")
    return campaign


def _schedule_payload(schedule: Sequence[CampaignFold]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "folds": [
            {
                "fold_id": item.spec.fold_id,
                "mode": item.mode,
                "run_directory": (
                    str(item.run_directory.resolve())
                    if item.run_directory is not None
                    else None
                ),
                "weight_filename": item.spec.filename,
                "weight_sha256": item.spec.sha256,
                "train_years": list(item.spec.train_years),
                "validation_year": item.spec.validation_year,
                "test_year": item.spec.test_year,
            }
            for item in schedule
        ],
    }


def run_campaign(campaign_directory: Path) -> dict[str, object]:
    """Execute exactly one sequential 12-fold campaign and stop on first failure."""
    campaign = _validate_campaign_directory(campaign_directory)
    global_lock = CAMPAIGN_ARTIFACTS_ROOT.resolve() / GLOBAL_LOCK_NAME
    if campaign.exists():
        raise FileExistsError(f"campaign lock already exists: {campaign}")
    campaign.mkdir(parents=True, exist_ok=False)
    campaign_id = campaign.name
    completed = 0
    fold_id: int | None = None
    stage = "load_manifest"
    root_lock_owned = False
    try:
        specs = load_pinned_manifest(PINNED_MANIFEST_PATH)
        schedule = build_campaign_schedule(specs, {2: ADOPTED_FOLD2_RUN})
        stage = "write_schedule"
        _acquire_immutable_json(
            campaign / "schedule.json", _schedule_payload(schedule)
        )
        lock_payload = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_directory": str(campaign),
            "schedule_sha256": sha256_file(campaign / "schedule.json"),
            "single_campaign_no_retry": True,
            "sequential": True,
        }
        stage = "acquire_root_lock"
        _acquire_immutable_json(global_lock, lock_payload)
        root_lock_owned = True

        stage = "qualify_fold2"
        adopted_record = qualify_existing_fold2(
            ADOPTED_FOLD2_RUN, spec_for_fold(specs, 2)
        )
        for item in schedule:
            if item.mode == "launch":
                fold_id = item.spec.fold_id
                stage = "prepare_weight"
                prepare_weight_cli(item.spec)

        for item in schedule:
            fold_id = item.spec.fold_id
            stage = "acquire_fold_lock"
            _acquire_immutable_json(
                campaign / f"fold{fold_id}.lock.json",
                {**lock_payload, "fold_id": fold_id, "mode": item.mode},
            )
            if item.mode == "adopt":
                stage = "adopt_fold"
                assert item.run_directory is not None
                record = adopted_record
                _require_campaign_record(record, fold_id)
            else:
                stage = "launch_fold"
                run = launch_one_fold_cli(item.spec)
                try:
                    stage = "verify_fold"
                    record = verify_one_fold_cli(run, item.spec)
                    _require_campaign_record(record, fold_id)
                except BaseException as error:
                    raise ValueError(
                        f"fold {fold_id} independent verification failed"
                    ) from error
            state = {"mode": item.mode, **record}
            stage = "write_fold_state"
            _write_json_atomic(campaign / f"fold{fold_id}.json", state)
            completed += 1
        result = {"status": "pass", "fold_count": 12, "next_fold": None}
        stage = "write_completion"
        _write_json_atomic(campaign / "completed.json", result)
        return result
    except BaseException as error:
        _write_json_atomic(
            campaign / "failure.json",
            {
                "status": "fail",
                "campaign_id": campaign_id,
                "stage": stage,
                "fold_id": fold_id,
                "error": f"{type(error).__name__}: {error}",
                "completed_fold_count": completed,
                "root_lock_path": str(global_lock.resolve()),
                "root_lock_owned_by_campaign": root_lock_owned,
                "immutable_locks_preserved": True,
                "automatic_retry_performed": False,
                "partial_aggregate_written": False,
            },
        )
        raise


def _read_json_object(path: Path, message: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(message)
    return dict(payload)


def _recovery_code_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.splitlines():
        raise ValueError("campaign resume recovery code worktree is dirty")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = revision.stdout.strip()
    if revision.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("campaign resume recovery code commit is invalid")
    return commit


def _assert_original_failure_seal(
    failure_path: Path, expected: Mapping[str, object]
) -> None:
    path = require_sealed_regular_file(
        failure_path,
        checked_root=failure_path.parent,
        description="campaign original failure",
    )
    if (
        expected != _file_seal(path)
        or expected.get("bytes") != path.stat().st_size
        or expected.get("sha256") != sha256_file(path)
    ):
        raise ValueError("campaign original failure seal changed")


def _validate_resume_state(
    campaign_directory: Path,
) -> tuple[
    Path,
    tuple[CampaignFold, ...],
    dict[str, object],
    Path,
    dict[str, object],
]:
    campaign = _validate_campaign_directory(campaign_directory)
    campaign = require_sealed_directory(
        campaign,
        checked_root=CAMPAIGN_ARTIFACTS_ROOT,
        description="campaign resume directory",
    )
    try:
        require_absent_recovery_path(
            campaign / "resume.lock.json",
            checked_root=campaign,
            description="campaign resume lock",
        )
    except ValueError as error:
        raise FileExistsError("campaign resume lock already exists") from error
    require_absent_recovery_path(
        campaign / "completed.json",
        checked_root=campaign,
        description="campaign completed marker",
    )
    if campaign.name != RECOVERY_CAMPAIGN_ID:
        raise ValueError("campaign resume is authorized only for the exact incident")
    specs = load_pinned_manifest(PINNED_MANIFEST_PATH)
    schedule = build_campaign_schedule(specs, {2: ADOPTED_FOLD2_RUN})
    expected_schedule = _schedule_payload(schedule)
    schedule_path = campaign / "schedule.json"
    require_sealed_regular_file(
        schedule_path, checked_root=campaign, description="campaign schedule"
    )
    if _read_json_object(schedule_path, "campaign resume schedule is invalid") != expected_schedule:
        raise ValueError("campaign resume schedule differs from the exact campaign")
    root_lock_path = CAMPAIGN_ARTIFACTS_ROOT.absolute() / GLOBAL_LOCK_NAME
    require_sealed_regular_file(
        root_lock_path,
        checked_root=CAMPAIGN_ARTIFACTS_ROOT,
        description="campaign root lock",
    )
    root_lock = _read_json_object(
        root_lock_path, "campaign resume root lock is invalid"
    )
    expected_root_lock = {
        "schema_version": 1,
        "campaign_id": campaign.name,
        "campaign_directory": str(campaign),
        "schedule_sha256": sha256_file(schedule_path),
        "single_campaign_no_retry": True,
        "sequential": True,
    }
    if root_lock != expected_root_lock:
        raise ValueError("campaign resume root lock mismatch")
    failure_path = campaign / "failure.json"
    require_sealed_regular_file(
        failure_path, checked_root=campaign, description="campaign original failure"
    )
    failure_stat = failure_path.stat()
    failure_sha256 = sha256_file(failure_path)
    if (
        failure_stat.st_size != RECOVERY_FAILURE_BYTES
        or failure_sha256 != RECOVERY_FAILURE_SHA256
    ):
        raise ValueError("campaign resume original failure bytes mismatch")
    failure = _read_json_object(failure_path, "campaign resume failure marker is invalid")
    expected_failure_fields = {
        "status": "fail",
        "campaign_id": campaign.name,
        "stage": "verify_fold",
        "fold_id": 0,
        "error": "ValueError: fold 0 independent verification failed",
        "completed_fold_count": 0,
        "root_lock_path": str(root_lock_path.resolve()),
        "root_lock_owned_by_campaign": True,
        "immutable_locks_preserved": True,
        "automatic_retry_performed": False,
        "partial_aggregate_written": False,
    }
    if failure != expected_failure_fields:
        raise ValueError("campaign resume is authorized only for the exact Fold 0 verifier failure")
    fold0_lock_path = campaign / "fold0.lock.json"
    require_sealed_regular_file(
        fold0_lock_path, checked_root=campaign, description="campaign Fold 0 lock"
    )
    fold0_lock = _read_json_object(
        fold0_lock_path, "campaign resume Fold 0 lock is invalid"
    )
    if fold0_lock != {**expected_root_lock, "fold_id": 0, "mode": "launch"}:
        raise ValueError("campaign resume Fold 0 lock mismatch")
    require_absent_recovery_path(
        campaign / "fold0.json",
        checked_root=campaign,
        description="campaign Fold 0 state",
    )
    for fold_id in range(1, 12):
        for future in (
            campaign / f"fold{fold_id}.lock.json",
            campaign / f"fold{fold_id}.json",
        ):
            require_absent_recovery_path(
                future,
                checked_root=campaign,
                description="campaign future fold state",
            )
    evaluator_lock_path = WEIGHT_ARTIFACTS_ROOT / "fold0-weight-evaluation.lock.json"
    require_sealed_regular_file(
        evaluator_lock_path,
        checked_root=WEIGHT_ARTIFACTS_ROOT,
        description="Fold 0 evaluator lock",
    )
    evaluator_lock = _read_json_object(
        evaluator_lock_path,
        "campaign resume Fold 0 evaluator lock is invalid",
    )
    fold0_run = Path(str(evaluator_lock.get("run_directory", ""))).absolute()
    if (
        fold0_run.parent != WEIGHT_ARTIFACTS_ROOT.resolve()
        or not fold0_run.name.startswith("fold0-weight-")
        or fold0_run.name != RECOVERY_FOLD0_RUN_NAME
    ):
        raise ValueError("campaign resume Fold 0 run identity mismatch")
    fold0_run = require_sealed_directory(
        fold0_run,
        checked_root=WEIGHT_ARTIFACTS_ROOT,
        description="Fold 0 run directory",
    )
    source = RECOVERY_OUTPUT_SOURCE.absolute()
    source = require_sealed_regular_file(
        source,
        checked_root=RECOVERY_OUTPUT_SOURCE.parent,
        description="Fold 0 recovery source output",
    )
    source_stat = source.stat()
    if (
        source_stat.st_size != RECOVERY_OUTPUT_BYTES
        or sha256_file(source) != RECOVERY_OUTPUT_SHA256
        or source_stat.st_ctime_ns != RECOVERY_OUTPUT_CTIME_NS
        or source_stat.st_mtime_ns != RECOVERY_OUTPUT_MTIME_NS
    ):
        raise ValueError("campaign resume Fold 0 source output identity mismatch")
    target = fold0_run / RECOVERY_OUTPUT_TARGET_NAME
    require_absent_recovery_path(
        target, checked_root=fold0_run, description="Fold 0 recovery target output"
    )
    for marker_name in (RECOVERY_AUTHORIZATION_NAME, RECOVERY_PROVENANCE_NAME):
        require_absent_recovery_path(
            fold0_run / marker_name,
            checked_root=fold0_run,
            description="Fold 0 recovery evidence",
        )
    require_absent_recovery_path(
        campaign / EXTERNAL_RECOVERY_AUTHORIZATION_NAME,
        checked_root=campaign,
        description="external recovery authorization",
    )
    require_pairwise_distinct_files(
        [
            schedule_path,
            root_lock_path,
            failure_path,
            fold0_lock_path,
            evaluator_lock_path,
            source,
            Path(__file__),
            EVALUATOR_CLI,
            VERIFIER_CLI,
            PATH_SECURITY_MODULE,
        ],
        description="campaign recovery seals",
    )
    recovery_code_commit = _recovery_code_commit()
    incident = {
        "recovery_code_commit": recovery_code_commit,
        "original_failure": {
            "path": str(failure_path.resolve()),
            "bytes": failure_stat.st_size,
            "sha256": failure_sha256,
        },
        "source_output": {
            "path": str(source),
            "bytes": source_stat.st_size,
            "sha256": RECOVERY_OUTPUT_SHA256,
            "ctime_ns": source_stat.st_ctime_ns,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "target_output": str(target.resolve()),
        "target_output_absent_before_resume": True,
    }
    return campaign, schedule, expected_root_lock, fold0_run, incident


def resume_existing_campaign(campaign_directory: Path) -> dict[str, object]:
    """Continue the one stopped campaign after offline Fold 0 finalization."""
    campaign, schedule, root_lock, fold0_run, incident = _validate_resume_state(
        campaign_directory
    )
    original_failure_path = campaign / "failure.json"
    original_failure_sha256 = sha256_file(original_failure_path)
    authorization_path = campaign / EXTERNAL_RECOVERY_AUTHORIZATION_NAME
    authorization = {
        "schema_version": 1,
        "status": "authorized",
        "authorized_action": "one-time Fold 0 output recovery without scientific relaunch",
        "campaign_id": campaign.name,
        "campaign_directory": str(campaign),
        "fold_id": 0,
        "run_directory": str(fold0_run),
        "schedule": _file_seal(campaign / "schedule.json"),
        "root_lock": _file_seal(CAMPAIGN_ARTIFACTS_ROOT / GLOBAL_LOCK_NAME),
        "fold0_lock": _file_seal(campaign / "fold0.lock.json"),
        "original_failure": incident["original_failure"],
        "source_output": incident["source_output"],
        "target_output": {
            "path": incident["target_output"],
            "absent": incident["target_output_absent_before_resume"],
        },
        "reviewed_code": {
            "commit": incident["recovery_code_commit"],
            "files": [
                _file_seal(Path(__file__)),
                _file_seal(EVALUATOR_CLI),
                _file_seal(VERIFIER_CLI),
                _file_seal(PATH_SECURITY_MODULE),
            ],
        },
    }
    _acquire_immutable_json(authorization_path, authorization)
    _acquire_immutable_json(
        campaign / "resume.lock.json",
        {
            **root_lock,
            "schema_version": 2,
            "resume_scope": "Fold 0 offline finalization then exact remaining schedule",
            "original_failure_sha256": original_failure_sha256,
            "fold0_run_directory": str(fold0_run),
            "fold0_scientific_child_relaunched": False,
            "automatic_retry_performed": False,
            **incident,
        },
    )
    completed = 0
    fold_id: int | None = 0
    stage = "finalize_existing_fold0"
    try:
        fold0_record = finalize_existing_fold(
            fold0_run, schedule[0].spec, authorization_path
        )
        _validate_recovered_fold0_result(
            fold0_run, schedule[0].spec, fold0_record
        )
        _require_campaign_record(fold0_record, 0)
        _write_json_atomic(
            campaign / "fold0.json",
            {
                "mode": "launch",
                **fold0_record,
                "recovered_via_finalize_existing": True,
                "scientific_child_relaunched": False,
            },
        )
        completed = 1
        stage = "qualify_fold2"
        adopted_record = qualify_existing_fold2(
            ADOPTED_FOLD2_RUN, schedule[2].spec
        )
        for item in schedule[1:]:
            if item.mode == "launch":
                fold_id = item.spec.fold_id
                stage = "prepare_weight"
                prepare_weight_cli(item.spec)
        for item in schedule[1:]:
            fold_id = item.spec.fold_id
            stage = "acquire_fold_lock"
            _acquire_immutable_json(
                campaign / f"fold{fold_id}.lock.json",
                {**root_lock, "fold_id": fold_id, "mode": item.mode},
            )
            if item.mode == "adopt":
                stage = "adopt_fold"
                record = adopted_record
                _require_campaign_record(record, fold_id)
            else:
                stage = "launch_fold"
                _assert_original_failure_seal(
                    original_failure_path, incident["original_failure"]
                )
                run = launch_one_fold_cli(item.spec)
                try:
                    stage = "verify_fold"
                    record = verify_one_fold_cli(run, item.spec)
                    _require_campaign_record(record, fold_id)
                except BaseException as error:
                    raise ValueError(
                        f"fold {fold_id} independent verification failed"
                    ) from error
            _write_json_atomic(
                campaign / f"fold{fold_id}.json", {"mode": item.mode, **record}
            )
            completed += 1
        _assert_original_failure_seal(
            original_failure_path, incident["original_failure"]
        )
        result = {"status": "pass", "fold_count": 12, "next_fold": None}
        _write_json_atomic(campaign / "completed.json", result)
        _write_json_atomic(
            campaign / "resume-completed.json",
            {
                **result,
                "original_failure_sha256": original_failure_sha256,
                "fold0_scientific_child_relaunched": False,
                "same_campaign_directory": True,
            },
        )
        return result
    except BaseException as error:
        _write_json_atomic(
            campaign / "resume-failure.json",
            {
                "status": "fail",
                "campaign_id": campaign.name,
                "stage": stage,
                "fold_id": fold_id,
                "error": f"{type(error).__name__}: {error}",
                "completed_fold_count": completed,
                "original_failure_sha256": original_failure_sha256,
                "original_failure_preserved": (
                    sha256_file(original_failure_path) == original_failure_sha256
                ),
                "immutable_locks_preserved": True,
                "automatic_retry_performed": False,
                "fold0_scientific_child_relaunched": False,
                "partial_aggregate_written": False,
            },
        )
        raise


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, choices=range(12))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--launch", action="store_true")
    action.add_argument("--finalize-existing", type=Path)
    action.add_argument("--resume-existing", type=Path)
    parser.add_argument("--campaign-directory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.launch:
        if arguments.campaign_directory is None or arguments.fold_id is not None:
            raise ValueError("campaign launch requires --campaign-directory only")
        result = run_campaign(arguments.campaign_directory)
    elif arguments.finalize_existing is not None:
        if arguments.fold_id is None or arguments.campaign_directory is not None:
            raise ValueError("finalize-existing requires exactly one --fold-id")
        specs = load_pinned_manifest(PINNED_MANIFEST_PATH)
        result = finalize_existing_fold(
            arguments.finalize_existing, spec_for_fold(specs, arguments.fold_id)
        )
    else:
        if arguments.fold_id is not None or arguments.campaign_directory is not None:
            raise ValueError("resume-existing requires exactly one campaign path")
        result = resume_existing_campaign(arguments.resume_existing)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

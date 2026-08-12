"""Run the pinned twelve-fold official-weight evaluation exactly once, sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
EVALUATOR_CLI = Path(__file__).with_name("evaluate_released_weight.py").resolve()
VERIFIER_CLI = Path(__file__).with_name("verify_released_weight.py").resolve()


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
    run_directory: Path, spec: WeightSpec
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
        ]
    )
    if finalized.get("status") != "pass" or finalized.get("fold_id") != spec.fold_id:
        raise ValueError(f"fold {spec.fold_id} finalization did not pass")
    return verify_one_fold_cli(run_directory, spec)


def _validate_campaign_directory(campaign_directory: Path) -> Path:
    campaign = campaign_directory.resolve()
    root = CAMPAIGN_ARTIFACTS_ROOT.resolve()
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


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, choices=range(12))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--launch", action="store_true")
    action.add_argument("--finalize-existing", type=Path)
    parser.add_argument("--campaign-directory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.launch:
        if arguments.campaign_directory is None or arguments.fold_id is not None:
            raise ValueError("campaign launch requires --campaign-directory only")
        result = run_campaign(arguments.campaign_directory)
    else:
        if arguments.fold_id is None or arguments.campaign_directory is not None:
            raise ValueError("finalize-existing requires exactly one --fold-id")
        specs = load_pinned_manifest(PINNED_MANIFEST_PATH)
        result = finalize_existing_fold(
            arguments.finalize_existing, spec_for_fold(specs, arguments.fold_id)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

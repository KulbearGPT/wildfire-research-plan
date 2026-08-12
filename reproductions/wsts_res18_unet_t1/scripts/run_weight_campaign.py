"""Run the pinned twelve-fold official-weight evaluation exactly once, sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluate_released_weight import (
    _launch_once as launch_one_fold_once,
    fetch_weight as prepare_weight,
    finalize_existing_weight,
    weight_cache_path,
)
from released_weight_contract import (
    WeightSpec,
    load_pinned_manifest,
    spec_for_fold,
)
from verify_released_weight import verify_run as verify_fold_run


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


def qualify_existing_fold2(
    run_directory: Path, spec: WeightSpec
) -> dict[str, object]:
    """Re-verify and adopt the one sealed Fold 2 run without modifying it."""
    run = run_directory.resolve()
    if spec.fold_id != 2 or run != ADOPTED_FOLD2_RUN.resolve():
        raise ValueError("campaign may adopt only the fixed sealed Fold 2 run")
    verified = verify_fold_run(run, weight_cache_path(spec), spec)
    _require_generic_pass(verified, 2)
    independent_path = run / "independent-verification.json"
    recorded = json.loads(independent_path.read_text(encoding="utf-8"))
    if not isinstance(recorded, Mapping) or dict(recorded) != dict(verified):
        raise ValueError("sealed Fold 2 independent verification differs from generic verifier")
    return _verification_record(run, spec, verified)


def verify_one_fold(run_directory: Path, spec: WeightSpec) -> dict[str, object]:
    """Run the generic independent verifier once and publish its fold-local result."""
    run = run_directory.resolve()
    verified = verify_fold_run(run, weight_cache_path(spec), spec)
    _require_generic_pass(verified, spec.fold_id)
    _write_json_atomic(run / "independent-verification.json", verified)
    return _verification_record(run, spec, verified)


def finalize_existing_fold(
    run_directory: Path, spec: WeightSpec
) -> dict[str, object]:
    """Finalize preserved output and verify it, without calling the launch path."""
    finalized = finalize_existing_weight(run_directory, spec)
    if finalized.get("status") != "pass":
        raise ValueError(f"fold {spec.fold_id} finalization did not pass")
    return verify_one_fold(run_directory, spec)


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
    campaign_lock = campaign / "campaign.lock.json"
    prospective_fold_locks = [
        campaign / f"fold{fold_id}.lock.json" for fold_id in range(12)
    ]
    existing = [
        path
        for path in (global_lock, campaign_lock, *prospective_fold_locks)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"campaign lock already exists: {existing[0].resolve()}")

    specs = load_pinned_manifest(PINNED_MANIFEST_PATH)
    schedule = build_campaign_schedule(specs, {2: ADOPTED_FOLD2_RUN})

    # All non-scientific prerequisites complete before any campaign launch lock.
    adopted_record = qualify_existing_fold2(
        ADOPTED_FOLD2_RUN, spec_for_fold(specs, 2)
    )
    for item in schedule:
        if item.mode == "launch":
            prepared = prepare_weight(item.spec)
            if prepared.get("fold_id", item.spec.fold_id) != item.spec.fold_id:
                raise ValueError(f"fold {item.spec.fold_id} weight preparation mismatch")

    campaign.mkdir(parents=True, exist_ok=True)
    _acquire_immutable_json(campaign / "schedule.json", _schedule_payload(schedule))
    lock_payload = {
        "schema_version": 1,
        "campaign_directory": str(campaign),
        "schedule_sha256": sha256_file(campaign / "schedule.json"),
        "single_campaign_no_retry": True,
        "sequential": True,
    }
    _acquire_immutable_json(global_lock, lock_payload)
    _acquire_immutable_json(campaign_lock, lock_payload)

    completed = 0
    for item in schedule:
        fold_id = item.spec.fold_id
        try:
            _acquire_immutable_json(
                campaign / f"fold{fold_id}.lock.json",
                {**lock_payload, "fold_id": fold_id, "mode": item.mode},
            )
            if item.mode == "adopt":
                assert item.run_directory is not None
                record = adopted_record
                _require_campaign_record(record, fold_id)
            else:
                run, child = launch_one_fold_once(item.spec)
                if child.get("status") != "pass":
                    raise ValueError(f"fold {fold_id} scientific child did not pass")
                try:
                    record = verify_one_fold(run, item.spec)
                    _require_campaign_record(record, fold_id)
                except BaseException as error:
                    raise ValueError(
                        f"fold {fold_id} independent verification failed"
                    ) from error
            state = {"mode": item.mode, **record}
            _write_json_atomic(campaign / f"fold{fold_id}.json", state)
            completed += 1
        except BaseException as error:
            _write_json_atomic(
                campaign / "failure.json",
                {
                    "status": "fail",
                    "fold_id": fold_id,
                    "error": f"{type(error).__name__}: {error}",
                    "completed_fold_count": completed,
                    "automatic_retry_performed": False,
                    "partial_aggregate_written": False,
                },
            )
            raise
    result = {"status": "pass", "fold_count": 12, "next_fold": None}
    _write_json_atomic(campaign / "completed.json", result)
    return result


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

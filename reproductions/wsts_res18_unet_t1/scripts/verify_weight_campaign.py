"""Independently reconstruct and aggregate a sealed twelve-fold weight campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import statistics
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from released_weight_contract import WeightSpec, load_pinned_manifest, spec_for_fold
from released_weight_path_security import (
    require_absent_recovery_path,
    require_pairwise_distinct_files,
    require_sealed_directory,
    require_sealed_regular_file,
    validate_reviewed_authorization_manifest,
)
from verify_released_weight import (
    EXPECTED_TEST_METRICS,
    capture_weight_raw_manifest as capture_fold_raw_manifest,
    verify_run as verify_fold_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST_PATH = REPRODUCTION_ROOT / "official_weights_manifest.json"
UPSTREAM_LOCK_PATH = REPRODUCTION_ROOT / "upstream.lock.json"
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
RESULTS_FILENAME = "official-weight-12fold-results.csv"
SUMMARY_FILENAME = "official-weight-12fold-summary.json"
INDEPENDENT_FILENAME = "independent-verification.json"
PUBLICATION_FILENAME = "publication.json"
GLOBAL_LOCK_NAME = "official-weight-12fold-campaign.lock.json"
EXTERNAL_RECOVERY_AUTHORIZATION_NAME = "fold0-recovery-authorization.json"
RECOVERY_REVIEWED_AUTHORIZATION_NAME = "fold0-recovery-reviewed-authorization.json"
RECOVERY_RECEIPT_NAME = "official-test-output-recovery-authorization.json"
RECOVERY_PROVENANCE_NAME = "official-test-output-provenance.json"
RECOVERY_TARGET_NAME = "official-test-pr-curve-data.npz"
CAMPAIGN_CONTROLLER_PATH = Path(__file__).with_name("run_weight_campaign.py").absolute()
EVALUATOR_PATH = Path(__file__).with_name("evaluate_released_weight.py").absolute()
FOLD_VERIFIER_PATH = Path(__file__).with_name("verify_released_weight.py").absolute()
PATH_SECURITY_MODULE = Path(__file__).with_name("released_weight_path_security.py").absolute()
RECOVERY_REVIEWED_FILES = (
    CAMPAIGN_CONTROLLER_PATH,
    EVALUATOR_PATH,
    FOLD_VERIFIER_PATH,
    PATH_SECURITY_MODULE,
)
ADOPTED_FOLD2_RUN = (
    WEIGHT_ARTIFACTS_ROOT
    / "fold2-weight-20260810T133336Z-2e071197"
).resolve()
CSV_FIELDS = (
    "fold_id",
    "train_years",
    "validation_year",
    "test_year",
    "weight_filename",
    "weight_sha256",
    "filename_ap",
    "test_AP",
    "test_f1",
    "test_iou",
    "test_loss",
    "test_precision",
    "test_recall",
    "test_AP_minus_filename_ap",
    "test_AP_absolute_difference_from_filename_ap",
    "wall_seconds",
    "run_directory",
    "fold_verifier_sha256",
    "raw_evidence_manifest_sha256",
)
_METRIC_NAME = r"test_(?:AP|f1|iou|loss|precision|recall)"
_FINITE_SCALAR = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exact_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _file_seal(path: Path, *, checked_root: Path, description: str) -> dict[str, object]:
    sealed = require_sealed_regular_file(
        path, checked_root=checked_root, description=description
    )
    return {
        "path": str(sealed),
        "bytes": sealed.stat().st_size,
        "sha256": _sha256(sealed),
    }


def _exact_file_seal(
    path: Path, *, checked_root: Path, description: str
) -> dict[str, object]:
    sealed = require_sealed_regular_file(
        path, checked_root=checked_root, description=description
    )
    raw = sealed.read_bytes()
    return {
        "path": str(sealed),
        "bytes_hex": raw.hex(),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _read_json_object(path: Path, *, checked_root: Path, description: str) -> dict[str, object]:
    sealed = require_sealed_regular_file(
        path, checked_root=checked_root, description=description
    )
    payload = json.loads(sealed.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"{description} must be an exact JSON object")
    return payload


def _verify_recovery_chain(campaign: Path, fold0_state: Mapping[str, object]) -> None:
    """Bind the final campaign result to the one reviewed Fold 0 recovery incident."""
    if (
        fold0_state.get("recovered_via_finalize_existing") is not True
        or fold0_state.get("scientific_child_relaunched") is not False
        or fold0_state.get("fold_id") != 0
        or fold0_state.get("mode") != "launch"
    ):
        raise ValueError("campaign Fold 0 recovery chain state mismatch")
    run = Path(str(fold0_state.get("run_directory", ""))).absolute()
    run = require_sealed_directory(
        run, checked_root=WEIGHT_ARTIFACTS_ROOT, description="recovery chain Fold 0 run"
    )
    paths = {
        "schedule": campaign / "schedule.json",
        "root_lock": CAMPAIGN_ARTIFACTS_ROOT / GLOBAL_LOCK_NAME,
        "fold0_lock": campaign / "fold0.lock.json",
        "original_failure": campaign / "failure.json",
        "resume_lock": campaign / "resume.lock.json",
        "authorization": campaign / EXTERNAL_RECOVERY_AUTHORIZATION_NAME,
        "reviewed": campaign / RECOVERY_REVIEWED_AUTHORIZATION_NAME,
        "completed": campaign / "completed.json",
        "resume_completed": campaign / "resume-completed.json",
        "receipt": run / RECOVERY_RECEIPT_NAME,
        "provenance": run / RECOVERY_PROVENANCE_NAME,
        "target": run / RECOVERY_TARGET_NAME,
    }
    checked_roots = {
        "root_lock": CAMPAIGN_ARTIFACTS_ROOT,
        "receipt": run,
        "provenance": run,
        "target": run,
    }
    for name, path in paths.items():
        require_sealed_regular_file(
            path,
            checked_root=checked_roots.get(name, campaign),
            description=f"recovery chain {name}",
        )
    require_pairwise_distinct_files(paths.values(), description="recovery chain")

    schedule_seal = _file_seal(
        paths["schedule"], checked_root=campaign, description="recovery chain schedule"
    )
    root_seal = _file_seal(
        paths["root_lock"],
        checked_root=CAMPAIGN_ARTIFACTS_ROOT,
        description="recovery chain root lock",
    )
    fold0_lock_seal = _file_seal(
        paths["fold0_lock"], checked_root=campaign, description="recovery chain Fold 0 lock"
    )
    failure_seal = _file_seal(
        paths["original_failure"],
        checked_root=campaign,
        description="recovery chain original failure",
    )
    authorization = _read_json_object(
        paths["authorization"], checked_root=campaign, description="recovery chain authorization"
    )
    authorization_keys = {
        "schema_version", "status", "authorized_action", "campaign_id",
        "campaign_directory", "fold_id", "run_directory", "schedule", "root_lock",
        "fold0_lock", "original_failure", "source_output", "target_output",
        "resume_lock", "reviewed_authorization",
    }
    if set(authorization) != authorization_keys:
        raise ValueError("campaign Fold 0 recovery chain authorization schema mismatch")
    reviewed = authorization.get("reviewed_authorization")
    if (
        type(reviewed) is not dict
        or set(reviewed) != {"seal", "payload"}
        or reviewed.get("seal")
        != _exact_file_seal(
            paths["reviewed"],
            checked_root=campaign,
            description="recovery chain reviewed authorization",
        )
        or _read_json_object(
            paths["reviewed"],
            checked_root=campaign,
            description="recovery chain reviewed authorization",
        )
        != reviewed.get("payload")
    ):
        raise ValueError("campaign Fold 0 recovery chain reviewed authorization mismatch")
    reviewed_payload = reviewed["payload"]
    independently_reviewed = validate_reviewed_authorization_manifest(
        paths["reviewed"],
        expected_path=paths["reviewed"],
        repository_root=REPOSITORY_ROOT,
        campaign_directory=campaign,
        run_directory=run,
        reviewed_paths=RECOVERY_REVIEWED_FILES,
    )
    if independently_reviewed != reviewed_payload:
        raise ValueError("campaign Fold 0 recovery chain reviewed authorization mismatch")
    source = authorization.get("source_output")
    target = authorization.get("target_output")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("status") != "authorized"
        or authorization.get("authorized_action")
        != "one-time Fold 0 output recovery without scientific relaunch"
        or authorization.get("campaign_id") != campaign.name
        or authorization.get("campaign_directory") != str(campaign)
        or authorization.get("fold_id") != 0
        or authorization.get("run_directory") != str(run)
        or authorization.get("schedule") != schedule_seal
        or authorization.get("root_lock") != root_seal
        or authorization.get("fold0_lock") != fold0_lock_seal
        or authorization.get("original_failure") != failure_seal
        or type(source) is not dict
        or set(source) != {"path", "bytes", "sha256", "ctime_ns", "mtime_ns"}
        or any(type(source[key]) is not int for key in ("bytes", "ctime_ns", "mtime_ns"))
        or not isinstance(source.get("path"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256"))) is None
        or target != {"path": str(paths["target"]), "absent": True}
        or authorization.get("resume_lock")
        != _exact_file_seal(
            paths["resume_lock"],
            checked_root=campaign,
            description="recovery chain resume lock",
        )
    ):
        raise ValueError("campaign Fold 0 recovery chain authorization mismatch")

    resume = _read_json_object(
        paths["resume_lock"], checked_root=campaign, description="recovery chain resume lock"
    )
    root_lock = _read_json_object(
        paths["root_lock"],
        checked_root=CAMPAIGN_ARTIFACTS_ROOT,
        description="recovery chain root lock",
    )
    expected_resume = {
        **root_lock,
        "schema_version": 2,
        "resume_scope": "Fold 0 offline finalization then exact remaining schedule",
        "original_failure_sha256": failure_seal["sha256"],
        "fold0_run_directory": str(run),
        "fold0_scientific_child_relaunched": False,
        "automatic_retry_performed": False,
        "recovery_code_commit": reviewed_payload.get("reviewed_commit"),
        "original_failure": failure_seal,
        "source_output": source,
        "target_output": str(paths["target"]),
        "target_output_absent_before_resume": True,
        "reviewed_authorization": reviewed,
    }
    if not _exact_json_value(resume, expected_resume):
        raise ValueError("campaign Fold 0 recovery chain resume lock mismatch")

    receipt = _read_json_object(
        paths["receipt"], checked_root=run, description="recovery chain receipt"
    )
    provenance = _read_json_object(
        paths["provenance"], checked_root=run, description="recovery chain provenance"
    )
    source_path = Path(str(source["path"]))
    try:
        require_absent_recovery_path(
            source_path,
            checked_root=source_path.parent,
            description="recovery chain consumed source",
        )
    except ValueError as error:
        raise ValueError("campaign Fold 0 recovery chain source was not consumed") from error
    receipt_expected_keys = {
        "schema_version", "status", "authorized_action", "scientific_child_relaunched",
        "fold_id", "run_directory", "weight", "source", "target", "child_pid",
        "command_sha256", "source_sha256", "source_bytes", "source_ctime_ns",
        "source_mtime_ns", "started_utc", "exit_marker_mtime_ns",
        "raw_evidence_manifest", "raw_evidence_manifest_sha256",
        "launch_provenance_sha256", "launch_provenance_manifest_sha256",
        "external_authorization_path", "external_authorization_sha256",
        "external_authorization",
    }
    if (
        set(receipt) != receipt_expected_keys
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "pass"
        or receipt.get("fold_id") != 0
        or receipt.get("run_directory") != str(run)
        or receipt.get("scientific_child_relaunched") is not False
        or receipt.get("source") != source["path"]
        or receipt.get("target") != str(paths["target"])
        or receipt.get("source_sha256") != source["sha256"]
        or receipt.get("source_bytes") != source["bytes"]
        or receipt.get("source_ctime_ns") != source["ctime_ns"]
        or receipt.get("source_mtime_ns") != source["mtime_ns"]
        or receipt.get("external_authorization_path") != str(paths["authorization"])
        or receipt.get("external_authorization_sha256") != _sha256(paths["authorization"])
        or not _exact_json_value(receipt.get("external_authorization"), authorization)
        or provenance.get("recovery_authorization_sha256") != _sha256(paths["receipt"])
        or provenance.get("target") != str(paths["target"])
        or provenance.get("sha256") != source["sha256"]
        or provenance.get("bytes") != source["bytes"]
        or _sha256(paths["target"]) != source["sha256"]
        or paths["target"].stat().st_size != source["bytes"]
    ):
        raise ValueError("campaign Fold 0 recovery chain receipt or provenance mismatch")
    completed = _read_json_object(
        paths["completed"], checked_root=campaign, description="recovery chain completion"
    )
    resume_completed = _read_json_object(
        paths["resume_completed"],
        checked_root=campaign,
        description="recovery chain resume completion",
    )
    expected_completed = {"status": "pass", "fold_count": 12, "next_fold": None}
    expected_resume_completed = {
        **expected_completed,
        "original_failure_sha256": failure_seal["sha256"],
        "fold0_scientific_child_relaunched": False,
        "same_campaign_directory": True,
    }
    if not _exact_json_value(completed, expected_completed) or not _exact_json_value(
        resume_completed, expected_resume_completed
    ):
        raise ValueError("campaign Fold 0 recovery chain completion mismatch")


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


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_link_or_reparse(path: Path) -> bool:
    metadata = path.lstat()
    is_junction = getattr(path, "is_junction", lambda: False)
    return (
        path.is_symlink()
        or is_junction()
        or bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    )


def _require_lexically_exact_path(
    path: Path, *, checked_root: Path, description: str
) -> Path:
    lexical = path.absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{description} is missing or inaccessible") from error
    if resolved != lexical:
        raise ValueError(f"{description} is aliased")
    current = lexical
    while True:
        if _is_link_or_reparse(current):
            raise ValueError(f"{description} is aliased")
        if current == checked_root:
            return lexical
        if current.parent == current:
            raise ValueError(f"{description} escapes its root")
        current = current.parent


def load_committed_publication(campaign_directory: Path) -> dict[str, object]:
    """Load the sole commit point and verify every referenced generation file."""
    campaign = campaign_directory.resolve()
    publication_path = campaign / PUBLICATION_FILENAME
    payload = json.loads(publication_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("campaign publication marker must be an object")
    generation = payload.get("generation")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "pass"
        or payload.get("fold_count") != 12
        or not isinstance(generation, str)
        or re.fullmatch(r"generations/[0-9a-f]{32}", generation) is None
    ):
        raise ValueError("campaign publication marker identity mismatch")
    outputs = payload.get("outputs")
    expected_names = {RESULTS_FILENAME, SUMMARY_FILENAME, INDEPENDENT_FILENAME}
    if not isinstance(outputs, Mapping) or set(outputs) != expected_names:
        raise ValueError("campaign publication output manifest mismatch")
    generations_root = campaign / "generations"
    generation_directory = campaign / generation
    if generation_directory.parent != generations_root:
        raise ValueError("campaign publication generation path escapes its root")
    generation_directory = _require_lexically_exact_path(
        generation_directory,
        checked_root=generations_root,
        description="campaign publication generation path",
    )
    output_paths: dict[str, Path] = {}
    for name in expected_names:
        seal = outputs[name]
        expected_relative = f"{generation}/{name}"
        if not isinstance(seal, Mapping) or set(seal) != {"path", "sha256"}:
            raise ValueError(f"campaign publication seal is invalid: {name}")
        if seal.get("path") != expected_relative:
            raise ValueError(f"campaign publication file seal mismatch: {name}")
        path = campaign / expected_relative
        if path.parent != generation_directory:
            raise ValueError(f"campaign publication file seal mismatch: {name}")
        path = _require_lexically_exact_path(
            path,
            checked_root=path,
            description="campaign publication output",
        )
        if not path.is_file():
            raise ValueError(f"campaign publication file seal mismatch: {name}")
        output_paths[name] = path
    output_items = list(output_paths.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1 :]:
            if left_path.samefile(right_path):
                raise ValueError(
                    "campaign publication output is aliased: "
                    f"{left_name}, {right_name}"
                )
    for name, path in output_paths.items():
        if outputs[name]["sha256"] != _sha256(path):
            raise ValueError(f"campaign publication file seal mismatch: {name}")
    independent = json.loads(
        output_paths[INDEPENDENT_FILENAME].read_text(encoding="utf-8")
    )
    if (
        not isinstance(independent, Mapping)
        or independent.get("status") != "pass"
        or independent.get("fold_count") != 12
        or independent.get("generation") != generation
        or independent.get("results_sha256")
        != outputs[RESULTS_FILENAME]["sha256"]
        or independent.get("summary_sha256")
        != outputs[SUMMARY_FILENAME]["sha256"]
    ):
        raise ValueError("campaign committed independent result mismatch")
    return dict(payload)


def _paper_reference() -> tuple[float, float]:
    payload = json.loads(UPSTREAM_LOCK_PATH.read_text(encoding="utf-8"))
    target = payload.get("paper", {}).get("target") if isinstance(payload, Mapping) else None
    match = re.fullmatch(r"(0\.\d+) \+/- (0\.\d+)", str(target))
    if match is None:
        raise ValueError("paper AP reference is missing from the pinned upstream lock")
    return float(match.group(1)), float(match.group(2))


def _validate_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, Mapping) or set(metrics) != EXPECTED_TEST_METRICS:
        raise ValueError("campaign verifier requires six finite metrics in [0,1]")
    if any(isinstance(value, bool) for value in metrics.values()):
        raise ValueError("campaign verifier requires six finite metrics in [0,1]")
    try:
        values = {name: float(value) for name, value in metrics.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(
            "campaign verifier requires six finite metrics in [0,1]"
        ) from error
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values.values()
    ):
        raise ValueError("campaign verifier requires six finite metrics in [0,1]")
    return values


def summarize_verified_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) != 12 or {int(row["fold_id"]) for row in rows} != set(range(12)):
        raise ValueError("campaign rows must cover folds exactly 0 through 11")
    ordered = sorted(rows, key=lambda row: int(row["fold_id"]))
    for row in ordered:
        _validate_metrics({name: row.get(name) for name in EXPECTED_TEST_METRICS})
    metric_summaries: dict[str, dict[str, object]] = {}
    for name in sorted(EXPECTED_TEST_METRICS):
        values = [float(row[name]) for row in ordered]
        minimum = min(values)
        maximum = max(values)
        metric_summaries[name] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "min": minimum,
            "min_fold_id": int(ordered[values.index(minimum)]["fold_id"]),
            "max": maximum,
            "max_fold_id": int(ordered[values.index(maximum)]["fold_id"]),
        }
    ap = metric_summaries["test_AP"]
    filename_values = [float(row["filename_ap"]) for row in ordered]
    filename_mean = statistics.fmean(filename_values)
    filename_std = statistics.pstdev(filename_values)
    paper_mean, paper_std = _paper_reference()
    runtimes = [float(row["wall_seconds"]) for row in ordered]
    if any(not math.isfinite(value) or value <= 0 for value in runtimes):
        raise ValueError("campaign runtimes must be finite and positive")
    return {
        "schema_version": 1,
        "status": "pass",
        "fold_count": 12,
        "metrics": metric_summaries,
        "ap_mean": ap["mean"],
        "ap_population_std": ap["population_std"],
        "ap_min": ap["min"],
        "ap_min_fold_id": ap["min_fold_id"],
        "ap_max": ap["max"],
        "ap_max_fold_id": ap["max_fold_id"],
        "runtime_total_seconds": math.fsum(runtimes),
        "runtime_median_seconds": statistics.median(runtimes),
        "filename_ap_mean": filename_mean,
        "filename_ap_population_std": filename_std,
        "paper_ap_mean": paper_mean,
        "paper_ap_population_std": paper_std,
        "paper_ap_reported_std": paper_std,
        "filename_reference": {
            "source": "official_weights_manifest.json filename labels",
            "manifest_sha256": _sha256(PINNED_MANIFEST_PATH),
            "paper_table_provenance": False,
            "mean": filename_mean,
            "population_std": filename_std,
        },
        "paper_reference": {
            "source": "upstream.lock.json paper.target",
            "upstream_lock_sha256": _sha256(UPSTREAM_LOCK_PATH),
            "reported": f"{paper_mean:.3f} +/- {paper_std:.3f}",
            "mean": paper_mean,
            "reported_std": paper_std,
        },
        "ap_mean_minus_filename_mean": float(ap["mean"]) - filename_mean,
        "ap_mean_absolute_difference_from_filename_mean": abs(
            float(ap["mean"]) - filename_mean
        ),
        "ap_population_std_minus_filename_population_std": (
            float(ap["population_std"]) - filename_std
        ),
        "ap_population_std_absolute_difference_from_filename_population_std": abs(
            float(ap["population_std"]) - filename_std
        ),
        "ap_mean_minus_paper_mean": float(ap["mean"]) - paper_mean,
        "ap_mean_absolute_difference_from_paper_mean": abs(
            float(ap["mean"]) - paper_mean
        ),
        "ap_population_std_minus_paper_population_std": (
            float(ap["population_std"]) - paper_std
        ),
        "ap_population_std_minus_paper_reported_std": (
            float(ap["population_std"]) - paper_std
        ),
        "ap_population_std_absolute_difference_from_paper_reported_std": abs(
            float(ap["population_std"]) - paper_std
        ),
    }


def _state_rows(campaign: Path) -> list[Mapping[str, object]]:
    paths = sorted(
        campaign.glob("fold*.json"),
        key=lambda path: path.name,
    )
    states: list[Mapping[str, object]] = []
    for path in paths:
        if re.fullmatch(r"fold\d+\.json", path.name) is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("campaign fold state must be a JSON object")
        states.append(payload)
    fold_ids = [state.get("fold_id") for state in states]
    if (
        len(states) != 12
        or any(type(fold_id) is not int for fold_id in fold_ids)
        or sorted(fold_ids) != list(range(12))
    ):
        raise ValueError("campaign fold states must cover folds exactly 0 through 11")
    return sorted(states, key=lambda state: int(state["fold_id"]))


def _validate_run_path(run_directory: Path, spec: WeightSpec) -> Path:
    run = run_directory.resolve()
    root = WEIGHT_ARTIFACTS_ROOT.resolve()
    if run.parent != root or not run.name.startswith(f"fold{spec.fold_id}-weight-"):
        raise ValueError("campaign fold run is outside the fixed raw artifact root")
    return run


def weight_path_for_spec(spec: WeightSpec) -> Path:
    return (REPRODUCTION_ROOT / ".local" / "released-weights" / spec.filename).resolve()


def _raw_metrics(run: Path) -> dict[str, float]:
    output = (run / "stdout.log").read_text(encoding="utf-8", errors="replace")
    output += "\n" + (run / "stderr.log").read_text(
        encoding="utf-8", errors="replace"
    )
    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    if exit_code != 0:
        raise ValueError("campaign fold child exit code is not zero")
    metrics: dict[str, float] = {}
    patterns = (
        rf"(?m)[│|]\s*({_METRIC_NAME})\s*[│|]\s*({_FINITE_SCALAR})",
        rf"(?m)(?:^|[\r\n])[ \t]*({_METRIC_NAME})[ \t]{{2,}}({_FINITE_SCALAR})"
        r"(?=[ \t]*(?:[\r\n]|$))",
    )
    for pattern in patterns:
        for name, raw in re.findall(pattern, output):
            if name in metrics:
                raise ValueError("campaign raw table contains duplicate test metrics")
            metrics[name] = float(raw)
    return _validate_metrics(metrics)


def verify_campaign(campaign_directory: Path) -> dict[str, object]:
    """Rebuild every result from raw fold evidence, then atomically publish aggregates."""
    campaign = campaign_directory.absolute()
    if campaign.parent != CAMPAIGN_ARTIFACTS_ROOT.absolute() or not campaign.name.startswith(
        "official-weight-12fold-"
    ):
        raise ValueError("campaign directory is outside the fixed artifact root")
    campaign = require_sealed_directory(
        campaign,
        checked_root=CAMPAIGN_ARTIFACTS_ROOT,
        description="campaign verifier directory",
    )
    specs = load_pinned_manifest(PINNED_MANIFEST_PATH)
    states = _state_rows(campaign)
    fold0_state = states[0]
    if fold0_state.get("recovered_via_finalize_existing") is True:
        _verify_recovery_chain(campaign, fold0_state)
    elif "recovered_via_finalize_existing" in fold0_state:
        raise ValueError("campaign Fold 0 recovery chain state mismatch")
    sealed_inputs: dict[int, dict[str, object]] = {}
    for state in states:
        fold_id = int(state["fold_id"])
        spec = spec_for_fold(specs, fold_id)
        if state.get("status") != "pass":
            raise ValueError(f"campaign fold {fold_id} state did not pass")
        expected_mode = "adopt" if fold_id == 2 else "launch"
        if state.get("mode") != expected_mode:
            raise ValueError(f"campaign fold {fold_id} mode mismatch")
        run = _validate_run_path(Path(str(state.get("run_directory", ""))), spec)
        if fold_id == 2 and run != ADOPTED_FOLD2_RUN.resolve():
            raise ValueError("campaign did not adopt the fixed sealed Fold 2 run")
        independent_path = run / "independent-verification.json"
        verifier_sha = _sha256(independent_path)
        if state.get("independent_verification_sha256") != verifier_sha:
            raise ValueError(f"fold {fold_id} verifier seal mismatch")
        sealed_inputs[fold_id] = {
            "run": run,
            "fold_verifier_sha256": verifier_sha,
            "raw_evidence_manifest": capture_fold_raw_manifest(
                run, weight_path_for_spec(spec)
            ),
        }

    rows: list[dict[str, object]] = []
    fold_seals: list[dict[str, object]] = []
    for state in states:
        fold_id = int(state["fold_id"])
        spec = spec_for_fold(specs, fold_id)
        sealed = sealed_inputs[fold_id]
        run = Path(str(sealed["run"]))
        independent_path = run / "independent-verification.json"
        verifier_sha_before = str(sealed["fold_verifier_sha256"])
        raw_before = sealed["raw_evidence_manifest"]
        if not isinstance(raw_before, Mapping):
            raise ValueError(f"fold {fold_id} raw evidence manifest is invalid")
        raw_metrics = _raw_metrics(run)
        verified = verify_fold_run(run, weight_path_for_spec(spec), spec)
        if verified.get("status") != "pass" or verified.get("fold_id") != fold_id:
            raise ValueError(f"fold {fold_id} generic independent verifier did not pass")
        generic_metrics = _validate_metrics(verified.get("test_metrics"))
        if raw_metrics != generic_metrics:
            raise ValueError(f"fold {fold_id} raw metrics disagree with generic verifier")
        if verified.get("raw_evidence_manifest") != raw_before:
            raise ValueError(f"fold {fold_id} generic raw evidence manifest mismatch")
        if verified.get("raw_evidence_unchanged_during_verification") is not True:
            raise ValueError(f"fold {fold_id} generic verifier did not seal raw evidence")
        recorded = json.loads(independent_path.read_text(encoding="utf-8"))
        if not isinstance(recorded, Mapping) or dict(recorded) != dict(verified):
            raise ValueError(f"fold {fold_id} sealed verifier result mismatch")
        wall_seconds = float(verified.get("wall_seconds", math.nan))
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError(f"fold {fold_id} runtime is invalid")
        row: dict[str, object] = {
            "fold_id": fold_id,
            "train_years": ";".join(str(year) for year in spec.train_years),
            "validation_year": spec.validation_year,
            "test_year": spec.test_year,
            "weight_filename": spec.filename,
            "weight_sha256": spec.sha256,
            "filename_ap": spec.filename_ap,
            **raw_metrics,
            "test_AP_minus_filename_ap": raw_metrics["test_AP"] - spec.filename_ap,
            "test_AP_absolute_difference_from_filename_ap": abs(
                raw_metrics["test_AP"] - spec.filename_ap
            ),
            "wall_seconds": wall_seconds,
            "run_directory": str(run),
            "fold_verifier_sha256": verifier_sha_before,
            "raw_evidence_manifest_sha256": _manifest_sha256(raw_before),
        }
        rows.append(row)
        fold_seals.append(
            {
                "fold_id": fold_id,
                "fold_verifier_sha256": verifier_sha_before,
                "raw_evidence_manifest": raw_before,
                "raw_evidence_manifest_sha256": _manifest_sha256(raw_before),
            }
        )

    for state in states:
        fold_id = int(state["fold_id"])
        spec = spec_for_fold(specs, fold_id)
        sealed = sealed_inputs[fold_id]
        run = Path(str(sealed["run"]))
        raw_after = capture_fold_raw_manifest(run, weight_path_for_spec(spec))
        verifier_sha_after = _sha256(run / "independent-verification.json")
        if (
            sealed["raw_evidence_manifest"] != raw_after
            or sealed["fold_verifier_sha256"] != verifier_sha_after
        ):
            raise ValueError(
                f"fold {fold_id} raw evidence changed during campaign verification"
            )

    summary = summarize_verified_rows(rows)
    generation_relative = Path("generations") / uuid.uuid4().hex
    generation_directory = campaign / generation_relative
    generation_directory.mkdir(parents=True, exist_ok=False)
    generation_name = generation_relative.as_posix()
    results_path = generation_directory / RESULTS_FILENAME
    summary_path = generation_directory / SUMMARY_FILENAME
    _write_csv_atomic(results_path, rows)
    _write_json_atomic(summary_path, summary)
    result = {
        "schema_version": 1,
        "status": "pass",
        "fold_count": 12,
        "generation": generation_name,
        "independent_implementation": True,
        "controller_metric_summaries_used": False,
        "raw_evidence_unchanged_during_verification": True,
        "results_sha256": _sha256(results_path),
        "summary_sha256": _sha256(summary_path),
        "fold_seals": fold_seals,
    }
    independent_path = generation_directory / INDEPENDENT_FILENAME
    _write_json_atomic(independent_path, result)
    output_paths = {
        RESULTS_FILENAME: results_path,
        SUMMARY_FILENAME: summary_path,
        INDEPENDENT_FILENAME: independent_path,
    }
    publication = {
        "schema_version": 1,
        "status": "pass",
        "fold_count": 12,
        "generation": generation_name,
        "outputs": {
            name: {
                "path": f"{generation_name}/{name}",
                "sha256": _sha256(path),
            }
            for name, path in output_paths.items()
        },
    }
    _write_json_atomic(campaign / PUBLICATION_FILENAME, publication)
    committed = load_committed_publication(campaign)
    if committed != publication:
        raise ValueError("campaign publication commit marker changed after write")
    return committed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-directory", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = verify_campaign(arguments.campaign_directory)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

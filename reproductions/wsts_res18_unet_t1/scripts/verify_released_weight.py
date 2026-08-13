"""Independent released-weight result validation primitives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from verify_calibration import (
    EXPECTED_COMMIT,
    EXPECTED_PATCH_SHA256,
    FIXED_DATA_ROOT,
    FIXED_ENVIRONMENT_PYTHON,
    rebuild_live_source_inventory,
    validate_import_scope_contents,
    validate_patch_text,
    validate_source_inventory_lineage,
)
from verify_full_fold import validate_provenance_copies
from released_weight_contract import (
    RELEASED_WEIGHT_FILENAMES as EXPECTED_RELEASED_WEIGHT_FILENAMES,
    REVISION as EXPECTED_REVISION,
    WEIGHT_PREFIX,
    WeightSpec,
    load_pinned_manifest,
    spec_for_fold,
    validate_local_weight,
)
from released_weight_path_security import (
    require_absent_recovery_path,
    require_pairwise_distinct_files,
    require_sealed_directory,
    require_sealed_regular_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DERIVED_UPSTREAM = (
    REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS-res18-runtime"
)
EXPECTED_ORIGINAL_UPSTREAM = REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS"
EXPECTED_PATCH = REPRODUCTION_ROOT / "patches" / "res18_import_scope.patch"
EXPECTED_WORKER_SMOKE = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1"
    / "fold2-train-val-smoke-workers8-20260809"
    / "smoke.json"
)
EXPECTED_FULL_ROOT = (
    REPOSITORY_ROOT / "artifacts" / "reproductions" / "wsts-res18-t1-full"
)
EXPECTED_WEIGHT_ROOT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1-official-weight"
)
PINNED_MANIFEST_PATH = REPRODUCTION_ROOT / "official_weights_manifest.json"
FILENAME_PROVENANCE = (
    "derived only from filenames in the pinned official All/T=1 weight manifest; "
    "not provenance for the paper table"
)
EXPECTED_LAUNCH_WEIGHT_CONTROLLER_SHA256 = (
    "de60464477f74d324bd2fadb451526814bc729fa71d1eeb167f53340626e080b"
)
EXPECTED_FOLD0_LAUNCH_WEIGHT_CONTROLLER_SHA256 = (
    "84db331675557425c1af7f98173bb623ae08088476a66cc14d5d0c0d05958b23"
)
EXPECTED_LAUNCH_WEIGHT_ENTRYPOINT_SHA256 = (
    "ea024f5982e1e3d34ddb5cdc53354d837dd00b755102249635db46f5cfeb3b04"
)
EXPECTED_TEST_METRICS = {
    "test_AP",
    "test_f1",
    "test_iou",
    "test_loss",
    "test_precision",
    "test_recall",
}
WEIGHT_RAW_RELATIVE_PATHS = (
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
OFFICIAL_TEST_OUTPUT_SOURCE_NAME = "test_pr_curve_data.npz"
OFFICIAL_TEST_OUTPUT_TARGET_NAME = "official-test-pr-curve-data.npz"
OFFICIAL_TEST_OUTPUT_PROVENANCE_NAME = "official-test-output-provenance.json"
OFFICIAL_TEST_OUTPUT_RECOVERY_AUTHORIZATION_NAME = (
    "official-test-output-recovery-authorization.json"
)
RECOVERY_AUTHORIZATION_RAW_RELATIVE_PATHS = (
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
    "completed.json",
)


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def derive_filename_manifest_evidence() -> dict[str, object]:
    values = [
        float(re.fullmatch(r"fold\d+_testAP(0\.\d+)\.pth", name).group(1))
        for name in EXPECTED_RELEASED_WEIGHT_FILENAMES
    ]
    return {
        "revision": EXPECTED_REVISION,
        "prefix": WEIGHT_PREFIX,
        "filenames": list(EXPECTED_RELEASED_WEIGHT_FILENAMES),
        "aggregate": {
            "count": len(values),
            "mean": sum(values) / len(values),
            "population_std": statistics.pstdev(values),
        },
        "paper_table_provenance": False,
        "provenance_note": FILENAME_PROVENANCE,
    }


def _fold2_spec() -> WeightSpec:
    return spec_for_fold(load_pinned_manifest(PINNED_MANIFEST_PATH), 2)


def _fold_claims(spec: WeightSpec, weight_path: Path) -> dict[str, object]:
    return {
        "fold_id": spec.fold_id,
        "train_years": list(spec.train_years),
        "validation_year": spec.validation_year,
        "test_year": spec.test_year,
        "weight_path": str(weight_path.resolve()),
        "weight_filename": spec.filename,
        "weight_size": spec.size,
        "weight_sha256": spec.sha256,
        "filename_ap": spec.filename_ap,
    }


def verify_fold_claims(
    payload: Mapping[str, object], *, spec: WeightSpec, weight_path: Path
) -> None:
    expected = _fold_claims(spec, weight_path)
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError("released-weight fold claims mismatch")


def _verify_fold_claims_compatible(
    payload: Mapping[str, object], *, spec: WeightSpec, weight_path: Path
) -> None:
    if "fold_id" in payload:
        verify_fold_claims(payload, spec=spec, weight_path=weight_path)
        return
    if spec.fold_id != 2:
        raise ValueError("released-weight fold claims are missing")
    legacy = {
        "weight_path": str(weight_path.resolve()),
        "weight_sha256": spec.sha256,
        "filename_ap": spec.filename_ap,
    }
    if any(payload.get(field) != value for field, value in legacy.items()):
        raise ValueError("released-weight legacy Fold-2 claims mismatch")


def validate_test_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    if set(metrics) != EXPECTED_TEST_METRICS:
        raise ValueError("independent verifier requires six legal test metrics")
    values = {name: float(value) for name, value in metrics.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("independent verifier requires six legal test metrics")
    if values["test_loss"] < 0 or any(
        not 0.0 <= value <= 1.0
        for name, value in values.items()
        if name != "test_loss"
    ):
        raise ValueError("independent verifier requires six legal test metrics")
    return values


def reconstruct_observer_statistics(
    started_utc: str,
    exit_mtime_ns: int,
    gpu_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    try:
        start = datetime.fromisoformat(started_utc).timestamp()
    except (TypeError, ValueError) as error:
        raise ValueError("independent weight start timestamp is invalid") from error
    wall = exit_mtime_ns / 1_000_000_000 - start
    if not math.isfinite(wall) or wall <= 0 or len(gpu_rows) < 2:
        raise ValueError("independent weight observer evidence is incomplete")
    observer = [float(row["observer_seconds"]) for row in gpu_rows]
    used = [float(row["memory_used_mib"]) for row in gpu_rows]
    utilization = [float(row["utilization_gpu_percent"]) for row in gpu_rows]
    child = [float(row["child_memory_mib"]) for row in gpu_rows]
    if not all(
        math.isfinite(value) and value >= 0
        for value in [*observer, *used, *utilization, *child]
    ):
        raise ValueError("independent weight GPU sample is invalid")
    intervals = [current - previous for previous, current in zip(observer, observer[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("independent weight GPU timestamps do not strictly increase")
    span = observer[-1] - observer[0]
    child_available = any(value > 0 for value in child)
    return {
        "wall_seconds": float(wall),
        "gpu_sample_count": float(len(gpu_rows)),
        "gpu_interval_count": len(intervals),
        "gpu_interval_mean_seconds": float(statistics.mean(intervals)),
        "gpu_interval_median_seconds": float(statistics.median(intervals)),
        "gpu_observed_effective_hz": float(len(intervals) / span),
        "gpu_peak_is_observed_sample_max": True,
        "gpu_peak_may_miss_between_sample_transients": True,
        "peak_gpu_used_mib": float(max(used)),
        "peak_child_process_mib": float(max(child)) if child_available else None,
        "child_process_memory_available": child_available,
        "gpu_utilization_min_percent": float(min(utilization)),
        "gpu_utilization_median_percent": float(statistics.median(utilization)),
        "gpu_utilization_max_percent": float(max(utilization)),
    }


def capture_weight_raw_manifest(
    run_directory: Path, weight_path: Path
) -> dict[str, object]:
    run = run_directory.resolve()
    entries: list[dict[str, object]] = []
    for relative in WEIGHT_RAW_RELATIVE_PATHS:
        path = run / relative
        if not path.is_file():
            raise ValueError(f"independent weight raw evidence is missing: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    weight = weight_path.resolve()
    entries.append(
        {
            "path": "released-weight::" + str(weight),
            "bytes": weight.stat().st_size,
            "sha256": _sha256(weight),
        }
    )
    return {"schema_version": 1, "entry_count": len(entries), "entries": entries}


def _capture_recovery_raw_manifest(run: Path) -> dict[str, object]:
    entries = []
    for relative in RECOVERY_AUTHORIZATION_RAW_RELATIVE_PATHS:
        path = run / relative
        if not path.is_file():
            raise ValueError(
                f"official test output recovery raw evidence is missing: {relative}"
            )
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return {"schema_version": 1, "entry_count": len(entries), "entries": entries}


def _json_sha256(payload: object) -> str:
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


def _is_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_aware_iso8601(value: object) -> bool:
    if type(value) is not str or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})",
        value,
    ) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _validate_inventory_schema(
    inventory: object, *, expected_source: Path, candidate_count: int
) -> list[dict[str, object]]:
    if type(inventory) is not dict or set(inventory) != {"cwd", "candidates"}:
        raise ValueError("official test output provenance inventory schema mismatch")
    candidates = inventory["candidates"]
    if (
        type(inventory["cwd"]) is not str
        or inventory["cwd"] != str(EXPECTED_DERIVED_UPSTREAM.absolute())
        or type(candidates) is not list
        or len(candidates) != candidate_count
    ):
        raise ValueError("official test output provenance inventory schema mismatch")
    validated: list[dict[str, object]] = []
    for candidate in candidates:
        if (
            type(candidate) is not dict
            or set(candidate)
            != {"path", "name", "bytes", "sha256", "ctime_ns", "mtime_ns"}
            or type(candidate["path"]) is not str
            or candidate["path"] != str(expected_source)
            or type(candidate["name"]) is not str
            or candidate["name"] != OFFICIAL_TEST_OUTPUT_SOURCE_NAME
            or type(candidate["bytes"]) is not int
            or candidate["bytes"] < 0
            or not _is_sha256(candidate["sha256"])
            or type(candidate["ctime_ns"]) is not int
            or candidate["ctime_ns"] < 0
            or type(candidate["mtime_ns"]) is not int
            or candidate["mtime_ns"] < 0
        ):
            raise ValueError("official test output provenance inventory candidate schema mismatch")
        validated.append(candidate)
    return validated


def _validate_schema_v2_marker(
    marker: object, *, expected_source: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    keys = {
        "schema_version", "status", "collection_mode", "atomic_move",
        "scientific_child_relaunched", "run_directory", "source", "target",
        "bytes", "sha256", "source_mtime_ns", "child_pid", "command",
        "command_sha256", "started_utc", "exit_marker_mtime_ns", "pre_inventory",
        "post_inventory", "recovery_authorization_sha256", "collected_utc",
    }
    if type(marker) is not dict or set(marker) != keys:
        raise ValueError("official test output provenance schema-v2 mismatch")
    if (
        type(marker["schema_version"]) is not int
        or marker["schema_version"] != 2
        or type(marker["status"]) is not str
        or type(marker["collection_mode"]) is not str
        or type(marker["atomic_move"]) is not bool
        or type(marker["scientific_child_relaunched"]) is not bool
        or type(marker["run_directory"]) is not str
        or type(marker["source"]) is not str
        or type(marker["target"]) is not str
        or type(marker["bytes"]) is not int
        or marker["bytes"] < 0
        or not _is_sha256(marker["sha256"])
        or type(marker["source_mtime_ns"]) is not int
        or marker["source_mtime_ns"] < 0
        or type(marker["child_pid"]) is not int
        or marker["child_pid"] <= 0
        or type(marker["command"]) is not list
        or any(type(item) is not str for item in marker["command"])
        or not _is_sha256(marker["command_sha256"])
        or not _is_aware_iso8601(marker["started_utc"])
        or type(marker["exit_marker_mtime_ns"]) is not int
        or marker["exit_marker_mtime_ns"] < 0
        or (
            marker["recovery_authorization_sha256"] is not None
            and not _is_sha256(marker["recovery_authorization_sha256"])
        )
        or not _is_aware_iso8601(marker["collected_utc"])
    ):
        raise ValueError("official test output provenance schema-v2 type mismatch")
    pre = marker["pre_inventory"]
    post = marker["post_inventory"]
    if type(pre) is not dict or type(post) is not dict:
        raise ValueError("official test output provenance inventory schema mismatch")
    mode = marker["collection_mode"]
    pre_key = (
        "pre_inventory_sha256" if mode == "live-postprocess" else "launch_preflight_sha256"
    )
    if (
        mode not in {"live-postprocess", "offline-finalize-existing"}
        or set(pre) != {"cwd", "candidates", "mode", pre_key}
        or type(pre["cwd"]) is not str
        or pre["cwd"] != str(EXPECTED_DERIVED_UPSTREAM.absolute())
        or type(pre["candidates"]) is not list
        or pre["candidates"] != []
        or type(pre["mode"]) is not str
        or not _is_sha256(pre[pre_key])
    ):
        raise ValueError("official test output provenance pre-inventory schema mismatch")
    candidates = _validate_inventory_schema(
        post, expected_source=expected_source, candidate_count=1
    )
    return marker, pre, candidates[0]


def _validate_legacy_fold2_marker(marker: object) -> dict[str, object]:
    keys = {
        "bytes", "child_pid", "command_sha256", "moved_utc", "reason", "sha256",
        "source", "status", "target",
    }
    if (
        type(marker) is not dict
        or set(marker) != keys
        or type(marker["bytes"]) is not int
        or marker["bytes"] < 0
        or type(marker["child_pid"]) is not int
        or marker["child_pid"] <= 0
        or not _is_sha256(marker["command_sha256"])
        or not _is_aware_iso8601(marker["moved_utc"])
        or type(marker["reason"]) is not str
        or not _is_sha256(marker["sha256"])
        or type(marker["source"]) is not str
        or type(marker["status"]) is not str
        or type(marker["target"]) is not str
    ):
        raise ValueError("official test output provenance legacy contract mismatch")
    return marker


def verify_official_test_output_provenance(
    run_directory: Path, *, fold_id: int
) -> dict[str, object]:
    """Independently bind the collected official NPZ to one child lineage."""
    run = require_sealed_directory(
        run_directory.absolute(),
        checked_root=run_directory.absolute(),
        description="official test output provenance run",
    )
    target = run / OFFICIAL_TEST_OUTPUT_TARGET_NAME
    marker_path = run / OFFICIAL_TEST_OUTPUT_PROVENANCE_NAME
    lineage_paths = [
        target,
        marker_path,
        run / "started.json",
        run / "effective-command.json",
        run / "completed.json",
        run / "exit-code.txt",
    ]
    for path in lineage_paths:
        require_sealed_regular_file(
            path, checked_root=run, description="official test output provenance evidence"
        )
    require_pairwise_distinct_files(
        lineage_paths, description="official test output provenance evidence"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    effective = json.loads(
        (run / "effective-command.json").read_text(encoding="utf-8")
    )
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, Mapping) for value in (marker, started, effective, completed)):
        raise ValueError("official test output provenance is invalid")
    expected_source = EXPECTED_DERIVED_UPSTREAM.absolute() / OFFICIAL_TEST_OUTPUT_SOURCE_NAME
    if type(marker) is not dict:
        raise ValueError("official test output provenance is invalid")
    if "schema_version" not in marker:
        marker = _validate_legacy_fold2_marker(marker)
    else:
        marker, pre, candidate = _validate_schema_v2_marker(
            marker, expected_source=expected_source
        )
    if (
        marker.get("source") != str(expected_source)
        or marker.get("target") != str(target)
        or marker.get("bytes") != target.stat().st_size
        or marker.get("sha256") != _sha256(target)
        or marker.get("child_pid") != started.get("pid")
        or marker.get("command_sha256") != started.get("command_sha256")
        or effective.get("command_sha256") != started.get("command_sha256")
        or effective.get("command") != started.get("command")
        or effective.get("cwd") != str(EXPECTED_DERIVED_UPSTREAM.resolve())
        or completed.get("status") != "pass"
        or completed.get("exit_code") != 0
        or completed.get("pid") != started.get("pid")
    ):
        raise ValueError("official test output provenance lineage mismatch")
    if "schema_version" not in marker:
        if (
            fold_id != 2
            or marker.get("status") != "preserved-official-test-output"
            or marker.get("reason") != (
                "official test emitted evaluation output in the derived runtime cwd; "
                "preserved unchanged inside the immutable run evidence directory so "
                "source-integrity verification can require the exact authorized patch only"
            )
        ):
            raise ValueError("official test output provenance legacy contract mismatch")
        return {
            "official_test_output_provenance_verified": True,
            "official_test_output_collection_mode": "legacy-fold2-preservation",
            "official_test_output_sha256": _sha256(target),
            "official_test_output_provenance_sha256": _sha256(marker_path),
        }
    post = marker["post_inventory"]
    if (
        marker.get("status") != "pass"
        or marker.get("atomic_move") is not True
        or marker.get("scientific_child_relaunched") is not False
        or marker.get("run_directory") != str(run)
        or marker.get("command") != started.get("command")
        or marker.get("started_utc") != started.get("started_utc")
        or post.get("cwd") != str(EXPECTED_DERIVED_UPSTREAM.absolute())
        or candidate.get("sha256") != _sha256(target)
        or candidate.get("bytes") != target.stat().st_size
        or candidate.get("mtime_ns") != marker.get("source_mtime_ns")
    ):
        raise ValueError("official test output provenance inventory mismatch")
    try:
        start_ns = int(
            datetime.fromisoformat(str(started["started_utc"])).timestamp() * 1e9
        )
        source_mtime_ns = int(marker["source_mtime_ns"])
        exit_ns = int(marker["exit_marker_mtime_ns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("official test output provenance timestamps are invalid") from error
    if (
        exit_ns != (run / "exit-code.txt").stat().st_mtime_ns
        or target.stat().st_mtime_ns != source_mtime_ns
        or not start_ns <= source_mtime_ns <= exit_ns + 2_000_000_000
        or os.path.lexists(expected_source)
    ):
        raise ValueError("official test output provenance time or move mismatch")
    mode = marker.get("collection_mode")
    if mode == "live-postprocess":
        pre_path = run / "official-test-output-cwd-pre.json"
        expected_pre_marker = {
            "schema_version": 1,
            "status": "pass",
            "inventory_scope": "immediately before released-weight scientific child",
            "cwd": str(EXPECTED_DERIVED_UPSTREAM.resolve()),
            "candidates": [],
        }
        pre_marker = json.loads(pre_path.read_text(encoding="utf-8"))
        if (
            pre_marker != expected_pre_marker
            or pre.get("mode") != "recorded-immediately-before-launch"
            or pre.get("cwd") != str(EXPECTED_DERIVED_UPSTREAM.resolve())
            or pre.get("candidates") != []
            or pre.get("pre_inventory_sha256") != _sha256(pre_path)
            or marker.get("recovery_authorization_sha256") is not None
        ):
            raise ValueError("official test output provenance live pre-inventory mismatch")
    elif mode == "offline-finalize-existing":
        authorization_path = run / OFFICIAL_TEST_OUTPUT_RECOVERY_AUTHORIZATION_NAME
        require_sealed_regular_file(
            authorization_path,
            checked_root=run,
            description="official test output recovery authorization",
        )
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if type(authorization) is not dict:
            raise ValueError("official test output provenance recovery authorization mismatch")
        external_path_value = authorization.get("external_authorization_path")
        if type(external_path_value) is not str:
            raise ValueError("official test output provenance recovery authorization mismatch")
        external_path = Path(external_path_value).absolute()
        require_sealed_regular_file(
            external_path,
            checked_root=external_path.parent,
            description="external recovery authorization",
        )
        external_authorization = json.loads(external_path.read_text(encoding="utf-8"))
        if type(external_authorization) is not dict:
            raise ValueError("official test output provenance recovery authorization mismatch")
        require_pairwise_distinct_files(
            [authorization_path, marker_path, target, external_path],
            description="official test output recovery authorization",
        )
        raw_manifest = _capture_recovery_raw_manifest(run)
        spec = spec_for_fold(load_pinned_manifest(PINNED_MANIFEST_PATH), fold_id)
        weight_path = (
            REPRODUCTION_ROOT / ".local" / "released-weights" / spec.filename
        ).resolve()
        launch_provenance = effective.get("provenance_sha256")
        expected_authorization = {
            "schema_version": 1,
            "status": "pass",
            "authorized_action": (
                "one-time atomic adoption of the existing completed child output only"
            ),
            "scientific_child_relaunched": False,
            "fold_id": fold_id,
            "run_directory": str(run),
            "weight": _fold_claims(spec, weight_path),
            "source": str(expected_source),
            "target": str(target),
            "child_pid": started.get("pid"),
            "command_sha256": started.get("command_sha256"),
            "source_sha256": _sha256(target),
            "source_bytes": target.stat().st_size,
            "source_ctime_ns": post["candidates"][0].get("ctime_ns"),
            "source_mtime_ns": source_mtime_ns,
            "started_utc": started.get("started_utc"),
            "exit_marker_mtime_ns": exit_ns,
            "raw_evidence_manifest": raw_manifest,
            "raw_evidence_manifest_sha256": _json_sha256(raw_manifest),
            "launch_provenance_sha256": launch_provenance,
            "launch_provenance_manifest_sha256": _json_sha256(launch_provenance),
            "external_authorization_path": str(external_path),
            "external_authorization_sha256": _sha256(external_path),
            "external_authorization": external_authorization,
        }
        if (
            fold_id != 0
            or pre.get("mode") != "reconstructed-from-exact-launch-preflight"
            or pre.get("cwd") != str(EXPECTED_DERIVED_UPSTREAM.resolve())
            or pre.get("candidates") != []
            or pre.get("launch_preflight_sha256") != _sha256(run / "preflight.json")
            or marker.get("recovery_authorization_sha256") != _sha256(authorization_path)
            or not _exact_json_value(authorization, expected_authorization)
        ):
            raise ValueError("official test output provenance recovery authorization mismatch")
    else:
        raise ValueError("official test output provenance collection mode mismatch")
    return {
        "official_test_output_provenance_verified": True,
        "official_test_output_collection_mode": mode,
        "official_test_output_sha256": _sha256(target),
        "official_test_output_provenance_sha256": _sha256(marker_path),
    }


def verify_first_verifier_failure(run_directory: Path) -> dict[str, bool]:
    marker_path = run_directory.resolve() / "independent-verifier-first-failure.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "status": "fail",
        "failure_stage": "independent_verifier_postprocess",
        "error": "ValueError: released-weight finite test metrics are missing",
        "scientific_child_exit_code": 0,
        "scientific_child_relaunched": False,
        "retrospective_documentation": True,
    }
    if not isinstance(marker, Mapping) or any(
        marker.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("first verifier failure evidence mismatch")
    return {"first_verifier_failure_preserved": True}


def _require_exact_marker(
    marker: object, expected: Mapping[str, object], message: str
) -> None:
    if not isinstance(marker, Mapping) or dict(marker) != dict(expected):
        raise ValueError(message)


def verify_retrospective_markers(
    run_directory: Path,
    *,
    filename_manifest: Mapping[str, object] | None = None,
    launch_recorded_dependency: object = None,
    current_dependency: Mapping[str, object] | None = None,
) -> dict[str, bool]:
    run = run_directory.resolve()
    first_path = run / "independent-verifier-first-failure.json"
    first_preserved = False
    if first_path.is_file():
        first_preserved = verify_first_verifier_failure(run)[
            "first_verifier_failure_preserved"
        ]
    preflight_augmentation = run / "offline-preflight-augmentation.json"
    if preflight_augmentation.is_file():
        if filename_manifest is None:
            raise ValueError("preflight augmentation context is missing")
        marker = json.loads(preflight_augmentation.read_text(encoding="utf-8"))
        _require_exact_marker(
            marker,
            {
                "status": "pass",
                "augmentation_scope": "offline filename-manifest provenance only",
                "launch_preflight_sha256": _sha256(run / "preflight.json"),
                "filename_manifest": dict(filename_manifest),
                "scientific_child_relaunched": False,
            },
            "preflight augmentation mismatch",
        )
    full_augmentation = run / "offline-full-dependency-augmentation.json"
    augmentation_present = full_augmentation.is_file()
    if augmentation_present:
        if current_dependency is None:
            raise ValueError("full dependency augmentation context is missing")
        validate_full_dependency_augmentation(
            run, launch_recorded_dependency, current_dependency
        )
    finalization_path = run / "offline-finalization.json"
    if finalization_path.is_file():
        marker = json.loads(finalization_path.read_text(encoding="utf-8"))
        _require_exact_marker(
            marker,
            {
                "status": "pass",
                "finalization_scope": "existing scientific output only",
                "weight_result_sha256": _sha256(run / "weight-result.json"),
                "scientific_child_relaunched": False,
            },
            "offline finalization mismatch",
        )
    return {
        "first_verifier_failure_preserved": first_preserved,
        "offline_full_dependency_augmentation_present": augmentation_present,
    }


def build_weight_provenance_contract(
    spec: WeightSpec | None = None,
) -> tuple[dict[str, Path], dict[str, str]]:
    spec = _fold2_spec() if spec is None else spec
    authoritative = {
        "upstream.lock.json": REPRODUCTION_ROOT / "upstream.lock.json",
        "smoke.json": EXPECTED_WORKER_SMOKE,
        "res18_import_scope.patch": EXPECTED_PATCH,
    }
    if spec.fold_id == 2:
        launch_time = {
            "official_weight_entrypoint.py": EXPECTED_LAUNCH_WEIGHT_ENTRYPOINT_SHA256,
            "evaluate_released_weight.py": EXPECTED_LAUNCH_WEIGHT_CONTROLLER_SHA256,
        }
    elif spec.fold_id == 0:
        launch_time = {
            "evaluate_released_weight.py": (
                EXPECTED_FOLD0_LAUNCH_WEIGHT_CONTROLLER_SHA256
            )
        }
        authoritative["official_weight_entrypoint.py"] = Path(__file__).with_name(
            "official_weight_entrypoint.py"
        )
    else:
        launch_time = {}
        authoritative.update(
            {
                "official_weight_entrypoint.py": Path(__file__).with_name(
                    "official_weight_entrypoint.py"
                ),
                "evaluate_released_weight.py": Path(__file__).with_name(
                    "evaluate_released_weight.py"
                ),
            }
        )
    return authoritative, launch_time


def build_expected_weight_command(
    run_directory: Path,
    derived_root: Path,
    weight_path: Path,
    spec: WeightSpec | None = None,
) -> list[str]:
    spec = _fold2_spec() if spec is None else spec
    run = run_directory.resolve()
    derived = derived_root.resolve()
    weight = weight_path.resolve()
    configs = derived / "cfgs"
    return [
        str(FIXED_ENVIRONMENT_PYTHON.resolve()),
        str(Path(__file__).with_name("official_weight_entrypoint.py").resolve()),
        "--upstream-root",
        str(derived),
        "--weights-path",
        str(weight),
        f"--config={(configs / 'unet' / 'res18_monotemporal.yaml').as_posix()}",
        f"--trainer={(configs / 'trainer_single_gpu.yaml').as_posix()}",
        f"--data={(configs / 'data_monotemporal_full_features.yaml').as_posix()}",
        f"--data.data_dir={FIXED_DATA_ROOT.resolve()}",
        f"--data.data_fold_id={spec.fold_id}",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        f"--trainer.default_root_dir={run}",
        "--do_train=false",
        "--do_test=false",
    ]


def verify_weight_command_lineage(
    payloads: Mapping[str, Mapping[str, object]],
    expected_command: Sequence[str],
    expected_hash: str,
) -> None:
    contracts = {
        "launch": ("command", "command_sha256"),
        "run_lock": ("command", "command_sha256"),
        "preflight": ("command", "command_sha256"),
        "started": ("command", "command_sha256"),
        "effective": ("command", "command_sha256"),
        "result": ("command_sha256",),
    }
    for label, fields in contracts.items():
        payload = payloads.get(label)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label} marker is missing")
        for field in fields:
            if field not in payload:
                raise ValueError(f"{label}.{field} is required")
        if "command" in fields:
            command = payload["command"]
            if not isinstance(command, list) or not all(
                isinstance(argument, str) for argument in command
            ):
                raise ValueError(f"{label}.command must be a list of strings")
            if command != list(expected_command):
                raise ValueError(f"{label}.command differs from exact test-only command")
        if not isinstance(payload["command_sha256"], str):
            raise ValueError(f"{label}.command_sha256 must be a string")
        if payload["command_sha256"] != expected_hash:
            raise ValueError(f"{label}.command_sha256 mismatch")


def reconstruct_weight_evidence(output: str, exit_code: int) -> dict[str, object]:
    if exit_code != 0:
        raise ValueError("released-weight child exit code is not zero")
    strict = re.findall(
        r"(?m)^WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=(\d+)\s*$", output
    )
    if len(strict) != 1 or int(strict[0]) <= 0:
        raise ValueError("released-weight strict-load evidence is missing or ambiguous")
    if re.search(r"Testing DataLoader 0:\s*100%", output) is None:
        raise ValueError("released-weight test completion evidence is missing")
    if re.search(r"\bEpoch\s+\d+:|Trainer\.fit|max_steps=", output):
        raise ValueError("released-weight output contains training evidence")
    if "Validation DataLoader" in output:
        raise ValueError("released-weight output contains validation evidence")
    if re.search(r"Predicting DataLoader|trainer\.predict", output):
        raise ValueError("released-weight output contains predict evidence")
    metrics = {
        name: float(raw)
        for name, raw in re.findall(
            r"(?m)[│|]\s*(test_(?:AP|f1|iou|loss|precision|recall))\s*[│|]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            output,
        )
    }
    metric_name = r"test_(?:AP|f1|iou|loss|precision|recall)"
    scalar = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    observed_matches = re.findall(
        rf"(?m)(?:^|[\r\n])[ \t]*({metric_name})[ \t]{{2,}}({scalar})"
        r"(?=[ \t]*(?:[\r\n]|$))",
        output,
    )
    for name, raw in observed_matches:
        if name in metrics:
            raise ValueError(f"released-weight duplicate test metric: {name}")
        metrics[name] = float(raw)
    metrics = validate_test_metrics(metrics)
    peak = re.findall(r"(?m)^WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\d+)\s*$", output)
    if len(peak) != 1 or int(peak[0]) <= 0:
        raise ValueError("released-weight peak allocation evidence is missing")
    return {
        "exit_code": 0,
        "strict_load": True,
        "loaded_tensor_count": int(strict[0]),
        "train_invoked": False,
        "validation_invoked": False,
        "predict_invoked": False,
        "test_metrics": metrics,
        "peak_allocated_bytes": int(peak[0]),
    }


def validate_weight_result(payload: Mapping[str, object]) -> dict[str, float]:
    if payload.get("exit_code") != 0:
        raise ValueError("released-weight child exit code is not zero")
    if payload.get("strict_load") is not True:
        raise ValueError("released weight was not loaded with strict=True")
    for action in ("train", "validation", "predict"):
        if payload.get(f"{action}_invoked") is not False:
            raise ValueError(f"released-weight evaluation invoked {action}")
    metrics = payload.get("test_metrics")
    if not isinstance(metrics, Mapping) or "test_AP" not in metrics:
        raise ValueError("released-weight test_AP is missing")
    test_ap = float(metrics["test_AP"])
    filename_ap = float(payload.get("filename_ap", math.nan))
    if not math.isfinite(test_ap) or not 0.0 <= test_ap <= 1.0:
        raise ValueError("released-weight test_AP is invalid")
    if not math.isfinite(filename_ap) or not 0.0 <= filename_ap <= 1.0:
        raise ValueError("released-weight filename AP is invalid")
    return {
        "test_AP": test_ap,
        "filename_ap": filename_ap,
        "filename_ap_absolute_difference": abs(test_ap - filename_ap),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write(os.linesep)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root.resolve()), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "git provenance verification failed")
    return result.stdout


def _verify_weight_runtime_provenance(
    run: Path,
    effective: Mapping[str, object],
    preflight: Mapping[str, object],
    spec: WeightSpec,
) -> dict[str, object]:
    original = EXPECTED_ORIGINAL_UPSTREAM.resolve()
    derived = EXPECTED_DERIVED_UPSTREAM.resolve()
    patch = EXPECTED_PATCH.resolve()
    original_commit = _git(original, "rev-parse", "HEAD").strip()
    derived_commit = _git(derived, "rev-parse", "HEAD").strip()
    if original_commit != EXPECTED_COMMIT or _git(
        original, "status", "--porcelain"
    ).splitlines():
        raise ValueError("original pinned checkout integrity failed")
    if derived_commit != EXPECTED_COMMIT or _git(
        derived, "status", "--porcelain"
    ).splitlines() != [" M src/models/__init__.py"]:
        raise ValueError("derived runtime checkout integrity failed")
    validate_import_scope_contents(
        (original / "src" / "models" / "__init__.py").read_text(encoding="utf-8"),
        (derived / "src" / "models" / "__init__.py").read_text(encoding="utf-8"),
    )
    validate_patch_text(patch.read_text(encoding="utf-8"))
    if _sha256(patch) != EXPECTED_PATCH_SHA256:
        raise ValueError("tracked runtime patch SHA-256 changed")
    if _git(derived, "diff", "--name-only").splitlines() != [
        "src/models/__init__.py"
    ]:
        raise ValueError("derived runtime contains an unauthorized change")
    actual_diff = _git(
        derived, "diff", "--no-ext-diff", "--", "src/models/__init__.py"
    )
    runtime_patch = preflight.get("runtime_patch")
    if not isinstance(runtime_patch, Mapping):
        raise ValueError("weight preflight runtime patch is missing")
    for field, expected in {
        "base_commit": EXPECTED_COMMIT,
        "patch_sha256": EXPECTED_PATCH_SHA256,
        "git_diff": actual_diff,
        "git_diff_sha256": hashlib.sha256(actual_diff.encode("utf-8")).hexdigest(),
        "affected_files": ["src/models/__init__.py"],
        "original_checkout_clean": True,
        "scientific_code_touched": False,
        "derived_root": str(derived),
    }.items():
        if runtime_patch.get(field) != expected:
            raise ValueError(f"weight preflight runtime_patch.{field} mismatch")
    authoritative_sources, launch_time_hashes = build_weight_provenance_contract(spec)
    copied_entrypoint = run / "provenance" / "official_weight_entrypoint.py"
    if spec.fold_id == 2 and (
        copied_entrypoint.is_file()
        and _sha256(copied_entrypoint) != EXPECTED_LAUNCH_WEIGHT_ENTRYPOINT_SHA256
    ):
        authoritative_sources["official_weight_entrypoint.py"] = Path(
            __file__
        ).with_name("official_weight_entrypoint.py")
        launch_time_hashes.pop("official_weight_entrypoint.py")
    copied_controller = run / "provenance" / "evaluate_released_weight.py"
    if spec.fold_id == 2 and (
        copied_controller.is_file()
        and _sha256(copied_controller) != EXPECTED_LAUNCH_WEIGHT_CONTROLLER_SHA256
    ):
        authoritative_sources["evaluate_released_weight.py"] = Path(__file__).with_name(
            "evaluate_released_weight.py"
        )
        launch_time_hashes.pop("evaluate_released_weight.py")
    recorded_hashes = effective.get("provenance_sha256")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError("weight effective provenance hash manifest is missing")
    validate_provenance_copies(
        run / "provenance",
        recorded_hashes,
        authoritative_sources,
        actual_diff,
        launch_time_hashes=launch_time_hashes,
    )
    return {
        "original_upstream_commit": original_commit,
        "original_upstream_clean": True,
        "derived_upstream_commit": derived_commit,
        "derived_status": [" M src/models/__init__.py"],
        "patch_sha256": EXPECTED_PATCH_SHA256,
        "provenance_copies_verified": True,
    }


def validate_full_dependency_augmentation(
    run: Path,
    launch_recorded: object,
    current: Mapping[str, object],
) -> None:
    marker = json.loads(
        (run.resolve() / "offline-full-dependency-augmentation.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "status": "pass",
        "augmentation_scope": "offline current full-run dependency hashes only",
        "launch_recorded_dependency": launch_recorded,
        "current_dependency": dict(current),
        "scientific_child_relaunched": False,
    }
    _require_exact_marker(marker, expected, "full dependency augmentation mismatch")


def _verify_full_dependency(
    run: Path, preflight: Mapping[str, object]
) -> dict[str, object]:
    lock = json.loads(
        (EXPECTED_FULL_ROOT / "fold2-full-launch.lock.json").read_text(encoding="utf-8")
    )
    full_run = Path(str(lock.get("run_directory", ""))).resolve()
    result_path = full_run / "full-result.json"
    independent_path = full_run / "independent-verification.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    if result.get("status") != "pass" or result.get("optimizer_steps") != 10_000:
        raise ValueError("full-run result dependency is not complete")
    if independent.get("status") != "pass" or independent.get(
        "optimizer_steps"
    ) != 10_000:
        raise ValueError("full-run independent dependency did not pass")
    for field in ("command_sha256", "checkpoint_sha256"):
        if result.get(field) != independent.get(field):
            raise ValueError(f"full-run dependency {field} mismatch")
    expected = {
        "run_directory": str(full_run),
        "result_sha256": _sha256(result_path),
        "independent_sha256": _sha256(independent_path),
        "independent_status": "pass",
    }
    launch_recorded = preflight.get("full_run_dependency")
    augmentation_path = run / "offline-full-dependency-augmentation.json"
    if augmentation_path.is_file():
        validate_full_dependency_augmentation(run, launch_recorded, expected)
    elif launch_recorded != expected:
        raise ValueError("full dependency augmentation is missing")
    return expected


def verify_run(
    run_directory: Path, weight_path: Path, spec: WeightSpec
) -> dict[str, object]:
    canonical = spec_for_fold(
        load_pinned_manifest(PINNED_MANIFEST_PATH), spec.fold_id
    )
    if spec != canonical:
        raise ValueError("caller spec differs from canonical pinned manifest spec")
    spec = canonical
    run = run_directory.resolve()
    weight = weight_path.resolve()
    if (
        run.parent != EXPECTED_WEIGHT_ROOT.resolve()
        or not run.name.startswith(f"fold{spec.fold_id}-weight-")
    ):
        raise ValueError("released-weight run directory is outside the fixed artifact root")
    expected_weight = (
        REPRODUCTION_ROOT / ".local" / "released-weights" / spec.filename
    ).resolve()
    if weight != expected_weight:
        raise ValueError("released-weight path is not the pinned fold cache path")
    validate_local_weight(weight, spec)
    raw_before = capture_weight_raw_manifest(run, weight)
    result = json.loads((run / "weight-result.json").read_text(encoding="utf-8"))
    preflight = json.loads((run / "preflight.json").read_text(encoding="utf-8"))
    effective = json.loads(
        (run / "effective-command.json").read_text(encoding="utf-8")
    )
    global_lock = json.loads(
        (run.parent / f"fold{spec.fold_id}-weight-evaluation.lock.json").read_text(
            encoding="utf-8"
        )
    )
    run_lock = json.loads((run / "launch.lock.json").read_text(encoding="utf-8"))
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    _verify_fold_claims_compatible(
        preflight, spec=spec, weight_path=weight
    )
    _verify_fold_claims_compatible(result, spec=spec, weight_path=weight)
    filename_manifest = derive_filename_manifest_evidence()
    if preflight.get("filename_manifest") != filename_manifest:
        augmentation_path = run / "offline-preflight-augmentation.json"
        if not augmentation_path.is_file():
            raise ValueError("weight preflight filename manifest evidence is missing")
        augmentation = json.loads(augmentation_path.read_text(encoding="utf-8"))
        expected_augmentation = {
            "status": "pass",
            "augmentation_scope": "offline filename-manifest provenance only",
            "launch_preflight_sha256": _sha256(run / "preflight.json"),
            "filename_manifest": filename_manifest,
            "scientific_child_relaunched": False,
        }
        for field, expected in expected_augmentation.items():
            if augmentation.get(field) != expected:
                raise ValueError(f"weight offline preflight augmentation mismatch: {field}")
    expected_command = build_expected_weight_command(
        run, EXPECTED_DERIVED_UPSTREAM, weight, spec
    )
    command_hash = _command_sha256(expected_command)
    verify_weight_command_lineage(
        {
            "launch": global_lock,
            "run_lock": run_lock,
            "preflight": preflight,
            "started": started,
            "effective": effective,
            "result": result,
        },
        expected_command,
        command_hash,
    )
    provenance = _verify_weight_runtime_provenance(run, effective, preflight, spec)
    full_dependency = _verify_full_dependency(run, preflight)
    retrospective = verify_retrospective_markers(
        run,
        filename_manifest=filename_manifest,
        launch_recorded_dependency=preflight.get("full_run_dependency"),
        current_dependency=full_dependency,
    )
    output_provenance = verify_official_test_output_provenance(
        run, fold_id=spec.fold_id
    )
    before = json.loads((run / "source-data-pre.json").read_text(encoding="utf-8"))
    after = json.loads((run / "source-data-post.json").read_text(encoding="utf-8"))
    live = rebuild_live_source_inventory(FIXED_DATA_ROOT)
    validate_source_inventory_lineage(before, after, live, FIXED_DATA_ROOT)
    output = (run / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (
        run / "stderr.log"
    ).read_text(encoding="utf-8", errors="replace")
    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    evidence = reconstruct_weight_evidence(output, exit_code)
    with (run / "gpu.csv").open(encoding="utf-8", newline="") as handle:
        gpu_rows = list(csv.DictReader(handle))
    observer = reconstruct_observer_statistics(
        str(started.get("started_utc")),
        (run / "exit-code.txt").stat().st_mtime_ns,
        gpu_rows,
    )
    independently_recorded = {
        "status": "pass",
        **evidence,
        **_fold_claims(spec, weight),
        "filename_manifest": filename_manifest,
        "command_sha256": command_hash,
        "child_pid": started.get("pid"),
        **observer,
    }
    summary = validate_weight_result(independently_recorded)
    independently_recorded.update(summary)
    for field, expected in independently_recorded.items():
        if field == "loaded_tensor_count" or (
            field
            in {
                "fold_id",
                "train_years",
                "validation_year",
                "test_year",
                "weight_filename",
                "weight_size",
            }
            and spec.fold_id == 2
            and field not in result
        ):
            continue
        if result.get(field) != expected:
            raise ValueError(f"released-weight result mismatch: {field}")
    if (
        completed.get("status") != "pass"
        or completed.get("exit_code") != 0
        or completed.get("pid") != started.get("pid")
    ):
        raise ValueError("released-weight completion marker mismatch")
    raw_after = capture_weight_raw_manifest(run, weight)
    if raw_before != raw_after:
        raise ValueError("weight raw evidence changed during independent verification")
    raw_manifest_sha = hashlib.sha256(
        json.dumps(raw_before, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "pass",
        "fold_id": spec.fold_id,
        "command_sha256": command_hash,
        "weight_sha256": spec.sha256,
        "loaded_tensor_count": evidence["loaded_tensor_count"],
        "test_metrics": evidence["test_metrics"],
        "filename_manifest": filename_manifest,
        "filename_aggregate": filename_manifest["aggregate"],
        **observer,
        "raw_evidence_manifest": raw_before,
        "raw_evidence_manifest_sha256": raw_manifest_sha,
        "raw_evidence_unchanged_during_verification": True,
        "data_inventory_live_verified": True,
        "data_file_count": before["file_count"],
        "data_total_bytes": before["total_bytes"],
        "full_run_dependency": full_dependency,
        **summary,
        **provenance,
        **retrospective,
        **output_provenance,
        "independent_implementation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, choices=range(12), default=2)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--weight-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    expected_output = arguments.run_directory.resolve() / "independent-verification.json"
    if arguments.output.resolve() != expected_output:
        raise ValueError("verifier output must be inside the selected run")
    spec = spec_for_fold(
        load_pinned_manifest(PINNED_MANIFEST_PATH), arguments.fold_id
    )
    result = verify_run(arguments.run_directory, arguments.weight_path, spec)
    _write_json_atomic(arguments.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

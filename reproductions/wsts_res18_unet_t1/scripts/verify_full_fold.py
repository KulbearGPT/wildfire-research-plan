"""Independently verify a preserved full Fold-2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from verify_calibration import (
    EXPECTED_COMMIT,
    EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT,
    EXPECTED_PATCH_SHA256,
    FIXED_DATA_ROOT,
    FIXED_ENVIRONMENT_PYTHON,
    derive_dynamic_positive_weight_provenance,
    rebuild_live_source_inventory,
    validate_import_scope_contents,
    validate_patch_text,
    validate_source_inventory_lineage,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORIGINAL_UPSTREAM = REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS"
EXPECTED_DERIVED_UPSTREAM = (
    REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS-res18-runtime"
)
EXPECTED_PATCH = REPRODUCTION_ROOT / "patches" / "res18_import_scope.patch"
EXPECTED_WORKER_SMOKE = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1"
    / "fold2-train-val-smoke-workers8-20260809"
    / "smoke.json"
)
EXPECTED_LAUNCH_RUNNER_SHA256 = (
    "36250bb080d702add45ffd0698c3d37af42c85432625658709bc47f11cee4e3d"
)
EXPECTED_OBSERVER_TEST_METRIC_FAILURE = (
    "ValueError: test_AP is missing from the Lightning test table"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_expected_full_command(run_directory: Path, derived_root: Path) -> list[str]:
    """Construct the full command without trusting any run artifact."""
    run = run_directory.resolve()
    derived = derived_root.resolve()
    configs = derived / "cfgs"
    return [
        str(FIXED_ENVIRONMENT_PYTHON.resolve()),
        str(Path(__file__).with_name("official_entrypoint.py").resolve()),
        "--upstream-root",
        str(derived),
        f"--config={(configs / 'unet' / 'res18_monotemporal.yaml').as_posix()}",
        f"--trainer={(configs / 'trainer_single_gpu.yaml').as_posix()}",
        f"--data={(configs / 'data_monotemporal_full_features.yaml').as_posix()}",
        f"--data.data_dir={FIXED_DATA_ROOT.resolve()}",
        "--data.data_fold_id=2",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        "--trainer.max_steps=10000",
        f"--trainer.default_root_dir={run}",
        "--do_test=true",
    ]


def validate_full_provenance_paths(
    original_root: Path, derived_root: Path, patch_path: Path
) -> None:
    if original_root.resolve() != EXPECTED_ORIGINAL_UPSTREAM.resolve():
        raise ValueError("original upstream path is not the pinned reproduction checkout")
    if derived_root.resolve() != EXPECTED_DERIVED_UPSTREAM.resolve():
        raise ValueError("derived upstream path is not the pinned runtime checkout")
    if patch_path.resolve() != EXPECTED_PATCH.resolve():
        raise ValueError("runtime patch path is not the tracked reproduction patch")


def validate_provenance_copies(
    provenance_directory: Path,
    recorded_hashes: Mapping[str, object],
    authoritative_sources: Mapping[str, Path],
    expected_diff: str,
    *,
    launch_time_hashes: Mapping[str, str] | None = None,
) -> None:
    """Require copied bytes/content to match both hashes and authoritative inputs."""
    launch_hashes = {} if launch_time_hashes is None else dict(launch_time_hashes)
    if set(launch_hashes) & (set(authoritative_sources) | {"derived-runtime.diff"}):
        raise ValueError("launch-time provenance names overlap authoritative inputs")
    expected_names = (
        set(authoritative_sources) | set(launch_hashes) | {"derived-runtime.diff"}
    )
    if set(recorded_hashes) != expected_names:
        raise ValueError("provenance hash manifest has an unexpected name set")
    provenance = provenance_directory.resolve()
    for name, expected_sha in launch_hashes.items():
        copied = provenance / name
        if not copied.is_file() or not isinstance(recorded_hashes[name], str):
            raise ValueError(f"provenance copy/hash is missing: {name}")
        if _sha256(copied) != recorded_hashes[name]:
            raise ValueError(f"provenance copy SHA-256 mismatch: {name}")
        if recorded_hashes[name] != expected_sha:
            raise ValueError(f"launch-time provenance SHA-256 mismatch: {name}")
    for name, source in authoritative_sources.items():
        copied = provenance / name
        if not copied.is_file() or not isinstance(recorded_hashes[name], str):
            raise ValueError(f"provenance copy/hash is missing: {name}")
        if _sha256(copied) != recorded_hashes[name]:
            raise ValueError(f"provenance copy SHA-256 mismatch: {name}")
        if copied.read_bytes() != source.resolve().read_bytes():
            raise ValueError(
                f"provenance copy differs from authoritative source: {name}"
            )
    copied_diff = provenance / "derived-runtime.diff"
    if not copied_diff.is_file() or not isinstance(
        recorded_hashes["derived-runtime.diff"], str
    ):
        raise ValueError("provenance copy/hash is missing: derived-runtime.diff")
    if _sha256(copied_diff) != recorded_hashes["derived-runtime.diff"]:
        raise ValueError("provenance copy SHA-256 mismatch: derived-runtime.diff")
    if copied_diff.read_text(encoding="utf-8") != expected_diff:
        raise ValueError("copied derived diff differs from the live runtime diff")


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


def verify_recorded_command(
    payloads: Mapping[str, Mapping[str, object]],
    expected_command: Sequence[str],
    expected_hash: str,
) -> None:
    required = {
        "launch": ("command", "command_sha256"),
        "run_lock": ("command", "command_sha256"),
        "preflight": ("command", "command_sha256"),
        "started": ("command", "command_sha256"),
        "effective": ("command", "command_sha256"),
        "result": ("command_sha256",),
    }
    for label, fields in required.items():
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
                raise ValueError(f"{label}.command differs from the exact full command")
        if not isinstance(payload["command_sha256"], str):
            raise ValueError(f"{label}.command_sha256 must be a string")
        if payload["command_sha256"] != expected_hash:
            raise ValueError(f"{label}.command_sha256 mismatch")


def verify_result_summary(
    raw: Mapping[str, object], recorded: Mapping[str, object]
) -> None:
    if dict(raw) != dict(recorded):
        raise ValueError("full result summary differs from independently reconstructed values")
    checkpoint = Path(str(raw.get("best_checkpoint", "")))
    if not checkpoint.is_file():
        raise ValueError("recorded best checkpoint is missing")
    if raw.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("best checkpoint SHA-256 mismatch")
    if raw.get("optimizer_steps") != 10_000:
        raise ValueError("full result optimizer step count mismatch")


def verify_completion_lineage(
    run: Path,
    recorded: Mapping[str, object],
    started: Mapping[str, object],
) -> dict[str, bool]:
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if not isinstance(completed, Mapping):
        raise ValueError("completed marker must be an object")
    pid = started.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("started.pid must be a positive integer")
    if (
        completed.get("status") != "pass"
        or completed.get("exit_code") != 0
        or completed.get("pid") != pid
    ):
        raise ValueError("completed marker does not match the exit-zero child")
    failure_path = run / "failure.json"
    recovery_path = run / "observer-recovery.json"
    if not failure_path.exists():
        if recovery_path.exists():
            raise ValueError("observer recovery exists without a prior failure")
        return {
            "completed_marker_verified": True,
            "observer_recovery_verified": False,
        }
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if not isinstance(failure, Mapping) or not isinstance(recovery, Mapping):
        raise ValueError("observer recovery lineage markers must be objects")
    if (
        failure.get("status") != "fail"
        or failure.get("error") != EXPECTED_OBSERVER_TEST_METRIC_FAILURE
        or failure.get("training_retry_performed") is not False
    ):
        raise ValueError("observer recovery prior failure mismatch")
    if recovery.get("failure_sha256") != _sha256(failure_path):
        raise ValueError("observer recovery failure SHA-256 mismatch")
    if recovery.get("full_result_sha256") != _sha256(run / "full-result.json"):
        raise ValueError("observer recovery full-result SHA-256 mismatch")
    test_metrics = recorded.get("test_metrics")
    if not isinstance(test_metrics, Mapping) or "test_AP" not in test_metrics:
        raise ValueError("recorded test_AP is missing from observer recovery lineage")
    expected_recovery = {
        "status": "pass",
        "recovery_type": "observer_parser_finalize_existing",
        "previous_error": EXPECTED_OBSERVER_TEST_METRIC_FAILURE,
        "command_sha256": recorded.get("command_sha256"),
        "checkpoint_sha256": recorded.get("checkpoint_sha256"),
        "optimizer_steps": 10_000,
        "test_AP": test_metrics["test_AP"],
        "previous_failure_preserved": True,
        "scientific_child_relaunched": False,
    }
    for field, expected in expected_recovery.items():
        if recovery.get(field) != expected:
            raise ValueError(f"observer recovery field mismatch: {field}")
    if (
        completed.get("finalized_existing") is not True
        or completed.get("observer_recovery") is not True
        or completed.get("completed_utc") != recovery.get("finalized_utc")
    ):
        raise ValueError("completed marker does not govern observer recovery")
    return {
        "completed_marker_verified": True,
        "observer_recovery_verified": True,
    }


def _parse_observed_windows_metrics(output: str) -> dict[str, float]:
    metric_name = r"test_(?:AP|f1|iou|loss|precision|recall)"
    scalar = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    matches = re.findall(
        rf"(?m)(?:^|[\r\n])[ \t]*({metric_name})[ \t]{{2,}}({scalar})"
        r"(?=[ \t]*(?:[\r\n]|$))",
        output,
    )
    values: dict[str, float] = {}
    for name, raw in matches:
        if name in values:
            raise ValueError(f"independent verifier found duplicate test metric: {name}")
        values[name] = float(raw)
    return values


def _parse_metrics(output: str) -> dict[str, float]:
    values = {
        name: float(raw)
        for name, raw in re.findall(
            r"(?m)[│|]\s*(test_(?:AP|f1|iou|loss|precision|recall))\s*[│|]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            output,
        )
    }
    observed = _parse_observed_windows_metrics(output)
    for name, value in observed.items():
        if name in values:
            raise ValueError(f"independent verifier found duplicate test metric: {name}")
        values[name] = value
    if "test_AP" not in values:
        raise ValueError("independent verifier could not reconstruct test_AP")
    return values


def verify_runtime_provenance(
    run: Path,
    original: Path,
    derived: Path,
    patch: Path,
    effective: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Verify live Git state, the sole patch, and every preserved provenance copy."""
    original_commit = _git(original, "rev-parse", "HEAD").strip()
    original_status = _git(original, "status", "--porcelain").splitlines()
    derived_commit = _git(derived, "rev-parse", "HEAD").strip()
    derived_status = _git(derived, "status", "--porcelain").splitlines()
    if original_commit != EXPECTED_COMMIT or original_status:
        raise ValueError("original pinned checkout integrity failed")
    if derived_commit != EXPECTED_COMMIT or derived_status != [
        " M src/models/__init__.py"
    ]:
        raise ValueError("derived runtime checkout integrity failed")

    original_initializer = (original / "src" / "models" / "__init__.py").read_text(
        encoding="utf-8"
    )
    derived_initializer = (derived / "src" / "models" / "__init__.py").read_text(
        encoding="utf-8"
    )
    validate_import_scope_contents(original_initializer, derived_initializer)
    patch_text = patch.read_text(encoding="utf-8")
    validate_patch_text(patch_text)
    if _sha256(patch) != EXPECTED_PATCH_SHA256:
        raise ValueError("tracked runtime patch SHA-256 changed")
    if _git(derived, "diff", "--name-only").splitlines() != [
        "src/models/__init__.py"
    ]:
        raise ValueError("derived runtime contains a change outside the initializer")
    actual_diff = _git(
        derived, "diff", "--no-ext-diff", "--", "src/models/__init__.py"
    )

    runtime_patch = preflight.get("runtime_patch")
    if not isinstance(runtime_patch, Mapping):
        raise ValueError("preflight.runtime_patch is missing")
    exact_runtime_fields = {
        "base_commit": EXPECTED_COMMIT,
        "patch_sha256": EXPECTED_PATCH_SHA256,
        "git_diff": actual_diff,
        "git_diff_sha256": hashlib.sha256(actual_diff.encode("utf-8")).hexdigest(),
        "affected_files": ["src/models/__init__.py"],
        "original_checkout_clean": True,
        "scientific_code_touched": False,
        "derived_root": str(derived.resolve()),
    }
    for field, expected in exact_runtime_fields.items():
        if runtime_patch.get(field) != expected:
            raise ValueError(f"preflight.runtime_patch.{field} mismatch")
    if preflight.get("upstream_commit") != EXPECTED_COMMIT:
        raise ValueError("preflight upstream commit mismatch")
    if preflight.get("data_file_count") != 607 or preflight.get(
        "data_total_bytes"
    ) != 24_242_259_023:
        raise ValueError("preflight source inventory summary mismatch")

    authoritative_sources = {
        "upstream.lock.json": REPRODUCTION_ROOT / "upstream.lock.json",
        "smoke.json": EXPECTED_WORKER_SMOKE,
        "official_entrypoint.py": Path(__file__).with_name("official_entrypoint.py"),
        "res18_import_scope.patch": patch,
        "res18_monotemporal.yaml": derived
        / "cfgs"
        / "unet"
        / "res18_monotemporal.yaml",
        "trainer_single_gpu.yaml": derived
        / "cfgs"
        / "trainer_single_gpu.yaml",
        "data_monotemporal_full_features.yaml": derived
        / "cfgs"
        / "data_monotemporal_full_features.yaml",
    }
    recorded_hashes = effective.get("provenance_sha256")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError("effective provenance hash manifest is missing")
    validate_provenance_copies(
        run / "provenance",
        recorded_hashes,
        authoritative_sources,
        actual_diff,
        launch_time_hashes={
            "run_full_fold.py": EXPECTED_LAUNCH_RUNNER_SHA256,
        },
    )
    return {
        "original_upstream_commit": original_commit,
        "original_upstream_clean": True,
        "derived_upstream_commit": derived_commit,
        "derived_status": derived_status,
        "patch_sha256": EXPECTED_PATCH_SHA256,
        "derived_diff_sha256": hashlib.sha256(actual_diff.encode("utf-8")).hexdigest(),
        "provenance_copies_verified": True,
    }


def verify_run(
    run_directory: Path,
    original_root: Path,
    derived_root: Path,
    patch_path: Path,
) -> dict[str, object]:
    run = run_directory.resolve()
    original = original_root.resolve()
    derived = derived_root.resolve()
    patch = patch_path.resolve()
    validate_full_provenance_paths(original, derived, patch)
    effective = json.loads((run / "effective-command.json").read_text(encoding="utf-8"))
    if not isinstance(effective, Mapping):
        raise ValueError("effective command marker must be an object")
    command = effective.get("command")
    if not isinstance(command, list) or not all(
        isinstance(argument, str) for argument in command
    ):
        raise ValueError("effective command must be a list of strings")
    expected_command = build_expected_full_command(run, derived)
    if command != expected_command:
        raise ValueError("effective command differs from independent full command")
    command_hash = _command_sha256(expected_command)
    global_lock = json.loads(
        (run.parent / "fold2-full-launch.lock.json").read_text(encoding="utf-8")
    )
    run_lock = json.loads((run / "launch.lock.json").read_text(encoding="utf-8"))
    preflight = json.loads((run / "preflight.json").read_text(encoding="utf-8"))
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    recorded = json.loads((run / "full-result.json").read_text(encoding="utf-8"))
    completion = verify_completion_lineage(run, recorded, started)
    verify_recorded_command(
        {
            "launch": global_lock,
            "run_lock": run_lock,
            "preflight": preflight,
            "started": started,
            "effective": effective,
            "result": recorded,
        },
        expected_command,
        command_hash,
    )
    provenance = verify_runtime_provenance(
        run, original, derived, patch, effective, preflight
    )
    source_pre = json.loads((run / "source-data-pre.json").read_text(encoding="utf-8"))
    source_post = json.loads((run / "source-data-post.json").read_text(encoding="utf-8"))
    live_source = rebuild_live_source_inventory(FIXED_DATA_ROOT)
    validate_source_inventory_lineage(
        source_pre, source_post, live_source, FIXED_DATA_ROOT
    )
    dynamic_weight = derive_dynamic_positive_weight_provenance(
        (derived / "cfgs" / "unet" / "res18_monotemporal.yaml").read_text(
            encoding="utf-8"
        ),
        (derived / "src" / "train.py").read_text(encoding="utf-8"),
        (run / "config.yaml").read_text(encoding="utf-8"),
    )
    if dynamic_weight["effective_pos_class_weight"] != EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT:
        raise ValueError("effective positive-class weight mismatch")
    output = (run / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (
        run / "stderr.log"
    ).read_text(encoding="utf-8", errors="replace")
    if int((run / "exit-code.txt").read_text(encoding="utf-8").strip()) != 0:
        raise ValueError("full child exit code is not zero")
    if re.search(r"max_steps=10000`?\s+reached", output) is None:
        raise ValueError("independent max_steps=10000 evidence is missing")
    if re.search(r"Testing DataLoader 0:\s*100%", output) is None:
        raise ValueError("independent test completion evidence is missing")
    if "Predicting DataLoader" in output:
        raise ValueError("predict action exists in full run output")
    checkpoints = sorted(run.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError("independent checkpoint inventory is not exactly one")
    metrics = _parse_metrics(output)
    raw = dict(recorded)
    checkpoint_match = re.search(
        r"best-epoch=(\d+)-val_avg_precision=([0-9.]+)\.ckpt$",
        checkpoints[0].name,
    )
    if checkpoint_match is None:
        raise ValueError("independent best checkpoint filename provenance is missing")
    peak_matches = re.findall(
        r"(?m)^WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\d+)\s*$", output
    )
    if len(peak_matches) != 1 or int(peak_matches[0]) <= 0:
        raise ValueError("independent peak-allocation sentinel is missing or ambiguous")
    raw["status"] = "pass"
    raw["fold"] = 2
    raw["command_sha256"] = command_hash
    raw["test_metrics"] = metrics
    raw["best_checkpoint"] = str(checkpoints[0].resolve())
    raw["checkpoint_sha256"] = _sha256(checkpoints[0])
    raw["optimizer_steps"] = 10_000
    raw["best_epoch"] = int(checkpoint_match.group(1))
    raw["best_validation_AP_from_filename"] = float(checkpoint_match.group(2))
    raw["peak_allocated_bytes"] = int(peak_matches[0])
    for field, value in dynamic_weight.items():
        if field in raw:
            raw[field] = value
    verify_result_summary(raw, recorded)
    return {
        "status": "pass",
        "command_sha256": command_hash,
        "optimizer_steps": 10_000,
        "test_metrics": metrics,
        "checkpoint_sha256": _sha256(checkpoints[0]),
        "best_epoch": int(checkpoint_match.group(1)),
        "best_validation_AP_from_filename": float(checkpoint_match.group(2)),
        "peak_allocated_bytes": int(peak_matches[0]),
        "data_inventory_live_verified": True,
        "data_file_count": source_pre["file_count"],
        "data_total_bytes": source_pre["total_bytes"],
        **dynamic_weight,
        **provenance,
        **completion,
        "independent_implementation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--original-upstream", required=True, type=Path)
    parser.add_argument("--derived-upstream", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = verify_run(
        arguments.run_directory,
        arguments.original_upstream,
        arguments.derived_upstream,
        arguments.patch,
    )
    arguments.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

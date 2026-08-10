"""Run the pinned official Fold-2 Res18-U-Net protocol once to 10,000 steps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from run_calibration import (
    DATA_ROOT,
    DERIVED_UPSTREAM_ROOT,
    ENVIRONMENT_PYTHON,
    EXPECTED_CODE_COMMIT,
    EXPECTED_WEIGHTS_REVISION,
    REPRODUCTION_ROOT,
    RUNTIME_PATCH_PATH,
    UPSTREAM_ROOT,
    WORKER_SMOKE_PATH,
    _build_child_environment,
    _gpu_preflight,
    _official_import_only_preflight,
    _prepare_derived_runtime,
    _runtime_preflight,
    _sample_nvidia_smi,
    _sha256,
    acquire_launch_lock,
    assert_source_inventories_identical,
    build_source_inventory,
    measure_wall_seconds_from_markers,
    observe_process,
    parse_peak_allocated,
    parse_progress,
    summarize_gpu_samples,
    validate_effective_runtime_config,
    validate_train_validation_smoke_evidence,
    verify_inventory,
    verify_upstream,
    write_json_atomic,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FULL_ARTIFACTS_ROOT = (
    REPOSITORY_ROOT / "artifacts" / "reproductions" / "wsts-res18-t1-full"
)
GLOBAL_LAUNCH_LOCK = FULL_ARTIFACTS_ROOT / "fold2-full-launch.lock.json"
EXPECTED_EFFECTIVE_POS_WEIGHT = 608.4653828020165
MINIMUM_DISK_FREE_BYTES = 20 * 1024**3
OBSERVER_TEST_METRIC_FAILURE = (
    "ValueError: test_AP is missing from the Lightning test table"
)
FULL_RAW_RELATIVE_PATHS = (
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
    "failure.json",
)
RAW_SEAL_SCOPE = "current offline re-finalization; not retrospective proof"


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_full_command(
    run_directory: Path,
    *,
    python_executable: Path = ENVIRONMENT_PYTHON,
    entrypoint: Path = Path(__file__).with_name("official_entrypoint.py"),
    upstream_root: Path = DERIVED_UPSTREAM_ROOT,
    data_root: Path = DATA_ROOT,
) -> list[str]:
    """Construct the complete ordered authors-protocol command."""
    upstream = upstream_root.resolve()
    run = run_directory.resolve()
    config_root = upstream / "cfgs"
    return [
        str(python_executable.resolve()),
        str(entrypoint.resolve()),
        "--upstream-root",
        str(upstream),
        f"--config={(config_root / 'unet' / 'res18_monotemporal.yaml').as_posix()}",
        f"--trainer={(config_root / 'trainer_single_gpu.yaml').as_posix()}",
        f"--data={(config_root / 'data_monotemporal_full_features.yaml').as_posix()}",
        f"--data.data_dir={data_root.resolve()}",
        "--data.data_fold_id=2",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        "--trainer.max_steps=10000",
        f"--trainer.default_root_dir={run}",
        "--do_test=true",
    ]


def validate_full_command(command: Sequence[str], run_directory: Path) -> None:
    expected = build_full_command(
        run_directory,
        python_executable=Path(command[0]) if command else ENVIRONMENT_PYTHON,
        entrypoint=Path(command[1]) if len(command) > 1 else Path(__file__).with_name(
            "official_entrypoint.py"
        ),
        upstream_root=(
            Path(command[3]) if len(command) > 3 else DERIVED_UPSTREAM_ROOT
        ),
        data_root=DATA_ROOT,
    )
    if list(command) != expected:
        raise ValueError("effective command differs from the exact full-fold command")


def parse_test_metrics(output: str) -> dict[str, float]:
    """Parse scalar Lightning test metrics from its rendered result table."""
    matches = re.findall(
        r"(?m)[│|]\s*(test_(?:AP|f1|iou|loss|precision|recall))\s*[│|]\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        output,
    )
    metric_name = r"test_(?:AP|f1|iou|loss|precision|recall)"
    scalar = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    matches.extend(
        re.findall(
            rf"(?m)(?:^|[\r\n])[ \t]*({metric_name})[ \t]{{2,}}({scalar})"
            r"(?=[ \t]*(?:[\r\n]|$))",
            output,
        )
    )
    metrics: dict[str, float] = {}
    for name, raw in matches:
        if name in metrics:
            raise ValueError(f"duplicate test metric: {name}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"test metric must be finite: {name}")
        if name != "test_loss" and not 0.0 <= value <= 1.0:
            raise ValueError(f"test metric must be within [0,1]: {name}")
        if name == "test_loss" and value < 0.0:
            raise ValueError("test loss must be non-negative")
        metrics[name] = value
    expected = {
        "test_AP",
        "test_f1",
        "test_iou",
        "test_loss",
        "test_precision",
        "test_recall",
    }
    if set(metrics) != expected:
        raise ValueError("all six test metrics are required from the Lightning table")
    return metrics


def validate_checkpoint_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    version = payload.get("pytorch-lightning_version")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("checkpoint global_step must be a positive integer")
    if not isinstance(version, str) or not version:
        raise ValueError("checkpoint Lightning version is missing")
    return {
        "best_epoch": epoch,
        "best_checkpoint_global_step": global_step,
        "checkpoint_lightning_version": version,
    }


def read_checkpoint_metadata(checkpoint: Path) -> dict[str, object]:
    import torch

    payload = torch.load(checkpoint.resolve(), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return validate_checkpoint_metadata(payload)


def parse_displayed_validation_ap(event_text: str, best_epoch: int) -> float:
    value: float | None = None
    progress = re.compile(
        rf"\bEpoch\s+{best_epoch}:.*?(\d+)\s*/\s*(\d+).*?"
        r"val_avg_precision=([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    )
    for line_number, line in enumerate(event_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            fragment = payload["text"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid stream event on line {line_number}") from error
        if not isinstance(fragment, str):
            raise ValueError("stream event text must be a string")
        match = progress.search(fragment)
        if match is not None and int(match.group(1)) == int(match.group(2)):
            value = float(match.group(3))
    if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("best-epoch displayed validation AP is missing or invalid")
    return value


def capture_full_raw_manifest(run_directory: Path) -> dict[str, object]:
    run = run_directory.resolve()
    checkpoints = sorted(run.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError("raw evidence requires exactly one checkpoint")
    relative_paths = [*FULL_RAW_RELATIVE_PATHS, checkpoints[0].relative_to(run).as_posix()]
    entries: list[dict[str, object]] = []
    for relative in relative_paths:
        path = run / Path(relative)
        if not path.is_file():
            raise ValueError(f"fixed raw evidence file is missing: {relative}")
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {"schema_version": 1, "entry_count": len(entries), "entries": entries}


def validate_full_success(
    event_text: str,
    combined_output: str,
    exit_code: int,
    checkpoints: Sequence[Path],
) -> dict[str, object]:
    """Fail closed unless one complete train-best-test protocol finished."""
    if exit_code != 0:
        raise ValueError(f"full child exit code is not zero: {exit_code}")
    samples = parse_progress(event_text)
    if not samples or samples[-1][0] != 10_000:
        observed = samples[-1][0] if samples else None
        raise ValueError(f"full run did not reach exactly 10000 optimizer steps: {observed}")
    if re.search(r"max_steps=10000`?\s+reached", combined_output) is None:
        raise ValueError("exact max_steps=10000 stop evidence is missing")
    if re.search(r"Testing DataLoader 0:\s*100%", combined_output) is None:
        raise ValueError("test completion evidence is missing")
    if re.search(r"Predicting DataLoader|trainer\.predict|\bdo_predict=true\b", combined_output):
        raise ValueError("predict action was invoked")
    if re.search(r"CUDA out of memory|DefaultCPUAllocator: not enough memory", combined_output):
        raise ValueError("OOM evidence exists in a nominally successful full run")
    if len(checkpoints) != 1:
        raise ValueError("full run must preserve exactly one best checkpoint")
    checkpoint = checkpoints[0].resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("best checkpoint is missing or empty")
    metrics = parse_test_metrics(combined_output)
    peak_allocated = parse_peak_allocated(combined_output)
    return {
        "optimizer_steps": 10_000,
        "test_metrics": metrics,
        "best_checkpoint": str(checkpoint),
        "peak_allocated_bytes": peak_allocated,
    }


def _create_run_directory() -> Path:
    FULL_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = FULL_ARTIFACTS_ROOT / f"fold2-full-{stamp}-{uuid.uuid4().hex[:8]}"
    target.mkdir()
    return target.resolve()


def _run_checked(command: Sequence[str], cwd: Path | None = None) -> str:
    import subprocess

    result = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=True, check=False, timeout=60
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise ValueError(f"preflight command failed: {detail}")
    return result.stdout.strip()


def _copy_provenance(
    run_directory: Path, runtime_patch: Mapping[str, object]
) -> dict[str, str]:
    provenance = run_directory / "provenance"
    provenance.mkdir()
    sources = {
        "upstream.lock.json": REPRODUCTION_ROOT / "upstream.lock.json",
        "smoke.json": WORKER_SMOKE_PATH,
        "official_entrypoint.py": Path(__file__).with_name("official_entrypoint.py"),
        "run_full_fold.py": Path(__file__),
        "res18_import_scope.patch": RUNTIME_PATCH_PATH,
        "res18_monotemporal.yaml": DERIVED_UPSTREAM_ROOT
        / "cfgs"
        / "unet"
        / "res18_monotemporal.yaml",
        "trainer_single_gpu.yaml": DERIVED_UPSTREAM_ROOT
        / "cfgs"
        / "trainer_single_gpu.yaml",
        "data_monotemporal_full_features.yaml": DERIVED_UPSTREAM_ROOT
        / "cfgs"
        / "data_monotemporal_full_features.yaml",
    }
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        target = provenance / name
        shutil.copy2(source, target)
        hashes[name] = _sha256(target)
    diff = provenance / "derived-runtime.diff"
    diff.write_text(str(runtime_patch["git_diff"]), encoding="utf-8")
    hashes[diff.name] = _sha256(diff)
    return hashes


def _preflight(run_directory: Path, *, require_lock_absent: bool = True) -> tuple[dict[str, object], dict[str, object]]:
    if require_lock_absent and GLOBAL_LAUNCH_LOCK.exists():
        raise FileExistsError(f"full-fold launch lock already exists: {GLOBAL_LAUNCH_LOCK}")
    if _run_checked(["git", "status", "--porcelain"], REPOSITORY_ROOT):
        raise ValueError("tracked reproduction worktree must be clean before full launch")
    smoke = json.loads(WORKER_SMOKE_PATH.read_text(encoding="utf-8"))
    validate_train_validation_smoke_evidence(smoke)
    verify_inventory(DATA_ROOT, selected_years=(2018, 2019, 2020, 2021))
    inventory = build_source_inventory(DATA_ROOT)
    if inventory["file_count"] != 607 or inventory["total_bytes"] != 24_242_259_023:
        raise ValueError("live WSTS source inventory does not match the frozen 607-file set")
    commit = verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
    runtime_patch = _prepare_derived_runtime()
    runtime = _runtime_preflight()
    gpu = _gpu_preflight()
    import_only = _official_import_only_preflight(run_directory)
    disk = shutil.disk_usage(run_directory)
    if disk.free < MINIMUM_DISK_FREE_BYTES:
        raise ValueError("artifact volume has less than 20 GiB free")
    command = build_full_command(run_directory)
    validate_full_command(command, run_directory)
    metadata: dict[str, object] = {
        "status": "pass",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "single Fold-2 full training and best-checkpoint test",
        "upstream_commit": commit,
        "weights_revision": EXPECTED_WEIGHTS_REVISION,
        "runtime_patch": runtime_patch,
        "runtime": runtime,
        "runtime_import_only": import_only,
        "gpu": gpu,
        "disk_free_bytes": disk.free,
        "data_file_count": inventory["file_count"],
        "data_total_bytes": inventory["total_bytes"],
        "command": command,
        "command_sha256": _command_sha256(command),
    }
    return metadata, inventory


def parse_full_result(run_directory: Path) -> dict[str, object]:
    """Finalize one terminal full-run directory from preserved raw evidence."""
    run = run_directory.resolve()
    effective = json.loads((run / "effective-command.json").read_text(encoding="utf-8"))
    stdout = (run / "stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr = (run / "stderr.log").read_text(encoding="utf-8", errors="replace")
    events = (run / "stream-events.jsonl").read_text(encoding="utf-8")
    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    checkpoints = sorted(run.rglob("*.ckpt"))
    summary = validate_full_success(events, stdout + "\n" + stderr, exit_code, checkpoints)
    checkpoint = Path(str(summary["best_checkpoint"]))
    checkpoint_match = re.search(
        r"best-epoch=(\d+)-val_avg_precision=([0-9.]+)\.ckpt$", checkpoint.name
    )
    if checkpoint_match is None:
        raise ValueError("best checkpoint filename lacks epoch/validation AP provenance")
    checkpoint_metadata = read_checkpoint_metadata(checkpoint)
    if checkpoint_metadata["best_epoch"] != int(checkpoint_match.group(1)):
        raise ValueError("checkpoint epoch differs from its filename label")
    displayed_validation_ap = parse_displayed_validation_ap(
        events, int(checkpoint_match.group(1))
    )
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    wall_seconds = measure_wall_seconds_from_markers(
        str(started["started_utc"]), (run / "exit-code.txt").stat().st_mtime_ns
    )
    with (run / "gpu.csv").open(encoding="utf-8", newline="") as handle:
        gpu_rows = list(csv.DictReader(handle))
    source_before = json.loads((run / "source-data-pre.json").read_text(encoding="utf-8"))
    source_after = json.loads((run / "source-data-post.json").read_text(encoding="utf-8"))
    assert_source_inventories_identical(source_before, source_after)
    effective_config = validate_effective_runtime_config(run / "config.yaml")
    if effective_config["effective_pos_class_weight"] != EXPECTED_EFFECTIVE_POS_WEIGHT:
        raise ValueError("full run effective positive-class weight changed")
    result: dict[str, object] = {
        "status": "pass",
        "fold": 2,
        "optimizer_steps": 10_000,
        "wall_seconds": wall_seconds,
        "wall_hours": wall_seconds / 3600,
        "best_checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        **checkpoint_metadata,
        "best_validation_AP_filename_rounded": float(checkpoint_match.group(2)),
        "best_validation_AP_displayed": displayed_validation_ap,
        "command_sha256": effective["command_sha256"],
        **effective_config,
        **summary,
        **summarize_gpu_samples(gpu_rows),
    }
    return result


def _launch_once() -> tuple[Path, dict[str, object]]:
    run = _create_run_directory()
    try:
        preflight, source_before = _preflight(run)
        write_json_atomic(run / "preflight.json", preflight)
        write_json_atomic(run / "source-data-pre.json", source_before)
        provenance = _copy_provenance(run, preflight["runtime_patch"])
        command = list(preflight["command"])
        command_hash = str(preflight["command_sha256"])
        child_environment, recorded_environment = _build_child_environment(
            run, preflight["runtime"]
        )
        launch = {
            "authorized_optimizer_steps": 10_000,
            "command": command,
            "command_sha256": command_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(run),
            "single_launch_no_retry": True,
        }
        acquire_launch_lock(GLOBAL_LAUNCH_LOCK, launch)
        acquire_launch_lock(run / "launch.lock.json", launch)
        write_json_atomic(
            run / "effective-command.json",
            {
                "command": command,
                "command_sha256": command_hash,
                "cwd": str(DERIVED_UPSTREAM_ROOT.resolve()),
                "environment": recorded_environment,
                "provenance_sha256": provenance,
            },
        )
        print(
            json.dumps(
                {"status": "launching-once", "run_directory": str(run), "command": command},
                ensure_ascii=False,
            ),
            flush=True,
        )
        observation = observe_process(
            command,
            cwd=DERIVED_UPSTREAM_ROOT,
            environment=child_environment,
            run_directory=run,
            gpu_sampler=_sample_nvidia_smi,
        )
        started = json.loads((run / "started.json").read_text(encoding="utf-8"))
        started["command_sha256"] = command_hash
        write_json_atomic(run / "started.json", started)
        write_json_atomic(run / "source-data-post.json", build_source_inventory(DATA_ROOT))
        if observation["gpu_errors"]:
            raise ValueError("GPU observer errors: " + "; ".join(observation["gpu_errors"]))
        result = parse_full_result(run)
        result["child_pid"] = observation["pid"]
        write_json_atomic(run / "full-result.json", result)
        write_json_atomic(
            run / "completed.json",
            {
                "status": "pass",
                "exit_code": observation["exit_code"],
                "pid": observation["pid"],
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return run, result
    except BaseException as error:
        write_json_atomic(
            run / "failure.json",
            {
                "status": "fail",
                "error": f"{type(error).__name__}: {error}",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "training_retry_performed": False,
            },
        )
        raise


def finalize_existing_run(run_directory: Path) -> dict[str, object]:
    """Recover an exit-zero run from the one authorized observer parser failure."""
    run = run_directory.resolve()
    artifacts_root = FULL_ARTIFACTS_ROOT.resolve()
    if run.parent != artifacts_root or not run.name.startswith("fold2-full-"):
        raise ValueError("finalize-existing run is outside the full-fold artifact root")
    existing_completed = (run / "completed.json").exists()
    if existing_completed:
        completed_before = json.loads(
            (run / "completed.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(completed_before, Mapping)
            or completed_before.get("status") != "pass"
            or completed_before.get("exit_code") != 0
            or completed_before.get("finalized_existing") is not True
            or completed_before.get("observer_recovery") is not True
            or not (run / "observer-recovery.json").is_file()
        ):
            raise ValueError("completed run lacks the authorized observer recovery lineage")
    failure_path = run / "failure.json"
    failure_bytes = failure_path.read_bytes()
    failure = json.loads(failure_bytes.decode("utf-8"))
    if not isinstance(failure, Mapping):
        raise ValueError("failure marker must be an object")
    if (
        failure.get("status") != "fail"
        or failure.get("error") != OBSERVER_TEST_METRIC_FAILURE
        or failure.get("training_retry_performed") is not False
    ):
        raise ValueError("finalize-existing requires the exact observer parser failure")
    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    if exit_code != 0:
        raise ValueError("finalize-existing requires scientific child exit code zero")
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    if not isinstance(started, Mapping):
        raise ValueError("started marker must be an object")
    pid = started.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("started.pid must be a positive integer")
    raw_before = capture_full_raw_manifest(run)
    result = parse_full_result(run)
    if (
        result.get("status") != "pass"
        or result.get("optimizer_steps") != 10_000
        or not isinstance(result.get("test_metrics"), Mapping)
        or "test_AP" not in result["test_metrics"]
    ):
        raise ValueError("finalize-existing reconstructed result is incomplete")
    raw_after = capture_full_raw_manifest(run)
    if raw_before != raw_after:
        raise ValueError("fixed raw evidence changed during offline finalization")
    finalized_utc = datetime.now(timezone.utc).isoformat()
    raw_seal = {
        "status": "pass",
        "sealed_utc": finalized_utc,
        "seal_scope": RAW_SEAL_SCOPE,
        "scientific_child_relaunched": False,
        "before": raw_before,
        "after": raw_after,
    }
    raw_seal_path = run / "raw-evidence-seal.json"
    write_json_atomic(raw_seal_path, raw_seal)
    result["raw_evidence_seal_sha256"] = _sha256(raw_seal_path)
    result_path = run / "full-result.json"
    write_json_atomic(result_path, result)
    recovery = {
        "status": "pass",
        "recovery_type": "observer_parser_finalize_existing",
        "finalized_utc": finalized_utc,
        "previous_error": OBSERVER_TEST_METRIC_FAILURE,
        "failure_sha256": hashlib.sha256(failure_bytes).hexdigest(),
        "full_result_sha256": _sha256(result_path),
        "command_sha256": result.get("command_sha256"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "optimizer_steps": result["optimizer_steps"],
        "test_AP": result["test_metrics"]["test_AP"],
        "previous_failure_preserved": True,
        "scientific_child_relaunched": False,
        "offline_refinalization": existing_completed,
        "retrospective_raw_integrity_claim": False,
        "raw_evidence_seal_sha256": result["raw_evidence_seal_sha256"],
        "raw_evidence_before_equals_after": True,
        "raw_evidence_scope": RAW_SEAL_SCOPE,
    }
    write_json_atomic(run / "observer-recovery.json", recovery)
    write_json_atomic(
        run / "completed.json",
        {
            "status": "pass",
            "exit_code": exit_code,
            "pid": pid,
            "completed_utc": finalized_utc,
            "finalized_existing": True,
            "observer_recovery": True,
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--finalize-existing", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.preflight_only:
            placeholder = FULL_ARTIFACTS_ROOT / "preflight-placeholder"
            placeholder.mkdir(parents=True, exist_ok=True)
            metadata, _ = _preflight(placeholder)
            print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
            return 0
        if arguments.finalize_existing is not None:
            result = finalize_existing_run(arguments.finalize_existing)
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        run, result = _launch_once()
        print(json.dumps({"run_directory": str(run), **result}, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

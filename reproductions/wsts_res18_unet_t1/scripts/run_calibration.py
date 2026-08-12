"""Externally observe the one authorized WSTS fold-2 timing calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

from control import (
    FROZEN_DATA_ROOT,
    validate_environment,
    validate_overrides,
    verify_inventory,
    verify_upstream,
    write_json_atomic,
)


EXPECTED_MAX_STEPS = 500
BATCH_SIZE = 64
WARMUP_LAST_STEP = 49
SELECTED_YEARS = (2018, 2019, 2020, 2021)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAIN_PROGRESS = re.compile(r"\bEpoch\s+(\d+):.*?(\d+)\s*/\s*(\d+)")
_METRIC = re.compile(
    r"\b(?:train_loss|train_f1|val_loss|val_avg_precision|val_f1)"
    r"(?:_step|_epoch)?\s*=\s*"
    r"([+-]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?))",
    re.IGNORECASE,
)
_VALIDATION_PROGRESS = re.compile(r"\bValidation DataLoader\s+\d+:.*?(\d+)\s*/\s*(\d+)")
EXPECTED_CODE_COMMIT = "ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad"
EXPECTED_WEIGHTS_REVISION = "acf70a37394849f4ec8d108a51d6f4325a554d0a"
OFFICIAL_ORIGIN = "https://github.com/slahrichi/WildfireSpreadTS.git"
EXPECTED_RUNTIME = {
    "python": "3.10.4",
    "setuptools": "80.9.0",
    "numpy": "1.23.5",
    "torch": "2.0.0+cu118",
    "torchvision": "0.15.1+cu118",
    "pytorch_lightning": "2.0.1",
}
EXPECTED_PATCH_DELETIONS = [
    "from .UTAELightning import UTAELightning",
    "from .SwinUnetLightning import SwinUnetLightning",
    "from .SwinUnetTempLightning import SwinUnetTempLightning",
    "from .UTAELightningDumb import UTAELightningDumb",
    "from .TransUnetLightning import TransUnetLightning",
    "from .SMPTempModel import SMPTempModel ",
    "from .SegFormerLightning import SegFormerLightning",
]
EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT = 608.4653828020165
SOURCE_YAML_POS_CLASS_WEIGHT = 236.0
REQUIRED_RES18_EXPORTS = [
    "from .BaseModel import BaseModel",
    "from .ConvLSTMLightning import ConvLSTMLightning",
    "from .LogisticRegression import LogisticRegression",
    "from .SMPModel import SMPModel",
]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS"
DERIVED_UPSTREAM_ROOT = REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS-res18-runtime"
RUNTIME_PATCH_PATH = REPRODUCTION_ROOT / "patches" / "res18_import_scope.patch"
ENVIRONMENT_PYTHON = Path(r"D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe")
DATA_ROOT = Path(FROZEN_DATA_ROOT)
ARTIFACTS_ROOT = REPOSITORY_ROOT / "artifacts" / "reproductions" / "wsts-res18-t1"
SMOKE_PATH = ARTIFACTS_ROOT / "fold2-smoke-20260809" / "smoke.json"
WORKER_SMOKE_PATH = (
    ARTIFACTS_ROOT / "fold2-train-val-smoke-workers8-20260809" / "smoke.json"
)
GLOBAL_LAUNCH_LOCK = ARTIFACTS_ROOT / "fold2-calibration-launch.lock.json"
GLOBAL_RECOVERY_LOCK = ARTIFACTS_ROOT / "fold2-calibration-pretraining-recovery.lock.json"
GLOBAL_WORKER_RECOVERY_LOCK = ARTIFACTS_ROOT / "fold2-calibration-worker-recovery.lock.json"
MINIMUM_GPU_FREE_MIB = 20_000
MINIMUM_DISK_FREE_BYTES = 10 * 1024**3


def validate_runtime_safety_patch(patch_path: Path, upstream_root: Path) -> dict[str, object]:
    """Require the one-file, seven-export Res18 import-scope safety patch."""
    patch = patch_path.read_text(encoding="utf-8")
    affected_files = [
        match.group(1)
        for match in re.finditer(r"(?m)^diff --git a/(\S+) b/(\S+)$", patch)
        if match.group(1) == match.group(2)
    ]
    deleted_lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added_lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if affected_files != ["src/models/__init__.py"]:
        raise ValueError("runtime safety patch must affect only src/models/__init__.py")
    if deleted_lines != EXPECTED_PATCH_DELETIONS or added_lines:
        raise ValueError("runtime safety patch must contain exactly seven approved deletions")
    pristine_lines = (
        upstream_root.resolve() / "src" / "models" / "__init__.py"
    ).read_text(encoding="utf-8").splitlines()
    if not all(export in pristine_lines for export in REQUIRED_RES18_EXPORTS):
        raise ValueError("pinned upstream is missing a required Res18 export")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(upstream_root.resolve()),
            "apply",
            "--check",
            "--ignore-space-change",
            str(patch_path.resolve()),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"runtime safety patch does not apply cleanly: {result.stderr.strip()}")
    return {
        "affected_files": affected_files,
        "added_lines": added_lines,
        "classification": "runtime safety import-scope patch; no scientific class/module body touched",
        "deleted_lines": deleted_lines,
        "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "required_exports_unchanged": REQUIRED_RES18_EXPORTS,
    }


def validate_import_only_evidence(exit_code: int, stdout: str, stderr: str) -> None:
    """Require import/help success without any Trainer, optimizer, or progress."""
    combined = stdout + "\n" + stderr
    if exit_code != 0:
        raise ValueError(f"import-only preflight exit code must be 0, got {exit_code}")
    if "Traceback (most recent call last)" in combined or "ModuleNotFoundError" in combined:
        raise ValueError("import-only preflight contains an import failure")
    if re.search(
        r"\bEpoch\s+\d+:.*\d+/\d+|Trainer\.fit|max_steps=500`?\s+reached|"
        r"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=",
        combined,
    ):
        raise ValueError("import-only preflight contains training evidence")


def _json_events(text: str) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            seconds = float(payload["seconds"])
            fragment = payload["text"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid observer event on line {line_number}") from error
        if not math.isfinite(seconds):
            raise ValueError("observer timestamps must be finite")
        if not isinstance(fragment, str):
            raise ValueError("observer event text must be a string")
        if events and seconds < events[-1][0]:
            raise ValueError("observer timestamps regress")
        events.append((seconds, fragment))
    return events


def parse_progress(text: str) -> list[tuple[int, float]]:
    """Extract monotonic global optimizer-step observations from JSONL events."""
    samples: list[tuple[int, float]] = []
    current_epoch: int | None = None
    epoch_offset = 0
    epoch_total: int | None = None
    epoch_position: int | None = None

    for seconds, raw_fragment in _json_events(text):
        fragment = _ANSI_ESCAPE.sub("", raw_fragment)
        match = _TRAIN_PROGRESS.search(fragment)
        if match is None or "Validation DataLoader" in fragment or "Sanity Checking" in fragment:
            continue
        epoch, position, total = (int(value) for value in match.groups())
        if total <= 0 or position < 0 or position > total:
            raise ValueError("invalid training progress values")

        if current_epoch is None:
            current_epoch = epoch
            epoch_total = total
        elif epoch < current_epoch:
            raise ValueError("training epoch regress detected")
        elif epoch == current_epoch:
            if total != epoch_total:
                raise ValueError("training progress denominator changed within an epoch")
            if epoch_position == epoch_total and position == 0:
                # Lightning tears down a completed epoch's tqdm bar by rendering
                # one same-epoch zero row before the next epoch is announced.
                continue
            if epoch_position is not None and position < epoch_position:
                raise ValueError("optimizer progress regress detected")
        else:
            if epoch != current_epoch + 1:
                raise ValueError("training epoch regress or discontinuity detected")
            if epoch_total is None or epoch_position != epoch_total:
                raise ValueError("new epoch observed before the preceding epoch completed")
            epoch_offset += epoch_total
            current_epoch = epoch
            epoch_total = total
            epoch_position = None

        global_step = epoch_offset + position
        epoch_position = position
        if samples and global_step < samples[-1][0]:
            raise ValueError("optimizer step regress detected")
        if samples and global_step == samples[-1][0]:
            continue
        samples.append((global_step, seconds))
    return samples


def parse_epoch_boundaries(text: str) -> list[tuple[int, int, float, int]]:
    """Record the first completed optimizer step and global boundary per epoch."""
    boundaries: list[tuple[int, int, float, int]] = []
    totals: list[int] = []
    seen_epochs: set[int] = set()
    for seconds, raw_fragment in _json_events(text):
        fragment = _ANSI_ESCAPE.sub("", raw_fragment)
        if "Validation DataLoader" in fragment or "Sanity Checking" in fragment:
            continue
        match = _TRAIN_PROGRESS.search(fragment)
        if match is None:
            continue
        epoch, position, total = map(int, match.groups())
        if position != 1 or epoch in seen_epochs:
            continue
        if epoch != len(boundaries) or total <= 0:
            raise ValueError("epoch boundary sequence is discontinuous")
        global_step = sum(totals) + 1
        boundaries.append((epoch, global_step, seconds, total))
        totals.append(total)
        seen_epochs.add(epoch)
    if len(boundaries) < 2:
        raise ValueError("at least two epoch first-step boundaries are required")
    return boundaries


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_timing(
    samples: Sequence[tuple[int, float]],
    wall_seconds: float,
    startup_seconds: float,
    validation_seconds: float,
) -> dict[str, float]:
    """Summarize post-warm-up optimizer timing and a 10,000-step estimate."""
    for name, value in (
        ("wall_seconds", wall_seconds),
        ("startup_seconds", startup_seconds),
        ("validation_seconds", validation_seconds),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    if len(samples) < 2:
        raise ValueError("at least two progress samples are required")

    per_step_seconds: list[float] = []
    previous_step, previous_seconds = samples[0]
    if type(previous_step) is not int or previous_step < 0 or not math.isfinite(previous_seconds):
        raise ValueError("progress values must be finite and non-negative")
    for step, seconds in samples[1:]:
        if type(step) is not int or step <= previous_step:
            raise ValueError("optimizer steps regress or repeat")
        if not math.isfinite(seconds):
            raise ValueError("progress timestamps must be finite")
        if seconds <= previous_seconds:
            raise ValueError("progress timestamps regress or repeat")
        if previous_step >= WARMUP_LAST_STEP:
            per_step_seconds.append((seconds - previous_seconds) / (step - previous_step))
        previous_step, previous_seconds = step, seconds
    if not per_step_seconds:
        raise ValueError("no post-warm-up timing samples are available")

    median_step_seconds = statistics.median(per_step_seconds)
    p25_step_seconds = _percentile(per_step_seconds, 0.25)
    p75_step_seconds = _percentile(per_step_seconds, 0.75)
    return {
        "wall_seconds": float(wall_seconds),
        "startup_seconds": float(startup_seconds),
        "validation_seconds": float(validation_seconds),
        "observed_progress_seconds": float(samples[-1][1] - samples[0][1]),
        "post_warmup_interval_count": float(len(per_step_seconds)),
        "median_step_seconds": float(median_step_seconds),
        "p25_step_seconds": float(p25_step_seconds),
        "p75_step_seconds": float(p75_step_seconds),
        "instantaneous_samples_per_second": float(BATCH_SIZE / median_step_seconds),
        "compute_only_10000_seconds": float(10_000 * median_step_seconds),
    }


def summarize_epoch_aware_timing(
    *,
    epoch_boundaries: Sequence[tuple[int, int, float, int]],
    startup_seconds: float,
    median_step_seconds: float,
    p25_step_seconds: float,
    p75_step_seconds: float,
    wall_seconds: float,
) -> dict[str, object]:
    """Project 10k steps from observed full loader/validation epoch cycles."""
    if len(epoch_boundaries) < 2:
        raise ValueError("at least two epoch boundaries are required")
    epoch_size = epoch_boundaries[0][3]
    if epoch_size <= 0 or any(boundary[3] != epoch_size for boundary in epoch_boundaries):
        raise ValueError("epoch boundary denominators must be positive and stable")
    cycles: list[float] = []
    for previous, current in zip(epoch_boundaries, epoch_boundaries[1:]):
        if current[0] != previous[0] + 1:
            raise ValueError("epoch boundaries must be consecutive")
        if current[1] - previous[1] != epoch_size:
            raise ValueError("global epoch boundary step distance changed")
        cycle = current[2] - previous[2]
        if not math.isfinite(cycle) or cycle <= 0:
            raise ValueError("epoch cycle time must be finite and positive")
        cycles.append(cycle)
    median_cycle = statistics.median(cycles)
    p25_cycle = _percentile(cycles, 0.25)
    p75_cycle = _percentile(cycles, 0.75)
    full_cycles, partial_steps = divmod(10_000, epoch_size)
    central = startup_seconds + full_cycles * median_cycle + partial_steps * median_step_seconds
    lower = startup_seconds + full_cycles * p25_cycle + partial_steps * p25_step_seconds
    upper = startup_seconds + full_cycles * p75_cycle + partial_steps * p75_step_seconds
    return {
        "epoch_size_steps": epoch_size,
        "epoch_first_step_boundaries": [list(boundary) for boundary in epoch_boundaries],
        "epoch_cycle_seconds": cycles,
        "epoch_cycle_count": len(cycles),
        "median_epoch_cycle_seconds": float(median_cycle),
        "p25_epoch_cycle_seconds": float(p25_cycle),
        "p75_epoch_cycle_seconds": float(p75_cycle),
        "projected_full_epoch_cycles": full_cycles,
        "projected_partial_steps": partial_steps,
        "epoch_aware_10000_central_seconds": float(central),
        "epoch_aware_10000_lower_seconds": float(lower),
        "epoch_aware_10000_upper_seconds": float(upper),
        "naive_wall_linear_10000_seconds": float(wall_seconds / 500 * 10_000),
        "end_to_end_samples_per_second": float(500 * BATCH_SIZE / wall_seconds),
    }


def measure_wall_seconds_from_markers(started_utc: str, exit_mtime_ns: int) -> float:
    """Recover child wall time when raw capture succeeded but postprocessing failed."""
    try:
        started_seconds = datetime.fromisoformat(started_utc).timestamp()
    except (TypeError, ValueError) as error:
        raise ValueError("started marker must contain an ISO-8601 timestamp") from error
    wall_seconds = exit_mtime_ns / 1_000_000_000 - started_seconds
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("exit marker must be later than the child start marker")
    return float(wall_seconds)


def build_official_command(
    *,
    python_executable: Path,
    entrypoint: Path,
    upstream_root: Path,
    data_root: Path,
    run_directory: Path,
    num_workers: int = 64,
) -> list[str]:
    """Build the exact official three-config command plus local allowlist."""
    if num_workers not in (64, 8):
        raise ValueError("calibration command num_workers must be the baseline 64 or authorized 8")
    upstream = upstream_root.resolve()
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
        f"--data.num_workers={num_workers}",
        "--trainer.max_steps=500",
        f"--trainer.default_root_dir={run_directory.resolve()}",
        "--do_test=false",
    ]


def validate_effective_command(
    command: Sequence[str], expected_command: Sequence[str]
) -> None:
    """Reject any action or scientific difference from the exact allowlist."""
    if list(command) != list(expected_command):
        raise ValueError("effective command differs from the exact allowlisted command")


def validate_effective_runtime_config(config_path: Path) -> dict[str, object]:
    """Require the official fold-aware positive-class override in saved config."""
    text = config_path.read_text(encoding="utf-8")
    matches = re.findall(r"(?m)^\s+pos_class_weight:\s*(\S+)\s*$", text)
    if len(matches) != 1:
        raise ValueError("effective runtime config must contain one positive-class weight")
    try:
        effective = float(matches[0])
    except ValueError as error:
        raise ValueError("effective positive-class weight must be numeric") from error
    if effective != EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT:
        raise ValueError(
            "effective positive-class weight does not match the official fold-2 dynamic override"
        )
    return {
        "effective_pos_class_weight": effective,
        "source_yaml_pos_class_weight": SOURCE_YAML_POS_CLASS_WEIGHT,
        "official_dynamic_override": True,
    }


def validate_success_evidence(
    timestamped_events: str,
    combined_output: str,
    exit_code: int,
    peak_allocated_bytes: float,
) -> list[tuple[int, float]]:
    """Require the exact successful, training-only 500-step observation."""
    if exit_code != 0:
        raise ValueError(f"training exit code must be 0, got {exit_code}")
    if not math.isfinite(peak_allocated_bytes) or peak_allocated_bytes <= 0:
        raise ValueError("peak allocated memory must be finite and positive")
    lowered = combined_output.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        raise ValueError("OOM evidence found in training output")
    if re.search(r"\b(?:testing|predicting) dataloader\b|runningstage\.(?:testing|predicting)", lowered):
        raise ValueError("test/predict invocation evidence found")
    if re.search(r"max_steps=500`?\s+reached", combined_output) is None:
        raise ValueError("exact max_steps=500 stop evidence is missing")

    samples = parse_progress(timestamped_events)
    if not samples or samples[-1][0] != EXPECTED_MAX_STEPS:
        observed = samples[-1][0] if samples else None
        raise ValueError(f"progress must establish exactly 500 optimizer steps; got {observed}")
    metric_values = [float(match) for match in _METRIC.findall(combined_output)]
    if not metric_values or not all(math.isfinite(value) for value in metric_values):
        raise ValueError("finite final loss/metric evidence is required")
    return samples


def build_source_inventory(data_root: Path) -> dict[str, object]:
    """Record selected source file paths, sizes, and nanosecond mtimes."""
    root = data_root.resolve()
    files: list[dict[str, object]] = []
    for year in SELECTED_YEARS:
        year_root = root / str(year)
        if not year_root.is_dir():
            raise ValueError(f"source year directory is missing: {year_root}")
        for path in sorted(year_root.glob("*.hdf5"), key=lambda candidate: candidate.name):
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "files": files,
    }


def assert_source_inventories_identical(
    before: Mapping[str, object], after: Mapping[str, object]
) -> None:
    """Reject any selected source-data size, mtime, or membership change."""
    if dict(before) != dict(after):
        raise ValueError("source data changed during calibration")


def acquire_launch_lock(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically reserve the sole calibration launch and never overwrite it."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"calibration launch lock already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _pid_is_alive(pid: int) -> bool:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result.returncode == 0


def validate_pretraining_failure_recovery(
    failed_run: Path, launch_lock_path: Path
) -> dict[str, object]:
    """Prove an authorized recovery follows a child that never reached training."""
    run_root = failed_run.resolve()
    launch = json.loads(launch_lock_path.read_text(encoding="utf-8"))
    started = json.loads((run_root / "started.json").read_text(encoding="utf-8"))
    failure = json.loads((run_root / "failure.json").read_text(encoding="utf-8"))
    if Path(launch.get("run_directory", "")).resolve() != run_root:
        raise ValueError("recovery failed run is not linked by the preserved global launch lock")
    if launch.get("attempt") != 1 or launch.get("command") != started.get("command"):
        raise ValueError("recovery launch provenance does not match the failed child")
    exit_code = int((run_root / "exit-code.txt").read_text(encoding="utf-8").strip())
    if exit_code != 1:
        raise ValueError(f"recovery requires the preserved pretraining exit code 1, got {exit_code}")
    pid = started.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("failed child PID is invalid")
    if _pid_is_alive(pid):
        raise ValueError(f"failed child PID {pid} is still alive and must be monitored, not replaced")
    stdout = (run_root / "stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr = (run_root / "stderr.log").read_text(encoding="utf-8", errors="replace")
    events = (run_root / "stream-events.jsonl").read_text(encoding="utf-8")
    combined = stdout + "\n" + stderr
    if stdout:
        raise ValueError("training progress or optimizer output exists in failed stdout")
    if "ModuleNotFoundError" not in stderr or "No module named 'src'" not in stderr:
        raise ValueError("failed child does not contain the authorized src import failure")
    if re.search(r"\bTrainer(?:\.fit)?\b|\boptimizer\b|\bEpoch\s+\d+:.*\d+/\d+", combined):
        raise ValueError("training progress, Trainer, or optimizer evidence exists in failed run")
    if parse_progress(events):
        raise ValueError("optimizer progress exists in failed observer events")
    if "max_steps=500" in combined or "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=" in combined:
        raise ValueError("failed run advanced beyond the authorized pretraining failure boundary")
    if failure.get("status") != "fail" or failure.get("training_retry_performed") is not False:
        raise ValueError("failed run does not preserve no-retry evidence")
    return {
        "authorized_reason": "entrypoint import-path integration failure before model/datamodule/Trainer/optimizer",
        "failed_command_sha256": launch.get("command_sha256"),
        "failed_child_pid": pid,
        "failed_exit_code": exit_code,
        "failed_run_directory": str(run_root),
        "failed_stderr_sha256": _sha256(run_root / "stderr.log"),
        "optimizer_steps_observed": 0,
        "pretraining_failure_recovery": 1,
        "prior_training_retry_performed": False,
    }


def validate_worker_failure_recovery(
    failed_run: Path,
    recovery_lock_path: Path,
    original_launch_lock_path: Path | None = None,
) -> dict[str, object]:
    """Prove the workers=64 retry failed in sanity validation before optimization."""
    run_root = failed_run.resolve()
    recovery = json.loads(recovery_lock_path.read_text(encoding="utf-8"))
    started = json.loads((run_root / "started.json").read_text(encoding="utf-8"))
    failure = json.loads((run_root / "failure.json").read_text(encoding="utf-8"))
    if Path(recovery.get("run_directory", "")).resolve() != run_root:
        raise ValueError("worker recovery failed run is not linked by the preserved recovery lock")
    if recovery.get("attempt") != 2 or recovery.get("pretraining_failure_recovery") != 1:
        raise ValueError("worker recovery lineage does not contain the authorized second attempt")
    if recovery.get("command") != started.get("command"):
        raise ValueError("worker recovery launch provenance does not match the failed child")
    command = started.get("command")
    if not isinstance(command, list) or "--data.num_workers=64" not in command:
        raise ValueError("worker recovery requires the preserved workers=64 command")

    lineage_run_text = recovery.get("failed_run_directory")
    if not isinstance(lineage_run_text, str):
        raise ValueError("worker recovery is missing the import-failure lineage")
    lineage_run = Path(lineage_run_text).resolve()
    if original_launch_lock_path is not None:
        validate_pretraining_failure_recovery(lineage_run, original_launch_lock_path)

    exit_code = int((run_root / "exit-code.txt").read_text(encoding="utf-8").strip())
    if exit_code == 0:
        raise ValueError("worker recovery requires a nonzero workers=64 child exit")
    pid = started.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("failed workers=64 child PID is invalid")
    if _pid_is_alive(pid):
        raise ValueError(f"failed child PID {pid} is still alive and must be monitored, not replaced")

    stdout = (run_root / "stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr = (run_root / "stderr.log").read_text(encoding="utf-8", errors="replace")
    events = (run_root / "stream-events.jsonl").read_text(encoding="utf-8")
    combined = stdout + "\n" + stderr
    if "Sanity Checking" not in stdout:
        raise ValueError("workers=64 failure did not reach validation sanity checking")
    if "Caught RuntimeError in DataLoader worker process 0" not in stderr:
        raise ValueError("workers=64 failure is not the exact validation DataLoader worker-0 error")
    if "DefaultCPUAllocator: not enough memory" not in stderr:
        raise ValueError("workers=64 failure is not the exact CPU allocator memory error")
    if parse_progress(events) or re.search(r"\bEpoch\s+\d+:.*\d+\s*/\s*\d+", combined):
        raise ValueError("optimizer training progress exists in failed workers=64 run")
    if re.search(r"max_steps=500`?\s+reached", combined):
        raise ValueError("workers=64 failure reached the configured optimizer-step stop")
    if "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=" in combined:
        raise ValueError("workers=64 failure reached the post-training allocation sentinel")
    if failure.get("status") != "fail" or failure.get("training_retry_performed") is not False:
        raise ValueError("workers=64 run does not preserve no-retry evidence")
    return {
        "authorized_reason": (
            "Windows workers=64 CPU allocator exhaustion in validation sanity checking; "
            "workers=8 changes loader/platform compatibility only"
        ),
        "failed_child_pid": pid,
        "failed_command_sha256": recovery.get("command_sha256"),
        "failed_exit_code": exit_code,
        "failed_run_directory": str(run_root),
        "failed_stderr_sha256": _sha256(run_root / "stderr.log"),
        "lineage_import_failed_run": str(lineage_run),
        "optimizer_steps_observed": 0,
        "prior_training_retry_performed": False,
        "worker_recovery": 1,
    }


def create_unique_run_directory(artifacts_root: Path) -> Path:
    """Create a collision-resistant ignored run directory."""
    root = artifacts_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"fold2-calibration-{timestamp}-{uuid.uuid4().hex[:8]}"
    target.mkdir()
    return target


def validate_smoke_evidence(smoke: Mapping[str, object]) -> None:
    """Require the exact passing Task-2 fold-2 loader gate."""
    expected = {
        "status": "pass",
        "purpose": "fold-2 real-data loader smoke only; no training performed",
        "upstream_commit": EXPECTED_CODE_COMMIT,
        "weights_revision": EXPECTED_WEIGHTS_REVISION,
        "fold_mapping": {
            "train": [2018, 2020],
            "validation": [2019],
            "test": [2021],
        },
        "inventory": {
            "2018": 176,
            "2019": 74,
            "2020": 201,
            "2021": 156,
            "total": 607,
        },
        "batch_size": 64,
        "num_workers": 64,
        "temporal_steps": 1,
        "model_channels": 40,
        "crop_side_length": 128,
        "test_loader_called": False,
        "test_sample_loaded": False,
        "training_started": False,
    }
    mismatches = [key for key, expected_value in expected.items() if smoke.get(key) != expected_value]
    if mismatches:
        raise ValueError("smoke evidence mismatch: " + ", ".join(mismatches))


def validate_train_validation_smoke_evidence(smoke: Mapping[str, object]) -> None:
    """Require workers=8 to load one exact train batch and one validation batch."""
    boundary = {
        "loader_input_shape": [64, 1, 40, 128, 128],
        "loader_target_shape": [64, 128, 128],
        "model_input_shape": [64, 40, 128, 128],
        "model_target_shape": [64, 1, 128, 128],
        "temporal_steps": 1,
        "model_channels": 40,
        "crop_side_length": 128,
        "inputs_finite": True,
        "targets_binary": True,
        "next_day_target_distinct": True,
    }
    expected = {
        "status": "pass",
        "purpose": "fold-2 real-data train+validation loader smoke only; no training performed",
        "upstream_commit": EXPECTED_CODE_COMMIT,
        "weights_revision": EXPECTED_WEIGHTS_REVISION,
        "fold_mapping": {
            "train": [2018, 2020],
            "validation": [2019],
            "test": [2021],
        },
        "inventory": {
            "2018": 176,
            "2019": 74,
            "2020": 201,
            "2021": 156,
            "total": 607,
        },
        "batch_size": 64,
        "num_workers": 8,
        "train": boundary,
        "validation": boundary,
        "validation_loader_called": True,
        "validation_sample_loaded": True,
        "test_loader_called": False,
        "test_sample_loaded": False,
        "training_started": False,
    }
    mismatches = [key for key, value in expected.items() if smoke.get(key) != value]
    ram = smoke.get("ram")
    if not isinstance(ram, Mapping):
        mismatches.append("ram")
    else:
        try:
            total = int(ram["ram_total_bytes"])
            before = int(ram["ram_available_before_bytes"])
            after = int(ram["ram_available_after_bytes"])
            minimum = int(ram["ram_minimum_available_bytes"])
            peak_used = int(ram["ram_peak_used_bytes"])
            sample_count = int(ram["ram_sample_count"])
            ram_valid = (
                total > 0
                and sample_count >= 3
                and 0 <= minimum <= before <= total
                and 0 <= minimum <= after <= total
                and peak_used == total - minimum
            )
        except (KeyError, TypeError, ValueError):
            ram_valid = False
        if not ram_valid:
            mismatches.append("ram")
    if mismatches:
        raise ValueError("train-validation smoke evidence mismatch: " + ", ".join(mismatches))


def measure_validation_seconds(timestamped_events: str) -> tuple[float, bool]:
    """Measure complete validation progress intervals, excluding sanity checks."""
    total_seconds = 0.0
    interval_start: float | None = None
    saw_validation = False
    incomplete = False
    for seconds, raw_fragment in _json_events(timestamped_events):
        fragment = _ANSI_ESCAPE.sub("", raw_fragment)
        if "Sanity Checking" in fragment:
            continue
        match = _VALIDATION_PROGRESS.search(fragment)
        if match is None:
            continue
        saw_validation = True
        position, total = (int(value) for value in match.groups())
        if total <= 0 or position < 0 or position > total:
            raise ValueError("invalid validation progress values")
        if position == 0:
            if interval_start is not None:
                incomplete = True
            interval_start = seconds
        elif interval_start is None:
            incomplete = True
        if position == total:
            if interval_start is None:
                incomplete = True
            else:
                total_seconds += seconds - interval_start
                interval_start = None
    if interval_start is not None:
        incomplete = True
    identifiable = saw_validation and not incomplete and total_seconds >= 0
    return float(total_seconds if identifiable else 0.0), identifiable


def parse_peak_allocated(output: str) -> int:
    """Extract exactly one positive peak-allocation sentinel."""
    matches = re.findall(r"(?m)^WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\S+)\s*$", output)
    if len(matches) != 1:
        raise ValueError("expected exactly one peak-allocation sentinel")
    try:
        value = int(matches[0])
    except ValueError as error:
        raise ValueError("peak-allocation sentinel must be a positive integer") from error
    if value <= 0:
        raise ValueError("peak-allocation sentinel must be a positive integer")
    return value


def summarize_gpu_samples(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """Summarize once-per-second GPU observer rows."""
    if not rows:
        raise ValueError("at least one GPU sample is required")
    used = [float(row["memory_used_mib"]) for row in rows]
    utilization = [float(row["utilization_gpu_percent"]) for row in rows]
    child = [float(row["child_memory_mib"]) for row in rows]
    observer_seconds = [float(row["observer_seconds"]) for row in rows]
    values = [*used, *utilization, *child, *observer_seconds]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("GPU samples must be finite and non-negative")
    if len(observer_seconds) < 2:
        raise ValueError("at least two GPU timestamps are required for cadence evidence")
    intervals = [
        current - previous
        for previous, current in zip(observer_seconds, observer_seconds[1:])
    ]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("GPU observer timestamps must strictly increase")
    sampling_span = observer_seconds[-1] - observer_seconds[0]
    child_available = any(value > 0 for value in child)
    return {
        "gpu_sample_count": float(len(rows)),
        "gpu_interval_count": len(intervals),
        "gpu_interval_mean_seconds": float(statistics.mean(intervals)),
        "gpu_interval_median_seconds": float(statistics.median(intervals)),
        "gpu_observed_effective_hz": float(len(intervals) / sampling_span),
        "gpu_peak_is_observed_sample_max": True,
        "gpu_peak_may_miss_between_sample_transients": True,
        "peak_gpu_used_mib": float(max(used)),
        "peak_child_process_mib": float(max(child)) if child_available else None,
        "child_process_memory_available": child_available,
        "gpu_utilization_min_percent": float(min(utilization)),
        "gpu_utilization_median_percent": float(statistics.median(utilization)),
        "gpu_utilization_max_percent": float(max(utilization)),
    }


def next_gpu_sample_deadline(
    previous_deadline: float, now: float, interval_seconds: float = 1.0
) -> float:
    """Advance an absolute monotonic sampling schedule without accumulating query time."""
    if not all(math.isfinite(value) for value in (previous_deadline, now, interval_seconds)):
        raise ValueError("GPU sampling schedule values must be finite")
    if interval_seconds <= 0:
        raise ValueError("GPU sampling interval must be positive")
    deadline = previous_deadline + interval_seconds
    if deadline <= now:
        missed = math.floor((now - deadline) / interval_seconds) + 1
        deadline += missed * interval_seconds
    return float(deadline)


GpuSampler = Callable[[int, float], Mapping[str, object] | None]


def _pump_stream(
    stream_name: str,
    source: BinaryIO,
    raw_path: Path,
    event_handle: object,
    event_lock: threading.Lock,
    started_at: float,
) -> None:
    buffer = b""
    with raw_path.open("wb", buffering=0) as raw_handle:
        while True:
            chunk = os.read(source.fileno(), 4096)
            if not chunk:
                break
            raw_handle.write(chunk)
            buffer += chunk
            while True:
                positions = [position for position in (buffer.find(b"\r"), buffer.find(b"\n")) if position >= 0]
                if not positions:
                    break
                end = min(positions) + 1
                fragment, buffer = buffer[:end], buffer[end:]
                with event_lock:
                    payload = {
                        "seconds": time.perf_counter() - started_at,
                        "stream": stream_name,
                        "text": fragment.decode("utf-8", errors="replace"),
                    }
                    event_handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    event_handle.flush()
        if buffer:
            with event_lock:
                payload = {
                    "seconds": time.perf_counter() - started_at,
                    "stream": stream_name,
                    "text": buffer.decode("utf-8", errors="replace"),
                }
                event_handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                event_handle.flush()


def observe_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    run_directory: Path,
    gpu_sampler: GpuSampler | None,
) -> dict[str, object]:
    """Launch once, externally preserve/timestamp streams, and never retry."""
    run_root = run_directory.resolve()
    start_perf = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        list(command),
        cwd=cwd.resolve(),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    write_json_atomic(
        run_root / "started.json",
        {"command": list(command), "pid": process.pid, "started_utc": start_utc},
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("child process pipes were not created")

    event_lock = threading.Lock()
    stop_gpu = threading.Event()
    gpu_rows: list[dict[str, object]] = []
    gpu_errors: list[str] = []
    gpu_headers = [
        "observer_seconds",
        "timestamp_utc",
        "gpu_index",
        "gpu_name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_gpu_percent",
        "power_draw_w",
        "child_memory_mib",
    ]

    def sample_gpu() -> None:
        if gpu_sampler is None:
            return
        deadline = start_perf
        while True:
            elapsed = time.perf_counter() - start_perf
            try:
                sample = gpu_sampler(process.pid, elapsed)
                if sample is not None:
                    gpu_rows.append(dict(sample))
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                gpu_errors.append(f"{type(error).__name__}: {error}")
            now = time.perf_counter()
            deadline = next_gpu_sample_deadline(deadline, now)
            if stop_gpu.wait(max(0.0, deadline - time.perf_counter())):
                break

    with (run_root / "stream-events.jsonl").open("w", encoding="utf-8", newline="\n") as events:
        stdout_thread = threading.Thread(
            target=_pump_stream,
            args=("stdout", process.stdout, run_root / "stdout.log", events, event_lock, start_perf),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_pump_stream,
            args=("stderr", process.stderr, run_root / "stderr.log", events, event_lock, start_perf),
            daemon=True,
        )
        gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        gpu_thread.start()
        exit_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        stop_gpu.set()
        gpu_thread.join()

    wall_seconds = time.perf_counter() - start_perf
    (run_root / "exit-code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    if gpu_sampler is not None:
        import csv

        with (run_root / "gpu.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=gpu_headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(gpu_rows)
    return {
        "exit_code": exit_code,
        "pid": process.pid,
        "started_utc": start_utc,
        "wall_seconds": wall_seconds,
        "gpu_rows": gpu_rows,
        "gpu_errors": gpu_errors,
    }


def _run_checked(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise ValueError(f"preflight command failed ({command[0]}): {detail}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_derived_runtime() -> dict[str, object]:
    patch_metadata = validate_runtime_safety_patch(RUNTIME_PATCH_PATH, UPSTREAM_ROOT)
    original_commit = verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
    if not DERIVED_UPSTREAM_ROOT.exists():
        DERIVED_UPSTREAM_ROOT.parent.mkdir(parents=True, exist_ok=True)
        _run_checked(
            [
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(UPSTREAM_ROOT.resolve()),
                str(DERIVED_UPSTREAM_ROOT.resolve()),
            ]
        )
        _run_checked(
            [
                "git",
                "-C",
                str(DERIVED_UPSTREAM_ROOT.resolve()),
                "remote",
                "set-url",
                "origin",
                OFFICIAL_ORIGIN,
            ]
        )
        _run_checked(
            [
                "git",
                "-C",
                str(DERIVED_UPSTREAM_ROOT.resolve()),
                "checkout",
                "--detach",
                EXPECTED_CODE_COMMIT,
            ]
        )
        _run_checked(
            [
                "git",
                "-C",
                str(DERIVED_UPSTREAM_ROOT.resolve()),
                "apply",
                "--ignore-space-change",
                str(RUNTIME_PATCH_PATH.resolve()),
            ]
        )

    derived_commit = _run_checked(
        ["git", "-C", str(DERIVED_UPSTREAM_ROOT.resolve()), "rev-parse", "HEAD"]
    ).stdout.strip()
    if derived_commit != EXPECTED_CODE_COMMIT:
        raise ValueError(f"derived runtime commit mismatch: {derived_commit}")
    derived_origin = _run_checked(
        ["git", "-C", str(DERIVED_UPSTREAM_ROOT.resolve()), "remote", "get-url", "origin"]
    ).stdout.strip()
    if derived_origin.rstrip("/") != OFFICIAL_ORIGIN.rstrip("/"):
        raise ValueError(f"derived runtime origin mismatch: {derived_origin}")
    status_lines = _run_checked(
        ["git", "-C", str(DERIVED_UPSTREAM_ROOT.resolve()), "status", "--porcelain"]
    ).stdout.splitlines()
    if status_lines != [" M src/models/__init__.py"]:
        raise ValueError(f"derived runtime has unexpected changes: {status_lines}")
    _run_checked(
        ["git", "-C", str(DERIVED_UPSTREAM_ROOT.resolve()), "diff", "--check"]
    )
    pristine_lines = (
        UPSTREAM_ROOT / "src" / "models" / "__init__.py"
    ).read_text(encoding="utf-8").splitlines()
    expected_lines = [line for line in pristine_lines if line not in EXPECTED_PATCH_DELETIONS]
    derived_lines = (
        DERIVED_UPSTREAM_ROOT / "src" / "models" / "__init__.py"
    ).read_text(encoding="utf-8").splitlines()
    if derived_lines != expected_lines or derived_lines != REQUIRED_RES18_EXPORTS:
        raise ValueError("derived runtime diff is not the exact approved seven-export deletion")
    git_diff = _run_checked(
        [
            "git",
            "-C",
            str(DERIVED_UPSTREAM_ROOT.resolve()),
            "diff",
            "--no-ext-diff",
            "--",
            "src/models/__init__.py",
        ]
    ).stdout
    verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
    return {
        **patch_metadata,
        "base_commit": original_commit,
        "derived_origin": derived_origin,
        "derived_root": str(DERIVED_UPSTREAM_ROOT.resolve()),
        "git_diff": git_diff,
        "git_diff_exact_match": True,
        "git_diff_sha256": hashlib.sha256(git_diff.encode("utf-8")).hexdigest(),
        "original_checkout_clean": True,
        "scientific_code_touched": False,
    }


def _official_import_only_preflight(run_directory: Path) -> dict[str, object]:
    cache_root = run_directory.resolve() / "import-only-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
        }
    )
    command = [
        str(ENVIRONMENT_PYTHON.resolve()),
        str(Path(__file__).with_name("official_entrypoint.py").resolve()),
        "--upstream-root",
        str(DERIVED_UPSTREAM_ROOT.resolve()),
        "--help",
    ]
    result = subprocess.run(
        command,
        cwd=DERIVED_UPSTREAM_ROOT.resolve(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    (run_directory / "import-only-stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_directory / "import-only-stderr.log").write_text(result.stderr, encoding="utf-8")
    (run_directory / "import-only-exit-code.txt").write_text(
        f"{result.returncode}\n", encoding="utf-8"
    )
    validate_import_only_evidence(result.returncode, result.stdout, result.stderr)
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        "training_started": False,
    }


def _runtime_preflight() -> dict[str, object]:
    code = (
        "import json,sys,setuptools,numpy,torch,torchvision,pytorch_lightning;"
        "print(json.dumps({"
        "'python':'.'.join(map(str,sys.version_info[:3])),"
        "'python_executable':sys.executable,'python_prefix':sys.prefix,"
        "'setuptools':setuptools.__version__,'numpy':numpy.__version__,"
        "'torch':torch.__version__,'torchvision':torchvision.__version__,"
        "'pytorch_lightning':pytorch_lightning.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count(),"
        "'cuda_device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'torch_cuda':torch.version.cuda,'torch_hub_dir':torch.hub.get_dir()}))"
    )
    result = _run_checked([str(ENVIRONMENT_PYTHON), "-c", code])
    try:
        runtime = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("runtime preflight did not return JSON") from error
    mismatches = [key for key, value in EXPECTED_RUNTIME.items() if runtime.get(key) != value]
    if mismatches:
        raise ValueError("runtime version mismatch: " + ", ".join(mismatches))
    if Path(runtime["python_executable"]).resolve() != ENVIRONMENT_PYTHON.resolve():
        raise ValueError("runtime Python executable is not the fixed environment")
    if Path(runtime["python_prefix"]).resolve() != ENVIRONMENT_PYTHON.parent.resolve():
        raise ValueError("runtime sys.prefix is not the fixed environment")
    if runtime.get("cuda_available") is not True or runtime.get("cuda_device_count", 0) < 1:
        raise ValueError("fixed runtime cannot access CUDA")
    if "RTX 3090" not in str(runtime.get("cuda_device_name")):
        raise ValueError("fixed runtime CUDA device is not the RTX 3090")
    torch_hub_dir = Path(runtime["torch_hub_dir"])
    encoder_weight = torch_hub_dir / "checkpoints" / "resnet18-f37072fd.pth"
    if not encoder_weight.is_file():
        raise ValueError(f"cached ResNet-18 encoder weights are missing: {encoder_weight}")
    runtime["encoder_weight"] = str(encoder_weight.resolve())
    runtime["encoder_weight_sha256"] = _sha256(encoder_weight)
    return runtime


def _gpu_preflight(
    *, minimum_free_mib: int = MINIMUM_GPU_FREE_MIB
) -> dict[str, object]:
    result = _run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"unexpected nvidia-smi preflight row: {line}")
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": float(fields[2]),
                "memory_free_mib": float(fields[3]),
                "utilization_gpu_percent": float(fields[4]),
                "driver_version": fields[5],
            }
        )
    matching = [device for device in devices if "RTX 3090" in str(device["name"])]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one RTX 3090, found {len(matching)}")
    selected = matching[0]
    if float(selected["memory_free_mib"]) < minimum_free_mib:
        raise ValueError(
            f"RTX 3090 has only {selected['memory_free_mib']} MiB free; "
            f"requires at least {minimum_free_mib} MiB"
        )
    return selected


def _perform_preflight(
    run_directory: Path,
    *,
    allow_preserved_launch_lock: bool = False,
    num_workers: int = 64,
    smoke_path: Path = SMOKE_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    if GLOBAL_LAUNCH_LOCK.exists() and not allow_preserved_launch_lock:
        raise FileExistsError(f"calibration launch lock already exists: {GLOBAL_LAUNCH_LOCK}")
    if not ENVIRONMENT_PYTHON.is_file():
        raise ValueError(f"fixed environment Python is missing: {ENVIRONMENT_PYTHON}")
    lock = json.loads((REPRODUCTION_ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    if lock["code"]["commit"] != EXPECTED_CODE_COMMIT:
        raise ValueError("code commit lock mismatch")
    if lock["weights"]["revision"] != EXPECTED_WEIGHTS_REVISION:
        raise ValueError("weights revision lock mismatch")

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if num_workers == 64:
        validate_smoke_evidence(smoke)
    elif num_workers == 8:
        validate_train_validation_smoke_evidence(smoke)
    else:
        raise ValueError("preflight num_workers must be the baseline 64 or authorized 8")
    verify_inventory(DATA_ROOT, selected_years=SELECTED_YEARS)
    source_inventory = build_source_inventory(DATA_ROOT)
    if source_inventory["file_count"] != 607:
        raise ValueError("selected source inventory must contain exactly 607 files")
    actual_commit = verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
    origin = _run_checked(["git", "-C", str(UPSTREAM_ROOT), "remote", "get-url", "origin"]).stdout.strip()
    if origin.rstrip("/") != OFFICIAL_ORIGIN.rstrip("/"):
        raise ValueError(f"upstream origin mismatch: {origin}")
    runtime_patch = _prepare_derived_runtime()

    overrides = {
        "data.data_fold_id": 2,
        "data.features_to_keep": None,
        "data.n_leading_observations": 1,
        "data.remove_duplicate_features": True,
        "trainer.max_steps": 500,
        "do_test": False,
    }
    environment_controls = {
        "data.data_dir": str(DATA_ROOT),
        "trainer.default_root_dir": str(run_directory.resolve()),
        "data.num_workers": num_workers,
        "WANDB_MODE": "disabled",
        "timing_logging": True,
        "progress_logging": True,
    }
    validate_overrides(overrides)
    validate_environment(environment_controls, ARTIFACTS_ROOT)

    runtime = _runtime_preflight()
    gpu = _gpu_preflight()
    import_only = _official_import_only_preflight(run_directory)
    disk = shutil.disk_usage(run_directory.resolve())
    if disk.free < MINIMUM_DISK_FREE_BYTES:
        raise ValueError(
            f"only {disk.free} bytes free on artifact volume; requires {MINIMUM_DISK_FREE_BYTES}"
        )
    config_paths = [
        DERIVED_UPSTREAM_ROOT / "cfgs" / "unet" / "res18_monotemporal.yaml",
        DERIVED_UPSTREAM_ROOT / "cfgs" / "trainer_single_gpu.yaml",
        DERIVED_UPSTREAM_ROOT / "cfgs" / "data_monotemporal_full_features.yaml",
    ]
    configs = {
        str(path.relative_to(DERIVED_UPSTREAM_ROOT).as_posix()): _sha256(path)
        for path in config_paths
    }
    metadata: dict[str, object] = {
        "status": "pass",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "timing calibration only; no test AP",
        "upstream_origin": origin,
        "upstream_commit": actual_commit,
        "weights_revision": EXPECTED_WEIGHTS_REVISION,
        "runtime": runtime,
        "runtime_import_only": import_only,
        "runtime_patch": runtime_patch,
        "gpu": gpu,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "data_file_count": source_inventory["file_count"],
        "data_total_bytes": source_inventory["total_bytes"],
        "smoke_path": str(smoke_path.resolve()),
        "smoke_sha256": _sha256(smoke_path),
        "configs_sha256": configs,
        "overrides": overrides,
        "environment_controls": environment_controls,
    }
    return metadata, source_inventory


def _build_child_environment(run_directory: Path, runtime: Mapping[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    cache_root = run_directory.resolve() / "cache"
    local_cache_values = {
        "WANDB_DIR": cache_root / "wandb-dir",
        "WANDB_CACHE_DIR": cache_root / "wandb-cache",
        "WANDB_CONFIG_DIR": cache_root / "wandb-config",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HOME": cache_root / "huggingface",
    }
    for path in local_cache_values.values():
        path.mkdir(parents=True, exist_ok=True)
    torch_hub_dir = Path(str(runtime["torch_hub_dir"])).resolve()
    torch_home = torch_hub_dir.parent
    recorded = {
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_HOME": str(torch_home),
        **{key: str(value) for key, value in local_cache_values.items()},
    }
    child = dict(os.environ)
    child.update(recorded)
    return child, recorded


def _sample_nvidia_smi(child_pid: int, observer_seconds: float) -> Mapping[str, object]:
    gpu_result = _run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    matching_lines = [line for line in gpu_result.stdout.splitlines() if "RTX 3090" in line]
    if len(matching_lines) != 1:
        raise ValueError("GPU sampler did not find exactly one RTX 3090")
    fields = [field.strip() for field in matching_lines[0].split(",")]
    if len(fields) != 7:
        raise ValueError(f"unexpected nvidia-smi sample row: {matching_lines[0]}")

    child_memory = 0.0
    process_result = _run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in process_result.stdout.splitlines():
        process_fields = [field.strip() for field in line.split(",")]
        if len(process_fields) != 2:
            continue
        try:
            pid = int(process_fields[0])
            memory = float(process_fields[1])
        except ValueError:
            continue
        if pid == child_pid:
            child_memory = max(child_memory, memory)
    return {
        "observer_seconds": f"{observer_seconds:.9f}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_index": fields[0],
        "gpu_name": fields[1],
        "memory_total_mib": fields[2],
        "memory_used_mib": fields[3],
        "memory_free_mib": fields[4],
        "utilization_gpu_percent": fields[5],
        "power_draw_w": fields[6],
        "child_memory_mib": f"{child_memory:.3f}",
    }


def _copy_provenance(
    run_directory: Path, runtime_patch: Mapping[str, object], smoke_path: Path = SMOKE_PATH
) -> dict[str, str]:
    provenance = run_directory / "provenance"
    provenance.mkdir()
    sources = {
        "upstream.lock.json": REPRODUCTION_ROOT / "upstream.lock.json",
        "smoke.json": smoke_path,
        "official_entrypoint.py": Path(__file__).with_name("official_entrypoint.py"),
        "run_calibration.py": Path(__file__),
        "res18_import_scope.patch": RUNTIME_PATCH_PATH,
        "res18_monotemporal.yaml": DERIVED_UPSTREAM_ROOT / "cfgs" / "unet" / "res18_monotemporal.yaml",
        "trainer_single_gpu.yaml": DERIVED_UPSTREAM_ROOT / "cfgs" / "trainer_single_gpu.yaml",
        "data_monotemporal_full_features.yaml": DERIVED_UPSTREAM_ROOT / "cfgs" / "data_monotemporal_full_features.yaml",
    }
    copied: dict[str, str] = {}
    for name, source in sources.items():
        target = provenance / name
        shutil.copy2(source, target)
        copied[name] = _sha256(target)
    diff_path = provenance / "derived-runtime.diff"
    diff_path.write_text(str(runtime_patch["git_diff"]), encoding="utf-8")
    copied[diff_path.name] = _sha256(diff_path)
    return copied


def _write_step_timing_csv(path: Path, samples: Sequence[tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "optimizer_step",
                "observer_seconds",
                "delta_steps",
                "delta_seconds",
                "seconds_per_step",
                "post_warmup",
            ],
        )
        writer.writeheader()
        previous: tuple[int, float] | None = None
        for step, seconds in samples:
            row: dict[str, object] = {
                "optimizer_step": step,
                "observer_seconds": f"{seconds:.9f}",
                "delta_steps": "",
                "delta_seconds": "",
                "seconds_per_step": "",
                "post_warmup": step >= 50,
            }
            if previous is not None:
                delta_steps = step - previous[0]
                delta_seconds = seconds - previous[1]
                row.update(
                    {
                        "delta_steps": delta_steps,
                        "delta_seconds": f"{delta_seconds:.9f}",
                        "seconds_per_step": f"{delta_seconds / delta_steps:.9f}",
                    }
                )
            writer.writerow(row)
            previous = (step, seconds)


def _write_calibration_summary_csv(
    path: Path, timing: Mapping[str, object]
) -> None:
    scalar_items = {
        key: value
        for key, value in timing.items()
        if value is None or isinstance(value, (bool, float, int, str))
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_items))
        writer.writeheader()
        writer.writerow(scalar_items)


def _local_report(timing: Mapping[str, object]) -> str:
    return (
        "# Fold-2 Res18-U-Net T=1 All timing calibration\n\n"
        "Status: PASS (timing calibration only; no test AP).\n\n"
        f"- Optimizer steps: {timing['optimizer_steps']}\n"
        f"- Wall time: {timing['wall_seconds']:.6f} s\n"
        f"- Median post-warm-up step: {timing['median_step_seconds']:.6f} s\n"
        f"- Instantaneous throughput (batch 64): {timing['instantaneous_samples_per_second']:.6f} samples/s\n"
        f"- End-to-end throughput: {timing['end_to_end_samples_per_second']:.6f} samples/s\n"
        f"- Peak PyTorch allocated: {timing['peak_allocated_mib']:.3f} MiB\n"
        f"- Peak nvidia-smi total GPU used: {timing['peak_gpu_used_mib']:.3f} MiB\n"
        f"- Observed GPU cadence: mean {timing['gpu_interval_mean_seconds']:.6f} s, "
        f"median {timing['gpu_interval_median_seconds']:.6f} s, "
        f"effective {timing['gpu_observed_effective_hz']:.6f} Hz\n"
        "- GPU peak is the sampled maximum at that cadence and may miss a between-sample transient\n"
        f"- Optimistic empirical compute-only 10,000-step reference: {timing['compute_only_10000_seconds']:.3f} s\n"
        f"- 10,000-step epoch-aware estimate: {timing['epoch_aware_10000_central_seconds']:.3f} s "
        f"[{timing['epoch_aware_10000_lower_seconds']:.3f}, {timing['epoch_aware_10000_upper_seconds']:.3f}]\n"
        f"- 10,000-step naive wall-linear estimate: {timing['naive_wall_linear_10000_seconds']:.3f} s\n\n"
        "`failure.json` records only the original observer postprocessing parser failure; "
        "the child exited 0 at step 500. `completed.json` is the governing status after "
        "authorized offline finalization of the same raw run.\n\n"
        "This calibration does not invoke test or predict and cannot establish reproduction of 0.460 +/- 0.084.\n"
    )


def finalize_existing_run(run_directory: Path) -> dict[str, object]:
    """Finalize preserved raw artifacts without launching or retrying a child."""
    run_root = run_directory.resolve()
    launch = json.loads(GLOBAL_WORKER_RECOVERY_LOCK.read_text(encoding="utf-8"))
    started = json.loads((run_root / "started.json").read_text(encoding="utf-8"))
    if Path(launch.get("run_directory", "")).resolve() != run_root:
        raise ValueError("existing run is not linked by the final worker-recovery lock")
    if launch.get("attempt") != 3 or launch.get("worker_recovery") != 1:
        raise ValueError("existing run is not the authorized final third attempt")
    if launch.get("command") != started.get("command"):
        raise ValueError("existing run command differs from its atomic launch marker")
    pid = started.get("pid")
    if type(pid) is not int or pid <= 0 or _pid_is_alive(pid):
        raise ValueError("existing child must have a valid dead PID before finalization")
    validate_worker_failure_recovery(
        Path(str(launch["failed_run_directory"])),
        GLOBAL_RECOVERY_LOCK,
        GLOBAL_LAUNCH_LOCK,
    )
    expected_command = build_official_command(
        python_executable=ENVIRONMENT_PYTHON,
        entrypoint=Path(__file__).with_name("official_entrypoint.py"),
        upstream_root=DERIVED_UPSTREAM_ROOT,
        data_root=DATA_ROOT,
        run_directory=run_root,
        num_workers=8,
    )
    validate_effective_command(started["command"], expected_command)

    exit_marker = run_root / "exit-code.txt"
    exit_code = int(exit_marker.read_text(encoding="utf-8").strip())
    wall_seconds = measure_wall_seconds_from_markers(
        str(started["started_utc"]), exit_marker.stat().st_mtime_ns
    )
    stdout_text = (run_root / "stdout.log").read_text(
        encoding="utf-8", errors="replace"
    )
    stderr_text = (run_root / "stderr.log").read_text(
        encoding="utf-8", errors="replace"
    )
    combined_output = stdout_text + "\n" + stderr_text
    event_text = (run_root / "stream-events.jsonl").read_text(encoding="utf-8")
    peak_allocated = parse_peak_allocated(combined_output)
    samples = validate_success_evidence(
        event_text, combined_output, exit_code, peak_allocated
    )
    epoch_boundaries = parse_epoch_boundaries(event_text)
    validation_seconds, validation_identifiable = measure_validation_seconds(event_text)
    summary = summarize_timing(
        samples,
        wall_seconds=wall_seconds,
        startup_seconds=samples[0][1],
        validation_seconds=validation_seconds,
    )
    epoch_summary = summarize_epoch_aware_timing(
        epoch_boundaries=epoch_boundaries,
        startup_seconds=samples[0][1],
        median_step_seconds=summary["median_step_seconds"],
        p25_step_seconds=summary["p25_step_seconds"],
        p75_step_seconds=summary["p75_step_seconds"],
        wall_seconds=wall_seconds,
    )
    with (run_root / "gpu.csv").open(encoding="utf-8", newline="") as handle:
        gpu_rows = list(csv.DictReader(handle))
    gpu_summary = summarize_gpu_samples(gpu_rows)
    gpu_span_seconds = float(gpu_rows[-1]["observer_seconds"]) - float(
        gpu_rows[0]["observer_seconds"]
    )

    source_before = json.loads(
        (run_root / "source-data-pre.json").read_text(encoding="utf-8")
    )
    source_after = json.loads(
        (run_root / "source-data-post.json").read_text(encoding="utf-8")
    )
    assert_source_inventories_identical(source_before, source_after)
    post_commit = verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
    post_runtime_patch = _prepare_derived_runtime()
    preflight = json.loads((run_root / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("environment_controls", {}).get("data.num_workers") != 8:
        raise ValueError("existing run preflight did not authorize workers=8")
    effective_runtime_config = validate_effective_runtime_config(run_root / "config.yaml")

    timing: dict[str, object] = {
        "status": "pass",
        "purpose": "timing calibration only; no test AP",
        "optimizer_steps": samples[-1][0],
        "warmup_excluded_steps": "0-49",
        "batch_size": BATCH_SIZE,
        "num_workers": 8,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": peak_allocated / 1024**2,
        "validation_separately_identifiable": validation_identifiable,
        "epoch_aware_formula": (
            "startup_seconds + 82 * median_full_epoch_first_step_cycle_seconds + "
            "78 * median_instantaneous_step_seconds"
        ),
        "compute_only_interpretation": (
            "optimistic empirical compute-only extrapolation/reference"
        ),
        "naive_wall_linear_is_conservative_empirical": True,
        "upstream_commit": post_commit,
        "weights_revision": EXPECTED_WEIGHTS_REVISION,
        "runtime_patch_sha256": post_runtime_patch["patch_sha256"],
        "child_pid": pid,
        "command_sha256": launch["command_sha256"],
        "gpu_sampling_span_seconds": gpu_span_seconds,
        "wall_seconds_basis": "started.json UTC to exit-code.txt NTFS mtime",
        "observer_postprocess_recovered": True,
        **effective_runtime_config,
        **summary,
        **epoch_summary,
        **gpu_summary,
    }
    _write_step_timing_csv(run_root / "step-timing.csv", samples)
    _write_calibration_summary_csv(run_root / "calibration.csv", timing)
    write_json_atomic(run_root / "timing.json", timing)
    provenance_sha256 = {
        path.name: _sha256(path)
        for path in sorted((run_root / "provenance").iterdir())
        if path.is_file()
    }
    _, recorded_environment = _build_child_environment(
        run_root, preflight["runtime"]
    )
    write_json_atomic(
        run_root / "effective-command.json",
        {
            "command": started["command"],
            "command_sha256": launch["command_sha256"],
            "cwd": str(DERIVED_UPSTREAM_ROOT.resolve()),
            "environment": recorded_environment,
            "provenance_sha256": provenance_sha256,
        },
    )
    original_failure = json.loads(
        (run_root / "failure.json").read_text(encoding="utf-8")
    )
    write_json_atomic(
        run_root / "observer-postprocess-recovery.json",
        {
            "authorized_action": "offline finalization of the existing raw run only",
            "child_relaunched": False,
            "current_runner_sha256": _sha256(Path(__file__)),
            "original_observer_failure": original_failure,
            "parser_fix": "ignore only a completed epoch's same-epoch tqdm teardown reset",
            "raw_sha256": {
                name: _sha256(run_root / name)
                for name in (
                    "stdout.log",
                    "stderr.log",
                    "stream-events.jsonl",
                    "gpu.csv",
                    "exit-code.txt",
                )
            },
            "recovered_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    (run_root / "report.md").write_text(_local_report(timing), encoding="utf-8")
    write_json_atomic(
        run_root / "completed.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code,
            "failure_json_classification": (
                "observer postprocess failure only; not a child or training failure"
            ),
            "governing_status_marker": "completed.json",
            "observer_postprocess_recovered": True,
            "pid": pid,
            "status": "pass",
        },
    )
    return timing


def _run_one_calibration(
    recovery_failed_run: Path | None = None,
    worker_failed_run: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    if recovery_failed_run is not None and worker_failed_run is not None:
        raise ValueError("only one explicit recovery mode may be selected")
    recovery_evidence: dict[str, object] | None = None
    worker_recovery_evidence: dict[str, object] | None = None
    if worker_failed_run is not None:
        if GLOBAL_WORKER_RECOVERY_LOCK.exists():
            raise FileExistsError(
                f"worker recovery lock already exists: {GLOBAL_WORKER_RECOVERY_LOCK}"
            )
        worker_recovery_evidence = validate_worker_failure_recovery(
            worker_failed_run,
            GLOBAL_RECOVERY_LOCK,
            GLOBAL_LAUNCH_LOCK,
        )
    elif recovery_failed_run is None:
        if GLOBAL_LAUNCH_LOCK.exists():
            raise FileExistsError(f"calibration launch lock already exists: {GLOBAL_LAUNCH_LOCK}")
    else:
        if GLOBAL_RECOVERY_LOCK.exists():
            raise FileExistsError(f"pretraining recovery lock already exists: {GLOBAL_RECOVERY_LOCK}")
        recovery_evidence = validate_pretraining_failure_recovery(
            recovery_failed_run, GLOBAL_LAUNCH_LOCK
        )
    num_workers = 8 if worker_recovery_evidence is not None else 64
    smoke_path = WORKER_SMOKE_PATH if worker_recovery_evidence is not None else SMOKE_PATH
    run_directory = create_unique_run_directory(ARTIFACTS_ROOT)
    try:
        preflight, source_before = _perform_preflight(
            run_directory,
            allow_preserved_launch_lock=(
                recovery_evidence is not None or worker_recovery_evidence is not None
            ),
            num_workers=num_workers,
            smoke_path=smoke_path,
        )
        write_json_atomic(run_directory / "preflight.json", preflight)
        write_json_atomic(run_directory / "source-data-pre.json", source_before)
        provenance_sha256 = _copy_provenance(
            run_directory, preflight["runtime_patch"], smoke_path
        )

        command = build_official_command(
            python_executable=ENVIRONMENT_PYTHON,
            entrypoint=Path(__file__).with_name("official_entrypoint.py"),
            upstream_root=DERIVED_UPSTREAM_ROOT,
            data_root=DATA_ROOT,
            run_directory=run_directory,
            num_workers=num_workers,
        )
        expected_command = build_official_command(
            python_executable=ENVIRONMENT_PYTHON,
            entrypoint=Path(__file__).with_name("official_entrypoint.py"),
            upstream_root=DERIVED_UPSTREAM_ROOT,
            data_root=DATA_ROOT,
            run_directory=run_directory,
            num_workers=num_workers,
        )
        validate_effective_command(command, expected_command)
        if worker_recovery_evidence is not None:
            previous_command = json.loads(
                GLOBAL_RECOVERY_LOCK.read_text(encoding="utf-8")
            )["command"]
            expected_previous_command = build_official_command(
                python_executable=ENVIRONMENT_PYTHON,
                entrypoint=Path(__file__).with_name("official_entrypoint.py"),
                upstream_root=DERIVED_UPSTREAM_ROOT,
                data_root=DATA_ROOT,
                run_directory=worker_failed_run,
                num_workers=64,
            )
            validate_effective_command(previous_command, expected_previous_command)
        child_environment, recorded_environment = _build_child_environment(
            run_directory, preflight["runtime"]
        )
        command_sha256 = hashlib.sha256(
            json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        launch_payload = {
            "attempt": (
                3
                if worker_recovery_evidence is not None
                else 2
                if recovery_evidence is not None
                else 1
            ),
            "authorized_optimizer_steps": 500,
            "command": command,
            "command_sha256": command_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(run_directory),
        }
        if recovery_evidence is None and worker_recovery_evidence is None:
            acquire_launch_lock(GLOBAL_LAUNCH_LOCK, launch_payload)
        elif recovery_evidence is not None:
            launch_payload.update(recovery_evidence)
            launch_payload["recovery_authorized_utc"] = datetime.now(timezone.utc).isoformat()
            acquire_launch_lock(GLOBAL_RECOVERY_LOCK, launch_payload)
            write_json_atomic(run_directory / "recovery-authorization.json", launch_payload)
        else:
            launch_payload.update(worker_recovery_evidence or {})
            launch_payload["worker_recovery_authorized_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            acquire_launch_lock(GLOBAL_WORKER_RECOVERY_LOCK, launch_payload)
            write_json_atomic(
                run_directory / "worker-recovery-authorization.json", launch_payload
            )
        acquire_launch_lock(run_directory / "launch.lock.json", launch_payload)
        print(
            json.dumps(
                {"status": "launching-once", "run_directory": str(run_directory), "command": command},
                ensure_ascii=False,
            ),
            flush=True,
        )
        observation = observe_process(
            command,
            cwd=DERIVED_UPSTREAM_ROOT,
            environment=child_environment,
            run_directory=run_directory,
            gpu_sampler=_sample_nvidia_smi,
        )

        source_after = build_source_inventory(DATA_ROOT)
        write_json_atomic(run_directory / "source-data-post.json", source_after)
        assert_source_inventories_identical(source_before, source_after)
        post_commit = verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
        post_runtime_patch = _prepare_derived_runtime()
        (run_directory / "upstream-status-post.txt").write_text(
            f"commit={post_commit}\nstatus=clean\n", encoding="utf-8"
        )
        if observation["gpu_errors"]:
            raise ValueError("GPU observer errors: " + "; ".join(observation["gpu_errors"]))

        stdout_text = (run_directory / "stdout.log").read_text(encoding="utf-8", errors="replace")
        stderr_text = (run_directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
        combined_output = stdout_text + "\n" + stderr_text
        event_text = (run_directory / "stream-events.jsonl").read_text(encoding="utf-8")
        peak_allocated = parse_peak_allocated(combined_output)
        samples = validate_success_evidence(
            event_text,
            combined_output,
            int(observation["exit_code"]),
            peak_allocated,
        )
        epoch_boundaries = parse_epoch_boundaries(event_text)
        validation_seconds, validation_identifiable = measure_validation_seconds(event_text)
        summary = summarize_timing(
            samples,
            wall_seconds=float(observation["wall_seconds"]),
            startup_seconds=samples[0][1],
            validation_seconds=validation_seconds,
        )
        epoch_summary = summarize_epoch_aware_timing(
            epoch_boundaries=epoch_boundaries,
            startup_seconds=samples[0][1],
            median_step_seconds=summary["median_step_seconds"],
            p25_step_seconds=summary["p25_step_seconds"],
            p75_step_seconds=summary["p75_step_seconds"],
            wall_seconds=float(observation["wall_seconds"]),
        )
        gpu_rows = observation["gpu_rows"]
        if not isinstance(gpu_rows, list):
            raise ValueError("GPU observer rows are malformed")
        gpu_summary = summarize_gpu_samples(gpu_rows)
        effective_runtime_config = validate_effective_runtime_config(
            run_directory / "config.yaml"
        )
        timing: dict[str, object] = {
            "status": "pass",
            "purpose": "timing calibration only; no test AP",
            "optimizer_steps": samples[-1][0],
            "warmup_excluded_steps": "0-49",
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "validation_separately_identifiable": validation_identifiable,
            "epoch_aware_formula": (
                "startup_seconds + 82 * median_full_epoch_first_step_cycle_seconds + "
                "78 * median_instantaneous_step_seconds"
            ),
            "compute_only_interpretation": (
                "optimistic empirical compute-only extrapolation/reference"
            ),
            "naive_wall_linear_is_conservative_empirical": True,
            "upstream_commit": post_commit,
            "weights_revision": EXPECTED_WEIGHTS_REVISION,
            "runtime_patch_sha256": post_runtime_patch["patch_sha256"],
            "child_pid": observation["pid"],
            "command_sha256": command_sha256,
            **effective_runtime_config,
            **summary,
            **epoch_summary,
            **gpu_summary,
        }
        _write_step_timing_csv(run_directory / "step-timing.csv", samples)
        _write_calibration_summary_csv(run_directory / "calibration.csv", timing)
        write_json_atomic(run_directory / "timing.json", timing)
        write_json_atomic(
            run_directory / "effective-command.json",
            {
                "command": command,
                "command_sha256": command_sha256,
                "cwd": str(DERIVED_UPSTREAM_ROOT.resolve()),
                "environment": recorded_environment,
                "provenance_sha256": provenance_sha256,
            },
        )
        (run_directory / "report.md").write_text(_local_report(timing), encoding="utf-8")
        write_json_atomic(
            run_directory / "completed.json",
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": observation["exit_code"],
                "pid": observation["pid"],
                "status": "pass",
            },
        )
        return run_directory, timing
    except BaseException as error:
        write_json_atomic(
            run_directory / "failure.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "status": "fail",
                "training_retry_performed": False,
            },
        )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--preflight-recovery", type=Path, metavar="FAILED_RUN")
    action.add_argument(
        "--authorized-pretraining-recovery", type=Path, metavar="FAILED_RUN"
    )
    action.add_argument("--preflight-worker-recovery", type=Path, metavar="FAILED_RUN")
    action.add_argument(
        "--authorized-worker-recovery", type=Path, metavar="FAILED_RUN"
    )
    action.add_argument("--finalize-existing", type=Path, metavar="RUN_DIRECTORY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.preflight_only:
            placeholder = ARTIFACTS_ROOT / "preflight-placeholder"
            placeholder.mkdir(parents=True, exist_ok=True)
            preflight, _ = _perform_preflight(placeholder)
            print(json.dumps(preflight, sort_keys=True, separators=(",", ":")))
            return 0
        if arguments.preflight_recovery is not None:
            if GLOBAL_RECOVERY_LOCK.exists():
                raise FileExistsError(
                    f"pretraining recovery lock already exists: {GLOBAL_RECOVERY_LOCK}"
                )
            recovery = validate_pretraining_failure_recovery(
                arguments.preflight_recovery, GLOBAL_LAUNCH_LOCK
            )
            placeholder = ARTIFACTS_ROOT / "preflight-recovery-placeholder"
            placeholder.mkdir(parents=True, exist_ok=True)
            preflight, _ = _perform_preflight(
                placeholder, allow_preserved_launch_lock=True
            )
            print(
                json.dumps(
                    {"preflight": preflight, "recovery": recovery},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if arguments.preflight_worker_recovery is not None:
            if GLOBAL_WORKER_RECOVERY_LOCK.exists():
                raise FileExistsError(
                    f"worker recovery lock already exists: {GLOBAL_WORKER_RECOVERY_LOCK}"
                )
            recovery = validate_worker_failure_recovery(
                arguments.preflight_worker_recovery,
                GLOBAL_RECOVERY_LOCK,
                GLOBAL_LAUNCH_LOCK,
            )
            placeholder = ARTIFACTS_ROOT / "preflight-worker-recovery-placeholder"
            placeholder.mkdir(parents=True, exist_ok=True)
            preflight, _ = _perform_preflight(
                placeholder,
                allow_preserved_launch_lock=True,
                num_workers=8,
                smoke_path=WORKER_SMOKE_PATH,
            )
            print(
                json.dumps(
                    {"preflight": preflight, "recovery": recovery},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if arguments.finalize_existing is not None:
            timing = finalize_existing_run(arguments.finalize_existing)
            print(
                json.dumps(
                    {
                        "run_directory": str(arguments.finalize_existing.resolve()),
                        "status": "pass",
                        "timing": timing,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        run_directory, timing = _run_one_calibration(
            arguments.authorized_pretraining_recovery,
            arguments.authorized_worker_recovery,
        )
    except (FileExistsError, OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        traceback.print_exc()
        print(f"calibration runner failed without retry: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"run_directory": str(run_directory), "status": "pass", "timing": timing},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independently recompute and verify one preserved calibration run."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path


EXPECTED_COMMIT = "ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad"
EXPECTED_PATCH_SHA256 = "e7b0211e762cb7a888b0ce699e2d22537b372a49cfc0baf52e078513fbdc0c83"
EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT = 608.4653828020165
SOURCE_YAML_POS_CLASS_WEIGHT = 236.0
AUTHORIZED_EXPORT_DELETIONS = [
    "from .UTAELightning import UTAELightning",
    "from .SwinUnetLightning import SwinUnetLightning",
    "from .SwinUnetTempLightning import SwinUnetTempLightning",
    "from .UTAELightningDumb import UTAELightningDumb",
    "from .TransUnetLightning import TransUnetLightning",
    "from .SMPTempModel import SMPTempModel ",
    "from .SegFormerLightning import SegFormerLightning",
]
REQUIRED_RES18_EXPORTS = [
    "from .BaseModel import BaseModel",
    "from .ConvLSTMLightning import ConvLSTMLightning",
    "from .LogisticRegression import LogisticRegression",
    "from .SMPModel import SMPModel",
]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
FIXED_ENVIRONMENT_PYTHON = Path(r"D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe")
FIXED_DATA_ROOT = Path(r"D:\WildFire Project\data\hdf5")
PROGRESS = re.compile(r"\bEpoch\s+(\d+):.*?(\d+)\s*/\s*(\d+)")
VALIDATION = re.compile(r"\bValidation DataLoader\s+\d+:.*?(\d+)\s*/\s*(\d+)")
METRIC = re.compile(
    r"\b(?:train_loss|train_f1|val_loss|val_avg_precision|val_f1)"
    r"(?:_step|_epoch)?\s*=\s*"
    r"([+-]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?))",
    re.IGNORECASE,
)


def build_exact_expected_command(run_root: Path, derived_root: Path) -> list[str]:
    """Independently construct the complete, ordered attempt-3 command."""
    run = run_root.resolve()
    derived = derived_root.resolve()
    config_root = derived / "cfgs"
    return [
        str(FIXED_ENVIRONMENT_PYTHON.resolve()),
        str((REPRODUCTION_ROOT / "scripts" / "official_entrypoint.py").resolve()),
        "--upstream-root",
        str(derived),
        f"--config={(config_root / 'unet' / 'res18_monotemporal.yaml').as_posix()}",
        f"--trainer={(config_root / 'trainer_single_gpu.yaml').as_posix()}",
        f"--data={(config_root / 'data_monotemporal_full_features.yaml').as_posix()}",
        f"--data.data_dir={FIXED_DATA_ROOT.resolve()}",
        "--data.data_fold_id=2",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        "--trainer.max_steps=500",
        f"--trainer.default_root_dir={run}",
        "--do_test=false",
    ]


def validate_exact_command(actual: Sequence[str], expected: Sequence[str]) -> None:
    if list(actual) != list(expected):
        raise ValueError("attempt-3 command differs from the full exact expected command")


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_command_lineage(
    run_root: Path, global_worker_lock: Path, expected_command: Sequence[str]
) -> str:
    """Cross-check command bytes/hash across every final-attempt marker."""
    run = run_root.resolve()
    expected = list(expected_command)
    expected_hash = _command_sha256(expected)
    marker_paths = [
        run / "started.json",
        global_worker_lock.resolve(),
        run / "launch.lock.json",
        run / "worker-recovery-authorization.json",
        run / "effective-command.json",
        run / "timing.json",
    ]
    for path in marker_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "command" in payload and payload["command"] != expected:
            raise ValueError(f"command lineage list mismatch: {path.name}")
        if "command_sha256" in payload and payload["command_sha256"] != expected_hash:
            raise ValueError(f"command lineage hash mismatch: {path.name}")
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    if _command_sha256(started["command"]) != expected_hash:
        raise ValueError("command lineage started command hash mismatch")
    return expected_hash


def validate_import_scope_contents(original_text: str, derived_text: str) -> None:
    """Derive the only authorized initializer result from pristine content."""
    original_lines_with_endings = original_text.splitlines(keepends=True)
    original_lines = [line.rstrip("\r\n") for line in original_lines_with_endings]
    if not all(line in original_lines for line in REQUIRED_RES18_EXPORTS):
        raise ValueError("original initializer lacks a required Res18 export")
    if not all(line in original_lines for line in AUTHORIZED_EXPORT_DELETIONS):
        raise ValueError("original initializer lacks an authorized unused export")
    expected_text = "".join(
        line
        for line in original_lines_with_endings
        if line.rstrip("\r\n") not in AUTHORIZED_EXPORT_DELETIONS
    )
    if derived_text != expected_text:
        raise ValueError("derived initializer differs beyond the exact seven deletions")


def validate_patch_text(patch: str) -> None:
    affected = re.findall(r"(?m)^diff --git a/(\S+) b/\1$", patch)
    deleted = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if affected != ["src/models/__init__.py"]:
        raise ValueError("patch contract must affect only src/models/__init__.py")
    if deleted != AUTHORIZED_EXPORT_DELETIONS or added:
        raise ValueError("patch contract must be exactly seven deletions and no additions")


def _single_pos_weight(text: str, label: str) -> float:
    matches = re.findall(r"(?m)^\s+pos_class_weight:\s*(\S+)(?:\s+#.*)?$", text)
    if len(matches) != 1:
        raise ValueError(f"{label} must contain one positive-class weight")
    return float(matches[0])


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def derive_dynamic_positive_weight_provenance(
    source_yaml: str, train_source: str, effective_config: str
) -> dict[str, object]:
    """Derive the official runtime override conclusion from source and AST."""
    source_weight = _single_pos_weight(source_yaml, "source YAML")
    effective_weight = _single_pos_weight(effective_config, "effective config")
    if source_weight != SOURCE_YAML_POS_CLASS_WEIGHT:
        raise ValueError("source YAML positive-class weight changed")
    if effective_weight != EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT:
        raise ValueError("effective config positive-class weight changed")
    tree = ast.parse(train_source)
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "before_instantiate_classes"
    ]
    if len(methods) != 1:
        raise ValueError("before_instantiate_classes method is missing or ambiguous")
    method = methods[0]
    computes_inverse = False
    assigns_to_config = False
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "pos_class_weight":
            value = node.value
            computes_inverse = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "float"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.BinOp)
                and isinstance(value.args[0].op, ast.Div)
                and isinstance(value.args[0].left, ast.Constant)
                and value.args[0].left.value == 1
                and isinstance(value.args[0].right, ast.Name)
                and value.args[0].right.id == "fire_rate"
            )
        if _attribute_path(target) == "self.config.model.init_args.pos_class_weight":
            assigns_to_config = (
                isinstance(node.value, ast.Name) and node.value.id == "pos_class_weight"
            )
    if not computes_inverse or not assigns_to_config:
        raise ValueError(
            "before_instantiate_classes does not assign float(1 / fire_rate) to config"
        )
    dynamic_override = source_weight != effective_weight and computes_inverse and assigns_to_config
    return {
        "source_yaml_pos_class_weight": source_weight,
        "effective_pos_class_weight": effective_weight,
        "official_dynamic_override": dynamic_override,
        "official_override_expression": "float(1 / fire_rate)",
    }


def _events(text: str) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        seconds = float(item["seconds"])
        fragment = item["text"]
        if not math.isfinite(seconds) or not isinstance(fragment, str):
            raise ValueError("invalid raw observer event")
        if result and seconds < result[-1][0]:
            raise ValueError("raw observer event time regressed")
        result.append((seconds, fragment))
    return result


def independent_progress(text: str) -> list[tuple[int, float]]:
    """Independently reconstruct global steps from Lightning epoch progress."""
    result: list[tuple[int, float]] = []
    current_epoch: int | None = None
    offset = 0
    total: int | None = None
    position: int | None = None
    for seconds, fragment in _events(text):
        if "Validation DataLoader" in fragment or "Sanity Checking" in fragment:
            continue
        match = PROGRESS.search(fragment)
        if match is None:
            continue
        next_epoch, next_position, next_total = map(int, match.groups())
        if next_total <= 0 or not 0 <= next_position <= next_total:
            raise ValueError("invalid progress coordinates")
        if current_epoch is None:
            current_epoch, total = next_epoch, next_total
        elif next_epoch == current_epoch:
            if next_total != total:
                raise ValueError("epoch denominator changed")
            if position == total and next_position == 0:
                continue
            if position is not None and next_position < position:
                raise ValueError("optimizer position regressed")
        elif next_epoch == current_epoch + 1 and position == total:
            offset += int(total)
            current_epoch, total, position = next_epoch, next_total, None
        else:
            raise ValueError("epoch sequence is discontinuous")
        position = next_position
        global_step = offset + next_position
        if result and global_step < result[-1][0]:
            raise ValueError("global step regressed")
        if not result or global_step != result[-1][0]:
            result.append((global_step, seconds))
    return result


def independent_effective_config(config_path: Path) -> float:
    """Independently read the saved effective positive-class weight."""
    matches = re.findall(
        r"(?m)^\s+pos_class_weight:\s*(\S+)\s*$",
        config_path.read_text(encoding="utf-8"),
    )
    if len(matches) != 1:
        raise ValueError("saved config must contain one positive-class weight")
    value = float(matches[0])
    if value != EXPECTED_EFFECTIVE_POS_CLASS_WEIGHT:
        raise ValueError("saved config positive-class weight is not the official dynamic value")
    return value


def independent_epoch_boundaries(text: str) -> list[tuple[int, int, float, int]]:
    """Independently select each epoch's first completed-step timestamp."""
    boundaries: list[tuple[int, int, float, int]] = []
    totals: list[int] = []
    seen: set[int] = set()
    for seconds, fragment in _events(text):
        if "Validation DataLoader" in fragment or "Sanity Checking" in fragment:
            continue
        match = PROGRESS.search(fragment)
        if match is None:
            continue
        epoch, position, total = map(int, match.groups())
        if position != 1 or epoch in seen:
            continue
        if epoch != len(boundaries) or total <= 0:
            raise ValueError("independent epoch boundaries are discontinuous")
        boundaries.append((epoch, sum(totals) + 1, seconds, total))
        totals.append(total)
        seen.add(epoch)
    if len(boundaries) < 2:
        raise ValueError("insufficient epoch boundaries")
    return boundaries


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def independent_timing(
    samples: Sequence[tuple[int, float]],
    *,
    wall_seconds: float,
    validation_seconds: float,
) -> dict[str, float]:
    intervals = [
        (current_seconds - previous_seconds) / (current_step - previous_step)
        for (previous_step, previous_seconds), (current_step, current_seconds) in zip(
            samples, samples[1:]
        )
        if previous_step >= 49
    ]
    if not intervals:
        raise ValueError("no post-warm-up intervals")
    median = statistics.median(intervals)
    p25, p75 = _percentile(intervals, 0.25), _percentile(intervals, 0.75)
    startup = samples[0][1]
    return {
        "wall_seconds": wall_seconds,
        "startup_seconds": startup,
        "validation_seconds": validation_seconds,
        "observed_progress_seconds": samples[-1][1] - samples[0][1],
        "post_warmup_interval_count": float(len(intervals)),
        "median_step_seconds": median,
        "p25_step_seconds": p25,
        "p75_step_seconds": p75,
        "instantaneous_samples_per_second": 64 / median,
        "compute_only_10000_seconds": 10_000 * median,
    }


def independent_epoch_projection(
    boundaries: Sequence[tuple[int, int, float, int]],
    *,
    startup_seconds: float,
    median_step_seconds: float,
    p25_step_seconds: float,
    p75_step_seconds: float,
    wall_seconds: float,
    observed_steps: int,
) -> dict[str, object]:
    epoch_size = boundaries[0][3]
    if any(boundary[3] != epoch_size for boundary in boundaries):
        raise ValueError("independent epoch size changed")
    cycles = [
        current[2] - previous[2]
        for previous, current in zip(boundaries, boundaries[1:])
    ]
    if not cycles or any(not math.isfinite(value) or value <= 0 for value in cycles):
        raise ValueError("independent epoch cycles are invalid")
    median_cycle = statistics.median(cycles)
    p25_cycle = _percentile(cycles, 0.25)
    p75_cycle = _percentile(cycles, 0.75)
    full_cycles, partial_steps = divmod(10_000, epoch_size)
    return {
        "epoch_size_steps": epoch_size,
        "epoch_first_step_boundaries": [list(boundary) for boundary in boundaries],
        "epoch_cycle_seconds": cycles,
        "epoch_cycle_count": len(cycles),
        "median_epoch_cycle_seconds": median_cycle,
        "p25_epoch_cycle_seconds": p25_cycle,
        "p75_epoch_cycle_seconds": p75_cycle,
        "projected_full_epoch_cycles": full_cycles,
        "projected_partial_steps": partial_steps,
        "epoch_aware_10000_central_seconds": (
            startup_seconds + full_cycles * median_cycle + partial_steps * median_step_seconds
        ),
        "epoch_aware_10000_lower_seconds": (
            startup_seconds + full_cycles * p25_cycle + partial_steps * p25_step_seconds
        ),
        "epoch_aware_10000_upper_seconds": (
            startup_seconds + full_cycles * p75_cycle + partial_steps * p75_step_seconds
        ),
        "naive_wall_linear_10000_seconds": wall_seconds / observed_steps * 10_000,
        "end_to_end_samples_per_second": observed_steps * 64 / wall_seconds,
    }


def _validation_seconds(text: str) -> float:
    total_seconds = 0.0
    start: float | None = None
    saw = False
    for seconds, fragment in _events(text):
        if "Sanity Checking" in fragment:
            continue
        match = VALIDATION.search(fragment)
        if match is None:
            continue
        saw = True
        position, total = map(int, match.groups())
        if position == 0:
            if start is not None:
                raise ValueError("overlapping validation progress")
            start = seconds
        elif start is None:
            raise ValueError("validation progress lacks a start")
        if position == total:
            if start is None:
                raise ValueError("validation progress lacks a start")
            total_seconds += seconds - start
            start = None
    if not saw or start is not None:
        raise ValueError("validation progress is not independently identifiable")
    return total_seconds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root.resolve()), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "git verification failed")
    return result.stdout


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify(
    run_root: Path, original_root: Path, derived_root: Path, patch_path: Path
) -> dict[str, object]:
    run = run_root.resolve()
    original = original_root.resolve()
    derived = derived_root.resolve()
    expected_original = (REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS").resolve()
    expected_derived = (
        REPRODUCTION_ROOT / ".local" / "WildfireSpreadTS-res18-runtime"
    ).resolve()
    expected_patch = (
        REPRODUCTION_ROOT / "patches" / "res18_import_scope.patch"
    ).resolve()
    if original != expected_original or derived != expected_derived:
        raise ValueError("upstream roots are not the exact pinned reproduction paths")
    if patch_path.resolve() != expected_patch:
        raise ValueError("patch path is not the exact tracked runtime patch")
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    command = started["command"]
    expected_command = build_exact_expected_command(run, derived)
    validate_exact_command(command, expected_command)
    command_sha256 = validate_command_lineage(
        run,
        run.parent / "fold2-calibration-worker-recovery.lock.json",
        expected_command,
    )

    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    stdout = (run / "stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr = (run / "stderr.log").read_text(encoding="utf-8", errors="replace")
    combined = stdout + "\n" + stderr
    if exit_code != 0 or re.search(r"max_steps=500`?\s+reached", combined) is None:
        raise ValueError("child did not exit cleanly at exact max_steps=500")
    if re.search(r"\b(?:testing|predicting) dataloader\b", combined, re.IGNORECASE):
        raise ValueError("test or predict invocation found")
    if "out of memory" in combined.lower():
        raise ValueError("OOM found in final child output")
    peak_matches = re.findall(
        r"(?m)^WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\d+)\s*$", combined
    )
    if len(peak_matches) != 1 or int(peak_matches[0]) <= 0:
        raise ValueError("unique positive PyTorch peak sentinel is missing")
    metrics = [float(value) for value in METRIC.findall(combined)]
    if not metrics or not all(math.isfinite(value) for value in metrics):
        raise ValueError("finite metrics are missing")

    event_text = (run / "stream-events.jsonl").read_text(encoding="utf-8")
    samples = independent_progress(event_text)
    if samples[0][0] != 0 or samples[-1][0] != 500 or len(samples) != 501:
        raise ValueError("independent progress reconstruction did not establish steps 0..500")
    validation_seconds = _validation_seconds(event_text)
    start_seconds = datetime.fromisoformat(started["started_utc"]).timestamp()
    wall_seconds = (run / "exit-code.txt").stat().st_mtime_ns / 1e9 - start_seconds
    timing = independent_timing(
        samples, wall_seconds=wall_seconds, validation_seconds=validation_seconds
    )
    epoch_boundaries = independent_epoch_boundaries(event_text)
    epoch_projection = independent_epoch_projection(
        epoch_boundaries,
        startup_seconds=timing["startup_seconds"],
        median_step_seconds=timing["median_step_seconds"],
        p25_step_seconds=timing["p25_step_seconds"],
        p75_step_seconds=timing["p75_step_seconds"],
        wall_seconds=wall_seconds,
        observed_steps=500,
    )

    with (run / "gpu.csv").open(encoding="utf-8", newline="") as handle:
        gpu = list(csv.DictReader(handle))
    used = [float(row["memory_used_mib"]) for row in gpu]
    utilization = [float(row["utilization_gpu_percent"]) for row in gpu]
    child = [float(row["child_memory_mib"]) for row in gpu]
    child_available = any(value > 0 for value in child)
    gpu_summary = {
        "gpu_sample_count": float(len(gpu)),
        "gpu_sampling_span_seconds": float(gpu[-1]["observer_seconds"])
        - float(gpu[0]["observer_seconds"]),
        "peak_gpu_used_mib": max(used),
        "peak_child_process_mib": max(child) if child_available else None,
        "child_process_memory_available": child_available,
        "gpu_utilization_min_percent": min(utilization),
        "gpu_utilization_median_percent": statistics.median(utilization),
        "gpu_utilization_max_percent": max(utilization),
    }
    gpu_seconds = [float(row["observer_seconds"]) for row in gpu]
    gpu_intervals = [
        current - previous
        for previous, current in zip(gpu_seconds, gpu_seconds[1:])
    ]
    if not gpu_intervals or any(interval <= 0 for interval in gpu_intervals):
        raise ValueError("GPU sampling timestamps are not strictly increasing")
    gpu_summary.update(
        {
            "gpu_interval_count": len(gpu_intervals),
            "gpu_interval_mean_seconds": statistics.mean(gpu_intervals),
            "gpu_interval_median_seconds": statistics.median(gpu_intervals),
            "gpu_observed_effective_hz": len(gpu_intervals)
            / (gpu_seconds[-1] - gpu_seconds[0]),
            "gpu_peak_is_observed_sample_max": True,
            "gpu_peak_may_miss_between_sample_transients": True,
        }
    )
    source_pre = json.loads((run / "source-data-pre.json").read_text(encoding="utf-8"))
    source_post = json.loads((run / "source-data-post.json").read_text(encoding="utf-8"))
    if source_pre != source_post:
        raise ValueError("source inventory changed")
    original_commit = _git(original, "rev-parse", "HEAD").strip()
    original_status = _git(original, "status", "--porcelain").splitlines()
    derived_commit = _git(derived, "rev-parse", "HEAD").strip()
    derived_status = _git(derived, "status", "--porcelain").splitlines()
    if original_commit != EXPECTED_COMMIT or original_status:
        raise ValueError("original pinned checkout integrity failed")
    if derived_commit != EXPECTED_COMMIT or derived_status != [" M src/models/__init__.py"]:
        raise ValueError("derived runtime integrity failed")
    original_init = (original / "src" / "models" / "__init__.py").read_text(
        encoding="utf-8"
    )
    derived_init = (derived / "src" / "models" / "__init__.py").read_text(
        encoding="utf-8"
    )
    validate_import_scope_contents(original_init, derived_init)
    patch_text = patch_path.read_text(encoding="utf-8")
    validate_patch_text(patch_text)
    patch_sha256 = _sha256(patch_path)
    if patch_sha256 != EXPECTED_PATCH_SHA256:
        raise ValueError("runtime patch hash changed")
    changed_names = _git(derived, "diff", "--name-only").splitlines()
    if changed_names != ["src/models/__init__.py"]:
        raise ValueError("derived runtime contains a change outside the initializer")
    actual_diff = _git(
        derived,
        "diff",
        "--no-ext-diff",
        "--",
        "src/models/__init__.py",
    )
    copied_diff = (run / "provenance" / "derived-runtime.diff").read_text(
        encoding="utf-8"
    )
    if copied_diff != actual_diff:
        raise ValueError("copied derived diff differs from the exact runtime diff")
    copied_patch = run / "provenance" / "res18_import_scope.patch"
    if copied_patch.read_bytes() != patch_path.read_bytes():
        raise ValueError("copied runtime patch differs from the tracked patch")
    preflight = json.loads((run / "preflight.json").read_text(encoding="utf-8"))
    recorded_patch = preflight["runtime_patch"]
    actual_diff_sha256 = hashlib.sha256(actual_diff.encode("utf-8")).hexdigest()
    if recorded_patch.get("patch_sha256") != patch_sha256:
        raise ValueError("preflight patch hash differs from the tracked patch")
    if recorded_patch.get("git_diff_sha256") != actual_diff_sha256:
        raise ValueError("preflight copied diff hash differs from the actual diff")

    pos_weight_provenance = derive_dynamic_positive_weight_provenance(
        (original / "cfgs" / "unet" / "res18_monotemporal.yaml").read_text(
            encoding="utf-8"
        ),
        (original / "src" / "train.py").read_text(encoding="utf-8"),
        (run / "config.yaml").read_text(encoding="utf-8"),
    )

    recorded = json.loads((run / "timing.json").read_text(encoding="utf-8"))
    effective_pos_class_weight = independent_effective_config(run / "config.yaml")
    if recorded.get("effective_pos_class_weight") != effective_pos_class_weight:
        raise ValueError("recorded effective positive-class weight mismatch")
    if recorded.get("source_yaml_pos_class_weight") != pos_weight_provenance[
        "source_yaml_pos_class_weight"
    ]:
        raise ValueError("recorded source-YAML positive-class weight mismatch")
    if recorded.get("official_dynamic_override") != pos_weight_provenance[
        "official_dynamic_override"
    ]:
        raise ValueError("recorded dynamic positive-class override flag is missing")
    if recorded.get("compute_only_interpretation") != (
        "optimistic empirical compute-only extrapolation/reference"
    ):
        raise ValueError("compute-only interpretation is missing or overstated")
    if "compute_only_is_hard_lower_bound" in recorded:
        raise ValueError("obsolete hard-lower-bound claim remains in timing")
    comparisons = {**timing, **epoch_projection, **gpu_summary}
    for key, value in comparisons.items():
        if isinstance(value, list) or value is None or isinstance(value, bool):
            matches = recorded[key] == value
        elif isinstance(value, (float, int)):
            matches = math.isclose(float(recorded[key]), float(value), rel_tol=0, abs_tol=1e-9)
        else:
            matches = recorded[key] == value
        if not matches:
            raise ValueError(f"recorded timing mismatch: {key}")
    peak_bytes = int(peak_matches[0])
    if recorded["peak_allocated_bytes"] != peak_bytes:
        raise ValueError("recorded peak allocation mismatch")

    raw_names = (
        "stdout.log",
        "stderr.log",
        "stream-events.jsonl",
        "gpu.csv",
        "exit-code.txt",
        "started.json",
    )
    return {
        "status": "pass",
        "independent_implementation": True,
        "runner_imported": False,
        "exit_code": exit_code,
        "optimizer_steps": samples[-1][0],
        "progress_sample_count": len(samples),
        "num_workers": 8,
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_mib": peak_bytes / 1024**2,
        "command_sha256": command_sha256,
        "exact_command_verified": True,
        "command_lineage_verified": True,
        "data_inventory_identical": True,
        "data_file_count": source_pre["file_count"],
        "data_total_bytes": source_pre["total_bytes"],
        "original_upstream_commit": original_commit,
        "original_upstream_clean": True,
        "derived_upstream_commit": derived_commit,
        "derived_status": derived_status,
        "patch_sha256": patch_sha256,
        "derived_diff_sha256": actual_diff_sha256,
        "derived_initializer_exact": True,
        "copied_patch_and_diff_exact": True,
        "effective_pos_class_weight": effective_pos_class_weight,
        **pos_weight_provenance,
        "effective_config_sha256": _sha256(run / "config.yaml"),
        "test_or_predict_invoked": False,
        "raw_sha256": {name: _sha256(run / name) for name in raw_names},
        "timing": {**timing, **epoch_projection},
        "gpu": gpu_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--original-upstream", required=True, type=Path)
    parser.add_argument("--derived-upstream", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify(
        args.run_directory, args.original_upstream, args.derived_upstream, args.patch
    )
    _write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent released-weight result validation primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from collections.abc import Sequence
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
EXPECTED_WEIGHT_PATH = (
    REPRODUCTION_ROOT
    / ".local"
    / "released-weights"
    / "fold2_testAP0.571.pth"
)
EXPECTED_WEIGHT_SIZE = 57_889_221
EXPECTED_WEIGHT_SHA256 = "e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec"
EXPECTED_WEIGHT_ROOT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1-official-weight"
)


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_expected_weight_command(
    run_directory: Path, derived_root: Path, weight_path: Path
) -> list[str]:
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
        "--data.data_fold_id=2",
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
    if "test_AP" not in metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("released-weight finite test metrics are missing")
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
    authoritative_sources = {
        "upstream.lock.json": REPRODUCTION_ROOT / "upstream.lock.json",
        "smoke.json": EXPECTED_WORKER_SMOKE,
        "official_weight_entrypoint.py": Path(__file__).with_name(
            "official_weight_entrypoint.py"
        ),
        "evaluate_released_weight.py": Path(__file__).with_name(
            "evaluate_released_weight.py"
        ),
        "res18_import_scope.patch": patch,
    }
    recorded_hashes = effective.get("provenance_sha256")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError("weight effective provenance hash manifest is missing")
    validate_provenance_copies(
        run / "provenance", recorded_hashes, authoritative_sources, actual_diff
    )
    return {
        "original_upstream_commit": original_commit,
        "original_upstream_clean": True,
        "derived_upstream_commit": derived_commit,
        "derived_status": [" M src/models/__init__.py"],
        "patch_sha256": EXPECTED_PATCH_SHA256,
        "provenance_copies_verified": True,
    }


def _verify_full_dependency(preflight: Mapping[str, object]) -> dict[str, object]:
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
    if preflight.get("full_run_dependency") != expected:
        raise ValueError("weight preflight full-run dependency provenance mismatch")
    return expected


def verify_run(run_directory: Path, weight_path: Path) -> dict[str, object]:
    run = run_directory.resolve()
    weight = weight_path.resolve()
    if run.parent != EXPECTED_WEIGHT_ROOT.resolve():
        raise ValueError("released-weight run directory is outside the fixed artifact root")
    if weight != EXPECTED_WEIGHT_PATH.resolve():
        raise ValueError("released-weight path is not the pinned Fold-2 cache path")
    if weight.stat().st_size != EXPECTED_WEIGHT_SIZE or _sha256(
        weight
    ) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("released-weight bytes differ from the pinned Fold-2 weight")
    result = json.loads((run / "weight-result.json").read_text(encoding="utf-8"))
    preflight = json.loads((run / "preflight.json").read_text(encoding="utf-8"))
    effective = json.loads(
        (run / "effective-command.json").read_text(encoding="utf-8")
    )
    global_lock = json.loads(
        (run.parent / "fold2-weight-evaluation.lock.json").read_text(encoding="utf-8")
    )
    run_lock = json.loads((run / "launch.lock.json").read_text(encoding="utf-8"))
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    expected_command = build_expected_weight_command(
        run, EXPECTED_DERIVED_UPSTREAM, weight
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
    provenance = _verify_weight_runtime_provenance(run, effective, preflight)
    full_dependency = _verify_full_dependency(preflight)
    before = json.loads((run / "source-data-pre.json").read_text(encoding="utf-8"))
    after = json.loads((run / "source-data-post.json").read_text(encoding="utf-8"))
    live = rebuild_live_source_inventory(FIXED_DATA_ROOT)
    validate_source_inventory_lineage(before, after, live, FIXED_DATA_ROOT)
    output = (run / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (
        run / "stderr.log"
    ).read_text(encoding="utf-8", errors="replace")
    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    evidence = reconstruct_weight_evidence(output, exit_code)
    independently_recorded = {
        "status": "pass",
        **evidence,
        "filename_ap": 0.571,
        "weight_path": str(weight),
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "command_sha256": command_hash,
        "child_pid": started.get("pid"),
    }
    summary = validate_weight_result(independently_recorded)
    independently_recorded.update(summary)
    for field, expected in independently_recorded.items():
        if field == "loaded_tensor_count":
            continue
        if result.get(field) != expected:
            raise ValueError(f"released-weight result mismatch: {field}")
    if (
        completed.get("status") != "pass"
        or completed.get("exit_code") != 0
        or completed.get("pid") != started.get("pid")
    ):
        raise ValueError("released-weight completion marker mismatch")
    return {
        "status": "pass",
        "command_sha256": command_hash,
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "loaded_tensor_count": evidence["loaded_tensor_count"],
        "test_metrics": evidence["test_metrics"],
        "data_inventory_live_verified": True,
        "data_file_count": before["file_count"],
        "data_total_bytes": before["total_bytes"],
        "full_run_dependency": full_dependency,
        **summary,
        **provenance,
        "independent_implementation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--weight-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = verify_run(arguments.run_directory, arguments.weight_path)
    arguments.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

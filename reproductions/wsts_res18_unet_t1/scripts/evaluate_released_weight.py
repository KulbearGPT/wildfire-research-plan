"""Fetch and evaluate the pinned official Fold-2 Res18-U-Net raw state dict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from run_calibration import (
    DATA_ROOT,
    DERIVED_UPSTREAM_ROOT,
    ENVIRONMENT_PYTHON,
    EXPECTED_CODE_COMMIT,
    REPRODUCTION_ROOT,
    RUNTIME_PATCH_PATH,
    UPSTREAM_ROOT,
    WORKER_SMOKE_PATH,
    _build_child_environment,
    _gpu_preflight,
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
    summarize_gpu_samples,
    validate_train_validation_smoke_evidence,
    verify_inventory,
    verify_upstream,
    write_json_atomic,
)
from run_full_fold import FULL_ARTIFACTS_ROOT, parse_test_metrics
from verify_released_weight import validate_weight_result


REVISION = "acf70a37394849f4ec8d108a51d6f4325a554d0a"
REPO_ID = "saadlahrichi/WSTSPlus"
WEIGHT_PREFIX = "trained_model_weights/Res18Unet_T1/All/"
FOLD2_PATH = WEIGHT_PREFIX + "fold2_testAP0.571.pth"
FOLD2_SIZE = 57_889_221
FOLD2_SHA256 = "e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WEIGHT_CACHE = (
    REPOSITORY_ROOT
    / "reproductions"
    / "wsts_res18_unet_t1"
    / ".local"
    / "released-weights"
    / "fold2_testAP0.571.pth"
)
WEIGHT_ARTIFACTS_ROOT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "reproductions"
    / "wsts-res18-t1-official-weight"
)
GLOBAL_WEIGHT_LOCK = WEIGHT_ARTIFACTS_ROOT / "fold2-weight-evaluation.lock.json"
RELEASED_WEIGHT_FILENAMES = (
    "fold0_testAP0.528.pth",
    "fold1_testAP0.426.pth",
    "fold2_testAP0.571.pth",
    "fold3_testAP0.307.pth",
    "fold4_testAP0.483.pth",
    "fold5_testAP0.322.pth",
    "fold6_testAP0.577.pth",
    "fold7_testAP0.474.pth",
    "fold8_testAP0.478.pth",
    "fold9_testAP0.471.pth",
    "fold10_testAP0.324.pth",
    "fold11_testAP0.474.pth",
)
FILENAME_PROVENANCE = (
    "derived only from filenames in the pinned official All/T=1 weight manifest; "
    "not provenance for the paper table"
)


def summarize_filename_aps(filenames: Sequence[str]) -> dict[str, float | int]:
    values: list[float] = []
    folds: set[int] = set()
    for name in filenames:
        match = re.fullmatch(r"fold(\d+)_testAP(0\.\d+)\.pth", name)
        if match is None:
            raise ValueError(f"released weight filename is malformed: {name}")
        fold = int(match.group(1))
        if fold in folds:
            raise ValueError("released weight filenames do not contain unique folds")
        folds.add(fold)
        values.append(float(match.group(2)))
    if folds != set(range(12)):
        raise ValueError("released weight filenames must cover folds 0 through 11")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
    }


def build_filename_manifest_evidence(
    filenames: Sequence[str] = RELEASED_WEIGHT_FILENAMES,
) -> dict[str, object]:
    ordered = sorted(
        filenames,
        key=lambda name: int(re.fullmatch(r"fold(\d+)_testAP0\.\d+\.pth", name).group(1))
        if re.fullmatch(r"fold(\d+)_testAP0\.\d+\.pth", name)
        else -1,
    )
    aggregate = summarize_filename_aps(ordered)
    return {
        "revision": REVISION,
        "prefix": WEIGHT_PREFIX,
        "filenames": ordered,
        "aggregate": aggregate,
        "paper_table_provenance": False,
        "provenance_note": FILENAME_PROVENANCE,
    }


def validate_weight_manifest(
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    matches = [
        item
        for item in items
        if item.get("type") == "file"
        and isinstance(item.get("path"), str)
        and str(item["path"]).startswith(WEIGHT_PREFIX)
    ]
    if len(matches) != 12:
        raise ValueError("All/T=1 manifest must contain exactly twelve released weights")
    filenames = [Path(str(item["path"])).name for item in matches]
    try:
        aggregate = summarize_filename_aps(filenames)
    except ValueError as error:
        if "unique folds" in str(error):
            raise
        raise ValueError(f"released weight manifest is invalid: {error}") from error
    filename_manifest = build_filename_manifest_evidence(filenames)
    if filename_manifest["filenames"] != list(RELEASED_WEIGHT_FILENAMES):
        raise ValueError("released weight filenames differ from the pinned manifest")
    selected = [item for item in matches if item.get("path") == FOLD2_PATH]
    if len(selected) != 1:
        raise ValueError("Fold 2 weight is missing or ambiguous")
    item = selected[0]
    lfs = item.get("lfs")
    sha = lfs.get("oid") if isinstance(lfs, Mapping) else None
    if item.get("size") != FOLD2_SIZE or sha != FOLD2_SHA256:
        raise ValueError("Fold 2 weight size or LFS SHA-256 differs from the pinned revision")
    return {
        "path": FOLD2_PATH,
        "size": FOLD2_SIZE,
        "lfs_sha256": FOLD2_SHA256,
        "filename_ap": 0.571,
        "filename_aggregate": aggregate,
        "filename_manifest": filename_manifest,
    }


def validate_download(
    path: Path, *, expected_size: int = FOLD2_SIZE, expected_sha256: str = FOLD2_SHA256
) -> None:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("released weight download size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("released weight download SHA-256 mismatch")


def query_manifest() -> list[dict[str, object]]:
    url = (
        f"https://huggingface.co/api/models/{REPO_ID}/tree/{REVISION}"
        "?recursive=true&expand=true&limit=100"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("Hugging Face tree response is not a list")
    return payload


def fetch_weight(target: Path = WEIGHT_CACHE) -> dict[str, object]:
    selected = validate_weight_manifest(query_manifest())
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_download(target)
        return {**selected, "local_path": str(target.resolve()), "download_reused": True}
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in FOLD2_PATH.split("/"))
    url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{encoded}?download=true"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        validate_download(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**selected, "local_path": str(target.resolve()), "download_reused": False}


def build_weight_command(
    *,
    run_directory: Path,
    weight_path: Path,
    python_executable: Path = ENVIRONMENT_PYTHON,
    entrypoint: Path = Path(__file__).with_name("official_weight_entrypoint.py"),
    upstream_root: Path = DERIVED_UPSTREAM_ROOT,
    data_root: Path = DATA_ROOT,
) -> list[str]:
    upstream = upstream_root.resolve()
    configs = upstream / "cfgs"
    return [
        str(python_executable.resolve()),
        str(entrypoint.resolve()),
        "--upstream-root",
        str(upstream),
        "--weights-path",
        str(weight_path.resolve()),
        f"--config={(configs / 'unet' / 'res18_monotemporal.yaml').as_posix()}",
        f"--trainer={(configs / 'trainer_single_gpu.yaml').as_posix()}",
        f"--data={(configs / 'data_monotemporal_full_features.yaml').as_posix()}",
        f"--data.data_dir={data_root.resolve()}",
        "--data.data_fold_id=2",
        "--data.features_to_keep=null",
        "--data.n_leading_observations=1",
        "--data.remove_duplicate_features=true",
        "--data.num_workers=8",
        f"--trainer.default_root_dir={run_directory.resolve()}",
        "--do_train=false",
        "--do_test=false",
    ]


def validate_weight_command(
    command: Sequence[str], run_directory: Path, weight_path: Path
) -> None:
    if len(command) < 6:
        raise ValueError("effective command differs from the exact released-weight command")
    expected = build_weight_command(
        run_directory=run_directory,
        weight_path=weight_path,
        python_executable=Path(command[0]),
        entrypoint=Path(command[1]),
        upstream_root=Path(command[3]),
        data_root=DATA_ROOT,
    )
    if list(command) != expected:
        raise ValueError("effective command differs from the exact released-weight command")


def validate_weight_output(combined_output: str, exit_code: int) -> dict[str, object]:
    if exit_code != 0:
        raise ValueError(f"released-weight child exit code is not zero: {exit_code}")
    strict = re.findall(
        r"(?m)^WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=(\d+)\s*$",
        combined_output,
    )
    if len(strict) != 1 or int(strict[0]) <= 0:
        raise ValueError("released-weight strict-load evidence is missing or ambiguous")
    if re.search(r"Testing DataLoader 0:\s*100%", combined_output) is None:
        raise ValueError("released-weight test completion evidence is missing")
    if re.search(r"\bEpoch\s+\d+:|Trainer\.fit|max_steps=", combined_output):
        raise ValueError("released-weight evaluation invoked training")
    if "Validation DataLoader" in combined_output:
        raise ValueError("released-weight evaluation invoked validation")
    if re.search(r"Predicting DataLoader|trainer\.predict", combined_output):
        raise ValueError("released-weight evaluation invoked predict")
    return {
        "strict_load": True,
        "loaded_tensor_count": int(strict[0]),
        "test_metrics": parse_test_metrics(combined_output),
        "peak_allocated_bytes": parse_peak_allocated(combined_output),
    }


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    return result


def _strict_load_preflight(command: Sequence[str]) -> dict[str, object]:
    strict_command = [
        command[0],
        command[1],
        "--strict-load-only",
        *command[2:],
    ]
    child_environment = dict(os.environ)
    child_environment.update(
        {
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = _run_checked(
        strict_command, cwd=DERIVED_UPSTREAM_ROOT, environment=child_environment
    )
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise ValueError("released-weight strict-load preflight failed: " + combined[-2000:])
    if len(re.findall(r"WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=\d+", combined)) != 1:
        raise ValueError("strict-load preflight lacks exact load evidence")
    if re.search(r"Testing DataLoader|Validation DataLoader|Predicting DataLoader|\bEpoch\s+\d+:", combined):
        raise ValueError("strict-load preflight invoked a data or training action")
    return {
        "status": "pass",
        "command": strict_command,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        "test_loader_invoked": False,
    }


def validate_full_run_dependency(lock_path: Path) -> dict[str, object]:
    """Require controller completion and a matching independent PASS artifact."""
    if not lock_path.is_file():
        raise ValueError("full Fold-2 run has not been launched")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    run = Path(str(lock.get("run_directory", ""))).resolve()
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    result = json.loads((run / "full-result.json").read_text(encoding="utf-8"))
    independent_path = run / "independent-verification.json"
    if not independent_path.is_file():
        raise ValueError("full Fold-2 independent verification is missing")
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    if (
        completed.get("status") != "pass"
        or completed.get("exit_code") != 0
        or result.get("status") != "pass"
        or result.get("optimizer_steps") != 10_000
    ):
        raise ValueError("full Fold-2 run is not complete")
    if independent.get("status") != "pass" or independent.get(
        "optimizer_steps"
    ) != 10_000:
        raise ValueError("full Fold-2 independent verification did not pass")
    for field in ("command_sha256", "checkpoint_sha256"):
        if independent.get(field) != result.get(field):
            raise ValueError(f"full Fold-2 independent {field} mismatch")
    return {
        "run_directory": str(run),
        "result_sha256": _sha256(run / "full-result.json"),
        "independent_sha256": _sha256(independent_path),
        "independent_status": "pass",
    }


def _full_run_complete() -> dict[str, object]:
    return validate_full_run_dependency(
        FULL_ARTIFACTS_ROOT / "fold2-full-launch.lock.json"
    )


def _create_run_directory() -> Path:
    WEIGHT_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    run = WEIGHT_ARTIFACTS_ROOT / f"fold2-weight-{stamp}-{uuid.uuid4().hex[:8]}"
    run.mkdir()
    return run.resolve()


def _preflight(
    run_directory: Path, *, require_full_complete: bool, require_lock_absent: bool = True
) -> tuple[dict[str, object], dict[str, object]]:
    if require_lock_absent and GLOBAL_WEIGHT_LOCK.exists():
        raise FileExistsError(f"released-weight launch lock already exists: {GLOBAL_WEIGHT_LOCK}")
    validate_download(WEIGHT_CACHE)
    repo_status = _run_checked(["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT)
    if repo_status.returncode != 0 or repo_status.stdout.strip():
        raise ValueError("tracked reproduction worktree must be clean before weight evaluation")
    smoke = json.loads(WORKER_SMOKE_PATH.read_text(encoding="utf-8"))
    validate_train_validation_smoke_evidence(smoke)
    verify_inventory(DATA_ROOT, selected_years=(2018, 2019, 2020, 2021))
    inventory = build_source_inventory(DATA_ROOT)
    if inventory["file_count"] != 607 or inventory["total_bytes"] != 24_242_259_023:
        raise ValueError("live WSTS source inventory differs from the frozen set")
    verify_upstream(UPSTREAM_ROOT, EXPECTED_CODE_COMMIT)
    runtime_patch = _prepare_derived_runtime()
    runtime = _runtime_preflight()
    gpu = _gpu_preflight()
    command = build_weight_command(run_directory=run_directory, weight_path=WEIGHT_CACHE)
    validate_weight_command(command, run_directory, WEIGHT_CACHE)
    strict_load = _strict_load_preflight(command)
    full_dependency = _full_run_complete() if require_full_complete else {"status": "deferred"}
    return (
        {
            "status": "pass",
            "purpose": "test-only evaluation of pinned official Fold-2 raw state dict",
            "weight_path": str(WEIGHT_CACHE.resolve()),
            "weight_sha256": FOLD2_SHA256,
            "filename_ap": 0.571,
            "filename_manifest": build_filename_manifest_evidence(),
            "runtime": runtime,
            "gpu": gpu,
            "runtime_patch": runtime_patch,
            "strict_load_preflight": strict_load,
            "full_run_dependency": full_dependency,
            "command": command,
            "command_sha256": _command_sha256(command),
            "data_file_count": inventory["file_count"],
            "data_total_bytes": inventory["total_bytes"],
        },
        inventory,
    )


def _copy_provenance(run: Path, preflight: Mapping[str, object]) -> dict[str, str]:
    import shutil

    provenance = run / "provenance"
    provenance.mkdir()
    sources = {
        "upstream.lock.json": REPRODUCTION_ROOT / "upstream.lock.json",
        "smoke.json": WORKER_SMOKE_PATH,
        "official_weight_entrypoint.py": Path(__file__).with_name(
            "official_weight_entrypoint.py"
        ),
        "evaluate_released_weight.py": Path(__file__),
        "res18_import_scope.patch": RUNTIME_PATCH_PATH,
    }
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        target = provenance / name
        shutil.copy2(source, target)
        hashes[name] = _sha256(target)
    diff = provenance / "derived-runtime.diff"
    diff.write_text(str(preflight["runtime_patch"]["git_diff"]), encoding="utf-8")
    hashes[diff.name] = _sha256(diff)
    return hashes


def parse_weight_run(run_directory: Path) -> dict[str, object]:
    run = run_directory.resolve()
    stdout = (run / "stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr = (run / "stderr.log").read_text(encoding="utf-8", errors="replace")
    exit_code = int((run / "exit-code.txt").read_text(encoding="utf-8").strip())
    evidence = validate_weight_output(stdout + "\n" + stderr, exit_code)
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    wall = measure_wall_seconds_from_markers(
        str(started["started_utc"]), (run / "exit-code.txt").stat().st_mtime_ns
    )
    with (run / "gpu.csv").open(encoding="utf-8", newline="") as handle:
        gpu_rows = list(__import__("csv").DictReader(handle))
    before = json.loads((run / "source-data-pre.json").read_text(encoding="utf-8"))
    after = json.loads((run / "source-data-post.json").read_text(encoding="utf-8"))
    assert_source_inventories_identical(before, after)
    payload: dict[str, object] = {
        "status": "pass",
        "exit_code": exit_code,
        "strict_load": evidence["strict_load"],
        "train_invoked": False,
        "validation_invoked": False,
        "predict_invoked": False,
        "test_metrics": evidence["test_metrics"],
        "filename_ap": 0.571,
        "filename_manifest": build_filename_manifest_evidence(),
        "weight_path": str(WEIGHT_CACHE.resolve()),
        "weight_sha256": FOLD2_SHA256,
        "wall_seconds": wall,
        "peak_allocated_bytes": evidence["peak_allocated_bytes"],
        **summarize_gpu_samples(gpu_rows),
    }
    payload.update(validate_weight_result(payload))
    return payload


def finalize_existing_weight(run_directory: Path) -> dict[str, object]:
    run = run_directory.resolve()
    if run.parent != WEIGHT_ARTIFACTS_ROOT.resolve() or not run.name.startswith(
        "fold2-weight-"
    ):
        raise ValueError("finalize-existing weight run is outside the fixed artifact root")
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if (
        not isinstance(completed, Mapping)
        or completed.get("status") != "pass"
        or completed.get("exit_code") != 0
        or int((run / "exit-code.txt").read_text(encoding="utf-8").strip()) != 0
    ):
        raise ValueError("finalize-existing requires a completed exit-zero weight child")
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    effective = json.loads((run / "effective-command.json").read_text(encoding="utf-8"))
    if (
        not isinstance(started, Mapping)
        or not isinstance(effective, Mapping)
        or type(started.get("pid")) is not int
        or started["pid"] != completed.get("pid")
        or not isinstance(started.get("command_sha256"), str)
        or started["command_sha256"] != effective.get("command_sha256")
    ):
        raise ValueError("finalize-existing weight launch lineage mismatch")
    result = parse_weight_run(run)
    result["command_sha256"] = started["command_sha256"]
    result["child_pid"] = started["pid"]
    if result.get("filename_manifest") != build_filename_manifest_evidence():
        raise ValueError("finalize-existing filename manifest reconstruction failed")
    write_json_atomic(run / "weight-result.json", result)
    preflight_path = run / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("filename_manifest") != result["filename_manifest"]:
        write_json_atomic(
            run / "offline-preflight-augmentation.json",
            {
                "status": "pass",
                "augmentation_scope": "offline filename-manifest provenance only",
                "launch_preflight_sha256": _sha256(preflight_path),
                "filename_manifest": result["filename_manifest"],
                "scientific_child_relaunched": False,
            },
        )
    launch_dependency = preflight.get("full_run_dependency")
    if isinstance(launch_dependency, Mapping):
        current_dependency = _full_run_complete()
        if launch_dependency != current_dependency:
            write_json_atomic(
                run / "offline-full-dependency-augmentation.json",
                {
                    "status": "pass",
                    "augmentation_scope": (
                        "offline current full-run dependency hashes only"
                    ),
                    "launch_recorded_dependency": launch_dependency,
                    "current_dependency": current_dependency,
                    "scientific_child_relaunched": False,
                },
            )
    write_json_atomic(
        run / "offline-finalization.json",
        {
            "status": "pass",
            "finalization_scope": "existing scientific output only",
            "weight_result_sha256": _sha256(run / "weight-result.json"),
            "scientific_child_relaunched": False,
        },
    )
    return result


def _launch_once() -> tuple[Path, dict[str, object]]:
    run = _create_run_directory()
    try:
        preflight, source_before = _preflight(run, require_full_complete=True)
        write_json_atomic(run / "preflight.json", preflight)
        write_json_atomic(run / "source-data-pre.json", source_before)
        provenance = _copy_provenance(run, preflight)
        command = list(preflight["command"])
        command_hash = str(preflight["command_sha256"])
        child_environment, recorded_environment = _build_child_environment(
            run, preflight["runtime"]
        )
        launch = {
            "command": command,
            "command_sha256": command_hash,
            "run_directory": str(run),
            "test_only": True,
            "single_launch_no_retry": True,
        }
        acquire_launch_lock(GLOBAL_WEIGHT_LOCK, launch)
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
        print(json.dumps({"status": "launching-test-only", "run_directory": str(run)}), flush=True)
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
        result = parse_weight_run(run)
        result["command_sha256"] = command_hash
        result["child_pid"] = observation["pid"]
        write_json_atomic(run / "weight-result.json", result)
        write_json_atomic(run / "completed.json", {"status": "pass", "pid": observation["pid"], "exit_code": 0})
        return run, result
    except BaseException as error:
        write_json_atomic(
            run / "failure.json",
            {
                "status": "fail",
                "error": f"{type(error).__name__}: {error}",
                "evaluation_retry_performed": False,
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--fetch-only", action="store_true")
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--finalize-existing", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.fetch_only:
        result = fetch_weight()
    elif arguments.preflight_only:
        placeholder = WEIGHT_ARTIFACTS_ROOT / "preflight-placeholder"
        placeholder.mkdir(parents=True, exist_ok=True)
        result, _ = _preflight(placeholder, require_full_complete=False)
    elif arguments.finalize_existing is not None:
        result = finalize_existing_weight(arguments.finalize_existing)
    else:
        run, result = _launch_once()
        result = {"run_directory": str(run), **result}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

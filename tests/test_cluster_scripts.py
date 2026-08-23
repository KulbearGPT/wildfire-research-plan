import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cluster" / "clusterctl.py"
SCRIPTS = SCRIPT.parent


def load_clusterctl():
    spec = importlib.util.spec_from_file_location("clusterctl_scripts", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("clusterctl module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Test"], check=True)
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", path, "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", path, "commit", "-qm", "initial"], check=True)


def write_profile(tmp_path: Path, data_root: Path) -> Path:
    for name in ("runs", "caches", "logs"):
        (tmp_path / name).mkdir(exist_ok=True)
    activation = tmp_path / "env" / "bin" / "activate"
    activation.parent.mkdir(parents=True, exist_ok=True)
    activation.write_text("# activation fixture\n", encoding="utf-8")
    values = {
        "CLUSTER_PROFILE_SCHEMA": "1",
        "SLURM_CPU_ACCOUNT": "cpu-project",
        "SLURM_GPU_ACCOUNT": "gpu-project",
        "SLURM_CPU_PARTITION": "cpu-short",
        "SLURM_GPU_PARTITION": "gpu-short",
        "SLURM_GPU_REQUEST": "--gpus=h100:1",
        "SLURM_ARRAY_CONCURRENCY": "4",
        "MODULES": "python/3.11,cuda/12.2",
        "ENV_ACTIVATE": activation.as_posix(),
        "PROJECT_ROOT": tmp_path.as_posix(),
        "DATA_ROOT": data_root.as_posix(),
        "RUNS_ROOT": (tmp_path / "runs").as_posix(),
        "CACHE_ROOT": (tmp_path / "caches").as_posix(),
        "LOG_ROOT": (tmp_path / "logs").as_posix(),
        "SCRATCH_ENV": "SLURM_TMPDIR",
        "SMOKE_TIME": "00:30:00",
        "CALIBRATION_TIME": "03:00:00",
        "FULL_TIME": "12:00:00",
    }
    profile = tmp_path / "profile.env"
    profile.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return profile


def make_data_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "hdf5" / "2016"
    root.mkdir(parents=True)
    (root / "event.hdf5").write_bytes(b"event")
    return root.parent


def test_wrappers_are_thin_lf_posix_files() -> None:
    for name in ("bootstrap.sh", "preflight.sh", "submit.sh", "job.sh", "collect.sh"):
        raw = (SCRIPTS / name).read_bytes()
        assert raw.startswith(b"#!/usr/bin/env bash\n")
        assert b"\r\n" not in raw
        assert b"set -euo pipefail" in raw
        assert len(raw.splitlines()) <= 80


def test_job_has_no_retry_or_scientific_override() -> None:
    text = (SCRIPTS / "job.sh").read_text(encoding="utf-8")
    assert "retry" not in text.lower()
    for forbidden in (
        "max_steps",
        "batch_size",
        "learning_rate",
        "fold_id",
        "features_to_keep",
    ):
        assert forbidden not in text
    assert "set +e" in text
    assert "exit_code=$?" in text


@pytest.mark.parametrize("name", ["bootstrap.sh", "job.sh"])
def test_module_load_precedes_first_python_call(name: str) -> None:
    text = (SCRIPTS / name).read_text(encoding="utf-8")

    assert text.index("module load") < text.index("python3")
    assert "profile validate" in text


def test_run_finish_writes_one_terminal_marker_and_preserves_start(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    data_root = make_data_fixture(tmp_path)
    profile = write_profile(tmp_path, data_root)
    run_dir = tmp_path / "run"

    ctl.write_run_start(
        run_dir,
        profile,
        ["python", "train.py"],
        {"SLURM_JOB_ID": "42", "SLURM_ARRAY_TASK_ID": "3"},
        repo_root=repo,
    )
    before = (run_dir / "started.json").read_bytes()
    terminal = ctl.write_run_finish(run_dir, 0)

    assert terminal["status"] == "completed"
    assert terminal["exit_code"] == 0
    assert terminal["started_sha256"] == hashlib.sha256(before).hexdigest()
    assert (run_dir / "completed.json").is_file()
    assert not (run_dir / "failure.json").exists()
    assert (run_dir / "started.json").read_bytes() == before


def test_nonzero_finish_is_failure_and_terminal_is_immutable(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    profile = write_profile(tmp_path, make_data_fixture(tmp_path))
    run_dir = tmp_path / "run"
    ctl.write_run_start(run_dir, profile, ["python", "fail.py"], {}, repo_root=repo)

    assert ctl.write_run_finish(run_dir, 7)["status"] == "failure"
    assert (run_dir / "failure.json").is_file()
    with pytest.raises(ValueError, match="terminal"):
        ctl.write_run_finish(run_dir, 0)


def test_run_start_rejects_dirty_formal_checkout(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    profile = write_profile(tmp_path, make_data_fixture(tmp_path))

    with pytest.raises(ValueError, match="dirty"):
        ctl.write_run_start(tmp_path / "run", profile, ["python", "x.py"], {}, repo_root=repo)


def test_finish_rejects_profile_mutation(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    profile = write_profile(tmp_path, make_data_fixture(tmp_path))
    run_dir = tmp_path / "run"
    ctl.write_run_start(run_dir, profile, ["python", "x.py"], {}, repo_root=repo)
    profile.write_text(profile.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="profile"):
        ctl.write_run_finish(run_dir, 0)
    assert not (run_dir / "completed.json").exists()
    assert not (run_dir / "failure.json").exists()


def test_preflight_verifies_manifest_roots_git_and_gpu(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    data_root = make_data_fixture(tmp_path)
    profile = write_profile(tmp_path, data_root)
    csv_path = tmp_path / "manifest.csv"
    summary_path = tmp_path / "summary.json"
    ctl.write_hdf5_manifest(data_root, csv_path, summary_path)
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n", stderr="")

    result = ctl.preflight(
        profile,
        csv_path,
        summary_path,
        require_gpu=True,
        repo_root=repo,
        runner=runner,
    )

    assert result["status"] == "pass"
    assert result["file_count"] == 1
    assert calls == [["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]]


def test_preflight_rejects_missing_manifest_and_no_cuda_device(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    data_root = make_data_fixture(tmp_path)
    profile = write_profile(tmp_path, data_root)

    with pytest.raises(ValueError, match="manifest"):
        ctl.preflight(profile, tmp_path / "missing.csv", tmp_path / "missing.json", repo_root=repo)

    csv_path = tmp_path / "manifest.csv"
    summary_path = tmp_path / "summary.json"
    ctl.write_hdf5_manifest(data_root, csv_path, summary_path)
    runner = lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(ValueError, match="CUDA"):
        ctl.preflight(
            profile,
            csv_path,
            summary_path,
            require_gpu=True,
            repo_root=repo,
            runner=runner,
        )


def test_preflight_rejects_invalid_output_root(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    data_root = make_data_fixture(tmp_path)
    profile = write_profile(tmp_path, data_root)
    text = profile.read_text(encoding="utf-8")
    bad_root = tmp_path / "not-a-directory"
    bad_root.write_text("file\n", encoding="utf-8")
    profile.write_text(
        text.replace(f"RUNS_ROOT={(tmp_path / 'runs').as_posix()}", f"RUNS_ROOT={bad_root.as_posix()}"),
        encoding="utf-8",
    )
    csv_path = tmp_path / "manifest.csv"
    summary_path = tmp_path / "summary.json"
    ctl.write_hdf5_manifest(data_root, csv_path, summary_path)

    with pytest.raises(ValueError, match="RUNS_ROOT"):
        ctl.preflight(profile, csv_path, summary_path, repo_root=repo)


def test_collection_packs_only_declared_small_json_csv_files(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metric = run_dir / "metrics.json"
    table = run_dir / "summary.csv"
    checkpoint = run_dir / "model.ckpt"
    metric.write_text('{"ap":0.5}\n', encoding="utf-8")
    table.write_text("fold,ap\n0,0.5\n", encoding="utf-8")
    checkpoint.write_bytes(b"large-model-placeholder")
    output = run_dir / "collection.json"

    result = ctl.collect_results(run_dir, output, [metric, table])

    assert result["status"] == "pass"
    assert result["files"] == ["metrics.json", "summary.csv"]
    assert (run_dir / "results.tar.gz").is_file()
    assert checkpoint.is_file()
    with pytest.raises(ValueError, match="existing collection"):
        ctl.collect_results(run_dir, output, [metric])


@pytest.mark.parametrize("bad_name", ["model.ckpt", "nested", "outside.json"])
def test_collection_rejects_checkpoint_directory_and_escape(tmp_path: Path, bad_name: str) -> None:
    ctl = load_clusterctl()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if bad_name == "nested":
        bad_path = run_dir / "nested"
        bad_path.mkdir()
    elif bad_name == "outside.json":
        bad_path = tmp_path / bad_name
        bad_path.write_text("{}\n", encoding="utf-8")
    else:
        bad_path = run_dir / bad_name
        bad_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError):
        ctl.collect_results(run_dir, run_dir / "collection.json", [bad_path])
    assert not (run_dir / "results.tar.gz").exists()

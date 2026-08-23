import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cluster" / "clusterctl.py"


def load_clusterctl():
    spec = importlib.util.spec_from_file_location("clusterctl", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("clusterctl module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def formal_profile_values() -> dict[str, str]:
    return {
        "CLUSTER_PROFILE_SCHEMA": "1",
        "SLURM_CPU_ACCOUNT": "cpu-project",
        "SLURM_GPU_ACCOUNT": "gpu-project",
        "SLURM_CPU_PARTITION": "cpu-short",
        "SLURM_GPU_PARTITION": "gpu-short",
        "SLURM_GPU_REQUEST": "--gpus=h100:1",
        "SLURM_ARRAY_CONCURRENCY": "4",
        "MODULES": "StdEnv/2023,gcc/12.3,cuda/12.2,python/3.11.5",
        "ENV_ACTIVATE": "/project/team/wildfire/envs/training/bin/activate",
        "PROJECT_ROOT": "/project/team/wildfire",
        "DATA_ROOT": "/project/team/wildfire/data/hdf5",
        "RUNS_ROOT": "/project/team/wildfire/runs",
        "CACHE_ROOT": "/project/team/wildfire/caches",
        "LOG_ROOT": "/project/team/wildfire/logs",
        "SCRATCH_ENV": "SLURM_TMPDIR",
        "SMOKE_TIME": "00:30:00",
        "CALIBRATION_TIME": "03:00:00",
        "FULL_TIME": "12:00:00",
    }


def write_profile(
    tmp_path: Path,
    values: dict[str, str],
    *,
    extra_lines: tuple[str, ...] = (),
) -> Path:
    path = tmp_path / "profile.env"
    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join([*lines, *extra_lines]) + "\n", encoding="utf-8")
    return path


def test_example_profile_is_complete_but_contains_only_placeholders() -> None:
    ctl = load_clusterctl()

    profile = ctl.parse_profile(
        ROOT / "configs" / "cluster" / "profile.example.env",
        allow_placeholders=True,
    )

    assert profile["CLUSTER_PROFILE_SCHEMA"] == "1"
    assert profile["SLURM_ARRAY_CONCURRENCY"] == "4"
    assert profile["SLURM_GPU_REQUEST"] == "--gpus=h100:1"
    assert profile["SLURM_GPU_ACCOUNT"] == "replace-me"


@pytest.mark.parametrize("bad_value", ["", "replace-me", "<GPU_ACCOUNT>"])
def test_formal_profile_rejects_placeholder_account(
    tmp_path: Path, bad_value: str
) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    profile["SLURM_GPU_ACCOUNT"] = bad_value

    with pytest.raises(ValueError, match="SLURM_GPU_ACCOUNT"):
        ctl.parse_profile(write_profile(tmp_path, profile))


@pytest.mark.parametrize(
    ("extra_lines", "message"),
    [
        (("SLURM_GPU_ACCOUNT=duplicate",), "duplicate"),
        (("UNKNOWN_SETTING=value",), "unknown"),
        (("not-an-assignment",), "malformed"),
    ],
)
def test_profile_rejects_duplicate_unknown_or_malformed_lines(
    tmp_path: Path, extra_lines: tuple[str, ...], message: str
) -> None:
    ctl = load_clusterctl()

    with pytest.raises(ValueError, match=message):
        ctl.parse_profile(
            write_profile(tmp_path, formal_profile_values(), extra_lines=extra_lines)
        )


@pytest.mark.parametrize("value", ["0", "5", "not-an-int"])
def test_profile_rejects_array_concurrency_outside_available_gpu_range(
    tmp_path: Path, value: str
) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    profile["SLURM_ARRAY_CONCURRENCY"] = value

    with pytest.raises(ValueError, match="SLURM_ARRAY_CONCURRENCY"):
        ctl.parse_profile(write_profile(tmp_path, profile))


@pytest.mark.parametrize(
    "gpu_request",
    [
        "--gres=gpu:h100:1,gpu:h100:1",
        "--gpus=h100:2",
        "--gres=gpu:h100:1,license:foo:1",
    ],
)
def test_profile_rejects_multi_resource_gpu_requests(
    tmp_path: Path, gpu_request: str
) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    profile["SLURM_GPU_REQUEST"] = gpu_request

    with pytest.raises(ValueError, match="exactly one GPU"):
        ctl.parse_profile(write_profile(tmp_path, profile))


def test_profile_rejects_relative_operational_root(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    profile["RUNS_ROOT"] = "relative/runs"

    with pytest.raises(ValueError, match="RUNS_ROOT.*absolute"):
        ctl.parse_profile(write_profile(tmp_path, profile))


@pytest.mark.parametrize("key", ["RUNS_ROOT", "CACHE_ROOT", "LOG_ROOT"])
def test_profile_rejects_mutable_root_inside_data(
    tmp_path: Path, key: str
) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    profile[key] = profile["DATA_ROOT"] + "/mutable"

    with pytest.raises(ValueError, match=f"{key}.*DATA_ROOT"):
        ctl.parse_profile(write_profile(tmp_path, profile))


def test_render_uses_one_gpu_and_bounded_array_without_running_sbatch(
    tmp_path: Path,
) -> None:
    ctl = load_clusterctl()
    path = write_profile(tmp_path, formal_profile_values())
    profile = ctl.parse_profile(path)

    command = ctl.render_submit_command(
        profile,
        profile_path=path,
        job_name="folds",
        time_class="full",
        command=["python", "train.py", "--seed", "0"],
        array="0-11",
    )

    assert command == [
        "sbatch",
        "--parsable",
        "--job-name=folds",
        "--account=gpu-project",
        "--partition=gpu-short",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=96G",
        "--gpus=h100:1",
        "--time=12:00:00",
        "--output=/project/team/wildfire/logs/%x-%A_%a.out",
        "--error=/project/team/wildfire/logs/%x-%A_%a.err",
        "--array=0-11%4",
        str(ROOT / "scripts" / "cluster" / "job.sh"),
        str(path.resolve()),
        "--",
        "python",
        "train.py",
        "--seed",
        "0",
    ]
    assert not any("h100:2" in token for token in command)


@pytest.mark.parametrize("time_class", ["", "week", "FULL"])
def test_render_rejects_unknown_time_class(tmp_path: Path, time_class: str) -> None:
    ctl = load_clusterctl()
    path = write_profile(tmp_path, formal_profile_values())
    profile = ctl.parse_profile(path)

    with pytest.raises(ValueError, match="time class"):
        ctl.render_submit_command(
            profile,
            profile_path=path,
            job_name="smoke",
            time_class=time_class,
            command=["python", "smoke.py"],
        )


@pytest.mark.parametrize("array", ["1", "3-2", "0-2%8", "0,1"])
def test_render_rejects_noncanonical_array_ranges(
    tmp_path: Path, array: str
) -> None:
    ctl = load_clusterctl()
    path = write_profile(tmp_path, formal_profile_values())
    profile = ctl.parse_profile(path)

    with pytest.raises(ValueError, match="array"):
        ctl.render_submit_command(
            profile,
            profile_path=path,
            job_name="folds",
            time_class="full",
            command=["python", "train.py"],
            array=array,
        )


def test_submit_dry_run_prints_command_without_calling_sbatch(tmp_path: Path) -> None:
    path = write_profile(tmp_path, formal_profile_values())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "submit",
            str(path),
            "--job-name",
            "smoke",
            "--time-class",
            "smoke",
            "--dry-run",
            "--",
            "python",
            "smoke.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.startswith("sbatch --parsable --job-name=smoke ")
    assert "--gpus=h100:1" in result.stdout
    assert result.stdout.endswith("-- python smoke.py\n")


def test_submit_cli_rejects_missing_scientific_command(tmp_path: Path) -> None:
    path = write_profile(tmp_path, formal_profile_values())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "submit",
            str(path),
            "--job-name",
            "empty",
            "--time-class",
            "smoke",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "scientific command" in result.stderr

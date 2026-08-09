import json
import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = (
    REPOSITORY_ROOT
    / "reproductions"
    / "wsts_res18_unet_t1"
    / "scripts"
    / "bootstrap.ps1"
)
FIXED_ENVIRONMENT = Path(r"D:\WildFire Project\.conda-envs\wsts-res18-t1")


def _powershell(*arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bootstrap_command_has_no_environment_prefix_override() -> None:
    escaped_path = str(BOOTSTRAP).replace("'", "''")
    result = _powershell(
        "-Command",
        f"$command = Get-Command -Name '{escaped_path}'; "
        "$command.Parameters.Keys | ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    parameters = json.loads(result.stdout)
    assert "EnvironmentPrefix" not in parameters
    assert "ValidateEnvironmentOnly" in parameters
    assert "ValidateCheckoutOnly" in parameters
    assert "ValidateRuntimeOnly" in parameters


def test_environment_preflight_rejects_the_target_prefix_when_it_is_active() -> None:
    environment = dict(os.environ)
    environment["CONDA_PREFIX"] = str(FIXED_ENVIRONMENT)

    result = _powershell(
        "-File",
        str(BOOTSTRAP),
        "-ValidateEnvironmentOnly",
        environment=environment,
    )

    assert result.returncode != 0
    assert "active Conda environment" in result.stderr


def test_environment_preflight_verifies_fixed_prefix_and_exact_python() -> None:
    environment = dict(os.environ)
    if Path(environment.get("CONDA_PREFIX", "")).resolve() == FIXED_ENVIRONMENT.resolve():
        environment.pop("CONDA_PREFIX")

    result = _powershell(
        "-File",
        str(BOOTSTRAP),
        "-ValidateEnvironmentOnly",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(payload["environment_prefix"]).resolve() == FIXED_ENVIRONMENT.resolve()
    assert payload["python_version"] == "3.10.4"
    assert Path(payload["python_prefix"]).resolve() == FIXED_ENVIRONMENT.resolve()


def test_checkout_preflight_rejects_nonofficial_origin(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "remote", "add", "origin", "https://example.invalid/fork.git"],
        check=True,
    )

    result = _powershell(
        "-File",
        str(BOOTSTRAP),
        "-ValidateCheckoutOnly",
        "-ValidationUpstreamRoot",
        str(tmp_path),
    )

    assert result.returncode != 0
    assert "origin URL mismatch" in result.stderr


def test_runtime_preflight_reports_verified_installed_versions() -> None:
    result = _powershell(
        "-File",
        str(BOOTSTRAP),
        "-ValidateRuntimeOnly",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "python": "3.10.4",
        "setuptools": "80.9.0",
        "numpy": "1.23.5",
        "torch": "2.0.0+cu118",
        "torchvision": "0.15.1+cu118",
    }

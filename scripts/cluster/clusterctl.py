#!/usr/bin/env python3
"""Small, site-profiled helpers for Slurm research jobs."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JOB_SCRIPT = REPOSITORY_ROOT / "scripts" / "cluster" / "job.sh"
REQUIRED_PROFILE_KEYS = (
    "CLUSTER_PROFILE_SCHEMA",
    "SLURM_CPU_ACCOUNT",
    "SLURM_GPU_ACCOUNT",
    "SLURM_CPU_PARTITION",
    "SLURM_GPU_PARTITION",
    "SLURM_GPU_REQUEST",
    "SLURM_ARRAY_CONCURRENCY",
    "MODULES",
    "ENV_ACTIVATE",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RUNS_ROOT",
    "CACHE_ROOT",
    "LOG_ROOT",
    "SCRATCH_ENV",
    "SMOKE_TIME",
    "CALIBRATION_TIME",
    "FULL_TIME",
)
TIME_KEYS = {
    "smoke": "SMOKE_TIME",
    "calibration": "CALIBRATION_TIME",
    "full": "FULL_TIME",
}
_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TIME_PATTERN = re.compile(r"^(?:[0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$")
_GPU_PATTERN = re.compile(r"^--(?:gpus|gres)=[^\s]+:1$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ARRAY_PATTERN = re.compile(r"^([0-9]+)-([0-9]+)$")


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    return (
        not stripped
        or stripped == "replace-me"
        or (stripped.startswith("<") and stripped.endswith(">"))
        or "/GROUP/" in stripped
    )


def parse_profile(
    path: Path | str, *, allow_placeholders: bool = False
) -> dict[str, str]:
    """Read a strict, non-expanding KEY=VALUE cluster profile."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise ValueError(f"cluster profile is not a file: {profile_path}")

    values: dict[str, str] = {}
    allowed = set(REQUIRED_PROFILE_KEYS)
    for line_number, raw_line in enumerate(
        profile_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed profile line {line_number}")
        key, value = line.split("=", 1)
        if key != key.strip() or not _KEY_PATTERN.fullmatch(key):
            raise ValueError(f"malformed profile key on line {line_number}")
        if key not in allowed:
            raise ValueError(f"unknown profile key: {key}")
        if key in values:
            raise ValueError(f"duplicate profile key: {key}")
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError(f"malformed profile value for {key}")
        values[key] = value

    missing = [key for key in REQUIRED_PROFILE_KEYS if key not in values]
    if missing:
        raise ValueError(f"missing cluster profile keys: {', '.join(missing)}")
    if values["CLUSTER_PROFILE_SCHEMA"] != "1":
        raise ValueError("CLUSTER_PROFILE_SCHEMA must be 1")

    if not allow_placeholders:
        for key, value in values.items():
            if _is_placeholder(value):
                raise ValueError(f"{key} contains an unresolved placeholder")

    try:
        concurrency = int(values["SLURM_ARRAY_CONCURRENCY"])
    except ValueError as error:
        raise ValueError("SLURM_ARRAY_CONCURRENCY must be an integer") from error
    if not 1 <= concurrency <= 4:
        raise ValueError("SLURM_ARRAY_CONCURRENCY must be between 1 and 4")
    if not _GPU_PATTERN.fullmatch(values["SLURM_GPU_REQUEST"]):
        raise ValueError("SLURM_GPU_REQUEST must request exactly one GPU")
    if not _ENV_NAME_PATTERN.fullmatch(values["SCRATCH_ENV"]):
        raise ValueError("SCRATCH_ENV must name one environment variable")
    for key in ("SMOKE_TIME", "CALIBRATION_TIME", "FULL_TIME"):
        if not _TIME_PATTERN.fullmatch(values[key]):
            raise ValueError(f"{key} must be a Slurm time value")
    return values


def render_submit_command(
    profile: dict[str, str],
    *,
    profile_path: Path | str,
    job_name: str,
    time_class: str,
    command: Sequence[str],
    array: str | None = None,
) -> list[str]:
    """Render one deterministic, one-GPU sbatch command."""

    if not _NAME_PATTERN.fullmatch(job_name):
        raise ValueError("job name contains unsupported characters")
    if time_class not in TIME_KEYS:
        raise ValueError(f"unknown time class: {time_class!r}")
    if not command or any(not token or "\n" in token or "\r" in token for token in command):
        raise ValueError("scientific command must contain nonempty tokens")

    rendered = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--account={profile['SLURM_GPU_ACCOUNT']}",
        f"--partition={profile['SLURM_GPU_PARTITION']}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=96G",
        profile["SLURM_GPU_REQUEST"],
        f"--time={profile[TIME_KEYS[time_class]]}",
        f"--output={profile['LOG_ROOT']}/%x-%A_%a.out",
        f"--error={profile['LOG_ROOT']}/%x-%A_%a.err",
    ]
    if array is not None:
        match = _ARRAY_PATTERN.fullmatch(array)
        if match is None:
            raise ValueError("array must use canonical START-END syntax")
        start, end = (int(value) for value in match.groups())
        if start >= end:
            raise ValueError("array end must be greater than start")
        rendered.append(
            f"--array={start}-{end}%{int(profile['SLURM_ARRAY_CONCURRENCY'])}"
        )
    return [
        *rendered,
        str(JOB_SCRIPT),
        str(Path(profile_path).resolve()),
        "--",
        *command,
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)

    profile_parser = subcommands.add_parser("profile")
    profile_actions = profile_parser.add_subparsers(
        dest="profile_action", required=True
    )
    validate = profile_actions.add_parser("validate")
    validate.add_argument("profile", type=Path)
    validate.add_argument("--allow-placeholders", action="store_true")

    submit = subcommands.add_parser("submit")
    submit.add_argument("profile", type=Path)
    submit.add_argument("--job-name", required=True)
    submit.add_argument("--time-class", required=True)
    submit.add_argument("--array")
    submit.add_argument("--dry-run", action="store_true")
    return parser


def _split_scientific_command(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    arguments = list(argv)
    if arguments[:1] != ["submit"] or "--" not in arguments:
        return arguments, []
    separator = arguments.index("--")
    return arguments[:separator], arguments[separator + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser_arguments, scientific_command = _split_scientific_command(raw_arguments)
    arguments = parser.parse_args(parser_arguments)
    try:
        if arguments.action == "profile":
            parse_profile(
                arguments.profile,
                allow_placeholders=arguments.allow_placeholders,
            )
            return 0

        profile = parse_profile(arguments.profile)
        if not scientific_command:
            raise ValueError("scientific command is required after --")
        command = render_submit_command(
            profile,
            profile_path=arguments.profile,
            job_name=arguments.job_name,
            time_class=arguments.time_class,
            command=scientific_command,
            array=arguments.array,
        )
        if arguments.dry_run:
            print(shlex.join(command))
            return 0
        return subprocess.run(command, check=False).returncode
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

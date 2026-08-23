#!/usr/bin/env python3
"""Small, site-profiled helpers for Slurm research jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
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
_YEAR_PATTERN = re.compile(r"^[0-9]{4}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_HEADER = ("relative_path", "size_bytes", "sha256")
DATASET_NAME = "WSTS+ active fixed event-level HDF5"
PRODUCTION_FILE_COUNT = 999
PRODUCTION_TOTAL_BYTES = 49_816_826_985
PRODUCTION_YEAR_COUNTS = {
    "2016": 92,
    "2017": 110,
    "2018": 176,
    "2019": 74,
    "2020": 201,
    "2021": 156,
    "2022": 122,
    "2023": 68,
}


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    size_bytes: int
    sha256: str


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


def _is_link_or_reparse(path: Path) -> bool:
    stat_result = path.lstat()
    attributes = getattr(stat_result, "st_file_attributes", 0)
    reparse_flag = getattr(os.stat_result, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not reparse_flag:
        reparse_flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative_hdf5_path(value: str) -> tuple[int, str]:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or len(pure.parts) != 2
        or any(part in {"", ".", ".."} for part in pure.parts)
        or not _YEAR_PATTERN.fullmatch(pure.parts[0])
        or not pure.parts[1].endswith(".hdf5")
        or "\\" in value
    ):
        raise ValueError(f"unsafe or noncanonical manifest path: {value!r}")
    return int(pure.parts[0]), pure.parts[1]


def build_hdf5_manifest(data_root: Path | str) -> list[ManifestEntry]:
    """Hash one canonical year/event.hdf5 tree without opening HDF5 content."""

    root = Path(data_root)
    if not root.is_dir() or _is_link_or_reparse(root):
        raise ValueError(f"data root must be one canonical directory: {root}")
    entries: list[ManifestEntry] = []
    for year_root in sorted(root.iterdir(), key=lambda path: path.name):
        if year_root.is_file() and not _is_link_or_reparse(year_root):
            continue
        if (
            not _YEAR_PATTERN.fullmatch(year_root.name)
            or not year_root.is_dir()
            or _is_link_or_reparse(year_root)
        ):
            raise ValueError(f"noncanonical entry in data root: {year_root.name}")
        for event_path in sorted(year_root.iterdir(), key=lambda path: path.name):
            if (
                not event_path.is_file()
                or _is_link_or_reparse(event_path)
                or event_path.suffix != ".hdf5"
            ):
                raise ValueError(f"noncanonical event entry: {event_path}")
            relative_path = f"{year_root.name}/{event_path.name}"
            _validate_relative_hdf5_path(relative_path)
            entries.append(
                ManifestEntry(
                    relative_path=relative_path,
                    size_bytes=event_path.stat().st_size,
                    sha256=_sha256_file(event_path),
                )
            )
    return entries


def _manifest_csv_bytes(entries: Sequence[ManifestEntry]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=MANIFEST_HEADER, lineterminator="\n"
    )
    writer.writeheader()
    for entry in entries:
        writer.writerow(
            {
                "relative_path": entry.relative_path,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
            }
        )
    return output.getvalue().encode("utf-8")


def _summary_for(
    entries: Sequence[ManifestEntry], csv_bytes: bytes
) -> dict[str, object]:
    years: dict[str, int] = {}
    for entry in entries:
        year = entry.relative_path.split("/", 1)[0]
        years[year] = years.get(year, 0) + 1
    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "file_count": len(entries),
        "total_bytes": sum(entry.size_bytes for entry in entries),
        "years": years,
        "manifest_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }


def _require_production_contract(summary: dict[str, object]) -> None:
    if (
        summary["file_count"] != PRODUCTION_FILE_COUNT
        or summary["total_bytes"] != PRODUCTION_TOTAL_BYTES
        or summary["years"] != PRODUCTION_YEAR_COUNTS
    ):
        raise ValueError("manifest does not match the production dataset contract")


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and _is_link_or_reparse(path):
        raise ValueError(f"refusing to replace linked output: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_hdf5_manifest(
    data_root: Path | str,
    csv_path: Path | str,
    summary_path: Path | str,
    *,
    require_production_contract: bool = False,
) -> dict[str, object]:
    entries = build_hdf5_manifest(data_root)
    csv_bytes = _manifest_csv_bytes(entries)
    summary = _summary_for(entries, csv_bytes)
    if require_production_contract:
        _require_production_contract(summary)
    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    csv_target = Path(csv_path)
    summary_target = Path(summary_path)
    if csv_target.resolve() == summary_target.resolve():
        raise ValueError("manifest CSV and summary paths must be distinct")
    _write_atomic_bytes(csv_target, csv_bytes)
    _write_atomic_bytes(summary_target, summary_bytes)
    return summary


def _read_manifest_entries(csv_bytes: bytes) -> list[ManifestEntry]:
    try:
        text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("manifest CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != MANIFEST_HEADER:
        raise ValueError("manifest CSV header is invalid")
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for row in reader:
        if set(row) != set(MANIFEST_HEADER) or None in row:
            raise ValueError("manifest CSV row shape is invalid")
        relative_path = row["relative_path"]
        _validate_relative_hdf5_path(relative_path)
        if relative_path in seen:
            raise ValueError(f"duplicate manifest path: {relative_path}")
        seen.add(relative_path)
        size_text = row["size_bytes"]
        if not size_text.isdigit() or (size_text.startswith("0") and size_text != "0"):
            raise ValueError(f"invalid manifest size for {relative_path}")
        sha256 = row["sha256"]
        if not _SHA256_PATTERN.fullmatch(sha256):
            raise ValueError(f"invalid manifest SHA-256 for {relative_path}")
        entries.append(ManifestEntry(relative_path, int(size_text), sha256))
    if entries != sorted(
        entries,
        key=lambda entry: _validate_relative_hdf5_path(entry.relative_path),
    ):
        raise ValueError("manifest rows are not in canonical order")
    return entries


def verify_hdf5_manifest(
    data_root: Path | str,
    csv_path: Path | str,
    summary_path: Path | str,
    *,
    require_production_contract: bool = False,
) -> dict[str, object]:
    csv_target = Path(csv_path)
    summary_target = Path(summary_path)
    if not csv_target.is_file() or _is_link_or_reparse(csv_target):
        raise ValueError("manifest CSV is missing or linked")
    if not summary_target.is_file() or _is_link_or_reparse(summary_target):
        raise ValueError("manifest summary is missing or linked")
    csv_bytes = csv_target.read_bytes()
    recorded_entries = _read_manifest_entries(csv_bytes)
    expected_entries = build_hdf5_manifest(data_root)
    if recorded_entries != expected_entries:
        raise ValueError("dataset tree does not match the manifest")
    try:
        summary = json.loads(summary_target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest summary is not valid UTF-8 JSON") from error
    expected_summary = _summary_for(recorded_entries, csv_bytes)
    if summary != expected_summary:
        if isinstance(summary, dict) and summary.get("manifest_sha256") != expected_summary["manifest_sha256"]:
            raise ValueError("summary manifest_sha256 does not match CSV bytes")
        raise ValueError("manifest summary does not match manifest rows")
    if require_production_contract:
        _require_production_contract(summary)
    return summary


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

    manifest_parser = subcommands.add_parser("manifest")
    manifest_actions = manifest_parser.add_subparsers(
        dest="manifest_action", required=True
    )
    for action in ("create", "verify"):
        operation = manifest_actions.add_parser(action)
        operation.add_argument("data_root", type=Path)
        operation.add_argument("csv_path", type=Path)
        operation.add_argument("summary_path", type=Path)
        operation.add_argument("--require-production-contract", action="store_true")
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

        if arguments.action == "manifest":
            function = (
                write_hdf5_manifest
                if arguments.manifest_action == "create"
                else verify_hdf5_manifest
            )
            summary = function(
                arguments.data_root,
                arguments.csv_path,
                arguments.summary_path,
                require_production_contract=arguments.require_production_contract,
            )
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
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

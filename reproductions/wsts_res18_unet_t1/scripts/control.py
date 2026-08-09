"""Guardrails for the WSTS Res18-U-Net fold-2 timing calibration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias


FROZEN_YEAR_COUNTS = {2018: 176, 2019: 74, 2020: 201, 2021: 156}
EXPECTED_OVERRIDES = {
    "data.data_fold_id": 2,
    "data.features_to_keep": None,
    "data.n_leading_observations": 1,
    "data.remove_duplicate_features": True,
    "trainer.max_steps": 500,
    "do_test": False,
    "do_predict": False,
    "do_validate": False,
}

CommandRunner: TypeAlias = Callable[[list[str]], subprocess.CompletedProcess[str]]


def verify_inventory(data_root: Path) -> dict[str, int]:
    """Require exactly the four official WSTS year inventories."""
    root = data_root.resolve()
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")

    counts: dict[str, int] = {}
    for year, expected_count in FROZEN_YEAR_COUNTS.items():
        year_dir = (root / str(year)).resolve()
        if not year_dir.is_dir():
            raise ValueError(f"required year directory is missing: {year_dir}")
        count = sum(
            path.is_file() and path.suffix == ".hdf5" for path in year_dir.iterdir()
        )
        if count != expected_count:
            raise ValueError(
                f"year {year} has {count} direct .hdf5 files; expected {expected_count}"
            )
        counts[str(year)] = count

    allowed_years = {str(year) for year in FROZEN_YEAR_COUNTS}
    extra_years = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and path.name not in allowed_years
        and any(
            child.is_file() and child.suffix == ".hdf5" for child in path.resolve().iterdir()
        )
    )
    if extra_years:
        raise ValueError(
            "calibration must not select additional year directories with .hdf5 files: "
            + ", ".join(extra_years)
        )

    counts["total"] = sum(counts.values())
    return counts


def verify_upstream(
    upstream_root: Path,
    expected_commit: str,
    *,
    command_runner: CommandRunner = None,
) -> str:
    """Return the pinned checkout commit or reject a provenance mismatch."""
    root = upstream_root.resolve()
    runner = command_runner or _run_command
    result = runner(["git", "-C", str(root), "rev-parse", "HEAD"])
    if result.returncode != 0:
        detail = result.stderr.strip() or "git rev-parse HEAD failed"
        raise ValueError(f"cannot determine upstream commit: {detail}")
    actual_commit = result.stdout.strip()
    if actual_commit != expected_commit:
        raise ValueError(
            f"upstream commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    return actual_commit


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def validate_overrides(overrides: Mapping[str, object]) -> None:
    """Reject every override except the fixed, non-scientific calibration diff."""
    expected_keys = set(EXPECTED_OVERRIDES)
    actual_keys = set(overrides)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(unexpected))
        raise ValueError("override keys must exactly match the allowlist (" + "; ".join(details) + ")")

    for key, expected_value in EXPECTED_OVERRIDES.items():
        actual_value = overrides[key]
        if type(actual_value) is not type(expected_value):
            raise ValueError(
                f"override {key} has type {type(actual_value).__name__}; "
                f"expected {type(expected_value).__name__}"
            )
        if actual_value != expected_value:
            raise ValueError(
                f"override {key} must be {expected_value!r}; got {actual_value!r}"
            )


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Publish JSON only after a complete same-directory temporary write."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_override_json(value: str) -> Mapping[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise ValueError("overrides JSON must be an object with string keys")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    inventory = subcommands.add_parser("inventory")
    inventory.add_argument("data_root", type=Path)

    upstream = subcommands.add_parser("upstream")
    upstream.add_argument("upstream_root", type=Path)
    upstream.add_argument("expected_commit")

    overrides = subcommands.add_parser("validate-overrides")
    overrides.add_argument("overrides_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "inventory":
            output: Mapping[str, object] = verify_inventory(arguments.data_root)
        elif arguments.command == "upstream":
            output = {
                "commit": verify_upstream(arguments.upstream_root, arguments.expected_commit)
            }
        else:
            validate_overrides(_parse_override_json(arguments.overrides_json))
            output = {"valid": True}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for the Phase 0 WSTS+ data gate."""

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from shutil import copy2

import pandas as pd

from wildfire_phase0.contract import ContractDecision, audit_contract, load_contract
from wildfire_phase0.inventory import inventory_dataset
from wildfire_phase0.path_safety import (
    canonical_root,
    require_contained_path,
    require_pairwise_disjoint_roots,
)
from wildfire_phase0.report import render_phase0_report
from wildfire_phase0.repair import stage_active_fire_repair
from wildfire_phase0.schema import EventInventory
from wildfire_phase0.splits import build_forward_split
from wildfire_phase0.target_gate import target_integrity_errors


_ARTIFACT_NAMES = (
    "inventory.csv",
    "split_manifest.csv",
    "contract_decision.json",
    "phase0_report.md",
)
_INVENTORY_COLUMNS = (
    "year",
    "fire_name",
    "path",
    "n_days",
    "n_channels",
    "height",
    "width",
    "dates",
    "nan_fraction",
    "target_days",
    "zero_target_days",
    "positive_target_pixels",
    "active_fire_min_positive",
    "active_fire_max_positive",
)


def _default_contract_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "wstsplus_field_contract.csv"


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {name: output_root / name for name in _ARTIFACT_NAMES}


def _remove_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _owned_sidecar(path: Path, suffix: str) -> Path:
    file_descriptor, sidecar_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(file_descriptor)
    return Path(sidecar_name)


def _write_text_temp(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_frame_temp(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _publish_staged_artifacts(
    paths: Sequence[Path], temp_paths: Mapping[Path, Path]
) -> None:
    preexisting = frozenset(path for path in paths if path.exists())
    backups: dict[Path, Path] = {}
    try:
        for path in paths:
            if path in preexisting:
                backup = _owned_sidecar(path, ".bak")
                backups[path] = backup
                copy2(path, backup)
        for path in paths:
            temp_paths[path].replace(path)
    except OSError:
        rollback_succeeded = False
        try:
            for path in paths:
                backup = backups.get(path)
                if backup is not None and backup.exists():
                    copy2(backup, path)
                elif path not in preexisting and path.exists():
                    path.unlink()
            rollback_succeeded = True
        finally:
            if rollback_succeeded:
                _remove_paths(tuple(backups.values()))
        raise
    else:
        _remove_paths(tuple(backups.values()))


def _inventory_frame(inventory: Sequence[EventInventory]) -> pd.DataFrame:
    records = [
        {
            "year": item.year,
            "fire_name": item.fire_name,
            "path": item.path.as_posix(),
            "n_days": item.n_days,
            "n_channels": item.n_channels,
            "height": item.height,
            "width": item.width,
            "dates": json.dumps(item.dates, separators=(",", ":")),
            "nan_fraction": item.nan_fraction,
            "target_days": item.target_days,
            "zero_target_days": item.zero_target_days,
            "positive_target_pixels": item.positive_target_pixels,
            "active_fire_min_positive": item.active_fire_min_positive,
            "active_fire_max_positive": item.active_fire_max_positive,
        }
        for item in sorted(inventory, key=lambda item: (item.year, item.fire_name))
    ]
    return pd.DataFrame(records, columns=_INVENTORY_COLUMNS)


def _blocked_decision(decision: ContractDecision, error: Exception) -> ContractDecision:
    error_text = f"{type(error).__name__}: {error}"
    return replace(
        decision,
        status="blocked",
        notes=(*decision.notes, f"Data-gate validation failed: {error_text}"),
    )


def _reproducible_commands(data_root: Path, output_root: Path) -> tuple[str, ...]:
    return (
        'python -m pip install -e ".[dev]"',
        "python -m pytest -q",
        "python -m wildfire_phase0.cli audit "
        f'--data-root "{data_root}" --output-root "{output_root}"',
    )


def run_audit(data_root: Path, output_root: Path) -> int:
    """Run the read-only audit and write the four derived gate artifacts."""
    data_root = canonical_root(Path(data_root), "data root")
    output_root = canonical_root(Path(output_root), "output root")
    require_pairwise_disjoint_roots(
        {"data root": data_root, "output root": output_root}
    )
    artifacts = {
        name: require_contained_path(
            output_root, path, "output root"
        )
        for name, path in _artifact_paths(output_root).items()
    }
    output_root.mkdir(parents=True, exist_ok=True)

    contract = load_contract(_default_contract_path())
    decision = audit_contract(contract)
    inventory: list[EventInventory] = []
    invalid_file_error: str | None = None
    split_manifest = build_forward_split([])
    try:
        inventory = inventory_dataset(data_root)
        split_manifest = build_forward_split(inventory)
        target_errors = target_integrity_errors(inventory, split_manifest)
        if target_errors:
            raise ValueError("; ".join(target_errors))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        invalid_file_error = f"{type(error).__name__}: {error}"
        decision = _blocked_decision(decision, error)

    temp_paths: dict[Path, Path] = {}
    try:
        report = render_phase0_report(
            inventory,
            split_manifest,
            decision,
            invalid_file_error,
            _reproducible_commands(data_root, output_root),
        )
        for path in artifacts.values():
            temp_paths[path] = _owned_sidecar(path, ".tmp")
        _write_frame_temp(temp_paths[artifacts["inventory.csv"]], _inventory_frame(inventory))
        _write_frame_temp(temp_paths[artifacts["split_manifest.csv"]], split_manifest)
        _write_text_temp(
            temp_paths[artifacts["contract_decision.json"]],
            json.dumps(asdict(decision), indent=2, sort_keys=True) + "\n",
        )
        _write_text_temp(temp_paths[artifacts["phase0_report.md"]], report)
        _publish_staged_artifacts(tuple(artifacts.values()), temp_paths)
    finally:
        _remove_paths(tuple(temp_paths.values()))

    return 0 if decision.status in {"continue_natural", "continue_controlled"} else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Phase 0 WSTS+ data gate artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Inventory WSTS+ files and write the gate artifacts.")
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    repair = subparsers.add_parser(
        "repair-active-fire",
        help="Stage active-fire label repairs and write repair evidence.",
    )
    repair.add_argument("--source-tiff-root", type=Path, required=True)
    repair.add_argument("--hdf5-root", type=Path, required=True)
    repair.add_argument("--staging-root", type=Path, required=True)
    repair.add_argument("--years", type=int, nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and return the data-gate status code."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "audit":
        return run_audit(arguments.data_root, arguments.output_root)
    if arguments.command == "repair-active-fire":
        decision = stage_active_fire_repair(
            arguments.source_tiff_root,
            arguments.hdf5_root,
            arguments.staging_root,
            arguments.years,
        )
        return 0 if decision.status == "ready" else 2
    raise AssertionError(f"unexpected command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())

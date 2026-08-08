import csv
import json
from pathlib import Path

import h5py
import numpy as np

from wildfire_phase0 import cli


_ARTIFACT_NAMES = (
    "inventory.csv",
    "split_manifest.csv",
    "contract_decision.json",
    "phase0_report.md",
)


def _write_event(path: Path, year: int, fire_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((2, 23, 4, 4), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values)
        data.attrs["year"] = year
        data.attrs["fire_name"] = fire_name
        data.attrs["img_dates"] = [f"{year}-08-01", f"{year}-08-02"]
        data.attrs["lnglat"] = [-120.5, 54.1]


def _write_string_event(path: Path, year: int, fire_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((2, 23, 4, 4), b"x", dtype="S1")
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values)
        data.attrs["year"] = year
        data.attrs["fire_name"] = fire_name
        data.attrs["img_dates"] = [f"{year}-08-01", f"{year}-08-02"]
        data.attrs["lnglat"] = [-120.5, 54.1]


def _run_audit(data_root: Path, output_root: Path) -> int:
    return cli.main([
        "audit",
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
    ])


def test_audit_writes_controlled_gate_artifacts_for_a_valid_event(tmp_path: Path) -> None:
    """Removing artifact serialization or the controlled decision must fail this test."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")

    assert _run_audit(data_root, output_root) == 0
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)

    report = (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert "continue_controlled" in report
    assert "next-calendar-day active-fire proxy" in report
    assert "2016–2020 train / 2021 validation / 2022–2023 test" in report

    decision = json.loads((output_root / "contract_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "continue_controlled"
    assert "target_validity" in decision["missing_required"]
    assert "event_roi_provenance" in decision["operational_blockers"]
    assert not list(output_root.glob("*.tmp"))


def test_audit_serializes_inventory_and_split_in_deterministic_order(tmp_path: Path) -> None:
    """Removing stable ordering or deterministic date serialization must fail this test."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2023" / "alpha_fire.hdf5", 2023, "alpha_fire")
    _write_event(data_root / "2016" / "zulu_fire.hdf5", 2016, "zulu_fire")

    assert _run_audit(data_root, output_root) == 0

    with (output_root / "inventory.csv").open(newline="", encoding="utf-8") as handle:
        inventory_rows = list(csv.DictReader(handle))
    assert [(row["year"], row["fire_name"], row["dates"]) for row in inventory_rows] == [
        ("2016", "zulu_fire", '["2016-08-01","2016-08-02"]'),
        ("2023", "alpha_fire", '["2023-08-01","2023-08-02"]'),
    ]

    with (output_root / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    assert [(row["event_id"], row["split"]) for row in split_rows] == [
        ("2016:zulu_fire", "train"),
        ("2023:alpha_fire", "test"),
    ]


def test_audit_blocks_on_invalid_event_and_leaves_final_artifacts(tmp_path: Path) -> None:
    """Removing invalid-file reporting or temp cleanup must fail this test."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    invalid_path = data_root / "2021" / "demo_fire.hdf5"
    invalid_path.parent.mkdir(parents=True)
    with h5py.File(invalid_path, "w"):
        pass
    output_root.mkdir(parents=True)
    for name in _ARTIFACT_NAMES:
        (output_root / f"{name}.tmp").write_text("stale", encoding="utf-8")

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)
    assert "missing data dataset" in (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert json.loads((output_root / "contract_decision.json").read_text(encoding="utf-8"))["status"] == "blocked"


def test_audit_blocks_with_artifacts_when_an_event_is_outside_the_frozen_protocol(
    tmp_path: Path,
) -> None:
    """Removing split-protocol failure reporting must fail this test."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2015" / "older_fire.hdf5", 2015, "older_fire")

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)
    assert "event year must be within the 2016-2023 protocol" in (
        output_root / "phase0_report.md"
    ).read_text(encoding="utf-8")


def test_audit_reports_string_typed_hdf5_payload_as_blocked(tmp_path: Path) -> None:
    """Removing TypeError handling for invalid HDF5 payloads must fail this test."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_string_event(data_root / "2021" / "string_fire.hdf5", 2021, "string_fire")

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)
    report = (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert "TypeError" in report
    assert "isnan" in report
    assert not list(output_root.glob("*.tmp"))

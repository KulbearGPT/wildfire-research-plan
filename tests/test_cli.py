import csv
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile

from wildfire_phase0 import cli


_ARTIFACT_NAMES = (
    "inventory.csv",
    "split_manifest.csv",
    "contract_decision.json",
    "phase0_report.md",
)


def _write_event(
    path: Path,
    year: int,
    fire_name: str,
    *,
    target_hour: float = 9,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((2, 23, 4, 4), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    values[1, 22, 0, 0] = target_hour
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


def _write_data_group(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_group("data")


def _write_zero_sized_event(path: Path, year: int, fire_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", shape=(0, 23, 4, 4), dtype=np.float32)
        data.attrs["year"] = year
        data.attrs["fire_name"] = fire_name
        data.attrs["img_dates"] = []
        data.attrs["lnglat"] = [-120.5, 54.1]


def _write_repair_event(
    source_tiff_root: Path,
    hdf5_root: Path,
    year: int,
    fire_name: str,
    active_values: tuple[float, ...] = (0.0, 6.0),
) -> Path:
    event_dir = source_tiff_root / str(year) / fire_name
    event_dir.mkdir(parents=True)
    dates = tuple(f"{year}-01-{index + 1:02d}" for index in range(len(active_values)))
    values = np.zeros((len(active_values), 23, 4, 4), dtype=np.float32)
    for index, (date, active) in enumerate(zip(dates, active_values)):
        image = np.zeros((4, 4, 23), dtype=np.float32)
        image[..., 0] = index + 1
        image[0, 0, 22] = active
        tifffile.imwrite(event_dir / f"{date}.tif", image, photometric="minisblack")
        values[index, 0] = index + 1
    hdf5_path = hdf5_root / str(year) / f"{fire_name}.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    with h5py.File(hdf5_path, "w") as handle:
        data = handle.create_dataset(
            "data", data=values, chunks=(1, 23, 4, 4), compression="lzf", shuffle=True
        )
        data.attrs["year"] = year
        data.attrs["fire_name"] = fire_name
        data.attrs["img_dates"] = dates
        data.attrs["lnglat"] = [float("nan"), float("nan")]
    return hdf5_path


def _run_repair(
    source_tiff_root: Path,
    hdf5_root: Path,
    staging_root: Path,
    years: tuple[int, ...],
) -> int:
    return cli.main(
        [
            "repair-active-fire",
            "--source-tiff-root",
            str(source_tiff_root),
            "--hdf5-root",
            str(hdf5_root),
            "--staging-root",
            str(staging_root),
            "--years",
            *(str(year) for year in years),
        ]
    )


def _run_audit(data_root: Path, output_root: Path) -> int:
    return cli.main([
        "audit",
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
    ])


def _seed_old_generation(output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True)
    old_contents = {name: f"old:{name}\n" for name in _ARTIFACT_NAMES}
    for name, content in old_contents.items():
        (output_root / name).write_text(content, encoding="utf-8")
    return old_contents


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
        inventory_reader = csv.DictReader(handle)
        inventory_rows = list(inventory_reader)
    assert inventory_reader.fieldnames == [
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
    ]
    assert [(row["year"], row["fire_name"], row["dates"]) for row in inventory_rows] == [
        ("2016", "zulu_fire", '["2016-08-01","2016-08-02"]'),
        ("2023", "alpha_fire", '["2023-08-01","2023-08-02"]'),
    ]
    assert [
        (
            row["target_days"],
            row["zero_target_days"],
            row["positive_target_pixels"],
            row["active_fire_min_positive"],
            row["active_fire_max_positive"],
        )
        for row in inventory_rows
    ] == [("1", "0", "1", "9.0", "9.0"), ("1", "0", "1", "9.0", "9.0")]

    with (output_root / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    assert [(row["event_id"], row["split"]) for row in split_rows] == [
        ("2016:zulu_fire", "train"),
        ("2023:alpha_fire", "test"),
    ]


def test_audit_blocks_all_zero_benchmark_year_and_split_with_complete_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(
        data_root / "2022" / "zero_fire.hdf5",
        2022,
        "zero_fire",
        target_hour=0,
    )
    blocker = (
        "ValueError: split test has zero positive target pixels; "
        "year 2022 has zero positive target pixels"
    )

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)
    report = (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert blocker in report
    decision = json.loads((output_root / "contract_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "blocked"
    assert f"Data-gate validation failed: {blocker}" in decision["notes"]
    assert not list(output_root.glob("*.tmp"))


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


@pytest.mark.parametrize(
    ("writer", "exact_error"),
    [
        (_write_data_group, "ValueError: data object must be an HDF5 dataset"),
        (
            lambda path: _write_zero_sized_event(path, 2021, "demo_fire"),
            "ValueError: data shape dimensions must be strictly positive",
        ),
    ],
)
def test_audit_blocks_non_dataset_or_zero_sized_data_with_complete_artifacts(
    tmp_path: Path,
    writer: Callable[[Path], None],
    exact_error: str,
) -> None:
    """Skipping object/extent checks would crash before a complete blocked generation."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    event_path = data_root / "2021" / "demo_fire.hdf5"
    writer(event_path)

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)
    assert exact_error in (output_root / "phase0_report.md").read_text(encoding="utf-8")
    decision = json.loads((output_root / "contract_decision.json").read_text(encoding="utf-8"))
    assert any(exact_error in note for note in decision["notes"])
    assert not list(output_root.glob("*.tmp"))


def test_audit_render_failure_preserves_existing_complete_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing any final before report rendering would corrupt the old generation."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    old_contents = _seed_old_generation(output_root)

    def fail_render(*args: object, **kwargs: object) -> str:
        raise RuntimeError("render failed")

    monkeypatch.setattr(cli, "render_phase0_report", fail_render)

    with pytest.raises(RuntimeError, match="render failed"):
        _run_audit(data_root, output_root)

    assert {
        name: (output_root / name).read_text(encoding="utf-8")
        for name in _ARTIFACT_NAMES
    } == old_contents
    assert not list(output_root.glob("*.tmp"))


def test_audit_serialization_failure_preserves_existing_complete_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing one final before all serializers finish would mix generations."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    old_contents = _seed_old_generation(output_root)
    original_to_csv = pd.DataFrame.to_csv
    calls = 0

    def fail_second_csv(self: pd.DataFrame, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("serialization failed")
        return original_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_second_csv)

    with pytest.raises(RuntimeError, match="serialization failed"):
        _run_audit(data_root, output_root)

    assert {
        name: (output_root / name).read_text(encoding="utf-8")
        for name in _ARTIFACT_NAMES
    } == old_contents
    assert not list(output_root.glob("*.tmp"))


def test_audit_publish_failure_rolls_back_every_existing_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-publication replacement failure must not expose a mixed generation."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    old_contents = _seed_old_generation(output_root)
    original_replace = Path.replace
    publish_calls = 0

    def fail_second_publish(source: Path, target: Path) -> Path:
        nonlocal publish_calls
        if source.name.endswith(".tmp") and Path(target).name in _ARTIFACT_NAMES:
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        _run_audit(data_root, output_root)

    assert publish_calls == 2
    assert {
        name: (output_root / name).read_text(encoding="utf-8")
        for name in _ARTIFACT_NAMES
    } == old_contents
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob("*.bak"))


def test_audit_publish_failure_removes_finals_without_predecessors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    original_replace = Path.replace
    publish_calls = 0

    def fail_second_publish(source: Path, target: Path) -> Path:
        nonlocal publish_calls
        if source.name.endswith(".tmp") and Path(target).name in _ARTIFACT_NAMES:
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        _run_audit(data_root, output_root)

    assert not any((output_root / name).exists() for name in _ARTIFACT_NAMES)
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob("*.bak"))


def test_repair_active_fire_cli_returns_zero_and_preserves_sources(tmp_path: Path) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    source_paths = [
        _write_repair_event(source_tiff_root, hdf5_root, year, f"fire_{year}")
        for year in (2016, 2022)
    ]
    source_hashes = {path: path.read_bytes() for path in source_paths}

    exit_code = _run_repair(
        source_tiff_root, hdf5_root, staging_root, (2016, 2022)
    )

    assert exit_code == 0
    assert {path: path.read_bytes() for path in source_paths} == source_hashes
    assert (staging_root / "active_fire_repair_manifest.csv").is_file()
    decision_path = staging_root / "active_fire_repair_decision.json"
    assert json.loads(decision_path.read_text(encoding="utf-8"))["status"] == "ready"
    assert not list(staging_root.rglob("*.tmp"))


def test_repair_to_audit_preserves_nonzero_labels_in_every_split(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    audit_root = tmp_path / "audit-data"
    output_root = tmp_path / "artifacts"
    source_paths = {
        year: _write_repair_event(
            source_tiff_root, hdf5_root, year, f"fire_{year}"
        )
        for year in (2016, 2021, 2022)
    }
    with h5py.File(source_paths[2021], "r+") as handle:
        handle["data"][1, 22, 0, 0] = 6.0

    assert _run_repair(
        source_tiff_root, hdf5_root, staging_root, (2016, 2022)
    ) == 0

    selected_paths = {
        2016: staging_root / "2016" / "fire_2016.hdf5",
        2021: source_paths[2021],
        2022: staging_root / "2022" / "fire_2022.hdf5",
    }
    for year, source_path in selected_paths.items():
        destination = audit_root / str(year) / source_path.name
        destination.parent.mkdir(parents=True)
        shutil.copy2(source_path, destination)

    assert _run_audit(audit_root, output_root) == 0
    report = (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert "| train | 1 | 1 | 0 | 1 |" in report
    assert "| validation | 1 | 1 | 0 | 1 |" in report
    assert "| test | 1 | 1 | 0 | 1 |" in report


def test_repair_active_fire_cli_returns_two_with_complete_error_evidence(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    source_path = _write_repair_event(
        source_tiff_root, hdf5_root, 2016, "hdf5_only"
    )
    for path in (source_tiff_root / "2016" / "hdf5_only").glob("*.tif"):
        path.unlink()
    source_bytes = source_path.read_bytes()

    exit_code = _run_repair(source_tiff_root, hdf5_root, staging_root, (2016,))

    assert exit_code == 2
    assert source_path.read_bytes() == source_bytes
    assert (staging_root / "active_fire_repair_manifest.csv").is_file()
    decision_path = staging_root / "active_fire_repair_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["status"] == "blocked"
    assert decision["errors"]
    assert not list(staging_root.rglob("*.tmp"))


def test_repair_active_fire_cli_returns_two_with_deterministic_invalid_year_evidence(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"

    first_exit = _run_repair(source_tiff_root, hdf5_root, staging_root, (2015,))
    manifest_path = staging_root / "active_fire_repair_manifest.csv"
    decision_path = staging_root / "active_fire_repair_decision.json"
    first_manifest = manifest_path.read_bytes()
    first_decision = decision_path.read_bytes()
    second_exit = _run_repair(source_tiff_root, hdf5_root, staging_root, (2015,))

    assert first_exit == second_exit == 2
    assert manifest_path.read_bytes() == first_manifest
    assert decision_path.read_bytes() == first_decision
    assert first_manifest.decode("utf-8").splitlines() == [
        "year,fire_name,source_hdf5,staged_hdf5,source_fingerprint,"
        "source_encoding,days,target_days,zero_target_days,"
        "positive_target_pixels,status,error"
    ]
    decision = json.loads(first_decision)
    assert decision == {
        "errors": ["ValueError: years must be unique integers within 2016..2023"],
        "excluded_empty_source_directories": [],
        "files_expected": 0,
        "files_staged": 0,
        "files_verified": 0,
        "requested_years": [2015],
        "status": "blocked",
    }
    assert first_decision.endswith(b"\n")
    assert not list(staging_root.rglob("*.tmp"))

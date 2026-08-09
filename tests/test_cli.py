import csv
import json
import os
import shutil
import subprocess
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
_RULE_ARTIFACT_NAMES = (
    "rule_event_metrics.csv",
    "rule_summary.csv",
    "rule_report.md",
)


def _make_directory_alias(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory aliases are unavailable: {symlink_error}")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"directory aliases are unavailable: {completed.stderr}")


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


def _write_protocol_events(
    data_root: Path,
    *,
    names: dict[int, str] | None = None,
    target_hours: dict[int, float] | None = None,
) -> None:
    event_names = names or {}
    hours = target_hours or {}
    for year in range(2016, 2024):
        fire_name = event_names.get(year, f"fire_{year}")
        _write_event(
            data_root / str(year) / f"{fire_name}.hdf5",
            year,
            fire_name,
            target_hour=hours.get(year, 9),
        )


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


def _write_rule_fixture(data_root: Path, manifest_path: Path) -> list[Path]:
    specifications = [
        (2022, "zulu_fire", "test"),
        (2020, "bravo_fire", "train"),
        (2021, "alpha_fire", "validation"),
    ]
    paths = []
    rows = []
    for year, fire_name, split in specifications:
        path = data_root / str(year) / f"{fire_name}.hdf5"
        _write_event(path, year, fire_name)
        paths.append(path)
        rows.append(
            {
                "event_id": f"{year}:{fire_name}",
                "year": year,
                "fire_name": fire_name,
                "path": path.relative_to(data_root).as_posix(),
                "split": split,
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False, lineterminator="\n")
    return paths


def _run_rules(data_root: Path, manifest_path: Path, output_root: Path) -> int:
    return cli.main(
        [
            "evaluate-rules",
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ]
    )


def _seed_old_rule_generation(output_root: Path) -> dict[str, bytes]:
    output_root.mkdir(parents=True)
    old_bytes = {name: f"old:{name}\n".encode() for name in _RULE_ARTIFACT_NAMES}
    for name, content in old_bytes.items():
        (output_root / name).write_bytes(content)
    return old_bytes


def test_evaluate_rules_writes_exact_deterministic_read_only_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "split_manifest.csv"
    output_root = tmp_path / "rule-artifacts"
    source_paths = _write_rule_fixture(data_root, manifest_path)
    source_bytes = {path: path.read_bytes() for path in source_paths}

    assert _run_rules(data_root, manifest_path, output_root) == 0
    first_generation = {
        name: (output_root / name).read_bytes() for name in _RULE_ARTIFACT_NAMES
    }
    assert {path.name for path in output_root.iterdir()} == set(_RULE_ARTIFACT_NAMES)
    assert _run_rules(data_root, manifest_path, output_root) == 0
    assert {
        name: (output_root / name).read_bytes() for name in _RULE_ARTIFACT_NAMES
    } == first_generation
    assert {path: path.read_bytes() for path in source_paths} == source_bytes

    with (output_root / "rule_event_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        event_rows = list(csv.DictReader(handle))
    assert [
        (row["split"], row["year"], row["fire_name"], row["baseline"])
        for row in event_rows
    ] == [
        ("test", "2022", "zulu_fire", "no_fire"),
        ("test", "2022", "zulu_fire", "persistence_latest"),
        ("train", "2020", "bravo_fire", "no_fire"),
        ("train", "2020", "bravo_fire", "persistence_latest"),
        ("validation", "2021", "alpha_fire", "no_fire"),
        ("validation", "2021", "alpha_fire", "persistence_latest"),
    ]
    with (output_root / "rule_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [(row["split"], row["baseline"]) for row in summary_rows] == [
        ("test", "no_fire"),
        ("test", "persistence_latest"),
        ("train", "no_fire"),
        ("train", "persistence_latest"),
        ("validation", "no_fire"),
        ("validation", "persistence_latest"),
    ]
    report = (output_root / "rule_report.md").read_text(encoding="utf-8")
    assert "fixed T=1 rules" in report
    assert "next-day target" in report
    assert "No threshold or model tuning" in report
    assert "Raw AP is prevalence-dependent" in report
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob("*.bak"))


def test_evaluate_rules_preserves_unknown_sidecars(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "split_manifest.csv"
    output_root = tmp_path / "rule-artifacts"
    _write_rule_fixture(data_root, manifest_path)
    output_root.mkdir()
    unknown = output_root / "rule_summary.csv.tmp"
    unknown.write_bytes(b"unowned-sidecar")

    assert _run_rules(data_root, manifest_path, output_root) == 0

    assert unknown.read_bytes() == b"unowned-sidecar"


def test_evaluate_rules_validation_failure_returns_two_without_mixed_generation(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "split_manifest.csv"
    output_root = tmp_path / "rule-artifacts"
    _write_rule_fixture(data_root, manifest_path)
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest["year"] == 2021, "split"] = "test"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    old_bytes = _seed_old_rule_generation(output_root)

    assert _run_rules(data_root, manifest_path, output_root) == 2

    assert {
        name: (output_root / name).read_bytes() for name in _RULE_ARTIFACT_NAMES
    } == old_bytes
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob("*.bak"))


def test_evaluate_rules_publish_error_bubbles_and_rolls_back_entire_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "split_manifest.csv"
    output_root = tmp_path / "rule-artifacts"
    _write_rule_fixture(data_root, manifest_path)
    old_bytes = _seed_old_rule_generation(output_root)
    original_replace = Path.replace
    publish_calls = 0

    def fail_second_publish(source: Path, target: Path) -> Path:
        nonlocal publish_calls
        if source.name.endswith(".tmp") and Path(target).name in _RULE_ARTIFACT_NAMES:
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("injected rule publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_publish)

    with pytest.raises(OSError, match="injected rule publish failure"):
        _run_rules(data_root, manifest_path, output_root)

    assert publish_calls == 2
    assert {
        name: (output_root / name).read_bytes() for name in _RULE_ARTIFACT_NAMES
    } == old_bytes
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob("*.bak"))


@pytest.mark.parametrize("relation", ("equal", "output-under-data", "output-above-data"))
def test_evaluate_rules_rejects_non_disjoint_roots_before_mutation(
    tmp_path: Path,
    relation: str,
) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "split_manifest.csv"
    output_root = tmp_path / "rule-artifacts"
    _write_rule_fixture(data_root, manifest_path)
    if relation == "equal":
        output_root = data_root
    elif relation == "output-under-data":
        output_root = data_root / "artifacts"
    else:
        output_root = tmp_path / "shared"
        moved_data_root = output_root / "data"
        output_root.mkdir()
        data_root.rename(moved_data_root)
        data_root = moved_data_root
        manifest = pd.read_csv(manifest_path)
        manifest.to_csv(manifest_path, index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="pairwise disjoint"):
        _run_rules(data_root, manifest_path, output_root)

    assert not any((output_root / name).exists() for name in _RULE_ARTIFACT_NAMES)


def test_evaluate_rules_rejects_artifact_symlink_escape_without_touching_target(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "split_manifest.csv"
    output_root = tmp_path / "rule-artifacts"
    _write_rule_fixture(data_root, manifest_path)
    output_root.mkdir()
    external = tmp_path / "external.csv"
    external.write_bytes(b"external")
    try:
        (output_root / "rule_summary.csv").symlink_to(external)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="outside designated output root"):
        _run_rules(data_root, manifest_path, output_root)

    assert external.read_bytes() == b"external"


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
    _write_protocol_events(data_root)

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
    _write_protocol_events(
        data_root, names={2016: "zulu_fire", 2023: "alpha_fire"}
    )

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
        ("2017", "fire_2017", '["2017-08-01","2017-08-02"]'),
        ("2018", "fire_2018", '["2018-08-01","2018-08-02"]'),
        ("2019", "fire_2019", '["2019-08-01","2019-08-02"]'),
        ("2020", "fire_2020", '["2020-08-01","2020-08-02"]'),
        ("2021", "fire_2021", '["2021-08-01","2021-08-02"]'),
        ("2022", "fire_2022", '["2022-08-01","2022-08-02"]'),
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
    ] == [
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
        ("1", "0", "1", "9.0", "9.0"),
    ]

    with (output_root / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    assert [(row["event_id"], row["split"]) for row in split_rows] == [
        ("2016:zulu_fire", "train"),
        ("2017:fire_2017", "train"),
        ("2018:fire_2018", "train"),
        ("2019:fire_2019", "train"),
        ("2020:fire_2020", "train"),
        ("2021:fire_2021", "validation"),
        ("2022:fire_2022", "test"),
        ("2023:alpha_fire", "test"),
    ]


def test_audit_blocks_all_zero_benchmark_year_and_split_with_complete_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_protocol_events(data_root, target_hours={2022: 0, 2023: 0})
    blocker = (
        "ValueError: split test has zero positive target pixels; "
        "year 2022 has zero positive target pixels; "
        "year 2023 has zero positive target pixels"
    )

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == set(_ARTIFACT_NAMES)
    report = (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert blocker in report
    decision = json.loads((output_root / "contract_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "blocked"
    assert f"Data-gate validation failed: {blocker}" in decision["notes"]
    assert not list(output_root.glob("*.tmp"))


def test_audit_blocks_incomplete_frozen_protocol_with_complete_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    for year in range(2016, 2023):
        _write_event(
            data_root / str(year) / f"fire_{year}.hdf5",
            year,
            f"fire_{year}",
        )

    assert _run_audit(data_root, output_root) == 2
    exact_error = "ValueError: benchmark year 2023 is missing from inventory"
    assert exact_error in (output_root / "phase0_report.md").read_text(encoding="utf-8")
    decision = json.loads(
        (output_root / "contract_decision.json").read_text(encoding="utf-8")
    )
    assert any(exact_error in note for note in decision["notes"])


def test_audit_blocks_on_invalid_event_and_leaves_final_artifacts(tmp_path: Path) -> None:
    """Unknown sidecars must survive while blocked evidence is published."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    invalid_path = data_root / "2021" / "demo_fire.hdf5"
    invalid_path.parent.mkdir(parents=True)
    with h5py.File(invalid_path, "w"):
        pass
    output_root.mkdir(parents=True)
    stale_sidecars = {}
    for name in _ARTIFACT_NAMES:
        path = output_root / f"{name}.tmp"
        path.write_text(f"stale:{name}", encoding="utf-8")
        stale_sidecars[path] = f"stale:{name}"

    assert _run_audit(data_root, output_root) == 2
    assert {path.name for path in output_root.iterdir()} == {
        *_ARTIFACT_NAMES,
        *(f"{name}.tmp" for name in _ARTIFACT_NAMES),
    }
    assert "missing data dataset" in (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert json.loads((output_root / "contract_decision.json").read_text(encoding="utf-8"))["status"] == "blocked"
    assert {path: path.read_text(encoding="utf-8") for path in stale_sidecars} == stale_sidecars


@pytest.mark.parametrize("relation", ("equal", "output-under-data", "output-above-data"))
def test_audit_rejects_non_disjoint_roots_before_mutation(
    tmp_path: Path, relation: str
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    if relation == "equal":
        output_root = data_root
    elif relation == "output-under-data":
        output_root = data_root / "artifacts"
    else:
        output_root = tmp_path / "shared"
        data_root = output_root / "data"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    marker = output_root / "keep.txt"
    if output_root.exists():
        marker.write_bytes(b"keep")

    with pytest.raises(ValueError, match="pairwise disjoint"):
        _run_audit(data_root, output_root)

    assert not any((output_root / name).exists() for name in _ARTIFACT_NAMES)
    if marker.exists():
        assert marker.read_bytes() == b"keep"


def test_audit_rejects_resolved_root_alias_before_mutation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    output_alias = tmp_path / "output-alias"
    _make_directory_alias(output_alias, data_root)

    with pytest.raises(ValueError, match="pairwise disjoint"):
        _run_audit(data_root, output_alias)

    assert not any((data_root / name).exists() for name in _ARTIFACT_NAMES)


def test_audit_rejects_artifact_symlink_escape_without_touching_target(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    output_root.mkdir()
    external = tmp_path / "external.csv"
    external.write_bytes(b"external")
    try:
        (output_root / "inventory.csv").symlink_to(external)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="outside designated output root"):
        _run_audit(data_root, output_root)

    assert external.read_bytes() == b"external"


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


def test_audit_synchronous_base_exception_rolls_back_every_existing_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catchable synchronous interruption must not expose a mixed generation."""

    class SynchronousInterruption(BaseException):
        pass

    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    _seed_old_generation(output_root)
    old_bytes = {
        name: (output_root / name).read_bytes() for name in _ARTIFACT_NAMES
    }
    original_replace = Path.replace
    publish_calls = 0

    def interrupt_second_publish(source: Path, target: Path) -> Path:
        nonlocal publish_calls
        if source.name.endswith(".tmp") and Path(target).name in _ARTIFACT_NAMES:
            publish_calls += 1
            if publish_calls == 2:
                raise SynchronousInterruption("injected synchronous interruption")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", interrupt_second_publish)

    with pytest.raises(
        SynchronousInterruption, match="injected synchronous interruption"
    ):
        _run_audit(data_root, output_root)

    assert publish_calls == 2
    assert {
        name: (output_root / name).read_bytes()
        for name in _ARTIFACT_NAMES
    } == old_bytes
    assert not list(output_root.glob("*.tmp"))
    assert not list(output_root.glob("*.bak"))


def test_audit_backup_copy_failure_preserves_existing_finals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete backup must never be used to restore an existing final."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    _write_event(data_root / "2021" / "demo_fire.hdf5", 2021, "demo_fire")
    _seed_old_generation(output_root)
    old_bytes = {
        name: (output_root / name).read_bytes() for name in _ARTIFACT_NAMES
    }
    original_copy2 = cli.copy2
    backup_copies = 0

    def fail_first_backup_copy(
        source: Path, destination: Path, *args: object, **kwargs: object
    ) -> Path:
        nonlocal backup_copies
        if (
            Path(source).name in _ARTIFACT_NAMES
            and Path(destination).suffix == ".bak"
        ):
            backup_copies += 1
            if backup_copies == 1:
                raise OSError("injected backup copy failure")
        return original_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(cli, "copy2", fail_first_backup_copy)

    with pytest.raises(OSError, match="injected backup copy failure"):
        _run_audit(data_root, output_root)

    assert backup_copies == 1
    assert {
        name: (output_root / name).read_bytes()
        for name in _ARTIFACT_NAMES
    } == old_bytes
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
    for year in (2016, 2017, 2022, 2023):
        _write_repair_event(source_tiff_root, hdf5_root, year, f"fire_{year}")
    original_paths = {}
    for year in (2018, 2019, 2020, 2021):
        path = hdf5_root / str(year) / f"fire_{year}.hdf5"
        _write_event(path, year, f"fire_{year}", target_hour=6)
        original_paths[year] = path

    assert _run_repair(
        source_tiff_root, hdf5_root, staging_root, (2016, 2017, 2022, 2023)
    ) == 0

    selected_paths = {
        2016: staging_root / "2016" / "fire_2016.hdf5",
        2017: staging_root / "2017" / "fire_2017.hdf5",
        **original_paths,
        2022: staging_root / "2022" / "fire_2022.hdf5",
        2023: staging_root / "2023" / "fire_2023.hdf5",
    }
    for year, source_path in selected_paths.items():
        destination = audit_root / str(year) / source_path.name
        destination.parent.mkdir(parents=True)
        shutil.copy2(source_path, destination)

    assert _run_audit(audit_root, output_root) == 0
    report = (output_root / "phase0_report.md").read_text(encoding="utf-8")
    assert "| train | 5 | 5 | 0 | 5 |" in report
    assert "| validation | 1 | 1 | 0 | 1 |" in report
    assert "| test | 2 | 2 | 0 | 2 |" in report


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


def test_repair_active_fire_cli_returns_two_for_missing_inputs_with_evidence(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "missing-tiff"
    hdf5_root = tmp_path / "missing-hdf5"
    staging_root = tmp_path / "staging"

    exit_code = _run_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert exit_code == 2
    manifest_path = staging_root / "active_fire_repair_manifest.csv"
    decision_path = staging_root / "active_fire_repair_decision.json"
    assert manifest_path.is_file()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["status"] == "blocked"
    assert decision["errors"] == [
        "ValueError: HDF5 root must be an existing directory",
        "ValueError: source TIFF root must be an existing directory",
    ]


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
        "year,fire_name,source_event_dir,source_hdf5,staged_hdf5,source_fingerprint,"
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
        "generation": "efc4c03d6ef18801cd783c5dc09a61e0d5d318f85e7a06c4c0ca5254faecc9d4",
        "manifest_sha256": "6be655dc500875332ae05ff17112447ff709cbb7a7c212309b1a0c990ae5eccf",
        "requested_years": [2015],
        "status": "blocked",
    }
    assert first_decision.endswith(b"\n")
    assert not list(staging_root.rglob("*.tmp"))

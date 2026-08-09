import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from wildfire_phase0.repair import (
    classify_active_fire_encoding,
    normalize_active_fire,
    source_fingerprint,
    stage_active_fire_repair,
    stage_event,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_tiff_event(
    event_dir: Path,
    active_values: list[float],
    dates: tuple[str, ...] | None = None,
) -> None:
    event_dir.mkdir(parents=True)
    event_dates = dates or tuple(
        f"{event_dir.parent.name}-01-{index + 1:02d}"
        for index in range(len(active_values))
    )
    for index, (date, active) in enumerate(zip(event_dates, active_values)):
        image = np.zeros((4, 4, 23), dtype=np.float32)
        image[..., 0] = index + 1
        image[0, 0, 22] = active
        tifffile.imwrite(event_dir / f"{date}.tif", image, photometric="minisblack")


def _write_hdf5_event(
    path: Path,
    active_values: list[float],
    dates: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True)
    year = int(path.parent.name)
    event_dates = dates or tuple(
        f"{year}-01-{index + 1:02d}" for index in range(len(active_values))
    )
    values = np.zeros((len(active_values), 23, 4, 4), dtype=np.float32)
    for index, active in enumerate(active_values):
        values[index, 0] = index + 1
        values[index, 22, 0, 0] = active
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset(
            "data", data=values, chunks=(1, 23, 4, 4), compression="lzf", shuffle=True
        )
        data.attrs["year"] = year
        data.attrs["fire_name"] = path.stem
        data.attrs["img_dates"] = event_dates
        data.attrs["lnglat"] = [float("nan"), float("nan")]


def test_normalize_preserves_hour_encoding_and_replaces_nan() -> None:
    values = np.array([[np.nan, 0.0, 6.0, 22.0]], dtype=np.float32)
    normalized, encoding = normalize_active_fire(values)
    assert encoding == "hour"
    assert normalized.tolist() == [[0.0, 0.0, 6.0, 22.0]]


def test_normalize_converts_hhmm_exactly_once() -> None:
    values = np.array([[0.0, 806.0, 1359.0, 2200.0]], dtype=np.float32)
    normalized, encoding = normalize_active_fire(values)
    assert encoding == "hhmm"
    assert normalized.tolist() == [[0.0, 8.0, 13.0, 22.0]]


def test_normalize_marks_zero_only_source() -> None:
    normalized, encoding = normalize_active_fire(
        np.array([[0.0, np.nan]], dtype=np.float32)
    )
    assert encoding == "no_positive_values"
    assert normalized.tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize(
    "values",
    [
        np.array([[-1.0]]),
        np.array([[6.5]]),
        np.array([[6.0, 806.0]]),
        np.array([[2360.0]]),
        np.array([[np.inf]]),
    ],
)
def test_classification_rejects_invalid_or_mixed_encoding(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="active-fire encoding"):
        classify_active_fire_encoding(values)


def test_source_fingerprint_is_order_independent_and_metadata_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "2016" / "fire_a" / "2016-01-01.tif"
    second = tmp_path / "2016" / "fire_a" / "2016-01-02.tif"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    forward = source_fingerprint([first, second], tmp_path)
    reverse = source_fingerprint([second, first], tmp_path)
    assert forward == reverse
    second.write_bytes(b"changed-size")
    assert source_fingerprint([first, second], tmp_path) != forward


def test_stage_event_repairs_only_active_channel_and_preserves_source(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [0.0, 6.0, 22.0])
    _write_hdf5_event(source_hdf5, active_values=[0.0, 0.0, 0.0])
    source_hash = _sha256(source_hdf5)

    record = stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert _sha256(source_hdf5) == source_hash
    assert record.status == "repaired"
    assert record.source_encoding == "hour"
    assert record.source_event_dir == event_dir.resolve().as_posix()
    with h5py.File(source_hdf5, "r") as source, h5py.File(staging_hdf5, "r") as staged:
        np.testing.assert_array_equal(source["data"][:, :22], staged["data"][:, :22])
        assert staged["data"][:, 22, 0, 0].tolist() == [0.0, 6.0, 22.0]
        assert staged["data"].attrs["active_fire_source_encoding"] == "hour"
        assert staged["data"].attrs["active_fire_stored_encoding"] == "hour"


def test_stage_event_preserves_unknown_preexisting_temp_file(tmp_path: Path) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [0.0, 6.0])
    _write_hdf5_event(source_hdf5, [0.0, 0.0])
    collision = staging_hdf5.with_name("fire_a.hdf5.tmp")
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"belongs-to-another-process")

    record = stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert record.status == "repaired"
    assert collision.read_bytes() == b"belongs-to-another-process"


def test_stage_event_rejects_date_mismatch_without_final_or_temp(tmp_path: Path) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(
        event_dir,
        [0.0, 6.0],
        dates=("2016-01-01", "2016-01-02"),
    )
    _write_hdf5_event(
        source_hdf5,
        [0.0, 0.0],
        dates=("2016-01-01", "2016-01-03"),
    )
    with pytest.raises(ValueError, match="dates"):
        stage_event(event_dir, source_hdf5, staging_hdf5, source_root)
    assert not staging_hdf5.exists()
    assert not staging_hdf5.with_name("fire_a.hdf5.tmp").exists()


def test_stage_event_rejects_tiff_shape_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [6.0, 7.0])
    tifffile.imwrite(
        event_dir / "2016-01-02.tif",
        np.zeros((5, 4, 23), dtype=np.float32),
        photometric="minisblack",
    )
    _write_hdf5_event(source_hdf5, [0.0, 0.0])

    with pytest.raises(ValueError, match="shape"):
        stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert not staging_hdf5.exists()
    assert not staging_hdf5.with_name("fire_a.hdf5.tmp").exists()


def test_stage_event_rejects_filename_and_fire_name_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_b.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [6.0])
    _write_hdf5_event(source_hdf5, [0.0])

    with pytest.raises(ValueError, match="fire name"):
        stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert not staging_hdf5.exists()


def test_stage_event_verifies_preexisting_stage_with_matching_fingerprint(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [0.0, 6.0])
    _write_hdf5_event(source_hdf5, [0.0, 0.0])
    stage_event(event_dir, source_hdf5, staging_hdf5, source_root)
    staged_hash = _sha256(staging_hdf5)

    record = stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert record.status == "verified"
    assert _sha256(staging_hdf5) == staged_hash
    assert not staging_hdf5.with_name("fire_a.hdf5.tmp").exists()


def test_stage_event_rejects_stale_preexisting_stage(tmp_path: Path) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [0.0, 6.0])
    _write_hdf5_event(source_hdf5, [0.0, 0.0])
    stage_event(event_dir, source_hdf5, staging_hdf5, source_root)
    tifffile.imwrite(
        event_dir / "2016-01-02.tif",
        np.ones((4, 4, 23), dtype=np.float32),
        photometric="minisblack",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert staging_hdf5.exists()
    assert not staging_hdf5.with_name("fire_a.hdf5.tmp").exists()


@pytest.mark.parametrize(
    ("channel", "error_match"),
    [(0, "non-active"), (22, "normalized source")],
)
def test_stage_event_rejects_corrupted_preexisting_stage(
    tmp_path: Path, channel: int, error_match: str
) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [0.0, 6.0])
    _write_hdf5_event(source_hdf5, [0.0, 0.0])
    stage_event(event_dir, source_hdf5, staging_hdf5, source_root)
    with h5py.File(staging_hdf5, "r+") as handle:
        handle["data"][1, channel, 0, 0] = 7.0

    with pytest.raises(ValueError, match=error_match):
        stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert not staging_hdf5.with_name("fire_a.hdf5.tmp").exists()


@pytest.mark.parametrize("mutation", ["altered", "missing", "extra"])
def test_stage_event_rejects_changed_preserved_attributes_on_retry(
    tmp_path: Path, mutation: str
) -> None:
    source_root = tmp_path / "tiff"
    event_dir = source_root / "2016" / "fire_a"
    source_hdf5 = tmp_path / "hdf5" / "2016" / "fire_a.hdf5"
    staging_hdf5 = tmp_path / "staging" / "2016" / "fire_a.hdf5"
    _write_tiff_event(event_dir, [0.0, 6.0])
    _write_hdf5_event(source_hdf5, [0.0, 0.0])
    stage_event(event_dir, source_hdf5, staging_hdf5, source_root)
    with h5py.File(staging_hdf5, "r+") as handle:
        attributes = handle["data"].attrs
        if mutation == "altered":
            attributes["lnglat"] = [-115.0, 52.0]
        elif mutation == "missing":
            del attributes["lnglat"]
        else:
            attributes["unexpected_preserved_attr"] = "unexpected"

    with pytest.raises(ValueError, match="attributes"):
        stage_event(event_dir, source_hdf5, staging_hdf5, source_root)

    assert staging_hdf5.exists()
    assert not staging_hdf5.with_name("fire_a.hdf5.tmp").exists()


def test_stage_active_fire_repair_stages_all_events_and_writes_evidence(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    for year, fire_name in ((2022, "zulu_fire"), (2016, "alpha_fire")):
        _write_tiff_event(source_tiff_root / str(year) / fire_name, [0.0, 6.0])
        _write_hdf5_event(hdf5_root / str(year) / f"{fire_name}.hdf5", [0.0, 0.0])

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016, 2022)
    )

    assert decision.status == "ready"
    assert decision.files_expected == 2
    assert decision.files_staged == 2
    assert decision.files_verified == 0
    assert not list(staging_root.rglob("*.tmp"))
    manifest_path = staging_root / "active_fire_repair_manifest.csv"
    decision_path = staging_root / "active_fire_repair_decision.json"
    assert manifest_path.is_file()
    assert decision_path.is_file()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["year"], row["fire_name"], row["status"]) for row in rows] == [
        ("2016", "alpha_fire", "repaired"),
        ("2022", "zulu_fire", "repaired"),
    ]
    assert list(rows[0]) == [
        "year",
        "fire_name",
        "source_event_dir",
        "source_hdf5",
        "staged_hdf5",
        "source_fingerprint",
        "source_encoding",
        "days",
        "target_days",
        "zero_target_days",
        "positive_target_pixels",
        "status",
        "error",
    ]
    assert json.loads(decision_path.read_text(encoding="utf-8"))["status"] == "ready"
    assert decision_path.read_bytes().endswith(b"\n")


def test_repair_evidence_preserves_unknown_preexisting_sidecars(tmp_path: Path) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    _write_tiff_event(source_tiff_root / "2016" / "fire_a", [0.0, 6.0])
    _write_hdf5_event(hdf5_root / "2016" / "fire_a.hdf5", [0.0, 0.0])
    staging_root.mkdir()
    unknown_sidecars = {
        staging_root / "active_fire_repair_manifest.csv.tmp": b"unknown-temp",
        staging_root / "active_fire_repair_decision.json.bak": b"unknown-backup",
    }
    for path, content in unknown_sidecars.items():
        path.write_bytes(content)

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert decision.status == "ready"
    assert {path: path.read_bytes() for path in unknown_sidecars} == unknown_sidecars


def test_stage_active_fire_repair_retry_verifies_existing_files(tmp_path: Path) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    _write_tiff_event(source_tiff_root / "2016" / "fire_a", [0.0, 6.0])
    _write_hdf5_event(hdf5_root / "2016" / "fire_a.hdf5", [0.0, 0.0])
    stage_active_fire_repair(source_tiff_root, hdf5_root, staging_root, (2016,))

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert decision.status == "ready"
    assert decision.files_staged == 0
    assert decision.files_verified == 1
    with (staging_root / "active_fire_repair_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert next(csv.DictReader(handle))["status"] == "verified"


def test_stage_active_fire_repair_records_empty_and_unmatched_sources(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    (source_tiff_root / "2022" / "empty_fire").mkdir(parents=True)
    _write_tiff_event(source_tiff_root / "2022" / "tiff_only", [6.0])
    _write_hdf5_event(hdf5_root / "2022" / "hdf5_only.hdf5", [0.0])

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2022,)
    )

    assert decision.status == "blocked"
    assert decision.excluded_empty_source_directories == ("2022/empty_fire",)
    assert any("tiff_only" in error and "without HDF5" in error for error in decision.errors)
    assert any(
        "hdf5_only" in error and "without a matching nonempty" in error
        for error in decision.errors
    )
    assert (staging_root / "active_fire_repair_manifest.csv").is_file()
    assert (staging_root / "active_fire_repair_decision.json").is_file()
    with (staging_root / "active_fire_repair_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [
        (
            row["fire_name"],
            row["source_event_dir"],
            row["source_hdf5"],
            row["staged_hdf5"],
            row["status"],
        )
        for row in rows
    ] == [
        (
            "hdf5_only",
            "",
            (hdf5_root / "2022" / "hdf5_only.hdf5").resolve().as_posix(),
            "",
            "error",
        ),
        (
            "tiff_only",
            (source_tiff_root / "2022" / "tiff_only").resolve().as_posix(),
            "",
            "",
            "error",
        ),
    ]
    assert not list(staging_root.rglob("*.tmp"))


def test_stage_active_fire_repair_records_one_row_for_matched_failure(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    event_dir = source_tiff_root / "2016" / "fire_a"
    source_hdf5 = hdf5_root / "2016" / "fire_a.hdf5"
    _write_tiff_event(
        event_dir,
        [0.0, 6.0],
        dates=("2016-01-01", "2016-01-02"),
    )
    _write_hdf5_event(
        source_hdf5,
        [0.0, 0.0],
        dates=("2016-01-01", "2016-01-03"),
    )

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert decision.status == "blocked"
    with (staging_root / "active_fire_repair_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["source_event_dir"] == event_dir.resolve().as_posix()
    assert rows[0]["source_hdf5"] == source_hdf5.resolve().as_posix()
    assert rows[0]["staged_hdf5"] == ""
    assert rows[0]["status"] == "error"
    assert "dates" in rows[0]["error"]


@pytest.mark.parametrize("years", [(2016, 2016), (2015,), (2024,)])
def test_stage_active_fire_repair_blocks_invalid_requested_years_with_evidence(
    tmp_path: Path, years: tuple[int, ...]
) -> None:
    staging_root = tmp_path / "staging"

    decision = stage_active_fire_repair(
        tmp_path / "tiff", tmp_path / "hdf5", staging_root, years
    )

    assert decision.status == "blocked"
    assert decision.errors == (
        "ValueError: years must be unique integers within 2016..2023",
    )
    assert (staging_root / "active_fire_repair_manifest.csv").is_file()
    assert (staging_root / "active_fire_repair_decision.json").is_file()
    assert not list(staging_root.rglob("*.tmp"))


def test_stage_active_fire_repair_blocks_empty_requested_years_with_evidence(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    source_tiff_root.mkdir()
    hdf5_root.mkdir()

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, ()
    )

    assert decision.status == "blocked"
    assert decision.errors == ("ValueError: years must not be empty",)
    assert (staging_root / "active_fire_repair_manifest.csv").is_file()
    assert (staging_root / "active_fire_repair_decision.json").is_file()


@pytest.mark.parametrize(
    ("missing", "exact_error"),
    (
        ("source", "ValueError: source TIFF root must be an existing directory"),
        ("hdf5", "ValueError: HDF5 root must be an existing directory"),
    ),
)
def test_stage_active_fire_repair_blocks_missing_input_root_with_evidence(
    tmp_path: Path, missing: str, exact_error: str
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    if missing != "source":
        (source_tiff_root / "2016").mkdir(parents=True)
    if missing != "hdf5":
        (hdf5_root / "2016").mkdir(parents=True)

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert decision.status == "blocked"
    assert decision.errors == (exact_error,)
    assert (staging_root / "active_fire_repair_manifest.csv").is_file()
    assert (staging_root / "active_fire_repair_decision.json").is_file()


def test_stage_active_fire_repair_blocks_empty_existing_roots_with_evidence(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    source_tiff_root.mkdir()
    hdf5_root.mkdir()

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert decision.status == "blocked"
    assert decision.errors == (
        "ValueError: HDF5 year directory does not exist: 2016",
        "ValueError: source TIFF year directory does not exist: 2016",
    )


def test_stage_active_fire_repair_blocks_any_missing_requested_year(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    _write_tiff_event(source_tiff_root / "2016" / "fire_a", [0.0, 6.0])
    _write_hdf5_event(hdf5_root / "2016" / "fire_a.hdf5", [0.0, 0.0])

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016, 2017)
    )

    assert decision.status == "blocked"
    assert decision.errors == (
        "ValueError: HDF5 year directory does not exist: 2017",
        "ValueError: source TIFF year directory does not exist: 2017",
    )


def test_stage_active_fire_repair_blocks_year_without_matched_nonempty_event(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    (source_tiff_root / "2016").mkdir(parents=True)
    (hdf5_root / "2016").mkdir(parents=True)

    decision = stage_active_fire_repair(
        source_tiff_root, hdf5_root, staging_root, (2016,)
    )

    assert decision.status == "blocked"
    assert decision.errors == (
        "ValueError: requested year has no matched nonempty events: 2016",
    )


@pytest.mark.parametrize(
    "relation",
    ("equal-inputs", "staging-under-source", "staging-above-source", "source-under-hdf5"),
)
def test_stage_active_fire_repair_rejects_non_disjoint_roots_before_mutation(
    tmp_path: Path, relation: str
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    if relation == "equal-inputs":
        hdf5_root = source_tiff_root
    elif relation == "staging-under-source":
        staging_root = source_tiff_root / "staging"
    elif relation == "staging-above-source":
        staging_root = tmp_path / "shared"
        source_tiff_root = staging_root / "tiff"
    else:
        source_tiff_root = hdf5_root / "tiff"
    source_tiff_root.mkdir(parents=True)
    hdf5_root.mkdir(parents=True, exist_ok=True)
    marker = staging_root / "keep.txt"
    if staging_root.exists():
        marker.write_bytes(b"keep")

    with pytest.raises(ValueError, match="pairwise disjoint"):
        stage_active_fire_repair(
            source_tiff_root, hdf5_root, staging_root, (2016,)
        )

    assert not (staging_root / "active_fire_repair_manifest.csv").exists()
    assert not (staging_root / "active_fire_repair_decision.json").exists()
    if marker.exists():
        assert marker.read_bytes() == b"keep"


def test_stage_active_fire_repair_rejects_resolved_root_alias_before_mutation(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    source_tiff_root.mkdir()
    staging_alias = tmp_path / "staging-alias"
    _make_directory_alias(staging_alias, source_tiff_root)
    hdf5_root = tmp_path / "hdf5"
    hdf5_root.mkdir()

    with pytest.raises(ValueError, match="pairwise disjoint"):
        stage_active_fire_repair(
            source_tiff_root, hdf5_root, staging_alias, (2016,)
        )

    assert not (source_tiff_root / "active_fire_repair_manifest.csv").exists()


def test_stage_active_fire_repair_rejects_staging_year_alias_escape(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    escaped_root = tmp_path / "escaped"
    source_tiff_root.mkdir()
    hdf5_root.mkdir()
    staging_root.mkdir()
    escaped_root.mkdir()
    _make_directory_alias(staging_root / "2016", escaped_root)

    with pytest.raises(ValueError, match="outside designated staging root"):
        stage_active_fire_repair(
            source_tiff_root, hdf5_root, staging_root, (2016,)
        )

    assert not any(escaped_root.iterdir())
    assert not (staging_root / "active_fire_repair_manifest.csv").exists()


def test_stage_active_fire_repair_rejects_source_event_alias_before_mutation(
    tmp_path: Path,
) -> None:
    source_tiff_root = tmp_path / "tiff"
    hdf5_root = tmp_path / "hdf5"
    staging_root = tmp_path / "staging"
    source_year = source_tiff_root / "2016"
    source_year.mkdir(parents=True)
    escaped_event = tmp_path / "escaped" / "fire_a"
    _write_tiff_event(escaped_event, [0.0, 6.0], dates=("2016-01-01", "2016-01-02"))
    _make_directory_alias(source_year / "fire_a", escaped_event)
    _write_hdf5_event(hdf5_root / "2016" / "fire_a.hdf5", [0.0, 0.0])

    with pytest.raises(ValueError, match="outside designated source TIFF root"):
        stage_active_fire_repair(
            source_tiff_root, hdf5_root, staging_root, (2016,)
        )

    assert not staging_root.exists()

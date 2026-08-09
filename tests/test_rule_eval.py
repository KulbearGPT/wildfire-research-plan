import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from wildfire_phase0.rule_eval import evaluate_rule_dataset, evaluate_rule_event


def _write_active_event(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((len(masks), 23, *masks.shape[1:]), dtype=np.float32)
    values[:, 22] = masks
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values)
        data.attrs["year"] = int(path.parent.name)
        data.attrs["fire_name"] = path.stem
        data.attrs["img_dates"] = [
            f"{path.parent.name}-08-{day:02d}" for day in range(1, len(masks) + 1)
        ]
        data.attrs["lnglat"] = [-120.5, 54.1]


def _manifest_row(path: Path, data_root: Path, split: str) -> dict[str, object]:
    year = int(path.parent.name)
    return {
        "event_id": f"{year}:{path.stem}",
        "year": year,
        "fire_name": path.stem,
        "path": path.relative_to(data_root).as_posix(),
        "split": split,
    }


def _replace_data_attribute(path: Path, name: str, value: object) -> None:
    with h5py.File(path, "r+") as handle:
        del handle["data"].attrs[name]
        handle["data"].attrs[name] = value


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


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_id", "year", "fire_name", "path", "split"])


def test_event_evaluation_uses_next_day_and_latest_persistence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    _write_active_event(path, masks)
    original_bytes = path.read_bytes()

    rows = evaluate_rule_event(path, tmp_path, "validation").set_index("baseline")

    persistence = rows.loc["persistence_latest"]
    assert persistence["target_days"] == 2
    assert persistence["zero_target_days"] == 1
    assert persistence["positive_target_pixels"] == 2
    assert persistence["total_pixels"] == 8
    assert (persistence["tp"], persistence["fp"], persistence["fn"]) == (1, 2, 1)
    assert persistence["tn"] == 4
    assert persistence["event_ap"] == pytest.approx(7 / 24)
    assert persistence["event_ap_defined"]
    assert persistence["zero_target_pixels"] == 4
    assert persistence["zero_target_predicted_positive_pixels"] == 2
    no_fire = rows.loc["no_fire"]
    assert (no_fire["tp"], no_fire["fp"], no_fire["fn"]) == (0, 0, 2)
    assert no_fire["tn"] == 6
    assert no_fire["event_ap"] == pytest.approx(1 / 4)
    assert no_fire["zero_target_predicted_positive_pixels"] == 0
    assert rows["event_id"].tolist() == ["2021:fire_a", "2021:fire_a"]
    assert rows["path"].tolist() == ["2021/fire_a.hdf5", "2021/fire_a.hdf5"]
    assert path.read_bytes() == original_bytes


def test_zero_positive_event_keeps_row_and_marks_ap_undefined(tmp_path: Path) -> None:
    path = tmp_path / "2022" / "zero_fire.hdf5"
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    _write_active_event(path, masks)

    rows = evaluate_rule_event(path, tmp_path, "test")

    assert rows["baseline"].tolist() == ["no_fire", "persistence_latest"]
    assert rows["event_ap_defined"].tolist() == [False, False]
    assert rows["event_ap"].isna().all()
    assert rows["zero_target_days"].tolist() == [2, 2]
    assert rows["zero_target_predicted_positive_pixels"].tolist() == [0, 1]


def test_event_evaluation_rejects_missing_data(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w"):
        pass

    with pytest.raises(ValueError, match="missing data dataset"):
        evaluate_rule_event(path, tmp_path, "validation")


@pytest.mark.parametrize("invalid_value", [-1, 1.5, 24, np.inf])
def test_event_evaluation_rejects_invalid_active_fire_values(
    tmp_path: Path, invalid_value: float
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    with h5py.File(path, "r+") as handle:
        handle["data"][0, 22, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="active-fire"):
        evaluate_rule_event(path, tmp_path, "validation")


def test_event_evaluation_requires_at_least_two_days(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((1, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="at least 2 days"):
        evaluate_rule_event(path, tmp_path, "validation")


def test_event_evaluation_rejects_fractional_year_attribute(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    _replace_data_attribute(path, "year", 2021.75)

    with pytest.raises(ValueError, match="year attribute"):
        evaluate_rule_event(path, tmp_path, "validation")


def test_event_evaluation_rejects_bool_year_attribute(tmp_path: Path) -> None:
    path = tmp_path / "1" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    _replace_data_attribute(path, "year", True)
    _replace_data_attribute(path, "img_dates", ["0001-08-01", "0001-08-02"])

    with pytest.raises(ValueError, match="year attribute"):
        evaluate_rule_event(path, tmp_path, "train")


def test_event_evaluation_rejects_non_string_fire_name_attribute(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2021" / "123.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    _replace_data_attribute(path, "fire_name", 123)

    with pytest.raises(ValueError, match="fire_name attribute"):
        evaluate_rule_event(path, tmp_path, "validation")


def test_event_evaluation_rejects_non_string_date_attributes(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    _replace_data_attribute(path, "img_dates", [20210801, 20210802])

    with pytest.raises(ValueError, match="img_dates attribute"):
        evaluate_rule_event(path, tmp_path, "validation")


@pytest.mark.parametrize(
    "lnglat",
    [
        [-120.5],
        [-120.5, 54.1, 0.0],
        ["west", "north"],
        [np.inf, 54.1],
        [-120.5, -np.inf],
    ],
)
def test_event_evaluation_rejects_malformed_lnglat_attribute(
    tmp_path: Path, lnglat: object
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    _replace_data_attribute(path, "lnglat", lnglat)

    with pytest.raises(ValueError, match="lnglat attribute"):
        evaluate_rule_event(path, tmp_path, "validation")


def test_event_evaluation_allows_nan_lnglat_for_missing_roi(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2022" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    _replace_data_attribute(path, "lnglat", [np.nan, np.nan])

    rows = evaluate_rule_event(path, tmp_path, "test")

    assert rows["baseline"].tolist() == ["no_fire", "persistence_latest"]


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("event_id", "2021:different"),
        ("year", 2020),
        ("fire_name", "different"),
        ("path", "2021/different.hdf5"),
    ],
)
def test_dataset_evaluation_rejects_manifest_path_or_event_mismatch(
    tmp_path: Path, column: str, replacement: object
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "validation")
    row[column] = replacement

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row]))


@pytest.mark.parametrize("invalid_year", [2021.0, True])
def test_dataset_evaluation_rejects_non_integral_manifest_year(
    tmp_path: Path, invalid_year: object
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "validation")
    row["year"] = invalid_year

    with pytest.raises(ValueError, match="manifest year must"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("event_id", 2021),
        ("fire_name", b"fire_a"),
        ("path", Path("2021/fire_a.hdf5")),
        ("path", ["2021/fire_a.hdf5"]),
        ("split", 1),
    ],
)
def test_dataset_evaluation_rejects_non_string_manifest_fields(
    tmp_path: Path, column: str, replacement: object
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "validation")
    row[column] = replacement

    with pytest.raises(ValueError, match=f"manifest {column} must be a string"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row]))


@pytest.mark.parametrize(
    "manifest_path",
    [
        "2021/../2021/fire_a.hdf5",
        "2021\\fire_a.hdf5",
        "./2021/fire_a.hdf5",
        "2021//fire_a.hdf5",
    ],
)
def test_dataset_evaluation_rejects_noncanonical_manifest_paths(
    tmp_path: Path, manifest_path: str
) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "validation")
    row["path"] = manifest_path

    with pytest.raises(ValueError, match="canonical POSIX relative"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row]))


def test_dataset_evaluation_rejects_absolute_manifest_path(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "validation")
    row["path"] = path.resolve().as_posix()

    with pytest.raises(ValueError, match="canonical POSIX relative"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row]))


def test_dataset_evaluation_rejects_year_directory_alias_before_enumeration(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    external_year = tmp_path / "external-year"
    external_year.mkdir()
    _make_directory_alias(data_root / "2021", external_year)

    with pytest.raises(ValueError, match="outside designated data root"):
        evaluate_rule_dataset(data_root, _empty_manifest())


def test_dataset_evaluation_rejects_candidate_file_symlink_before_file_check(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    year_directory = data_root / "2021"
    year_directory.mkdir(parents=True)
    external_path = tmp_path / "external" / "2021" / "fire_a.hdf5"
    _write_active_event(external_path, np.zeros((2, 2, 2), dtype=np.uint8))
    try:
        (year_directory / "fire_a.hdf5").symlink_to(external_path)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="outside designated data root"):
        evaluate_rule_dataset(data_root, _empty_manifest())


def test_dataset_evaluation_rejects_duplicate_manifest_entries(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "validation")

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row, row]))


def test_dataset_evaluation_rejects_missing_manifest_entries(tmp_path: Path) -> None:
    first = tmp_path / "2021" / "fire_a.hdf5"
    second = tmp_path / "2022" / "fire_b.hdf5"
    _write_active_event(first, np.zeros((2, 2, 2), dtype=np.uint8))
    _write_active_event(second, np.zeros((2, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_rule_dataset(
            tmp_path, pd.DataFrame([_manifest_row(first, tmp_path, "validation")])
        )


def test_dataset_evaluation_rejects_extra_manifest_entries(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    rows = [
        _manifest_row(path, tmp_path, "validation"),
        {
            "event_id": "2022:missing",
            "year": 2022,
            "fire_name": "missing",
            "path": "2022/missing.hdf5",
            "split": "test",
        },
    ]

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame(rows))


def test_dataset_evaluation_rejects_non_frozen_split(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
    row = _manifest_row(path, tmp_path, "test")

    with pytest.raises(ValueError, match="frozen split"):
        evaluate_rule_dataset(tmp_path, pd.DataFrame([row]))


def test_dataset_evaluation_has_deterministic_sorted_rows(tmp_path: Path) -> None:
    specifications = [
        (2022, "zulu", "test"),
        (2020, "bravo", "train"),
        (2020, "alpha", "train"),
    ]
    manifest_rows = []
    for year, fire_name, split in specifications:
        path = tmp_path / str(year) / f"{fire_name}.hdf5"
        _write_active_event(path, np.zeros((2, 2, 2), dtype=np.uint8))
        manifest_rows.append(_manifest_row(path, tmp_path, split))

    rows = evaluate_rule_dataset(tmp_path, pd.DataFrame(manifest_rows))

    assert rows[["split", "year", "fire_name", "baseline"]].values.tolist() == [
        ["test", 2022, "zulu", "no_fire"],
        ["test", 2022, "zulu", "persistence_latest"],
        ["train", 2020, "alpha", "no_fire"],
        ["train", 2020, "alpha", "persistence_latest"],
        ["train", 2020, "bravo", "no_fire"],
        ["train", 2020, "bravo", "persistence_latest"],
    ]

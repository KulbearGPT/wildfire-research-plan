import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from wildfire_phase0.verify_repair import (
    ExpectedYear,
    main,
    verify_active_fire_dataset,
)


def _raw_image(day: int, active: float) -> np.ndarray:
    image = np.zeros((4, 4, 23), dtype=np.float32)
    image[..., 0] = day + 1
    image[0, 0, 22] = active
    return image


def _write_raw_event(path: Path, active_values: list[float]) -> tuple[str, ...]:
    path.mkdir(parents=True)
    dates = tuple(f"2016-08-{day + 1:02d}" for day in range(len(active_values)))
    for day, (date, active) in enumerate(zip(dates, active_values)):
        image = _raw_image(day, active)
        if day == len(active_values) - 1:
            image = np.moveaxis(image, -1, 0)
        tifffile.imwrite(path / f"{date}.tif", image, photometric="minisblack")
    return dates


def _write_staged_event(
    path: Path,
    active_values: list[float],
    *,
    dates: tuple[str, ...] | None = None,
    compression: str | None = "lzf",
    channels: int = 23,
) -> None:
    path.parent.mkdir(parents=True)
    dates = dates or tuple(
        f"2016-08-{day + 1:02d}" for day in range(len(active_values))
    )
    values = np.zeros((len(active_values), channels, 4, 4), dtype=np.float32)
    for day, active in enumerate(active_values):
        values[day, 0] = day + 1
        if channels > 22:
            values[day, 22, 0, 0] = active
    create_options: dict[str, object] = {}
    if compression is not None:
        create_options = {
            "chunks": (1, channels, 4, 4),
            "compression": compression,
            "shuffle": True,
        }
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values, **create_options)
        data.attrs["year"] = 2016
        data.attrs["fire_name"] = path.stem
        data.attrs["img_dates"] = dates
        data.attrs["active_fire_source_encoding"] = "hour"
        data.attrs["active_fire_stored_encoding"] = "hour"
        data.attrs["active_fire_repair_version"] = "1"
        data.attrs["active_fire_source_fingerprint"] = "fixture-fingerprint"


def _write_verification_fixture(
    tmp_path: Path, mutation: str
) -> tuple[Path, Path, dict[int, ExpectedYear]]:
    source_root = tmp_path / "source"
    hdf5_root = tmp_path / "staged"
    raw_event = source_root / "2016" / "fire_a"
    dates = _write_raw_event(raw_event, [0.0, 9.0])
    staged_event = hdf5_root / "2016" / "fire_a.hdf5"
    compression = None if mutation == "missing_lzf" else "lzf"
    channels = 22 if mutation == "wrong_channels" else 23
    active_values = [0.0, 24.0] if mutation == "out_of_range" else [0.0, 9.0]
    _write_staged_event(
        staged_event,
        active_values,
        dates=dates,
        compression=compression,
        channels=channels,
    )
    if mutation == "missing_shuffle":
        with h5py.File(staged_event, "r") as source:
            values = np.asarray(source["data"])
            attributes = dict(source["data"].attrs)
        staged_event.unlink()
        with h5py.File(staged_event, "w") as handle:
            data = handle.create_dataset(
                "data",
                data=values,
                chunks=(1, 23, 4, 4),
                compression="lzf",
                shuffle=False,
            )
            for name, value in attributes.items():
                data.attrs[name] = value
    if mutation == "missing_repair_attr":
        with h5py.File(staged_event, "r+") as handle:
            del handle["data"].attrs["active_fire_repair_version"]
    if mutation == "date_mismatch":
        with h5py.File(staged_event, "r+") as handle:
            handle["data"].attrs["img_dates"] = [dates[0], "2016-08-03"]
    if mutation == "raw_sample_mismatch":
        image = tifffile.imread(raw_event / f"{dates[0]}.tif")
        image[0, 0, 0] = 99
        tifffile.imwrite(
            raw_event / f"{dates[0]}.tif", image, photometric="minisblack"
        )
    if mutation == "raw_mixed_encoding":
        path = raw_event / f"{dates[-1]}.tif"
        image = tifffile.imread(path)
        image[22, 0, 1] = 930
        tifffile.imwrite(path, image, photometric="minisblack")
    files = 2 if mutation == "wrong_file_count" else 1
    target_days = 2 if mutation == "wrong_target_days" else 1
    zero_days = 1 if mutation == "wrong_zero_days" else 0
    positive_pixels = 2 if mutation == "wrong_positive_count" else 1
    return hdf5_root, source_root, {
        2016: ExpectedYear(files, target_days, zero_days, positive_pixels)
    }


def test_verifier_checks_counts_compression_range_and_raw_samples(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    hdf5_root = tmp_path / "staged"
    dates = _write_raw_event(source_root / "2016" / "fire_a", [0.0, 9.0])
    _write_staged_event(
        hdf5_root / "2016" / "fire_a.hdf5", [0.0, 9.0], dates=dates
    )
    expected = {2016: ExpectedYear(1, 1, 0, 1)}

    summary = verify_active_fire_dataset(hdf5_root, source_root, expected)

    assert summary[2016].files == 1
    assert summary[2016].positive_target_pixels == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_positive_count", "positive target pixels"),
        ("wrong_file_count", "files"),
        ("wrong_target_days", "target days"),
        ("wrong_zero_days", "zero target days"),
        ("missing_lzf", "compression"),
        ("missing_shuffle", "compression"),
        ("out_of_range", "0-23"),
        ("raw_sample_mismatch", "raw comparison"),
        ("raw_mixed_encoding", "mixed hour and HHMM"),
        ("wrong_channels", "23 channels"),
        ("missing_repair_attr", "repair attribute"),
        ("date_mismatch", "TIFF stems"),
    ],
)
def test_verifier_rejects_independent_invariant_failures(
    tmp_path: Path, mutation: str, message: str
) -> None:
    hdf5_root, source_root, expected = _write_verification_fixture(
        tmp_path, mutation
    )

    with pytest.raises(ValueError, match=message):
        verify_active_fire_dataset(hdf5_root, source_root, expected)


def test_verifier_main_parses_exact_expectation_and_writes_sorted_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hdf5_root, source_root, _ = _write_verification_fixture(tmp_path, "valid")

    exit_code = main(
        [
            "--hdf5-root",
            str(hdf5_root),
            "--source-tiff-root",
            str(source_root),
            "--expect",
            "2016:1:1:0:1",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        json.dumps(
            {
                "2016": {
                    "files": 1,
                    "positive_target_pixels": 1,
                    "target_days": 1,
                    "zero_target_days": 0,
                }
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_verifier_independently_normalizes_raw_hhmm_samples(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    hdf5_root = tmp_path / "staged"
    dates = _write_raw_event(source_root / "2016" / "fire_a", [0.0, 930.0])
    _write_staged_event(
        hdf5_root / "2016" / "fire_a.hdf5", [0.0, 9.0], dates=dates
    )
    with h5py.File(hdf5_root / "2016" / "fire_a.hdf5", "r+") as handle:
        handle["data"].attrs["active_fire_source_encoding"] = "hhmm"

    summary = verify_active_fire_dataset(
        hdf5_root, source_root, {2016: ExpectedYear(1, 1, 0, 1)}
    )

    assert summary[2016].positive_target_pixels == 1


def test_verifier_rejects_cross_day_mixed_raw_encoding(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    hdf5_root = tmp_path / "staged"
    dates = _write_raw_event(source_root / "2016" / "fire_a", [9.0, 930.0])
    _write_staged_event(
        hdf5_root / "2016" / "fire_a.hdf5", [9.0, 9.0], dates=dates
    )

    with pytest.raises(ValueError, match="mixed event encoding"):
        verify_active_fire_dataset(
            hdf5_root, source_root, {2016: ExpectedYear(1, 1, 0, 1)}
        )


@pytest.mark.parametrize(
    "expectations",
    [
        ["2016:1:1:0:1", "2016:1:1:0:1"],
        ["2016:1:1:0"],
        ["year:1:1:0:1"],
        ["2016:1:-1:0:1"],
    ],
)
def test_verifier_main_rejects_duplicate_years_and_malformed_expectations(
    tmp_path: Path, expectations: list[str]
) -> None:
    with pytest.raises(ValueError, match="expect"):
        main(
            [
                "--hdf5-root",
                str(tmp_path / "staged"),
                "--source-tiff-root",
                str(tmp_path / "source"),
                *(item for expectation in expectations for item in ("--expect", expectation)),
            ]
        )

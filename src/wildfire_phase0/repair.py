from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import tifffile


ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]
REPAIR_VERSION = "1"
_REPAIR_ATTRIBUTES = frozenset(
    {
        "active_fire_source_encoding",
        "active_fire_stored_encoding",
        "active_fire_repair_version",
        "active_fire_source_fingerprint",
    }
)


@dataclass(frozen=True)
class RepairRecord:
    year: int
    fire_name: str
    source_hdf5: str
    staged_hdf5: str
    source_fingerprint: str
    source_encoding: str
    days: int
    target_days: int
    zero_target_days: int
    positive_target_pixels: int
    status: str
    error: str


@dataclass(frozen=True)
class RepairDecision:
    status: str
    requested_years: tuple[int, ...]
    files_expected: int
    files_staged: int
    files_verified: int
    excluded_empty_source_directories: tuple[str, ...]
    errors: tuple[str, ...]


def _finite_positive_integers(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.any(np.isinf(array)):
        raise ValueError("invalid active-fire encoding: infinity is not allowed")
    finite = array[np.isfinite(array)]
    if np.any(finite < 0) or np.any(finite != np.floor(finite)):
        raise ValueError("invalid active-fire encoding: values must be nonnegative integers")
    return finite[finite > 0]


def classify_active_fire_encoding(values: np.ndarray) -> ActiveFireEncoding:
    positive = _finite_positive_integers(values)
    if positive.size == 0:
        return "no_positive_values"
    hour_mask = positive <= 23
    if np.all(hour_mask):
        return "hour"
    if np.any(hour_mask):
        raise ValueError("invalid active-fire encoding: mixed hour and HHMM values")
    hours = np.floor_divide(positive, 100)
    minutes = np.mod(positive, 100)
    if np.any(hours > 23) or np.any(minutes > 59):
        raise ValueError("invalid active-fire encoding: invalid HHMM value")
    return "hhmm"


def normalize_active_fire(
    values: np.ndarray,
) -> tuple[np.ndarray, ActiveFireEncoding]:
    encoding = classify_active_fire_encoding(values)
    normalized = np.nan_to_num(np.asarray(values), nan=0.0).copy()
    if encoding == "hhmm":
        normalized = np.floor_divide(normalized, 100)
    if np.any(normalized < 0) or np.any(normalized > 23):
        raise ValueError("invalid active-fire encoding after normalization")
    return normalized, encoding


def source_fingerprint(paths: Sequence[Path], source_root: Path) -> str:
    root = Path(source_root).resolve()
    digest = sha256()
    relatives = sorted(Path(item).resolve().relative_to(root) for item in paths)
    for relative_path in relatives:
        relative = relative_path.as_posix()
        stat = (root / relative_path).stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _decoded_strings(values: object) -> tuple[str, ...]:
    array = np.atleast_1d(values)
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    )


def _active_channel(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape == (height, width, 23):
        return image[..., 22]
    if image.shape == (23, height, width):
        return image[22]
    raise ValueError(
        "TIFF shape must be either "
        f"({height}, {width}, 23) or (23, {height}, {width}); got {image.shape}"
    )


def _target_counts(values: np.ndarray) -> tuple[int, int, int]:
    targets = values[1:]
    target_days = int(targets.shape[0])
    positive_by_day = np.count_nonzero(targets > 0, axis=(1, 2))
    zero_target_days = int(np.count_nonzero(positive_by_day == 0))
    positive_target_pixels = int(np.count_nonzero(targets > 0))
    return target_days, zero_target_days, positive_target_pixels


def _normalized_active_fire(
    tiff_paths: Sequence[Path], height: int, width: int
) -> tuple[np.ndarray, ActiveFireEncoding]:
    active_arrays = [
        _active_channel(np.asarray(tifffile.imread(path)), height, width)
        for path in tiff_paths
    ]
    return normalize_active_fire(np.stack(active_arrays))


def _attribute_values_equal(source_value: object, staged_value: object) -> bool:
    source_array = np.asarray(source_value)
    staged_array = np.asarray(staged_value)
    if source_array.dtype != staged_array.dtype or source_array.shape != staged_array.shape:
        return False
    try:
        return bool(np.array_equal(source_array, staged_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(source_array, staged_array))


def _require_preserved_attributes(
    source_attributes: h5py.AttributeManager,
    staged_attributes: h5py.AttributeManager,
) -> None:
    source_names = set(source_attributes) - _REPAIR_ATTRIBUTES
    staged_names = set(staged_attributes) - _REPAIR_ATTRIBUTES
    if source_names != staged_names or any(
        not _attribute_values_equal(source_attributes[name], staged_attributes[name])
        for name in source_names
    ):
        raise ValueError("staged preserved attributes do not match source HDF5")


def _validate_staged_event(
    staged_path: Path,
    source_hdf5: Path,
    fingerprint: str,
    expected_normalized: np.ndarray | None = None,
    expected_source_encoding: ActiveFireEncoding | None = None,
) -> RepairRecord:
    with h5py.File(source_hdf5, "r") as source, h5py.File(staged_path, "r") as staged:
        source_data = source.get("data")
        staged_data = staged.get("data")
        if not isinstance(source_data, h5py.Dataset) or not isinstance(
            staged_data, h5py.Dataset
        ):
            raise ValueError("data object must be an HDF5 dataset")
        if source_data.shape != staged_data.shape:
            raise ValueError("staged data shape does not match source HDF5 shape")
        if staged_data.compression != "lzf" or not staged_data.shuffle:
            raise ValueError("staged data must preserve LZF compression and shuffle")
        if staged_data.attrs.get("active_fire_repair_version") != REPAIR_VERSION:
            raise ValueError("staged repair version does not match")
        if staged_data.attrs.get("active_fire_stored_encoding") != "hour":
            raise ValueError("staged active-fire encoding must be hour")
        if staged_data.attrs.get("active_fire_source_fingerprint") != fingerprint:
            raise ValueError("staged source fingerprint does not match")
        source_encoding = str(staged_data.attrs.get("active_fire_source_encoding", ""))
        if source_encoding not in {"hour", "hhmm", "no_positive_values"}:
            raise ValueError("staged source encoding is invalid")
        if (
            expected_source_encoding is not None
            and source_encoding != expected_source_encoding
        ):
            raise ValueError("staged source encoding does not match normalized source")
        if int(staged_data.attrs.get("year", -1)) != int(
            source_data.attrs.get("year", -1)
        ) or str(staged_data.attrs.get("fire_name")) != str(
            source_data.attrs.get("fire_name")
        ):
            raise ValueError("staged event identity does not match source HDF5")
        if _decoded_strings(staged_data.attrs.get("img_dates", ())) != _decoded_strings(
            source_data.attrs.get("img_dates", ())
        ):
            raise ValueError("staged dates do not match source HDF5")
        _require_preserved_attributes(source.attrs, staged.attrs)
        _require_preserved_attributes(source_data.attrs, staged_data.attrs)
        if not np.array_equal(
            source_data[:, :22], staged_data[:, :22], equal_nan=True
        ):
            raise ValueError("staged non-active channels do not match source HDF5")
        active = np.asarray(staged_data[:, 22])
        if np.any(~np.isfinite(active)) or np.any(active < 0) or np.any(active > 23):
            raise ValueError("staged active-fire values must be finite and within [0, 23]")
        if np.any(active != np.floor(active)):
            raise ValueError("staged active-fire values must be integers")
        if expected_normalized is not None and not np.array_equal(
            active, expected_normalized
        ):
            raise ValueError("staged active-fire values do not match normalized source")
        target_days, zero_target_days, positive_target_pixels = _target_counts(active)
        year = int(staged_data.attrs["year"])
        fire_name = str(staged_data.attrs["fire_name"])
    return RepairRecord(
        year=year,
        fire_name=fire_name,
        source_hdf5=Path(source_hdf5).as_posix(),
        staged_hdf5=Path(staged_path).as_posix(),
        source_fingerprint=fingerprint,
        source_encoding=source_encoding,
        days=int(active.shape[0]),
        target_days=target_days,
        zero_target_days=zero_target_days,
        positive_target_pixels=positive_target_pixels,
        status="repaired",
        error="",
    )


def stage_event(
    source_event_dir: Path,
    source_hdf5: Path,
    staging_hdf5: Path,
    source_root: Path,
) -> RepairRecord:
    source_event_dir = Path(source_event_dir)
    source_hdf5 = Path(source_hdf5)
    staging_hdf5 = Path(staging_hdf5)
    source_root = Path(source_root)
    temp_path = staging_hdf5.with_name(f"{staging_hdf5.name}.tmp")
    try:
        tiff_paths = sorted(source_event_dir.glob("*.tif"), key=lambda path: path.name)
        if not tiff_paths:
            raise ValueError(f"source event is empty: {source_event_dir}")
        if source_hdf5.resolve() == staging_hdf5.resolve():
            raise ValueError("staging HDF5 must differ from source HDF5")

        fire_name = source_event_dir.name
        try:
            year = int(source_event_dir.parent.name)
        except ValueError as error:
            raise ValueError("source event year directory must be an integer") from error
        with h5py.File(source_hdf5, "r") as handle:
            data = handle.get("data")
            if not isinstance(data, h5py.Dataset):
                raise ValueError("data object must be an HDF5 dataset")
            if source_hdf5.stem != fire_name or str(data.attrs.get("fire_name")) != fire_name:
                raise ValueError("HDF5 filename and fire name must match source event")
            if int(data.attrs.get("year", -1)) != year:
                raise ValueError("HDF5 year must match source event year")
            hdf5_dates = _decoded_strings(data.attrs.get("img_dates", ()))
            tiff_dates = tuple(path.stem for path in tiff_paths)
            if hdf5_dates != tiff_dates:
                raise ValueError("HDF5 img_dates must exactly match TIFF dates")
            if len(data.shape) != 4 or data.shape[0] != len(tiff_paths) or data.shape[1] != 23:
                raise ValueError("HDF5 data shape must be (days, 23, height, width)")
            _, _, height, width = data.shape

        for path in tiff_paths:
            with tifffile.TiffFile(path) as handle:
                shape = handle.series[0].shape
            if shape not in {(height, width, 23), (23, height, width)}:
                raise ValueError(
                    "TIFF shape must match the HDF5 spatial shape and contain 23 channels"
                )

        fingerprint = source_fingerprint(tiff_paths, source_root)
        if staging_hdf5.exists():
            normalized, source_encoding = _normalized_active_fire(
                tiff_paths, height, width
            )
            return replace(
                _validate_staged_event(
                    staging_hdf5,
                    source_hdf5,
                    fingerprint,
                    expected_normalized=normalized,
                    expected_source_encoding=source_encoding,
                ),
                status="verified",
            )

        staging_hdf5.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_hdf5, temp_path)
        normalized, source_encoding = _normalized_active_fire(tiff_paths, height, width)
        with h5py.File(temp_path, "r+") as handle:
            data = handle["data"]
            data[:, 22] = normalized
            data.attrs["active_fire_source_encoding"] = source_encoding
            data.attrs["active_fire_stored_encoding"] = "hour"
            data.attrs["active_fire_repair_version"] = REPAIR_VERSION
            data.attrs["active_fire_source_fingerprint"] = fingerprint
            handle.flush()
        record = _validate_staged_event(
            temp_path,
            source_hdf5,
            fingerprint,
            expected_normalized=normalized,
            expected_source_encoding=source_encoding,
        )
        temp_path.replace(staging_hdf5)
        return replace(record, staged_hdf5=staging_hdf5.as_posix())
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _error_record(
    year: int,
    fire_name: str,
    source_hdf5: Path,
    staged_hdf5: Path,
    days: int,
    error: str,
) -> RepairRecord:
    return RepairRecord(
        year=year,
        fire_name=fire_name,
        source_hdf5=source_hdf5.as_posix(),
        staged_hdf5=staged_hdf5.as_posix(),
        source_fingerprint="",
        source_encoding="",
        days=days,
        target_days=max(days - 1, 0),
        zero_target_days=0,
        positive_target_pixels=0,
        status="error",
        error=error,
    )


def _evidence_temp(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _evidence_backup(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak")


def _remove_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _publish_evidence(paths: Sequence[Path]) -> None:
    preexisting = frozenset(path for path in paths if path.exists())
    backups = tuple(_evidence_backup(path) for path in paths)
    try:
        for path in preexisting:
            shutil.copy2(path, _evidence_backup(path))
        for path in paths:
            _evidence_temp(path).replace(path)
    except OSError:
        rollback_succeeded = False
        try:
            for path in paths:
                backup = _evidence_backup(path)
                if backup.exists():
                    shutil.copy2(backup, path)
                elif path not in preexisting and path.exists():
                    path.unlink()
            rollback_succeeded = True
        finally:
            if rollback_succeeded:
                _remove_paths(backups)
        raise
    else:
        _remove_paths(backups)


def _write_repair_evidence(
    staging_root: Path,
    records: Sequence[RepairRecord],
    decision: RepairDecision,
) -> None:
    manifest = staging_root / "active_fire_repair_manifest.csv"
    decision_path = staging_root / "active_fire_repair_decision.json"
    paths = (manifest, decision_path)
    temp_paths = tuple(_evidence_temp(path) for path in paths)
    backup_paths = tuple(_evidence_backup(path) for path in paths)
    _remove_paths((*temp_paths, *backup_paths))
    try:
        with _evidence_temp(manifest).open("w", newline="", encoding="utf-8") as handle:
            field_names = [field.name for field in fields(RepairRecord)]
            writer = csv.DictWriter(handle, fieldnames=field_names, lineterminator="\n")
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        _evidence_temp(decision_path).write_text(
            json.dumps(asdict(decision), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _publish_evidence(paths)
    finally:
        _remove_paths(temp_paths)


def stage_active_fire_repair(
    source_tiff_root: Path,
    hdf5_root: Path,
    staging_root: Path,
    years: Sequence[int],
) -> RepairDecision:
    requested_years = tuple(years)
    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    if (
        len(set(requested_years)) != len(requested_years)
        or any(not isinstance(year, int) or not 2016 <= year <= 2023 for year in requested_years)
    ):
        decision = RepairDecision(
            status="blocked",
            requested_years=tuple(sorted(requested_years)),
            files_expected=0,
            files_staged=0,
            files_verified=0,
            excluded_empty_source_directories=(),
            errors=("ValueError: years must be unique integers within 2016..2023",),
        )
        _write_repair_evidence(staging_root, (), decision)
        return decision
    sorted_years = tuple(sorted(requested_years))
    source_tiff_root = Path(source_tiff_root)
    hdf5_root = Path(hdf5_root)

    records: list[RepairRecord] = []
    errors: list[str] = []
    exclusions: list[str] = []
    files_expected = 0
    for year in sorted_years:
        source_year = source_tiff_root / str(year)
        hdf5_year = hdf5_root / str(year)
        source_events: dict[str, Path] = {}
        if source_year.is_dir():
            for event_dir in sorted(
                (path for path in source_year.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            ):
                if any(event_dir.glob("*.tif")):
                    source_events[event_dir.name] = event_dir
                else:
                    exclusions.append(event_dir.relative_to(source_tiff_root).as_posix())
        hdf5_events = {
            path.stem: path
            for path in sorted(hdf5_year.glob("*.hdf5"), key=lambda path: path.name)
        }
        files_expected += len(hdf5_events)

        for fire_name in sorted(source_events.keys() - hdf5_events.keys()):
            errors.append(
                f"{year}/{fire_name}: nonempty source directory without HDF5"
            )
        for fire_name in sorted(hdf5_events.keys() - source_events.keys()):
            error = (
                f"{year}/{fire_name}: HDF5 without a matching nonempty source directory"
            )
            errors.append(error)
            records.append(
                _error_record(
                    year,
                    fire_name,
                    hdf5_events[fire_name],
                    staging_root / str(year) / f"{fire_name}.hdf5",
                    0,
                    error,
                )
            )

        for fire_name in sorted(source_events.keys() & hdf5_events.keys()):
            source_event = source_events[fire_name]
            source_hdf5 = hdf5_events[fire_name]
            staged_hdf5 = staging_root / str(year) / source_hdf5.name
            try:
                record = stage_event(
                    source_event,
                    source_hdf5,
                    staged_hdf5,
                    source_tiff_root,
                )
            except (OSError, TypeError, ValueError) as error:
                error_text = f"{year}/{fire_name}: {type(error).__name__}: {error}"
                errors.append(error_text)
                records.append(
                    _error_record(
                        year,
                        fire_name,
                        source_hdf5,
                        staged_hdf5,
                        len(tuple(source_event.glob("*.tif"))),
                        error_text,
                    )
                )
            else:
                records.append(record)

    ordered_records = tuple(sorted(records, key=lambda record: (record.year, record.fire_name)))
    files_staged = sum(record.status == "repaired" for record in ordered_records)
    files_verified = sum(record.status == "verified" for record in ordered_records)
    ordered_errors = tuple(sorted(errors))
    decision = RepairDecision(
        status=(
            "ready"
            if not ordered_errors and files_staged + files_verified == files_expected
            else "blocked"
        ),
        requested_years=sorted_years,
        files_expected=files_expected,
        files_staged=files_staged,
        files_verified=files_verified,
        excluded_empty_source_directories=tuple(sorted(exclusions)),
        errors=ordered_errors,
    )
    _write_repair_evidence(staging_root, ordered_records, decision)
    return decision

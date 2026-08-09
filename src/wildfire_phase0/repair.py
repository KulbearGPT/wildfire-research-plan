from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

import h5py
import numpy as np
import tifffile

from wildfire_phase0.path_safety import (
    canonical_root,
    require_contained_path,
    require_pairwise_disjoint_roots,
)


ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]
REPAIR_VERSION = "1"
_MANIFEST_NAME = "active_fire_repair_manifest.csv"
_DECISION_NAME = "active_fire_repair_decision.json"
_TRANSACTION_PREFIX = ".active_fire_repair_evidence."
_TRANSACTION_SUFFIX = ".txn.json"
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
    source_event_dir: str | None
    source_hdf5: str | None
    staged_hdf5: str | None
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
    manifest_sha256: str = ""
    generation: str = ""


@dataclass(frozen=True)
class _EvidenceTransaction:
    invocation_id: str
    manifest: Path
    decision: Path
    manifest_temp: Path
    decision_temp: Path
    manifest_backup: Path
    decision_backup: Path
    journal_temp: Path
    journal: Path
    preexisting_manifest: bool
    preexisting_decision: bool
    manifest_sha256: str
    generation: str


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
    source_event_dir: Path,
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
        source_event_dir=Path(source_event_dir).as_posix(),
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
    source_root = canonical_root(Path(source_root), "source TIFF root", must_exist=True)
    source_event_dir = require_contained_path(
        source_root, Path(source_event_dir), "source TIFF root"
    )
    source_hdf5 = Path(source_hdf5).resolve(strict=False)
    staging_hdf5 = Path(staging_hdf5).resolve(strict=False)
    temp_path: Path | None = None
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
                    source_event_dir,
                    source_hdf5,
                    fingerprint,
                    expected_normalized=normalized,
                    expected_source_encoding=source_encoding,
                ),
                status="verified",
            )

        staging_hdf5.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{staging_hdf5.name}.",
            suffix=".tmp",
            dir=staging_hdf5.parent,
        )
        os.close(file_descriptor)
        temp_path = Path(temp_name)
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
            source_event_dir,
            source_hdf5,
            fingerprint,
            expected_normalized=normalized,
            expected_source_encoding=source_encoding,
        )
        temp_path.replace(staging_hdf5)
        return replace(record, staged_hdf5=staging_hdf5.as_posix())
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _error_record(
    year: int,
    fire_name: str,
    source_event_dir: Path | None,
    source_hdf5: Path | None,
    staged_hdf5: Path | None,
    days: int,
    error: str,
) -> RepairRecord:
    return RepairRecord(
        year=year,
        fire_name=fire_name,
        source_event_dir=(
            None if source_event_dir is None else source_event_dir.as_posix()
        ),
        source_hdf5=None if source_hdf5 is None else source_hdf5.as_posix(),
        staged_hdf5=None if staged_hdf5 is None else staged_hdf5.as_posix(),
        source_fingerprint="",
        source_encoding="",
        days=days,
        target_days=max(days - 1, 0),
        zero_target_days=0,
        positive_target_pixels=0,
        status="error",
        error=error,
    )


def _remove_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _evidence_generation(
    manifest_digest: str, decision_payload: Mapping[str, object]
) -> str:
    decision_core = dict(decision_payload)
    decision_core.pop("manifest_sha256", None)
    decision_core.pop("generation", None)
    canonical_core = json.dumps(
        decision_core, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(f"{manifest_digest}\n".encode("ascii") + canonical_core).hexdigest()


def _decision_from_payload(payload: Mapping[str, object]) -> RepairDecision:
    try:
        return RepairDecision(
            status=str(payload["status"]),
            requested_years=tuple(int(year) for year in payload["requested_years"]),
            files_expected=int(payload["files_expected"]),
            files_staged=int(payload["files_staged"]),
            files_verified=int(payload["files_verified"]),
            excluded_empty_source_directories=tuple(
                str(path) for path in payload["excluded_empty_source_directories"]
            ),
            errors=tuple(str(error) for error in payload["errors"]),
            manifest_sha256=str(payload["manifest_sha256"]),
            generation=str(payload["generation"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("repair decision has invalid integrity metadata") from error


def _verify_evidence_paths(manifest: Path, decision_path: Path) -> RepairDecision:
    try:
        manifest_bytes = manifest.read_bytes()
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("repair evidence is incomplete or unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("repair decision must be a JSON object")
    manifest_digest = sha256(manifest_bytes).hexdigest()
    if payload.get("manifest_sha256") != manifest_digest:
        raise ValueError("manifest digest does not match repair decision")
    expected_generation = _evidence_generation(manifest_digest, payload)
    if payload.get("generation") != expected_generation:
        raise ValueError("repair evidence generation does not match decision payload")
    return _decision_from_payload(payload)


def verify_repair_evidence(staging_root: Path) -> RepairDecision:
    """Verify and load the committed manifest/decision evidence generation."""
    root = canonical_root(Path(staging_root), "staging root", must_exist=True)
    manifest = require_contained_path(
        root, root / _MANIFEST_NAME, "staging root"
    )
    decision_path = require_contained_path(
        root, root / _DECISION_NAME, "staging root"
    )
    return _verify_evidence_paths(manifest, decision_path)


def _transaction_path(root: Path, name: str) -> Path:
    return require_contained_path(root, root / name, "staging root")


def _transaction_for(
    root: Path,
    invocation_id: str,
    *,
    preexisting_manifest: bool,
    preexisting_decision: bool,
    manifest_sha256: str,
    generation: str,
) -> _EvidenceTransaction:
    return _EvidenceTransaction(
        invocation_id=invocation_id,
        manifest=_transaction_path(root, _MANIFEST_NAME),
        decision=_transaction_path(root, _DECISION_NAME),
        manifest_temp=_transaction_path(
            root, f".{_MANIFEST_NAME}.{invocation_id}.tmp"
        ),
        decision_temp=_transaction_path(
            root, f".{_DECISION_NAME}.{invocation_id}.tmp"
        ),
        manifest_backup=_transaction_path(
            root, f".{_MANIFEST_NAME}.{invocation_id}.bak"
        ),
        decision_backup=_transaction_path(
            root, f".{_DECISION_NAME}.{invocation_id}.bak"
        ),
        journal_temp=_transaction_path(
            root, f"{_TRANSACTION_PREFIX}{invocation_id}.txn.tmp"
        ),
        journal=_transaction_path(
            root, f"{_TRANSACTION_PREFIX}{invocation_id}{_TRANSACTION_SUFFIX}"
        ),
        preexisting_manifest=preexisting_manifest,
        preexisting_decision=preexisting_decision,
        manifest_sha256=manifest_sha256,
        generation=generation,
    )


def _transaction_sidecars(transaction: _EvidenceTransaction) -> tuple[Path, ...]:
    return (
        transaction.manifest_temp,
        transaction.decision_temp,
        transaction.manifest_backup,
        transaction.decision_backup,
        transaction.journal_temp,
        transaction.journal,
    )


def _new_transaction(
    root: Path, manifest_sha256: str, generation: str
) -> _EvidenceTransaction:
    while True:
        transaction = _transaction_for(
            root,
            uuid4().hex,
            preexisting_manifest=(root / _MANIFEST_NAME).exists(),
            preexisting_decision=(root / _DECISION_NAME).exists(),
            manifest_sha256=manifest_sha256,
            generation=generation,
        )
        if not any(path.exists() for path in _transaction_sidecars(transaction)):
            return transaction


def _reserve_owned_file(path: Path) -> None:
    with path.open("xb"):
        pass


def _journal_payload(transaction: _EvidenceTransaction) -> dict[str, object]:
    return {
        "generation": transaction.generation,
        "invocation_id": transaction.invocation_id,
        "manifest_sha256": transaction.manifest_sha256,
        "preexisting_decision": transaction.preexisting_decision,
        "preexisting_manifest": transaction.preexisting_manifest,
        "version": 1,
    }


def _prepare_transaction(
    transaction: _EvidenceTransaction,
    manifest_bytes: bytes,
    decision_bytes: bytes,
) -> None:
    created: list[Path] = []
    try:
        for path, content in (
            (transaction.manifest_temp, manifest_bytes),
            (transaction.decision_temp, decision_bytes),
        ):
            _reserve_owned_file(path)
            created.append(path)
            path.write_bytes(content)
        for preexisting, final_path, backup_path in (
            (
                transaction.preexisting_manifest,
                transaction.manifest,
                transaction.manifest_backup,
            ),
            (
                transaction.preexisting_decision,
                transaction.decision,
                transaction.decision_backup,
            ),
        ):
            if preexisting:
                _reserve_owned_file(backup_path)
                created.append(backup_path)
                shutil.copy2(final_path, backup_path)
        _reserve_owned_file(transaction.journal_temp)
        created.append(transaction.journal_temp)
        transaction.journal_temp.write_text(
            json.dumps(_journal_payload(transaction), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        transaction.journal_temp.replace(transaction.journal)
    except Exception:
        _remove_paths(tuple(created))
        raise


def _rollback_transaction(transaction: _EvidenceTransaction) -> None:
    for preexisting, final_path, backup_path in (
        (
            transaction.preexisting_manifest,
            transaction.manifest,
            transaction.manifest_backup,
        ),
        (
            transaction.preexisting_decision,
            transaction.decision,
            transaction.decision_backup,
        ),
    ):
        if preexisting:
            if not backup_path.is_file():
                raise ValueError("repair evidence transaction backup is missing")
            shutil.copy2(backup_path, final_path)
        elif final_path.exists():
            final_path.unlink()


def _cleanup_transaction(transaction: _EvidenceTransaction) -> None:
    _remove_paths(_transaction_sidecars(transaction))


def _publish_transaction(transaction: _EvidenceTransaction) -> None:
    try:
        transaction.manifest_temp.replace(transaction.manifest)
        transaction.decision_temp.replace(transaction.decision)
        committed = _verify_evidence_paths(
            transaction.manifest, transaction.decision
        )
        if committed.generation != transaction.generation:
            raise ValueError("published repair evidence has the wrong generation")
    except (OSError, ValueError):
        _rollback_transaction(transaction)
        _cleanup_transaction(transaction)
        raise
    else:
        _cleanup_transaction(transaction)


def _valid_transaction_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _reserved_transaction_invocation_id(name: str) -> str | None:
    for prefix, suffix in (
        (f".{_MANIFEST_NAME}.", ".tmp"),
        (f".{_DECISION_NAME}.", ".tmp"),
        (f".{_MANIFEST_NAME}.", ".bak"),
        (f".{_DECISION_NAME}.", ".bak"),
        (_TRANSACTION_PREFIX, ".txn.tmp"),
        (_TRANSACTION_PREFIX, _TRANSACTION_SUFFIX),
    ):
        if name.startswith(prefix) and name.endswith(suffix):
            invocation_id = name[len(prefix) : -len(suffix)]
            if _valid_transaction_id(invocation_id):
                return invocation_id
    return None


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_transaction(root: Path, journal: Path) -> _EvidenceTransaction:
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unrecognized repair evidence transaction: {journal.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"unrecognized repair evidence transaction: {journal.name}")
    invocation_id = payload.get("invocation_id")
    if (
        payload.get("version") != 1
        or not _valid_transaction_id(invocation_id)
        or not _valid_digest(payload.get("manifest_sha256"))
        or not _valid_digest(payload.get("generation"))
        or type(payload.get("preexisting_manifest")) is not bool
        or type(payload.get("preexisting_decision")) is not bool
    ):
        raise ValueError(f"unrecognized repair evidence transaction: {journal.name}")
    transaction = _transaction_for(
        root,
        str(invocation_id),
        preexisting_manifest=bool(payload["preexisting_manifest"]),
        preexisting_decision=bool(payload["preexisting_decision"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        generation=str(payload["generation"]),
    )
    if journal not in (transaction.journal, transaction.journal_temp):
        raise ValueError(f"unrecognized repair evidence transaction: {journal.name}")
    return transaction


def _recover_transaction(transaction: _EvidenceTransaction) -> None:
    manifest_candidates = tuple(
        path
        for path in (transaction.manifest, transaction.manifest_temp)
        if path.is_file()
        and sha256(path.read_bytes()).hexdigest() == transaction.manifest_sha256
    )
    decision_candidates = tuple(
        path
        for path in (transaction.decision, transaction.decision_temp)
        if path.is_file()
    )
    target_pair: tuple[Path, Path] | None = None
    for manifest_candidate in manifest_candidates:
        for decision_candidate in decision_candidates:
            try:
                decision = _verify_evidence_paths(
                    manifest_candidate, decision_candidate
                )
            except ValueError:
                continue
            if decision.generation == transaction.generation:
                target_pair = (manifest_candidate, decision_candidate)
                break
        if target_pair is not None:
            break
    if target_pair is None:
        _rollback_transaction(transaction)
        _cleanup_transaction(transaction)
        return
    manifest_candidate, decision_candidate = target_pair
    if manifest_candidate != transaction.manifest:
        manifest_candidate.replace(transaction.manifest)
    if decision_candidate != transaction.decision:
        decision_candidate.replace(transaction.decision)
    committed = _verify_evidence_paths(transaction.manifest, transaction.decision)
    if committed.generation != transaction.generation:
        raise ValueError("recovered repair evidence has the wrong generation")
    _cleanup_transaction(transaction)


def _recover_interrupted_evidence(staging_root: Path) -> None:
    reserved_invocation_ids = {
        invocation_id
        for path in staging_root.iterdir()
        if (
            invocation_id := _reserved_transaction_invocation_id(path.name)
        )
        is not None
    }
    journals = tuple(
        sorted(
            staging_root.glob(f"{_TRANSACTION_PREFIX}*{_TRANSACTION_SUFFIX}"),
            key=lambda path: path.name,
        )
    )
    journal_temps = tuple(
        sorted(
            staging_root.glob(f"{_TRANSACTION_PREFIX}*.txn.tmp"),
            key=lambda path: path.name,
        )
    )
    if len(journals) + len(journal_temps) > 1 or len(reserved_invocation_ids) > 1:
        raise ValueError("multiple interrupted repair evidence transactions found")
    if not journals and not journal_temps and reserved_invocation_ids:
        raise ValueError("incomplete interrupted repair evidence transaction found")
    if journals:
        journal = require_contained_path(staging_root, journals[0], "staging root")
        _recover_transaction(_load_transaction(staging_root, journal))
    elif journal_temps:
        journal_temp = require_contained_path(
            staging_root, journal_temps[0], "staging root"
        )
        transaction = _load_transaction(staging_root, journal_temp)
        journal_temp.replace(transaction.journal)
        _recover_transaction(transaction)


def _render_manifest(records: Sequence[RepairRecord]) -> bytes:
    buffer = io.StringIO(newline="")
    field_names = [field.name for field in fields(RepairRecord)]
    writer = csv.DictWriter(buffer, fieldnames=field_names, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(record) for record in records)
    return buffer.getvalue().encode("utf-8")


def _write_repair_evidence(
    staging_root: Path,
    records: Sequence[RepairRecord],
    decision: RepairDecision,
) -> RepairDecision:
    manifest = require_contained_path(
        staging_root,
        staging_root / "active_fire_repair_manifest.csv",
        "staging root",
    )
    decision_path = require_contained_path(
        staging_root,
        staging_root / "active_fire_repair_decision.json",
        "staging root",
    )
    manifest_bytes = _render_manifest(records)
    manifest_digest = sha256(manifest_bytes).hexdigest()
    decision_payload = asdict(decision)
    committed_decision = replace(
        decision,
        manifest_sha256=manifest_digest,
        generation=_evidence_generation(manifest_digest, decision_payload),
    )
    decision_bytes = (
        json.dumps(asdict(committed_decision), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    transaction = _new_transaction(
        staging_root, committed_decision.manifest_sha256, committed_decision.generation
    )
    if transaction.manifest != manifest or transaction.decision != decision_path:
        raise AssertionError("repair evidence transaction paths changed unexpectedly")
    _prepare_transaction(transaction, manifest_bytes, decision_bytes)
    _publish_transaction(transaction)
    return committed_decision


def stage_active_fire_repair(
    source_tiff_root: Path,
    hdf5_root: Path,
    staging_root: Path,
    years: Sequence[int],
) -> RepairDecision:
    requested_years = tuple(years)
    source_tiff_root = canonical_root(Path(source_tiff_root), "source TIFF root")
    hdf5_root = canonical_root(Path(hdf5_root), "HDF5 root")
    staging_root = canonical_root(Path(staging_root), "staging root")
    require_pairwise_disjoint_roots(
        {
            "source TIFF root": source_tiff_root,
            "HDF5 root": hdf5_root,
            "staging root": staging_root,
        }
    )
    require_contained_path(
        staging_root,
        staging_root / "active_fire_repair_manifest.csv",
        "staging root",
    )
    require_contained_path(
        staging_root,
        staging_root / "active_fire_repair_decision.json",
        "staging root",
    )
    if staging_root.is_dir():
        _recover_interrupted_evidence(staging_root)
    for year in requested_years:
        source_year = require_contained_path(
            source_tiff_root, source_tiff_root / str(year), "source TIFF root"
        )
        if source_year.is_dir():
            for event_entry in source_year.iterdir():
                event_path = require_contained_path(
                    source_tiff_root, event_entry, "source TIFF root"
                )
                if event_path.is_dir():
                    for tiff_path in event_path.glob("*.tif"):
                        require_contained_path(
                            source_tiff_root, tiff_path, "source TIFF root"
                        )
        hdf5_year = require_contained_path(
            hdf5_root, hdf5_root / str(year), "HDF5 root"
        )
        if hdf5_year.is_dir():
            for hdf5_path in hdf5_year.glob("*.hdf5"):
                require_contained_path(hdf5_root, hdf5_path, "HDF5 root")
        require_contained_path(
            staging_root, staging_root / str(year), "staging root"
        )
    staging_root.mkdir(parents=True, exist_ok=True)
    if not requested_years:
        decision = RepairDecision(
            status="blocked",
            requested_years=(),
            files_expected=0,
            files_staged=0,
            files_verified=0,
            excluded_empty_source_directories=(),
            errors=("ValueError: years must not be empty",),
        )
        return _write_repair_evidence(staging_root, (), decision)
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
        return _write_repair_evidence(staging_root, (), decision)
    sorted_years = tuple(sorted(requested_years))
    root_errors = []
    if not source_tiff_root.is_dir():
        root_errors.append(
            "ValueError: source TIFF root must be an existing directory"
        )
    if not hdf5_root.is_dir():
        root_errors.append("ValueError: HDF5 root must be an existing directory")
    if root_errors:
        decision = RepairDecision(
            status="blocked",
            requested_years=sorted_years,
            files_expected=0,
            files_staged=0,
            files_verified=0,
            excluded_empty_source_directories=(),
            errors=tuple(sorted(root_errors)),
        )
        return _write_repair_evidence(staging_root, (), decision)

    records: list[RepairRecord] = []
    errors: list[str] = []
    exclusions: list[str] = []
    files_expected = 0
    for year in sorted_years:
        source_year = source_tiff_root / str(year)
        hdf5_year = hdf5_root / str(year)
        source_year_exists = source_year.is_dir()
        hdf5_year_exists = hdf5_year.is_dir()
        if not source_year_exists:
            errors.append(
                f"ValueError: source TIFF year directory does not exist: {year}"
            )
        if not hdf5_year_exists:
            errors.append(
                f"ValueError: HDF5 year directory does not exist: {year}"
            )
        source_events: dict[str, Path] = {}
        if source_year_exists:
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
        if source_year_exists and hdf5_year_exists and not (
            source_events.keys() & hdf5_events.keys()
        ):
            errors.append(
                f"ValueError: requested year has no matched nonempty events: {year}"
            )

        for fire_name in sorted(source_events.keys() - hdf5_events.keys()):
            error = (
                f"{year}/{fire_name}: nonempty source directory without HDF5"
            )
            errors.append(error)
            source_event = source_events[fire_name]
            records.append(
                _error_record(
                    year,
                    fire_name,
                    source_event,
                    None,
                    None,
                    len(tuple(source_event.glob("*.tif"))),
                    error,
                )
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
                    None,
                    hdf5_events[fire_name],
                    None,
                    0,
                    error,
                )
            )

        for fire_name in sorted(source_events.keys() & hdf5_events.keys()):
            source_event = source_events[fire_name]
            source_hdf5 = hdf5_events[fire_name]
            staged_hdf5 = require_contained_path(
                staging_root,
                staging_root / str(year) / source_hdf5.name,
                "staging root",
            )
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
                        source_event,
                        source_hdf5,
                        staged_hdf5 if staged_hdf5.is_file() else None,
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
    return _write_repair_evidence(staging_root, ordered_records, decision)

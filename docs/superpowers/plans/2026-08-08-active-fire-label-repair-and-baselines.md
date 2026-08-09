# Active-Fire Label Repair and Rule Baselines Implementation Plan

> **Execution note:** This combined draft is superseded by
> `2026-08-08-active-fire-label-repair.md` and
> `2026-08-08-rule-baseline-evaluation.md`, which add explicit empty-source
> exclusions, independent repair verification, and smaller reviewable tasks.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the four WSTS+ added-year active-fire label channels through a staged, reversible workflow, strengthen Phase 0 so invalid target years cannot pass, and produce deterministic real-data no-fire and persistence baseline results.

**Architecture:** Add a focused `active_fire` module that normalizes source encodings and stages event-level HDF5 repairs without touching active data. Extend the existing inventory scan with target statistics and enforce year/split target integrity in the Phase 0 CLI. After staged data pass independent acceptance checks and are activated with recoverable directory renames, add a streaming rule-baseline evaluator that reads only channel 22 and writes atomic CSV/JSON/Markdown artifacts.

**Tech Stack:** Python 3.13, NumPy 2.x, h5py 3.x, pandas 2.x, scikit-learn 1.x, tifffile, imagecodecs, pytest, standard-library argparse/hashlib/json/shutil, PowerShell for same-volume activation.

## Global Constraints

- Keep WSTS+ as the primary dataset and retain the approved target, backbone, lightweight gating idea, frozen `2016--2020 train / 2021 validation / 2022--2023 test` split, and ten-month research route.
- Never modify an active source HDF5 file in place; repair only a copied file below an explicit staging root.
- Preserve channels 0--21, event identity, dates, spatial shape, LZF compression, shuffle, ROI attributes, and all unrelated dataset attributes.
- Store active-fire values as integer-valued hours in `[0, 23]`, replacing source NaNs with zero.
- Infer encoding per event: all positives in `[1, 23]` means `hour`; all positives greater than 23 and valid HHMM means `hhmm`; no positives means `no_positive_values`; mixed or invalid values are errors.
- Do not activate staged data if any repair error, temporary file, file-count mismatch, label-count mismatch, or schema mismatch remains.
- Keep the four original added-year HDF5 directories under `D:\WildFire Project\data\hdf5-active-fire-bug-backup` until repaired-data audit and baseline smoke evaluation both pass.
- Do not tune a model, score threshold, calibration, or preprocessing choice on 2022--2023 targets.
- Commit after each independently testable task; stage only the files named by that task.

---

### Task 1: Active-Fire Encoding Primitives

**Files:**
- Modify: `pyproject.toml`
- Create: `src/wildfire_phase0/active_fire.py`
- Create: `tests/test_active_fire.py`

**Interfaces:**
- Produces: `ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]`.
- Produces: `normalize_active_fire(values: numpy.ndarray) -> tuple[numpy.ndarray, ActiveFireEncoding]`.
- Produces: `source_fingerprint(paths: Sequence[Path], source_root: Path) -> str`.
- Later tasks consume normalized arrays with the original shape and `float32` dtype.

- [ ] **Step 1: Add bounded TIFF decoder dependencies and failing encoding tests**

Add these project dependencies:

```toml
  "tifffile>=2025.10,<2027",
  "imagecodecs>=2025.11,<2027",
```

Create `tests/test_active_fire.py` with:

```python
from pathlib import Path

import numpy as np
import pytest

from wildfire_phase0.active_fire import normalize_active_fire, source_fingerprint


def test_hour_encoding_is_preserved_and_nan_becomes_zero() -> None:
    values = np.array([[np.nan, 0, 6, 22]], dtype=np.float32)
    normalized, encoding = normalize_active_fire(values)
    assert encoding == "hour"
    assert normalized.tolist() == [[0.0, 0.0, 6.0, 22.0]]


def test_hhmm_encoding_is_converted_once() -> None:
    values = np.array([[0, 806, 1359, 2200]], dtype=np.float32)
    normalized, encoding = normalize_active_fire(values)
    assert encoding == "hhmm"
    assert normalized.tolist() == [[0.0, 8.0, 13.0, 22.0]]


def test_zero_only_encoding_is_explicit() -> None:
    normalized, encoding = normalize_active_fire(np.zeros((2, 2), dtype=np.float32))
    assert encoding == "no_positive_values"
    assert not normalized.any()


@pytest.mark.parametrize(
    "values",
    [
        np.array([[8, 1300]], dtype=np.float32),
        np.array([[-1, 0]], dtype=np.float32),
        np.array([[12.5, 0]], dtype=np.float32),
        np.array([[1261, 0]], dtype=np.float32),
        np.array([[2400, 0]], dtype=np.float32),
        np.array([[np.inf, 0]], dtype=np.float32),
    ],
)
def test_invalid_or_mixed_encoding_is_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="active-fire"):
        normalize_active_fire(values)


def test_source_fingerprint_is_ordered_and_root_relative(tmp_path: Path) -> None:
    first = tmp_path / "2016" / "fire_a" / "2016-01-01.tif"
    second = tmp_path / "2016" / "fire_a" / "2016-01-02.tif"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"bb")
    value = source_fingerprint((first, second), tmp_path)
    assert len(value) == 64
    assert value == source_fingerprint((first, second), tmp_path)
    assert value != source_fingerprint((second, first), tmp_path)
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run:

```powershell
python -m pytest tests/test_active_fire.py -v
```

Expected: collection fails because `wildfire_phase0.active_fire` does not exist.

- [ ] **Step 3: Implement normalization and fingerprinting**

Create `src/wildfire_phase0/active_fire.py` with these public definitions and rules:

```python
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
from typing import Literal

import numpy as np


ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]


def normalize_active_fire(values: np.ndarray) -> tuple[np.ndarray, ActiveFireEncoding]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or np.isinf(array).any():
        raise ValueError("active-fire values must be a non-empty finite-or-NaN array")
    finite = array[np.isfinite(array)]
    if (finite < 0).any() or not np.equal(finite, np.floor(finite)).all():
        raise ValueError("active-fire values must be non-negative integers")

    positive = finite[finite > 0]
    normalized = np.nan_to_num(array, nan=0.0).astype(np.float32, copy=True)
    if positive.size == 0:
        return normalized, "no_positive_values"

    hour_values = positive <= 23
    if hour_values.all():
        return normalized, "hour"
    if hour_values.any():
        raise ValueError("active-fire values mix hour and HHMM encodings")

    hours = np.floor_divide(positive.astype(np.int64), 100)
    minutes = np.remainder(positive.astype(np.int64), 100)
    if (hours > 23).any() or (minutes > 59).any():
        raise ValueError("active-fire HHMM values are out of range")
    normalized = np.floor_divide(normalized, 100).astype(np.float32, copy=False)
    return normalized, "hhmm"


def source_fingerprint(paths: Sequence[Path], source_root: Path) -> str:
    root = Path(source_root).resolve()
    digest = hashlib.sha256()
    for path in paths:
        item = Path(path).resolve()
        try:
            relative = item.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("source TIFF must be inside source_root") from error
        stat = item.stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_active_fire.py -v
python -m pytest -q
```

Expected: all active-fire tests pass and the existing 80-test suite remains green.

- [ ] **Step 5: Commit the primitives**

```powershell
git add pyproject.toml src/wildfire_phase0/active_fire.py tests/test_active_fire.py
git commit -m "feat: validate active-fire encodings"
```

---

### Task 2: Event-Atomic Staging Engine

**Files:**
- Modify: `src/wildfire_phase0/active_fire.py`
- Modify: `tests/test_active_fire.py`

**Interfaces:**
- Produces: immutable `RepairRecord` and `RepairFailure` dataclasses.
- Produces: `stage_event_active_fire(source_event_dir: Path, source_hdf5: Path, staging_hdf5: Path, source_root: Path) -> RepairRecord`.
- Produces: `stage_active_fire_repair(source_tiff_root: Path, hdf5_root: Path, staging_root: Path, years: Sequence[int]) -> tuple[tuple[RepairRecord, ...], tuple[RepairFailure, ...]]`.
- The engine writes only below `staging_root`; source HDF5 files are read and copied but never opened writable.

- [ ] **Step 1: Add failing synthetic TIFF/HDF5 staging tests**

Append these fixture helpers and tests using `tifffile.imwrite` for `(height, width, 23)` arrays and `h5py` for `(days, 23, height, width)` source files:

```python
import hashlib

import h5py
import tifffile

from wildfire_phase0.active_fire import stage_active_fire_repair, stage_event_active_fire


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_repair_fixture(
    tmp_path: Path, fire_name: str = "fire_demo", year: int = 2022
) -> tuple[Path, Path, Path]:
    raw_event = tmp_path / "raw" / str(year) / fire_name
    raw_event.mkdir(parents=True)
    dates = (f"{year}-08-01", f"{year}-08-02")
    for date, hour in zip(dates, (8, 20)):
        image = np.arange(4 * 5 * 23, dtype=np.float32).reshape(4, 5, 23)
        image[..., 22] = hour
        tifffile.imwrite(raw_event / f"{date}.tif", image)

    source = tmp_path / "hdf5" / str(year) / f"{fire_name}.hdf5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as handle:
        data = handle.create_dataset(
            "data", data=np.moveaxis(np.stack([
                tifffile.imread(raw_event / f"{dates[0]}.tif"),
                tifffile.imread(raw_event / f"{dates[1]}.tif"),
            ]), -1, 1), compression="lzf", shuffle=True,
        )
        data[:, 22] = 0
        data.attrs["year"] = year
        data.attrs["fire_name"] = fire_name
        data.attrs["img_dates"] = dates
        data.attrs["lnglat"] = [np.nan, np.nan]
    staged = tmp_path / "staging" / str(year) / f"{fire_name}.hdf5"
    return raw_event, source, staged


def test_stage_event_repairs_only_channel_22(tmp_path: Path) -> None:
    raw_event, source, staged = _write_repair_fixture(tmp_path)
    before = _sha256(source)
    record = stage_event_active_fire(raw_event, source, staged, tmp_path / "raw")
    assert _sha256(source) == before
    with h5py.File(staged, "r") as handle:
        values = handle["data"]
        assert values[:, 22, 0, 0].tolist() == [8.0, 20.0]
        assert values.compression == "lzf" and values.shuffle
        assert values.attrs["active_fire_source_encoding"] == "hour"
        assert values.attrs["active_fire_stored_encoding"] == "hour"
    assert record.positive_target_pixels == 20
    assert record.zero_target_days == 0


def test_stage_event_rejects_date_mismatch_and_removes_temp(tmp_path: Path) -> None:
    raw_event, source, staged = _write_repair_fixture(tmp_path)
    with h5py.File(source, "r+") as handle:
        handle["data"].attrs["img_dates"] = ["2022-08-02", "2022-08-03"]
    with pytest.raises(ValueError, match="dates"):
        stage_event_active_fire(raw_event, source, staged, tmp_path / "raw")
    assert not staged.exists()
    assert not staged.with_name(staged.name + ".tmp").exists()


def test_stage_event_rejects_source_shape_mismatch(tmp_path: Path) -> None:
    raw_event, source, staged = _write_repair_fixture(tmp_path)
    replacement = np.zeros((5, 5, 23), dtype=np.float32)
    tifffile.imwrite(raw_event / "2022-08-02.tif", replacement)
    with pytest.raises(ValueError, match="shape"):
        stage_event_active_fire(raw_event, source, staged, tmp_path / "raw")


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [("year", 2023, "year"), ("fire_name", "other_fire", "fire_name")],
)
def test_stage_event_rejects_hdf5_identity_mismatch(
    tmp_path: Path, attribute: str, value: object, message: str
) -> None:
    raw_event, source, staged = _write_repair_fixture(tmp_path)
    with h5py.File(source, "r+") as handle:
        handle["data"].attrs[attribute] = value
    with pytest.raises(ValueError, match=message):
        stage_event_active_fire(raw_event, source, staged, tmp_path / "raw")


def test_existing_stage_rejects_changed_source_fingerprint(tmp_path: Path) -> None:
    raw_event, source, staged = _write_repair_fixture(tmp_path)
    stage_event_active_fire(raw_event, source, staged, tmp_path / "raw")
    changed = raw_event / "2022-08-02.tif"
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="fingerprint"):
        stage_event_active_fire(raw_event, source, staged, tmp_path / "raw")


def test_batch_collects_one_success_and_one_failure(tmp_path: Path) -> None:
    _write_repair_fixture(tmp_path, "fire_a")
    raw_b, _, _ = _write_repair_fixture(tmp_path, "fire_b")
    (raw_b / "2022-08-02.tif").write_bytes(b"not-a-tiff")
    records, failures = stage_active_fire_repair(
        tmp_path / "raw", tmp_path / "hdf5", tmp_path / "staging", (2022,)
    )
    assert [record.fire_name for record in records] == ["fire_a"]
    assert [failure.fire_name for failure in failures] == ["fire_b"]
```

- [ ] **Step 2: Run the new staging tests and verify they fail**

```powershell
python -m pytest tests/test_active_fire.py -k "stage" -v
```

Expected: failures name the missing `RepairRecord`, `stage_event_active_fire`, or `stage_active_fire_repair` interface.

- [ ] **Step 3: Implement the staging dataclasses and event repair**

Add these dataclasses and stable version constant:

```python
from dataclasses import dataclass
from shutil import copy2

import h5py
import tifffile


ACTIVE_FIRE_REPAIR_VERSION = "1"


@dataclass(frozen=True)
class RepairRecord:
    year: int
    fire_name: str
    source_hdf5: Path
    staged_hdf5: Path
    n_days: int
    source_encoding: ActiveFireEncoding
    source_fingerprint: str
    positive_target_pixels: int
    zero_target_days: int
    status: Literal["converted", "verified_existing"]


@dataclass(frozen=True)
class RepairFailure:
    year: int
    fire_name: str
    source_event_dir: Path
    source_hdf5: Path
    error: str
```

`stage_event_active_fire` must perform these exact operations:

```python
tiff_paths = tuple(sorted(source_event_dir.glob("*.tif")))
if not tiff_paths:
    raise ValueError("source event has no TIFF files")
dates = tuple(path.stem for path in tiff_paths)
fingerprint = source_fingerprint(tiff_paths, source_root)
raw_days = []
for path in tiff_paths:
    image = tifffile.imread(path)
    if image.ndim != 3:
        raise ValueError("source TIFF must be three-dimensional")
    if image.shape[-1] == 23:
        active = image[..., 22]
    elif image.shape[0] == 23:
        active = image[22]
    else:
        raise ValueError("source TIFF must contain 23 bands")
    raw_days.append(np.asarray(active, dtype=np.float32))
raw_active = np.stack(raw_days)
normalized, encoding = normalize_active_fire(raw_active)
```

Validate source HDF5 key/type/shape/attributes and exact dates before copying. Copy to `staging_hdf5.with_name(staging_hdf5.name + ".tmp")`, open only the temporary copy as `r+`, write `data[:, 22] = normalized`, and set the three repair attributes plus `active_fire_source_fingerprint`. Reopen the temporary file read-only and require exact normalized equality, unchanged non-label shape, LZF compression, and shuffle before replacing `staging_hdf5`. Always unlink a surviving temporary file in `finally`.

When `staging_hdf5` already exists, reopen it and return `verified_existing` only if repair version, fingerprint, source/stored encodings, dates, shape, positive-target count, and zero-target-day count all match the source. Otherwise raise `ValueError` without overwriting it.

- [ ] **Step 4: Implement deterministic batch traversal**

`stage_active_fire_repair` must sort unique requested years, reject years outside `2016--2023`, iterate source HDF5 files by `(year, filename)`, resolve the matching raw directory `<source_tiff_root>/<year>/<fire_name>`, and collect a `RepairFailure` for each caught `OSError`, `TypeError`, or `ValueError`. Reject an empty HDF5 year before iteration. Return records and failures sorted by `(year, fire_name)`.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m pytest tests/test_active_fire.py -v
python -m pytest -q
```

Expected: all tests pass; source-file hash tests prove staging is non-mutating.

- [ ] **Step 6: Commit the staging engine**

```powershell
git add src/wildfire_phase0/active_fire.py tests/test_active_fire.py
git commit -m "feat: stage active-fire label repairs"
```

---

### Task 3: Repair CLI and Atomic Evidence

**Files:**
- Modify: `src/wildfire_phase0/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/experiments/phase0.md`

**Interfaces:**
- Produces CLI subcommand `repair-active-fire` with required `--source-tiff-root`, `--hdf5-root`, `--staging-root`, and one-or-more integer `--years`.
- Produces `active_fire_repair_manifest.csv` and `active_fire_repair_decision.json` below `staging_root`.
- Produces `run_repair_active_fire(source_tiff_root: Path, hdf5_root: Path, staging_root: Path, years: Sequence[int]) -> int`, returning `0` only with no failures and the expected staged-file count.

- [ ] **Step 1: Add a failing end-to-end CLI test**

Add a two-event synthetic fixture and invoke:

```python
exit_code = cli.main([
    "repair-active-fire",
    "--source-tiff-root", str(raw_root),
    "--hdf5-root", str(hdf5_root),
    "--staging-root", str(staging_root),
    "--years", "2022",
])
assert exit_code == 0
manifest = pd.read_csv(staging_root / "active_fire_repair_manifest.csv")
assert manifest[["year", "fire_name", "source_encoding", "status"]].to_dict("records") == [
    {"year": 2022, "fire_name": "fire_a", "source_encoding": "hour", "status": "converted"},
    {"year": 2022, "fire_name": "fire_b", "source_encoding": "hour", "status": "converted"},
]
decision = json.loads((staging_root / "active_fire_repair_decision.json").read_text())
assert decision == {"errors": [], "failed": 0, "staged": 2, "status": "complete", "years": [2022]}
```

Add this failure-path assertion after creating a valid two-event fixture:

```python
bad_tiff = raw_root / "2022" / "fire_b" / "2022-08-02.tif"
bad_tiff.write_bytes(b"not-a-tiff")
exit_code = cli.main([
    "repair-active-fire",
    "--source-tiff-root", str(raw_root),
    "--hdf5-root", str(hdf5_root),
    "--staging-root", str(staging_root),
    "--years", "2022",
])
assert exit_code == 2
decision = json.loads((staging_root / "active_fire_repair_decision.json").read_text())
assert decision["status"] == "failed"
assert decision["failed"] == 1
assert decision["errors"][0]["year"] == 2022
assert decision["errors"][0]["fire_name"] == "fire_b"
assert decision["errors"][0]["source_event_dir"].endswith("2022/fire_b")
assert decision["errors"][0]["source_hdf5"].endswith("2022/fire_b.hdf5")
assert "TiffFileError" in decision["errors"][0]["error"]
```

- [ ] **Step 2: Run the CLI tests and verify they fail**

```powershell
python -m pytest tests/test_cli.py -k "repair_active_fire" -v
```

Expected: parser rejects the unknown subcommand or `run_repair_active_fire` is absent.

- [ ] **Step 3: Implement atomic repair artifacts and parser dispatch**

Add artifact names:

```python
_REPAIR_ARTIFACT_NAMES = (
    "active_fire_repair_manifest.csv",
    "active_fire_repair_decision.json",
)
```

Build manifest rows with these exact columns:

```python
(
    "year", "fire_name", "source_hdf5", "staged_hdf5", "n_days",
    "source_encoding", "source_fingerprint", "positive_target_pixels",
    "zero_target_days", "status",
)
```

Serialize relative paths against their respective roots. Write both artifacts to sibling `.tmp` paths and publish them through the existing backup-and-rollback helper. Decision JSON contains sorted error dictionaries and keys `status`, `years`, `staged`, `failed`, and `errors`. Return `2` on any failure.

Extend `_parser()` with:

```python
repair = subparsers.add_parser(
    "repair-active-fire", help="Stage repaired active-fire channels without activating data."
)
repair.add_argument("--source-tiff-root", type=Path, required=True)
repair.add_argument("--hdf5-root", type=Path, required=True)
repair.add_argument("--staging-root", type=Path, required=True)
repair.add_argument("--years", type=int, nargs="+", required=True)
```

Dispatch to `run_repair_active_fire` from `main`.

- [ ] **Step 4: Document the staging command and explicit non-activation guarantee**

Add the exact operator command to `docs/experiments/phase0.md` and state that it only writes staging data, never renames or deletes active year directories, and requires manifest/decision validation before activation.

- [ ] **Step 5: Run tests and commit**

```powershell
python -m pytest tests/test_cli.py tests/test_active_fire.py -v
python -m pytest -q
git diff --check
git add src/wildfire_phase0/cli.py tests/test_cli.py docs/experiments/phase0.md
git commit -m "feat: add active-fire repair command"
```

---

### Task 4: Target Statistics and Blocking Gate

**Files:**
- Modify: `src/wildfire_phase0/schema.py`
- Modify: `src/wildfire_phase0/inventory.py`
- Modify: `src/wildfire_phase0/cli.py`
- Modify: `src/wildfire_phase0/report.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_inventory.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Extends `EventInventory` with `target_days`, `zero_target_days`, `positive_target_pixels`, `active_fire_min_positive`, and `active_fire_max_positive`.
- Produces `_target_integrity_error(inventory: Sequence[EventInventory], split_manifest: pandas.DataFrame) -> str | None`.
- Inventory CSV and Markdown report expose target statistics by event, year, and split.

- [ ] **Step 1: Add failing inventory tests for target statistics and stored encoding validation**

Update the synthetic HDF5 helper so day 0 contains one active pixel, day 1 is zero-target, and day 2 contains two active pixels with hours 8 and 20. Assert:

```python
item = inspect_hdf5(path, tmp_path)
assert item.target_days == 2
assert item.zero_target_days == 1
assert item.positive_target_pixels == 2
assert item.active_fire_min_positive == 8
assert item.active_fire_max_positive == 20
```

Add parameterized files containing active-fire NaN, infinity, fractional value, negative value, and 24; each must raise `ValueError` naming stored active-fire values.

- [ ] **Step 2: Add failing CLI tests for all-zero year/split and missing benchmark year**

Build eight tiny synthetic year directories. Give every year one positive next-day target for the passing case. Add three failing cases:

```python
assert _run_audit(data_root_with_zero_2022, output_root) == 2
assert "zero positive target pixels: 2022" in report

assert _run_audit(data_root_with_zero_test_split, output_root) == 2
assert "zero positive target pixels: test" in report

assert _run_audit(data_root_missing_2023, output_root) == 2
assert "missing benchmark years: 2023" in report
```

Keep a 2017 event with no positive target alongside another positive 2017 event and assert the year and train split remain non-blocking.

- [ ] **Step 3: Run focused tests and verify they fail**

```powershell
python -m pytest tests/test_schema.py tests/test_inventory.py tests/test_cli.py tests/test_report.py -v
```

Expected: assertions fail because target fields, validation, and report sections are absent.

- [ ] **Step 4: Extend schema and compute target statistics in the existing daily scan**

Add fields to `EventInventory` after `nan_fraction` in this order:

```python
target_days: int
zero_target_days: int
positive_target_pixels: int
active_fire_min_positive: float | None
active_fire_max_positive: float | None
```

In `inspect_hdf5`, reuse each loaded `day = data[day_index]` for NaN counting. Validate `active = day[22]` is finite, integer-valued, and in `[0, 23]`. For `day_index > 0`, increment target days, positive pixels, zero-target days, and min/max positive values. Update `validate_inventory_row`, all constructors, `_INVENTORY_COLUMNS`, and `_inventory_frame` with exact integer/range invariants:

```python
if target_days != max(n_days - 1, 0):
    raise ValueError("target_days must equal n_days minus one")
if not 0 <= zero_target_days <= target_days:
    raise ValueError("zero_target_days must be between zero and target_days")
if positive_target_pixels < 0:
    raise ValueError("positive_target_pixels must be non-negative")
```

Require both min/max to be `None` when there are no positive target pixels; otherwise require `1 <= min <= max <= 23`.

- [ ] **Step 5: Implement benchmark-year and split gates**

Use exact expected years `frozenset(range(2016, 2024))`. `_target_integrity_error` returns one deterministic semicolon-separated message listing:

- missing benchmark years;
- years with zero positive target pixels;
- frozen splits with zero positive target pixels.

In `run_audit`, call it after `build_forward_split`. Convert a non-`None` message into `ValueError(message)` and reuse `_blocked_decision`. Do not reject individual zero-positive events or days.

- [ ] **Step 6: Render target tables and rerun tests**

Add Markdown sections `Target summary by year` and `Target summary by split` with columns `events`, `target days`, `zero-target days`, and `positive target pixels`. Ensure empty/blocked reports render deterministic zero-row tables.

Run:

```powershell
python -m pytest tests/test_schema.py tests/test_inventory.py tests/test_cli.py tests/test_report.py -v
python -m pytest -q
```

Expected: all tests pass and an all-zero year cannot return `continue_controlled`.

- [ ] **Step 7: Commit the strengthened gate**

```powershell
git add src/wildfire_phase0/schema.py src/wildfire_phase0/inventory.py src/wildfire_phase0/cli.py src/wildfire_phase0/report.py tests/test_schema.py tests/test_inventory.py tests/test_cli.py tests/test_report.py
git commit -m "fix: block invalid active-fire targets"
```

---

### Task 5: Stage, Validate, Activate, and Re-Audit Real Data

**Files:**
- Modify: `docs/experiments/phase0.md`
- Modify: `index.html`
- Derived only: `artifacts/phase0/*` (ignored; do not commit)
- External data only: `D:\WildFire Project\data\hdf5-active-fire-staging`, `D:\WildFire Project\data\hdf5-active-fire-bug-backup`, and active HDF5 year directories.

**Interfaces:**
- Consumes the repair CLI and target-aware Phase 0 audit from Tasks 3--4.
- Produces an activated, recoverable eight-year HDF5 dataset and corrected Phase 0 evidence.

- [ ] **Step 1: Run fresh code verification before touching real data**

```powershell
python -m pytest -q
git status --short --branch
```

Expected: all tests pass and the branch is clean.

- [ ] **Step 2: Verify exact real-data roots and free space**

Read-only checks must prove:

- source TIFF root is `D:\WildFire Project\data\raw\WSTSPlus`;
- active HDF5 root is `D:\WildFire Project\data\hdf5`;
- staging and backup roots do not yet exist;
- active counts are 92/110/122/68 for 2016/2017/2022/2023;
- at least 30 GiB is free on drive D.

Stop on any mismatch.

Run this exact check:

```powershell
$sourceRoot = 'D:\WildFire Project\data\raw\WSTSPlus'
$activeRoot = 'D:\WildFire Project\data\hdf5'
$stagingRoot = 'D:\WildFire Project\data\hdf5-active-fire-staging'
$backupRoot = 'D:\WildFire Project\data\hdf5-active-fire-bug-backup'
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) { throw 'missing source root' }
if (-not (Test-Path -LiteralPath $activeRoot -PathType Container)) { throw 'missing active root' }
if (Test-Path -LiteralPath $stagingRoot) { throw 'staging root already exists' }
if (Test-Path -LiteralPath $backupRoot) { throw 'backup root already exists' }
$expected = @{2016=92; 2017=110; 2022=122; 2023=68}
foreach ($year in $expected.Keys) {
  $count = @(Get-ChildItem -LiteralPath (Join-Path $activeRoot $year) -File -Filter '*.hdf5').Count
  if ($count -ne $expected[$year]) { throw "unexpected active count for $year`: $count" }
}
if ((Get-PSDrive -Name D).Free -lt 30GB) { throw 'less than 30 GiB free on D' }
```

- [ ] **Step 3: Run the staged repair**

```powershell
python -m wildfire_phase0.cli repair-active-fire `
  --source-tiff-root "D:\WildFire Project\data\raw\WSTSPlus" `
  --hdf5-root "D:\WildFire Project\data\hdf5" `
  --staging-root "D:\WildFire Project\data\hdf5-active-fire-staging" `
  --years 2016 2017 2022 2023
```

Expected: exit `0`, 392 staged records, no failures, no `.tmp` files.

- [ ] **Step 4: Independently validate staging acceptance criteria**

Use a read-only verification script independent of repair manifest aggregation. Require exact values:

| year | files | target days | zero-target days | positive target pixels |
| --- | ---: | ---: | ---: | ---: |
| 2016 | 92 | 2,102 | 886 | 303,648 |
| 2017 | 110 | 2,490 | 1,481 | 177,972 |
| 2022 | 122 | 3,424 | 2,158 | 105,377 |
| 2023 | 68 | 2,442 | 1,297 | 167,952 |

Open all 392 staged files and require `data` shape `(days, 23, height, width)`, LZF, shuffle, repair attributes, stored active-fire range `[0, 23]`, and no temporary files. Compare the first and last day of the first and last event in each year against source TIFFs: exact channels 0--21 and normalized-exact channel 22.

Run this independent verifier from the repository environment:

```powershell
@'
from pathlib import Path
import h5py
import numpy as np
import tifffile
from wildfire_phase0.active_fire import normalize_active_fire

raw_root = Path(r'D:\WildFire Project\data\raw\WSTSPlus')
staged_root = Path(r'D:\WildFire Project\data\hdf5-active-fire-staging')
expected = {
    2016: (92, 2102, 886, 303648),
    2017: (110, 2490, 1481, 177972),
    2022: (122, 3424, 2158, 105377),
    2023: (68, 2442, 1297, 167952),
}
assert not list(staged_root.rglob('*.tmp'))
for year, wanted in expected.items():
    files = sorted((staged_root / str(year)).glob('*.hdf5'))
    target_days = zero_days = positives = 0
    for path in files:
        with h5py.File(path, 'r') as handle:
            data = handle['data']
            assert data.ndim == 4 and data.shape[1] == 23
            assert data.compression == 'lzf' and data.shuffle
            assert data.attrs['active_fire_stored_encoding'] == 'hour'
            assert data.attrs['active_fire_repair_version'] == '1'
            active = data[:, 22]
            assert np.isfinite(active).all()
            assert np.equal(active, np.floor(active)).all()
            assert 0 <= active.min() <= active.max() <= 23
            day_counts = np.count_nonzero(active[1:] > 0, axis=(1, 2))
            target_days += len(day_counts)
            zero_days += int(np.count_nonzero(day_counts == 0))
            positives += int(day_counts.sum())
    assert (len(files), target_days, zero_days, positives) == wanted

    raw_events = sorted(p for p in (raw_root / str(year)).iterdir() if p.is_dir() and any(p.glob('*.tif')))
    for event in (raw_events[0], raw_events[-1]):
        tiffs = sorted(event.glob('*.tif'))
        with h5py.File(staged_root / str(year) / f'{event.name}.hdf5', 'r') as handle:
            data = handle['data']
            for index in (0, len(tiffs) - 1):
                image = tifffile.imread(tiffs[index])
                raw = np.moveaxis(image, -1, 0) if image.shape[-1] == 23 else image
                normalized, _ = normalize_active_fire(raw[22])
                assert np.array_equal(data[index, :22], raw[:22], equal_nan=True)
                assert np.array_equal(data[index, 22], normalized)
print('staging acceptance: ok')
'@ | python -
```

- [ ] **Step 5: Activate using recoverable same-volume renames**

Resolve and verify all exact paths before moving. Create only the backup root. For each year, move active to backup, then staged to active. If the second move fails, immediately move that year's backup directory back. Do not delete backup or source TIFF data.

```powershell
$activeRoot = 'D:\WildFire Project\data\hdf5'
$stagingRoot = 'D:\WildFire Project\data\hdf5-active-fire-staging'
$backupRoot = 'D:\WildFire Project\data\hdf5-active-fire-bug-backup'
New-Item -ItemType Directory -Path $backupRoot -ErrorAction Stop | Out-Null
foreach ($year in 2016,2017,2022,2023) {
  $activeYear = Join-Path $activeRoot $year
  $stagedYear = Join-Path $stagingRoot $year
  $backupYear = Join-Path $backupRoot $year
  if (-not (Test-Path -LiteralPath $activeYear) -or
      -not (Test-Path -LiteralPath $stagedYear) -or
      (Test-Path -LiteralPath $backupYear)) { throw "unsafe activation state for $year" }
  Move-Item -LiteralPath $activeYear -Destination $backupYear
  try { Move-Item -LiteralPath $stagedYear -Destination $activeYear }
  catch { Move-Item -LiteralPath $backupYear -Destination $activeYear; throw }
}
```

- [ ] **Step 6: Run the full target-aware Phase 0 audit**

```powershell
python -m wildfire_phase0.cli audit `
  --data-root "D:\WildFire Project\data\hdf5" `
  --output-root "artifacts\phase0"
```

Expected: exit `0`, 999 events, split counts 653/156/190, every year and split has positive labels, and the contract decision remains `continue_controlled` only because availability/QA/coverage/target-validity provenance remains unavailable.

- [ ] **Step 7: Correct the website and experiment documentation**

Update both files to state:

- the original added-year conversion double-converted hour labels;
- staged repair restored the four exact label totals;
- Phase 0 now blocks missing/all-zero benchmark years and splits;
- the research route remains unchanged but model experiments use only repaired data;
- derived audit artifacts and the 22.31 GiB backup remain local.

Update the bilingual translation map in `index.html` for every new visible Chinese text node.

- [ ] **Step 8: Verify and commit the corrected real-data result**

```powershell
python -m pytest -q
git diff --check
node -e "const fs=require('fs'),vm=require('vm');const s=fs.readFileSync('index.html','utf8');const m=s.match(/<script>([\s\S]*?)<\/script>/);if(!m)throw new Error('script missing');new vm.Script(m[1]);"
git add docs/experiments/phase0.md index.html
git commit -m "docs: correct phase0 target audit"
```

Keep backup data and the staging root outside git.

---

### Task 6: Streaming T=1 Rule-Baseline Evaluation

**Files:**
- Create: `src/wildfire_phase0/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `src/wildfire_phase0/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/experiments/phase0.md`
- Modify: `index.html`

**Interfaces:**
- Produces: `BinaryScoreCounts` with `true_positive_score_one`, `false_positive_score_one`, `positives`, and `total`.
- Produces: `binary_score_average_precision(counts: BinaryScoreCounts) -> float` without retaining all pixels.
- Produces: `evaluate_rule_baselines(data_root: Path, split_manifest: pandas.DataFrame) -> pandas.DataFrame`.
- Produces CLI `baseline --data-root PATH --split-manifest PATH --output-root PATH` and atomic `baseline_results.csv`, `baseline_results.json`, and `baseline_report.md`.

- [ ] **Step 1: Write failing count-based AP tests**

Create `tests/test_evaluation.py`:

```python
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from wildfire_phase0.evaluation import (
    BinaryScoreCounts,
    binary_score_average_precision,
    evaluate_rule_baselines,
)


def test_binary_count_ap_matches_sklearn() -> None:
    target = np.array([1, 1, 0, 0, 1, 0])
    score = np.array([1, 0, 1, 0, 1, 0])
    counts = BinaryScoreCounts(
        true_positive_score_one=2,
        false_positive_score_one=1,
        positives=3,
        total=6,
    )
    assert binary_score_average_precision(counts) == pytest.approx(
        average_precision_score(target, score)
    )


def test_binary_count_ap_is_nan_without_positive_target() -> None:
    counts = BinaryScoreCounts(0, 2, 0, 8)
    assert np.isnan(binary_score_average_precision(counts))
```

- [ ] **Step 2: Write a failing synthetic event evaluation test**

Append this exact two-split fixture and evaluation test:

```python
def _write_active_event(path: Path, year: int, fire_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((3, 23, 2, 2), dtype=np.float32)
    values[0, 22, 0, 0] = 8
    values[1, 22, 0, 0] = 9
    values[1, 22, 0, 1] = 9
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values)
        data.attrs["year"] = year
        data.attrs["fire_name"] = fire_name
        data.attrs["img_dates"] = [
            f"{year}-08-01", f"{year}-08-02", f"{year}-08-03"
        ]
        data.attrs["lnglat"] = [-120.0, 54.0]


def test_rule_baselines_stream_events_and_report_zero_target_far(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_active_event(data_root / "2020" / "train_fire.hdf5", 2020, "train_fire")
    _write_active_event(data_root / "2022" / "test_fire.hdf5", 2022, "test_fire")
    split_manifest = pd.DataFrame([
        {"event_id": "2020:train_fire", "year": 2020, "fire_name": "train_fire",
         "path": "2020/train_fire.hdf5", "split": "train"},
        {"event_id": "2022:test_fire", "year": 2022, "fire_name": "test_fire",
         "path": "2022/test_fire.hdf5", "split": "test"},
    ])
    result = evaluate_rule_baselines(data_root, split_manifest)
    assert result[["model", "split"]].to_records(index=False).tolist() == [
        ("no_fire", "train"), ("persistence_latest", "train"),
        ("no_fire", "test"), ("persistence_latest", "test"),
    ]
    persistence = result[
        (result["model"] == "persistence_latest") & (result["split"] == "test")
    ].iloc[0]
    assert persistence["target_days"] == 2
    assert persistence["zero_target_days"] == 1
    assert persistence["positive_pixels"] == 2
    assert persistence["zero_target_false_alarm_rate"] == pytest.approx(0.5)
    assert persistence["event_macro_ap"] == pytest.approx(7 / 24)
    no_fire = result[(result["model"] == "no_fire") & (result["split"] == "test")].iloc[0]
    assert no_fire["zero_target_false_alarm_rate"] == 0.0
    assert no_fire["pooled_ap"] == pytest.approx(0.25)
```

Assert the result columns are exactly:

```python
(
    "model", "split", "events", "defined_ap_events", "undefined_ap_events",
    "target_days", "zero_target_days", "positive_pixels", "total_pixels",
    "prevalence", "pooled_ap", "event_macro_ap", "zero_target_false_alarm_rate",
)
```

Assert no-fire FAR is zero and persistence FAR equals the hand-computed fraction.

- [ ] **Step 3: Run tests and verify they fail**

```powershell
python -m pytest tests/test_evaluation.py -v
```

Expected: collection fails because `wildfire_phase0.evaluation` does not exist.

- [ ] **Step 4: Implement streaming binary-score metrics and evaluator**

For a binary-score count summary with positives `P`, total `N`, high-score true positives `TP1`, and high-score false positives `FP1`, compute exact scikit-learn AP as:

```python
if P == 0:
    return float("nan")
prevalence = P / N
recall_at_one = TP1 / P
precision_at_one = TP1 / (TP1 + FP1) if TP1 + FP1 else 1.0
return recall_at_one * precision_at_one + (1 - recall_at_one) * prevalence
```

Validate non-negative counts, `TP1 <= P`, `TP1 + FP1 <= N`, and `P <= N`.

`evaluate_rule_baselines` must:

1. Validate unique split-manifest paths and split values.
2. Open one event at a time and read only `data[:, 22, :, :]`.
3. For each target day `t >= 1`, use `target = active[t] > 0`, no-fire scores all zero, and latest persistence scores `active[t - 1] > 0`.
4. Aggregate exact binary counts per event and split without retaining cross-event pixel arrays.
5. Average AP only across events with positive targets; report the defined and undefined counts explicitly.
6. Aggregate false alarms only across zero-positive target days, weighted by their pixels.
7. Return deterministic model/split row order.

- [ ] **Step 5: Add atomic baseline CLI artifacts**

Add parser arguments and `run_baseline`. Serialize the dataframe to CSV, JSON records, and a Markdown report containing metric definitions, split/year provenance, undefined AP counts, and a statement that no threshold or parameter was selected on test. Reuse the existing multi-artifact atomic publish helper and return `2` on invalid data.

- [ ] **Step 6: Run tests and commit the evaluator**

```powershell
python -m pytest tests/test_evaluation.py tests/test_cli.py -v
python -m pytest -q
git diff --check
git add src/wildfire_phase0/evaluation.py tests/test_evaluation.py src/wildfire_phase0/cli.py tests/test_cli.py
git commit -m "feat: evaluate rule baselines"
```

- [ ] **Step 7: Run real baselines on repaired data**

```powershell
python -m wildfire_phase0.cli baseline `
  --data-root "D:\WildFire Project\data\hdf5" `
  --split-manifest "artifacts\phase0\split_manifest.csv" `
  --output-root "artifacts\phase0\baselines"
```

Expected: exit `0`; six rows cover two models across train/validation/test; every row reports event coverage, prevalence, pooled AP, event-macro AP, and zero-target FAR.

- [ ] **Step 8: Publish measured baseline results and commit**

Update `docs/experiments/phase0.md` and the visible bilingual Phase 0 website callout with exact baseline values copied from generated artifacts. State that these are deterministic T=1 references, not learned-model results. Run full tests, HTML/JavaScript static checks, and `git diff --check`, then:

```powershell
git add docs/experiments/phase0.md index.html
git commit -m "docs: publish rule baseline results"
```

Do not commit generated baseline artifacts or source/backup data.

---

## Plan Self-Review

- Spec coverage: Tasks 1--3 implement encoding detection, staging, fingerprints, atomic evidence, and explicit non-activation. Task 4 implements the target-integrity gate. Task 5 performs reversible real-data activation and corrects Phase 0 evidence. Task 6 runs and publishes the approved deterministic next experiment.
- Safety: no implementation step opens active source HDF5 files writable; activation uses exact same-volume paths and retains the complete backup.
- Type consistency: `ActiveFireEncoding`, `RepairRecord`, `RepairFailure`, inventory field names, CLI artifact names, and baseline result columns are defined once and consumed with the same spelling in later tasks.
- Scope: learned neural baselines, controlled corruption, gating models, bootstrap inference, and TS-SatFire remain excluded.
- Placeholder scan: the plan contains no deferred implementation markers; every task names files, interfaces, tests, commands, expected outcomes, and a commit boundary.

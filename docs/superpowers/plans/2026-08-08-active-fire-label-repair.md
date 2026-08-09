# Active-Fire Label Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested staged repair for the four WSTS+ added-year active-fire channels, strengthen Phase 0 so an all-zero year or split is blocked, activate only validated data, and publish the corrected audit result.

**Architecture:** Add a focused `repair.py` module that classifies source active-fire encoding, stages event-atomic copies, and emits deterministic repair evidence without activating data. Extend the existing inventory scan with target statistics and add a separate target-integrity decision that the CLI combines with the contract decision. Real activation remains an explicit same-volume directory rename after all 392 staged events satisfy independently derived counts.

**Tech Stack:** Python 3.13, NumPy 2.x, h5py 3.x, pandas 2.x, tifffile, imagecodecs, pytest, standard-library `argparse`, `csv`, `dataclasses`, `hashlib`, `json`, `pathlib`, and `shutil`.

## Global Constraints

- Keep WSTS+ as the primary dataset and retain the approved forecasting target, frozen `2016--2020 train / 2021 validation / 2022--2023 test` split, backbone route, gating idea, and ten-month schedule.
- Repair only channel 22 in the four added years `2016`, `2017`, `2022`, and `2023`; preserve channels 0--21, event identity, dates, shape, attributes, LZF compression, and shuffle.
- Source hour encoding contains finite positive integers only in `[1, 23]`; source HHMM encoding contains finite positive integers all greater than 23 with valid hours `[0, 23]` and minutes `[0, 59]`; mixed encodings are rejected.
- Stored active-fire data are integer-valued hours in `[0, 23]`, with source NaNs replaced by zero.
- The repair command writes only to a caller-supplied staging root. It never activates, deletes, or overwrites an active year directory.
- Use `hdf5-active-fire-bug-backup` as the exact deterministic backup root during manual activation, and retain it through the repaired audit and baseline smoke run.
- Use SHA-256 over ordered source-relative TIFF paths, file sizes, and nanosecond modification timestamps as the source fingerprint.
- A present benchmark year or frozen split with zero positive next-day target pixels is a blocking data error; individual zero-positive events and days remain valid and must be reported.
- Staged real-data acceptance requires 392 HDF5 files, no `*.tmp`, no repair errors, LZF plus shuffle, and exact source-derived label counts: 2016 `303648`, 2017 `177972`, 2022 `105377`, 2023 `167952`.
- Source data and active HDF5 files are immutable until the explicit activation step. All repository file edits use `apply_patch`; all commits are small and intentional.

---

### Task 1: Active-Fire Encoding and Normalization Core

**Files:**
- Create: `src/wildfire_phase0/repair.py`
- Create: `tests/test_repair.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]`.
- Produces: `classify_active_fire_encoding(values: numpy.ndarray) -> ActiveFireEncoding`.
- Produces: `normalize_active_fire(values: numpy.ndarray) -> tuple[numpy.ndarray, ActiveFireEncoding]`.
- Produces: `source_fingerprint(paths: Sequence[Path], source_root: Path) -> str`.
- Adds bounded runtime dependencies `tifffile>=2025.10,<2027` and `imagecodecs>=2025.11,<2027`.

- [ ] **Step 1: Write the failing encoding tests**

Add these tests to `tests/test_repair.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from wildfire_phase0.repair import (
    classify_active_fire_encoding,
    normalize_active_fire,
    source_fingerprint,
)


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
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
python -m pytest tests/test_repair.py -v
```

Expected: collection fails because `wildfire_phase0.repair` does not exist.

- [ ] **Step 3: Add bounded TIFF dependencies**

Modify `pyproject.toml` dependencies to include exactly:

```toml
  "tifffile>=2025.10,<2027",
  "imagecodecs>=2025.11,<2027",
```

Run:

```powershell
python -m pip install -e ".[dev]"
```

Expected: editable install succeeds without changing the existing NumPy major version.

- [ ] **Step 4: Implement the minimal normalization core**

Create `src/wildfire_phase0/repair.py` with these public functions and rules:

```python
from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np


ActiveFireEncoding = Literal["hour", "hhmm", "no_positive_values"]
REPAIR_VERSION = "1"


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
```

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_repair.py -v
python -m pytest -q
```

Expected: new tests pass and the existing suite remains green.

- [ ] **Step 6: Commit the normalization core**

```powershell
git add pyproject.toml src/wildfire_phase0/repair.py tests/test_repair.py
git commit -m "feat: classify active-fire source encoding"
```

---

### Task 2: Event-Atomic Staging, Manifest, and Repair CLI

**Files:**
- Modify: `src/wildfire_phase0/repair.py`
- Modify: `src/wildfire_phase0/cli.py`
- Modify: `tests/test_repair.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `normalize_active_fire` and `source_fingerprint` from Task 1.
- Produces: immutable `RepairRecord` and `RepairDecision` dataclasses.
- Produces: `stage_event(source_event_dir: Path, source_hdf5: Path, staging_hdf5: Path, source_root: Path) -> RepairRecord`.
- Produces: `stage_active_fire_repair(source_tiff_root: Path, hdf5_root: Path, staging_root: Path, years: Sequence[int]) -> RepairDecision`.
- Produces CLI: `python -m wildfire_phase0.cli repair-active-fire --source-tiff-root PATH --hdf5-root PATH --staging-root PATH --years 2016 2017 2022 2023`.
- Writes `active_fire_repair_manifest.csv` and `active_fire_repair_decision.json` under the staging root.

- [ ] **Step 1: Add failing event-staging tests**

Extend `tests/test_repair.py` with fixture helpers that use `tifffile.imwrite` to
write HWC 23-band TIFFs and `h5py` to write CHW event HDF5 files. Add tests:

```python
import hashlib

import h5py
import tifffile

from wildfire_phase0.repair import stage_event


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        tifffile.imwrite(event_dir / f"{date}.tif", image)


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
    with h5py.File(source_hdf5, "r") as source, h5py.File(staging_hdf5, "r") as staged:
        np.testing.assert_array_equal(source["data"][:, :22], staged["data"][:, :22])
        assert staged["data"][:, 22, 0, 0].tolist() == [0.0, 6.0, 22.0]
        assert staged["data"].attrs["active_fire_source_encoding"] == "hour"
        assert staged["data"].attrs["active_fire_stored_encoding"] == "hour"


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
```

Also add tests for shape mismatch, filename/fire-name mismatch, a valid
preexisting staged file with matching fingerprint returning `status="verified"`,
and a stale preexisting staged file being rejected.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```powershell
python -m pytest tests/test_repair.py -v
```

Expected: import or attribute failure because staging interfaces do not exist.

- [ ] **Step 3: Implement event staging**

Add dataclasses with exact fields:

```python
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
```

Implement `stage_event` in this order:

1. Sort `*.tif` by filename and reject an empty event.
2. Read and decode HDF5 `img_dates`; require exact equality with TIFF stems.
3. Require HDF5 shape `(len(tiffs), 23, height, width)` and each TIFF shape
   `(height, width, 23)` or `(23, height, width)`.
4. Compute one source fingerprint for the ordered event TIFFs.
5. Copy `source_hdf5` to `staging_hdf5.name + ".tmp"` using `shutil.copy2`.
6. Read each TIFF with `tifffile.imread`, extract channel 22, and collect all
   active arrays before classification so mixed event encoding is rejected.
7. Normalize once, write staged `data[:, 22]`, and add the three repair attrs
   plus `active_fire_source_fingerprint`.
8. Compute next-day counts from normalized days 1 onward.
9. Flush, close, reopen read-only, validate attrs/range/counts/compression, then
   replace the final staged path.
10. Always remove the sibling temp in `finally`.

- [ ] **Step 4: Add failing dataset-level and CLI tests**

Add a synthetic source with one valid event in each requested year. Assert:

```python
decision = stage_active_fire_repair(source_tiff_root, hdf5_root, staging_root, (2016, 2022))
assert decision.status == "ready"
assert decision.files_expected == 2
assert decision.files_staged == 2
assert not list(staging_root.rglob("*.tmp"))
assert (staging_root / "active_fire_repair_manifest.csv").is_file()
assert (staging_root / "active_fire_repair_decision.json").is_file()
```

Add a CLI test calling `cli.main([...])` with `repair-active-fire`; require exit
`0` for `ready`, exit `2` for any repair error, complete evidence artifacts in
both cases, and source HDF5 hashes unchanged.

- [ ] **Step 5: Implement dataset orchestration and CLI parsing**

`stage_active_fire_repair` must:

- validate unique years within `2016..2023` and sort them;
- map every existing source HDF5 to exactly one nonempty TIFF event directory;
- record empty source directories as sorted nonblocking exclusions and require
  exactly the five known exclusions in the real-data acceptance check;
- treat any nonempty source directory without HDF5, or HDF5 without a matching
  nonempty source directory, as an error;
- call `stage_event` deterministically by `(year, fire_name)`;
- write the manifest with the exact `RepairRecord` field order;
- write the decision JSON with sorted keys and a final newline;
- stage both evidence files through sibling `.tmp` paths before replacing them;
- return `ready` only when every source HDF5 has a valid staged file and there
  are no errors;
- never rename or remove active data directories.

Extend `_parser()` and `main()` in `cli.py` with the exact arguments in the
interface block. Return `0` for `ready` and `2` otherwise.

- [ ] **Step 6: Run focused and full verification**

Run:

```powershell
python -m pytest tests/test_repair.py tests/test_cli.py -v
python -m pytest -q
```

Expected: all tests pass; no fixture leaves `.tmp` files.

- [ ] **Step 7: Commit the staged repair workflow**

```powershell
git add src/wildfire_phase0/repair.py src/wildfire_phase0/cli.py tests/test_repair.py tests/test_cli.py
git commit -m "feat: stage active-fire label repairs"
```

---

### Task 3: Target-Aware Inventory and Blocking Gate

**Files:**
- Modify: `src/wildfire_phase0/schema.py`
- Modify: `src/wildfire_phase0/inventory.py`
- Create: `src/wildfire_phase0/target_gate.py`
- Modify: `src/wildfire_phase0/cli.py`
- Modify: `src/wildfire_phase0/report.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_inventory.py`
- Create: `tests/test_target_gate.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Extends `EventInventory` with `target_days: int`, `zero_target_days: int`, `positive_target_pixels: int`, `active_fire_min_positive: float | None`, and `active_fire_max_positive: float | None`.
- Produces: `target_integrity_errors(inventory: Sequence[EventInventory], split_manifest: pandas.DataFrame) -> tuple[str, ...]`.
- Extends `inventory.csv` with the five target fields.
- Extends `phase0_report.md` with target counts by year and split.

- [ ] **Step 1: Write failing schema and inventory tests**

Update the inventory fixture so day 1 contains active-fire hour `9` at one
pixel. Assert:

```python
item = inspect_hdf5(path, tmp_path)
assert item.target_days == 2
assert item.zero_target_days == 1
assert item.positive_target_pixels == 1
assert item.active_fire_min_positive == 9
assert item.active_fire_max_positive == 9
```

Add tests that reject stored positive active-fire values above 23, negative
values, and non-integer values. Extend schema tests to reject negative counts,
`zero_target_days > target_days`, and non-`None` min/max for zero-positive
events.

- [ ] **Step 2: Run focused tests to verify RED**

Run:

```powershell
python -m pytest tests/test_schema.py tests/test_inventory.py -v
```

Expected: failures because the target fields do not exist.

- [ ] **Step 3: Compute target evidence during the existing daily scan**

Extend `EventInventory` and `validate_inventory_row`. In `inspect_hdf5`, while
each `data[day_index]` array is already in memory for NaN counting:

```python
day = np.asarray(data[day_index])
nan_count += int(np.isnan(day).sum())
active = day[22]
finite_active = active[np.isfinite(active)]
if np.any(finite_active < 0) or np.any(finite_active != np.floor(finite_active)):
    raise ValueError("stored active-fire values must be nonnegative integer hours")
if np.any(finite_active > 23):
    raise ValueError("stored active-fire values must be within 0-23 hours")
if day_index > 0:
    positives = int(np.count_nonzero(finite_active > 0))
    target_days += 1
    zero_target_days += int(positives == 0)
    positive_target_pixels += positives
```

Track the global positive min/max over all days, not only target days, because
the stored encoding contract applies to inputs and labels.

- [ ] **Step 4: Write failing target-gate tests**

Create `tests/test_target_gate.py` with typed inventory fixtures. Required cases:

```python
def test_gate_blocks_present_year_and_split_with_zero_positive_targets() -> None:
    inventory = [
        _event(2016, "train_positive", positives=1),
        _event(2021, "validation_positive", positives=1),
        _event(2022, "test_zero", positives=0),
    ]
    split = build_forward_split(inventory)
    errors = target_integrity_errors(inventory, split)
    assert any("year 2022" in error for error in errors)
    assert any("split test" in error for error in errors)


def test_gate_allows_zero_positive_events_when_year_and_split_are_positive() -> None:
    inventory = [
        _event(2017, "zero_event", positives=0),
        _event(2017, "positive_event", positives=2),
    ]
    assert target_integrity_errors(inventory, build_forward_split(inventory)) == ()
```

Also test empty inventory returns no target error so the existing invalid-file
error remains the primary blocker.

- [ ] **Step 5: Implement target integrity and combine it with the gate**

Implement `target_integrity_errors` deterministically:

- aggregate positive target pixels for every present year;
- aggregate by `split_manifest.split` through event IDs, rejecting any manifest
  mismatch;
- emit sorted, exact messages for each present year or split with total zero;
- do not reject individual zero-positive events or days.

In `run_audit`, call it only after inventory and split construction succeed. If
errors exist, create one `ValueError("; ".join(errors))`, set the invalid-file
error, and call `_blocked_decision`. This preserves the existing atomic artifact
behavior and exit code `2`.

- [ ] **Step 6: Extend CSV and report tests before implementation**

Update `_INVENTORY_COLUMNS`, `_inventory_frame`, and report expectations. The
Markdown target section must contain tables with columns:

```text
year | events | target days | zero-target days | positive target pixels
split | events | target days | zero-target days | positive target pixels
```

Update valid CLI fixtures to contain at least one positive target in every
present year/split. Add a CLI regression test with all-zero 2022 data that
asserts exit `2`, complete artifacts, and exact blocker text in both report and
decision notes.

- [ ] **Step 7: Run all focused and full tests**

Run:

```powershell
python -m pytest tests/test_schema.py tests/test_inventory.py tests/test_target_gate.py tests/test_report.py tests/test_cli.py -v
python -m pytest -q
```

Expected: all tests pass and artifact rollback tests remain green.

- [ ] **Step 8: Commit the strengthened data gate**

```powershell
git add src/wildfire_phase0/schema.py src/wildfire_phase0/inventory.py src/wildfire_phase0/target_gate.py src/wildfire_phase0/cli.py src/wildfire_phase0/report.py tests/test_schema.py tests/test_inventory.py tests/test_target_gate.py tests/test_cli.py tests/test_report.py
git commit -m "fix: block invalid active-fire targets"
```

---

### Task 4: Independent Verifier, Operator Documentation, and End-to-End Test

**Files:**
- Create: `src/wildfire_phase0/verify_repair.py`
- Create: `tests/test_verify_repair.py`
- Modify: `docs/experiments/phase0.md`
- Modify: `index.html`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces independent CLI: `python -m wildfire_phase0.verify_repair --hdf5-root PATH --source-tiff-root PATH --expect YEAR:FILES:TARGET_DAYS:ZERO_DAYS:POSITIVE_PIXELS [...]`.
- The verifier imports neither `wildfire_phase0.repair` nor `wildfire_phase0.inventory` and emits a JSON summary to stdout; exit `0` means every expectation passed.
- Documents the repair command, non-activation guarantee, exact activation preconditions, backup root, rollback, and audit command.
- Adds a synthetic end-to-end test that stages data and then audits the staged tree.

- [ ] **Step 1: Write failing independent-verifier tests**

Create `tests/test_verify_repair.py` with synthetic staged HDF5 and raw TIFF
fixtures. Import only `wildfire_phase0.verify_repair` and assert:

```python
def test_verifier_checks_counts_compression_range_and_raw_samples(tmp_path: Path) -> None:
    # One two-day 2016 event: raw hour labels 0 and 9, staged labels 0 and 9.
    _write_raw_event(source_root / "2016" / "fire_a", [0.0, 9.0])
    _write_staged_event(hdf5_root / "2016" / "fire_a.hdf5", [0.0, 9.0])
    expected = {2016: ExpectedYear(1, 1, 0, 1)}
    summary = verify_active_fire_dataset(hdf5_root, source_root, expected)
    assert summary[2016].files == 1
    assert summary[2016].positive_target_pixels == 1


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("wrong_positive_count", "positive target pixels"),
        ("missing_lzf", "compression"),
        ("out_of_range", "0-23"),
        ("raw_sample_mismatch", "raw comparison"),
    ],
)
def test_verifier_rejects_independent_invariant_failures(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _write_verification_fixture(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        verify_active_fire_dataset(hdf5_root, source_root, expected)
```

Add a `main([...])` test for the exact `--expect 2016:1:1:0:1` syntax and JSON
stdout. Require rejection of duplicate years and malformed expectations.

- [ ] **Step 2: Run verifier tests to verify RED**

```powershell
python -m pytest tests/test_verify_repair.py -v
```

Expected: collection fails because `wildfire_phase0.verify_repair` does not
exist.

- [ ] **Step 3: Implement the independent verifier**

Define exact dataclasses:

```python
@dataclass(frozen=True)
class ExpectedYear:
    files: int
    target_days: int
    zero_target_days: int
    positive_target_pixels: int


@dataclass(frozen=True)
class VerifiedYear:
    files: int
    target_days: int
    zero_target_days: int
    positive_target_pixels: int
```

Implement:

```python
def verify_active_fire_dataset(
    hdf5_root: Path,
    source_tiff_root: Path,
    expected: Mapping[int, ExpectedYear],
) -> dict[int, VerifiedYear]:
```

For every expected year, sort `*.hdf5`, require exact file count, and open every
file read-only. Require dataset `data`, 4D shape with 23 channels, LZF,
`shuffle=True`, repair attrs, dates matching the event's TIFF stems, and active
values that are finite integer hours in `[0, 23]`. Count target days from index
1, zero-target days, and positive target pixels, then compare the complete
year totals to `ExpectedYear`.

For the lexicographically first and last event in each year, compare the first
and last day against raw TIFF. Read raw HWC or CHW, require exact equality with
staged channels 0--21, and independently normalize raw channel 22 by these
local rules: all positive values `<=23` are hours; all positive values `>23`
must be valid HHMM; mixed values are rejected. Require exact normalized active
equality. Do not import the production repair or inventory modules.

The module `main(argv)` parses repeated `--expect` values, prints sorted JSON
with a final newline, returns `0` on success, and lets validation errors produce
a nonzero process exit.

- [ ] **Step 4: Run verifier and full tests**

```powershell
python -m pytest tests/test_verify_repair.py -v
python -m pytest -q
```

Expected: independent verifier tests and the full suite pass.

- [ ] **Step 5: Add the synthetic repair-to-audit test**

In `tests/test_cli.py`, create a synthetic source for years 2016, 2021, and
2022 where 2016/2022 TIFF labels are hour encoded and active HDF5 labels are
zeroed, while 2021 HDF5 is already correct. Run `repair-active-fire` for 2016
and 2022, assemble a test audit root from staged 2016/2022 plus original 2021,
then assert audit exit `0` and nonzero label totals in train, validation, and
test report rows.

- [ ] **Step 6: Run the end-to-end test to verify coverage**

Run:

```powershell
python -m pytest tests/test_cli.py -k "repair_to_audit" -v
```

Expected: PASS only when Tasks 2 and 3 compose correctly; if it exposes an
interface mismatch, fix the producing task interface rather than weakening the
test.

- [ ] **Step 7: Document the operator workflow**

Add to `docs/experiments/phase0.md`:

```powershell
python -m wildfire_phase0.cli repair-active-fire `
  --source-tiff-root "$wstsPlusTiffRoot" `
  --hdf5-root "$wstsHdf5Root" `
  --staging-root "$repairStagingRoot" `
  --years 2016 2017 2022 2023
```

Document these hard preconditions before activation:

- decision status `ready`;
- 392 staged HDF5 files;
- zero errors and zero `.tmp` files;
- exact four-year label counts from Global Constraints;
- all staged datasets use LZF and shuffle;
- the active four year paths and `hdf5-active-fire-bug-backup` are resolved and
  checked on the same volume.

Document that activation is an explicit same-volume rename, that the backup is
retained, and that rollback reverses those renames. Do not include a broad or
recursive delete command.

Update the website Phase 0 callout to disclose the source encoding mismatch and
state that the published counts are pending corrected audit until Task 5.
Update the English translation map for every new visible text node.

- [ ] **Step 8: Run repository and website checks**

Run:

```powershell
python -m pytest -q
git diff --check
node -e "const fs=require('fs'),vm=require('vm');const s=fs.readFileSync('index.html','utf8');const m=s.match(/<script>([\\s\\S]*?)<\\/script>/);if(!m)throw new Error('script missing');new vm.Script(m[1]);"
```

Expected: all tests pass, diff check is clean, and inline JavaScript parses.

- [ ] **Step 9: Commit the verifier and operator workflow**

```powershell
git add src/wildfire_phase0/verify_repair.py tests/test_verify_repair.py docs/experiments/phase0.md index.html tests/test_cli.py
git commit -m "feat: verify staged active-fire repairs"
```

---

### Task 5: Stage, Activate, Re-Audit, and Publish Corrected Real-Data Evidence

**Files:**
- Derived only before activation: external staging and backup directories under `D:\WildFire Project\data`
- Modify after verified audit: `docs/experiments/phase0.md`
- Modify after verified audit: `index.html`

**Interfaces:**
- Consumes the Task 2 repair CLI and Task 3 target-aware audit CLI.
- Produces a validated active 999-event HDF5 dataset, retained pre-repair backup, corrected local Phase 0 artifacts, and corrected website result.

- [ ] **Step 1: Capture immutable preconditions**

Run read-only checks and record output in the task log:

```powershell
git status --short --branch
Get-PSDrive -Name D
Get-ChildItem -LiteralPath 'D:\WildFire Project\data\hdf5' -Directory
Test-Path -LiteralPath 'D:\WildFire Project\data\hdf5-active-fire-repair-staging'
Test-Path -LiteralPath 'D:\WildFire Project\data\hdf5-active-fire-bug-backup'
```

Require a clean worktree, more than 30 GiB free, and absent staging/backup roots.
If either root exists, stop and inspect it; never merge with or overwrite an
unknown prior run.

- [ ] **Step 2: Run the real staged repair**

Run:

```powershell
python -m wildfire_phase0.cli repair-active-fire `
  --source-tiff-root 'D:\WildFire Project\data\raw\WSTSPlus' `
  --hdf5-root 'D:\WildFire Project\data\hdf5' `
  --staging-root 'D:\WildFire Project\data\hdf5-active-fire-repair-staging' `
  --years 2016 2017 2022 2023
```

Expected exit `0` and decision `ready`. Preserve the complete console output.

- [ ] **Step 3: Independently validate staging before activation**

Run the independent verifier from Task 4:

```powershell
python -m wildfire_phase0.verify_repair `
  --hdf5-root 'D:\WildFire Project\data\hdf5-active-fire-repair-staging' `
  --source-tiff-root 'D:\WildFire Project\data\raw\WSTSPlus' `
  --expect 2016:92:2102:886:303648 `
  --expect 2017:110:2490:1481:177972 `
  --expect 2022:122:3424:2158:105377 `
  --expect 2023:68:2442:1297:167952
```

Expected exit `0`; this opens all 392 files, checks shape/compression/repair
attrs/ranges, validates exact year totals, and compares representative raw
first/last event days independently of production repair code.

Then validate repair evidence and exclusions:

```powershell
$stagingRoot = 'D:\WildFire Project\data\hdf5-active-fire-repair-staging'
$decision = Get-Content -LiteralPath (Join-Path $stagingRoot 'active_fire_repair_decision.json') -Raw | ConvertFrom-Json
$expectedExclusions = @(
  '2022/fire_CA4186812327820220730',
  '2022/fire_ID4570411652620220904',
  '2022/fire_OR4513211711020220825',
  '2022/fire_WA4687912083320220803',
  '2022/fire_WA4796412068520220909'
)
if ($decision.status -ne 'ready') { throw "repair decision is not ready" }
if (@(Get-ChildItem -LiteralPath $stagingRoot -Recurse -File -Filter '*.tmp').Count -ne 0) { throw "staging contains temp files" }
if (Compare-Object $expectedExclusions @($decision.excluded_empty_source_directories)) { throw "empty-source exclusions differ" }
```

If any check fails, leave active data untouched and fix the implementation in a
new commit before rerunning staging.

- [ ] **Step 4: Activate through exact same-volume directory renames**

Resolve and verify these exact paths before mutation:

```text
D:\WildFire Project\data\hdf5\2016
D:\WildFire Project\data\hdf5\2017
D:\WildFire Project\data\hdf5\2022
D:\WildFire Project\data\hdf5\2023
D:\WildFire Project\data\hdf5-active-fire-repair-staging\2016
D:\WildFire Project\data\hdf5-active-fire-repair-staging\2017
D:\WildFire Project\data\hdf5-active-fire-repair-staging\2022
D:\WildFire Project\data\hdf5-active-fire-repair-staging\2023
D:\WildFire Project\data\hdf5-active-fire-bug-backup
```

Create the exact backup root. For each year, move the active year directory to
the backup root, then move the staged year directory into the active HDF5 root.
Use native PowerShell `Move-Item -LiteralPath` throughout. Do not delete the
staging evidence files or backup directories.

- [ ] **Step 5: Validate all active data and rerun Phase 0**

Run the verifier against the activated four repaired years using the exact
Task 5 Step 3 command with `--hdf5-root` changed to
`D:\WildFire Project\data\hdf5`. Independently validate structural and stored
active-fire invariants for all 999 files:

```powershell
@'
from pathlib import Path
import h5py
import numpy as np

root = Path(r"D:\WildFire Project\data\hdf5")
expected = {2016: 92, 2017: 110, 2018: 176, 2019: 74,
            2020: 201, 2021: 156, 2022: 122, 2023: 68}
total = 0
for year, expected_files in expected.items():
    files = sorted((root / str(year)).glob("*.hdf5"))
    assert len(files) == expected_files, (year, len(files))
    total += len(files)
    for path in files:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            assert data.ndim == 4 and data.shape[1] == 23, path
            assert data.compression == "lzf" and data.shuffle, path
            for day in range(data.shape[0]):
                active = np.asarray(data[day, 22])
                finite = active[np.isfinite(active)]
                assert np.all((finite >= 0) & (finite <= 23)), path
                assert np.all(finite == np.floor(finite)), path
assert total == 999
print({"files": total, "status": "valid"})
'@ | python -
```

Then run:

```powershell
python -m wildfire_phase0.cli audit `
  --data-root 'D:\WildFire Project\data\hdf5' `
  --output-root 'artifacts\phase0'
```

Expected exit `0`, status `continue_controlled`, 999 events, train/validation/test
counts `653/156/190`, and nonzero target pixels for all eight years and all
three splits. The exact NaN totals may be unchanged; report the freshly
generated values rather than copying earlier numbers.

If validation or audit fails, reverse the exact four directory moves to restore
the backup and preserve the failed repaired directories for diagnosis.

- [ ] **Step 6: Publish corrected real-data evidence**

Update `docs/experiments/phase0.md` and the visible bilingual `index.html`
callout with:

- the discovered source encoding mismatch and repair version;
- exact target counts by year and split from the fresh report;
- the strengthened target-integrity gate;
- retained contract blockers and `continue_controlled` claim boundary;
- confirmation that the research route and frozen split remain unchanged.

Do not commit local absolute paths, raw data, HDF5 files, or ignored generated
artifacts.

- [ ] **Step 7: Run fresh final verification and commit**

Run:

```powershell
python -m pytest -q
git diff --check
node -e "const fs=require('fs'),vm=require('vm');const s=fs.readFileSync('index.html','utf8');const m=s.match(/<script>([\\s\\S]*?)<\\/script>/);if(!m)throw new Error('script missing');new vm.Script(m[1]);"
git status --short
```

Expected: complete green suite, clean static checks, and only the two intended
documentation files modified.

Commit:

```powershell
git add docs/experiments/phase0.md index.html
git commit -m "docs: publish corrected phase0 target audit"
```

Do not delete the backup or push the branch without explicit user direction.

---

## Follow-On Boundary

The repaired dataset and strengthened gate are the prerequisite for the
separate `2026-08-08-rule-baseline-evaluation.md` plan. Do not start that plan
until Task 5 exits with verified nonzero target labels in every year and split.

# Phase 0 Data Audit and Rule Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Phase 0 pipeline that inventories WSTS+ HDF5 events, audits the recoverability of time/quality/validity fields, freezes the forward split, and produces no-fire and persistence baselines without changing the approved research route.

**Architecture:** Add a small installable Python package under `src/wildfire_phase0`. The package treats the public WSTS+ files as immutable inputs, emits CSV/JSON/Markdown artifacts under `artifacts/phase0`, and keeps data-contract decisions separate from model code. All logic is first exercised on synthetic HDF5 fixtures; the same CLI then runs against the real dataset through an explicit `--data-root` argument.

**Tech Stack:** Python 3.13, NumPy, pandas, h5py, scikit-learn, pytest, standard-library `argparse` and `json`. PyTorch 2.11.0+cu126 is reserved for the later learned-baseline plan and is not required by Phase 0.

## Global Constraints

- Keep WSTS+ as the primary dataset and retain the ten-month research route.
- Interpret the public task as next-UTC-calendar-day VIIRS active-fire proxy forecasting, not a uniform physical `t+24h` operational forecast.
- Never modify or redistribute source data; write derived artifacts only under `artifacts/phase0`.
- Use official folds only for literature comparison; freeze `2016–2020 train / 2021 validation / 2022–2023 untouched test` for confirmatory evaluation.
- Treat natural missingness as a main estimand only when coverage/QA and target validity are recoverable.
- Do not load the 2022–2023 target arrays during model or threshold selection.
- Use Windows-safe paths and accept the dataset location only through CLI arguments.

---

### Task 1: Package Skeleton and Manifest Schema

**Files:**
- Create: `pyproject.toml`
- Create: `src/wildfire_phase0/__init__.py`
- Create: `src/wildfire_phase0/schema.py`
- Create: `tests/test_schema.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `EventInventory` dataclass and `validate_inventory_row(row: Mapping[str, object]) -> EventInventory`.
- Produces: editable package installation through `python -m pip install -e ".[dev]"`.

- [ ] **Step 1: Write the failing schema test**

```python
from pathlib import Path

import pytest

from wildfire_phase0.schema import EventInventory, validate_inventory_row


def test_validate_inventory_row_rejects_non_monotonic_dates() -> None:
    row = {
        "year": 2021,
        "fire_name": "demo_fire",
        "path": "2021/demo_fire.hdf5",
        "n_days": 3,
        "n_channels": 23,
        "height": 8,
        "width": 8,
        "dates": ["2021-08-03", "2021-08-01", "2021-08-02"],
        "nan_fraction": 0.1,
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_inventory_row(row)


def test_event_inventory_keeps_relative_path() -> None:
    item = EventInventory(2021, "demo_fire", Path("2021/demo_fire.hdf5"), 3, 23, 8, 8,
                          ("2021-08-01", "2021-08-02", "2021-08-03"), 0.1)
    assert item.path == Path("2021/demo_fire.hdf5")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_schema.py -v`

Expected: FAIL because `wildfire_phase0.schema` does not exist.

- [ ] **Step 3: Add the minimal package and dataclass implementation**

`EventInventory` must contain exactly: `year`, `fire_name`, `path`, `n_days`, `n_channels`, `height`, `width`, `dates`, and `nan_fraction`. `validate_inventory_row` must parse ISO dates, require strictly increasing dates, require `n_channels == 23`, require positive spatial dimensions, and constrain `nan_fraction` to `[0, 1]`.

`pyproject.toml` must define:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "wildfire-phase0"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
dependencies = [
  "h5py>=3.12,<4",
  "numpy>=2.1,<3",
  "pandas>=2.2,<3",
  "scikit-learn>=1.6,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8.3,<9"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore` must preserve the existing `.worktrees/` entry and add `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `data/`, `artifacts/phase0/*.csv`, `artifacts/phase0/*.json`, and `artifacts/phase0/*.md`, while retaining `artifacts/phase0/.gitkeep`.

- [ ] **Step 4: Install and run the schema tests**

Run: `python -m pip install -e ".[dev]"`

Run: `python -m pytest tests/test_schema.py -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit the package skeleton**

```powershell
git add pyproject.toml .gitignore src/wildfire_phase0 tests/test_schema.py
git commit -m "build: add phase0 experiment package"
```

---

### Task 2: WSTS+ HDF5 Inventory Adapter

**Files:**
- Create: `src/wildfire_phase0/inventory.py`
- Create: `tests/test_inventory.py`

**Interfaces:**
- Consumes: WSTS/WSTS+ layout `<data_root>/<year>/<fire_name>.hdf5` with dataset key `data` and attributes `year`, `fire_name`, `img_dates`, and `lnglat`.
- Produces: `inspect_hdf5(path: Path, data_root: Path) -> EventInventory`.
- Produces: `inventory_dataset(data_root: Path) -> list[EventInventory]` sorted by `(year, fire_name)`.

- [ ] **Step 1: Write the synthetic HDF5 fixture test**

```python
from pathlib import Path

import h5py
import numpy as np

from wildfire_phase0.inventory import inspect_hdf5, inventory_dataset


def _write_event(path: Path) -> None:
    path.parent.mkdir(parents=True)
    values = np.zeros((3, 23, 8, 8), dtype=np.float32)
    values[0, 2, 0, 0] = np.nan
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("data", data=values)
        data.attrs["year"] = 2021
        data.attrs["fire_name"] = "demo_fire"
        data.attrs["img_dates"] = ["2021-08-01", "2021-08-02", "2021-08-03"]
        data.attrs["lnglat"] = [-120.5, 54.1]


def test_inspect_hdf5_reads_official_layout(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "demo_fire.hdf5"
    _write_event(path)
    item = inspect_hdf5(path, tmp_path)
    assert (item.year, item.fire_name, item.n_days, item.n_channels) == (2021, "demo_fire", 3, 23)
    assert item.nan_fraction == 1 / (3 * 23 * 8 * 8)


def test_inventory_dataset_is_deterministic(tmp_path: Path) -> None:
    _write_event(tmp_path / "2021" / "demo_fire.hdf5")
    assert [x.fire_name for x in inventory_dataset(tmp_path)] == ["demo_fire"]
```

- [ ] **Step 2: Run the inventory tests to verify they fail**

Run: `python -m pytest tests/test_inventory.py -v`

Expected: FAIL because `wildfire_phase0.inventory` does not exist.

- [ ] **Step 3: Implement read-only inspection**

Open files with `h5py.File(path, "r")`; reject a missing `data` key, missing required attributes, shapes other than `(days, 23, height, width)`, folder-year/attribute-year mismatch, and filename/fire-name mismatch. Decode byte-string attributes to UTF-8. Compute `nan_fraction` in chunks of one day so a full event is never duplicated in memory.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_inventory.py tests/test_schema.py -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit the inventory adapter**

```powershell
git add src/wildfire_phase0/inventory.py tests/test_inventory.py
git commit -m "feat: inventory WSTS event files"
```

---

### Task 3: Data-Contract Registry and Blocking Audit

**Files:**
- Create: `configs/wstsplus_field_contract.csv`
- Create: `src/wildfire_phase0/contract.py`
- Create: `tests/test_contract.py`

**Interfaces:**
- Produces: `load_contract(path: Path) -> pandas.DataFrame`.
- Produces: `audit_contract(frame: pandas.DataFrame) -> ContractDecision` with `status` equal to `continue_natural`, `continue_controlled`, or `blocked`.
- `ContractDecision` exposes `missing_required`, `operational_blockers`, and `notes` as tuples of strings.

- [ ] **Step 1: Write decision tests**

```python
import pandas as pd

from wildfire_phase0.contract import audit_contract


def test_public_wsts_contract_downgrades_to_controlled_missingness() -> None:
    frame = pd.DataFrame([
        {"modality": "active_fire", "observation_time": "positive_pixel_hour_only",
         "availability_time": "unavailable", "qa": "filtered_not_retained",
         "coverage": "unavailable", "target_validity": "unavailable"},
        {"modality": "viirs_reflectance", "observation_time": "daily_aggregate_only",
         "availability_time": "unavailable", "qa": "unavailable",
         "coverage": "nan_unknown_cause", "target_validity": "not_applicable"},
    ])
    decision = audit_contract(frame)
    assert decision.status == "continue_controlled"
    assert "target_validity" in decision.missing_required


def test_traceable_contract_allows_natural_missingness() -> None:
    frame = pd.DataFrame([
        {"modality": "active_fire", "observation_time": "traceable",
         "availability_time": "traceable", "qa": "traceable",
         "coverage": "traceable", "target_validity": "traceable"},
    ])
    assert audit_contract(frame).status == "continue_natural"
```

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `python -m pytest tests/test_contract.py -v`

Expected: FAIL because `wildfire_phase0.contract` does not exist.

- [ ] **Step 3: Add the registry and audit rules**

The CSV must contain rows for `active_fire`, `viirs_reflectance`, `ndvi_evi`, `gridmet`, `gfs_forecast`, `terrain`, and `land_cover`, plus `source_url` and `audit_note`. Encode the verified public state: only positive active-fire pixels retain within-day detection hour; other modalities lack per-sample acquisition/availability metadata and original QA; target validity is unavailable; VNP13A1 is a composite product; GRIDMET historical versions can be updated; land cover is annual.

`audit_contract` must return `continue_controlled` when source files are usable but any of `coverage`, `target_validity`, or modality-level `observation_time` is unavailable. It must return `blocked` only when event identity, dates, labels, or data arrays cannot be interpreted.

- [ ] **Step 4: Run the contract and full test suites**

Run: `python -m pytest tests/test_contract.py -v`

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the contract audit**

```powershell
git add configs/wstsplus_field_contract.csv src/wildfire_phase0/contract.py tests/test_contract.py
git commit -m "feat: audit WSTS data contract"
```

---

### Task 4: Frozen Forward Split Manifest

**Files:**
- Create: `src/wildfire_phase0/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Consumes: `Sequence[EventInventory]`.
- Produces: `build_forward_split(events) -> pandas.DataFrame` with columns `event_id`, `year`, `fire_name`, `path`, and `split`.
- Split mapping is exact: 2016–2020 `train`, 2021 `validation`, 2022–2023 `test`; any other year raises `ValueError`.

- [ ] **Step 1: Write split tests**

```python
from pathlib import Path

from wildfire_phase0.schema import EventInventory
from wildfire_phase0.splits import build_forward_split


def _event(year: int, name: str) -> EventInventory:
    return EventInventory(year, name, Path(f"{year}/{name}.hdf5"), 2, 23, 8, 8,
                          (f"{year}-08-01", f"{year}-08-02"), 0.0)


def test_forward_split_is_event_disjoint_and_temporal() -> None:
    frame = build_forward_split([_event(2020, "train_fire"), _event(2021, "val_fire"),
                                 _event(2022, "test_fire")])
    assert frame.set_index("fire_name")["split"].to_dict() == {
        "train_fire": "train", "val_fire": "validation", "test_fire": "test"
    }
    assert frame["event_id"].is_unique
```

- [ ] **Step 2: Run the split test to verify it fails**

Run: `python -m pytest tests/test_splits.py -v`

Expected: FAIL because `wildfire_phase0.splits` does not exist.

- [ ] **Step 3: Implement and validate the frozen mapping**

Build `event_id` as `"{year}:{fire_name}"`; sort by `(year, fire_name)`; assert uniqueness before returning. Do not implement random splitting.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the split manifest logic**

```powershell
git add src/wildfire_phase0/splits.py tests/test_splits.py
git commit -m "feat: freeze forward event split"
```

---

### Task 5: Rule Baselines and Event-Level Metrics

**Files:**
- Create: `src/wildfire_phase0/baselines.py`
- Create: `src/wildfire_phase0/metrics.py`
- Create: `tests/test_baselines.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `no_fire(shape: tuple[int, int]) -> numpy.ndarray`.
- Produces: `persistence(history: numpy.ndarray, mode: Literal["latest", "all"]) -> numpy.ndarray` for binary history shaped `(time, height, width)`.
- Produces: `event_macro_ap(records: pandas.DataFrame) -> float`, where records contain `event_id`, `y_true`, and `y_score` arrays.
- Produces: `zero_target_false_alarm_rate(y_true, y_score, threshold) -> float`.

- [ ] **Step 1: Write rule-baseline tests**

```python
import numpy as np

from wildfire_phase0.baselines import no_fire, persistence


def test_persistence_modes() -> None:
    history = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.uint8)
    assert no_fire((2, 2)).sum() == 0
    assert persistence(history, "latest").tolist() == [[0, 1], [0, 0]]
    assert persistence(history, "all").tolist() == [[1, 1], [0, 0]]
```

- [ ] **Step 2: Write metric tests for ordinary and zero-positive targets**

```python
import numpy as np

from wildfire_phase0.metrics import average_precision_safe, zero_target_false_alarm_rate


def test_average_precision_safe_marks_zero_positive_target() -> None:
    value, has_positive = average_precision_safe(np.zeros(4), np.array([0.0, 0.2, 0.7, 0.1]))
    assert np.isnan(value)
    assert has_positive is False
    assert zero_target_false_alarm_rate(np.zeros(4), np.array([0.0, 0.2, 0.7, 0.1]), 0.5) == 0.25
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_baselines.py tests/test_metrics.py -v`

Expected: FAIL because the modules do not exist.

- [ ] **Step 4: Implement deterministic baselines and safe metrics**

Use `sklearn.metrics.average_precision_score` only when at least one positive exists. Never silently remove zero-positive event-days: return `NaN` plus `has_positive=False`, and aggregate their false-alarm rate separately. `event_macro_ap` must average per-event AP values with equal event weight after concatenating days within each event.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/wildfire_phase0/baselines.py src/wildfire_phase0/metrics.py tests/test_baselines.py tests/test_metrics.py
git commit -m "feat: add wildfire rule baselines"
```

---

### Task 6: Phase 0 CLI, Artifacts, and Real-Data Gate

**Files:**
- Create: `src/wildfire_phase0/cli.py`
- Create: `src/wildfire_phase0/report.py`
- Create: `tests/test_cli.py`
- Create: `artifacts/phase0/.gitkeep`
- Create: `docs/experiments/phase0.md`

**Interfaces:**
- Produces CLI: `python -m wildfire_phase0.cli audit --data-root PATH --output-root artifacts/phase0`.
- Produces `inventory.csv`, `split_manifest.csv`, `contract_decision.json`, and `phase0_report.md`.
- The command exits `0` for `continue_natural` or `continue_controlled`, and `2` for `blocked`.

- [ ] **Step 1: Write an end-to-end CLI test using one synthetic event**

The test must invoke `cli.main(["audit", "--data-root", str(data_root), "--output-root", str(output_root)])`, assert exit code `0`, assert all four artifact filenames exist, and assert the report contains `continue_controlled`, `next-calendar-day active-fire proxy`, and the frozen split definition.

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL because `wildfire_phase0.cli` and `wildfire_phase0.report` do not exist.

- [ ] **Step 3: Implement atomic artifact generation**

Write each artifact to a sibling `*.tmp` file and replace the final file only after serialization succeeds. The Markdown report must include counts by year/split, invalid-file errors, NaN summaries, missing contract fields, operational blockers, the exact continue/pivot decision, and the commands used.

- [ ] **Step 4: Document the operator workflow**

`docs/experiments/phase0.md` must contain these commands:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
$wstsDataRoot = (Resolve-Path -LiteralPath $env:WSTSPLUS_DATA_ROOT).Path
python -m wildfire_phase0.cli audit --data-root "$wstsDataRoot" --output-root artifacts\phase0
```

It must state that `continue_controlled` preserves the approved project route but removes natural-missingness and operational claims.

- [ ] **Step 5: Run the full synthetic gate**

Run: `python -m pytest -q`

Expected: all tests pass.

Run against the real dataset after its location is known:

```powershell
$wstsDataRoot = (Resolve-Path -LiteralPath $env:WSTSPLUS_DATA_ROOT).Path
python -m wildfire_phase0.cli audit --data-root "$wstsDataRoot" --output-root artifacts\phase0
```

Expected: exit `0` with `continue_natural` or `continue_controlled`, or exit `2` with a report naming the exact blocker.

- [ ] **Step 6: Commit the Phase 0 gate**

```powershell
git add src/wildfire_phase0/cli.py src/wildfire_phase0/report.py tests/test_cli.py artifacts/phase0/.gitkeep docs/experiments/phase0.md
git commit -m "feat: add phase0 data gate"
```

---

## Plan Self-Review

- Every approved Phase 0 requirement maps to a task: field audit (Task 3), event/file inventory (Task 2), forward holdout (Task 4), rule baselines and zero-target handling (Task 5), and a reproducible decision artifact (Task 6).
- The plan deliberately excludes learned neural baselines, synthetic corruption generation, gating models, bootstrap inference, and TS-SatFire; each depends on the Phase 0 decision and belongs in the next implementation plan.
- Public WSTS+ HDF5 assumptions match the official converter: one file per event, dataset key `data`, and attributes `year`, `fire_name`, `img_dates`, and `lnglat`.
- No task writes to source data or reads 2022–2023 targets for tuning.
- Function names and data types are consistent across tasks; no placeholder implementation step remains.

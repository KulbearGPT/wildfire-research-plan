# Artifact Publication Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two remaining artifact-publication safety defects without changing the Phase 0 research protocol, data, or experiment route.

**Architecture:** Keep both existing transaction designs. The audit publisher must distinguish a successfully copied rollback image from a merely reserved backup path. Repair-evidence restart handling must recognize a fully written invocation-owned journal temp before scanning final journals, promote that temp atomically, and let the existing journal-based recovery finish the interrupted generation.

**Tech Stack:** Python 3, pathlib, pytest, pandas.

## Global Constraints

- Do not mutate, copy, restage, rename, or delete any real HDF5, TIFF, backup, or repair-staging dataset directories.
- Do not change the frozen 2016–2023 protocol, split boundaries, field contract, repair semantics, or research route.
- Preserve unknown legacy sidecars that do not use the reserved invocation-owned transaction naming scheme.
- Use strict TDD: record the focused RED result before production edits and the focused GREEN result afterward.
- Limit production changes to `src/wildfire_phase0/cli.py` and `src/wildfire_phase0/repair.py`; limit regression tests to their existing test modules.

---

### Task 1: Make audit backup creation rollback-safe

**Files:**
- Modify: `src/wildfire_phase0/cli.py:82-108`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `_owned_sidecar(path: Path, suffix: str) -> Path` and `shutil.copy2` imported as `copy2`.
- Produces: `_publish_staged_artifacts(paths, temp_paths)` with the same signature and all-or-nothing publication contract.

- [ ] **Step 1: Write the failing regression test**

Add a test beside the existing audit publication-failure tests. Seed a complete old generation, inject an `OSError` from `wildfire_phase0.cli.copy2` only while copying the first final into its newly reserved `.bak`, invoke the real audit command, and assert the exception is raised, every old final retains its exact bytes, and no invocation-owned `.tmp` or `.bak` remains.

- [ ] **Step 2: Run the focused test and record RED**

Run: `python -m pytest -q tests/test_cli.py::test_audit_backup_copy_failure_preserves_existing_finals`

Expected: FAIL because the first pre-existing final is overwritten from an empty or partial reserved backup during rollback.

- [ ] **Step 3: Implement the minimal correction**

Track all reserved backup paths separately for cleanup, but add a `path -> backup` entry to the restorable-backup mapping only after `copy2(path, backup)` returns successfully. On a backup-copy failure, rollback must not copy the incomplete backup over its original. Cleanup must remove every backup path reserved by this invocation while preserving unrelated sidecars.

- [ ] **Step 4: Verify GREEN and regression coverage**

Run the focused test, then `python -m pytest -q tests/test_cli.py`.

- [ ] **Step 5: Commit**

Commit message: `fix: preserve audit artifacts on backup failure`

### Task 2: Recover a pre-journal repair interruption

**Files:**
- Modify: `src/wildfire_phase0/repair.py:557-598,730-741`
- Test: `tests/test_repair.py`

**Interfaces:**
- Consumes: the reserved `.active_fire_repair_evidence.<32 lowercase hex>.txn.tmp` and `.txn.json` namespace plus `_load_transaction` and `_recover_transaction`.
- Produces: `_recover_interrupted_evidence(staging_root: Path)` that handles a valid journal temp left after process interruption before journal promotion.

- [ ] **Step 1: Write the failing restart regression test**

Add a test beside the existing repair interruption tests. Inject a custom `BaseException` when `Path.replace` promotes a fully written `.txn.tmp` to `.txn.json`, assert the first run leaves the reserved transaction sidecars, restore `Path.replace`, then invoke the real repair flow again. Assert restart completes with `status == "ready"`, `verify_repair_evidence` returns that decision, all invocation-owned transaction temps/backups/journals are gone, and any separately seeded unknown legacy sidecar retains exact bytes.

- [ ] **Step 2: Run the focused test and record RED**

Run: `python -m pytest -q tests/test_repair.py::test_repair_evidence_recovers_interruption_before_journal_promotion`

Expected: FAIL because restart ignores `.txn.tmp`, starts a new transaction, and leaves the interrupted invocation's sidecars behind.

- [ ] **Step 3: Implement the minimal correction**

At restart, scan for both completed journals and reserved journal temps. If there is exactly one valid journal temp and no completed journal, parse it with the same strict payload and filename validation used for a completed journal, atomically promote it to its matching `.txn.json`, and invoke existing recovery. Fail closed on ambiguous multiple/mixed reserved transaction journals. Do not glob-delete sidecars and do not touch unknown legacy names.

- [ ] **Step 4: Verify GREEN and regression coverage**

Run the focused test, then `python -m pytest -q tests/test_repair.py`.

- [ ] **Step 5: Commit**

Commit message: `fix: recover repair journal promotion interruptions`


# WSTS+ Controlled-Missingness Implementation Plan

**Goal:** Implement the complete deterministic M00--M07 diagnostic pipeline
without accessing 2022--2023 before the clean-replication gate.

**Design:** `docs/superpowers/specs/2026-08-23-wsts-post-control-design.md`

## Task 1: Raw-space corruption engine

**Files:**

- Add: `reproductions/wsts_fast_track/missingness.py`
- Add: `tests/test_wsts_fast_track_missingness.py`
- Modify: `reproductions/wsts_fast_track/matrix.py`

1. Test M00 identity and non-mutation.
2. Test raw-channel effects for M01--M05 and strict M02 stale-history input.
3. Test exact-area, stable-hash M06/M07 blocks and key sensitivity.
4. Reject tensors outside `(T, 23, H, W)` and non-finite stale fire history.
5. Mark scenarios implemented only after every corruption test passes.

## Task 2: Deterministic evaluation dataset

**Files:**

- Add: `reproductions/wsts_fast_track/evaluation.py`
- Add: `tests/test_wsts_fast_track_evaluation.py`

1. Construct the pinned upstream dataset with `is_train=false`, train-only
   statistics, HDF5 loading, and effective history adjustment six.
2. Derive event-relative path and target date from HDF5 metadata without
   accepting labels or predictions as corruption-key inputs.
3. Load one extra preceding raw day for M02 while returning the same target
   indices for every scenario.
4. Apply corruption before upstream preprocessing and preserve the exact
   model-specific feature-selection behavior.
5. Expose a 2021-only engineering constructor; reject every other year until a
   separately reviewed held-out authorization exists.

## Task 3: Evaluation manifest and immutable result contract

**Files:**

- Add: `reproductions/wsts_fast_track/missingness_manifest.py`
- Add: `tests/test_wsts_fast_track_missingness_manifest.py`

1. Require the six accepted clean 10K completion records before a formal
   matrix can be rendered.
2. Render the Cartesian product of six checkpoints and eight scenarios with
   deterministic ordering and no Slurm call.
3. Freeze split, schema, checkpoint identity, scenario identity, metric names,
   and non-overwriting output paths.
4. Permit a 2021 engineering manifest before replication only when explicitly
   marked non-scientific and keep 2022--2023 unavailable.

## Task 4: Model evaluation and aggregation

**Files:**

- Add: `reproductions/wsts_fast_track/evaluate_missingness.py`
- Add: `reproductions/wsts_fast_track/aggregate_missingness.py`
- Add tests for both modules.

1. Strict-load one declared checkpoint into its C00/C02 architecture.
2. Evaluate AP, F1, IoU, precision, recall, and loss without training or
   checkpoint selection.
3. Seal one immutable result per checkpoint/scenario/year.
4. Aggregate same-year deltas from M00 and seed mean/dispersion; never compare
   raw AP across years as an improvement claim.

## Task 5: Verification and execution boundary

1. Run all fast-track tests, shell syntax checks, and `git diff --check`.
2. Run only synthetic and 2021 engineering validation while clean replication
   is incomplete.
3. After all six clean records pass, render the formal matrix once and execute
   2022 and 2023 separately without tuning or retry-based selection.

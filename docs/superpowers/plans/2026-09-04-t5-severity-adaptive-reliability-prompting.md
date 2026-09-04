# T=5 Severity-Adaptive Reliability Prompting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a matched quantitative T=5 test of D12's severity-adaptive prompt-depth mechanism on corrected Res18-UTAE.

**Architecture:** Add a corrected T=5 FireDrop+BlockDrop base (`B5`), then branch a standard and a SARP single-corrupt-view continuation from its exact checkpoint. T5-SARP injects the severe input token per time step and mild multi-scale prompts before the unchanged LTAE/skip aggregation.

**Tech Stack:** Python 3.10, PyTorch, segmentation-models-pytorch, the pinned WSTS runtime, pytest, Bash, Slurm on Alliance Nibi.

**Spec:** `docs/superpowers/specs/2026-09-04-t5-severity-adaptive-reliability-prompting-design.md`

## Global Constraints

- C02 Res18-UTAE, `T=5`, 33 Multi features, corrected 2016--2020 training indices.
- Seed 0, batch 64, AdamW `1e-3`, 3,000 steps per run, unchanged focal loss.
- Severity threshold exactly `0.375`; no sweep and no held-out selection.
- Only focused synthetic tests run on the login node; real data/model compute runs through Slurm.
- Every T=5 training job requests 64GB host memory.

---

### Task 1: Register the corrected T=5 robust base

**Files:**
- Modify: `reproductions/wsts_fast_track/corrected_baselines.py`
- Modify: `reproductions/wsts_fast_track/run_corrected_baseline_on_nibi.sh`
- Modify: `tests/test_wsts_fast_track_corrected_baselines.py`
- Modify: `tests/test_wsts_fast_track_corrected_runner.py`

**Interfaces:**
- Produces: `corrected_baseline_spec("B5")` with experiment `C02` and policy `fire-block`.
- Produces: the existing corrected baseline runner accepting `B5` without changing B0--B4.

- [ ] **Step 1: Write the failing B5 registration and runner assertions**

```python
assert corrected_baselines.corrected_baseline_spec("B5").experiment_id == "C02"
assert corrected_baselines.corrected_baseline_spec("B5").training_policy == "fire-block"
assert "B0|B1|B2|B3|B4|B5" in RUNNER.read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_wsts_fast_track_corrected_baselines.py tests/test_wsts_fast_track_corrected_runner.py`

Expected: failure because `B5` is unknown and the runner rejects it.

- [ ] **Step 3: Implement the minimal B5 registration**

Add `"B5": CorrectedBaselineSpec("B5", "C02", "fire-block")` and extend only the runner's accepted-ID case and error message.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_wsts_fast_track_corrected_baselines.py tests/test_wsts_fast_track_corrected_runner.py`

Expected: all selected tests pass.

Commit: `git commit -m "feat: register corrected T5 robust baseline"`

### Task 2: Implement processed T=5 corruption and temporal SARP

**Files:**
- Create: `reproductions/wsts_fast_track/temporal_reliability_prompting.py`
- Create: `tests/test_wsts_fast_track_t5_sarp.py`

**Interfaces:**
- Produces: `apply_processed_temporal_reliability_corruption(x, fire_drop, block_fraction, key_digest, active_fire_missing_value) -> Tensor`.
- Produces: `ProcessedTemporalReliabilityDataset(base_dataset, active_fire_missing_value)`.
- Produces: `TemporalSeverityAdaptiveReliabilityPrompting(base_model)` accepting `[B,5,34,H,W]`.

- [ ] **Step 1: Write failing feature-map and routing tests**

The corruption test constructs `[5,33,8,8]`, verifies the appended invalidity channel, verifies C02 active-fire positions are derived from `MULTI_FEATURES.index(38/39)`, and verifies BlockDrop does not modify unregistered static channels. The routing test uses a tiny temporal base exposing `encoder`, `ltae`, `temporal_aggregator`, `decoder`, and `segmentation_head`; 25% invalidity must give gradients only to deep tokens and 50% invalidity only to the input token.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_wsts_fast_track_t5_sarp.py`

Expected: import failure because `temporal_reliability_prompting.py` does not exist.

- [ ] **Step 3: Implement only the required data and model paths**

Derive C02 channel positions from `MULTI_FEATURES`, reuse `structured_block_mask`, reuse `InputReliabilityTokenConv2d`, and reuse `apply_reliability_prompts`. Encode all five time steps, prompt their post-input scales, then call the unchanged base `ltae`, `temporal_aggregator`, decoder, and segmentation head.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_wsts_fast_track_t5_sarp.py`

Expected: all T5 data/routing tests pass.

Commit: `git commit -m "feat: add temporal severity-adaptive prompts"`

### Task 3: Add matched training, evaluation, and Nibi runners

**Files:**
- Create: `reproductions/wsts_fast_track/train_temporal_reliability_prompting.py`
- Create: `reproductions/wsts_fast_track/evaluate_temporal_reliability_prompting.py`
- Create: `reproductions/wsts_fast_track/run_t5_reliability_on_nibi.sh`
- Create: `reproductions/wsts_fast_track/run_t5_sarp_heldout_on_nibi.sh`
- Modify: `tests/test_wsts_fast_track_t5_sarp.py`

**Interfaces:**
- Consumes: a completed corrected `B5` record.
- Produces: `D13-STD-T5.pt` or `D13-SARP-T5.pt` with exact metadata and state dict.
- Produces: M00/M01/M06/M07 summary JSON for 2021 or an explicitly authorized held-out year.

- [ ] **Step 1: Write failing checkpoint-contract and shell-contract tests**

```python
assert validate_t5_reliability_checkpoint(std_payload) == "standard"
assert validate_t5_reliability_checkpoint(sarp_payload) == "sarp"
```

Also run `bash -n` on both runners and verify the training runner rejects an unknown variant before touching Slurm state.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_wsts_fast_track_t5_sarp.py`

Expected: failure because the trainer, evaluator, validator, and runners are missing.

- [ ] **Step 3: Implement the minimal two-arm flow**

The trainer builds the C02 dataset, loads the exact B5 checkpoint, strips the invalidity channel for `standard`, retains it for `sarp`, and saves candidate IDs `D13-STD-T5` and `D13-SARP-T5`. The evaluator reconstructs the matching model, uses `build_controlled_dataset(... experiment_id="C02", routing_mask_channel=True)`, and writes the existing summary schema. The held-out runner archives one clean commit and serially evaluates both checkpoints for 2022 and 2023.

- [ ] **Step 4: Verify GREEN, compilation, shell syntax, and commit**

Run:

```bash
python -m pytest -q tests/test_wsts_fast_track_t5_sarp.py \
  tests/test_wsts_fast_track_corrected_baselines.py \
  tests/test_wsts_fast_track_corrected_runner.py
python -m py_compile \
  reproductions/wsts_fast_track/temporal_reliability_prompting.py \
  reproductions/wsts_fast_track/train_temporal_reliability_prompting.py \
  reproductions/wsts_fast_track/evaluate_temporal_reliability_prompting.py
bash -n reproductions/wsts_fast_track/run_t5_reliability_on_nibi.sh
bash -n reproductions/wsts_fast_track/run_t5_sarp_heldout_on_nibi.sh
```

Expected: all focused checks pass without warnings or syntax errors.

Commit: `git commit -m "feat: run matched T5 SARP experiment"`

### Task 4: Execute the frozen experiment and record evidence

**Files:**
- Modify: `docs/experiments/quantitative_reliability_ledger.md`
- Modify: `README.md` only if T5 produces a final retained or rejected result.

**Interfaces:**
- Consumes: B5, D13-STD-T5, and D13-SARP-T5 Slurm artifacts.
- Produces: a committed 2021 gate decision and, if authorized, a committed three-year decision.

- [ ] **Step 1: Select resources and submit B5**

Inspect `squeue` and compare `sbatch --test-only` predictions for available H100 MIG/full classes. Submit B5 with 8 CPU, 64GB host memory, and a time limit supported by the prior 44-minute B1 run. Do not execute training on the login node.

- [ ] **Step 2: Monitor B5 at approximately 30-minute intervals**

Require Slurm `COMPLETED`, exit `0:0`, `completed.json`, and `results-2021/summary.json` before submitting dependents.

- [ ] **Step 3: Submit the matched STD and SARP continuations**

Choose the fastest predicted resource class, keep resources and time limits identical, and submit both from the same B5 record. Monitor at approximately 30-minute intervals and repair only concrete failures.

- [ ] **Step 4: Apply the frozen 2021 gate**

Read the two summaries and compute M06/M07 mean delta plus M00/M01 guardrails. If the gate fails, record the rejection and stop without reading new held-out results. If it passes, commit authorization before held-out submission.

- [ ] **Step 5: Run the fixed held-out comparison if authorized**

Submit one Slurm allocation evaluating STD and SARP on both 2022 and 2023. Compute per-year and three-year module deltas without changing the method.

- [ ] **Step 6: Verify evidence, update documentation, and commit**

Run a read-only aggregation directly from summary JSON, `git diff --check`, and a focused search for every recorded job ID and metric. Commit with `docs: record T5 SARP transfer result` and leave the worktree clean.

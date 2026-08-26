# P09 Year-Corruption GroupDRO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the P02 expert with 15-group year-corruption GroupDRO and evaluate its routed P09 forecast on the fixed 2021 selection scenarios.

**Architecture:** A P09-only dataset patch appends a year-by-block group ID and a weighted sampler balances years. A pure GroupDRO objective updates 15 exponentiated weights and returns a differentiable weighted loss. The evaluator reuses P00, P02, controlled missingness, and spatial routing so only the expert checkpoint changes.

**Tech Stack:** Python 3.10, PyTorch, pytest, Bash, Slurm on Nibi.

**Spec:** `docs/superpowers/specs/2026-08-26-p09-year-corruption-groupdro-design.md`

## Global Constraints

- Initialize from accepted P02 and never modify P00/P02 artifacts.
- Use years 2016--2020, three equally likely block states, 15 groups, seed 0, batch 64, AdamW `1e-4`, GroupDRO step size `0.1`, and exactly 3,000 steps.
- Select only on 2021 M00/M01/M06/M07; access 2022--2023 only after the exact written gate passes.
- Add one focused CPU test file and no generic optimization framework.
- Run all training and dataset evaluation through Nibi Slurm compute nodes.

---

### Task 1: Environment and GroupDRO primitives

**Files:**
- Create: `tests/test_wsts_fast_track_environment_dro.py`
- Create: `reproductions/wsts_fast_track/environment_dro.py`

**Interfaces:**
- Produces: `environment_group(year: int, block_state: int) -> int`.
- Produces: `balanced_year_sampling_weights(dataset) -> torch.DoubleTensor`.
- Produces: `group_dro_objective(per_sample_losses, group_ids, log_weights, step_size=0.1)`.
- Produces: `install_training_environment_groups(upstream_root)`.

- [x] Write tests asserting exact group IDs, equal total sampler mass per year, movement toward a higher-loss group, and gradients to every per-sample loss.
- [x] Run the test and confirm RED because the module does not exist.
- [x] Implement the four minimal interfaces with exact validation and 15 fixed groups.
- [x] Run the test and confirm GREEN.
- [x] Commit the primitive and test.

### Task 2: Training, evaluation, and Nibi launchers

**Files:**
- Create: `reproductions/wsts_fast_track/train_environment_dro.py`
- Create: `reproductions/wsts_fast_track/evaluate_environment_dro.py`
- Create: `reproductions/wsts_fast_track/run_environment_dro_on_nibi.sh`
- Create: `reproductions/wsts_fast_track/run_environment_dro_evaluation_on_nibi.sh`

**Interfaces:**
- Training consumes accepted P02 record and emits `P09-YearCorruptionGroupDRO` checkpoint metadata plus a strict full-model state dict.
- Evaluation consumes P00, P02, P09 and emits same-population P00/P03/P09 metrics for M00/M01/M06/M07.

- [x] Implement the fixed 3,000-step trainer using inverse-year sampling, per-sample focal loss, and the pure GroupDRO objective.
- [x] Implement strict checkpoint loading and routed evaluation for 2021 or an explicitly authorized 2022/2023.
- [x] Add one selection launcher and one fixed-year launcher that archive the committed project before compute execution.
- [x] Run the focused test, Python compilation, both `bash -n` checks, and `git diff --check`.
- [x] Commit the executable campaign.

### Task 3: Run the registered decision

**Files:**
- Modify: `docs/experiments/p00_p06_rapid_reliability.md`
- Modify: `docs/research-roadmap.md`

- [x] Resolve P00/P02 records and submit one bounded H100 selection job.
- [x] Confirm allocation and monitor training plus 2021 evaluation to terminal status.
- [x] Apply the three-condition 2021 gate from the spec.
- [x] If selected, submit exactly one authorized evaluation job for each of 2022 and 2023 and apply the fixed temporal gate; otherwise stop without test access.
- [x] Record job IDs, metrics, decision, and artifact hashes; run final focused verification and commit the result.

# P08 Teacher-Posterior Belief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and submit one 3,000-step P08 run whose clean-input training posterior teaches a corrupted-input inference prior.

**Architecture:** Preserve the existing 41-channel evaluation contract. During P08 training only, append the clean 40-channel tensor to the corrupted input, forming 81 channels. Frozen P00 produces clean and corrupted decoder features; separate posterior and prior heads parameterize 16-channel diagonal Gaussians. Posterior samples train a shared residual output head plus a feature-reconstruction head, while evaluation samples only from the prior. Routing restores exact P00 logits outside the missing mask.

**Tech Stack:** Python 3.10, PyTorch, pytest, Bash, Slurm on Nibi.

**Spec:** `docs/superpowers/specs/2026-08-26-p08-teacher-posterior-belief-design.md`

## Constraints

- Freeze P00 and train only the posterior, prior, reconstruction, and output heads.
- Fixed values only: seed 0, four samples, Adam `1e-3`, 3,000 steps, KL weight `1e-3`, reconstruction weight `1e-2`.
- Select only on 2021 M00/M01/M06/M07 in this job.
- Add only one focused CPU behavior test; no generic framework or CLI test matrix.
- Never train or evaluate on the login node. Submit one Slurm GPU job.

### Task 1: Add the teacher-posterior model contract

**Files:**
- Modify: `tests/test_wsts_fast_track_residual_gate.py`
- Modify: `reproductions/wsts_fast_track/residual_gate.py`

- [x] Add one failing test covering 81-channel posterior training, finite forecast/KL/reconstruction losses, gradients in all four heads, 41-channel prior inference, non-negative uncertainty, and exact unmasked P00 preservation.
- [x] Run the focused test and confirm RED because P08 does not exist.
- [x] Add `FrozenTeacherPosteriorBelief`, P08 constants, prior/posterior parameterization with log-scale clamp `[-2, 2]`, reparameterized four-sample prediction, and the weighted training objective.
- [x] Run the focused test and confirm GREEN.
- [x] Commit the model and test.

### Task 2: Wire the fixed training/evaluation variant

**Files:**
- Modify: `reproductions/wsts_fast_track/residual_gate.py`
- Modify: `reproductions/wsts_fast_track/train_residual_gate.py`
- Modify: `reproductions/wsts_fast_track/evaluate_residual_gate.py`
- Modify: `reproductions/wsts_fast_track/run_residual_gate_on_nibi.sh`

- [x] Add a P08-only training preprocessor that emits `[corrupted40, mask1, clean40]`; leave all older preprocessors unchanged.
- [x] Add mutually exclusive `--teacher-belief` trainer/evaluator branches, fixed hyperparameters, loss-component logging every 100 steps, and strict checkpoint save/load for all four heads.
- [x] Add launcher variant `teacher` labelled `P08-teacher-posterior-belief`.
- [x] Run the one focused pytest, Python compilation, `bash -n`, and `git diff --check`.
- [x] Commit the wiring.

### Task 3: Submit and inspect one Nibi job

**Files:**
- Modify: `docs/experiments/p00_p06_rapid_reliability.md`

- [x] Resolve the accepted P00 record and confirm the target run directory does not exist.
- [x] Submit one bounded H100 Slurm job using launcher variant `teacher`; training and all 2021 evaluation must occur on the compute node.
- [x] Confirm scheduler command, allocation, and job state with `squeue`/`scontrol`.
- [x] Record commit, job ID, fixed experiment contract, and pending state in the experiment report; commit the record.
- [x] Monitor the job to completion, inspect its summary and losses, apply the written promotion rule, and update the report with the result.

# Three Directions Validation Implementation Plan

> Execute inline using executing-plans; preserve the user's lightweight research scope.

**Goal:** Obtain and report actual fixed-protocol evidence for N, W and D.
**Architecture:** Small experiment module reuses cross_history data/models and
shared evaluator. A normalization wrapper stores shared or conditional moments;
a separate driver calibrates/merges/trains and records immutable artifacts.
**Tech Stack:** Existing Python3.10/PyTorch2.6 environment, Nibi Slurm.
**Spec:** `docs/experiments/three-directions.md`

## Global constraints

Use the registered protocol verbatim. Never run Torch/data/model work on login.
Use final matched checkpoints, no test-year selection, no new dependencies.

## 1. Source manifest and correctness tests

- [ ] Freeze source metadata and SHA256; reject mismatched initialization,
  history, seed, method, updates or checkpoint structure.
- [ ] Add `tests/test_three_directions.py` using unittest: conditional BN
  equivalence before calibration; separate known means/variances; untouched
  affine; four mask groups including compound; midpoint and mismatched keys;
  same-input teacher choice; zero KL for identical logits and finite gradients.
- [ ] Observe failing CPU Slurm tests before implementation.

## 2. Normalization and midpoint experiments

- [ ] Implement `reproductions/three_directions/methods.py` with
  `condition_ids(packed)`, `StatsBatchNorm`, `StatisticsModel`,
  `midpoint_state(left,right)`, and `bernoulli_kl(student,teacher)`.
- [ ] Implement `run.py` that reads the source manifest, loads Forecaster
  strictly, preserves paired RNG, calibrates variants, writes checkpoints and
  four-scenario summaries using evaluate_batches. Modes: normalization, merge,
  student_control, student_distill, evaluate. Smoke limits are explicitly marked.
- [ ] Verify unit tests and real T1/T5 GPU smoke, then commit runnable code.
- [ ] Submit N/W seed0 full calibration and 2021 evaluation with source archive.

## 3. Distillation and evidence

- [ ] Complete registered four-teacher same-input routing and the matched
  no-KL student in driver; verify per-example teacher choice and detach.
- [ ] Submit D/control seed0 3000-update jobs after GPU smoke passes.
- [ ] Save source commit, source weights, job/time/memory, per-scenario AP,
  checkpoint parameter/storage size and finite loss logs.
- [ ] Implement lightweight paired report/gates and verify on synthetic rows
  that failures, missing histories, and missing scenarios cannot count as pass.
- [ ] Follow the registered confirmation gates; inspect actual terminal jobs
  and artifacts before advancing. Record negative results without retuning.
- [ ] Commit final tables and explain which hypotheses survived and why.

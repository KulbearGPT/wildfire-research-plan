# Cross-architecture Table 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal cross-history backbone interface and obtain gated SwinUnet T=1/T=5 evidence for the paper's Table 1.

**Architecture:** Preserve the existing feature-aware `Forecaster` and add a direct wrapper for schedule- and sampling-only methods. Checkpoints carry an explicit architecture identity and new backbones bootstrap once per seed before matched continuations branch from the same state.

**Tech Stack:** Python 3.10, PyTorch, pinned WSTS+ upstream models, Slurm, pytest, JSON comparison scripts.

**Spec:** `docs/superpowers/specs/2026-09-08-cross-architecture-table1-design.md`

## Global Constraints

- Run every model forward, smoke, training job, and evaluation job inside Slurm.
- Run SwinUnet T=1/T=5, SegFormer-B2 T=1, and ConvLSTM T=5 seed-0
  screens in parallel. Keep seeds 1/2 and held-out evaluations gated per
  architecture.
- Preserve the corrected 2016-2020/2021/2022-2023 split and paired target dates.
- Use one focused unit test per code boundary and no repository-wide validation suite.
- Commit each completed task before submitting dependent jobs.

---

### Task 1: Architecture registry and direct wrapper

**Files:**
- Create: `reproductions/cross_history/architectures.py`
- Modify: `reproductions/cross_history/models.py`
- Test: `tests/test_cross_history_architectures.py`

**Interfaces:**
- Produces: `canonical_architecture(history: int) -> str`.
- Produces: `validate_architecture(architecture: str, history: int) -> None`.
- Produces: `make_architecture(architecture: str, history: int, hparams: dict | None) -> torch.nn.Module`.
- Produces: `DirectForecaster(base: nn.Module, history: int)` with `forward` and `compute_loss`.

- [x] Write a focused test that verifies canonical backward compatibility,
  rejects `convlstm` with T=1, and confirms that the direct wrapper removes the
  two routing-mask channels before calling a dummy base model.
- [x] Run
  `pytest -q tests/test_cross_history_architectures.py` and confirm it fails
  because `reproductions.cross_history.architectures` does not exist.
- [x] Implement the registry with fixed compatibility sets:
  `res18_unet={1}`, `res18_utae={5}`, `swin_unet={1,5}`,
  `segformer_b2={1,5}`, and `convlstm={5}`. Implement the direct wrapper without
  importing heavyweight upstream models at module import time.
- [x] Run the focused test and confirm it passes.
- [x] Commit with `research: add cross-history architecture boundary`.

### Task 2: Runner checkpoint and bootstrap contract

**Files:**
- Modify: `reproductions/cross_history/run.py`
- Modify: `reproductions/cross_history/models.py`
- Test: `tests/test_cross_history_architectures.py`

**Interfaces:**
- Consumes: the Task 1 architecture registry and direct wrapper.
- Produces CLI flags `--architecture`, `--initial-checkpoint`, and
  `--bootstrap`.
- Produces checkpoint metadata fields `architecture` and `bootstrap`.

- [x] Extend the focused test with pure argument/checkpoint helpers proving
  that old history-only checkpoints infer the canonical architecture, explicit
  mismatches fail, and noncanonical training requires exactly one of bootstrap
  or an initial checkpoint.
- [x] Run the focused test and confirm the new assertions fail.
- [x] Refactor runner argument validation into importable pure helpers. Keep
  canonical default behavior byte-compatible. Select `DirectForecaster` only
  for noncanonical `control`, `cosine_erm`, or `block_specialist`; reject every
  other noncanonical method before calling `setup()` or loading data.
- [x] Save architecture/bootstrap metadata and enforce architecture, history,
  method, and block-fraction identity during evaluate-only loading.
- [x] Run the focused test and confirm it passes.
- [x] Commit with `research: add matched backbone bootstrap runs`.

### Task 3: Swin initialization portability

**Files:**
- Modify: `reproductions/cross_history/architectures.py`
- Modify: `reproductions/cross_history/run_slurm.sh`
- Test: `tests/test_cross_history_architectures.py`

**Interfaces:**
- Consumes: `make_architecture` from Task 1.
- Produces: an environment-independent Swin pretrained checkpoint path rooted
  under `/project/6085198/kulbear/wildfire/cache`.

- [x] Add a test that inspects the resolved Swin constructor arguments and
  proves no developer home-directory path is used.
- [x] Run the focused test and confirm it fails against the pinned upstream
  hard-coded path behavior.
- [x] Cache the exact WSTS+ Swin-T ImageNet checkpoint, record its SHA-256 in
  the architecture registry, and install the expected path only inside each
  Slurm run archive. Fail with a clear checksum error before model creation if
  it is absent or different.
- [x] Run the focused test and `bash -n reproductions/cross_history/run_slurm.sh`.
- [x] Commit with `cluster: make Swin initialization reproducible`.

### Task 4: Swin real-data smokes and seed-0 screen

**Files:**
- Modify: `docs/experiments/t1_t5_innovations.md`
- Create via jobs: `/project/6085198/kulbear/wildfire/runs/cross-history-{1,5}-*-<jobid>/result/`

**Interfaces:**
- Consumes: committed Task 3 runner and checksum-pinned initialization.
- Produces: T=1/T=5 smoke summaries, bootstrap checkpoints, and 2021 seed-0
  matched summaries for constant ERM, cosine ERM, and BlockDrop specialists.

- [x] Inspect Nibi GPU availability and test-only start estimates for 10GB and
  20GB H100 slices; request the earliest option within the two-times resource
  bound.
- [x] Submit one-step T=1/T=5 Swin smokes with the smallest physical batch that
  preserves effective batch 64. Verify output shape, finite loss, and
  checkpoint metadata from Slurm output.
- [ ] Submit one 10,000-step bootstrap per history for seed 0, then matched
  3,000-step constant, cosine, mixed BlockDrop, 25% BlockDrop, and 50%
  BlockDrop continuations with `afterok` dependencies.
- [ ] Evaluate 2021 and compose the full route. Stop a mechanism unless its
  primary delta is at least +0.005 in both histories and M00 is at least
  -0.010.
- [ ] Record job IDs, allocation choices, AP values, gates, and artifact paths
  in the campaign document.
- [ ] Commit with `results: record Swin seed-0 transfer screen`.

### Task 5: Promotion and Table 1 artifact

**Files:**
- Create: `reproductions/cross_history/compose_table1.py`
- Create: `tests/test_cross_history_table1.py`
- Modify: `docs/experiments/t1_t5_innovations.md`

**Interfaces:**
- Consumes: matched summary JSONs grouped by architecture, history, seed, year,
  and method.
- Produces: `table1.json` and Markdown rows containing test-only scenario AP,
  primary mean, seed standard deviation, and delta against ERM.

- [x] Write a tiny fixture test proving 2021 is excluded, 2022/2023 are both
  required, unmatched seed/backbone rows fail, and deltas use matched ERM.
- [x] Run the focused test and confirm it fails because the composer is absent.
- [x] Implement only the validated schema and Markdown rendering needed by the
  paper table.
- [x] Run the focused test and confirm it passes.
- [ ] If the Swin gate passed, submit seeds 1/2 and frozen 2022/2023 evaluations
  for advancing methods; otherwise emit no held-out jobs and record the failed
  architecture-transfer result.
- [ ] Generate the authoritative Table 1 artifact, run `git diff --check`, and
  verify all cited job states and JSON files before committing.
- [ ] Commit with `results: add cross-architecture Table 1 evidence`.

### Task 6: Conditional recurrent and second-Transformer breadth

**Files:**
- Modify: `reproductions/cross_history/architectures.py`
- Modify: `docs/experiments/t1_t5_innovations.md`

**Interfaces:**
- Consumes: the frozen method and gate from Task 5.
- Produces: optional ConvLSTM T=5 and SegFormer-B2 T=1 matched ERM/Ours rows.

- [x] Open the smoke/bootstrap/seed-0 portion in parallel; the user explicitly
  requested early submission while Swin remains queued.
- [ ] Run one-step Slurm smokes for ConvLSTM T=5 and SegFormer-B2 T=1 using
  their published initialization/loss/initial-LR recipes.
- [ ] Run seed-0 bootstrap plus matched ERM/Ours only; do not repeat X14/X22
  search or add architecture-specific tuning.
- [ ] Promote seeds 1/2 and 2022/2023 only when seed-0 primary delta is at least
  +0.005 with M00 at least -0.010.
- [ ] Regenerate Table 1, document negative outcomes as architecture limits,
  and commit with `results: extend Table 1 architecture breadth`.

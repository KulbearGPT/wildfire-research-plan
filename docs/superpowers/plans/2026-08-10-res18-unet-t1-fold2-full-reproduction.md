# Res18-U-Net T=1 Fold-2 Full Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete one official-protocol Fold 2 training/test run and independently verify the authors' released Fold 2 weight on the same test split.

**Architecture:** Add a full-run controller and verifier beside the existing calibration tooling, reusing only its audited process-observation and provenance primitives. Add a separate raw-state-dict evaluation entrypoint/controller so released weights cannot accidentally resume training or use Lightning checkpoint semantics. Every scientific launch is single-use and artifact-first.

**Tech Stack:** Python 3.10.4, PyTorch 2.0.0+cu118, PyTorch Lightning 2.0.1, official WildfireSpreadTS, pytest, standard-library JSON/CSV/hash verification, PowerShell process monitoring.

## Global Constraints

- Freeze the scientific protocol and revisions exactly as specified in `docs/superpowers/specs/2026-08-10-res18-unet-t1-fold2-full-reproduction-design.md`.
- Run the full experiment to exactly 10,000 optimizer steps; 1.5 hours is a progress checkpoint, not permission to truncate.
- The full run may launch once and may not automatically retry.
- The released-weight evaluation must start only after the full run reaches a terminal state.
- Do not push or merge. Keep ignored model/data artifacts out of Git.

---

### Task 1: Full Fold Controller and Independent Verifier

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/scripts/run_full_fold.py`
- Create: `reproductions/wsts_res18_unet_t1/scripts/verify_full_fold.py`
- Create: `tests/test_wsts_full_fold_runner.py`
- Create: `tests/test_wsts_full_fold_verifier.py`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

**Interfaces:**
- Produces `build_full_command(run_directory: Path) -> list[str]`, `validate_full_command(command: Sequence[str], run_directory: Path) -> None`, `parse_full_result(run_directory: Path) -> dict[str, object]`, and CLI actions `--preflight-only`, `--launch`, `--finalize-existing`.
- Produces independent CLI `verify_full_fold.py --run-directory ... --original-upstream ... --derived-upstream ... --patch ... --output ...`.

- [x] **Step 1: Write RED contract tests**

  Assert exact ordered command fields: three official YAMLs, Fold 2, All,
  `T=1`, deduplication, workers 8, `max_steps=10000`, local root, and
  `do_test=true`, with no extra argument. Assert atomic lock refusal, clean
  provenance gates, exact success sentinels, finite test metrics, unique best
  checkpoint, and malformed/missing raw artifacts fail closed.

- [x] **Step 2: Run RED tests**

  Run `python -m pytest tests/test_wsts_full_fold_runner.py tests/test_wsts_full_fold_verifier.py -q` and require failure because both modules are absent.

- [x] **Step 3: Implement the minimal controller and verifier**

  Reuse `observe_process`, `_sample_nvidia_smi`, `_prepare_derived_runtime`,
  inventory builders, atomic JSON writing, and the fixed environment from the
  calibration module. Use a new full-run artifact root and global lock. Copy
  provenance before launch, never retry, and validate post-run evidence before
  writing `completed.json` and `full-result.json`.

- [x] **Step 4: Verify GREEN and preflight**

  Run focused tests, the full suite, `git diff --check`, then
  `run_full_fold.py --preflight-only`. Confirm no training process or full-run
  lock exists.

- [x] **Step 5: Commit Task 1**

  Commit tracked files with `feat: add full fold reproduction runner`.

### Task 2: Released Weight Fetch and Evaluation

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/scripts/official_weight_entrypoint.py`
- Create: `reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py`
- Create: `reproductions/wsts_res18_unet_t1/scripts/verify_released_weight.py`
- Create: `tests/test_wsts_released_weight.py`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

**Interfaces:**
- Produces `validate_weight_manifest(items: Sequence[Mapping[str, object]]) -> dict[str, object]`, `validate_download(path: Path) -> None`, `build_weight_command(run_directory: Path, weight_path: Path) -> list[str]`, and CLI actions `--fetch-only`, `--preflight-only`, `--launch`, `--finalize-existing`.
- The entrypoint loads the raw state dict using `strict=True` and invokes exactly `Trainer.test`.

- [x] **Step 1: Write RED contract tests**

  Assert exact Hub revision/path/size/SHA, twelve-file manifest and filename AP
  aggregate, strict raw-state loading contract, test-only command, atomic
  evaluation lock, finite metric parsing, filename-versus-recomputed AP delta,
  and failures for training/predict/validate evidence.

- [x] **Step 2: Run RED tests**

  Run `python -m pytest tests/test_wsts_released_weight.py -q` and require
  failure because the evaluation modules are absent.

- [x] **Step 3: Implement fetch, test-only entrypoint, and verifier**

  Query the pinned Hub tree, download only Fold 2 into the ignored local cache,
  verify exact bytes/hash, instantiate the same official CLI/config, load with
  `strict=True`, call `Trainer.test`, preserve raw evidence, and independently
  verify all claims.

- [x] **Step 4: Verify GREEN and fetch-only gate**

  Run focused/full tests and `--fetch-only`; check the downloaded SHA and run
  `--preflight-only` without invoking a test loader.

- [x] **Step 5: Commit Task 2**

  Commit tracked files with `feat: add released weight verification`.

### Task 3: Execute Full Fold Once

**Files:**
- Produces ignored artifacts under `artifacts/reproductions/wsts-res18-t1-full/`.
- Modify after completion: `docs/experiments/res18_unet_t1_reproduction.md`

- [x] **Step 1: Run final launch gates**

  Re-run full tests, preflight, GPU/disk/process checks, exact command hash, and
  absence of the global full-run lock.

- [x] **Step 2: Launch once and monitor the same PID**

  Start `run_full_fold.py --launch` hidden, record the outer and child PID, and
  inspect saved stream events at the 1.5-hour checkpoint. Do not stop a healthy
  process and do not launch a replacement.

- [x] **Step 3: Finalize and independently verify**

  Require exact step 10,000, best checkpoint test completion, exit zero,
  unchanged sources, and independent verifier PASS. Record runtime, best
  validation AP/epoch/step, test metrics, GPU/memory statistics, and checkpoint
  SHA.

### Task 4: Evaluate Official Fold-2 Weight

**Files:**
- Produces ignored artifacts under `artifacts/reproductions/wsts-res18-t1-official-weight/`.
- Modify after completion: `docs/experiments/res18_unet_t1_reproduction.md`

- [x] **Step 1: Re-run weight preflight after training terminates**

  Require the exact pinned weight SHA, free GPU, no training process, unchanged
  data/code, and absence of the weight-evaluation launch lock.

- [x] **Step 2: Launch test-only evaluation once**

  Monitor the same PID to terminal state. Require strict load and forbid train,
  validation, predict, or checkpoint-resume actions.

- [x] **Step 3: Verify and compare**

  Independently recompute the result artifact, compare test AP with filename
  `0.571`, calculate the twelve-filename aggregate, and state its relation to
  the paper's `0.460 +/- 0.084` without rewriting provenance.

### Task 5: Final Scientific Report and Verification

**Files:**
- Modify: `docs/experiments/res18_unet_t1_reproduction.md`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

- [x] **Step 1: Write the evidence-bounded report**

  Present trained versus official-weight Fold 2 metrics, runtime/resource
  observations, configuration discrepancies, and next recommendation. Keep
  single-fold and twelve-fold claims separate.

- [x] **Step 2: Run completion audit**

  Run both independent verifiers, the full pytest suite, `git diff --check`,
  tracked/upstream/derived status checks, raw artifact hashes, and confirm no
  relevant Python process remains.

- [x] **Step 3: Commit without push or merge**

  Commit the final report with `docs: report fold 2 full reproduction`.

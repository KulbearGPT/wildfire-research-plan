# Res18-U-Net T=1 Fold-2 500-Step Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one provenance-controlled 500-optimizer-step timing calibration of the authors' Res18-U-Net, `T=1`, All-features model on official WSTS fold 2 and report a defensible 10,000-step runtime estimate.

**Architecture:** Keep the pinned authors' checkout and Python 3.10 environment outside tracked source. Add a small standard-library control layer that validates provenance, data inventory, effective overrides, and timing artifacts; invoke the official `src/train.py` unchanged through a subprocess wrapper that timestamps Lightning progress externally and samples `nvidia-smi`.

**Tech Stack:** Python 3.13 for tracked control/tests, Conda Python 3.10 for the official stack, PyTorch 2.0.0, torchvision 0.15.1, PyTorch Lightning 2.0.1, segmentation-models-pytorch 0.3.2, pytest, PowerShell, NVIDIA RTX 3090.

## Global Constraints

- Pin official code to `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad` and record the released-weight revision `acf70a37394849f4ec8d108a51d6f4325a554d0a`; do not modify the upstream checkout.
- Use only original WSTS years 2018--2021 and require file counts `2018=176`, `2019=74`, `2020=201`, `2021=156`, total `607`.
- Fold 2 is train `2018, 2020`, validation `2019`, test `2021`; the test loader must not be invoked.
- Freeze Res18-U-Net, `T=1`, All features, batch 64, crop 128, FP32, seed 0, focal loss, AdamW learning rate `0.001`, and the authors' dynamically derived positive-class weight.
- The only experiment overrides are `trainer.max_steps=500`, `do_test=false`, local paths, `WANDB_MODE=disabled`, logging/progress needed for timing, and one recorded Windows-safe `num_workers` value if 64 fails smoke testing.
- Keep `D:\WildFire Project\data\hdf5` read-only and do not modify the existing Python 3.13 environment.
- Calibration artifacts are local and ignored; only control code, tests, provenance documentation, and the final timing note are committed. Do not publish a 500-step AP claim on the website.
- Stop after exactly optimizer step 500. Do not continue to 10,000 steps, another fold, or checkpoint evaluation.

---

### Task 1: Reproduction Control Layer

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/upstream.lock.json`
- Create: `reproductions/wsts_res18_unet_t1/README.md`
- Create: `reproductions/wsts_res18_unet_t1/scripts/control.py`
- Create: `tests/test_wsts_reproduction_control.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `verify_inventory(data_root: Path) -> dict[str, int]`, `verify_upstream(upstream_root: Path, expected_commit: str) -> str`, `validate_overrides(overrides: Mapping[str, object]) -> None`, and `write_json_atomic(path: Path, payload: Mapping[str, object]) -> None`.
- Produces: immutable JSON keys `code.url`, `code.commit`, `weights.url`, `weights.revision`, `paper.target`, and `calibration` for later scripts.

- [x] **Step 1: Write failing control tests**

  Add tests that create fake year directories and assert exact acceptance of the four frozen counts; rejection of missing, extra, or wrong-year `.hdf5` files; exact acceptance of the allowlisted override keys and values; rejection of `do_test=true`, non-fold-2, non-All-features, non-500-step, batch-size, precision, optimizer, or model changes; atomic JSON replacement; and upstream commit mismatch. Stub `git rev-parse HEAD` through an injected command runner rather than touching a real repository.

- [x] **Step 2: Run the focused tests and confirm RED**

  Run `python -m pytest tests/test_wsts_reproduction_control.py -q` and require failure because `control.py` does not yet exist.

- [x] **Step 3: Implement the minimal standard-library control module**

  Define constants for frozen years/counts and the exact effective override mapping. Resolve every path before reading it, count only direct `*.hdf5` children of the four allowed year directories, reject any additional year selected for the calibration, obtain the upstream commit with `git -C <root> rev-parse HEAD`, compare mappings using both key and type equality, and publish JSON through a same-directory temporary file followed by `os.replace`.

  Provide a CLI with `inventory`, `upstream`, and `validate-overrides` subcommands. Each successful command prints one compact JSON object; every validation error writes a precise message to stderr and exits 2.

- [x] **Step 4: Add the lock manifest, operator README, and ignore rules**

  The lock manifest must carry the two frozen revisions and paper target `0.460 +/- 0.084`. The README must state that 500-step timing is not a scientific reproduction result and list the official baseline command. Add `/artifacts/reproductions/` and `/reproductions/wsts_res18_unet_t1/.local/` to `.gitignore` without broadening existing ignores.

- [x] **Step 5: Verify and commit Task 1**

  Run `python -m pytest tests/test_wsts_reproduction_control.py -q`, `python -m pytest -q`, and `git diff --check`. Commit only Task 1 files with `feat: add WSTS reproduction controls`.

### Task 2: Isolated Official Environment and Fold-2 Data Gate

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/scripts/bootstrap.ps1`
- Create: `reproductions/wsts_res18_unet_t1/scripts/smoke_fold2.py`
- Create: `tests/test_wsts_fold2_smoke.py`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

**Interfaces:**
- Consumes: the lock manifest and control functions from Task 1.
- Produces: an ignored pinned checkout at `reproductions/wsts_res18_unet_t1/.local/WildfireSpreadTS`, a dedicated Conda environment at `D:\WildFire Project\.conda-envs\wsts-res18-t1`, and `smoke.json` in the selected local run directory.
- Produces: `inspect_batch(batch: object) -> dict[str, object]` and `assert_fold_mapping(datamodule: object) -> dict[str, list[int]]` for the real smoke run.

- [x] **Step 1: Write failing smoke-unit tests**

  Use small NumPy arrays to require that `inspect_batch` records loader and model-boundary shapes, accepts finite `T=1`, 40-channel, 128x128 tensors with binary targets, and rejects non-finite inputs, non-binary targets, wrong temporal/channel/spatial dimensions, or equality between the input-day active-fire mask and next-day target. Use a fake datamodule to require the exact fold-2 year mapping.

- [x] **Step 2: Run the focused tests and confirm RED**

  Run `python -m pytest tests/test_wsts_fold2_smoke.py -q` and require failure because `smoke_fold2.py` does not yet exist.

- [x] **Step 3: Implement bootstrap and smoke scripts**

  `bootstrap.ps1` must be idempotent: clone the official repository only when absent, fetch and checkout the frozen commit in detached mode, fail if `git status --porcelain` is non-empty, create the prefix Conda environment with Python 3.10.4 only when absent, install the authors' pinned requirements using the CUDA-compatible PyTorch 2.0.0 wheels, and export `conda-list.txt`, `pip-freeze.txt`, GPU/driver, CPU, RAM, and disk metadata into the ignored run directory. It must never activate or install into the current environment.

  `smoke_fold2.py` must put only the pinned upstream `src` on `sys.path`, instantiate the official fold-2 datamodule with the frozen All-feature/T=1/crop settings, call `setup('fit')`, take exactly one seeded training batch, record both official loader output and flattened 40-channel model-boundary shapes, run the semantic assertions, and atomically emit `smoke.json`. It must not call `setup('test')`, `test_dataloader`, `trainer.test`, or read a 2021 sample.

- [x] **Step 4: Materialize the isolated environment and pass the real gate**

  Record source-data file size and modification-time inventories before the smoke. Run bootstrap, then run smoke initially with upstream `num_workers=64`. If native Windows fails specifically during worker creation/IPC, record the complete failure and retry loader smoke only with `8`, then `4`, then `0`, stopping at the first passing value. Do not change batch size or scientific settings. Cache the ResNet-18 ImageNet encoder weights before timing.

- [x] **Step 5: Verify and commit Task 2**

  Require the real `smoke.json` to state total 607, fold mapping `train=[2018,2020]`, `validation=[2019]`, `test=[2021]`, `T=1`, model channels 40, crop 128, finite inputs, binary targets, distinct next-day target, and no test-loader call. Recompute the source-data inventory and require byte size and modification time unchanged. Run both focused tests, the full suite, and `git diff --check`; commit tracked Task 2 files with `feat: add official fold 2 smoke gate`.

### Task 3: External Timing Runner and 500-Step Calibration

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/scripts/run_calibration.py`
- Create: `reproductions/wsts_res18_unet_t1/scripts/official_entrypoint.py`
- Create: `tests/test_wsts_calibration_runner.py`
- Create: `docs/experiments/res18_unet_t1_reproduction.md`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

**Interfaces:**
- Consumes: the pinned checkout/environment, passing `smoke.json`, and selected `num_workers` from Task 2.
- Produces: `parse_progress(text: str) -> list[tuple[int, float]]`, `summarize_timing(samples: Sequence[tuple[int, float]], wall_seconds: float, startup_seconds: float, validation_seconds: float) -> dict[str, float]`, and a CLI that creates one unique ignored run directory.
- Produces: `timing.json`, `calibration.csv`, `effective-command.json`, `stdout.log`, `stderr.log`, `gpu.csv`, `exit-code.txt`, provenance copies, and a local Markdown report.

- [x] **Step 1: Write failing timing and command tests**

  Provide captured Lightning progress fragments containing carriage-return updates and assert monotonic `(optimizer_step, timestamp)` extraction, duplicate-step collapse, exclusion of warm-up steps 0--49, median instantaneous seconds per step, samples/second at batch 64, and `10,000 * median_step_seconds` as an optimistic empirical compute-only extrapolation/reference. Separately reconstruct first-step epoch boundaries so recurring loader/validation gaps produce an epoch-aware 10,000-step central/range estimate; also report complete-wall linear scaling and end-to-end throughput. Require rejection when progress never reaches exactly 500, steps regress, values are non-finite, or the effective command contains `do_test=true`, a test/predict action, another fold, another feature subset, or a non-allowlisted scientific override.

- [x] **Step 2: Run focused tests and confirm RED**

  Run `python -m pytest tests/test_wsts_calibration_runner.py -q` and require failure because `run_calibration.py` does not yet exist.

- [x] **Step 3: Implement the observer-only runner**

  Build the official command from the three upstream YAML files and these explicit overrides: `data.data_dir`, `data.data_fold_id=2`, `data.features_to_keep=null`, `data.n_leading_observations=1`, `data.remove_duplicate_features=true`, selected `data.num_workers`, `trainer.max_steps=500`, local `trainer.default_root_dir`, and `do_test=false`. Set `WANDB_MODE=disabled`, `WANDB_SILENT=true`, deterministic cache locations, and do not change the upstream files. Do not add explicit `do_predict` or `do_validate` overrides: they are unapproved changes to official defaults.

  Keep the original pinned checkout clean. Create an ignored derived checkout
  at the same commit and apply the tracked runtime-safety import-scope patch,
  which only removes seven unrelated eager architecture exports while leaving
  the four imports required by `train.py` unchanged. `official_entrypoint.py`
  must add only that derived checkout's `src` directory to `sys.path`, execute
  the unchanged official `src/train.py` in the same
  process, and after normal return print one machine-readable sentinel holding
  `torch.cuda.max_memory_allocated()`. It must not monkeypatch the model,
  datamodule, optimizer, callbacks, trainer, or metrics.

  Launch that entrypoint with the official environment's Python, timestamp
  stdout/stderr progress externally, and schedule `nvidia-smi` against absolute
  monotonic one-second deadlines into `gpu.csv` so query duration does not
  accumulate. Always record the actual mean/median interval and effective Hz;
  an observed peak may miss between-sample transients. Always record the child exit code. On success, require exact
  step 500, finite final loss, a finite peak-allocated sentinel, no OOM, no
  test/predict markers, and a clean upstream checkout; then atomically write
  the timing summaries and effective command. Do not automatically retry the
  training process.

- [x] **Step 4: Run the authorized calibration lineage**

  Recheck GPU availability, provenance, disk space, data inventory, and `smoke.json`. Every launch must have an atomic lineage lock and the runner must never retry automatically. Preserve an import-time failure before Trainer as attempt 1 and an explicit runtime-patch recovery as attempt 2. If Windows `num_workers=64` exhausts CPU allocator memory during validation sanity checking before any optimizer step, require a separate no-training train+validation one-batch smoke at workers 8 with RAM evidence, then permit one final explicit worker-recovery attempt whose only command change is `data.num_workers=8` (besides the unique output directory). No fourth launch is permitted. If the outer shell times out while a child remains alive, attach to and monitor that same PID; never launch a replacement. Record peak allocated GPU memory when Lightning exposes it, peak `nvidia-smi` memory, utilization distribution, total wall time, observed training/progress interval, median post-warm-up step time, throughput, startup/validation overhead, and the extrapolated 10,000-step range.

- [x] **Step 5: Independently verify artifacts and analyze the result**

  Recompute timing statistics from raw timestamp/progress and GPU CSV files in a separate process. Require each command-lineage marker's schema-specific command/hash fields with exact types and values. Validate both source snapshots internally: exact data root, 607 unique direct year/file paths, year counts `176/74/201/156`, entry count, summed byte size, and integer nanosecond mtimes; independently stat the live 2018--2021 tree and require exact equality with the post snapshot. Verify child exit code 0, exact step 500, finite peak allocated memory from the wrapper sentinel, absence of test invocation, and unchanged upstream status. Compare the extrapolated one-fold time with the paper's approximate `0.4 h` entry while explicitly noting that the paper does not document hardware and that its time is not directly equivalent to this end-to-end calibration.

- [x] **Step 6: Document and commit Task 3**

  Write the exact hardware, selected worker count, environment revisions, measured times, throughput, GPU memory/utilization, extrapolation formula/range, warnings, and interpretation to `docs/experiments/res18_unet_t1_reproduction.md`. State prominently that this is timing calibration only, contains no test AP, and cannot establish reproduction of `0.460 +/- 0.084`. Run the focused and full test suites, `git diff --check`, and a secret/path scan; commit tracked files with `docs: record fold 2 timing calibration`.

### Task 4: Whole-Run Verification and Handoff

**Files:**
- Modify only if verification finds an evidence defect: files created in Tasks 1--3.

**Interfaces:**
- Consumes: all committed control code and the complete ignored artifact directory.
- Produces: a reviewer-approved branch and a concise PI-facing result report; no push and no merge.

- [ ] **Step 1: Run final verification**

  Run `python -m pytest -q`, `git diff --check`, `git status --short`, confirm the pinned upstream checkout is clean, independently validate each source-data snapshot and compare pre/post/live metadata, and validate every required artifact exists and is mutually consistent.

- [ ] **Step 2: Conduct broad review**

  Review the whole plan diff for provenance correctness, accidental scientific changes, test-loader leakage, unsupported claims, ignored large artifacts, secret leakage, and exact numerical agreement between raw evidence, JSON, CSV, Markdown, and the user-facing report. Resolve any Critical or Important finding through the prescribed fix/re-review loop.

- [ ] **Step 3: Report without publishing**

  Tell the user the measured 500-step wall time, steady-state step time, throughput, peak GPU memory, utilization, startup/validation overhead, 10,000-step one-fold estimate, and practical 12-fold implication. Separate observation from inference, describe any official-code/platform issue encountered, link the committed plan and experiment note, and explicitly state that nothing was pushed or merged.

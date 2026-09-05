# T=1 Research Mainline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Leave one reproducible `T=1` research mainline while preserving all removed Git and Nibi artifacts in explicit archives.

**Architecture:** Freeze the current Git tree on an archive branch, collapse the retained B0/B2/B3, D1, D2-STD, and D12 code into focused modules, and replace chronological experiment notes with current-results and rejected-results documents. Move obsolete `/project` run directories and Slurm logs to a same-filesystem archive using an explicit manifest.

**Tech Stack:** Python 3.10+, PyTorch, Bash/Slurm, Git, Markdown

**Spec:** `docs/superpowers/specs/2026-09-04-t1-research-mainline-cleanup-design.md`

## Global Constraints

- Do not submit Slurm jobs or perform training/evaluation on the login node.
- Preserve pre-cleanup Git state at `archive/pre-t1-cleanup-2026-09-04` commit `4b843ad`.
- Move ignored run artifacts to `/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs`; do not compress them.
- Keep active artifacts only for B0, B2, B3, D1-ERM, D1-KL, D2-STD, D12-SARP, and their selected 2022/2023 evaluations.
- Keep shared data, environments, upstream caches, official weights, and data-preparation evidence outside artifact cleanup.
- Use one document for current quantitative results and one document for rejected/out-of-scope results.
- Run only lightweight targeted verification.

---

### Task 1: Freeze the recovery points and inventory artifacts

**Files:**
- Create: `docs/experiments/artifact-archive-manifest.tsv`
- Modify: `docs/superpowers/plans/2026-09-04-t1-research-mainline-cleanup.md`

**Interfaces:**
- Consumes: Git commit `4b843ad`; top-level entries in `/project/6085198/kulbear/wildfire/runs`.
- Produces: one tab-separated row per moved artifact with `kind`, `source`, `destination`, `bytes`, and `reason`.

- [ ] **Step 1: Verify the archive branch and clean worktrees**

Run `git show --no-patch --oneline archive/pre-t1-cleanup-2026-09-04`, `git status --short`, and `git worktree list`.

- [ ] **Step 2: Verify no Slurm jobs are active**

Run `squeue -u kulbear -o '%.18i %.9P %.30j %.8T %.10M %.10l %.6D %R'`; the artifact move proceeds only when no rows follow the header.

- [ ] **Step 3: Classify every top-level run entry**

Retain data-preparation evidence plus the exact successful training/evaluation entries for B0, B2, B3, D1-ERM, D1-KL, D2-STD, and D12-SARP. Mark every other top-level run directory or `slurm-*.out` file for archival with one of: `T5`, `invalid-resolver`, `rejected`, `failed-retry`, or `superseded`.

- [ ] **Step 4: Write and reconcile the manifest**

Generate explicit source/destination paths and byte sizes; require that retained plus archived entries equal the complete pre-move top-level inventory.

- [ ] **Step 5: Commit the recovery manifest with later artifact-state updates**

The initial classification remains uncommitted until Task 4 fills the final status and archive totals.

### Task 2: Collapse code to the retained `T=1` dependency graph

**Files:**
- Create: `reproductions/wsts_fast_track/corruption_training.py`
- Create: `reproductions/wsts_fast_track/processed_reliability.py`
- Create: `reproductions/wsts_fast_track/severity_adaptive_reliability_prompting.py`
- Create: `reproductions/wsts_fast_track/train_standard_reliability_control.py`
- Create: `reproductions/wsts_fast_track/evaluate_standard_reliability_control.py`
- Create: `reproductions/wsts_fast_track/train_severity_adaptive_reliability_prompting.py`
- Create: `reproductions/wsts_fast_track/evaluate_severity_adaptive_reliability_prompting.py`
- Create: `reproductions/wsts_fast_track/run_standard_reliability_control_on_nibi.sh`
- Modify: `reproductions/wsts_fast_track/corrected_baselines.py`
- Modify: `reproductions/wsts_fast_track/run_corrected_baseline_on_nibi.sh`
- Modify: `reproductions/wsts_fast_track/run_reliability_evaluation_on_nibi.sh`
- Modify: `reproductions/wsts_fast_track/run_severity_adaptive_reliability_prompting_on_nibi.sh`
- Modify: `reproductions/wsts_fast_track/run_d12_heldout_on_nibi.sh`
- Modify: retained imports and tests under `reproductions/wsts_fast_track/` and `tests/`
- Delete: rejected prototype modules, evaluators, trainers, runners, completion helpers, and tests listed by the design spec.

**Interfaces:**
- Consumes: corrected five-year index resolution; B3 completion record; 40-channel processed C00 input; appended binary invalidity map.
- Produces: `CORRECTED_BASELINES={B0,B2,B3}`; D1 ERM/KL runners; D2-STD checkpoint contract; D12-SARP checkpoint contract and held-out evaluator.

- [ ] **Step 1: Extract corrected index and B2/B3 corruption code**

Move `resolve_dataset_index`, FireDrop, and BlockDrop support into the retained modules; remove P00/P01/P02 IDs, the rejected validity-channel path, and year-balanced B4 code.

- [ ] **Step 2: Restrict corrected baselines**

Register only B0, B2, and B3 and restrict `run_corrected_baseline_on_nibi.sh` to those identifiers.

- [ ] **Step 3: Extract the shared processed reliability dataset**

Keep `apply_processed_reliability_corruption` and `ProcessedReliabilityDataset` without RNC/D4 model variants.

- [ ] **Step 4: Make D2-STD a focused control**

The trainer always feeds the first 40 channels, writes candidate `D2-STD`, variant `standard`, matched pair `D2`, and the evaluator rejects all other checkpoint identities.

- [ ] **Step 5: Make D12 a focused implementation**

Keep the input invalid-region token, deep prompt tokens, threshold `0.375`, mild/deep and severe/input routing, and exact D12 checkpoint contract. Remove D10/D11 selectable variants and validators.

- [ ] **Step 6: Restrict held-out evaluation dispatch**

Allow only `baseline`, `d1`, `d2`, and `sarp`, dispatching D2 and SARP to the newly focused evaluators.

- [ ] **Step 7: Remove rejected experiment files**

Delete T5, P-series, belief-state, VIIRS, GroupDRO, RNC, counterfactual D5--D9, standalone D4/D10/D11, promotion, and stale matrix orchestration entry points plus tests that exist only for them.

- [ ] **Step 8: Run targeted code verification**

Run `python -m compileall -q` on retained packages, `bash -n` on retained Nibi runners, and targeted pytest files for corrected baselines, D1, D2-STD, D12, evaluation, corruption, and cluster scripts.

- [ ] **Step 9: Commit code cleanup**

Commit retained implementation and test changes as `refactor: focus fast track on retained T1 methods`.

### Task 3: Replace historical narrative with two quantitative records

**Files:**
- Rewrite: `README.md`
- Rewrite: `reproductions/wsts_fast_track/README.md`
- Rewrite: `docs/experiments/quantitative_reliability_ledger.md`
- Create: `docs/experiments/rejected_experiments.md`
- Modify: `index.html`
- Modify: `docs/research-roadmap.md`
- Delete: `docs/experiments/p00_p06_rapid_reliability.md`
- Delete: obsolete files under `docs/superpowers/plans/` and `docs/superpowers/specs/`, except this cleanup plan/design.

**Interfaces:**
- Consumes: frozen quantitative values and job IDs already recorded in the old ledger.
- Produces: one current-results source of truth and one rejected-results source of truth.

- [ ] **Step 1: Write the current result table**

Record official T1 reproduction, B0/B2/B3, D1-ERM/KL, D2-STD, and D12 for 2021--2023, including matched deltas and the clean-performance boundary.

- [ ] **Step 2: Write the rejected result table**

Preserve concise numeric evidence and job IDs for invalid legacy runs, P01--P13, belief-state, VIIRS, B4, D2-RNC, D3--D11, and T5/D13; link the Git and artifact archives.

- [ ] **Step 3: Rewrite entry documentation**

Make README, fast-track README, roadmap, and public status describe only the retained T1 execution path and point to the two result documents.

- [ ] **Step 4: Remove obsolete plans and experiment narratives**

Keep the cleanup design/plan plus foundation experiment reports; use the archive branch for detailed historical plans.

- [ ] **Step 5: Check references**

Use `rg` to find references to removed paths and runnable B1/B4/B5/D2-RNC/D4--D13/P-series/T5 interfaces; allow those identifiers only inside `rejected_experiments.md` when they identify historical evidence.

- [ ] **Step 6: Commit documentation cleanup**

Commit as `docs: present the retained T1 research mainline`.

### Task 4: Move obsolete Nibi artifacts and remove stale worktrees

**Files:**
- Finalize: `docs/experiments/artifact-archive-manifest.tsv`
- Remove locally: generated `.pytest_cache`, `__pycache__`, and rejected clean worktrees.

**Interfaces:**
- Consumes: reconciled manifest from Task 1.
- Produces: clean active `/project/.../runs`, recoverable archive directory, finalized manifest.

- [ ] **Step 1: Recheck queue and exact source roots**

Require an empty user queue, resolve every source under the exact runs root, reject symlink targets, and require the archive destination not to exist.

- [ ] **Step 2: Create the archive directory**

Create `/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs` with the existing project group inheritance.

- [ ] **Step 3: Move manifest entries**

Move each explicitly listed top-level entry to its explicit destination and update its manifest status to `archived`.

- [ ] **Step 4: Reconcile active and archived inventories**

Confirm that each retained entry remains active, every archive entry exists at its destination, no manifest source remains, and byte totals match the pre-move classification.

- [ ] **Step 5: Remove rejected clean worktrees and generated caches**

Remove only worktrees with empty `git status --short`; keep their branches. Remove generated test caches after tests finish.

- [ ] **Step 6: Commit artifact manifest**

Commit as `chore: archive obsolete wildfire experiment artifacts`.

### Task 5: Final lightweight audit

**Files:**
- Modify only if verification exposes a retained-path defect.

**Interfaces:**
- Consumes: cleaned Git tree, active run root, artifact archive, and manifest.
- Produces: fresh evidence that the requested scope is met.

- [ ] **Step 1: Run retained tests and syntax checks**

Run the targeted tests from Task 2 and every retained shell script through `bash -n`.

- [ ] **Step 2: Audit repository scope**

Run `git status --short`, `git diff --check HEAD~3..HEAD`, `rg --files`, and identifier/path searches against the documentation rules.

- [ ] **Step 3: Audit archives and active artifacts**

Verify archive branch commit, external manifest reconciliation, active run allowlist, and total archived byte count.

- [ ] **Step 4: Report commits and recovery paths**

Report exact commit hashes, retained scientific chain, test counts, active/archive artifact counts and sizes, and recovery instructions.

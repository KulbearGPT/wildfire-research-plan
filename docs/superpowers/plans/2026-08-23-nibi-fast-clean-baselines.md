# Nibi Fast Clean-Baseline Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the first two WSTS+ clean learned experiments on Nibi: C00 Res18-UNet T=1 All and C02 UTAE(Res18) T=5 Multi, each for 3,000 optimizer steps with seed 0.

**Architecture:** Reuse the pinned upstream training stack and the qualified Nibi Python/CUDA environment. Add only a project-level fast-track contract and entrypoint that enforce the frozen train/validation years, prevent frozen-test evaluation, render the two exact commands, and seal compact run evidence. Prepare the 999-event active-fixed WSTS+ tree from the two official Zenodo archives before either GPU job starts.

**Tech Stack:** Python 3.13 control tests, Python 3.10 scientific environment, PyTorch 2.6.0, PyTorch Lightning 2.0.2, Slurm, HDF5, Zenodo WSTS/WSTS+ archives, pinned WildfireSpreadTS commit `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`.

**Spec:** `docs/research-roadmap.md`, with the user-approved Fast Experiment Track override from 2026-08-23: defer Stage 0 numerical equivalence and Stage 1 sensitivity, retain only data/split/numerical/run-lineage guards, and begin Stage 2 seed-0 screening immediately.

## Global Constraints

- Use exactly 999 active-fixed event HDF5 files: 2016=92, 2017=110, 2018=176, 2019=74, 2020=201, 2021=156, 2022=122, 2023=68.
- Train on 2016–2020, validate on 2021, and never evaluate or select on 2022–2023 in this campaign.
- Use seed 0, 3,000 optimizer steps, one H100, 32-bit precision, and the same training budget for C00 and C02.
- C00 is Res18-UNet, T=1, All 40 features; C02 is the upstream `SMPTempModel` published as Res18-UTAE, T=5, Multi 33 features, whose temporal block uses relative positions `0..4` internally.
- Preserve upstream source and archives. Runtime compatibility changes live in a derived checkout and are captured as a diff in every run.
- A GPU job writes a new immutable run directory once and never silently retries.
- Run only focused contract tests during implementation; the full historical reproduction suite is not a launch gate.

---

### Task 1: Publish the Fast Experiment Track

**Files:**
- Modify: `README.md`
- Modify: `docs/research-roadmap.md`
- Create: `reproductions/wsts_fast_track/README.md`

**Interfaces:**
- Consumes: the approved Fast Experiment Track and frozen Phase 0 split.
- Produces: the authoritative immediate order C00/C02 seed-0 screening, followed by promotion rather than Stage 0/1 completion.

- [ ] **Step 1: Update the human-facing roadmap**

  State that the Nibi engineering smoke passed, the remaining cluster-equivalence and positive-weight sensitivity work is deferred, and the immediate launch gate is only the 999-event inventory plus a finite one-batch preflight embedded in each experiment job.

- [ ] **Step 2: Document result boundaries**

  Label 3,000-step results as seed-0 screening, prohibit 2022–2023 access, and require 10,000-step promotion before any model-quality claim.

- [ ] **Step 3: Check Markdown and commit with Task 2**

  Run `git diff --check`. Human prose receives no brittle source-text test.

### Task 2: Add the Frozen-Split Fast-Track Contract

**Files:**
- Create: `reproductions/wsts_fast_track/contract.py`
- Create: `reproductions/wsts_fast_track/compute_stats.py`
- Create: `tests/test_wsts_fast_track.py`

**Interfaces:**
- Produces: `ExperimentSpec`, `experiment_spec(experiment_id)`, `frozen_fit_split()`, `validate_inventory(root)`, and `upstream_arguments(spec, upstream_root, data_root, run_root)`.
- Produces: `compute_stats.compute_paths(paths)` and a CLI that writes train-only `means`, `stds`, and `missing_values` arrays.
- Consumes: direct year directories containing event-level `.hdf5` files.

- [ ] **Step 1: Write RED tests for the two experiment commands**

  ```python
  def test_c00_and_c02_change_only_declared_model_inputs(tmp_path: Path) -> None:
      c00 = contract.upstream_arguments(
          contract.experiment_spec("C00"), tmp_path / "data", tmp_path / "c00"
      )
      c02 = contract.upstream_arguments(
          contract.experiment_spec("C02"), tmp_path / "data", tmp_path / "c02"
      )
      assert "--trainer.max_steps=3000" in c00
      assert "--trainer.max_steps=3000" in c02
      assert "--data.n_leading_observations=1" in c00
      assert "--data.n_leading_observations=5" in c02
      assert "--data.features_to_keep=null" in c00
      assert "--data.features_to_keep=[0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,38,39]" in c02
      assert "--do_test=false" in c00
      assert "--do_test=false" in c02
  ```

- [ ] **Step 2: Run RED**

  Run `python -m pytest tests/test_wsts_fast_track.py -q` and require import failure because the module does not exist.

- [ ] **Step 3: Implement literal specs and command rendering**

  `C00` selects upstream `cfgs/unet/res18_monotemporal.yaml` and `cfgs/data_monotemporal_full_features.yaml`. `C02` uses the same focal-loss/AdamW base, selects `cfgs/data_multitemporal_multi_features.yaml`, and overrides the class path to `models.SMPTempModel`, the upstream Res18 encoder plus LTAE implementation used for the released `Res18UTAE_T5` weights. It does not request day-of-year input because that model constructs relative positions `0..T-1` internally. Both set `do_train=true`, `do_validate=true`, `do_test=false`, `trainer.max_steps=3000`, `trainer.num_sanity_val_steps=0`, batch size 64, and eight workers.

- [ ] **Step 4: Write RED inventory tests**

  Build literal empty `.hdf5` fixtures with the eight expected year counts and assert success. Remove one event, add a ninth year, and add a nested event separately; each mutation must raise `ValueError`.

- [ ] **Step 5: Implement the lightweight inventory check**

  Inspect only direct file names and year counts. Do not hash or open all HDF5 payloads here; the job preflight opens one train and one validation sample.

- [ ] **Step 6: Run GREEN**

  Run `python -m pytest tests/test_wsts_fast_track.py -q` and require all focused tests to pass.

### Task 3: Add the Frozen-Split Upstream Entrypoint

**Files:**
- Create: `reproductions/wsts_fast_track/entrypoint.py`
- Modify: `tests/test_wsts_fast_track.py`

**Interfaces:**
- Produces CLI: `entrypoint.py --upstream-root PATH --experiment {C00,C02} --data-root PATH --run-root PATH --stats-path PATH`.
- Consumes: `contract.upstream_arguments` and the pinned upstream `src/train.py`.

- [ ] **Step 1: Write RED entrypoint guard tests**

  ```python
  def test_entrypoint_uses_frozen_fit_years_and_withholds_test() -> None:
      assert entrypoint.frozen_fit_split(0, False) == (
          [2016, 2017, 2018, 2019, 2020], [2021], [2021]
      )

  def test_entrypoint_rejects_any_test_enablement(tmp_path: Path) -> None:
      with pytest.raises(ValueError, match="frozen test is withheld"):
          entrypoint.validate_forwarded_arguments(["--do_test=true"])
  ```

- [ ] **Step 2: Run RED**

  Require failure because `entrypoint.py` is absent.

- [ ] **Step 3: Implement the minimal runtime adapter**

  Add upstream `src` to `sys.path`, import `FireSpreadDataModule`, replace only its `split_fires` static method with the frozen fit split, reject caller-supplied test or split overrides, set `sys.argv` from the literal contract, and execute pinned `train.py` with `runpy.run_path`. Return test years as `[2021]` because upstream constructs a test dataset even during fit; `do_test=false` guarantees it is never evaluated and prevents any 2022–2023 file access.

  Load the canonical train-only NPZ and replace both upstream references to `get_means_stds_missing_values`. Return copies and preserve upstream handling for direction and land-cover features. This is necessary because upstream does not publish statistics for the frozen five-year training tuple.

- [ ] **Step 4: Run GREEN and import the real derived upstream**

  Run the focused pytest file, then invoke the entrypoint with `--help` against the derived checkout to prove real import compatibility without starting training.

### Task 4: Prepare the Active-Fixed WSTS+ Tree on Nibi

**Files:**
- Produces ignored/external data under `/project/6085198/kulbear/wildfire/hdf5/wstsplus-active-fixed`.
- Produces a conversion record under `/project/6085198/kulbear/wildfire/runs/nibi-wstsplus-data-20260823`.

**Interfaces:**
- Consumes: original WSTS archive MD5 `dc1a04e63ccc70037b277d585b8fe761` and WSTS+ v1 archive MD5 `42da7598cc33a170064e78d8027148c9`.
- Produces: the exact eight-year event counts required by Task 2.

- [ ] **Step 1: Download and checksum WSTS+ v1 once**

  Run CPU job `20346120`. Activate `/project/6085198/kulbear/wildfire/downloads/WSTSPlus.zip` only after the declared MD5 passes.

- [ ] **Step 2: Inspect the archive and select the minimal conversion path**

  If it contains HDF5, extract directly to staging. If it contains GeoTIFF event directories, run the pinned upstream converter to staging. Preserve source archive and extracted TIFFs until repair completes.

- [ ] **Step 3: Apply the existing active-fire repair to added years**

  Use `wildfire_phase0 repair-active-fire` for 2016, 2017, 2022, and 2023 against staging, then run `verify-active-fire-repair` with the frozen year totals from `docs/experiments/phase0.md`.

- [ ] **Step 4: Assemble and inventory the active tree**

  Reuse the converted original WSTS years 2018–2021 without modifying them, activate the 392 nonempty added-year events, and require exact counts `92/110/176/74/201/156/122/68`. Open one train HDF5 and one validation HDF5, produce one dataset item for T=1 and T=5, and require finite tensors and next-day targets.

- [ ] **Step 5: Seal a compact data summary**

  Run `compute_stats.py` across the 653 training events and save `train-2016-2020-stats.npz`. For active fire, treat zero as absence and compute detection-hour moments only over positive pixels, matching the upstream positive-class-weight convention. Record archive URLs/checksums, per-year counts, total bytes, repair verifier result, roots, statistics path and positive-fire rate, and conversion Slurm job IDs in `data-summary.json`.

### Task 5: Run C00 and C02 Seed-0 Screening

**Files:**
- Produces external run directories under `/project/6085198/kulbear/wildfire/runs/fast-track-{C00,C02}-JOBID`.

**Interfaces:**
- Consumes: Task 3 entrypoint, Task 4 data root, the qualified PyTorch 2.6.0 `sm_90` environment, and the derived upstream checkout.
- Produces: best checkpoint, final 2021 validation metrics, effective config, logs, environment freeze, source diff, GPU identity, peak CUDA allocation, and `completed.json` for each experiment.

- [ ] **Step 1: Qualify C02 imports on CPU**

  Restore only the required `SMPTempModel` export in the derived upstream. Ensure both the upstream root and `src` are importable because this model refers to `src.models.utae_paps_models`. If import fails under PyTorch 2.6, diagnose each incompatibility and record a minimal runtime-only patch; do not restore unrelated Swin/Transformer modules.

- [ ] **Step 2: Run one embedded H100 preflight per model**

  The same job that will train loads one train and one validation batch, runs forward/backward, asserts finite loss/gradients, then begins the 3,000-step fit. A failed preflight exits the job before optimizer training.

- [ ] **Step 3: Submit both jobs once**

  Request one H100, eight CPUs, 96 GiB RAM, and a time limit derived from the completed one-step smoke plus an initial conservative ceiling. Do not submit an automatic retry.

- [ ] **Step 4: Verify each terminal result**

  Require Slurm `COMPLETED` with exit `0:0`, the exact `max_steps=3000` stop marker, a positive CUDA peak allocation, a loadable best checkpoint, and a finite 2021 `val_avg_precision`. Confirm the recorded command contains `do_test=false` and no 2022/2023 path was opened.

- [ ] **Step 5: Publish the screening comparison**

  Add C00/C02 runtime, parameters, best validation AP, loss, F1, checkpoint path, and job IDs to `reproductions/wsts_fast_track/README.md`. State that one seed and 3,000 steps establish screening evidence only; recommend which model(s) should be promoted to 10,000 steps.

- [ ] **Step 6: Run final focused verification and commit**

  Run `python -m pytest tests/test_wsts_fast_track.py tests/test_project_research_organization.py -q`, `git diff --check`, inspect `git status`, and commit the tracked fast-track implementation and reviewed result summary.

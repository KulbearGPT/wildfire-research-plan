# Res18-U-Net T=1 Official Weights 12-Fold Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute and independently verify all twelve official Res18-U-Net `T=1` All-feature released-weight test results, then publish the measured twelve-fold aggregate with bounded comparisons to the paper and filename aggregates.

**Architecture:** Extract a fold-neutral immutable weight contract from the already audited Fold 2 tooling, generalize the single-fold controller/verifier without changing its scientific entrypoint, and add a sequential campaign controller plus an independent raw-evidence aggregator. Reuse the qualifying Fold 2 run, launch each other fold at most once, stop on first failure, and update the experiment report and website only after all twelve independent verifications pass.

**Tech Stack:** Python 3.10.4, PyTorch 2.0.0+cu118, PyTorch Lightning 2.0.1, official WildfireSpreadTS commit `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`, Hugging Face model revision `acf70a37394849f4ec8d108a51d6f4325a554d0a`, pytest, standard-library JSON/CSV/hash verification, PowerShell process monitoring.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-11-res18-unet-t1-official-weights-12fold-design.md` exactly.
- Evaluate folds `0..11` from the official hard-coded `FireSpreadDataModule.split_fires` tuples; do not invent or substitute a split file.
- Use Res18-U-Net, `T=1`, all 40 features, batch 64, crop 128, FP32, seed 0, `num_workers=8`, and the same three official YAML files used for Fold 2.
- Keep the original upstream checkout clean and the derived checkout limited to the already authorized seven deletions in `src/models/__init__.py`.
- Each raw state dictionary must load with `strict=True`; the only scientific action is `Trainer.test`.
- Training, validation, prediction, checkpoint selection, optimizer steps, resume, and scientific automatic retries are forbidden.
- Reuse the existing Fold 2 test-only run only after the new verifier proves it satisfies the campaign contract; do not rerun it.
- Launch folds sequentially on the single RTX 3090. Stop on the first child or verification failure and publish no partial aggregate.
- Keep ignored weights and run artifacts out of Git. Do not push or merge.

---

### Task 1: Frozen Twelve-Weight and Fold Contract

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/scripts/released_weight_contract.py`
- Create: `reproductions/wsts_res18_unet_t1/official_weights_manifest.json`
- Create: `tests/test_wsts_released_weight_contract.py`
- Modify: `reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py`
- Modify: `reproductions/wsts_res18_unet_t1/scripts/verify_released_weight.py`

**Interfaces:**
- Produces immutable `WeightSpec(fold_id: int, filename: str, hub_path: str, size: int, sha256: str, filename_ap: float, train_years: tuple[int, int], validation_year: int, test_year: int)`.
- Produces `OFFICIAL_FOLDS: tuple[tuple[int, int, int, int], ...]`, `parse_pinned_tree(items: Sequence[Mapping[str, object]]) -> tuple[WeightSpec, ...]`, `load_pinned_manifest(path: Path) -> tuple[WeightSpec, ...]`, `validate_local_weight(path: Path, spec: WeightSpec) -> None`, and `write_manifest_atomic(path: Path, specs: Sequence[WeightSpec]) -> None`.
- Existing filename aggregate helpers import the fold-neutral constants instead of maintaining duplicate Fold 2 lists.

- [ ] **Step 1: Write the failing contract tests**

  Add tests that require the exact official fold tuples, exactly one weight for
  every fold, filename/fold/AP agreement, positive size, 64-character lowercase
  SHA-256, exact frozen revision/path prefix, deterministic fold ordering, and
  rejection of missing, duplicate, additional, renamed, wrong-size, and
  wrong-hash entries.

  ```python
  def test_contract_freezes_official_twelve_folds() -> None:
      assert contract.OFFICIAL_FOLDS == (
          (2018, 2019, 2020, 2021),
          (2018, 2019, 2021, 2020),
          (2018, 2020, 2019, 2021),
          (2018, 2020, 2021, 2019),
          (2018, 2021, 2019, 2020),
          (2018, 2021, 2020, 2019),
          (2019, 2020, 2018, 2021),
          (2019, 2020, 2021, 2018),
          (2019, 2021, 2018, 2020),
          (2019, 2021, 2020, 2018),
          (2020, 2021, 2018, 2019),
          (2020, 2021, 2019, 2018),
      )

  def test_pinned_tree_fails_closed_on_extra_weight(tree_items) -> None:
      tree_items.append(dict(tree_items[0], path=contract.WEIGHT_PREFIX + "fold12_testAP0.500.pth"))
      with pytest.raises(ValueError, match="exactly twelve"):
          contract.parse_pinned_tree(tree_items)
  ```

- [ ] **Step 2: Run the RED tests**

  Run:

  ```powershell
  python -m pytest tests/test_wsts_released_weight_contract.py -q
  ```

  Expected: collection fails because `released_weight_contract.py` does not
  exist.

- [ ] **Step 3: Implement the minimal immutable contract**

  Define the frozen revision, repository, prefix, filenames, and fold tuples in
  the new module. Derive `train_years`, `validation_year`, `test_year`, and
  filename AP from the fold ID and filename, but accept size/SHA only from the
  pinned Hub tree. Serialize a schema-versioned JSON object whose weight rows
  are sorted by `fold_id`; atomic publication uses a same-directory temporary
  file plus `os.replace`.

  ```python
  @dataclass(frozen=True, slots=True)
  class WeightSpec:
      fold_id: int
      filename: str
      hub_path: str
      size: int
      sha256: str
      filename_ap: float
      train_years: tuple[int, int]
      validation_year: int
      test_year: int

  def spec_for_fold(specs: Sequence[WeightSpec], fold_id: int) -> WeightSpec:
      selected = [spec for spec in specs if spec.fold_id == fold_id]
      if len(selected) != 1:
          raise ValueError(f"fold {fold_id} weight is missing or ambiguous")
      return selected[0]
  ```

- [ ] **Step 4: Query the pinned Hub tree and freeze the manifest**

  Add a non-scientific CLI action that queries only revision
  `acf70a37394849f4ec8d108a51d6f4325a554d0a`, validates all twelve entries,
  and atomically writes the tracked manifest. Run:

  ```powershell
  D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe reproductions\wsts_res18_unet_t1\scripts\evaluate_released_weight.py --pin-manifest reproductions\wsts_res18_unet_t1\official_weights_manifest.json
  ```

  Review all twelve sizes and SHA-256 values against the Hub response, then run
  the contract tests again. This action must not import torch, instantiate a
  datamodule, or create a launch lock.

- [ ] **Step 5: Verify and commit Task 1**

  Run:

  ```powershell
  python -m pytest tests/test_wsts_released_weight_contract.py tests/test_wsts_released_weight.py -q
  git diff --check
  ```

  Commit only the contract, manifest, tests, and import deduplication:

  ```powershell
  git add reproductions/wsts_res18_unet_t1/scripts/released_weight_contract.py reproductions/wsts_res18_unet_t1/official_weights_manifest.json reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py reproductions/wsts_res18_unet_t1/scripts/verify_released_weight.py tests/test_wsts_released_weight_contract.py
  git commit -m "feat: freeze official twelve-weight manifest"
  ```

### Task 2: Fold-Neutral Test-Only Controller and Independent Verifier

**Files:**
- Modify: `reproductions/wsts_res18_unet_t1/scripts/official_weight_entrypoint.py`
- Modify: `reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py`
- Modify: `reproductions/wsts_res18_unet_t1/scripts/verify_released_weight.py`
- Modify: `tests/test_wsts_released_weight.py`
- Create: `tests/test_wsts_released_weight_folds.py`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

**Interfaces:**
- `build_weight_command(*, spec: WeightSpec, run_directory: Path, weight_path: Path, ...) -> list[str]` places exactly `--data.data_fold_id={spec.fold_id}` in the official command.
- `fetch_weight(spec: WeightSpec, target: Path | None = None) -> dict[str, object]` downloads one pinned file and verifies its exact size/SHA.
- Controller CLI supports `--fold-id 0..11` with exactly one action from `--fetch-only`, `--preflight-only`, `--launch`, and `--finalize-existing RUN`.
- `verify_run(run_directory: Path, weight_path: Path, spec: WeightSpec) -> dict[str, object]` reconstructs all scientific and provenance claims independently.

- [ ] **Step 1: Write RED tests for folds 0, 5, and 11**

  Require that each generated command differs only in fold ID, weight path, and
  run directory. Assert the official split years are captured in preflight and
  independently checked against the exact fold tuple. Assert global/run locks
  are fold-specific, the result records the matching filename AP/SHA, and
  cross-fold run/weight/lock aliases fail closed.

  ```python
  @pytest.mark.parametrize("fold_id", [0, 5, 11])
  def test_command_is_strict_test_only_for_every_fold(tmp_path: Path, specs) -> None:
      spec = contract.spec_for_fold(specs, fold_id)
      command = controller.build_weight_command(
          spec=spec, run_directory=tmp_path / f"fold{fold_id}", weight_path=tmp_path / spec.filename
      )
      assert f"--data.data_fold_id={fold_id}" in command
      assert "--do_train=false" in command
      assert "--do_test=false" in command
      assert not any("predict" in argument or "validate" in argument for argument in command)
  ```

- [ ] **Step 2: Run focused RED tests**

  Run:

  ```powershell
  python -m pytest tests/test_wsts_released_weight.py tests/test_wsts_released_weight_folds.py -q
  ```

  Expected: failures show the controller/verifier are still Fold-2 constants.

- [ ] **Step 3: Generalize the controller without changing scientific code**

  Replace Fold 2 constants with `WeightSpec` arguments through manifest
  selection, cache path, command, lock, run-directory naming, preflight,
  provenance, result parsing, and finalization. Keep the existing Fold 2 CLI
  behavior compatible by defaulting `--fold-id` to `2`, so its sealed artifact
  remains verifiable. Change only the entrypoint's misleading Fold-2 error text
  to `released weight must be a raw state_dict`; do not alter load or test calls.

  ```python
  parser.add_argument("--fold-id", type=int, choices=range(12), default=2)
  spec = spec_for_fold(load_pinned_manifest(MANIFEST_PATH), arguments.fold_id)
  lock_path = WEIGHT_ARTIFACTS_ROOT / f"fold{spec.fold_id}-weight-evaluation.lock.json"
  ```

- [ ] **Step 4: Generalize the independent verifier and preserve Fold 2 compatibility**

  Rebuild the expected command, lock path, fold years, weight size/SHA, filename
  AP, and raw-manifest weight entry exclusively from the tracked manifest and
  `fold_id`. Do not trust controller result fields. The old Fold 2 run may carry
  retrospective parser/provenance markers; verify those only when present and
  never require them for new clean runs. Require current raw seals to remain
  byte-identical before and after verification.

- [ ] **Step 5: Run full verification and commit Task 2**

  Run focused tests, the current Fold 2 independent verifier, then the full
  suite:

  ```powershell
  python -m pytest tests/test_wsts_released_weight.py tests/test_wsts_released_weight_folds.py -q
  D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe reproductions\wsts_res18_unet_t1\scripts\verify_released_weight.py --fold-id 2 --run-directory artifacts\reproductions\wsts-res18-t1-official-weight\fold2-weight-20260810T133336Z-2e071197 --weight-path reproductions\wsts_res18_unet_t1\.local\released-weights\fold2_testAP0.571.pth --output artifacts\reproductions\wsts-res18-t1-official-weight\fold2-weight-20260810T133336Z-2e071197\independent-verification.json
  python -m pytest -q
  git diff --check
  ```

  Commit:

  ```powershell
  git add reproductions/wsts_res18_unet_t1/scripts/official_weight_entrypoint.py reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py reproductions/wsts_res18_unet_t1/scripts/verify_released_weight.py reproductions/wsts_res18_unet_t1/README.md tests/test_wsts_released_weight.py tests/test_wsts_released_weight_folds.py
  git commit -m "feat: generalize official weight evaluation by fold"
  ```

### Task 3: Sequential Campaign Controller and Independent Aggregator

**Files:**
- Create: `reproductions/wsts_res18_unet_t1/scripts/run_weight_campaign.py`
- Create: `reproductions/wsts_res18_unet_t1/scripts/verify_weight_campaign.py`
- Create: `tests/test_wsts_weight_campaign.py`
- Create: `tests/test_wsts_weight_campaign_verifier.py`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`

**Interfaces:**
- `qualify_existing_fold2(run_directory: Path, spec: WeightSpec) -> dict[str, object]` accepts the sealed Fold 2 run only after the generic independent verifier passes.
- `build_campaign_schedule(specs: Sequence[WeightSpec], adopted: Mapping[int, Path]) -> tuple[CampaignFold, ...]` returns folds `0..11` once, with Fold 2 marked `adopt` and eleven folds marked `launch`.
- `run_campaign(campaign_directory: Path) -> dict[str, object]` launches one fold controller at a time, invokes its independent verifier, writes atomic state, and stops on the first non-pass outcome.
- `verify_campaign(campaign_directory: Path) -> dict[str, object]` independently reconstructs all twelve rows from raw fold evidence and writes `official-weight-12fold-results.csv`, `official-weight-12fold-summary.json`, and `independent-verification.json` atomically.

- [ ] **Step 1: Write RED campaign controller tests**

  Test deterministic `0..11` ordering, exact Fold 2 adoption, campaign/global
  and per-fold lock refusal, no duplicate launch, stop-on-first-failure, no next
  fold after verifier failure, and `finalize-existing` without process launch.

  ```python
  def test_campaign_stops_before_next_fold_when_verifier_fails(fake_launcher, specs) -> None:
      fake_launcher.fail_verifier_for = 4
      with pytest.raises(ValueError, match="fold 4 independent verification failed"):
          campaign.run_campaign(fake_launcher.root)
      assert fake_launcher.scientific_folds == [0, 1, 3, 4]
      assert 5 not in fake_launcher.scientific_folds
  ```

- [ ] **Step 2: Write RED independent aggregation tests**

  Use twelve synthetic raw Lightning tables and require six finite `[0,1]`
  metrics per fold, exact official year mappings, unique full fold coverage,
  raw-seal stability, AP mean via `statistics.fmean`, population standard
  deviation via `statistics.pstdev`, min/max fold IDs, runtime totals/median,
  filename deltas, and paper/filename aggregate deltas. Reject controller-only
  summaries, missing or duplicate folds, tampered raw logs, NaN/Inf/out-of-range
  metrics, and sample standard deviation.

  ```python
  def test_aggregate_uses_population_standard_deviation(rows) -> None:
      result = verifier.summarize_verified_rows(rows)
      values = [row["test_AP"] for row in rows]
      assert result["ap_mean"] == statistics.fmean(values)
      assert result["ap_population_std"] == statistics.pstdev(values)
      assert result["fold_count"] == 12
  ```

- [ ] **Step 3: Run RED tests**

  Run:

  ```powershell
  python -m pytest tests/test_wsts_weight_campaign.py tests/test_wsts_weight_campaign_verifier.py -q
  ```

  Expected: collection fails because both campaign modules are absent.

- [ ] **Step 4: Implement the minimal sequential controller**

  The campaign controller first qualifies Fold 2, verifies/downloads all eleven
  missing weights, records one immutable schedule, and atomically acquires the
  campaign lock. For each launch entry it calls the single-fold controller once,
  requires terminal exit, calls the independent fold verifier once, records the
  run directory and verifier SHA, then advances. Any exception writes a failure
  marker and exits without invoking the next fold.

  ```python
  for item in schedule:
      if item.mode == "adopt":
          result = qualify_existing_fold2(item.run_directory, item.spec)
      else:
          result = launch_one_fold_once(item.spec)
      require_independent_pass(result)
      write_campaign_state_atomic(campaign_directory, item.spec.fold_id, result)
  ```

- [ ] **Step 5: Implement the independent campaign verifier**

  Ignore campaign-authored metric summaries. Resolve each recorded run under
  the fixed artifact root, invoke the generic independent fold verifier, parse
  the sealed stdout/stderr again, compare all six metrics exactly, and then
  aggregate. Write CSV field order explicitly and JSON with sorted keys. Seal
  the twelve fold-verifier files and raw evidence manifests; assert they remain
  unchanged during campaign verification.

- [ ] **Step 6: Verify and commit Task 3**

  Run:

  ```powershell
  python -m pytest tests/test_wsts_weight_campaign.py tests/test_wsts_weight_campaign_verifier.py -q
  python -m pytest -q
  git diff --check
  ```

  Commit:

  ```powershell
  git add reproductions/wsts_res18_unet_t1/scripts/run_weight_campaign.py reproductions/wsts_res18_unet_t1/scripts/verify_weight_campaign.py reproductions/wsts_res18_unet_t1/README.md tests/test_wsts_weight_campaign.py tests/test_wsts_weight_campaign_verifier.py
  git commit -m "feat: add twelve-fold weight campaign"
  ```

### Task 4: Execute the Official-Weight Campaign to Completion

**Files:**
- Produces ignored artifacts under `artifacts/reproductions/wsts-res18-t1-official-weight/`.
- Produces ignored campaign artifacts under `artifacts/reproductions/wsts-res18-t1-official-weight-12fold/`.
- Modify after terminal completion: `.superpowers/sdd/2026-08-11-res18-unet-t1-official-weights-12fold/task-4-report.md`

**Interfaces:**
- Consumes the committed manifest and Tasks 1-3 CLIs.
- Produces one qualifying run per fold, `official-weight-12fold-results.csv`, `official-weight-12fold-summary.json`, and an independent PASS artifact.

- [ ] **Step 1: Run final non-scientific gates**

  Require a clean tracked worktree, full test suite PASS, exact Python/library
  environment, 607 files/24,242,259,023 bytes, clean official checkout at the
  pinned commit, exact seven-deletion derived patch, current Fold 2 independent
  verifier PASS, all twelve local weight size/SHA checks PASS, no campaign lock,
  no conflicting scientific Python process, and adequate GPU/disk memory.

- [ ] **Step 2: Publish the exact schedule and launch once**

  Record that Fold 2 is adopted and the launch set is exactly
  `0,1,3,4,5,6,7,8,9,10,11`. Start one hidden campaign controller process:

  ```powershell
  $process = Start-Process -FilePath 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -ArgumentList @('reproductions\wsts_res18_unet_t1\scripts\run_weight_campaign.py','--launch') -WorkingDirectory 'D:\WildFire Project\wildfire-research-plan\.worktrees\rule-baseline-evaluation' -WindowStyle Hidden -PassThru
  $process.Id
  ```

  Do not start a replacement process.

- [ ] **Step 3: Monitor the same lineage to terminal state**

  At least once per fold, record controller/scientific PID, fold ID, official
  test progress, elapsed time, fatal-error count, GPU utilization/memory, and
  whether the next fold has started only after the prior verifier passed. Do not
  interrupt a healthy process. If a fold fails, preserve artifacts and stop.

- [ ] **Step 4: Finalize and independently verify all twelve folds**

  Require campaign exit zero, eleven newly launched fold directories, the
  adopted Fold 2 directory, twelve independent fold PASS artifacts, unchanged
  raw evidence, and no train/validation/predict evidence. Run the campaign
  verifier separately and record output hashes. Confirm no campaign or
  scientific process remains.

- [ ] **Step 5: Record Task 4 evidence**

  Write exact commands, lock/run paths, manifest SHA, per-fold run/verifier
  paths, scientific launch count, failures or parser-only recoveries, aggregate
  artifact hashes, test counts, source/upstream status, and campaign commit to
  the ignored task report. Do not claim success before the independent campaign
  verifier passes.

### Task 5: Publish Results, Website Update, and Completion Audit

**Files:**
- Modify: `docs/experiments/res18_unet_t1_reproduction.md`
- Modify: `reproductions/wsts_res18_unet_t1/README.md`
- Modify: `index.html`
- Create: `tests/test_wsts_weight_campaign_publication.py`
- Modify: `.superpowers/progress/2026-08-11-res18-unet-t1-official-weights-12fold.md`

**Interfaces:**
- Consumes only the independently passing campaign summary and twelve-row CSV.
- Produces the PI-facing scientific interpretation, bilingual website summary, exact-value publication tests, and final tracked commit.

- [ ] **Step 1: Write publication RED tests**

  Assert that docs and both website languages contain the exact recomputed AP
  mean/population standard deviation, all twelve fold AP values, min/max fold,
  paper aggregate, filename aggregate, and their explicit provenance labels.
  Assert the text says test-only released weights, not twelve new trainings, and
  keeps the focal-alpha issue out of this baseline claim.

- [ ] **Step 2: Write the evidence-bounded report and website content**

  Include a fold table with six metrics and filename deltas, aggregate/runtime
  tables, comparison to `0.460 +/- 0.084` and
  `0.45291666666666663 +/- 0.08827179460179917`, observed failure modes, and a
  restrained scientific conclusion. State that agreement supports executable
  released-weight reproducibility but does not prove provenance identity with
  the paper table. Record the discovered official focal-alpha behavior as a
  separate future ablation, not a correction applied here.

- [ ] **Step 3: Run the completion audit**

  Run both the generic Fold 2 verifier and campaign verifier, focused publication
  tests, the full pytest suite, `git diff --check`, secret/path scans, tracked
  worktree status, original/derived upstream status, source inventory comparison,
  all twelve weight hashes, aggregate artifact hashes, and process checks. Match
  every numbered completion criterion in the design spec to direct evidence.

- [ ] **Step 4: Commit without push or merge**

  Commit the verified documentation, website, tests, and progress ledger:

  ```powershell
  git add docs/experiments/res18_unet_t1_reproduction.md reproductions/wsts_res18_unet_t1/README.md index.html tests .superpowers/progress/2026-08-11-res18-unet-t1-official-weights-12fold.md
  git commit -m "docs: publish twelve-fold official weight results"
  ```

- [ ] **Step 5: Request final code/scientific review**

  Give the reviewer the design, plan, complete branch diff, tracked manifest,
  Task 4 report, campaign summary/CSV, all twelve independent verifier outputs,
  and final test/status evidence. Address only verified Critical/Important
  findings, rerun the completion audit after fixes, and do not rerun scientific
  children for observer/parser/documentation defects.

# Quantitative Reliability Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce corrected-index baselines and retain at least three follow-on directions only after matched quantitative evidence supports them.

**Architecture:** A small registry defines B0--B3 and installs the already tested first-match resolver before any upstream dataset is constructed. One generic train/evaluate runner emits immutable records for the four baselines; follow-on methods reuse the same evaluation path so each experiment changes one factor and names its matched control.

**Tech Stack:** Python 3.10, PyTorch, LightningCLI, segmentation-models-pytorch, NumPy, pytest, Bash, Slurm on Nibi.

**Spec:** `docs/superpowers/specs/2026-09-03-quantitative-reliability-baselines-design.md`

## Global Constraints

- Train on 2016--2020, select on 2021, and evaluate frozen promoted settings separately on 2022 and 2023.
- Run seed 0 and 3,000 steps for screens; do not launch 10,000 steps or extra seeds without quantitative promotion evidence.
- Never run training or full evaluation on a login node.
- Inspect Nibi demand immediately before every submission wave and request the smallest fast-starting compatible GPU.
- Keep tests focused on changed behavior; do not run the full pytest suite.
- A direction is not reliable without the quantitative evidence level defined in the spec.

---

### Task 1: Corrected baseline registry and resolver installation

**Files:**
- Create: `reproductions/wsts_fast_track/corrected_baselines.py`
- Create: `tests/test_wsts_fast_track_corrected_baselines.py`

**Interfaces:**
- Consumes: `environment_dro.resolve_dataset_index`, `prototype.install_training_fire_dropout`, and `prototype.install_training_fire_and_block_dropout`.
- Produces: `CorrectedBaselineSpec`, `CORRECTED_BASELINES`, `corrected_baseline_spec(baseline_id)`, and `install_corrected_baseline(upstream_root, baseline_id)`.

- [x] **Step 1: Write the failing registry/resolver test**

```python
def test_corrected_baselines_are_from_scratch_single_variable_controls():
    assert corrected_baseline_spec("B0").training_policy == "clean"
    assert corrected_baseline_spec("B1").experiment_id == "C02"
    assert corrected_baseline_spec("B2").training_policy == "fire"
    assert corrected_baseline_spec("B3").training_policy == "fire-block"
    with pytest.raises(ValueError, match="unknown corrected baseline"):
        corrected_baseline_spec("P00")
```

The same test module uses a fake upstream dataset module and asserts that
`install_corrected_baseline` replaces
`find_image_index_from_dataset_index` with the first-match resolver before
installing the selected corruption policy.

- [x] **Step 2: Verify RED**

Run:
`/project/6085198/kulbear/wildfire/envs/wildfire-audit/bin/python -m pytest -q tests/test_wsts_fast_track_corrected_baselines.py`

Expected: collection fails because `corrected_baselines` does not exist.

- [x] **Step 3: Implement the minimal registry**

```python
@dataclass(frozen=True)
class CorrectedBaselineSpec:
    baseline_id: str
    experiment_id: Literal["C00", "C02"]
    training_policy: Literal["clean", "fire", "fire-block"]
    seed: int = 0
    max_steps: int = 3_000

CORRECTED_BASELINES = {
    "B0": CorrectedBaselineSpec("B0", "C00", "clean"),
    "B1": CorrectedBaselineSpec("B1", "C02", "clean"),
    "B2": CorrectedBaselineSpec("B2", "C00", "fire"),
    "B3": CorrectedBaselineSpec("B3", "C00", "fire-block"),
}
```

`install_corrected_baseline` imports the upstream dataset class, assigns
`resolve_dataset_index`, and installs only the selected training corruption.

- [x] **Step 4: Verify GREEN**

Run the Task 1 test command and expect all tests to pass.

- [x] **Step 5: Commit**

```bash
git add reproductions/wsts_fast_track/corrected_baselines.py tests/test_wsts_fast_track_corrected_baselines.py
git commit -m "feat: define corrected reliability baselines"
```

### Task 2: Generic corrected-index training and completion records

**Files:**
- Create: `reproductions/wsts_fast_track/train_corrected_baseline.py`
- Create: `reproductions/wsts_fast_track/complete_corrected_baseline.py`
- Create: `tests/test_wsts_fast_track_corrected_training.py`

**Interfaces:**
- Consumes: `corrected_baseline_spec`, `install_corrected_baseline`, `entrypoint._install_runtime_contract`, `contract.upstream_arguments`, and `completion.last_metric`.
- Produces: CLI `python -m reproductions.wsts_fast_track.train_corrected_baseline --baseline-id B0 ...` and immutable `completed.json` with `corrected_index: true` and `initialization: from_scratch`.

- [x] **Step 1: Write failing completion tests**

Create a synthetic 3,000-step training log and fake checkpoint loader. Assert
that `finalize_corrected_baseline` accepts the exact marker, records B0's
identity and policy, and rejects a missing completion marker or pre-existing
output.

- [x] **Step 2: Verify RED**

Run:
`/project/6085198/kulbear/wildfire/envs/wildfire-audit/bin/python -m pytest -q tests/test_wsts_fast_track_corrected_training.py`

Expected: collection fails because the two modules do not exist.

- [x] **Step 3: Implement training entrypoint and completion writer**

The entrypoint validates the inventory/statistics, installs the upstream
runtime contract, installs the corrected baseline before dataset creation,
uses `max_steps=3_000` and `seed=0`, prints a JSON configuration marker, and
runs the pinned `train.py`. The completion writer requires one checkpoint, a
positive CUDA peak marker, finite validation AP/F1/loss, and writes the exact
training policy plus the corrected-index/from-scratch flags.

- [x] **Step 4: Verify GREEN**

Run the Task 2 test command and expect all tests to pass.

- [x] **Step 5: Commit**

```bash
git add reproductions/wsts_fast_track/train_corrected_baseline.py reproductions/wsts_fast_track/complete_corrected_baseline.py tests/test_wsts_fast_track_corrected_training.py
git commit -m "feat: train corrected baselines from scratch"
```

### Task 3: Matched rapid evaluation and Nibi runner

**Files:**
- Create: `reproductions/wsts_fast_track/evaluate_corrected_baseline.py`
- Create: `reproductions/wsts_fast_track/run_corrected_baseline_on_nibi.sh`
- Create: `tests/test_wsts_fast_track_corrected_runner.py`

**Interfaces:**
- Consumes: corrected completion record, `evaluation.build_controlled_dataset`, and `evaluate_missingness.evaluate_batches`.
- Produces: one Slurm job per baseline containing training plus frozen 2021 M00/M01/M06/M07 evaluation and `results-2021/summary.json`.

- [x] **Step 1: Write failing record-boundary and shell tests**

Assert that evaluation accepts only a passing B0--B3 record with
`max_steps=3000`, `seed=0`, `corrected_index=true`,
`initialization=from_scratch`, and 2021 without held-out authorization. Run
`bash -n` on the runner and assert it contains no `sbatch`, archives committed
HEAD, accepts exactly B0--B3, and invokes train, completion, and evaluation.

- [x] **Step 2: Verify RED**

Run:
`/project/6085198/kulbear/wildfire/envs/wildfire-audit/bin/python -m pytest -q tests/test_wsts_fast_track_corrected_runner.py`

Expected: collection fails because the evaluator and runner do not exist.

- [x] **Step 3: Implement evaluator and shell runner**

The evaluator loads one checkpoint, evaluates exactly the four declared
scenarios on 2021, and writes AP/F1/IoU/precision/recall/loss/Brier plus AP
deltas from M00. The shell runner records git/upstream state, environment,
command, GPU inventory, resource metadata, and refuses existing run roots.

- [x] **Step 4: Verify GREEN and focused regression**

Run:
`/project/6085198/kulbear/wildfire/envs/wildfire-audit/bin/python -m pytest -q tests/test_wsts_fast_track_corrected_baselines.py tests/test_wsts_fast_track_corrected_training.py tests/test_wsts_fast_track_corrected_runner.py tests/test_wsts_fast_track_prototype.py tests/test_wsts_fast_track_environment_dro.py`

Expected: all focused tests pass. Also run
`bash -n reproductions/wsts_fast_track/run_corrected_baseline_on_nibi.sh`.

- [x] **Step 5: Commit**

```bash
git add reproductions/wsts_fast_track/evaluate_corrected_baseline.py reproductions/wsts_fast_track/run_corrected_baseline_on_nibi.sh tests/test_wsts_fast_track_corrected_runner.py
git commit -m "feat: evaluate corrected baseline wave"
```

### Task 4: Submit and classify B0--B3

**Files:**
- Modify: `docs/experiments/quantitative_reliability_ledger.md`
- Modify: `docs/research-roadmap.md`

**Interfaces:**
- Consumes: committed Task 3 runner and live Nibi scheduler state.
- Produces: four terminal records and a matched 2021 baseline table.

- [x] **Step 1: Inspect scheduler demand**

Run `squeue -h -t PD -o '%b' | sort | uniq -c | sort -nr`, `sinfo -p gpubackfill -N -o '%N %G %t %C'`, and `sprio -u kulbear`. Compare 10GB/20GB/40GB MIG, A100, and full-H100 demand.

- [x] **Step 2: Submit the smallest fast-starting compatible requests**

Use one GPU per job, four CPUs, 16--32GB host RAM, and a one-hour wall time.
Submit B0--B3 only; do not run any training on the login node. Record job IDs
and exact resource requests in the ledger.

- [ ] **Step 3: Monitor terminal state and inspect records**

Poll no more frequently than every 30 minutes unless Slurm reports a terminal
failure. On failure, inspect the log, apply the smallest root-cause fix under a
new test, commit, and resubmit only the affected job.

- [ ] **Step 4: Classify the corrected baselines**

Create a table with M00/M01/M06/M07 AP and deltas against B0. Explicitly state
whether B1, B2, and B3 are baseline evidence, tuning evidence, or a rejected
configuration; do not call them new method contributions.

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/quantitative_reliability_ledger.md docs/research-roadmap.md
git commit -m "docs: record corrected baseline evidence"
```

### Task 5: D1 clean-corrupt consistency with matched continuation

**Files:**
- Create: `reproductions/wsts_fast_track/predictive_consistency.py`
- Create: `reproductions/wsts_fast_track/train_predictive_consistency.py`
- Create: `tests/test_wsts_fast_track_predictive_consistency.py`
- Modify: `reproductions/wsts_fast_track/run_corrected_baseline_on_nibi.sh`

**Interfaces:**
- Consumes: frozen B3 checkpoint and paired clean/corrupt samples.
- Produces: `bernoulli_kl_from_logits(clean_logits, corrupt_logits)` and two matched 3,000-step continuations, D1-KL and D1-ERM.

- [x] **Step 1: Write a failing loss test**

Use literal logits and assert KL is zero for identical predictions, positive
for different predictions, finite for logits `[-20, 20]`, and sends gradients
only through corrupt logits when clean logits are detached.

- [x] **Step 2: Verify RED**

Run the D1 test module and expect import failure.

- [x] **Step 3: Implement the minimal paired-view objective**

Train D1-KL and D1-ERM from the same B3 checkpoint, with identical sample
order, corruptions, optimizer, learning rate, and 3,000 additional steps. The
only difference is `lambda_consistency=0.1` versus `0.0`.

- [ ] **Step 4: Verify and submit after a fresh queue check**

Run only the D1 and corrected-runner tests, commit, inspect Nibi demand, and
submit the two matched jobs. Evaluate M00/M01/M06/M07 on 2021.

- [ ] **Step 5: Freeze or reject**

If D1-KL does not meet the screen-positive rule against D1-ERM, reject it. If
it passes, freeze lambda 0.1 and evaluate both frozen checkpoints on 2022 and
2023 once.

### Task 6: D2 reliability-normalized first encoder convolution

**Files:**
- Create: `reproductions/wsts_fast_track/reliability_normalized_conv.py`
- Create: `reproductions/wsts_fast_track/train_reliability_normalized.py`
- Create: `tests/test_wsts_fast_track_reliability_normalized_conv.py`

**Interfaces:**
- Consumes: B3 policy and spatial validity generated before NaN replacement.
- Produces: `ReliabilityNormalizedConv2d` and one D2 checkpoint matched to a standard-convolution control.

- [x] **Step 1: Write failing operator tests**

Assert exact equality with `torch.nn.functional.conv2d` for an all-valid mask,
invariance to placeholder values under an invalid mask, correct output shape,
and finite gradients for weights and valid inputs.

- [x] **Step 2: Verify RED**

Run the D2 test module and expect import failure.

- [x] **Step 3: Implement the one-layer intervention**

Compute masked convolution and divide by the valid fraction in each receptive
field with a clamped denominator. Preserve the original bias and initialize
from the matched standard convolution. Add no deeper mask propagation in this
first prototype.

- [ ] **Step 4: Verify and submit after a fresh queue check**

Run only D2 and corrected-baseline tests, commit, then submit the D2 and matched
standard-convolution jobs with identical B3 corruption and 3,000-step budgets.

- [ ] **Step 5: Freeze or reject**

Use mean M06/M07 AP as primary. A passing 2021 configuration is frozen and
evaluated on 2022 and 2023 once; otherwise record rejection.

### Task 7: D3 reliability-conditioned temporal fusion

**Files:**
- Create: `reproductions/wsts_fast_track/reliability_temporal_fusion.py`
- Create: `reproductions/wsts_fast_track/train_reliability_temporal.py`
- Create: `tests/test_wsts_fast_track_reliability_temporal_fusion.py`

**Interfaces:**
- Consumes: B1 C02 input sequence and time-resolved validity masks.
- Produces: masked temporal-softmax helper and a C02-compatible model class.

- [x] **Step 1: Write failing masked-softmax tests**

Assert all-valid equivalence to ordinary softmax, exactly zero mass for invalid
time steps, unit mass over remaining steps, and finite behavior when only one
time step is valid.

- [x] **Step 2: Verify RED**

Run the D3 test module and expect import failure.

- [ ] **Step 3: Implement validity-conditioned temporal aggregation**

Reuse the C02 encoder/decoder and add the validity bias immediately before
temporal attention normalization. Do not add reconstruction, an auxiliary
network, or additional temporal history.

- [ ] **Step 4: Verify and submit only if B1 remains eligible**

If B1 is not competitive with B0 or shows no useful temporal robustness, record
D3 as gated and do not spend GPU time. Otherwise run focused tests, commit,
check the scheduler, and submit D3 plus the matched C02 control.

- [ ] **Step 5: Freeze or reject**

Use the predeclared primary AP and the same 2021-to-held-out procedure as D1/D2.

### Task 8: Quantitative contribution audit

**Files:**
- Modify: `docs/experiments/quantitative_reliability_ledger.md`
- Modify: `docs/research-roadmap.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all terminal JSON records from Tasks 4--7.
- Produces: final baseline comparison and a list containing only directions with level-3 or level-4 quantitative support.

- [ ] **Step 1: Build the evidence table**

For each candidate, report matched baseline, parameter delta, training budget,
2021/2022/2023 primary AP, per-year delta, three-year mean delta, clean AP
delta, and evidence level.

- [ ] **Step 2: Enforce contribution accounting**

Classify each retained item as reproduction/baseline, tuning/training,
incremental module, or method. Count at most one tuning/training contribution.

- [ ] **Step 3: Continue or stop honestly**

If fewer than three directions reach level 3, report the shortfall and design
the next smallest one-variable candidate instead of relabeling unsupported
hypotheses. If three reach level 3, identify which ones merit multi-seed 10K
confirmation.

- [ ] **Step 4: Run fresh verification**

Run focused tests for every changed method, `bash -n` for every submitted
runner, parse every cited JSON record, and inspect `git diff --check` and
`git status --short`.

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/quantitative_reliability_ledger.md docs/research-roadmap.md README.md
git commit -m "docs: report quantitative reliability directions"
```

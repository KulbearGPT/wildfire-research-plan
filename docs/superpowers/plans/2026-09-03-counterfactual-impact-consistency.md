# Counterfactual Impact-Weighted Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and screen CIWC against the completed D1-ERM and D1-KL controls, advancing to 2022/2023 only if the frozen 2021 `+0.020` gate passes.

**Architecture:** Reuse D1's exactly aligned clean/corrupt dataset and add one stop-gradient, per-sample impact-weighted Bernoulli KL primitive. A dedicated fixed-contract trainer, evaluator, and Nibi runner keep the candidate metadata honest while reusing the B3 initialization and existing controlled scenarios.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, Bash, Slurm on Nibi.

**Spec:** `docs/superpowers/specs/2026-09-03-counterfactual-impact-consistency-design.md`

## Global Constraints

- Keep C00 ResNet-18 U-Net, corrected 2016--2020 data, B3 initialization, seed 0, AdamW `1e-3`, and 3,000 optimizer steps fixed.
- Use `lambda_ciwc=0.1` without a sweep.
- Train and evaluate only inside Slurm allocations; the login node may run only focused CPU unit tests and result parsing.
- The 2021 primary delta versus D1-ERM must be at least `+0.020`, must exceed D1-KL, and must satisfy the per-scenario and M00 guardrails before any 2022/2023 access.
- Do not run the broad repository test suite, multi-seed experiments, or longer-budget confirmation during screening.

---

### Task 1: CIWC loss and fixed training objective

**Files:**
- Create: `reproductions/wsts_fast_track/counterfactual_impact_consistency.py`
- Create: `reproductions/wsts_fast_track/train_counterfactual_impact_consistency.py`
- Create: `tests/test_wsts_fast_track_counterfactual_impact_consistency.py`

**Interfaces:**
- Consumes: `CleanCorruptPairDataset` and D1's B3 checkpoint/data contract.
- Produces: `counterfactual_impact_kl_from_logits(clean_logits, corrupt_logits) -> torch.Tensor` and `ciwc_training_objective(model, clean_logits, corrupt_logits, target) -> tuple[torch.Tensor, dict[str, float]]`.

- [ ] **Step 1: Write focused failing loss tests**

```python
def test_ciwc_focuses_disagreement_and_stops_clean_gradient():
    clean = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    corrupt = torch.tensor([[[-4.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    loss = counterfactual_impact_kl_from_logits(clean, corrupt)
    assert float(loss.detach()) > float(bernoulli_kl_from_logits(clean, corrupt))
    loss.backward()
    assert clean.grad is None
    assert torch.count_nonzero(corrupt.grad) == 1


def test_ciwc_zero_impact_is_exactly_zero():
    logits = torch.tensor([[[0.0, 1.0]]])
    assert float(counterfactual_impact_kl_from_logits(logits, logits)) == 0.0
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_wsts_fast_track_counterfactual_impact_consistency.py`

Expected: collection fails because `counterfactual_impact_consistency` does not exist.

- [ ] **Step 3: Implement the minimal impact-weighted KL**

```python
def counterfactual_impact_kl_from_logits(clean_logits, corrupt_logits):
    if clean_logits.shape != corrupt_logits.shape or clean_logits.numel() == 0:
        raise ValueError("clean and corrupt logits must have the same nonempty shape")
    teacher = clean_logits.detach()
    student = corrupt_logits
    teacher_probability = torch.sigmoid(teacher)
    impact = (teacher_probability - torch.sigmoid(student.detach())).abs()
    reduce_dims = tuple(range(1, impact.ndim))
    mean_impact = impact.mean(dim=reduce_dims, keepdim=True)
    weight = torch.where(
        mean_impact > 0.0,
        impact / mean_impact.clamp_min(1e-6),
        torch.zeros_like(impact),
    )
    per_pixel = teacher_probability * (
        F.logsigmoid(teacher) - F.logsigmoid(student)
    ) + (1.0 - teacher_probability) * (
        F.logsigmoid(-teacher) - F.logsigmoid(-student)
    )
    return (weight * per_pixel).mean()
```

Validate rank-three or greater logits so normalization is per sample over all non-batch dimensions. Implement `ciwc_training_objective` as the same two supervised terms as D1 plus exactly `0.1 * CIWC`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_wsts_fast_track_counterfactual_impact_consistency.py tests/test_wsts_fast_track_predictive_consistency.py`

Expected: all tests pass.

- [ ] **Step 5: Add the fixed 3,000-step trainer**

Copy only D1's runtime setup and loop into `train_counterfactual_impact_consistency.py`. Remove the consistency CLI choice, call `ciwc_training_objective`, and save this exact metadata:

```python
payload.update({
    "candidate_id": "D5-CIWC",
    "matched_pair": "D5",
    "base_control": "D1-ERM",
    "closest_ablation": "D1-KL",
    "lambda_ciwc": 0.1,
    "impact_weighting": "per-sample-absolute-probability-change",
})
```

- [ ] **Step 6: Commit the loss and trainer**

```bash
git add reproductions/wsts_fast_track/counterfactual_impact_consistency.py \
  reproductions/wsts_fast_track/train_counterfactual_impact_consistency.py \
  tests/test_wsts_fast_track_counterfactual_impact_consistency.py
git commit -m "feat: add counterfactual impact consistency"
```

### Task 2: Candidate evaluation and Nibi launch contract

**Files:**
- Create: `reproductions/wsts_fast_track/evaluate_counterfactual_impact_consistency.py`
- Create: `reproductions/wsts_fast_track/run_counterfactual_impact_consistency_on_nibi.sh`
- Modify: `reproductions/wsts_fast_track/run_reliability_evaluation_on_nibi.sh`
- Modify: `tests/test_wsts_fast_track_counterfactual_impact_consistency.py`

**Interfaces:**
- Consumes: a `D5-CIWC` checkpoint and the existing controlled M00/M01/M06/M07 datasets.
- Produces: one 2021 summary compatible with `quantitative_evidence.py` and a `ciwc` held-out runner kind.

- [ ] **Step 1: Write failing checkpoint and runner tests**

```python
def test_ciwc_checkpoint_contract():
    payload = {
        "schema_version": 1, "status": "pass", "candidate_id": "D5-CIWC",
        "matched_pair": "D5", "experiment": "C00", "steps": 3000,
        "seed": 0, "lambda_ciwc": 0.1, "impact_weighting":
        "per-sample-absolute-probability-change", "hyper_parameters": {},
        "state_dict": {"weight": torch.tensor(1.0)},
    }
    assert validate_ciwc_checkpoint(payload) == "D5-CIWC"


def test_ciwc_runner_is_allocation_only():
    text = RUNNER.read_text()
    assert "sbatch" not in text
    assert "train_counterfactual_impact_consistency" in text
    assert "evaluate_counterfactual_impact_consistency" in text
```

- [ ] **Step 2: Run the new test module and verify RED**

Run: `pytest -q tests/test_wsts_fast_track_counterfactual_impact_consistency.py`

Expected: import or path assertions fail because evaluator and runner are absent.

- [ ] **Step 3: Implement the evaluator and runners**

Implement `validate_ciwc_checkpoint` with exact schema, candidate, pair,
experiment, steps, seed, weight, weighting-name, hyperparameter, and state-dict
checks. Reuse the D1 evaluator flow for the four scenarios. The training runner
must archive committed source, train once, evaluate 2021, and never call
`sbatch`. Add `ciwc` to the generic held-out runner and dispatch it to the new
evaluator.

- [ ] **Step 4: Verify focused Python and Bash checks**

Run:

```bash
pytest -q tests/test_wsts_fast_track_counterfactual_impact_consistency.py \
  tests/test_wsts_fast_track_predictive_consistency.py
bash -n reproductions/wsts_fast_track/run_counterfactual_impact_consistency_on_nibi.sh
bash -n reproductions/wsts_fast_track/run_reliability_evaluation_on_nibi.sh
```

Expected: tests and both shell syntax checks pass.

- [ ] **Step 5: Commit the runnable candidate**

```bash
git add reproductions/wsts_fast_track/evaluate_counterfactual_impact_consistency.py \
  reproductions/wsts_fast_track/run_counterfactual_impact_consistency_on_nibi.sh \
  reproductions/wsts_fast_track/run_reliability_evaluation_on_nibi.sh \
  tests/test_wsts_fast_track_counterfactual_impact_consistency.py
git commit -m "feat: run CIWC reliability screen"
```

### Task 3: Nibi screen, frozen decision, and evidence record

**Files:**
- Modify: `docs/experiments/quantitative_reliability_ledger.md`
- Modify: `docs/research-roadmap.md`
- Modify: `README.md` only if CIWC reaches the final three-year gate.

**Interfaces:**
- Consumes: completed D1-ERM/D1-KL summaries and CIWC summaries.
- Produces: a frozen pass/fail decision with absolute AP values and job IDs.

- [ ] **Step 1: Verify source and inspect current Nibi demand**

Run the focused checks from Task 2, `git status --short`, `squeue -u "$USER"`,
and `sinfo` for available GPU/MIG classes. Select the smallest compatible GPU
class with the shortest apparent queue; request one GPU, 8 CPUs, 32 GB, and one
hour or less.

- [ ] **Step 2: Submit exactly one 2021 candidate job**

Run `sbatch` from the repository worktree with the CIWC runner and the exact B3
completion record used by D1. Record the returned job ID immediately in the
ledger. No workload is executed directly on the login node.

- [ ] **Step 3: Inspect the terminal job state and apply the frozen gate**

After the job becomes terminal, inspect `sacct`, training log tail, checkpoint
metadata, and `results-2021/summary.json`. Calculate D1-ERM and D1-KL
comparisons from their existing summaries. Record every gate condition,
including a rejection.

- [ ] **Step 4: Submit held-out evaluation only after a passing screen**

If and only if all 2021 conditions pass, submit one compact Slurm job that
evaluates the frozen checkpoint on both 2022 and 2023 using the `ciwc` evaluator
with `--heldout-authorized`. Otherwise stop CIWC and begin the next approved
candidate design without reading test-year outcomes.

- [ ] **Step 5: Record the final three-year decision and commit**

```bash
git add docs/experiments/quantitative_reliability_ledger.md \
  docs/research-roadmap.md README.md
git commit -m "docs: report CIWC quantitative result"
```

Report CIWC as retained only when all final criteria in the spec are proven by
the three summary files. If rejected, the commit must say so and preserve the
exact next candidate in the failure ladder.

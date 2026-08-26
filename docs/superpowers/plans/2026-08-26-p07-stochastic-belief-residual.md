# P07 Stochastic Belief Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and submit the minimal P07 stochastic belief residual that marginalizes four missing-region forecasts on top of frozen P00.

**Architecture:** Reuse the P04/P05 decoder-feature path. A small `3x3` head emits latent mean and log-scale, four reparameterized samples share a `1x1` residual output head, and their probabilities are averaged only inside the routing mask; outside the mask the returned logits are exactly P00.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing WSTS fast-track code, Slurm on Nibi.

**Spec:** `docs/superpowers/specs/2026-08-25-p07-stochastic-belief-residual-design.md`

## Global Constraints

- Keep P00 frozen and train only the belief and output heads.
- Use `T=1`, four stochastic samples, one seed, and one fixed 3,000-step run.
- Reuse existing 25%/50% structured BlockDrop and 2016--2020 training data.
- Select on 2021 M00/M01/M06/M07; do not access 2022--2023 in this first job.
- Add no configuration framework, registry, generic uncertainty API, multi-seed orchestration, or exhaustive tests.
- Run only import/shape/forward-backward checks on the login node. Run training and full evaluation through Slurm on a Nibi compute node.

---

### Task 1: Stochastic belief model and metric contract

**Files:**
- Modify: `tests/test_wsts_fast_track_residual_gate.py`
- Modify: `tests/test_wsts_fast_track_evaluate_missingness.py`
- Modify: `reproductions/wsts_fast_track/residual_gate.py`
- Modify: `reproductions/wsts_fast_track/evaluate_missingness.py`

**Interfaces:**
- Produces: `FrozenStochasticBeliefResidual(default_model, sample_count=4)`.
- Produces: `forward_with_uncertainty(routed_input) -> tuple[mean_logits, predictive_variance]`.
- Produces: `evaluate_batches(...)` metrics containing `brier` and, for P07, `mean_predictive_variance`.

- [ ] **Step 1: Write the failing behavior tests**

Add tests showing that P07 returns exact P00 logits and zero variance outside the missing mask, produces finite non-negative variance inside it, exposes trainable belief/output heads, and that `evaluate_batches` reports the hand-computed Brier score.

```python
def test_stochastic_belief_preserves_unmasked_logits_and_reports_variance():
    gate = FrozenStochasticBeliefResidual(_TinyDefault(), sample_count=4)
    routed_input = torch.zeros((1, 1, 41, 2, 2))
    routed_input[:, :, 0] = 2.0
    routed_input[:, :, 40, 0, 1] = 1.0
    torch.manual_seed(0)
    logits, variance = gate.forward_with_uncertainty(routed_input)
    torch.testing.assert_close(logits[~routed_input[:, 0, 40:41].bool()], torch.full((3,), 4.0))
    assert torch.count_nonzero(variance[~routed_input[:, 0, 40:41].bool()]) == 0
    assert torch.isfinite(variance).all()
    assert torch.all(variance >= 0)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_wsts_fast_track_residual_gate.py`

Expected: failure because `FrozenStochasticBeliefResidual` and `brier` do not exist.

- [ ] **Step 3: Implement the minimum stochastic head**

Add `P07-StochasticBeliefResidual`, `BELIEF_TRAINING_STEPS = 3_000`, and a module that concatenates the 16-channel detached decoder feature with the mask, predicts 16-channel `mu` and bounded `log_sigma`, draws four samples, maps them through one shared output head, averages probabilities, and restores exact default logits outside the mask. Add streaming Brier accumulation to `evaluate_batches`; if the model supplies `forward_with_uncertainty`, consume it once and accumulate mean variance.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_wsts_fast_track_residual_gate.py tests/test_wsts_fast_track_evaluate_missingness.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_wsts_fast_track_residual_gate.py tests/test_wsts_fast_track_evaluate_missingness.py reproductions/wsts_fast_track/residual_gate.py reproductions/wsts_fast_track/evaluate_missingness.py
git commit -m "feat: add P07 stochastic belief residual"
```

### Task 2: Reuse the residual training and evaluation path

**Files:**
- Modify: `reproductions/wsts_fast_track/train_residual_gate.py`
- Modify: `reproductions/wsts_fast_track/evaluate_residual_gate.py`
- Modify: `reproductions/wsts_fast_track/run_residual_gate_on_nibi.sh`

**Interfaces:**
- Consumes: `FrozenStochasticBeliefResidual`, `BELIEF_TRAINING_STEPS`, and the P07 prototype ID.
- Produces: `--stochastic-belief` CLI variant and a P07 checkpoint containing `belief_head`, `output_head`, `sample_count=4`, `steps=3000`, and `learning_rate=0.001`.

- [ ] **Step 1: Reuse the core round-trip test**

The user's fast-prototype constraint excludes unit tests for argparse and shell
glue. Extend the Task 1 core model test to assert that the two P07 state dicts
round-trip and retain identical seeded outputs; this directly protects the
checkpoint data consumed by the evaluator.

```python
torch.manual_seed(7)
expected, _ = gate.forward_with_uncertainty(routed_input)
clone.belief_head.load_state_dict(gate.belief_head.state_dict())
clone.output_head.load_state_dict(gate.output_head.state_dict())
torch.manual_seed(7)
actual, _ = clone.forward_with_uncertainty(routed_input)
torch.testing.assert_close(actual, expected)
```

- [ ] **Step 2: Verify the core test remains GREEN before glue changes**

Run: `pytest -q tests/test_wsts_fast_track_residual_gate.py`

Expected: pass. No source-text or argparse change-detector test is added.

- [ ] **Step 3: Add the P07 variant**

Add `--stochastic-belief` to the trainer/evaluator. Train P07 for exactly 3,000 steps with Adam at `1e-3`; seed torch and NumPy with zero. Save and strictly validate both head state dicts and the P07 metadata. Seed torch with zero immediately before each scenario evaluation. Add launcher variant `belief` with label `P07-stochastic-belief`.

- [ ] **Step 4: Run minimal verification**

Run: `pytest -q tests/test_wsts_fast_track_residual_gate.py tests/test_wsts_fast_track_evaluate_missingness.py`

Run: `python -m py_compile reproductions/wsts_fast_track/residual_gate.py reproductions/wsts_fast_track/train_residual_gate.py reproductions/wsts_fast_track/evaluate_residual_gate.py`

Expected: tests and compilation pass.

- [ ] **Step 5: Commit**

```bash
git add reproductions/wsts_fast_track/train_residual_gate.py reproductions/wsts_fast_track/evaluate_residual_gate.py reproductions/wsts_fast_track/run_residual_gate_on_nibi.sh tests/test_wsts_fast_track_residual_gate.py
git commit -m "feat: wire P07 fast-track campaign"
```

### Task 3: Submit the immutable Nibi prototype job

**Files:**
- Modify after submission: `docs/experiments/p00_p06_rapid_reliability.md`

**Interfaces:**
- Consumes: the accepted P00 completion record and launcher variant `belief`.
- Produces: one Slurm job running 3,000-step P07 training followed by 2021 M00/M01/M06/M07 evaluation on a compute node.

- [ ] **Step 1: Verify the submission inputs read-only**

Resolve the P00 record, Nibi site profile, current clean commit, data root, stats file, and launcher path. Confirm the run root derived from the future Slurm job ID does not exist.

- [ ] **Step 2: Submit one compute job**

Reuse the accepted P04 Nibi allocation shape, with a longer bounded wall time for
3,000 training steps. Invoke:

```bash
sbatch --account=def-vislearn_gpu --qos=interac --job-name=P07-stochastic-belief-H100MIG --time=00:30:00 --cpus-per-task=8 --mem=32G --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --output=/project/6085198/kulbear/wildfire/runs/slurm-P07-stochastic-belief-%j.out --chdir=/scratch/kulbear/wildfire-research-plan/.worktrees/nibi-minimal-res18 reproductions/wsts_fast_track/run_residual_gate_on_nibi.sh /project/6085198/kulbear/wildfire/runs/prototype-P00-FireDrop-C00-20398173/completed.json belief
```

- [ ] **Step 3: Verify scheduler acceptance**

Run `squeue -j "$job_id"` and `scontrol show job "$job_id"`, where `job_id` is
the integer returned by `sbatch --parsable`. Confirm the command points to the
P07 launcher and requests one H100 MIG GPU.

- [ ] **Step 4: Record the submission**

Append a short P07 section to the experiment report with commit, job ID, 3,000-step budget, four-sample belief residual, and pending status. Do not claim scientific results before the job completes.

- [ ] **Step 5: Verify and commit the record**

Run: `git diff --check && git status --short`

Commit:

```bash
git add docs/experiments/p00_p06_rapid_reliability.md
git commit -m "docs: record P07 Nibi submission"
```

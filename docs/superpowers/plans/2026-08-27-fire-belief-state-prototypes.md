# Fire Belief-State Prototypes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, submit, and screen six matched fire belief-state prototypes: recurrent filter, temporal attention, and reconstruction-first, each at `T=1` and `T=5`.

**Architecture:** A shared processed-space corruption and observation-correction contract feeds three small state estimators. Every estimator changes only unavailable pixels in processed active-fire channel 39 and sends the corrected latest day through the same frozen P00 Res18-U-Net. One trainer, evaluator, and Nibi launcher keep data, optimization, metrics, and checkpoint format matched.

**Tech Stack:** Python 3.10, NumPy, PyTorch, torchvision focal loss, pytest, Bash, Slurm on Nibi.

**Spec:** `docs/superpowers/specs/2026-08-27-fire-belief-state-design.md`

## Global Constraints

- Train on corrected-index 2016--2020 with inverse-year sampling; select only on 2021; open 2022--2023 only for the surviving candidate and its matched reconstruction control.
- Use `[B,T,40,H,W]`, with detection hour at channel 38, binary active fire at 39, and validity `R` packed as channel 40 where one means observed.
- Use `T=1` day 5 or `T=5` days 1--5 for the identical day-6 target population; controlled evaluation uses history adjustment six.
- State inference sees channels `0:33`, active-fire evidence `38:40`, and `R`; channels `33:38` cannot enter a state estimator. Frozen P00 still receives all latest-day channels.
- Freeze P00. Use seed 0, 3,000 optimizer steps, AdamW `1e-3`, effective batch 64, and uniform M01/M06/M07 training corruption.
- If one `T=5` job is out of memory, resubmit only it with physical batch 32 and two-step gradient accumulation.
- A/B use unweighted sigmoid focal forecast loss plus `0.1` masked state focal. C uses only masked state focal and never uses the next-day target in its objective.
- M00 correction and logits must be exactly P00. Screen on mean corrupted AP, at least two improving scenarios, and no scenario below P00 by more than `0.01 AP`.
- Run only the focused belief-state test, Python compilation, `bash -n`, and `git diff --check`. No global pytest, multi-seed run, broad smoke test, or login-node GPU/data work.
- Tasks 2--4 run in separate Git worktrees after Task 1 and are integrated as individual reviewed commits.

## File Structure

- `latent_state_common.py`: constants, processed corruption, dataset adapters, loss, observation correction, frozen-P00 base, checkpoint helpers, and screen rule.
- `state_filter.py`: recurrent tiny U-Net state proposal.
- `state_attention.py`: prior-token pixelwise temporal attention proposal.
- `reconstruction_baseline.py`: reconstruction-only TinyMaskUNet proposal.
- `train_belief_state.py`: one fixed trainer selected by method and history.
- `evaluate_belief_state.py`: direct-P00 comparison, four-scenario metrics, exact M00 check, and screen summary.
- `run_belief_state_on_nibi.sh`: immutable train-then-evaluate runner for one method/history.
- `test_wsts_fast_track_belief_state.py`: the only focused behavior tests.
- `p00_p06_rapid_reliability.md`: submission, result, and decision record.

---

### Task 1: Shared tensor, corruption, loss, and frozen-P00 contract

**Files:**
- Create: `reproductions/wsts_fast_track/latent_state_common.py`
- Create: `tests/test_wsts_fast_track_belief_state.py`

**Interfaces:**
- Consumes: `structured_block_mask()`, `ControlledMissingnessDataset`, and `sigmoid_focal_loss`.
- Produces: `pack_observations`, `split_observations`, `apply_processed_corruption`, `observation_consistent_state`, `masked_unweighted_focal`, `BeliefStateTrainingDataset`, `BeliefStateEvaluationDataset`, `FrozenP00BeliefModel`, trainable-state save/load helpers, and `screen_2021`.
- Packed input is always `[B,T,41,H,W]`; channel 40 is `R`.

- [ ] **Step 1: Write the failing common-contract test**

Define the one fake downstream predictor used by all focused model tests. It
has no trainable parameters and deliberately depends on active-fire channel
39, so gradients to a missing-state proposal are observable:

```python
class _TinyDefault(torch.nn.Module):
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed[:, 0, 39:40] * 2.0

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()
```

```python
def test_common_contract_preserves_observations_and_masks_state_loss() -> None:
    from reproductions.wsts_fast_track.latent_state_common import (
        apply_processed_corruption, masked_unweighted_focal,
        observation_consistent_state, pack_observations, split_observations,
    )
    for history in (1, 5):
        clean = torch.ones((history, 40, 8, 8))
        m01, r01 = apply_processed_corruption(
            clean, "M01", normalized_active_fire_zero=-2.0,
            key_digest="00" * 32,
        )
        assert torch.count_nonzero(r01) == 0
        assert torch.count_nonzero(m01[:, 39]) == 0
        m06, r06 = apply_processed_corruption(
            clean, "M06", normalized_active_fire_zero=-2.0,
            key_digest="00" * 32,
        )
        assert int((r06 == 0).sum()) == history * 16
        packed = pack_observations(m06[None], r06[None])
        x, reliability = split_observations(packed)
        proposal = torch.zeros((1, 1, 8, 8))
        corrected = observation_consistent_state(
            x[:, -1, 39:40], proposal, reliability[:, -1]
        )
        observed = reliability[:, -1].bool()
        assert torch.equal(corrected[observed], x[:, -1, 39:40][observed])
        loss = masked_unweighted_focal(
            proposal, clean[None, -1, 39:40], ~observed
        )
        assert torch.isfinite(loss)
```

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_wsts_fast_track_belief_state.py::test_common_contract_preserves_observations_and_masks_state_loss`.

Expected: import failure because `latent_state_common` does not exist.

- [ ] **Step 3: Implement tensor, corruption, and loss functions**

Use these exact names and values:

```python
PROCESSED_FEATURES = 40
ACTIVE_HOUR = 38
ACTIVE_BINARY = 39
RELIABILITY = 40
STATE_CONTEXT = slice(0, 33)
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))
SCENARIOS = ("M00", "M01", "M06", "M07")
TRAIN_SCENARIOS = ("M01", "M06", "M07")
```

`pack_observations(features,reliability)` validates `[B,T,40,H,W]` plus `[B,T,1,H,W]` and concatenates them. `split_observations(packed)` requires exactly 41 channels. `observation_consistent_state()` requires equal shapes and returns `torch.where(reliability.bool(), observed, torch.sigmoid(proposal_logits))` so observed values are exact.

`masked_unweighted_focal()` calls torchvision focal with `alpha=-1.0`, `gamma=2.0`, and `reduction="none"`; it averages all pixels when mask is absent and exactly the selected pixels otherwise, rejecting an empty mask.

`apply_processed_corruption(clean,scenario_id,normalized_active_fire_zero,key_digest)` clones `[T,40,H,W]`. M00 leaves it intact. M01 sets channel 38 to normalized raw zero, channel 39 to zero, and all `R` to zero. M06/M07 build one 25%/50% `structured_block_mask`, reuse it across time, zero `PROCESSED_DYNAMIC_NON_FIRE` and channel 39 inside it, set channel 38 to normalized raw zero, and set `R=0` inside it. Reject every other scenario.

- [ ] **Step 4: Implement the two narrow dataset adapters**

`BeliefStateTrainingDataset(base,normalized_active_fire_zero)` obtains the already aligned clean crop, draws one of M01/M06/M07 using the worker-seeded NumPy RNG, and returns `(packed, clean[-1,39:40], target, scenario_index)`. Compute sampler weights on `base`, not this wrapper.

The trainer derives the normalized raw-zero value from the installed training
statistics exactly as
`-base.means[0,22,0,0] / base.stds[0,22,0,0]`; it is not a tunable value.

`BeliefStateEvaluationDataset(controlled)` requires `routing_mask_channel=True`, splits its 41st channel as missing mask, overrides that mask to all ones for M01, sets `R=1-missing`, and returns `(packed,target)`. Thus M00 gets all-one `R` and M06/M07 retain their controlled block.

- [ ] **Step 5: Implement the frozen-P00 base and checkpoint helpers**

```python
class FrozenP00BeliefModel(torch.nn.Module):
    def __init__(self, default_model, history):
        super().__init__()
        if history not in {1, 5}:
            raise ValueError("history must be 1 or 5")
        self.default_model = default_model.requires_grad_(False)
        self.default_model.eval()
        self.history = history

    def train(self, mode=True):
        super().train(mode)
        self.default_model.eval()
        return self

    def infer_state_logits(self, features, reliability):
        raise NotImplementedError

    def forward_state(self, packed):
        features, reliability = split_observations(packed)
        if features.shape[1] != self.history:
            raise ValueError("packed history differs from model history")
        state_logits = self.infer_state_logits(features, reliability)
        state = observation_consistent_state(
            features[:, -1, 39:40], state_logits, reliability[:, -1]
        )
        latest = features[:, -1].clone()
        latest[:, 39:40] = state
        forecast = self.default_model(latest[:, None])
        return forecast, state_logits, state

    def forward(self, packed):
        return self.forward_state(packed)[0]

    def compute_loss(self, logits, target):
        return masked_unweighted_focal(logits.squeeze(1), target)
```

`trainable_state_dict()` stores only state-dict keys outside `default_model.`; the load helper requires exact key equality. `screen_2021()` calculates per-scenario AP deltas and returns mean delta, count above zero, worst delta, and pass status for the three frozen rules.

- [ ] **Step 6: Run GREEN and commit**

Run the focused node, `python -m py_compile reproductions/wsts_fast_track/latent_state_common.py`, and `git diff --check`; then commit only the common module and focused test as `feat: add fire belief-state data contract`.

---

### Task 2: Direction A recurrent filter

**Files:**
- Create: `reproductions/wsts_fast_track/state_filter.py`
- Modify: `tests/test_wsts_fast_track_belief_state.py`

**Interfaces:**
- Consumes: Task 1's base class, constants, and correction.
- Produces: `RecurrentBeliefFilter(default_model,history)` with `[B,1,H,W]` state logits.

- [ ] **Step 1: Write and run the failing recurrent test**

```python
def test_recurrent_filter_supports_both_histories_and_keeps_m00_exact() -> None:
    from reproductions.wsts_fast_track.latent_state_common import pack_observations
    from reproductions.wsts_fast_track.state_filter import RecurrentBeliefFilter
    for history in (1, 5):
        features = torch.randn((2, history, 40, 16, 16))
        reliability = torch.ones((2, history, 1, 16, 16))
        model = RecurrentBeliefFilter(_TinyDefault(), history)
        forecast, state_logits, state = model.forward_state(
            pack_observations(features, reliability)
        )
        assert state_logits.shape == state.shape == (2, 1, 16, 16)
        assert torch.equal(forecast, model.default_model(features[:, -1:]))
        reliability[:, -1, :, :8] = 0
        model(pack_observations(features, reliability)).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.trainable_parameters())
```

Run its exact pytest node and expect the missing-module failure.

- [ ] **Step 2: Implement the shared recurrent updater**

Implement one three-level tiny U-Net with GroupNorm/SiLU, widths `16/32/64/96`, bilinear upsampling, skip concatenation, 35 input channels, and one output logit. Unroll the same updater once or five times:

```python
state = features.new_zeros((features.shape[0], 1, *features.shape[-2:]))
for step in range(self.history):
    proposal = self.updater(torch.cat(
        (features[:, step, :33], state, reliability[:, step]), dim=1
    ))
    state = observation_consistent_state(
        features[:, step, 39:40], proposal, reliability[:, step]
    )
return proposal
```

Expose only updater parameters. Add no ConvLSTM, stochastic state, or forecast decoder.

- [ ] **Step 3: Run and commit**

Run the recurrent node, compile `state_filter.py`, run `git diff --check`, and commit model plus test as `feat: add recurrent fire belief filter`.

---

### Task 3: Direction B prior-token temporal attention

**Files:**
- Create: `reproductions/wsts_fast_track/state_attention.py`
- Modify: `tests/test_wsts_fast_track_belief_state.py`

**Interfaces:**
- Consumes: Task 1's base and tensor constants.
- Produces: `TemporalAttentionBeliefState(default_model,history)` and `attention_weights(features,reliability)` returning `[B,T+1,H,W]`.

- [ ] **Step 1: Write and run the failing attention test**

```python
def test_attention_state_is_history_matched_and_missing_fire_invariant() -> None:
    from reproductions.wsts_fast_track.latent_state_common import pack_observations
    from reproductions.wsts_fast_track.state_attention import TemporalAttentionBeliefState
    one = TemporalAttentionBeliefState(_TinyDefault(), 1)
    five = TemporalAttentionBeliefState(_TinyDefault(), 5)
    assert sum(p.numel() for p in one.trainable_parameters()) == sum(
        p.numel() for p in five.trainable_parameters()
    )
    features = torch.randn((2, 5, 40, 12, 12))
    reliability = torch.zeros((2, 5, 1, 12, 12))
    changed = features.clone()
    changed[:, :, 38:40] = 1000.0
    torch.testing.assert_close(
        five.infer_state_logits(features, reliability),
        five.infer_state_logits(changed, reliability),
    )
    weights = five.attention_weights(features, reliability)
    assert weights.shape == (2, 6, 12, 12)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    reliability[:, :-1] = 1
    forecast = five(pack_observations(features, reliability))
    forecast.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in five.trainable_parameters())

    reliability.fill_(1)
    assert torch.equal(
        five(pack_observations(features, reliability)),
        five.default_model(features[:, -1:]),
    )
```

Run its exact pytest node and expect the missing-module failure.

- [ ] **Step 2: Implement parameter-matched temporal attention**

Use 16 channels: `Conv2d(34,16,1)` context, `Conv2d(3,16,1)` fire evidence, `Conv2d(34,16,1)` prior, shared depthwise `3x3` plus GroupNorm/SiLU, and a `Conv2d(16,1,3,padding=1)` state head. Append fixed lag `(step-(T-1))/4` to context `0:33`. Multiply the fire projection by `R` before token fusion, so missing active-fire values cannot affect output. Use latest context with lag zero as prior/query, dot-product attention over T evidence tokens plus prior, scale by `sqrt(16)`, and softmax across `T+1`. Do not add learned Q/K/V or transformer blocks.

- [ ] **Step 3: Run and commit**

Run the attention node, compile `state_attention.py`, run `git diff --check`, and commit model plus test as `feat: add temporal attention fire state`.

---

### Task 4: Direction C reconstruction-first baseline

**Files:**
- Create: `reproductions/wsts_fast_track/reconstruction_baseline.py`
- Modify: `tests/test_wsts_fast_track_belief_state.py`

**Interfaces:**
- Consumes: Task 1's base and tensor constants.
- Produces: `ReconstructionFirstBeliefState(default_model,history)` with a one-logit `TinyMaskUNet`.

- [ ] **Step 1: Write and run the failing reconstruction test**

```python
def test_reconstruction_baseline_uses_state_loss_and_preserves_m00() -> None:
    from reproductions.wsts_fast_track.latent_state_common import (
        masked_unweighted_focal, pack_observations,
    )
    from reproductions.wsts_fast_track.reconstruction_baseline import ReconstructionFirstBeliefState
    for history in (1, 5):
        features = torch.randn((2, history, 40, 16, 16))
        reliability = torch.ones((2, history, 1, 16, 16))
        model = ReconstructionFirstBeliefState(_TinyDefault(), history)
        assert torch.equal(
            model(pack_observations(features, reliability)),
            model.default_model(features[:, -1:]),
        )
        reliability[:, -1, :, :8] = 0
        _, state_logits, _ = model.forward_state(pack_observations(features, reliability))
        loss = masked_unweighted_focal(
            state_logits, features[:, -1, 39:40], ~reliability[:, -1].bool()
        )
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.trainable_parameters())
```

Run its exact pytest node and expect the missing-module failure.

- [ ] **Step 2: Implement reconstruction-only TinyMaskUNet**

Concatenate context `0:33`, active evidence `38:40`, and `R` for each step, flatten time to `36*T`, and use the same three-level GroupNorm/SiLU U-Net pattern and `16/32/64/96` widths as A. Output only the latest binary-fire logit:

```python
state_inputs = torch.cat(
    (features[:, :, :33], features[:, :, 38:40], reliability), dim=2
).flatten(1, 2)
return self.reconstructor(state_inputs)
```

Expose only reconstructor parameters. Forecast labels cannot enter this module.

- [ ] **Step 3: Run and commit**

Run the reconstruction node, compile `reconstruction_baseline.py`, run `git diff --check`, and commit model plus test as `feat: add reconstruction-first fire state`.

---

### Task 5: Unified trainer, evaluator, and Nibi runner

**Files:**
- Create: `reproductions/wsts_fast_track/train_belief_state.py`
- Create: `reproductions/wsts_fast_track/evaluate_belief_state.py`
- Create: `reproductions/wsts_fast_track/run_belief_state_on_nibi.sh`
- Modify: `tests/test_wsts_fast_track_belief_state.py`

**Interfaces:**
- Consumes: Task 1--4 models, corrected resolver and sampler, P00 loader, controlled dataset, and `evaluate_batches`.
- Produces: `build_model`, `combine_losses`, `screen_payload`, one checkpoint schema, one evaluator summary, and runner arguments `P00_COMPLETED_RECORD METHOD HISTORY`.

- [ ] **Step 1: Write and run the failing objective/screen test**

```python
def test_objective_and_screen_are_frozen() -> None:
    from reproductions.wsts_fast_track.evaluate_belief_state import screen_payload
    from reproductions.wsts_fast_track.train_belief_state import combine_losses
    forecast, state = torch.tensor(2.0), torch.tensor(3.0)
    torch.testing.assert_close(combine_losses("filter", forecast, state), torch.tensor(2.3))
    torch.testing.assert_close(combine_losses("attention", forecast, state), torch.tensor(2.3))
    torch.testing.assert_close(combine_losses("reconstruction", forecast, state), state)
    results = {
        scenario: {"model": {"metrics": {"avg_precision": a}},
                   "p00": {"metrics": {"avg_precision": b}}}
        for scenario, a, b in (
            ("M01", .11, .10), ("M06", .31, .30), ("M07", .19, .20)
        )
    }
    screen = screen_payload(results)
    assert screen["pass"] is True
    assert screen["improved_scenarios"] == 2
```

Run the exact node and expect trainer/evaluator import failure.

- [ ] **Step 2: Implement the fixed trainer**

Require method, history, P00 record, upstream/data/stats/output paths, and Git commit. Defaults are physical batch 64, accumulation one, workers eight, and CUDA. Install the C00 runtime contract, assign `resolve_dataset_index` to the upstream dataset class, instantiate full features with chosen history and no duplicate removal, wrap with `BeliefStateTrainingDataset`, and use `balanced_year_sampling_weights(base)` in a seed-0 `WeightedRandomSampler`.

`build_model()` maps filter, attention, and reconstruction to their three exact classes. `combine_losses()` returns state loss for reconstruction and `forecast + 0.1*state` otherwise. One optimizer step consumes the configured number of microbatches; divide by accumulation before backward and step AdamW once. For C, call `infer_state_logits()` directly, ignore the batched next-day target, and skip P00 entirely during training; record forecast loss as null. Reject non-finite losses, non-CUDA CUDA requests, effective batch other than 64, and existing output.

Save schema version one, prototype `fire-belief-state-v1`, method/history, P00 paths, Git commit, train years, seed, steps, optimizer, learning rate, physical/effective batch, accumulation, corruption list, state weight, parameter count, final loss components, and only `trainable_state_dict(model)`.

- [ ] **Step 3: Implement the matched evaluator**

Strict-check checkpoint metadata, reconstruct P00 plus selected state model, and strict-load only non-P00 state. For M00/M01/M06/M07 build full-feature controlled datasets with selected history, adjustment six, `routing_mask_channel=True`, then wrap them with `BeliefStateEvaluationDataset`.

Evaluate belief model and a `LatestDayP00` wrapper on identical loaders using `evaluate_batches`. On every M00 batch require `torch.equal(belief_logits,p00_logits)`. Write scenario JSON with both metrics and AP delta. Write `summary.json` with identities, all scenarios, `m00_exact: true`, `screen_2021(results)`, and `scientific_claim: false`. Export `screen_payload = screen_2021` for the focused test.

- [ ] **Step 4: Implement the compute-node runner**

Require exactly P00 record, `{filter|attention|reconstruction}`, and `{1|5}`. Use established Nibi paths and environment. Create `/project/6085198/kulbear/wildfire/runs/prototype-fire-belief-${method}-t${history}-${SLURM_JOB_ID}`, archive the submitted commit, record status and hashes, disable W&B and HDF5 locking, run trainer, then evaluator. The runner itself performs no login-node work.

- [ ] **Step 5: Run focused verification and commit**

Run `pytest -q tests/test_wsts_fast_track_belief_state.py`, compile the six new Python modules, run `bash -n` on the runner, and run `git diff --check`. Commit trainer, evaluator, runner, and test as `feat: wire matched fire belief experiments`.

---

### Task 6: Submit six Nibi screens, monitor, and record the decision

**Files:**
- Modify: `docs/experiments/p00_p06_rapid_reliability.md`

**Interfaces:**
- Consumes: accepted P00 completed record, Task 5 runner, and six summaries.
- Produces: six job IDs/run roots, 2021 table, and frozen promote/stop decision.

- [ ] **Step 1: Resolve immutable inputs**

Locate the accepted P00 record referenced by P13, verify record/checkpoint, require a clean implementation branch, and record HEAD. Use only read-only filesystem commands on the login node.

- [ ] **Step 2: Submit the six independent jobs**

Submit all combinations without dependencies using the successful P13
account/resource form: the 40 GB H100 MIG slice, eight CPUs, 32 GB, interactive
GPU QoS, and one hour. Use this exact loop from the repository worktree:

```bash
p00_record=/project/6085198/kulbear/wildfire/runs/prototype-P00-FireDrop-C00-20398173/completed.json
for method in filter attention reconstruction; do
  for history in 1 5; do
    sbatch --account=def-vislearn_gpu --qos=interac \
      --job-name="belief-${method}-t${history}" --time=01:00:00 \
      --cpus-per-task=8 --mem=32G \
      --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 \
      --output="/project/6085198/kulbear/wildfire/runs/slurm-belief-${method}-t${history}-%j.out" \
      --chdir=/scratch/kulbear/wildfire-research-plan/.worktrees/nibi-minimal-res18 \
      reproductions/wsts_fast_track/run_belief_state_on_nibi.sh \
      "${p00_record}" "${method}" "${history}"
  done
done
```

This matches P13 job `20611586`: account `def-vislearn_gpu`, QoS `interac`, and
resource `nvidia_h100_80gb_hbm3_3g.40gb`. Record every returned ID mapped to
method/history.

- [ ] **Step 3: Commit the pending record immediately**

Add implementation commits, P00 record, fixed protocol, six job mappings, pending status, and the statement that the login node only ran submission/scheduler inspection. Run `git diff --check` and commit as `docs: record fire belief-state submissions`.

- [ ] **Step 4: Monitor coarsely and repair only concrete failures**

Use `squeue`, `sacct`, and logs about every 30 minutes or once the scheduler reports a terminal state. If only a `T=5` job OOMs, add bounded runner pass-through for physical batch 32 plus accumulation two, commit, and resubmit only that configuration. For any other traceback, diagnose it and make the smallest scoped fix; add no global validation framework.

- [ ] **Step 5: Apply the frozen 2021 screen**

Tabulate method, history, parameters, M00 exactness, M01/M06/M07 AP, P00 deltas, corrupted mean, signal pass, and A/B delta against reconstruction at the same history. Also show M01 beside the frozen P13 M01 diagnostic and M06/M07 beside the frozen P10 block diagnostic; these are comparisons, not new routing or tuning targets. A/B is a candidate only if it passes the signal screen and exceeds matched C. If neither qualifies, stop without tuning or opening 2022--2023.

- [ ] **Step 6: Commit final evidence**

Record terminal states, run roots, losses, metrics, screen, any repair, and exact next action. Run `git diff --check` and commit as `docs: record fire belief-state screen`. The branch must finish clean.

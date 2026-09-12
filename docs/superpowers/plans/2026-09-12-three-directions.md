# Three reliability directions implementation and validation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the bounded mechanism implementation and review. The controller owns Slurm integration, experiment execution and reporting.

**Goal:** Validate all three user-approved directions and report positive or negative empirical results.
**Architecture:** Reuse the canonical T1 Res18-U-Net and T5 Res18-UTAE, their corrected data contract, and the existing evaluator. Add one small mechanisms module and a separate experiment driver; no unrelated refactoring.
**Tech Stack:** Existing pinned PyTorch/SMP runtime on Nibi Slurm.
**Spec:** The three designs and acceptance rules below implement the user's approved September 12 proposals.

## Global constraints

- All model execution and tensor tests run in Slurm. Login nodes perform source, metadata, Git and submissions only.
- Train/calibrate only 2016–2020; selection 2021, T1/T5, seed 0. Full four-scenario evaluation: M00, M01, M06, M07; 3181 validation samples per scenario.
- Matched continuation uses the same original B3/B5 initialization, AdamW, initial LR .001, cosine-to-zero, 3000 updates, effective batch 64. Physical batch 64 if smoke fits; otherwise rerun all matched arms with the same feasible physical batch.
- No hyperparameter search. Final-step checkpoints only. Main AP is mean M01/M06/M07; block AP mean M06/M07. Clean guardrail −.010 absolute AP.
- Fresh matched cosine control is the attribution reference for learned changes. Reuse existing checkpoints as teachers, never select teachers using new results.
- A negative valid screen completes that direction's screening; it is not evidence the entire class cannot work. A positive screen requires seeds 1/2 and fixed 2022/2023 reporting before adoption. Historical test exposure must be disclosed.

## Task 1: Small mechanism module and focused tests

Files: reproductions/cross_history/three_directions.py, tests/test_cross_history_three_directions.py.

- [ ] Test first, observe expected failure in a CPU Slurm allocation, implement, then verify.
- [ ] Implement BN moment recalibration without gradients or parameter updates. Model stays eval; hooks collect pre-BN sums/squares/counts for all BatchNorm layers. The fixed-stat evaluation forward makes all variants collect from exactly the same activations, avoids dropout and physical batch statistical confounds, and preserves T5 frame ordering. Support aggregate and block-present/block-absent banks, float64 accumulated moments, pooled unbiased variance, empty-bank errors. Apply banks by module names with shape checking; leave affine weights, GroupNorm, original weights unchanged. Bank routing depends only on observed spatial mask.
- [ ] Implement shallow residual input stems: each path is 3x3 conv (bias=False) -> SiLU -> 3x3 conv (bias=False); hidden width 16, final conv zero initialized. Mixed arm processes all channels; typed arm separately processes static and dynamic channel groups, then reassembles original ordering. Both have exactly matching parameter counts and initial identity. This preserves pretrained initialization while learning nonlinear within-group features before the original backbone first mixes groups. Dynamic includes all processed dynamic nonfire columns AND active fire columns 38/39, mapped through MULTI_FEATURES for T5; static is the complement. No masks/gates/auxiliary losses in this direction.
- [ ] Implement observable route IDs: clean=0, fire-only=1, block<=.375=2, block>.375=3; block has priority when block and fire missingness coexist. Inputs contain last two channels spatial/fire masks. Route does not read target or scenario ID.
- [ ] Implement detached teacher Bernoulli KL using logits/logsigmoid, mean pixel reduction and finite checks. Fixed loss weight .1 in runner. Teacher/student receive identical corrupt input; no clean teacher view.
- [ ] Focused tests verify moment math, no parameter mutation/dropout activation, bank independence and reload, exact stem identity/parameter matching/gradient reachability/group separation, observable route precedence, KL identity/nonnegativity/extreme finite logits/teacher detached/student gradients.

## Task 2: Slurm driver and provenance

Files: reproductions/cross_history/run_three_directions.py, run_three_directions_slurm.sh, docs/experiments/three_directions_manifest.json.

- [ ] Freeze explicit paths and SHA256 hashes of canonical seed-matched control/cosine/mild/severe teacher checkpoints; validate history, architecture, seed and method/fraction when loaded.
- [ ] BN diagnostic starts from existing X22 checkpoint. Collect moments from 4096 deterministic training examples under existing .3/.3 corruption distribution, fixed physical batches and no augmentation changes across banks. Evaluate unchanged checkpoint, common recalibration, and conditional recalibration. Only if conditional-minus-common primary >0 in both histories, conditional also improves original X22 primary, and clean guardrail passes versus both references, train separate BN affine parameters as the predefined extension; learned screen requires +.005 in both histories versus matched shared BN.
- [ ] Train cosine control, mixed shallow stem, typed shallow stem, and routed-teacher student from identical original B3/B5 weights and RNG/data ordering. Student trains all existing parameters. Teacher routes are fixed ERM for clean, X22 for fire-only, .25/.5 specialists for blocks. Evaluate frozen teacher route on the same four scenarios.
- [ ] Architecture screen: typed-minus-mixed primary >=+.005 in both histories, clean >=−.010; additionally typed must improve original cosine control. Distillation compression screen: student primary >= teacher−.003, student primary > no-KD cosine control, clean >=control−.010 in both histories. Report all per-scenario differences and physical model storage; do not claim routing latency reduction.
- [ ] One-step real-data smokes cover both histories, stem backward, teacher routing, checkpoint reload equivalence and BN banks. Unit tests and unchanged relevant evaluator tests must pass before full screens.
- [ ] Commit source before job submission; each job archives the immutable commit, records allocation, source hash, inputs, budgets, losses, parameters, time and results. No duplicate live jobs; inspect sacct/squeue before retries.

## Task 3: Execute gates and report

- [ ] Complete both-history seed-0 measurements for every direction and every required control; no selective cancellation of missing comparison cells.
- [ ] Follow predefined positive gates with seeds 1/2, then frozen 2022/2023 evaluation; negative gates stop without retuning.
- [ ] Produce docs/experiments/three_directions_results.md and small machine-readable paired results, reporting all outcomes, uncertainty, attribution controls, runtime/storage and limitations.
- [ ] Independently review experimental code and result evidence; resolve load-bearing issues, commit final artifacts and report results to user. Goal is complete only when required jobs and reports are complete.

## Calibration correction after the fixed-forward diagnostic (before revised AP)

The fixed-activation probe completed for both histories (21787290/91). T1 conditional BN had M06/M07 AP .01745/.01140; T5 .24568/.09005. This is retained as diagnostic evidence, not erased. CPU21787565 verifies nonzero variances and pooled-bank algebra; GPU21787609 shows direct/routed predictions exactly equal, but T1 BN activation RMS grows from .48 to 1.99e6 after replacing all statistics. Downstream statistics were estimated under old upstream normalization, so simultaneously substituting them creates an inconsistent activation chain.

One bounded correction uses the SAME 4096 examples, masks, grouping and frozen weights, but enables per-batch normalization while collecting moments. Only BN modules use training mode with track_running_stats=False; Dropout and all other modules stay eval. Restore BN flags afterwards, and require every state_dict entry to remain unchanged. This follows the batch-stat normalization principle of the installed torch.optim.swa_utils.update_bn, without its model-wide dropout activation or unweighted running-stat updates. Both common/conditional banks still share exactly the same calibration forwards.

Keep `fixed` and `batch_stats` source/summary metadata separate and never mix them in a comparison. Apply the previously frozen positive/negative gate to this corrected calibration assay; there is no coefficient, mask, architecture, sample-budget or test-year search. The optional BN-affine extension remains gated on corrected positive evidence.

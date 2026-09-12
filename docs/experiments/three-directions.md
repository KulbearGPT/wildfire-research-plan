# Three retained-line extensions: frozen validation protocol

Registered 2026-09-12 before new evaluation. User authorized completing all
three directions and reporting results. Scope: existing canonical Res18 T1/T5,
controlled M00/M01/M06/M07, aligned target dates and original train split.
These are hypotheses, not three established independent contributions.

## Fixed experiments

1. **Conditional BN statistics (N).** Freeze X22 weights and affine parameters.
   Compare original X22, shared recalibration, and four-condition recalibration
   of every BatchNorm2d (encoder and decoder). Conditions: no added corruption,
   global FireDrop only, spatial BlockDrop only, both. Pool 25/50% severities.
   Other normalization and dropout remain in eval mode. During calibration,
   normalize each group using batch moments and accumulate exact channel sums,
   squared sums and pixel counts; use global unbiased variance for inference.
   Shared and conditional variants see identical 2048 training crops (128
   batches of 16), shuffle seed fixed to run seed, independent .3/.3 corruption.
   No gradients, label use, test adaptation or affine fitting. Check every
   condition has support. Preserve masks at all five history frames.
2. **Weight midpoint (W).** Same history, architecture, seed, original
   initialization and 3000-update X22 / original constant-LR X14 endpoints.
   Fixed alpha=.5; no coefficient search. Average parameters, then recalibrate
   BatchNorm2d with the same shared protocol above for midpoint AND endpoints.
   Keep non-BN buffers equal or reject incompatibility. Evaluate all scenarios
   without output routing. Original uncalibrated endpoints remain references.
3. **Same-input routed distillation (D).** One canonical student initialized
   from the matched final X22. Four frozen teachers reproduce the retained
   route: fresh ERM for M00, X22 for FireDrop, fixed25/fixed50 X17 experts for
   the corresponding block severities. Both-corruption training samples use
   X22 (fixed fallback). Teacher and student receive the same corrupted input.
   Supervised original focal loss + .1 mean Bernoulli KL(teacher || student),
   temperature 1, detached teachers. Matched control has no KL and identical
   initialization/data/RNG/budget. Both use 3000 AdamW updates, lr .001 cosine
   to zero, effective batch64, physical batch16, and .3/.3 corruption. Teacher
   inference must not change student RNG/BatchNorm. All student parameters
   train; a single checkpoint serves all evaluation conditions.

## Decisions and completion

- Complete real-data smoke and all four 2021 scenarios for both histories for
  all three directions; failure of one does not cancel another direction.
- N advances when primary gain >=.005 against BOTH original X22 and shared BN
  in both histories, with M00 delta >=-.010 against original X22.
- W advances when primary gain >=.005 against recalibrated X22 in both histories,
  primary positive against original X22, and M00 >=-.010 against original X22.
  Report recalibrated X14 and the existing route as additional controls.
- D advances when both histories retain routed-teacher primary within .005,
  no scenario AP drops more than .010 from routed teachers, and primary beats
  the no-KL student. Report parameter bytes/checkpoint storage, measured time,
  and training cost; do not infer per-sample speedup from teacher count.
- A failed fixed seed-0 screen closes that exact recipe with numeric evidence,
  not a claim of universal impossibility. No coefficient/BN-budget sweep.
- Passing directions run matched seeds1/2. Confirm grouped mean positive versus
  nearest control (D: same noninferiority and no-KL superiority conditions),
  with M00 guardrail. Only confirmed recipes receive fixed 2022/23 evaluation.
- 2022/23 were already exposed in prior research: report these as retrospective
  robustness replication, never an untouched independent test or deployment
  claim. No new-event holdout is fabricated from previously used data.
- Completion means all three fixed recipes have actual paired T1/T5 results,
  their gates are evaluated, any gate-required confirmations/evaluations finish,
  and code/configuration/job IDs/metrics/limitations are committed and reported.

## Implementation and provenance

Worktree: `.worktrees/three-directions`, branch `research/three-directions`,
base `864f675`. Reuse existing runtime, dataset, Forecaster and exact pooled AP.
Artifacts stay under `/project/6085198/kulbear/wildfire/runs/three-directions-*`.
Source runs and checksums are recorded in `reproductions/three_directions/sources.json`.
Tests and all model execution run in Slurm. Login work is metadata-only.

## Execution ledger

- Source audit: all 30 checkpoints have frozen SHA256 and matching metadata.
  T5 seed0 ERM has a completed M00 scene but no original full summary; that
  scene is used only as the retained route's M00 reference. Its checkpoint is
  strictly loaded in the smoke and its training metadata requires 3000 steps.
- Six mechanism unit tests passed in CPU job `21786795`; six report/gate tests
  passed on stdlib Python. Tests first failed before their implementation.
- T1/T5 all-mode real-data smoke jobs `21786838/21786839` both completed with
  exit `0:0` in 1m52s/2m13s. Every mode produced all four finite scene metrics.
  Smoke uses only two evaluation samples and provides no AP effectiveness evidence.
- Within each history, four calibration variants had identical sample grouping.
  T1 groups were `[62,24,29,13]`, T5 `[58,29,30,11]` for 128 smoke crops.
  First supervised loss matched exactly between control and KD student:
  T1 `.006080592866055667`, T5 `.006270543788559735`.
  Peak smoke GPU memory for KD was 0.968/1.982 GB at T1/T5.
- Full-screen commands and job IDs are in `three-directions-jobs.json`.
  Source `e5593d3` is pinned for the 12 formal runs. No confirmation gate has
  been evaluated yet; all three directions remain unverified scientifically.

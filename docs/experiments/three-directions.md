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
  Source `e5593d3` is pinned for the 12 formal runs. All seed-0 gates have now been evaluated; see the final screen below.

## Completed normalization and midpoint screens

All eight non-training runs completed with Slurm exit `0:0`; each scene contains
3181 aligned samples / 52,117,504 pixels. Deltas below are absolute AP, not
relative percentages. Source result paths and all scene deltas are recorded in
`three-directions-partial.json` (D is explicitly incomplete).

| Direction | History | M00 | M01 | M06 | M07 | Primary delta vs shared recalibrated X22 | Screen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| N conditional BN | T1 | .570710 | .075217 | .051037 | .051107 | -.240553 | fail |
| N conditional BN | T5 | .597608 | .341844 | .379071 | .194650 | -.020333 | fail |
| W fixed midpoint | T1 | .585799 | .214325 | .383715 | .200754 | -.033409 | fail |
| W fixed midpoint | T5 | .588583 | .201571 | .389147 | .204937 | -.060303 | fail |

N: shared recalibration largely preserves the original X22, but separating
statistics without updating weights/affine damages both histories, especially
T1. Read-only CPU audit `21787230` completed with exit `0:0`: real M00/M01/
M06/M07 examples route to the intended conditions and teachers; the wrapper is
exactly equivalent before calibration (maximum logit difference 0.0); every
network parameter remains bit-identical to X22. All conditions have training
support. Three examples per scenario reveal greatly expanded T1 logit ranges
(e.g. M06 approximately [-80.62,13.23] versus original [-4.09,-.46]). This is
consistent with disrupted feature scaling after replacing statistics, not
proof that every conditional-normalization design must fail. Detailed audit:
`three-directions-statistics-audit.json` and `audit_statistics.py`.

W: T1 block mean rises only .003249 against shared X22 while M01 loses .106724;
T5 primary also falls. A fixed weight midpoint does not preserve the two
parents' complementary capabilities. This rejects the registered alpha=.5
recipe, not all possible model-merging algorithms.

By the pre-registered stop rules, N/W do not advance to seeds1/2 or exposed
test-year evaluation. Their implementation and negative evidence are retained.
D used jobs `21787047/21787048` (T1 control/KD) and
`21787053/21787054` (T5 control/KD) each target all 3000 updates and full 2021
four-scene evaluation. No early training loss is used as an AP conclusion.

### D: T1 seed-0 pair

Both T1 jobs completed 3000 updates and all 3181 evaluation samples per scene,
with Slurm exit `0:0`. The no-KL student AP is
`.597544/.326218/.387318/.202660`; KD is
`.598244/.327507/.387521/.205879` for M00/M01/M06/M07.
KD's primary increment over the matched no-KL student is `+.001570`; its
primary delta against the retained four-teacher route is `+.000710`, with
worst scene delta `-.002396`. T1 therefore passes its fixed screen gate, but
the final cross-history decision also requires T5 (reported below).

The KD checkpoint occupies 57,885,172 bytes versus 231,564,112 bytes for the
four retained teacher checkpoints (ratio .249975). The no-KL student is also
one checkpoint and is already within the route's primary noninferiority
margin; storage reduction alone is not attributable to KD. T1 training times
were 2319.91 s (KD) / 2331.26 s (control), which do not establish a speedup given
shared-node execution. The mechanism's incremental AP gain is small and needs
the registered both-history/three-seed checks before any adoption claim.

T1 saved-checkpoint audit `21788864` completed with exit `0:0`. Both final
student files strict-load into the declared architecture, contain only finite
state tensors, record 3000 updates, match the reported parameter/file sizes,
and differ from the initial X22 state in 182 tensors. Checkpoint SHA256 and
full checks are retained in `three-directions-t1-checkpoints.json`; the audit
is reproducible with `audit_checkpoints.py` inside a CPU Slurm allocation.

## Final seed-0 decision

All 12 formal jobs completed with Slurm exit `0:0`. Full paired evidence is
`three-directions-screen.json`; the earlier partial file is an interim snapshot.
T5 control/KD jobs `21787053/21787054` each completed 3000 updates and all four
2021 scenes (3181 samples / 52,117,504 pixels each).

| T5 student | M00 | M01 | M06 | M07 |
| --- | ---: | ---: | ---: | ---: |
| No KL | .597558 | .374744 | .389190 | .197049 |
| Routed KD | .604209 | .380023 | .395268 | .202814 |

T5 KD improves primary over no-KL by .005707, but loses .005418 primary and
.013862 on M07 against the retained route. Both exceed the registered tolerances
(.005 primary / .010 any scene). T1 passes, T5 fails, so D does not advance.
Neither seed1/2 confirmation nor 2022/23 evaluation is required or performed.
Do not reinterpret the positive no-KL comparison as preserved expert capability.

T5 KD/control training cost is 5803.92/5746.66 seconds; peak allocated GPU memory
is 2,270,857,728/1,487,167,488 bytes. KD stores 58,936,324 bytes versus
235,771,792 for the four-teacher route, but control also stores one checkpoint.
These paired runs do not establish inference acceleration or statistical
significance. The original route already executes one selected teacher per input.

All three fixed recipes are closed by their registered seed-0 gates. This is
negative evidence for these configurations, not a universal rejection of their
method families. The retained expert route remains the stronger reference for
heavy block missingness; distillation shows an incremental teaching signal but
has not met the capability-preservation requirement. Chinese report:
`three-directions-results.md`.

T5 saved-checkpoint audit `21790289` completed with exit `0:0` (44 s).
Both files strict-load, have finite state and 3000-update metadata, and differ
from the initial X22 in 198 state tensors. SHA256 records are in
`three-directions-t5-checkpoints.json`. All four student training logs have
finite recorded numeric values and end at step 3000. Seven report/gate tests
passed in the final verification.

# Cross-history innovation campaign — 2026-09-06

Origin: main `8c9c072`. User authorizes autonomous design, implementation,
cluster submission, repair, and iteration until three supported directions
are found. This document is the design, execution plan, and live evidence log.

## Frozen problem and evidence rule

WSTS+ next-calendar-day active-fire proxy under controlled missingness.
Train 2016–2020, select on 2021, report fixed 2022/2023 tests. T=1 uses
Res18-U-Net (40 channels); T=5 uses the previously reproduced Res18-UTAE
(33 selected channels per day). Both evaluate identical event/target dates
using history adjustment six and M00/M01/M06/M07. T changes architecture and
feature selection too: this demonstrates transfer across existing settings,
not a pure history-length causal effect.

The paired target-date contract is exact. The upstream dataset sets
`skip_initial_samples = 6 - T`; the controlled wrapper then uses
`target_index = in_fire_index + skip_initial_samples + T`, which equals
`in_fire_index + 6` for both T=1 and T=5. Thus both settings evaluate the same
event-relative target dates; only their available history and model differ.
The authoritative B3 and archived B5 2021 summaries both contain exactly
3,181 samples and 52,117,504 evaluated pixels per scenario, ruling out a
history-dependent sample-count difference in the paired comparison.

Fresh matched controls and candidates start from corrected B3 (T1) or B5
(T5), run 3000 AdamW steps, lr=0.001, effective batch 64, identical seed,
augmentation, normalization, sampling, and loss. Final-step checkpoint only.
First screen seed=0; confirmation seeds=1,2 for candidates passing both T.
Primary metric: mean AP over M01/M06/M07. Also report M06/M07 block mean,
M00, individual corrupted AP, runtime and parameter count. A candidate counts
only when primary improvement is positive in both T settings across the
confirmation seed mean and each fixed test year, with no M00 regression
greater than 0.01. Seed-0 screening signal: primary >=0.005 in both T and
M00 guardrail. These are practical thresholds, not significance claims.
Seek >=0.02 average improvement for at least one direction, preserving the
user's earlier magnitude requirement. Do not count three hyperparameters of
one method as three directions. Keep all negative results. No selection using
2022/2023; if test evidence fails, mark failure and disclose adaptive reuse
before any later campaign rather than pretending a fresh unseen test.

## First candidates and falsifiable mechanisms

- **X1 context transport:** at encoder skip levels, replace missing-region
  features by learned residuals from normalized valid-region context pooled
  at multiple spatial resolutions. Zero-initialized residuals preserve the
  initial predictor. Spatial mask comes only from the controlled intervention.
  Distinct from rejected D2-RNC: retain standard convolutions; transport
  multi-scale feature content across holes, not input convolution rescaling.
- **X2 fire-weighted feature distillation:** frozen initial model sees clean
  training inputs; student matches normalized deep features primarily inside
  missing regions and near next-day positive targets. Auxiliary weight 0.05.
  Distinct from D1 output KL and failed ranking losses: intermediate spatial
  representation targets, with no extra inference parameters.
- **X3 latent transition head:** decoder predicts previous-day occupancy,
  conditional continuation, and new activity. Forecast marginalizes uncertain
  occupancy in missing regions, clamps occupancy to observed latest fire where
  valid, and receives latest-fire auxiliary supervision from clean train
  inputs. This is joint forecasting, not a separate frozen reconstruction
  pipeline. Conditional heads initialize to the original forecast.

Prior art: [partial convolutions](https://arxiv.org/abs/1804.07723),
[privileged multimodal segmentation](https://pubmed.ncbi.nlm.nih.gov/34633927/),
[DIS2 remote-sensing distillation](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/papers/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.pdf),
[wildfire reconstruction then prediction](https://arxiv.org/abs/2603.09042).
These establish related mechanisms, not novelty or quantitative support for
our implementations. A publication claim needs precise comparison to them.

## Closest-work and claim boundary

The closest task-specific work is Yang et al.'s 2026
[reconstruction-to-prediction framework](https://arxiv.org/abs/2603.09042),
which explicitly reconstructs a clean fire-history sequence before a separate
T=5 forecaster. Consequently, neither partial-observability wildfire
forecasting nor missing-fire reconstruction is a novelty claim here. X1 and
X3 instead test end-to-end, forecast-conditioned alternatives: X1 transports
valid latent context directly into spatial holes, while X3 marginalizes a
latent latest-fire state inside the forecasting head and clamps observations
where valid. Neither produces or supervises a standalone reconstructed fire
map, and both are evaluated under T=1 and T=5.

A targeted search of directly adjacent next-day methods found raster
single-/multi-day forecasting
([Lahrichi et al.](https://arxiv.org/abs/2502.12003)), query-based ignition-set
prediction ([WISP](https://arxiv.org/abs/2605.10298)), and the two-stage
reconstruction framework above, but did not identify an observed-state-clamped
survival/new-activity marginalization matching X3. This is a scoped positioning
observation, not an absolute priority claim; novelty language remains
conditional on broader review and positive matched evidence.

Missing-modality remote-sensing methods already perform learned compensation
and distillation, including
[DIS2](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html),
and recent teacher-student segmentation work already targets robustness with
little full-modality degradation
([RobustSeg](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.html)).
Thus feature distillation and consistency alone are not novelty claims. The
`local_consistency` experiment is explicitly an incremental, spatially
localized extension of retained D1; it can count only with a positive matched
ablation in both history settings. Original feature distillation has already
failed T1 and remains negative evidence.

## Minimal implementation plan

- [x] Add a separate `reproductions/cross_history/` package for the campaign.
  Reuse corrected resolver, normalization, raw evaluation corruptions, exact
  AP code and archived B5 checkpoint (explicit relocated path).
- [x] Implement shared T1/T5 encoder-decoder access and the three methods.
  Keep clean training targets only in the training loss, absent at inference.
- [x] Add one Slurm runner with train/eval modes, job-specific output path,
  committed source snapshot, fixed seed metadata, checkpoint and results.
- [x] Run targeted GPU one-batch checks inside Slurm for the initial methods
  and every subsequently added execution path, including original-vs-control
  output agreement and backward pass. No global tests.
- [x] Submit eight seed-0 experiments (control + three methods, both T).
  Inspect resource availability; prefer 10/20GB slices, at most 2x overflow
  unless a blocking prerequisite requires more. Revisit jobs pending >10min.
- [ ] Compare against fresh controls, diagnose failures, register any changed
  hypothesis before further experiments; confirm and test only passing recipes.
- [ ] Commit all progress and report three only when evidence satisfies above.

## Status

T1 seed-0 evidence is available for the first wave; cross-history support is
not established until the matched T5 screens finish. Historical B3/B5 and
D2/D13 controls are context, not proof of any new direction.

### Frozen candidate register

| Direction | Matched attribution control | Independent contribution? | T1 seed-0 | T5 seed-0 |
| --- | --- | --- | --- | --- |
| X1 context transport | fresh continuation; observable spatial route | yes | routed +.005954 | routed +.000874; cross-T reject |
| X1 frozen adapter | frozen initial B3/B5 plus fresh continuation | alternative X1 implementation | +.012151 vs frozen, -.013537 vs fresh; reject | cancelled after T1 adoption failure |
| X2 feature distillation | fresh continuation | yes | original and block-local rejected | running |
| X3 latent transition | fresh continuation | yes | +.011831 | -.009413; decoupled repair pending |
| X4 spatial risk weighting | fresh continuation | one training-objective contribution | routed +.003463 | routed -.001659; reject |
| X5 localized consistency | unchanged global D1 consistency | incremental method contribution | -.002294; reject | cancelled after T1 failure |
| X6 balanced corruption coverage | fresh continuation | sole corruption-rate tuning contribution | pending | pending |
| X7 missingness-conditioned residual experts | fresh continuation | no; overlaps archived D7-CRA | cancelled after overlap audit | cancelled after overlap audit |
| X8 counterfactual-impact consistency | unchanged global D1 consistency | incremental objective contribution | screen `21214685` | screen `21214686` |
| X1+X3 composition | fresh continuation / component ablations | no; interaction only | +.008200, below X3; reject | cancelled |

X7 was preregistered and implemented while the already submitted screens were
running, before seeing their outcomes. The subsequent archive audit below
invalidated its independence before a screen ran. X8 is a justified reopening
of quantitatively promising D5 under the new cross-history objective; the
register is otherwise closed until these results resolve. A failed row may
receive one mechanism-driven repair, but no unrelated direction is added merely
to accumulate positive experiments. Only rows with positive matched evidence
in both history settings can advance to seed confirmation.

Implementation: `357e475`, numerical mixture fix `8c654e2`. Shared raw
evaluation permits T5 only by an explicit opt-in; mainline T1 default stays.
Physical batch 16, accumulation 4 applies to every new control/candidate.
Existing full-batch D2/D13 numbers are context, not the matched comparison.

GPU smoke jobs: T1 `21210705`, T5 `21210706`, each checked control/context/
distill/transition in sequence on a 20GB H100 slice, 4 CPUs, 32GB host memory.
Earlier 10GB requests `21210683/21210684` were cancelled while pending because
no start estimate was available; 20GB had an approximately 11-minute estimate.
No model or dataset computation has been run on the login node.

New submissions export `WILDFIRE_SOURCE_COMMIT` so queued jobs archive the
submission-time revision rather than whatever HEAD exists when allocation
eventually begins. Every run still records the resolved commit in `commit.txt`.
After three T5 jobs completed training but exhausted 64GB host RAM during
batch-64 evaluation, new snapshots cap evaluation batch at 16 while leaving
training batch 64 unchanged. AP aggregation and samples are identical; only
evaluation memory and wall time change. Saved checkpoints recover old jobs
through evaluate-only rather than repeating training.

Both smoke jobs completed with exit `0:0`. All eight real-data combinations
completed one optimizer step, backward pass, checkpoint reload path, and one
M06 2021 inference. Initial maximum absolute output difference from the base
was exactly zero for control/context/distill and `2.38e-7` (T1) / `4.77e-7`
(T5) for the transition mixture. Peak GPU allocation at physical batch 16 was
0.70GB (T1) and 1.97GB (T5); formal screens therefore use physical/effective
batch 64 on 20GB slices to reduce wall time, identically for all variants.

Capacity is not the source of a large-model advantage. The base models contain
14,444,241 (T1) and 14,704,785 (T5) parameters. X1 adds 66,640 parameters in
either setting (about 0.46%/0.45%); its adapter variant trains only those
66,640. X3 adds only 51 parameters. Their composition adds 66,691. Risk,
distillation, and both consistency objectives add no inference parameters.

Next action: run eight 3000-step jobs (four methods x two T), compare the three
candidates with their fresh same-history controls, and stop failed directions
before confirmation.

Seed-0 screen jobs (all physical/effective batch 64, seed 0, 3000 steps,
20GB H100 slice, 8 CPU, 64GB host memory): T1 control/context/distill/
transition `21211219/21211220/21211221/21211222`; T5 equivalents
`21211223/21211224/21211225/21211226`. All were initially pending without a
start estimate. Recheck after ten minutes and change slice only when queue
evidence supports a faster start under the resource rule.

### First T1 screen

| Method | M00 | M01 | M06 | M07 | primary | primary delta | block delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh control | .585066 | .308313 | .370825 | .189912 | .289683 | — | — |
| context | .593755 | .304883 | .385249 | .193349 | .294494 | +.004811 | +.008930 |
| distill | .584336 | .286450 | .372542 | .194401 | .284464 | -.005219 | +.003103 |
| distill_block | .578880 | .271517 | .365712 | .191154 | .276127 | -.013556 | -.001936 |
| risk 3x | .583485 | .276321 | .377532 | .193596 | .282483 | -.007201 | +.005195 |
| transition | .584545 | .330915 | .377251 | .196377 | .301514 | +.011831 | +.006445 |

Transition passes the T1 gate. Context misses the all-corruption threshold by
.000189 because M01, which has no spatial hole, regresses .003430. Its declared
observable route uses the control for M00/M01 and context for M06/M07; routed
primary delta is +.005954 and clean is identical to control. This is the same
specialist-routing evidence convention already used by B2/B3.

Original distillation fails because its mask included complete FireDrop and
the target focus also applied to non-spatial examples: M01 drops .021863. The
registered `distill_block` revision applies feature matching only inside true
spatial holes and weights positive neighborhoods only within those holes. A
separate `risk` candidate upweights segmentation risk 3x inside spatial holes
while retaining the original loss elsewhere. These are direct responses to
the diagnosed failure and remain in the same controlled-missingness problem.
Static inspection of the pinned upstream `BaseModel.compute_loss` confirms
that `risk` uses the identical torchvision focal per-pixel term, alpha and
gamma; with no spatial hole its normalized objective is exactly the original
mean. Its only intervention is the declared 3x spatial-hole weighting.
The `distill_block` audit likewise confirms that its auxiliary weight is
identically zero whenever the true spatial-hole mask is empty; FireDrop-only
and clean samples retain the control objective. The frozen teacher consumes
the exactly paired clean tensor, while only the three deepest normalized
features inside the same spatial mask contribute to the auxiliary term.

The T1 block-local distillation revision nevertheless fails: primary delta
`-.013556` and block delta `-.001936`; X2 is rejected without another repair.
Risk 3x has block delta `+.005195`, but its observable routed primary delta is
only `+.003463`, below the `+.005` gate. This directional result registers the
single allowed X4 repair, `risk_strong`: identical code and loss with spatial
weight increased from 3x to 5x. There is no weight grid; X4 remains one tuning
contribution and is rejected if this fixed repair misses the gate.
The fixed T1/T5 5x jobs are `21213326/21213327`, pinned to commit `2385e76`
and using the minimum 10GB GPU slice supported by observed peak usage.
Before meaningful computation, the completed T5 3x result showed routed
primary `-.001659` and block mean `-.002489`, contradicting the stronger-weight
repair mechanism. Both 5x jobs were cancelled after 11 seconds and X4 closed;
no 5x result is claimed.

Queue correction: original T5 distill/transition `21211225/21211226` were
cancelled pending after >10 minutes. Same-resource replacements
`21211486/21211487` started immediately alongside T5 control/context
`21211223/21211224`.

T5 control/distill/transition completed all 3000 updates and saved checkpoints
but exhausted 64GB host RAM during the old batch-64 evaluation path. Their
batch-16 evaluate-only recovery jobs are `21213246/21213247/21213248`; no
training is repeated. T5 context completed its full evaluation directly.
Its absolute M00/M01/M06/M07 AP is
`.597931/.340972/.388529/.200548`; no matched delta is claimed until the fresh
control recovery finishes.

Recovered T5 fresh-control M00/M01/M06/M07 AP is
`.595552/.364196/.389710/.196743`. Context changes raw primary by `-.006867`
and block mean by `+.001312`; its declared spatial route changes primary by
only `+.000874`. Thus X1 full-model context fails the T5 `+.005` screen and is
not advanced. Across T1/T5, its routed mean primary delta is `+.003414`.

Recovered T5 transition AP is `.597599/.348957/.383617/.189836`: primary
delta `-.009413`, block delta `-.006500`, despite the T1 `+.011831`. X3
therefore does not advance in its original form. The single registered repair,
`transition_decoupled`, keeps the identical inference head and marginalization
but computes auxiliary state/conditional supervision from detached decoder
features. Auxiliary gradients update only the 51-parameter head, while the
main forecast loss remains end-to-end; this directly tests whether T5 failure
came from auxiliary distortion of the shared temporal representation.
Fixed T1/T5 decoupled jobs are `21213902/21213903`, pinned to commit
`7277cb8`; this is X3's only repair and uses the post-OOM evaluation cap.

Original T5 distillation AP is `.586798/.355377/.380389/.191319`, for primary
delta `-.007855` and block delta `-.007372`. Together with both failed T1
forms, this closes X2 as negative evidence.

Follow-up seed-0 screens: T1 `distill_block/risk` are
`21211826/21211827`; T5 equivalents are `21211828/21211829`. They use the
same 20GB slice and matched train/evaluation contract and were initially
pending behind the active first wave.
T5 risk completed at M00/M01/M06/M07
`.590760/.338343/.385910/.195566`, yielding raw primary `-.010277` and block
mean `-.002489`. T5 distill_block completed training but its old batch-64
evaluation exhausted host RAM; no recovery was submitted because T1 had
already rejected X2 on both primary and block metrics.

The registered fallback `local_consistency` transfers the mainline D1 idea
but localizes Bernoulli KL to the union of observed FireDrop/BlockDrop pixels
and weights it by clean-view teacher confidence. The student still uses the
exact D1 paired supervised loss, `0.5 * (clean + corrupt)`, and inference has
no extra parameters or clean view. A same-runner `global_consistency` control
ports D1 unchanged to both history settings; local consistency counts only if
it improves over that control, not merely over ERM. This tests an incremental
spatially targeted objective, not another architecture or general-purpose
hyperparameter search.

The registered composition `context_transition` combines X1 feature transport
with X3 transition marginalization without changing either component or its
loss weight. This is an interaction/scale experiment, not a fourth independent
contribution: it tests whether spatial context recovery and fire-state dynamics
are complementary and whether their total gain reaches the requested +.02.
It will be screened only after the component implementations pass a real-data
smoke check; a gain smaller than the better component will be treated as
negative interaction rather than retuned post hoc.

T1 composition AP is `.592994/.319796/.380838/.193017`, giving primary
`+.008200` and block mean `+.006559` versus fresh control. Because this is
smaller than X3 alone (`+.011831`), the registered interaction criterion fails.
T5 composition `21212156` was cancelled at 37 minutes to free its slice; a
second-history result cannot rescue a composition already non-complementary in
T1, and the composition is not counted as an independent direction.

After X1/X2/X4 and the composition closed, X6 registers one data-coverage
direction without changing the research problem. Independent FireDrop and
BlockDrop probabilities increase from 0.3 to 0.5, changing the expected
clean/fire/block/both mix from 49/21/21/9% to 25/25/25/25%. Model, focal loss,
3000-step optimizer contract, effective batch, and evaluation remain matched.
This is the campaign's single corruption-rate tuning contribution; probability
0.5 is fixed by symmetry and will not be searched.
Fixed T1/T5 X6 screens are `21214127/21214128`, using 10GB slices and pinned
to implementation commit `b9c02f2`.

Composition smoke jobs T1/T5 `21212079/21212081` completed successfully in
25/49 seconds. They covered a real batch, backward pass and inference; initial
maximum output deviations were `2.38e-7/4.77e-7`. Full seed-0 composition
screens are `21212155/21212156` with the same 3000-step contract.

Local-consistency jobs `21211876/21211877` remained pending for ten minutes
with an estimated 04:17 start. Test-only probes of 10/20/40GB slices and
75/90/180-minute limits all returned 04:28, so the original 20GB jobs were
retained; changing resources would not start sooner.

Those two jobs were subsequently cancelled before execution after an
attribution audit found that they used corrupt-only supervision rather than
D1's paired clean/corrupt supervised objective. Commit `a22c589` corrects the
objective and adds the unchanged global-D1 control. Minimal paired-path smoke
jobs are T1-global `21212258` and T5-local `21212272`; full jobs will use only
the corrected snapshot after these pass. No result from the cancelled jobs is
eligible evidence.

Both corrected smoke jobs completed with exit `0:0`, exact initial output
agreement, a real paired clean/corrupt backward step, and inference. Corrected
seed-0 full jobs, all pinned to commit `92beb6c`, are global T1/T5
`21212411/21212412` and localized T1/T5 `21212413/21212414`.

T1 local consistency completed at M00/M01/M06/M07
`.595788/.297506/.373543/.188356`. Against its unchanged global-D1 control,
the deltas are `+.004449/+.003244/-.004912/-.005212`: primary `-.002294`
and block mean `-.005062`. Localizing KL to the input hole therefore shifts
performance in exactly the wrong spatial scenarios, supporting the hypothesis
that future forecast impact need not coincide with the missing input support.
X5 is rejected. T5-local `21212414` was cancelled at 50:31 while training;
T5-global remains active because it is also X8's required attribution control.

The registered `context_adapter` variant freezes the entire initial forecaster
and trains only the zero-initialized context-transport layers. Because every
transport residual is multiplied by the observed spatial-hole mask, this
single checkpoint is exactly the original control whenever no spatial hole is
present; it removes the two-checkpoint routing caveat and isolates the X1
mechanism. It is an alternative implementation of X1, not an additional
contribution. Its fixed recipe uses the same 3000 steps and optimizer settings.
Its attribution control is the same frozen initial checkpoint evaluated by the
cross-history runner with zero update steps; comparison to the fresh full-model
continuation is a required total-system adoption check. The adapter can be
attributed against frozen B3/B5, but it counts toward the goal only if it also
has positive primary delta against the fresh 3000-step control in both T
settings; a cheaper but weaker model does not satisfy the performance goal.
Adapter smoke jobs T1/T5 `21212176/21212177` completed with exit `0:0` and
exact initial output agreement. Peak allocation was only 0.51/0.86GB at batch
16, so full seed-0 screens `21212420/21212421` use the minimum 10GB H100 slice
at batch 64 and are pinned to commit `e906291`.
After T5 adapter remained pending for over ten minutes with a 02:41 estimate,
test-only 10/20GB and 2/3-hour probes all returned 02:59. The existing 10GB
job was retained because it starts earlier and uses fewer resources.

T1 adapter `21212420` completed at M00/M01/M06/M07
`.558318/.276890/.362162/.189388`. It improves primary AP by `+.012151`
and block AP by `+.018227` over the frozen B3 attribution control, with exact
unchanged M00/M01 as designed. However, it is `-.013537` primary and
`-.004593` block below the fresh 3000-step continuation; M00 is also
`-.026747` lower. It therefore demonstrates that X1 itself learns useful
spatial corrections, but fails the preregistered total-system adoption rule.
T5 adapter `21212421` was cancelled at 19:14 to release its slice because no
T5 result could rescue a method already ineligible on T1. X1 is closed.

Frozen attribution reference T1 `21212205` completed with M00/M01/M06/M07 AP
`.558318/.276890/.343010/.172086`. T5 reference `21212206` failed before a
complete result because a DataLoader worker exceeded the requested 32GB host
RAM; the traceback was a Slurm cgroup OOM, not a GPU or model error. Replacement
`21212423` keeps the minimum 10GB GPU and increases only host RAM to the
already-used 64GB, with zero training steps and a pinned source revision.
The four T1 values match the source B3 `results-2021/summary.json` exactly at
full stored precision, independently confirming that the new wrapper and AP
path introduce no evaluation drift before adapter attribution.
For scale, the T1 fresh control improves primary AP over frozen B3 by
`.025688`; the adapter must recover that gap before it can count as a net
performance contribution, regardless of its parameter efficiency.
The archived B5 reference fixes the expected T5 values at
M00/M01/M06/M07 `.557715/.288461/.336086/.158021`. Replacement `21212423`
completed all four at a maximum absolute AP difference of `5.50e-7`, accepted
as GPU numerical equivalence rather than requiring inappropriate bit identity.

While GPU jobs run, `reproductions/cross_history/compare.py` provides the
minimal downstream result path. It pairs summaries by history/seed/year,
computes individual, primary, block, clean-guardrail and multi-run statistics,
and can apply the declared spatial-specialist route. Its gates require explicit
T1+T5 coverage: seed-0 2021 screening, seeds 0/1/2 2021 confirmation, and all
four history/year cells on the fixed 2022/2023 tests. A single-history input
can no longer be reported as a cross-history pass. It separately reports
`magnitude_target_met` at mean primary delta >=.02; this is a campaign-level
requirement for at least one direction, not a gate imposed on all three.
`run.py --evaluate-only`
already performs immutable 2022/2023 evaluation from a selected checkpoint.

X7 used two zero-initialized late residual experts selected by the observed
corruption masks: one for global FireDrop evidence and one for a spatial data
hole. A shared 3x3 decoder bottleneck receives decoder features plus both local
masks, so each expert can adjust the full forecast while knowing where evidence
was removed. Neither expert activates on a clean sample, and the untrained
model is exactly the base forecast. This is distinct from X1 encoder transport,
X3 fire-state marginalization, and loss/data-distribution candidates X5/X6.
The same small module was implemented unchanged for both histories at
`76cd083`; real-data one-batch T1/T5 smoke jobs `21214411/21214412` completed
in 29/24 seconds with exact initial equivalence, successful backward and
inference, and peak GPU allocation of 0.64/1.61GB. Before the formal screens
produced evidence, an archive audit found that its 3x3 late residual, explicit
reliability maps, whole-forecast correction and jointly trained base materially
repeat archived D7-CRA; two heads instead of one and removal of CIWC are not an
independent contribution. Formal T1/T5 jobs `21214506/21214507` were therefore
cancelled at 21/0 seconds. X7 is retained only as an overlap-audit record.

X8 revisits archived D5 counterfactual-impact weighted consistency because the
current goal changes its relevant decision boundary. D5 had corrected T1
primary delta `+.015468` over D1-ERM, driven by M01 `+.043494`, and stopped
only because the old campaign required every candidate to reach `+.020`; it
was never tested at T5. Unlike X5-local, which assumes forecast impact lies on
the input invalidity support, X8 weights clean-to-corrupt Bernoulli KL by the
detached absolute change in predicted probability at each future pixel and
normalizes impact per sample. It adds no inference parameters. X8 must improve
over the unchanged global-D1 consistency control, not merely ERM, in both
histories to count as a new incremental objective. The exact fixed 0.1 recipe
is ported in `35695ec`; T1/T5 real-batch smoke jobs are `21214598/21214599`.
Both completed in 22/24 seconds with exact initial equivalence, one successful
paired backward step and inference. Formal T1/T5 screens are
`21214685/21214686`. T1 uses a 10GB slice; T5 uses 20GB because the same
batch-64 paired path measured just over 10GB, preserving the matched physical
batch rather than changing BatchNorm behavior to fit the smaller slice.

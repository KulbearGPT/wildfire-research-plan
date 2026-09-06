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
greater than 0.01. An incremental module must pass this against its closest
attribution control, and its routed final system must also pass against fresh
ERM before it counts as an adopted improvement. Seed-0 screening signal:
primary >=0.005 in both T and
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
| X2 feature distillation | fresh continuation | yes | original and block-local rejected | original rejected; block-local stopped |
| X3 latent transition | fresh continuation | yes | original +.011831; decoupled -.045065 | original -.009413; repair cancelled |
| X4 spatial risk weighting | fresh continuation | one training-objective contribution | routed +.003463 | routed -.001659; reject |
| X5 localized consistency | unchanged global D1 consistency | incremental method contribution | -.002294; reject | cancelled after T1 failure |
| X6 balanced corruption coverage | fresh continuation | sole corruption-rate tuning contribution | +.005203 | +.002426 with M00 -.017601; reject |
| X7 missingness-conditioned residual experts | fresh continuation | no; overlaps archived D7-CRA | cancelled after overlap audit | cancelled after overlap audit |
| X8 counterfactual-impact consistency | unchanged global D1 consistency; fresh ERM adoption | incremental FireDrop specialist | incrementally reliable; ERM adoption fails 2022 | incrementally reliable and ERM-positive |
| X9 spatial-impact FiLM | fresh continuation with spatial route | yes; context-conditioned decoder modulation | routed -.000552; reject | cancelled after T1 failure |
| X10 forecast-aware dynamic inpainting | fresh continuation with spatial route | yes; typed input restoration optimized by forecast loss | 2022/23 -.001356/-.000347; reject cross-history | 2022/23 +.004783/+.002139; T5-only positive |
| X11 identifiable dynamic restoration | X10 forecast-only inpainting | incremental reconstruction objective | -.004862 vs X10; reject | cancelled after T1 failure |
| X12 counterfactual FireDrop specialist | fresh continuation; plain specialist ablation if screen passes | one specialist-training contribution | routed +.001589; reject | cancelled after T1 failure |
| X13 normalized diffusion inpainting | same frozen control checkpoint with spatial route | yes; parameter-free typed spatial propagation | routed -.024799; reject | cancelled after T1 failure |
| X14 BlockDrop specialist continuation | fresh continuation with spatial route | one block-specialization training contribution | reliable: +.005467/+.001416/+.006792 in 2021/22/23 | reliable: +.016685/+.011528/+.005485 in 2021/22/23 |
| X15 distance-to-evidence prompting | fresh continuation with spatial route | yes; continuous missing-geometry encoder prompt | routed -.002072; reject | cancelled after T1 failure |
| X16 block-specialized dynamic restoration | X14 specialist plus fresh-control total check | no; restoration loses to X14 | -.000452 vs X14; reject | cancelled after T1 attribution failure |
| X17 severity-factorized block specialists | X14 mixed-severity specialist plus fresh ERM | no; may strengthen/supersede X14 | 3-seed +.007392 vs ERM; block +.002887 vs X14 | 3-seed +.017792 vs ERM; block +.001662 vs X14; heldout running |
| X18 block-specialized context transport | X14 specialist plus fresh ERM | yes only if positive vs X14 | +.013205 vs ERM; +.004405 vs X14 | +.006524 vs ERM but -.000922 vs X14; reject |
| X19 severity-conditioned latent adapters | X14 specialist; X17 two-checkpoint upper bound; fresh ERM | yes only if positive vs X14 | +.012193 vs ERM; block +.005090 vs X14 | seed-0 running |
| X20 ERM-anchored impact consistency | fresh ERM; X8 diagnoses the repair | same impact-consistency family as X8 | -.003678; reject | cancelled after T1 failure |
| X21 first-layer reliability calibration | D4/D12 and fresh ERM | no; pre-run closest-work reject | not run | not run |
| X22 cosine-decayed ERM | fresh constant-LR ERM | sole optimizer/tuning contribution | seed-0 submitted | seed-0 submitted |
| X23 impact-consistent BlockDrop specialist | X14 BlockDrop specialist plus fresh ERM | incremental counterfactual-impact objective; inference unchanged | implementation ready | implementation ready |
| X1+X3 composition | fresh continuation / component ablations | no; interaction only | +.008200, below X3; reject | cancelled |

X7 was preregistered and implemented while the already submitted screens were
running, before seeing their outcomes. The subsequent archive audit below
invalidated its independence before a screen ran. X8 is a justified reopening
of quantitatively promising D5 under the new cross-history objective. X9 was
registered only after X6 closed and targets the remaining block-specific gap;
X10 was registered before either X9 screen started and tests a distinct
input-space mechanism rather than a revision selected from X9 results. The
register is otherwise closed until these results resolve. A failed row may
receive one mechanism-driven repair, but no unrelated direction is added merely
to accumulate positive experiments. Only rows with positive matched evidence
in both history settings can advance to seed confirmation.

X23 was registered while X17/X19/X22 were still unresolved. It is the smallest
evidence-driven reuse of X8: train only on spatial BlockDrop examples exactly
as X14 does, pair each corrupt view with a no-gradient clean view, and add the
same fixed `0.1` counterfactual-impact-weighted Bernoulli KL. Unlike X8's
FireDrop specialist, the missing support now matches X14's successful spatial
regime. The predictor and inference route are unchanged, so attribution is
strictly X23 versus X14; adoption additionally requires the routed result to
beat fresh ERM in both histories. There is no consistency-weight sweep. The
seed-0 gate remains primary `>= +.005` versus X14 in both T settings with the
M00 guardrail; failure in either history closes X23 before confirmation.
The T1/T5 real-data smoke jobs are `21228439/21228440`; dependency-gated
seed-0 screens are `21228441/21228442`, all pinned to `a89c06f`. Queue probes
placed the 10GB smoke profile about five minutes earlier than 20GB, so both
smokes and T1 formal use 10GB. T5 formal uses 20GB because the identical
paired clean/corrupt path previously exceeded 10GB at physical batch 64.
The formal jobs retain 3000 steps and the X14 sampling recipe; no computation
was performed on the login node.

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

The decoupled T1 repair completed at M00/M01/M06/M07
`.472336/.248200/.311505/.174151`, changing primary by `-.045065`, block
mean by `-.037541`, and M00 by `-.112730` versus fresh control. Although the
initial mixture was equivalent, removing auxiliary gradients from the decoder
left the 51-parameter transition head to move under the forecast mixture and
produced severe degradation rather than the intended T5 stabilization. This
falsifies the registered repair. T5 job `21213903` was cancelled at 37:33;
X3 is closed without a weight search.

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

T1 X6 completed at M00/M01/M06/M07 `.587003/.319053/.373156/.192450`.
Every scenario improves over fresh continuation; primary delta is `+.005203`,
block mean `+.002434`, and M00 `+.001937`. It passes the preregistered T1
screen narrowly and remains eligible pending the unchanged T5 job `21214128`.

T5 X6 completed at M00/M01/M06/M07 `.577951/.375596/.383052/.199280`.
Raw primary is only `+.002426`; M00 falls `-.017601`, M06 falls `-.006658`,
and block mean is `-.002060` versus fresh continuation. It therefore fails
both the T5 signal threshold and clean guardrail. The positive M01 change does
not justify a corruption-probability search or post-hoc specialist because its
single-scenario routed primary would be only `+.003800` at T5. X6 is closed.

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
T5-global was retained because it is also X8's required attribution control.

T5 global-D1 control `21212412` completed at M00/M01/M06/M07
`.609360/.357955/.391061/.196798`. This is the frozen attribution reference
for X8 at T5; its own change versus fresh ERM is not an X8 contribution.

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
`submit_heldout.py` closes the remaining handoff without adding a workflow
framework: it refuses to act unless a saved comparison reports
`confirmation_pass=true`, accepts the selected training result directories,
and emits the fixed 2022/2023 evaluate-only jobs. It is dry-run by default;
`--submit` is the sole state-changing switch. This keeps heldout data out of
selection while avoiding manual construction of 24 commands for a six-pair,
three-seed confirmation set.

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

T1 X8 completed at M00/M01/M06/M07 `.591929/.315978/.379551/.193669`.
Against global-D1, all four deltas are nonnegative:
`+.000590/+.021716/+.001096/+.000101`; primary is `+.007638` and block mean
`+.000599`. The gain is concentrated in complete FireDrop, matching archived
D5's diagnosis, while clean and block performance are preserved. X8 passes
the T1 screen and remains eligible pending unchanged T5 job `21214686`.

T5 X8 `21214686` completed all 3000 optimizer steps and wrote its 57MB final
checkpoint, then the old job reached the 64GB host-memory cgroup limit during
evaluation. This does not invalidate training. Evaluate-only recovery
`21216721` loads that exact checkpoint, uses batch 16 on a 10GB slice, and
does not repeat or change any optimizer step.

Recovery completed at M00/M01/M06/M07 `.606770/.373352/.386005/.192072`.
Against global-D1, raw primary is `+.001871`: M01 improves `+.015397`, while
M06/M07 change `-.005056/-.004727` and M00 changes `-.002590`. This misses the
raw T5 screen. However, archived D5 and the T1 result had already identified
counterfactual-impact weighting as a complete-FireDrop specialist, and the
FireDrop state is directly observable from the declared reliability mask.
The single mechanism repair therefore freezes a two-checkpoint route: use X8
only for M01 and global-D1 for M00/M06/M07. Its effective primary delta is
`+.007239` at T1 and `+.005132` at T5, with exact zero clean/block changes;
combined seed-0 mean is `+.006185`. Because this route was formalized after
reading seed-0 T5, it is explicitly adaptive and cannot be called reliable
until prospective seeds and held-out years pass unchanged.

Fixed confirmation jobs, all pinned to route implementation commit `ea6c759`,
are seed 1 T1 global/X8 `21217000/21217001`, T5 global/X8
`21217002/21217003`; seed 2 equivalents are
`21217004/21217005/21217006/21217007`. T1 uses 10GB slices and T5 uses 20GB;
all retain physical batch 64, 3000 steps, and batch-16 evaluation.

The prospective T1 seed-1 global/X8 M01 AP is `.308054/.319033`, and seed 2 is
`.316577/.319691`. The fixed FireDrop route therefore improves primary AP by
`+.003660` and `+.001038`, respectively, with exact routed M00/M06/M07 deltas
of zero. Together with seed 0's `+.007239`, all three seeds are positive and
their mean is `+.003979`. X8 passes T1 confirmation without changing its
recipe. Both T5 prospective pairs remain required for the cross-history
confirmation decision at this intermediate checkpoint.

The completed T5 prospective pairs preserve the positive effect: the
three-seed routed primary mean is `+.005834`. The authoritative six-pair
comparison is
`cross-history-analysis/x8-impact-fire-route-confirmation.json`; across all
six runs its mean is `+.004906` (population standard deviation `.002429`),
worst routed clean delta is exactly zero, and `confirmation_pass=true`.
Consequently, the fixed 2022/2023 evaluation opened without changing any
checkpoint or route. Twenty-four minimum-10GB evaluate-only jobs cover both
methods, histories, three seeds, and two years: contiguous job range
`21218965`--`21218988`. Their exact commands are recorded in
`cross-history-analysis/x8-heldout-jobs.txt`. Heldout results remain unread
until completion and are not used to alter this recipe.

All 24 heldout jobs completed successfully. The final authoritative artifact
is `cross-history-analysis/x8-impact-fire-route-final.json`. Three-seed primary
deltas for T1 are `+.003979/+.004533/+.004699` in 2021/2022/2023; T5 deltas
are `+.005834/+.003820/+.005231`. Every history/year cell is positive, routed
M00 and block deltas are exactly zero, `confirmation_pass=true`,
`heldout_pass=true`, and `goal_evidence_pass=true` against its frozen D1
attribution control. The 18 matched rows have overall mean `+.004683` and
population standard deviation `.004062`. X8 is therefore a reliable
incremental objective effect. It does not meet the separate +.02 magnitude
target and is not represented as doing so.

A later completion audit additionally compared the actually routed X8 system
against fresh ERM, using the already completed X8, global-D1, and ERM summaries
for all 18 history/seed/year cells. T1 primary deltas are
`+.003841/-.002114/+.002570` in 2021/2022/2023, while T5 deltas are
`+.016981/+.015578/+.010831`. The T1 2022 regression makes
`heldout_pass=false`, despite an overall mean `+.007948`. Thus X8 remains valid
positive module-attribution evidence but is not by itself an adopted
cross-history improvement toward the three-direction goal. Artifact:
`cross-history-analysis/x8-impact-fire-route-vs-erm-final.json`.

A stricter no-training deployment audit also removed global consistency from
the non-FireDrop scenarios: X8 supplies only M01, while fresh ERM supplies
M00/M06/M07. The three-seed primary deltas are T1
`+.002870/-.001856/+.001999` and T5
`+.011074/+.011713/+.007816` for 2021/2022/2023. Thus T1/2022 still regresses;
the failure is intrinsic to the old X8 FireDrop checkpoint rather than merely
its global-consistency fallback. This closes recomposition of old checkpoints
and leaves X20's ERM-anchored training repair as the only active member of the
impact-consistency family.

X9 spatial-impact FiLM addresses the failure revealed by X5-local: the future
forecast pixels affected by a missing input block need not lie inside that
block. At the final decoder resolution, it pools feature context only over
observed locations and combines it with the observed missing fraction. A
zero-initialized 832-parameter MLP produces per-channel scale and bias that
modulate the full decoder map only when a spatial hole exists. The initial
function is exactly the base model; M00 and complete FireDrop bypass the module.
Its declared evaluation is therefore the observable spatial route (candidate
for M06/M07, fresh continuation for M00/M01). This differs from X1's residual
transport inside encoder holes, D12's local fixed channel prompts, and archived
D7's forecast-logit residual. Implementation is `a39fc53`; real-data T1/T5
one-batch smoke jobs `21217053/21217054` completed in 22/26 seconds on 10GB
slices. Both reported exact initial equivalence, a successful backward update,
and inference; peak allocation was 0.61/1.58GB. Fixed seed-0 3000-step screens
are T1/T5 `21217161/21217162`, using the same physical batch 64 and minimum
10GB slices, pinned to `a39fc53`.

X10 forecast-aware dynamic inpainting targets the part of M06/M07 that neither
fire-map reconstruction nor decoder calibration models explicitly: the block
also removes dynamic environmental fields. For every history day, a compact
dilated CNN receives the corrupt typed input, its valid-region channel means,
static local context, and the spatial mask. It predicts corrections only for
the 18 T1 or 11 T5 retained dynamic non-fire channels and applies them only
inside the missing block. Observed values, static variables, active-fire
channels, M00, and complete-FireDrop inputs are unchanged by construction. The
zero-initialized final layer makes the initial predictor exactly the base.
There is no reconstruction coefficient: the downstream forecast loss decides
which physically typed values are useful, avoiding an image-fidelity tuning
branch.

The bounded claim is forecast-aware restoration of missing environmental
drivers, not novelty of masked reconstruction itself. Yang et al.'s
[wildfire reconstruction pipeline](https://arxiv.org/abs/2603.09042) and its
[official implementation](https://github.com/LS-Wireless/Robust-Wildfire-Forecasting)
reconstruct fire maps before a separate forecaster and assume environmental
fields remain observed. General masked spatiotemporal pretraining is covered by
[STD-MAE](https://arxiv.org/abs/2312.00516), while
[DIS2](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html)
compensates missing remote-sensing modalities in latent space. X10 differs in
the acted-on object (typed wildfire drivers), its observation-clamped input
correction, and direct forecast supervision. A reviewer-style pre-run audit is
`Accept with Revisions, pending the validation experiment`: the main risk is
incremental novelty, defended only if the same fixed module improves routed
M06/M07 performance in both T settings. Its implementation is intentionally
small (33,202 T1 / 28,939 T5 parameters); the decisive experiment is the same
seed-0 spatial-route screen used for X9, followed by unchanged confirmation
and heldout gates rather than an imputation benchmark or coefficient sweep.
Implementation commit is `52cf5b8`; T1/T5 real-data one-step smoke jobs are
`21217300/21217301` on minimum 10GB slices. Formal seed-0 screens
`21217349/21217350` have strict `afterok` dependencies on those smokes, so a
broken path cannot consume a training allocation. T1 requests 10GB; T5 uses a
20GB slice because applying the input correction across five frames increases
activation memory, while retaining the frozen physical batch 64.
Both smokes completed in 21/26 seconds with exact initial equivalence,
successful backward/inference, and peak allocations of 0.84/2.79GB. Their
dependencies released both formal screens, which started without queue delay;
the T5 training path currently peaks at 10.49GB, validating the 20GB request.

X9 T1 completed at M00/M01/M06/M07
`.589071/.312300/.371126/.187955`. Under its declared spatial route, only
M06/M07 count and change by `+.000300/-.001957`; block mean is `-.000828` and
primary is `-.000552` versus fresh control. The modulation mechanism is thus
beaten by its matched baseline rather than merely missing a threshold. X9 is
rejected, and its still-running T5 screen `21217162` was cancelled at 28:17
because no second-history result could restore a direction that must improve
both histories.

X10 T1 completed at M00/M01/M06/M07
`.593509/.309703/.383099/.199782`. The fixed spatial route uses the fresh
control for M00/M01 and yields M06/M07 deltas `+.012274/+.009870`, block mean
`+.011072`, and primary `+.007381`. It passes the T1 seed-0 screen with exact
routed clean preservation. The unchanged T5 job `21217350` remains the only
missing screen evidence before X10 can advance.

T5 job `21217350` completed all 3000 steps, saved its 57MB checkpoint, and
evaluated M00/M01/M06 before the 64GB host cgroup killed it during M07. The
available AP values are `.599411/.361960/.397225`; training is valid but no
screen decision is made without M07. Evaluate-only recovery `21219331` loads
that exact checkpoint, uses the minimum 10GB GPU slice, batch 16, three workers,
and 128GB host RAM. It changes no training state and resolves the sole blocking
metric without repeating 3000 optimizer steps.

Recovery completed at M00/M01/M06/M07
`.599411/.361960/.397225/.206876`. Against fresh T5 control, the spatial-route
M06/M07 deltas are `+.007515/+.010134`, block mean `+.008824`, and primary
`+.005883`, with exact routed M00/M01 preservation. The seed-0 cross-history
artifact `cross-history-analysis/x10-dynamic-inpaint-route-seed0.json` reports
T1/T5 primary `+.007381/+.005883`, combined mean `+.006632`, and
`screen_pass=true`.

Prospective confirmation jobs use the identical source `52cf5b8`, batch 64,
3000 steps, and seed values 1/2. T1 control/X10 are
`21219734/21219735` and `21219739/21219740`; T5 pairs are
`21219736/21219737` and `21219741/21219742`. T1 and T5 controls use minimum
10GB slices. T5 X10 uses 20GB GPU and 128GB host memory, based solely on the
observed training peak and post-training cgroup failure; no scientific setting
changes.

X11 is preregistered while the X10 confirmation jobs wait for allocation. It
tests whether X10's missing-driver correction becomes more reliable when the
latent correction is identifiable from the paired clean training input. X11
uses exactly the X10 architecture and forecast loss, and adds a `0.002`-weight
Smooth-L1 objective only on dynamic non-fire values inside the synthetically
removed spatial block. Observed pixels, static fields, active-fire fields, and
the inference graph are unchanged; M00/M01 still bypass the restoration by
construction. The coefficient is fixed before any X11 result and is intended
to keep reconstruction auxiliary to the roughly `0.003`--`0.007` forecast
loss observed in X10 rather than start a tuning sweep. Attribution is X11
versus matched X10; usefulness for the final system additionally requires a
positive routed delta versus fresh control in both T settings. One T1 and one
T5 one-step smoke will precede the seed-0 screens, with no broad test suite.
T1/T5 smoke jobs `21220222/21220223` completed successfully with exact initial
equivalence, backward update, checkpoint reload, and real-data inference. Their
peak GPU allocations were `0.93/3.11GB`. Fixed seed-0 screens
`21220298/21220299` use 10/20GB slices respectively, batch 64, and 3000 steps.
T1 completed at M00/M01/M06/M07 `.589165/.295700/.374371/.193925`.
Against matched X10, the spatial-route M06/M07 deltas are
`-.008728/-.005857`, block mean `-.007293`, and primary `-.004862`.
The auxiliary reconstruction constraint therefore harms the exact forecast
metric it was intended to improve; X11 is rejected and T5 job `21220299` was
cancelled at 25:46. The comparison artifact is
`cross-history-analysis/x11-reconstruction-vs-x10-t1.json`.

X12 targets the separate magnitude requirement through the dominant complete
active-fire-history failure. It continues the same initial model for the same
3000 steps using FireDrop on every corrupt branch, no BlockDrop, and X8's fixed
counterfactual-impact consistency objective; evaluation routes the candidate
only to observable M01. The recipe and `0.1` consistency coefficient are fixed
before results. If and only if it passes both seed-0 history screens, a plain
FireDrop-specialist continuation will isolate the consistency term from the
specialization schedule before confirmation. This can count as at most one
training-strategy contribution, not as architectural novelty.

A pre-run idea audit rates this `Accept with Revisions, pending the validation
experiment`: effectiveness and missingness robustness have high mechanism-based
potential because retained R1 improved routed M01 substantially and X8 is
positive across histories, while novelty is deliberately bounded. Modality
dropout is established in missing-input learning, including
[Lau et al.](https://arxiv.org/abs/1908.06683) and
[Woo et al.](https://ojs.aaai.org/index.php/AAAI/article/view/25378); recent
sequential modality dropout also combines dropout with optional reconstruction
([Yang and Zhang](https://arxiv.org/abs/2608.10240)). Therefore X12 is justified
as a cheap, falsifiable magnitude probe for this wildfire failure regime, not a
general missing-modality method claim. Failure modes are saturation of M01 from
the already robust B3 initialization and loss of useful mixed-corruption replay.
T1/T5 smoke jobs `21220240/21220241` completed successfully with exact initial
equivalence and peak GPU allocations `0.97/2.70GB`. Fixed seed-0 screens
`21220300/21220301` use the same 10/20GB resource policy and 3000-step protocol.
T1 completed at M00/M01/M06/M07 `.588671/.313079/.324425/.134607`.
Under the declared FireDrop route only M01 counts: it improves `+.004767`, so
primary improves just `+.001589`, far below both the `+.005` screen and the
campaign `+.02` magnitude target. The already robust initialization is
saturated rather than rescued by full specialization. X12 is rejected without
the conditional plain-specialist ablation, and T5 job `21220301` was cancelled
at 25:46. The comparison artifact is
`cross-history-analysis/x12-fire-specialist-t1.json`.

The already available seed-0 X8/X10 specialists were also composed against one
common fresh control using `compose_routes.py`: X8 supplies M01, X10 supplies
M06/M07, and control supplies M00. T1/T5 primary deltas are
`+.009936/+.008935`, with cross-history mean `+.009436`. The authoritative
artifact is `cross-history-analysis/x8-x10-route-seed0.json`. This is a useful
final-system route but does not meet the `+.02` magnitude target and is not an
independent contribution; it quantitatively rules out satisfying that target
by merely adding the two current seed-0 gains.

X10's prospective T1 seed-1 spatial-route primary delta is `+.000630`; seed 2
is `-.004452`. Together with seed 0, the three-seed mean remains positive at
`+.001186` (block mean `+.001780`). This satisfies the frozen mean-positive T1
confirmation condition but exposes substantial seed variance. The partial
artifact is `cross-history-analysis/x10-t1-confirmation-partial.json`; no
cross-history confirmation decision is made until both T5 pairs complete.

Both T5 pairs then completed. Seed-1/2 routed primary deltas are
`+.002198/+.020875`; with seed 0, the T5 three-seed mean is `+.009652` and
block mean is `+.014478`. The authoritative six-pair artifact
`cross-history-analysis/x10-dynamic-inpaint-route-confirmation.json` reports
overall primary `+.005419`, population standard deviation `.007892`, exact
routed clean preservation, and `confirmation_pass=true`. The large variance is
retained as a limitation rather than hidden by the mean. The frozen 2022/2023
evaluation therefore opened: 24 evaluate-only jobs `21221885`--`21221908`
cover controls/candidates, both histories, three seeds, and both years using
minimum 10GB slices and source commit `52cf5b8`.
All 24 jobs completed successfully. The final artifact
`cross-history-analysis/x10-dynamic-inpaint-route-final.json` reports T1
2022/2023 three-seed primary deltas `-.001356/-.000347`, so
`heldout_pass=false` and X10 is not a reliable cross-history direction. T5
remains positive at `+.004783/+.002139` on those years. Across all 18 rows the
mean is `+.002676` with population standard deviation `.007657`; this supports
a T5-specific observation but not the required T1/T5 generalization claim.

The full X8(M01)+X10(M06/M07) route was also recomputed against one common
fresh control in `cross-history-analysis/x8-x10-route-final.json`. Its T1
2022 primary is `-.003212`, so the composition also has
`heldout_pass=false`; overall mean `+.008279` does not meet `+.02`. This route
is rejected as a final cross-history system and remains non-independent.

X13 addresses a concrete limitation of X10 rather than adding another loss:
X10's two convolutions cannot transport observed values to the centre of a
25%/50% missing block. X13 uses parameter-free normalized spatial averaging at
successively larger radii to propagate only observed dynamic non-fire fields
into the hole, while clamping every observed value and leaving static and fire
channels unchanged. M00/M01 bypass it exactly. The first decisive experiment
loads the existing seed-0 fresh-control checkpoints without training and routes
X13 only to M06/M07. It advances only if primary gain is at least `+.005` in
both histories; otherwise it is rejected before any continuation.

The pre-run idea verdict is `Accept with Revisions, pending the validation
experiment`. Its strengths are a direct large-hole mechanism, no trainable
parameters, and identical T1/T5 behavior. Its novelty boundary is narrow:
normalized/partial convolution and image inpainting are established, so the
claim can only be observation-clamped propagation of typed wildfire drivers
for robust forecasting. The main failure mode is distribution mismatch because
the retained control was trained with zero-filled blocks rather than propagated
values; the frozen-checkpoint screen measures that risk directly.
T1/T5 real-data smoke jobs `21221083/21221084` completed in 22/25 seconds with
successful backward, checkpoint reload, and inference; peak GPU allocations
were only `0.62/1.73GB`. The transformation intentionally changes corrupted
inputs, producing nonzero base-output differences `1.840/1.551`; M00/M01
bypass is enforced by the zero spatial mask rather than global equivalence on
the M06 smoke. Zero-training seed-0 evaluations `21221335/21221336` load the
existing fresh-control checkpoints and use minimum 10GB slices.
T1 completed at M00/M01/M06/M07 `.585066/.308313/.338536/.147805`; the
spatial transformation strongly lowers both block metrics relative to control.
The original T5 evaluation failed before model loading because it incorrectly
referenced the evaluate-only recovery directory, which contains no checkpoint.
Recovery `21221884` points to the actual seed-0 T5 training checkpoint; no
optimizer or scientific setting changes.
The exact spatial-route deltas are M06 `-.032289`, M07 `-.042107`, block mean
`-.037198`, and primary `-.024799`. This is a large mechanism failure, not a
near-threshold result: propagated smooth values are more harmful to the model
than its trained zero-fill convention. X13 is rejected; recovery `21221884`
was cancelled before allocation, and the quantitative artifact is
`cross-history-analysis/x13-normalized-inpaint-t1.json`.

X14 is the fixed magnitude-target probe after complete FireDrop specialization
saturated. It continues the same B3/B5 initialization for 3000 steps with
FireDrop probability `0`, BlockDrop probability `1`, the unchanged optimizer,
batch, model, and forecast loss. The candidate is routed only to observable
M06/M07; control supplies M00/M01. This is distinct from X6, which changed both
corruption probabilities together from `.3` to `.5`, and from X10, which adds
a restoration module under the original mixed sampling. It can count as at
most one training-strategy contribution. The seed-0 gate remains routed
primary `>=+.005` in both histories, while the separate `+.02` target is only
reported if actually reached. Closest missing-modality work already establishes
dropout training, so no architectural novelty is claimed. Failure modes are
over-specialization to synthetic block geometry and saturation from the B3/B5
mixed-corruption initialization.
T1/T5 smoke replacements `21222278/21222279` completed successfully with exact
initial equivalence and peak GPU allocations `0.58/1.55GB`. Formal seed-0 jobs
`21222548/21222549` use batch 64 and 3000 steps on 10GB slices; T5 receives
128GB host memory solely to avoid the already observed post-training evaluator
cgroup failure.
T1 completed at M00/M01/M06/M07 `.575114/.117167/.381522/.205614`.
The raw non-block metrics confirm deliberate over-specialization and are not
used outside the observable route. Against fresh control, routed M06/M07 gain
`+.010697/+.015702`, block mean `+.013199`, and primary `+.008800`; M00/M01
are exactly preserved by control. X14 passes the T1 seed-0 gate pending the
unchanged T5 result. Artifact:
`cross-history-analysis/x14-block-specialist-t1.json`.
T5 completed at M00/M01/M06/M07 `.596387/.079035/.394624/.214166`.
Under the spatial route, M06/M07 improve `+.004914/+.017423`, block mean
`+.011169`, and primary `+.007446`. The cross-history artifact
`cross-history-analysis/x14-block-specialist-seed0.json` reports mean primary
`+.008123` and `screen_pass=true`. Prospective seed-1/2 candidates are T1
`21223825/21223827` and T5 `21223826/21223828`; they reuse already complete
fresh controls with identical seeds rather than spend four redundant training
allocations.
The T1 seed-1/2 routed primary deltas are `+.004258/+.003344`; all three seeds
are positive and their mean is `+.005467` (block mean `+.008200`). X14 passes
T1 confirmation. Partial artifact:
`cross-history-analysis/x14-t1-confirmation-partial.json`.
The T5 seed-1/2 primary deltas are `+.012953/+.029655`; with seed 0, the T5
three-seed mean is `+.016685` and block mean is `+.025027`. All six individual
seed/history pairs are positive. The authoritative confirmation artifact
`cross-history-analysis/x14-block-specialist-confirmation.json` reports
overall primary `+.011076`, block mean `+.016614`, routed clean delta `0`, and
`confirmation_pass=true`. Twelve evaluate-only held-out jobs
`21224999`--`21225010` now evaluate only the six X14 checkpoints on 2022/2023;
the corresponding ERM summaries already exist and no training is repeated.
Commands are recorded in `cross-history-analysis/x14-heldout-jobs.txt`.
All twelve jobs completed successfully. The final artifact
`cross-history-analysis/x14-block-specialist-final.json` has
`confirmation_pass=true`, `heldout_pass=true`, and `goal_evidence_pass=true`.
T1 three-seed primary deltas for 2021/2022/2023 are
`+.005467/+.001416/+.006792`; T5 deltas are
`+.016685/+.011528/+.005485`. Every history/year cell is positive, routed M00
and M01 are exactly the ERM control, and the 18-row overall primary/block means
are `+.007895/+.011843`. X14 is the first adopted contribution from this
campaign that improves fresh ERM under both histories and both held-out years.

X15 targets cross-year block instability with an explicit geometry signal. For
each spatial hole it computes normalized erosion depth (zero at observed cells,
largest at pixels furthest from observable context) and injects a learned,
zero-initialized channel token scaled by that depth at every nontrivial encoder
resolution. It adds 1,024 parameters, activates only for M06/M07, and uses the
same implementation in T1/T5. Unlike D12 SARP's pooled local coverage and hard
global severity switch, X15 represents continuous interior depth; unlike X1 it
does not transport feature content. The matched control, optimizer, corruption
schedule, 3000 steps, and spatial route remain unchanged. The fixed seed-0 gate
is primary `>=+.005` in each history. This is an incremental reliability-prompt
module whose novelty and usefulness both depend on the matched ablation; likely
failure modes are redundancy with convolutional mask boundaries and insufficient
information to correct a missing field without its values.
T1/T5 smoke jobs `21222667/21222668` completed successfully with exact initial
equivalence, backward/checkpoint/inference coverage, and peak GPU allocations
`0.59/1.55GB`. Formal seed-0 screens `21222911/21222912` use batch 64, 3000
steps, minimum 10GB slices, and 64/128GB host memory for T1/T5 respectively.
T1 completed at M00/M01/M06/M07 `.580461/.294675/.365261/.189261`.
Against fresh control, the spatial-route M06/M07 deltas are
`-.005565/-.000651`, block mean `-.003108`, and primary `-.002072`. Continuous
hole depth does not supply enough missing content and degrades the matched
forecast, so X15 is rejected. T5 `21222912` was cancelled at 20:51; artifact:
`cross-history-analysis/x15-distance-prompt-t1.json`.

X16 is the mechanism-driven interaction between the two strongest block
hypotheses, not a new unrelated branch. It trains X10's forecast-aware dynamic
restoration under X14's block-only schedule (`fire=0`, `block=1`) for the same
3000 steps. X10's mixed schedule produced positive 2021 confirmation but lost
small amounts on T1 2022/2023; the falsifiable hypothesis is that every update
must expose a spatial hole for the restoration module to learn stable typed
corrections. X14 is the matched attribution control for the module; fresh
control measures total adoption value. X16 can count as an independent module
contribution only if it improves over X14 as well as fresh control in both
histories. Otherwise it is merely a joint recipe or another rejection. The
spatial route and all existing gates remain frozen.
T1/T5 smoke jobs `21223439/21223440` completed successfully with exact initial
equivalence and peak GPU allocations `0.84/2.79GB`. Formal seed-0 screens
`21223829/21223830` use 10/20GB slices, batch 64, and 3000 steps.
T1 completed at M00/M01/M06/M07 `.583476/.062043/.383286/.202494`.
Against fresh control its spatial-route primary is `+.008348`, but the decisive
module attribution against matched X14 is `-.000452` (block mean `-.000678`):
M06 gains `+.001764` while M07 loses `-.003120`. The restoration module does
not add value under the specialist schedule, so X16 is rejected and T5
`21223830` was cancelled at 22:59. Artifacts:
`cross-history-analysis/x16-vs-control-t1.json` and
`cross-history-analysis/x16-vs-x14-t1.json`.

X17 tests whether X14's remaining error comes from forcing one specialist to
fit two visibly different corruption severities. It trains two otherwise
identical BlockDrop-only continuations: one always receives 25% blocks and the
other always receives 50% blocks. At evaluation, the observed missing fraction
selects the 25% expert for M06 and the 50% expert for M07; the fresh control
still supplies M00/M01. This differs from archived P03, which routes a single
mixed-severity expert only inside missing pixels, and from D12, which switches
prompt depth inside one model. Because X17 only factorizes X14's training
distribution, it may strengthen or supersede X14 but cannot be counted as a
separate independent contribution. The seed-0 adoption gate is unchanged:
routed primary gain `>=+.005` against fresh control in both T=1 and T=5. Its
mechanism gate additionally requires positive block-mean gain over X14 in both
histories. Only then are seeds 1/2 run. Fixed-severity support and the offline
composer are implemented before observing any X17 result; no training is run
on the login node. Real-data T1-25%/T5-50% smoke jobs `21224594/21224595`
completed in 22/29 seconds with exit `0:0`. The four seed-0 screens are T1
25%/50% `21224659/21224660` and T5 25%/50% `21224661/21224662`, all on the
minimum 10GB slice; T5 uses 128GB host memory only for the evaluator.
The completed T1 severity route uses fixed-25% M06 and fixed-50% M07. Relative
to fresh ERM it improves M06/M07 by `+.019091/+.014924`, block mean by
`+.017008`, and primary by `+.011338`. Relative to mixed-severity X14 it gains
`+.008394` on M06 and loses `-.000778` on M07, for positive block-mean module
attribution `+.003808`. X17 therefore passes its T1 mechanism and adoption
checks, pending the unchanged T5 pair. Partial artifact:
`cross-history-analysis/x17-severity-factorized-t1-partial.json`.
The completed T5 severity route improves fresh ERM M06/M07 by
`+.011317/+.019933`, block mean by `+.015625`, and primary by `+.010416`.
It also improves mixed X14 by `+.006403/+.002510`, giving positive block-mean
module attribution `+.004456`. The two-history artifact
`cross-history-analysis/x17-severity-factorized-seed0.json` reports mean
primary `+.010877`, `screen_pass=true`, and `mechanism_screen_pass=true`.
Prospective confirmation trains both fixed-severity experts for seeds 1/2:
T1/T5 fixed-25% jobs `21225830/21225831` and fixed-50% jobs
`21225832/21225833` for seed 1; seed-2 equivalents are
`21225834/21225835` and `21225836/21225837`. Existing ERM and X14 checkpoints
remain the two controls, so no redundant control training is submitted.
After the four T1 10GB requests waited more than 18 minutes with start
estimates 42--100 minutes away, Slurm test-only probes placed equivalent 20GB
requests immediately. The unstarted T1 jobs were therefore replaced, within
the 2x resource rule, by fixed-25/fixed-50 seed-1 `21226523/21226524` and
seed-2 `21226525/21226526`. The already running T5 jobs are unchanged.
All four replacements completed successfully. Across seeds 0/1/2, the T1
severity route improves fresh ERM primary by
`+.011338/+.007649/+.003189` and block mean by
`+.017008/+.011473/+.004783`. The three-seed means are `+.007392` primary
and `+.011088` block. Relative to matched mixed-severity X14, block deltas are
`+.003808/+.005086/-.000232`, for a positive three-seed mechanism mean
`+.002887`. The frozen rule aggregates the predetermined seeds rather than
requiring every seed to be positive, so X17 passes T1 confirmation. Artifact:
`cross-history-analysis/x17-severity-factorized-t1-confirmation-partial.json`.

All four T5 confirmation jobs completed successfully. Across seeds 0/1/2,
the fixed-severity route improves fresh ERM primary by a three-seed mean of
`+.017792` and block mean of `+.026688`. Relative to matched mixed-severity
X14, the T5 block-mean attribution is `+.001662`. Together with T1, the
six-run overall primary/block means versus ERM are `+.012592/+.018888`;
`confirmation_pass=true` and `mechanism_confirmation_pass=true` in
`cross-history-analysis/x17-severity-factorized-confirmation.json`. Fixed
2022/2023 evaluation now covers all twelve expert checkpoints in 24 minimum
10GB evaluate-only jobs `21228523`--`21228546`. No training is repeated and
the held-out results will not change the recipe.

An evidence-reuse audit tested whether the unchanged global clean--corrupt
consistency control could itself be promoted against ordinary ERM. This uses
only already completed, seed-matched X8-control and X10-control summaries; it
adds no training and does not select a new recipe on test years. T=5 is
consistently positive, with three-seed primary deltas
`+.011147/+.011758/+.005600` in 2021/2022/2023. T=1 changes are
`-.000138/-.006647/-.002129`, however, so both confirmation and held-out gates
fail. The overall 18-row mean is `+.003265`. Global consistency alone is
therefore a T=5-specific observation, not a second reliable cross-history
direction. Artifact:
`cross-history-analysis/global-consistency-vs-erm-final.json`.

X18 is the architectural follow-up to X1 and the attribution follow-up to X14.
It trains X1's zero-initialized multi-scale latent context transport under the
otherwise unchanged X14 BlockDrop-only schedule. X1's mixed-corruption screen
had a routed `+.005954` T=1 primary gain but only `+.000874` at T=5, while X14
shows that persistent block supervision transfers across both histories. The
falsifiable hypothesis is therefore that the transport module lacked enough
block-active updates, rather than that latent context is intrinsically useless
for T=5. X18 is an independent module only if its routed block mean is positive
against X14 in both histories; adoption also requires the ordinary `+.005`
seed-0 primary gate against fresh ERM in both. A failure against X14 ends the
module with no coefficient or architecture sweep. This remains the same
forecast, corruption scenarios, initialization, optimizer, and 3000-step
budget; it does not add a reconstruction task or consume held-out years.
T1/T5 smoke jobs `21224666/21224667` completed in 24/26 seconds with exit
`0:0`, exact initial equivalence (`0.0`), successful backward/inference, and
peak GPU allocation `0.70/1.66GB`. Dependency-gated formal seed-0 screens
`21224668/21224669` then started automatically, so the smoke gate consumed no
manual wait and a failed path could not have consumed a training allocation.
T1 completed at M00/M01/M06/M07 `.581210/.106990/.391443/.208909`. Under the
observable spatial route, it improves fresh ERM M06/M07 by
`+.020617/+.018997`, block mean by `+.019807`, and primary by `+.013205`.
Against matched X14, the module still improves M06/M07 by
`+.009920/+.003295`, block mean by `+.006608`, and primary by `+.004405`.
Thus X18 passes both T1 adoption and module-attribution checks; no decision is
made until the unchanged T5 screen completes. Artifacts:
`cross-history-analysis/x18-vs-erm-t1.json` and
`cross-history-analysis/x18-vs-x14-t1.json`.
T5 completed at M00/M01/M06/M07 `.597367/.077257/.394835/.211189`. Its spatial
route remains useful relative to ERM, improving M06/M07 by
`+.005125/+.014447`, block mean by `+.009786`, and primary by `+.006524`.
The decisive module attribution against X14 is negative, however: M06 changes
`+.000211`, M07 `-.002976`, block mean `-.001383`, and primary `-.000922`.
Block-only supervision does not make latent context transport add value beyond
plain specialization in T=5. X18 is therefore rejected without seeds 1/2;
its T1 gain remains a history-specific observation. Artifacts:
`cross-history-analysis/x18-vs-erm-t5.json` and
`cross-history-analysis/x18-vs-x14-t5.json`.

X19 tests whether X17's measured benefit from severity factorization can be
captured inside one shared predictor rather than two full checkpoints. At each
nontrivial encoder scale, a small shared bottleneck reads the current feature
and downsampled spatial-invalidity mask; observed missing fraction selects a
separate zero-initialized mild (`<=.375`) or severe (`>.375`) residual head.
The correction is applied only inside an observed hole, so M00/M01 bypass it.
Training otherwise exactly matches X14's BlockDrop-only schedule and 3000-step
contract. The closest-work audit found dynamic modality experts in
[SimMLM](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html),
dynamic adapters in
[Synergistic Prompting](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Synergistic_Prompting_for_Robust_Visual_Recognition_with_Missing_Modalities_ICCV_2025_paper.html),
and multi-resolution compensation in
[DIS2](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html).
Therefore X19 does not claim any of those general mechanisms as new;
its bounded candidate contribution is within-modality spatial-severity routing
of content-dependent latent corrections for wildfire forecasting.

The pre-run idea verdict is `Accept with Revisions, pending the validation
experiment`. X17 supplies quantitative mechanism support: splitting full
experts improves X14 block mean by `+.003808/+.004456` in T1/T5 seed 0.
The main fatal-flaw risk is crowded prior art, defended only by the narrow
object/granularity claim and direct controls. X19 advances only if it improves
X14 block mean in both histories and retains routed primary `>=+.005` versus
fresh ERM; X17 is reported as the two-model upper bound, not an attribution
baseline that a compact model must beat. A failure ends the adapter without a
width, depth, or threshold sweep. Compute/data/engineering risk is low because
the existing paired tensors, reliability masks, runner, and Nibi slices are
reused; novelty and effectiveness remain the decisive risks.
T1/T5 real-data smoke jobs are `21225918/21225919`; dependency-gated seed-0
screens are `21225920/21225921`. All request the minimum 10GB GPU slice, while
T5 uses 128GB host memory only on the formal job for post-training evaluation.
Those 10GB smoke requests subsequently waited more than 15 minutes with
estimated starts 88--100 minutes away. An equivalent 20GB test-only probe was
placeable immediately, so all four unstarted jobs were replaced within the 2x
rule: smoke `21226527/21226528`, dependency-gated seed-0
`21226529/21226530`. The implementation and recipe remain pinned to `8f4a052`.
Both replacement smoke jobs completed in 23/26 seconds with exit `0:0`, exact
initial equivalence (`0.0`), successful backward/inference, and peak GPU
allocation 0.63/1.59GB. The dependency-gated T1/T5 seed-0 screens then started.
T1 seed 0 completed. Under the spatial route, X19 improves fresh ERM by
`+.019301/+.017277` on M06/M07, block mean `+.018289`, and primary
`+.012193`. Against the matched X14 specialist it changes M06/M07 by
`+.008605/+.001575`, hence block mean `+.005090` (routed primary `+.003393`).
This passes X19's T1 total-effect and module-attribution checks. T5 remains the
unchanged cross-history decision; no confirmation jobs are opened early.
Artifacts: `cross-history-analysis/x19-vs-erm-t1-partial.json` and
`cross-history-analysis/x19-vs-x14-t1-partial.json`.

X20 is the loss-side follow-up to X8's completion audit. X8 established that
counterfactual-impact weighting itself transfers across T=1/T=5 relative to
the unchanged global-consistency control, but its paired clean/corrupt
supervised objective made the final routed system regress against fresh ERM
on T=1/2022. X20 therefore leaves ERM's corrupt-sample supervised loss and
sampling distribution unchanged. A clean inference-mode view supplies only a
detached online teacher; the same fixed 0.1 impact-weighted Bernoulli KL is
added on the corrupt prediction. Running-stat updates also remain confined to
the ERM branch. The method has no inference-time parameters or routing.

This is one bounded repair, not a hyperparameter search. Seed 0 advances only
if primary AP improves by at least `+.005` over fresh ERM in both histories,
with M00 no worse than `-.010`; otherwise the loss direction stops. If it
passes, seeds 1/2 and fixed 2022/2023 evaluation use the same recipe. X20 and
X8 are one impact-consistency contribution family and cannot be counted as
two independent innovations.
The implementation is frozen at `74b3dbd`. T1/T5 real-data smoke jobs are
`21226332/21226333`; dependency-gated seed-0 screens are
`21226334/21226335`. The scheduler rejected an explicitly combined partition
request for the 20GB profile, so the successful submissions leave partition
selection to Nibi; Slurm resolved them to `gpubase_bygpu_b1,gpubackfill`.
The 20GB profile is exactly twice the minimum 10GB slice and was selected only
after the older 10GB smoke jobs showed estimated waits well beyond ten minutes.
Both smoke jobs completed in 26 seconds with exit `0:0`, exact initial
equivalence (`0.0`), successful backward/inference, and peak GPU allocation
of 0.63/1.75GB. Both formal screens then started after about four minutes.
T1 completed with deltas versus fresh ERM of M00/M01/M06/M07
`-.006519/+.009667/-.009673/-.011027`: primary `-.003678` and block mean
`-.010350`. This fails the raw two-history gate. Although FireDrop improves,
the preregistered rule forbids scenario routing from rescuing a failed X20,
and the old X8 family already showed held-out FireDrop instability. T5 job
`21226335` was cancelled at 33:48 to release its slice; there is no coefficient
or teacher variant sweep. Artifact:
`cross-history-analysis/x20-erm-impact-t1-partial.json`.

X21 was considered while those jobs waited: a type-separated affine
calibration of the first convolution driven by local valid-pixel coverage.
The fatal-flaw audit rejected it before implementation. Partial-convolution
padding already reweights convolution outputs by valid coverage
([Liu et al., 2018](https://arxiv.org/abs/1811.11718)); gated convolution makes
the spatial/channel selection learnable
([Yu et al., 2019](https://arxiv.org/abs/1806.03589)); and MADF generates
location-specific filters from the mask and pairs them with point-wise affine
normalization
([Zhu et al., 2021](https://arxiv.org/abs/2104.13743)). Within this repository,
D4/D12 already test local reliability tokens. Separating FireDrop from
BlockDrop changes the application granularity but not the core mechanism, so
the novelty defect is critical for an independent contribution. No code or
Slurm job was created.

X22 is the campaign's sole optimizer/tuning contribution. A post-hoc read of
the logged 100-step snapshots does not support blindly extending training:
mean sampled T1 loss over steps 2100--3000 is `.005259`, but T5 is `.006373`
versus `.005755` over steps 1100--2000. X22 therefore retains AdamW, initial
learning rate `.001`, batch 64, corruption distribution, initialization, and
3000 updates, while applying one standard cosine decay to zero. It adds no
model parameter and is explicitly a training-recipe result, not a novel
method. The fixed gate is primary AP `>=+.005` versus constant-LR ERM in each
history with M00 `>=-.010`; one seed-0 pair decides whether it advances, with
no alternative schedule, endpoint, warmup, or longer-budget sweep. The code is
frozen at `735700e`. At the scheduled queue check, 10GB and 20GB test-only
probes had the same immediate start estimate, so the minimum 10GB profile was
selected. T1/T5 smoke jobs are `21227409/21227410`; dependency-gated seed-0
screens are `21227412/21227413`. T5 alone requests 128GB host memory for the
known evaluation footprint.
The T1 10GB formal request subsequently remained pending for more than
30 minutes with a `10:51` start estimate. A same-command 20GB probe estimated
`10:09`, so the unstarted `21227412` was cancelled and replaced within the
2x rule by `21228612`. The source commit, seed, optimizer, batch, steps, host
memory, and time limit are unchanged; only its GPU slice is larger.

`compose_complete_routes.py` prepares the final system audit without opening
another model direction. For each matched history/seed/year row it takes M00
from fresh ERM, M01 from a separately validated FireDrop component, M06 from
the fixed-25% X17 expert, and M07 from the fixed-50% X17 expert. It reports the
same confirmation, held-out, and mean-primary `+.020` checks as the other
comparators. This system-level artifact cannot establish component novelty or
attribution; each component must first pass its own frozen closest-control
gate. In particular, X20 is eligible for this composition only if its raw
two-history gate passes, so scenario routing cannot rescue a failed X20.

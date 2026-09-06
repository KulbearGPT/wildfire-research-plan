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

- [ ] Add a separate `reproductions/cross_history/` package for the campaign.
  Reuse corrected resolver, normalization, raw evaluation corruptions, exact
  AP code and archived B5 checkpoint (explicit relocated path).
- [ ] Implement shared T1/T5 encoder-decoder access and the three methods.
  Keep clean training targets only in the training loss, absent at inference.
- [ ] Add one Slurm runner with train/eval modes, job-specific output path,
  committed source snapshot, fixed seed metadata, checkpoint and results.
- [ ] Run a GPU one-batch check inside Slurm for each method/history, including
  original-vs-control output agreement and backward pass. No global tests.
- [ ] Submit eight seed-0 experiments (control + three methods, both T).
  Inspect resource availability; prefer 10/20GB slices, at most 2x overflow
  unless a blocking prerequisite requires more. Revisit jobs pending >10min.
- [ ] Compare against fresh controls, diagnose failures, register any changed
  hypothesis before further experiments; confirm and test only passing recipes.
- [ ] Commit all progress and report three only when evidence satisfies above.

## Status

No new quantitative result yet. Historical B3/B5 and D2/D13 controls are
starting evidence, not proof of any new direction.

Implementation: `357e475`, numerical mixture fix `8c654e2`. Shared raw
evaluation permits T5 only by an explicit opt-in; mainline T1 default stays.
Physical batch 16, accumulation 4 applies to every new control/candidate.
Existing full-batch D2/D13 numbers are context, not the matched comparison.

GPU smoke jobs: T1 `21210705`, T5 `21210706`, each checked control/context/
distill/transition in sequence on a 20GB H100 slice, 4 CPUs, 32GB host memory.
Earlier 10GB requests `21210683/21210684` were cancelled while pending because
no start estimate was available; 20GB had an approximately 11-minute estimate.
No model or dataset computation has been run on the login node.

Both smoke jobs completed with exit `0:0`. All eight real-data combinations
completed one optimizer step, backward pass, checkpoint reload path, and one
M06 2021 inference. Initial maximum absolute output difference from the base
was exactly zero for control/context/distill and `2.38e-7` (T1) / `4.77e-7`
(T5) for the transition mixture. Peak GPU allocation at physical batch 16 was
0.70GB (T1) and 1.97GB (T5); formal screens therefore use physical/effective
batch 64 on 20GB slices to reduce wall time, identically for all variants.

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

Queue correction: original T5 distill/transition `21211225/21211226` were
cancelled pending after >10 minutes. Same-resource replacements
`21211486/21211487` started immediately alongside T5 control/context
`21211223/21211224`.

Follow-up seed-0 screens: T1 `distill_block/risk` are
`21211826/21211827`; T5 equivalents are `21211828/21211829`. They use the
same 20GB slice and matched train/evaluation contract and were initially
pending behind the active first wave.

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

The registered `context_adapter` variant freezes the entire initial forecaster
and trains only the zero-initialized context-transport layers. Because every
transport residual is multiplied by the observed spatial-hole mask, this
single checkpoint is exactly the original control whenever no spatial hole is
present; it removes the two-checkpoint routing caveat and isolates the X1
mechanism. It is an alternative implementation of X1, not an additional
contribution. Its fixed recipe uses the same 3000 steps and optimizer settings.
Its attribution control is the same frozen initial checkpoint evaluated by the
cross-history runner with zero update steps; comparison to the fresh full-model
continuation will be reported only as a secondary total-system comparison.

While GPU jobs run, `reproductions/cross_history/compare.py` provides the
minimal downstream result path. It pairs summaries by history/seed/year,
computes individual, primary, block, clean-guardrail and multi-run statistics,
and can apply the declared spatial-specialist route. Its gates require explicit
T1+T5 coverage: seed-0 2021 screening, seeds 0/1/2 2021 confirmation, and all
four history/year cells on the fixed 2022/2023 tests. A single-history input
can no longer be reported as a cross-history pass. `run.py --evaluate-only`
already performs immutable 2022/2023 evaluation from a selected checkpoint.

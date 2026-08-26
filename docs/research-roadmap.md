# Research Roadmap After Baseline Reproduction

This is the authoritative experiment order for the next project phase. It
converts the completed data and baseline work into a sequence of small
scientific decisions. It is intentionally not a model wish list: each stage has
one question, the minimum runs needed to answer it, a gate, and a compact
output.

## Frozen evidence entering this roadmap

- WSTS+ contains 999 active event-level HDF5 files across 2016–2023 after the
  documented label repair and Phase 0 audit.
- The split is fixed to 2016–2020 train, 2021 validation, and 2022–2023 test.
- No-fire and latest-mask persistence are the deterministic controls.
- The official Res18-U-Net `T=1` pipeline is executable, one Fold-2 training run
  is sealed, and all twelve official released weights have independently
  verified test-only results.
- The release campaign supports released-weight executable reproducibility but
  does not prove paper-table provenance identity.
- Missing provenance limits the main study to controlled missingness. The work
  does not establish natural-missingness or operational-deployment performance.

**Indexing correction discovered 2026-08-26:** the pinned upstream multi-year
dataset resolver continues its outer year loop after finding a sample, so the
found year is overwritten by the final included year. Local pooled-year
training runs before P09 therefore do not establish the recorded 2016--2020
training exposure, although single-year controlled evaluation is unaffected.
P09 installs a local first-match resolver. Earlier checkpoints remain frozen
legacy comparisons; corrected-data claims require matched reruns.

## Active Fast Experiment Track — 2026-08-23

The immediate priority is to begin new learned experiments. The remaining
Stage 0 checkpoint-equivalence work and Stage 1 positive-weight sensitivity
study are deferred; they remain documented below but no longer block Stage 2.

The launch gate was deliberately small: exact active-fixed WSTS+ year counts,
the frozen 2016–2020 train / 2021 validation split, one finite train and
validation batch inside each job, a locked 2022–2023 test, and compact run
lineage. The first matched seed-0 screening runs were:

1. C00 Res18-U-Net, `T=1`, All features, 3,000 optimizer steps;
2. C02 Res18-UTAE, `T=5`, Multi features, 3,000 optimizer steps.

Both completed with passing records. C00 job `20346980` reached validation AP
0.547372 and C02 job `20346981` reached 0.584540; this screening difference
does not eliminate either baseline. Fresh matched seed-0 10,000-step jobs
`20353582` and `20353584` have been submitted. After both pass, seed 1 and seed
2 are the declared replication set. The canonical run IDs and gates live in
[`reproductions/wsts_fast_track/`](../reproductions/wsts_fast_track/README.md).
No Fast Experiment Track artifact reads 2022–2023 before the declared final
test action, and a screening result is not a clean-performance or held-out-
test-performance claim.

## Stage 0 — Cluster migration and equivalence

**Status:** Deferred; not a Fast Experiment Track launch prerequisite.

**Question:** Does the selected cluster environment execute the pinned baseline
with numerically equivalent test behavior before any new training begins?

**Required runs:** environment/import smoke, one train and validation batch
without optimization, one official Fold-2 checkpoint test-only equivalence
run, and one 500-step timing calibration after equivalence passes.

**Gate:** data counts, bytes, year distribution, input/output shapes, finite values,
official weight strict loading, test metrics within a declared numerical
tolerance, and complete environment provenance must pass. Failure stops all new
training.

**Output:** one reviewed cluster-qualification record containing Git commit,
dataset manifest, environment, Slurm job IDs, metric comparison, throughput,
memory observation, and the selected time tier.

## Stage 1 — Training-contract sensitivity

**Status:** Deferred until after the first Stage 2 seed-0 screening results.

**Question:** Does the official configuration/code positive-weight discrepancy
materially affect the Fold-2 result?

**Required runs:** a paired same-environment experiment that changes only the
positive-weight behavior: official runtime recomputation (`608.465...` in the
sealed Fold-2 run) versus fixed YAML value `236`.

**Gate:** freeze the materiality rule and any predetermined expansion folds
before launch. If the difference is immaterial, retain official runtime
behavior. If it is material, verify on the predetermined additional folds and
report both contracts rather than selecting the better result post hoc.

**Output:** one paired table with AP, F1, IoU, precision, recall, loss,
calibration diagnostics, runtime, and the exact one-variable diff.

## Stage 2 — WSTS+ learned controls

**Status:** Runtime-complete but scientifically reopened by the multi-year
indexing defect. The six C00/C02 seed-0/1/2 10K records passed their original
runtime gates but do not prove pooled 2016--2020 training exposure.

**Question:** What clean-observation learned performance is available on the
actual frozen WSTS+ split before robustness mechanisms are introduced?

**Required runs:**

1. Res18-U-Net, `T=1`, All features, seed 0, as the bridge from the completed
   official reproduction.
2. UTAE(Res18), `T=5`, Multi features, seed 0, as the stronger
   clean-observation backbone candidate.
3. Fresh `C00-S0-10K` and `C02-S0-10K` runs after both 3,000-step screening
   records pass; neither resumes a screening checkpoint.
4. `C00/C02-S1-10K` and `C00/C02-S2-10K` only after both seed-0 10K data,
   numerical, runtime, and metric gates pass.

**Gate:** no split leakage; finite training; checkpoint selection fixed before
launch; test results reported separately for 2022 and 2023; and improvement
interpreted relative to the same-year control rather than by comparing raw AP
across different prevalence levels.

**Output:** a reviewed clean-baseline table with per-year and aggregate metrics,
seed variability, calibration, runtime, and parameter/training-budget counts.

The replication gate and controlled-missingness implementation boundary are
specified in
[`2026-08-23-wsts-post-control-design.md`](superpowers/specs/2026-08-23-wsts-post-control-design.md).
The current upstream validation dataset enables training augmentation, so the
formal Stage 3 evaluator must instead construct a deterministic
`is_train=false` path before any M00--M07 result is recorded.

## Stage 3 — Controlled-missingness diagnosis

**Status:** Complete; all 96 M00--M07 tasks over the accepted clean checkpoints
and 2022--2023 finished. M01 active-fire-history absence was the dominant
failure, followed by severe structured missingness.

**Question:** Which prespecified observation failures cause stable and
scientifically meaningful degradation in the learned controls?

**Required runs:** apply the frozen, test-only M00–M07 matrix declared in
[`reproductions/wsts_fast_track/`](../reproductions/wsts_fast_track/README.md)
to accepted clean checkpoints. It covers clean reference, missing and
one-day-stale fire history, observed-weather loss, forecast-weather loss,
combined weather loss, and deterministic structured blocks covering 25% and
50% of dynamic-input area. The identifiers are declared now; corruption
transforms remain non-launchable until their later implementation is reviewed.

**Gate:** corruption generation must be deterministic, label-independent, and
free of future information. Continue to method development only if degradation
is reproducible across relevant years/events and exceeds the frozen practical
materiality rule.

**Output:** missingness-intensity curves, same-year deltas from the clean
checkpoint, event/scenario slices, and representative failure cases. Do not
claim that simulated corruption is observed natural missingness.

## Stage 4 — Matched robust baselines and main method

**Status:** Active. P00 established the minimal training-corruption baseline:
30% training-only active-fire dropout. Its controlled 2021 AP changed from
0.584368 to 0.585322 on M00 and from 0.045920 to 0.299465 on M01. P01 keeps
that contract fixed and adds one explicit active-fire-validity channel.
P01 did not improve P00. P02 added 30% structured BlockDrop and improved P00
by 0.0183 AP on M06 and 0.0303 on M07, but degraded M00 by 0.0200 and M01 by
0.0322. It therefore exposed a robustness tradeoff and was not promoted as a
single model. P03 uses P02 only inside known missing blocks and P00 elsewhere.
The training-free router exactly preserved P00 AP on 2021 M00/M01 and improved
M06/M07 by 0.0339/0.0378, making P03 the 2021-selected routed prototype.
P04 compressed routing to one frozen P00 pass plus a 17-parameter residual
head. It preserved M00/M01 and improved P00 on M06/M07 by 0.0151/0.0107, but
trailed P03 by 0.0188/0.0271 and was not promoted.
P05 replaced P04's `1x1` residual with a 145-parameter `3x3` residual, but its
M06/M07 AP decreased to 0.3277/0.1388. Merely enlarging the output correction
does not close the gap to P03.
P06 adapted the final decoder block and head but reached only 0.3302/0.1410 on
M06/M07. The spatial-router capacity escalation is complete. P03's final
held-out M06/M07 deltas were -0.0231/-0.0261 in 2022 and +0.0135/+0.0096 in
2023. Because the gain is not cross-year stable, P03 is not promoted and P00
remains the frozen legacy checkpoint. P07 collapsed to an effectively
deterministic stochastic residual, and P08 generated nonzero uncertainty but
regressed AP. P09 corrected the upstream year resolver and fine-tuned P02 with
15-group year-corruption GroupDRO. It improved M06/M07 over P00 in 2021 and in
both fixed test years. P10 then held the corrected resolver, sampler, data,
seed, optimizer, learning rate, and 3,000-step budget fixed while replacing
GroupDRO with ordinary ERM. P09 and P10 were effectively tied in 2021; P10 was
slightly better in 2022 and slightly worse in 2023. The GroupDRO attribution is
therefore rejected: corrected, balanced five-year exposure explains the gain.
P10 is the parsimonious leading candidate. The 2022--2023 results remain
reporting-only and cannot select further settings.

**Question:** Does explicit reliability conditioning improve robustness beyond
simple filling, validity masks, uniform fusion, reconstruction, capacity, and
modality specialization controls?

**Required runs:** implement and screen in increasing cost order:

1. zero-fill and last-observation-carried-forward;
2. direct concatenate-plus-validity-mask;
3. uniform available-modality fusion;
4. lightweight direct reliability gating;
5. MaskUNet reconstruction followed by the same frozen forecaster;
6. FireEx-style modality experts and a capacity-matched generalist ensemble;
7. residual reliability gating with candidate semantic-prototype reliability,
   while retaining the same backbone, data, corruption, and training budget.

**Rapid-prototype decision:** complete. Do not tune P00--P06 against the now
opened 2022--2023 results. Any future method-development phase must freeze a
new hypothesis and use only the training/validation evidence available before
this final held-out check; P00 is its matched mainline baseline.

**Gate:** a method must preserve clean performance within the frozen tolerance,
improve multiple prespecified missingness regimes, and beat the matched
capacity/control baseline. A larger parameter count or an unmatched training
budget is not evidence for reliability modeling.

**Output:** one main comparison table, clean-versus-corrupted curves, gate or
reconstruction diagnostics, parameter and compute controls, and exact
ablation-to-claim mapping.

## Stage 5 — Conditional extensions

These extensions are not initial implementation commitments. Every item has
status **Candidate, primary-source verification required** until its primary
paper or official implementation has been checked against the wildfire task
and label contract.

- MaskCVAE: consider only if severe block missingness makes the reconstruction
  route competitive enough to justify generative uncertainty and extra cost.
- STARS-style semantic alignment: consider only if measured representation
  drift remains after the simpler reliability baseline.
- Arbitrary-modal attention or set fusion: add only if the direct gate needs a
  stronger generic any-subset fusion control.
- Direct/reconstruction hybrid routing: add only if the two routes win in
  distinct, reproducible missingness regimes.
- Timestamp-centric approaches such as AnytimeFormer/AGFlow: defer unless
  defensible acquisition or availability timestamps are recovered.
- Other ideas from the shared planning conversation, including SGMA/MAGIC-like
  components: retain as hypotheses until primary-source verification and a
  one-variable experiment justify inclusion.

## Reporting rules shared by all stages

1. Record Git commit/status, data and weight manifests, environment, Slurm job
   ID, seed, configuration, command, checkpoint, and terminal status.
2. Never silently retry a scientific process or replace a failed run with a new
   run under the same identity.
3. Report AP together with F1, IoU, precision, recall, loss, prevalence context,
   undefined-event counts where applicable, and calibration diagnostics.
4. Keep 2022 and 2023 visible. Do not let one aggregate hide cross-year failure.
5. Treat active-fire predictions as proxy-label forecasting, not complete fire
   perimeters or deployment-ready spread simulation.
6. Promote a candidate method only after the experiment answering its gate has
   passed; otherwise stop, simplify, or revise the research premise.

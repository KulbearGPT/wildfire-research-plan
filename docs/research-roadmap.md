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

## Stage 0 — Cluster migration and equivalence

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

**Question:** What clean-observation learned performance is available on the
actual frozen WSTS+ split before robustness mechanisms are introduced?

**Required runs:**

1. Res18-U-Net, `T=1`, All features, seed 0, as the bridge from the completed
   official reproduction.
2. UTAE(Res18), `T=5`, Multi features, seed 0, as the stronger
   clean-observation backbone candidate.
3. Additional seeds only after the seed-0 data, numerical, runtime, and metric
   gates pass.

**Gate:** no split leakage; finite training; checkpoint selection fixed before
launch; test results reported separately for 2022 and 2023; and improvement
interpreted relative to the same-year control rather than by comparing raw AP
across different prevalence levels.

**Output:** a reviewed clean-baseline table with per-year and aggregate metrics,
seed variability, calibration, runtime, and parameter/training-budget counts.

## Stage 3 — Controlled-missingness diagnosis

**Question:** Which prespecified observation failures cause stable and
scientifically meaningful degradation in the learned controls?

**Required runs:** apply a frozen test-only corruption matrix to the clean
checkpoints. The minimum matrix covers missing fire history, one-day-stale fire
history, loss of one dynamic modality, combined dynamic-modality loss, and
structured spatial block missingness at prespecified severities.

**Gate:** corruption generation must be deterministic, label-independent, and
free of future information. Continue to method development only if degradation
is reproducible across relevant years/events and exceeds the frozen practical
materiality rule.

**Output:** missingness-intensity curves, same-year deltas from the clean
checkpoint, event/scenario slices, and representative failure cases. Do not
claim that simulated corruption is observed natural missingness.

## Stage 4 — Matched robust baselines and main method

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

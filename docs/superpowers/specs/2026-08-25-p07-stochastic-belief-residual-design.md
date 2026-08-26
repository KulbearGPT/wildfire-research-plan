# P07 Stochastic Belief Residual: Fast-Prototype Design

## Decision

Prototype the belief-state hypothesis with the smallest change to P00. P07 will
keep P00 frozen, learn a stochastic residual only inside known missing blocks,
and marginalize four sampled forecasts. This is a scientific probe, not a
general uncertainty framework or production subsystem.

The fixed temporal split remains 2016--2020 train, 2021 validation, and
2022--2023 test. The final split is described as a fixed temporal test set, not
as untouched data. P07 is selected on 2021, frozen, and then evaluated without
further tuning on 2022 and 2023.

## Research hypothesis

A deterministic missing-region correction collapses an ambiguous observation
into one state. A small stochastic belief residual should better represent
observation-induced ambiguity and improve the marginalized next-day forecast
under structured missingness without changing clean or observed-region P00
predictions.

## Minimal architecture

P07 reuses the existing P04/P05 single-forward path:

1. Run frozen P00 through its encoder and decoder to obtain its 16-channel
   decoder feature and default logits.
2. Apply one small `3x3` convolutional belief head to the detached decoder
   feature and routing mask. The head emits per-pixel `mu` and bounded
   `log_sigma` tensors for a compact residual latent.
3. During training, draw four reparameterized samples. A shared `1x1` output
   head maps each sample to a residual logit.
4. Add each residual only inside the known missing block. Outside it, every
   sample is exactly P00.
5. Average `sigmoid` probabilities across samples. Convert the mean probability
   back to logits only where the existing loss interface requires logits.
6. Save the predictive mean and sample variance. The variance is diagnostic
   evidence for observation-induced uncertainty, not a calibrated guarantee.

The first prototype remains `T=1`. A temporal belief encoder, CVAE, diffusion
model, learned expert router, and new backbone are explicitly deferred until
this probe succeeds.

## Training and evaluation

- Train only the belief and output heads on 2016--2020.
- Use one fixed seed and the existing 25%/50% structured BlockDrop training
  corruption.
- Use one fixed 3,000-step run. The stochastic head has higher optimization
  variance than P04--P06, whose single-batch losses were still noisy at step
  1,000. Do not add early stopping or sweep intermediate checkpoints.
- Select using 2021 M00, M01, M06, and M07.
- Compare P07 with P00 and the existing deterministic residual result; no new
  baseline campaign is required.
- If selected, freeze the checkpoint and evaluate the same four scenarios for
  2022 and 2023 on Nibi compute nodes.
- Primary metric: AP. Report Brier score or NLL plus mean predictive variance
  only to test whether stochasticity adds information. Do not create a broad
  uncertainty benchmark.

## Promotion rule

Promote P07 to the full-method branch only when:

1. M00 and M01 preserve P00 by construction, within numerical tolerance.
2. The combined 2021 M06/M07 AP effect is positive.
3. Against the capacity-matched deterministic residual, P07 improves AP or a
   proper scoring rule without a material AP regression.

Failure closes this particular stochastic-residual implementation; it does not
falsify every belief-state model. Regardless of outcome, a selected and frozen
P07 is reported on both 2022 and 2023 without post-test retuning.

## Implementation boundary

Prefer one model file plus minimal edits to the existing residual trainer,
evaluator, and Nibi launcher. Reuse current checkpoint loading, corruption,
metrics, records, and Slurm conventions. Do not introduce configuration
frameworks, registries, generic sampling APIs, multi-seed orchestration,
publication packaging, or exhaustive unit tests.

Verification is limited to a local import/shape check and one tiny synthetic
forward/backward check. No training or GPU-heavy evaluation runs on the login
node; all scientific workloads run through Slurm on Nibi compute nodes.

## Paper positioning if successful

P07 is the cheapest empirical test of the paper's technique claim: forecast
from a distribution over plausible latent observation states rather than from
one reconstructed history or one deterministic expert choice. Benchmark and
uncertainty analyses support that method claim but are not standalone headline
contributions.

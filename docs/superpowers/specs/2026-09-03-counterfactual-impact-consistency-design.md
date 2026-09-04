# Counterfactual Impact-Weighted Consistency Design

## Objective

Find one additional wildfire-forecasting method direction whose improvement is
both attributable and practically material. The retained method must improve
the three-year mean of M01/M06/M07 average precision by at least `+0.020`
against its matched ERM control. A candidate that only improves through a
larger training budget, a different sample stream, routing, or test-year model
selection does not qualify.

## Motivation and novelty boundary

The completed D1 experiment shows that paired clean-to-corrupt prediction
consistency is useful but small: global Bernoulli KL improves the three-year
mean primary AP by `+0.008970`. In wildfire forecasting, a missing input cell
and the output cells affected one day later need not have the same spatial
support. Uniformly averaging KL over the complete forecast therefore spends
most of the consistency signal on background pixels that did not respond to
the corruption.

The proposed **Counterfactual Impact-Weighted Consistency (CIWC)** uses the
clean/corrupt prediction change itself to identify the forecast locations
counterfactually affected within the forecasting model by the synthetic
observation failure. It applies no
reconstruction stage, needs no independent teacher, and adds no inference-time
module.

This is a deliberately bounded novelty claim. MIC applies an EMA teacher to
masked target images for domain adaptation, M3L distils a full-modality teacher
into a missing-modality student, and recent partial-observability wildfire work
reconstructs the fire history before forecasting. Disagreement-guided
distillation also exists for compressing medical segmentation models. CIWC's
differing axis is a same-model, paired counterfactual intervention whose
induced *future-forecast impact* determines the consistency support. The work
must not claim that masking, consistency, or disagreement weighting by itself
is new.

Closest references:

- Hoyer et al., "MIC: Masked Image Consistency for Context-Enhanced Domain
  Adaptation," CVPR 2023, <https://arxiv.org/abs/2212.01322>.
- Maheshwari et al., "Missing Modality Robustness in Semi-Supervised
  Multi-Modal Semantic Segmentation," WACV 2024,
  <https://arxiv.org/abs/2304.10756>.
- Yang et al., "Robust Wildfire Forecasting under Partial Observability: From
  Reconstruction to Prediction," 2026, <https://arxiv.org/abs/2603.09042>.
- Oksuz, "Disagreement-Guided Knowledge Distillation for Efficient Kidney
  Segmentation in Abdominal CT," Applied Sciences 2026,
  <https://doi.org/10.3390/app16115573>.

## Method

For an aligned clean input `x_c`, corrupted input `x_m`, and one forecasting
model `f`, define clean teacher probability `q` and corrupt student probability
`p`:

```text
q = sigmoid(f(x_c)).detach()
p = sigmoid(f(x_m))
d = abs(q - p.detach())
w[b] = d[b] / max(mean_over_H_W(d[b]), 1e-6)
L_CIWC = mean(w * KL_Bernoulli(q || p))
```

The impact weight `w` is stop-gradiented. If a sample produces exactly zero
counterfactual change, its CIWC term is zero. Per-sample normalization keeps
the average weight at one whenever an impact exists, so `lambda=0.1` remains
directly comparable with D1-KL. Supervision remains the mean of clean and
corrupt focal losses:

```text
L = 0.5 * (L_sup_clean + L_sup_corrupt) + 0.1 * L_CIWC
```

CIWC changes only the consistency weighting. It reuses D1's single upstream
crop followed by processed-space corruption, ensuring that the two predictions
are spatially aligned.

## Matched experiment contract

- Initialization: the exact corrected B3 checkpoint used by D1-ERM/D1-KL.
- Model: C00 ResNet-18 U-Net, one input timestep.
- Train years: 2016--2020 with the corrected first-match resolver.
- Corruption: D1's 30% FireDrop plus 30% BlockDrop sampling.
- Optimizer: AdamW, learning rate `1e-3`.
- Budget: 3,000 optimizer steps, seed 0, identical batch size and loader seed.
- Method weight: `lambda_ciwc=0.1`; no tuning sweep.
- Matched ERM control: completed D1-ERM.
- Closest method ablation: completed global D1-KL at the same weight.
- Compute: training and evaluation run only through Slurm on Nibi. The login
  node is limited to source editing, focused unit tests, queue inspection, and
  result parsing.

## Decision gates

The 2021 selection rule is frozen before training. CIWC advances only if all
conditions hold:

1. mean M01/M06/M07 AP minus D1-ERM is at least `+0.020`;
2. mean M01/M06/M07 AP is greater than D1-KL at the same `lambda=0.1`;
3. no individual primary scenario is more than `0.005` below D1-ERM;
4. M00 AP is no more than `0.010` below D1-ERM.

Only a passing checkpoint is evaluated once on 2022 and 2023. It is retained
as a quantitatively reliable direction only if:

1. its D1-ERM primary delta is positive in every year;
2. the mean of the three yearly primary deltas is at least `+0.020`;
3. its three-year mean primary AP exceeds D1-KL;
4. no yearly M00 delta from D1-ERM is below `-0.010`.

All absolute AP values, deltas, job identifiers, and failure decisions are
recorded. Test-year results never choose a variant or hyperparameter.

## Minimal implementation

Add one loss primitive and one CIWC trainer/runner by reusing D1's dataset,
checkpoint loading, evaluator, and completed controls. Focused CPU tests cover
the loss normalization, stop-gradient contract, zero-impact behavior,
checkpoint metadata, and shell syntax. No broad test suite, multi-seed run, or
long-budget confirmation is part of the screening stage.

## Failure ladder

If CIWC fails the 2021 gate, record it as a negative result and do not access
2022--2023. The next approved candidate is an AP-aligned counterfactual ranking
loss; if that also fails, the third candidate is a jointly trained
reliability-conditioned forecast-logit adapter. Each receives its own frozen
2021 design and the same `+0.020`/clean-guardrail standard. Existing negative
RNC, missing-token, frozen residual, GroupDRO, and routing branches are not
reopened.

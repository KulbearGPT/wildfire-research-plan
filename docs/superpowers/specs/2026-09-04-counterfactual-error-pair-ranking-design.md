# Counterfactual Error-Pair Ranking Design

## Objective

D9 tests a metric-aligned training method after closing the output-adapter
family. It must improve 2021 mean M01/M06/M07 AP by at least `+0.020` over the
matched D1-ERM continuation, then satisfy the same frozen cross-year rule. It
adds no inference parameters or routing.

## Evidence and novelty boundary

D5's counterfactual impact weighting reached `+0.015468` primary AP, while
D6's label-free spatial distribution matching fell to `+0.008733`. This shows
that focusing on corruption-induced changes is useful, but forcing the whole
clean ranking onto the corrupt prediction is not. The evaluation metric is
average precision, so the next loss should correct only label-confirmed
positive/negative ordering failures caused by the observation intervention.

Ranking losses and hard-pair selection are established. AP-Loss (CVPR 2019,
<https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.html>),
Rank & Sort Loss (ICCV 2021,
<https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html>),
and Adaptive Pairwise Error (CVPR 2022,
<https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Revisiting_AP_Loss_for_Dense_Object_Detection_Adaptive_Ranking_Pair_CVPR_2022_paper.html>)
are closest prior art. D9 does not claim pairwise ranking, hard-negative mining,
or AP optimization as new.

The bounded novelty candidate is **intervention-selected error-pair ranking**
for partial-observation hazard forecasting: aligned clean/corrupt predictions
identify which labelled fire pixels lost confidence and which labelled
background pixels gained confidence specifically because observations were
removed; only those two counterfactual error sets form ranking pairs. This is
different from confidence-only hard mining and D6's label-free distillation.

## Fixed loss

For each synthetically corrupted sample, let `q=sigmoid(z_clean).detach()` and
`p=sigmoid(z_corrupt).detach()`. Define intervention errors:

```text
positive_drop = relu(q - p) on target-positive pixels
negative_rise = relu(p - q) on target-negative pixels
```

Select at most the top 64 strictly positive entries from each set. Their
corrupt logits form all selected positive/negative pairs:

```text
L_pair = mean softplus(z_corrupt_negative - z_corrupt_positive) / log(2)
```

Samples without both error types contribute zero. Uncorrupted paired samples
contribute zero. Selection and clean predictions are stop-gradiented; gradients
flow only through corrupt logits in the pair loss. The complete fixed objective
is:

```text
L = 0.5 * (L_sup_clean + L_sup_corrupt)
    + 0.1 * L_CIWC
    + 0.01 * L_pair
```

The CIWC term is exactly D5's. `top_k=64` and `lambda_pair=0.01` are frozen once
without a sweep. Division by `log(2)` makes equal-logit pairs have unit loss.

## Matched contract

- Exact corrected B3 initialization, C00 ResNet-18 U-Net, T=1.
- Exact D1/D5 2016--2020 clean/corrupt stream, random-call order, and seed.
- AdamW `1e-3`, batch 32, 3,000 steps, seed 0.
- No architecture, inference, sampler, or optimizer change.
- Matched control D1-ERM; closest ablations D5-CIWC and D6-CIRC.
- Training/evaluation only inside Slurm; login node runs focused synthetic CPU
  tests, queue inspection, and small-result parsing only.

## Frozen gates

D9 advances beyond 2021 only if mean M01/M06/M07 AP minus D1-ERM is at least
`+0.020`, exceeds both D1-KL and D5-CIWC, has no primary delta below `-0.005`,
and has M00 delta at least `-0.010`. A passing checkpoint is evaluated once on
2022 and 2023. Final retention requires positive primary delta in every year,
a three-year mean delta at least `+0.020`, mean above D1-KL, and every M00
delta at least `-0.010`. Held-out years cannot select `top_k`, coefficient, or
variant.

## Minimal implementation

Add one loss primitive plus a thin trainer/evaluator/runner reusing D5 and the
opt-in reliability maps already emitted for D7. Focused tests cover exact pair
selection, zero cases, stop-gradient, objective composition, checkpoint
contract, and shell syntax. Do not run broad tests, a coefficient sweep,
multiple seeds, or long-budget training during screening.

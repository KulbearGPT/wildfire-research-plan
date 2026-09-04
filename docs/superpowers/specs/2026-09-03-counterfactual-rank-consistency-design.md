# Counterfactual Impact-and-Rank Consistency Design

## Objective

D6 tests whether explicitly preserving the spatial ordering of forecast risk
under missing observations can turn D5-CIWC's `+0.015468` 2021 primary AP gain
into a material result. The retained method must still reach at least `+0.020`
mean M01/M06/M07 AP against the matched D1-ERM control; the target is not
relaxed after seeing D5.

## Evidence and scope

D5-CIWC improved 2021 M01 by `+0.043494`, but changed M06 and M07 by only
`-0.000076` and `+0.002985`. Pixelwise Bernoulli consistency therefore finds
the complete FireDrop case but does not explicitly constrain which spatial
locations should rank as highest risk inside a missing block. D6 keeps D5's
impact-weighted term and adds one listwise spatial-risk term. It does not add a
new network, reconstruction target, data source, or inference-time operation.

Spatial-distribution knowledge distillation is established prior art (Shu et
al., ICCV 2021, <https://arxiv.org/abs/2011.13256>). The bounded novelty claim
here is not spatial softmax or distillation itself. It is the combination of
(1) same-model paired clean/corrupt counterfactual wildfire forecasts,
(2) counterfactual impact-weighted pixel consistency, and (3) preservation of
the future-risk spatial ordering under an observation intervention. The clean
prediction is a stop-gradient target; no external teacher is used.

## Fixed method

For clean logits `z_c` and corrupt logits `z_m`, flatten each sample's spatial
dimensions and form distributions over forecast locations:

```text
q = softmax(flatten(z_c.detach()), dim=space)
p = softmax(flatten(z_m), dim=space)
m = 0.5 * (q + p)
L_rank = 0.5 * KL(q || m) + 0.5 * KL(p || m)
L_rank_normalized = mean_batch(L_rank) / log(2)
```

The normalized Jensen-Shannon divergence is bounded in `[0, 1]`, symmetric as
a value, and differentiates only through the corrupt prediction because the
clean branch is detached. Its listwise spatial softmax responds to risk
ordering rather than independent probability calibration. Temperature is
fixed at `1.0`.

The complete objective is frozen as:

```text
L = 0.5 * (L_sup_clean + L_sup_corrupt)
    + 0.1 * L_CIWC
    + 0.01 * L_rank_normalized
```

The CIWC definition and coefficient are exactly D5's. The rank coefficient is
fixed once at `0.01`; there is no sweep.

## Matched experiment contract

- Exact corrected B3 initialization used by D1 and D5.
- C00 ResNet-18 U-Net, one input timestep, 2016--2020 training years.
- D1's 30% FireDrop plus 30% BlockDrop stream and identical loader seed.
- AdamW at `1e-3`, batch size 32, 3,000 steps, seed 0.
- Matched control: D1-ERM. Closest ablations: D1-KL and D5-CIWC.
- All training/evaluation runs through Slurm; the login node only edits,
  performs focused CPU checks, inspects the queue, and parses results.

## Frozen decision gates

D6 advances beyond 2021 only if all conditions hold:

1. mean M01/M06/M07 AP minus D1-ERM is at least `+0.020`;
2. the same primary mean exceeds both D1-KL and D5-CIWC;
3. no primary scenario is more than `0.005` below D1-ERM;
4. M00 is no more than `0.010` below D1-ERM.

Only a passing checkpoint is evaluated once on 2022 and 2023. It is retained
only when its D1-ERM primary delta is positive in every year, its three-year
mean primary delta is at least `+0.020`, its mean exceeds D1-KL, and no yearly
M00 delta is below `-0.010`. Held-out years never select a coefficient or
variant.

## Minimal implementation and stop rule

Add one loss primitive, one trainer, and thin evaluator/Slurm wrappers by
reusing D5. Focused tests cover identical distributions, changed ordering,
stop-gradient behavior, objective composition, metadata, and shell syntax.
No broad suite or long-budget run is part of screening. If D6 misses the 2021
gate, record the negative result without evaluating 2022--2023 and proceed to
the already approved jointly trained reliability-conditioned logit adapter.

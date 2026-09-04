# Failure-Factorized Counterfactual Adapter Design

## Decision

D8 tests one evidence-driven structural change, not a D7 hyperparameter sweep.
D5-CIWC is retained as the mechanism for complete FireDrop, while a learned
forecast correction is enabled only when a spatial BlockDrop is present. This
factorizes the response by failure geometry inside one model and removes the
shared-head interference measured in D7.

## Quantitative premise

On 2021, D5-CIWC reached M01/M06/M07 AP
`0.294861/0.361160/0.179137`. D7's shared two-regime adapter changed those
values by `-0.013426/+0.005763/+0.006305`. The adapter supplies the desired
block correction but damages the global FireDrop regime. This sign-separated
result supports an asymmetric design: no adapter for FireDrop-only samples,
and a spatial correction for BlockDrop samples. The materiality threshold
remains `+0.020` mean primary AP versus D1-ERM.

## Novelty boundary

Missing-modality adapters, dropout, distillation, reliability gating, and
mixtures of experts are existing ideas and are not novelty claims. D8's
bounded method contribution is **failure-geometry factorization within a
paired counterfactual hazard predictor**: global sensor-history absence is
handled through invariant clean-to-corrupt risk consistency, whereas local
observation occlusion activates a small, globally acting future-risk residual
conditioned on block support and extent. It uses one shared forecast backbone,
not routed checkpoints, and has exactly zero adapter effect on clean and
FireDrop-only inputs.

The closest literature remains M3L, MDA-KD, CLoE, and DIS2 listed in the D7
design. The closest internal ablations are D5 (no adapter), D7 (one shared
adapter for both geometries), D4 (input-level invalid token), and P04--P06
(frozen block residuals).

## Fixed architecture and loss

For corrupt decoder features `h_m`, spatial block map `r_block`, and its
broadcast fraction `f_block`:

```text
u = concat(h_m, r_block, broadcast(mean(r_block)))  # 16 + 2 channels
a_block = Conv3x3(18, 16) -> GELU -> Conv1x1(16, 1)
z_adapted = z_m + has_block * a_block(u)
```

The final convolution is zero-initialized. The adapter has exactly 2,625
parameters. `has_block` is one only if the sample contains BlockDrop. Thus
M00 and FireDrop-only inference are exact base-model paths; a combined
FireDrop+BlockDrop training sample uses the block adapter.

The objective is unchanged from D5 and D7:

```text
L = 0.5 * (L_sup(z_clean, y) + L_sup(z_adapted, y))
    + 0.1 * L_CIWC(z_clean.detach(), z_adapted)
```

No rank loss, direct residual target, feature loss, alternate coefficient, or
adapter-width sweep is allowed.

## Matched contract and gates

- Exact corrected B3 initialization, C00 ResNet-18 U-Net, T=1.
- Exact D1/D5 2016--2020 clean/corrupt random stream and loader seed.
- AdamW `1e-3`, batch 32, 3,000 steps, seed 0.
- Jointly train backbone and adapter from a zero adapter initialization.
- 2021 gate: primary mean delta versus D1-ERM at least `+0.020`; primary mean
  must exceed D1-KL and D5; no primary delta below `-0.005`; M00 delta not
  below `-0.010`.
- Only a passing checkpoint may run once on 2022 and 2023. Final retention
  requires positive primary delta in every year, three-year mean at least
  `+0.020`, mean above D1-KL, and every M00 delta at least `-0.010`.
- All training and evaluation run through Slurm. No real-data/model compute on
  the login node.

## Minimal implementation

Add a `block` scope to the existing D7 adapter, trainer, evaluator, and runner.
Keep D7 defaults valid. Focused tests prove FireDrop-only bypass, block-only
gradient flow, fixed parameter count, metadata, and shell dispatch. Do not run
the global test suite, a sweep, or multiple seeds during screening.

# Complete Reliability Prompt Pyramid Design

## Decision

D11 combines two independently positive reliability representations into one
coherent encoder hierarchy. D4's input-stage invalid token improves its D2-STD
control by `+0.004960` primary AP. D10's five post-input prompts improve the
same control by `+0.002915` and produce a `+0.026610` total gain over D1-ERM,
but neither clears D10's attribution rule. The missing ablation is a complete
input-to-semantic reliability prompt pyramid.

## Fixed method

The observable invalid-cell map enters the ResNet-18 U-Net in two forms:

1. **local input prompt:** D4's exact first-convolution token, scaled by local
   invalid coverage under the original convolution kernel;
2. **hierarchical prompts:** D10's exact `(64, 64, 128, 256, 512)` channel
   tokens, scaled by area-averaged invalid coverage at each encoder feature.

```text
h_1 = conv1(features) + local_coverage(invalid) * token_input
h'_l = h_l + area_pool_l(invalid) * token_l,  l=1..5
forecast = decoder(h_0, h'_1, ..., h'_5)
```

All tokens start at zero. The module adds exactly `64 + 1024 = 1088`
parameters. When invalidity is zero, both input and hierarchical prompt terms
are exactly zero for any learned token values.

## Novelty boundary

Prompt learning, missing-modality prompts, multi-scale features, and learned
missing tokens are prior art. D11's bounded contribution candidate is a
**coverage-preserving spatial reliability prompt hierarchy** for a single
partially observed hazard field: the same physical invalid-support map is
propagated through the convolution footprint and every semantic scale, rather
than choosing a modality-level prompt or reconstructing the missing inputs.
The novelty claim includes the complete hierarchy and wildfire forecasting
setting, not any individual token operation.

Closest external work remains Multimodal Prompting with Missing Modalities
(CVPR 2023) and Synergistic Prompting (ICCV 2025), linked in the D10 design.
Closest internal ablations are D2-STD (no reliability module), D4-TOKEN
(input-only), and D10-RPP (post-input only).

## Matched contract

- Exact corrected B3 initialization; C00 ResNet-18 U-Net, T=1.
- Exact D2/D4/D10 single-corrupt-view dataset and random stream.
- 2016--2020 corrected training years, seed 0, batch 64.
- AdamW `1e-3`, exactly 3,000 steps, alpha-disabled focal loss only.
- No auxiliary loss, reconstruction, routing, or coefficient.
- Training/evaluation only through Slurm; no real-data/model compute on login.

## Frozen 2021 gate

D11 advances only if:

1. total mean M01/M06/M07 AP delta versus D1-ERM is at least `+0.020`;
2. module mean delta versus D2-STD is at least `+0.005`;
3. primary mean exceeds both D4-TOKEN and D10-RPP;
4. no primary delta versus D2-STD is below `-0.005`;
5. M00 delta versus D1-ERM and D2-STD is at least `-0.010`.

## Frozen held-out rule

A passing D11 unlocks one fixed Slurm allocation that evaluates both D11 and
the already frozen D2-STD checkpoint on 2022 and 2023. Retain D11 only if its
total primary delta versus D1-ERM is positive in every year and at least
`+0.020` on the three-year mean, its module delta versus D2-STD is positive in
every year and at least `+0.005` on the three-year mean, and all M00 deltas
versus both controls are at least `-0.010`. Test years never select a prompt
scale or variant.

## Minimal implementation

Add one complete wrapper by reusing D4's first-convolution module and D10's
prompt primitive. Reuse the D10 trainer/evaluator through one fixed variant and
add one thin runner. Focused tests cover zero identity, both prompt families'
gradients, exact 1,088-parameter count, checkpoint dispatch, and shell syntax.
No broad suite, sweep, multi-seed screen, or long training run is required.

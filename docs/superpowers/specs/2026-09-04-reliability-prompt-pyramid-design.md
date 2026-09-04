# Reliability Prompt Pyramid Design

## Decision

D10 moves to the quantitatively stronger single-corrupt-view training family.
The matched D2-STD control already improves 2021 mean M01/M06/M07 AP over
D1-ERM by `+0.023695`. D4's first-convolution invalid token raises the same
total delta to `+0.027689`, but its isolated module effect over D2-STD is only
`+0.004960`, narrowly below the original `+0.005` module gate. D10 tests
whether scale-matched reliability prompts are the missing structural upgrade.

## Method and novelty boundary

For the ResNet-18 encoder, let `h_l` be each post-input encoder feature at
channels `(64, 64, 128, 256, 512)`. Downsample the observable spatial
invalidity map `r` to every feature resolution using area averaging and add one
learned channel vector per scale:

```text
c_l = adaptive_average_pool(r, spatial_shape(h_l))
h'_l = h_l + c_l * token_l
forecast = decoder(h_0, h'_1, ..., h'_5)
```

All five tokens start at zero, adding 1,024 parameters. A fully valid input is
therefore an exact base-model path for every parameter value. Unlike D4, the
invalid support is represented at shallow spatial and deep semantic scales
instead of only after the first convolution.

Prompt learning for missing modalities is established prior art, including
Multimodal Prompting with Missing Modalities (CVPR 2023,
<https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Multimodal_Prompting_With_Missing_Modalities_for_Visual_Recognition_CVPR_2023_paper.html>)
and Synergistic Prompting (ICCV 2025,
<https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Synergistic_Prompting_for_Robust_Visual_Recognition_with_Missing_Modalities_ICCV_2025_paper.html>).
Multi-scale features are also standard. D10 does not claim prompts or feature
pyramids as new.

The bounded novelty candidate is a **spatial reliability prompt pyramid for
partial-observation hazard forecasting**: a physically observable missing-cell
support map is area-matched to every convolutional encoder scale and modulates
learned channel tokens, preserving partial block coverage instead of choosing
a modality-level prompt. There is no input reconstruction, external teacher,
transformer prompt sequence, or inference-time model routing.

## Matched experiment contract

- Exact corrected B3 checkpoint initialization and corrected 2016--2020 index.
- C00 ResNet-18 U-Net, T=1; AdamW `1e-3`, batch 64, 3,000 steps, seed 0.
- Exact D2/D4 `ProcessedReliabilityDataset` and random stream: a single view
  with 30% FireDrop plus 30% BlockDrop, not D1's paired objective.
- Exact legacy alpha-disabled focal supervision; no auxiliary loss.
- Matched training control D2-STD. Closest module ablation D4-TOKEN.
- Training and evaluation run only in Slurm. Login-node work is restricted to
  focused synthetic tests, editing, queue inspection, and small JSON parsing.

## Frozen 2021 gates

D10 advances only if all conditions hold:

1. total mean M01/M06/M07 AP delta versus D1-ERM is at least `+0.020`;
2. isolated mean M01/M06/M07 AP delta versus D2-STD is at least `+0.005`;
3. mean M01/M06/M07 AP exceeds D4-TOKEN;
4. no primary scenario is more than `0.005` below D2-STD;
5. M00 is no more than `0.010` below both D1-ERM and D2-STD.

The total gain and module-attributable gain are always reported separately.
Only a passing D10 checkpoint unlocks one fixed 2022/2023 evaluation of both
D10 and the frozen D2-STD control in the same allocation.

## Frozen held-out rule

D10 is retained only if:

1. its total primary delta versus D1-ERM is positive in every year and at least
   `+0.020` on the three-year mean;
2. its module delta versus D2-STD is positive in every year and at least
   `+0.005` on the three-year mean;
3. no yearly M00 delta versus either control is below `-0.010`.

2022--2023 cannot select scale count, pooling, tokens, or any coefficient.

## Minimal implementation

Add one wrapper around the existing encoder/decoder and thin D10
trainer/evaluator/Slurm runners reusing D2's dataset. Focused tests cover zero
identity, exact scale pooling, token-only invalid-region gradients, parameter
count, checkpoint contract, and shell syntax. Do not run a global test suite,
prompt variant, sweep, multi-seed job, or long-budget confirmation during the
screen.

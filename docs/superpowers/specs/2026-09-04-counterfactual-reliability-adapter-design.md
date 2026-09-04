# Counterfactual Reliability Adapter Design

## Objective

D7 is the final candidate in the approved failure ladder. It tests whether a
small, jointly trained forecast-logit correction conditioned on the *type and
support* of an observation failure can provide the missing gain beyond
D5-CIWC. It must improve 2021 mean M01/M06/M07 AP by at least `+0.020` over
D1-ERM before any held-out evaluation; D5 and D6 do not relax this target.

## Evidence and novelty boundary

D5 established that counterfactual impact consistency helps complete
FireDrop (`+0.043494` M01 AP) but barely changes BlockDrop. D4 independently
showed a small `+0.004960` mean block gain from explicitly representing
invalid input support. D6 showed that imposing a global spatial-distribution
penalty is counterproductive. The next minimal test is therefore a learned
correction that sees an explicit failure description while retaining D5's
paired counterfactual training.

Adapters, modality dropout, missing-modality distillation, and reliability
gating are prior art and are not claimed as new. Relevant closest work
includes M3L (WACV 2024, <https://openaccess.thecvf.com/content/WACV2024/html/Maheshwari_Missing_Modality_Robustness_in_Semi-Supervised_Multi-Modal_Semantic_Segmentation_WACV_2024_paper.html>),
MDA-KD (CVPR 2024, <https://openaccess.thecvf.com/content/CVPR2024/html/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.html>),
DIS2 (WACV Workshops 2026,
<https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html>),
and CLoE (2026, <https://arxiv.org/abs/2603.09316>).

The bounded candidate novelty is **intervention-conditioned counterfactual
forecast correction for a single partially observed hazard field**:

1. encode complete active-fire absence and local dynamic-field blockage as
   two distinct, observable reliability maps rather than missing modalities;
2. condition a future-risk logit residual on corrupt decoder features, the
   local block support, and global failure extent;
3. jointly train the base forecast and correction from aligned clean/corrupt
   predictions using counterfactual impact consistency;
4. bypass the correction exactly for fully observed inputs.

This is a defensible novelty candidate, not a claim that literature search can
mathematically guarantee no unpublished or differently named equivalent.

## Fixed architecture

Each processed sample carries two reliability maps:

- `r_fire`: all ones only when the active-fire history is completely removed;
- `r_block`: one on the structured missing block and zero elsewhere.

For corrupt decoder features `h_m`, also broadcast the scalar block fraction
`mean(r_block)` over space. A small adapter predicts one residual map:

```text
u = concat(h_m, r_fire, r_block, broadcast(mean(r_block)))  # 16 + 3 channels
a = Conv3x3(19, 16) -> GELU -> Conv1x1(16, 1)
z_adapted = z_m + any_missing * a(u)
```

The final `1x1` convolution is zero-initialized, so the initial function is
exactly B3. `any_missing` is one per sample if either reliability map contains
a missing observation, otherwise zero. The residual may affect the complete
future-risk field because D5 demonstrated that input corruption support and
future forecast impact need not coincide. The clean path and M00 path bypass
the adapter exactly. The adapter has 2,769 trainable parameters; the base
forecast is jointly trainable, unlike the rejected frozen P04--P08 heads.

## Fixed objective

Let `z_c` be the clean base forecast and `z_a` the reliability-adapted corrupt
forecast. The objective is:

```text
L = 0.5 * (L_sup(z_c, y) + L_sup(z_a, y))
    + 0.1 * L_CIWC(z_c.detach(), z_a)
```

This is D5's exact supervised and CIWC objective; only the explicit adapter
path changes. There is no feature reconstruction, adapter-only auxiliary
loss, rank loss, or coefficient sweep.

## Matched experiment contract

- Exact corrected B3 initialization used by D1/D5/D6.
- C00 ResNet-18 U-Net, T=1, 2016--2020 training years.
- Exact D1 clean/corrupt sample stream: 30% FireDrop and 30% BlockDrop with
  the same random-call order and loader seed.
- AdamW `1e-3`, batch size 32, 3,000 steps, seed 0.
- Base and adapter jointly train; the adapter starts as an exact zero residual.
- Matched control: D1-ERM. Closest ablations: D5-CIWC, D4-TOKEN, and the
  frozen residual P04--P06 family.
- Training/evaluation only through Slurm. Login-node work is limited to source
  edits, focused CPU tests, queue inspection, and parsing small JSON/log files.

## Frozen decision gates

D7 advances beyond 2021 only if all conditions hold:

1. mean M01/M06/M07 AP minus D1-ERM is at least `+0.020`;
2. its primary mean exceeds both D1-KL and D5-CIWC;
3. no primary scenario is more than `0.005` below D1-ERM;
4. M00 is no more than `0.010` below D1-ERM.

Only a passing checkpoint is evaluated once on 2022 and 2023. Final retention
requires a positive D1-ERM primary delta in every year, a three-year mean
primary delta of at least `+0.020`, a mean above D1-KL, and no yearly M00 delta
below `-0.010`. Held-out results never choose architecture or coefficients.

## Minimal implementation

Extend the paired dataset behind an opt-in flag to return the two reliability
maps without changing D1/D5/D6 defaults. Add one adapter wrapper, one trainer,
one evaluator, and thin Slurm dispatch. Focused tests cover random-stream
compatibility, zero-residual identity, clean bypass, gradient flow, objective
composition, checkpoint contract, and shell syntax. No full suite, sweep,
multi-seed run, or broad validation is part of this screen.

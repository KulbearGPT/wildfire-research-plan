# T=5 Severity-Adaptive Reliability Prompting Design

## Goal

Test whether D12's observed-severity-to-prompt-depth mechanism transfers from
the C00 Res18-U-Net (`T=1`, All features) to the C02 Res18-UTAE (`T=5`, Multi
features) without selecting a new threshold or reading held-out years during
model selection.

This is a cross-architecture confirmation experiment, not a new contribution
claim. The candidate is named `D13-SARP-T5` and the matched no-prompt control
is `D13-STD-T5`.

## Why a new model path is required

D12 accepts one 40-channel time step plus one spatial-invalidity channel and
injects prompts into a standard ResNet encoder pyramid. C02 accepts five
33-channel time steps, encodes every step independently, uses LTAE attention
at the last scale, aggregates skip features with the same temporal attention,
and then decodes. A D12 checkpoint cannot be reshaped or directly loaded into
this temporal model.

## Frozen experiment chain

1. Train `B5`, a corrected-index C02 baseline from scratch for 3,000 steps
   with the exact B3 FireDrop + BlockDrop policy.
2. Start both continuations from the same completed B5 checkpoint:
   - `D13-STD-T5`: standard single-corrupt-view continuation;
   - `D13-SARP-T5`: the identical continuation plus severity-adaptive prompts.
3. Use 2016--2020 training, seed 0, AdamW `1e-3`, batch 64, 3,000 continuation
   steps, the existing alpha-disabled focal loss, and the existing corruption
   probabilities and random stream.
4. Evaluate M00/M01/M06/M07 on 2021. Only a passing 2021 result authorizes one
   fixed allocation evaluating both continuation checkpoints on 2022 and
   2023. Held-out results cannot change the threshold, prompt locations, loss,
   optimizer, or training budget.

B1, the corrected clean C02 baseline, is retained only as a secondary
whole-recipe comparator. The primary module attribution is always
`D13-SARP-T5` versus `D13-STD-T5`.

## Temporal SARP architecture

The processed C02 input has shape `[B, 5, 33, H, W]`. Training and controlled
evaluation append the same spatial-invalidity map to every time step, yielding
`[B, 5, 34, H, W]`.

For each sample, severity is the mean invalid fraction over time and space.
The D12 threshold remains exactly `0.375`:

- no spatial invalidity: neither prompt family is active;
- `0 < severity <= 0.375`: the input invalidity is zero and deep prompts are
  added to every time step at all post-input ResNet encoder scales;
- `severity > 0.375`: the deep invalidity is zero and the input reliability
  token is applied independently to every time step before the encoder.

After per-time-step encoding and prompting, the unchanged C02 LTAE aggregates
the last scale. The unchanged temporal aggregator applies the resulting
attention to the prompted skip scales, and the unchanged U-Net decoder
produces the forecast. The prompt channels are the existing ResNet-18
post-input channels `(64, 64, 128, 256, 512)` and the total prompt parameter
count remains 1,088.

M01 does not create a spatial-invalidity mask, so prompts remain inactive for
that scenario; any M01 change comes from the matched corruption-training
continuation rather than hidden prompt routing.

## Processed C02 corruption

The C02 feature selection is frozen to the existing 33-entry `MULTI_FEATURES`
tuple. The active-fire value and binary channels are the positions
corresponding to processed C00 channels 38 and 39. BlockDrop modifies only the
selected dynamic channels that are already modified by the C00 processed-space
corruption and appends one invalidity channel. No feature set or missing value
is re-estimated.

## Frozen gates

The 2021 transfer screen passes only if:

1. mean M06/M07 AP for `D13-SARP-T5` exceeds `D13-STD-T5` by at least `+0.005`;
2. neither M06 nor M07 is more than `0.005` below `D13-STD-T5`;
3. M00 and M01 are each no more than `0.010` below `D13-STD-T5`.

The B1 whole-recipe comparison is reported but does not control prompt-module
attribution. If the screen passes, cross-year transfer is retained only if the
M06/M07 delta versus `D13-STD-T5` is positive in 2021, 2022, and 2023 and its
three-year mean is at least `+0.005`; all M00/M01 deltas must remain at least
`-0.010`.

## Minimal implementation and compute boundary

Add one C02 baseline registration, one processed C02 corruption wrapper, one
temporal SARP wrapper, thin train/evaluate entrypoints, and thin Nibi runners.
Focused tests cover feature-channel mapping, input shape, mild/severe routing,
gradient routing, checkpoint metadata, and runner syntax. Do not run the global
test suite or a hyperparameter sweep.

All real-data training and evaluation run through Slurm. Login-node work is
limited to source inspection, focused synthetic tests, queue queries, result
parsing, and documentation. C02 previously required nearly 32GB host memory,
so every T=5 training job requests 64GB host memory. GPU class is selected from
Nibi's current predicted start time before submission.

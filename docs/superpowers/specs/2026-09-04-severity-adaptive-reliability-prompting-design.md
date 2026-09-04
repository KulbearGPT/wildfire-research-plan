# Severity-Adaptive Reliability Prompting Design

## Decision

D12 uses the measured complementarity of D4 and D10 rather than adding their
prompts simultaneously. D10's post-input pyramid is stronger on 25% BlockDrop
(M06), while D4's input token is stronger on 50% BlockDrop (M07). D11 showed
that unconditional addition nearly passes but does not combine those benefits.
D12 chooses prompt depth from the directly observed block fraction.

## Fixed method

The model contains D4's 64-parameter input token and D10's 1,024-parameter
hierarchical tokens. For invalidity map `r`, compute per-sample fraction
`s=mean(r)` and use the fixed midpoint of the two registered severities:

```text
mild = 0 < s <= 0.375
severe = s > 0.375
r_input = r * severe
r_deep = r * mild
```

The encoder receives `r_input` through D4's local-coverage token. The five
post-input scales receive `r_deep` through D10's area-matched prompts. Thus
M06 uses deep prompts, M07 uses the input token, and M00/M01 use neither.
Combined synthetic corruptions remain classified by their spatial block
fraction; complete FireDrop alone has `s=0`. All tokens start at zero and the
module contains exactly 1,088 parameters.

The threshold `0.375` is not tuned: it is the arithmetic midpoint of the only
two predeclared BlockDrop fractions, `0.25` and `0.50`.

## Novelty boundary

Dynamic prompting and missing-modality prompting are prior art, including SyP
(ICCV 2025). D12 does not claim conditional prompts or severity estimation as
new. Its bounded contribution candidate is **observation-severity-to-prompt-
depth routing for a spatially partial hazard field**: a physical invalid-cell
fraction selects whether reliability enters at high-resolution sensing or
across semantic scales. It is a single shared predictor with parameter prompts,
not a route among separately trained forecasting checkpoints.

Closest internal ablations are D2-STD (no prompt), D4 (severe/input endpoint),
D10 (mild/deep endpoint), and D11 (both prompt families always active).

## Matched contract

- Exact corrected B3 initialization and D2/D4/D10/D11 data stream.
- C00 ResNet-18 U-Net, T=1; 2016--2020 corrected-index training.
- AdamW `1e-3`, batch 64, 3,000 steps, seed 0.
- Single-corrupt-view alpha-disabled focal loss only.
- No auxiliary objective, coefficient, threshold search, or test-time ensemble.
- Training/evaluation only through Slurm; no real-data/model compute on login.

## Frozen gates

On 2021, D12 advances only if:

1. total mean M01/M06/M07 AP delta versus D1-ERM is at least `+0.020`;
2. module mean M06/M07 AP delta versus D2-STD is at least `+0.005`;
3. mean M06/M07 AP exceeds D4, D10, and D11;
4. neither M06 nor M07 is more than `0.005` below D2-STD;
5. M00 and M01 are each no more than `0.010` below D2-STD.

A passing D12 unlocks one Slurm allocation evaluating D12 and frozen D2-STD on
2022 and 2023. Final retention requires total joint delta versus D1-ERM
positive in every year and at least `+0.020` on the three-year mean; module
block delta versus D2-STD positive in every year and at least `+0.005` on the
three-year mean; and all M00/M01 deltas versus D2-STD at least `-0.010`.
Held-out years cannot select the threshold or prompt family.

## Minimal implementation

Add one subclass of the complete prompt wrapper that masks the two reliability
paths by observed severity. Reuse the D10/D11 trainer, evaluator, and loss via
one fixed variant plus a thin runner. Focused tests cover the M06/M07 routing,
zero-input behavior, gradients to only the selected token family, metadata,
and shell syntax. No global test suite, threshold sweep, or multiple seeds.

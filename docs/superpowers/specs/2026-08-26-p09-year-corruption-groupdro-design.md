# P09 Year-Corruption GroupDRO: Fast-Prototype Design

## Decision

P09 is the first prototype after closing the belief-state residual branch. It
tests whether explicitly optimizing the worst training environments makes the
block-robust P02 expert less temporally brittle. The model architecture and
P03 routing rule remain unchanged; only the P02 expert's fine-tuning objective
changes.

## Alternatives considered

1. **Year-corruption GroupDRO fine-tuning (selected).** Fine-tune P02 across
   five training years and three block severities, upweighting high-loss
   groups. This directly targets the observed cross-year failure with one
   bounded run.
2. **Five leave-one-year-out experts.** This would expose temporal variance
   more directly but requires at least five trainings plus a selection rule.
3. **A `T=5` temporal backbone.** This changes both temporal context and
   robustness optimization, making a fast result difficult to interpret.

## Training environments and objective

Use P02's accepted 10,000-step checkpoint as initialization. Training data
remain 2016--2020 with statistics computed only from those years. Each sampled
item receives independent 30% active-fire dropout and one uniformly sampled
block state: clean, 25%, or 50%. The Cartesian product of five years and three
block states defines 15 environments.

Use inverse-year sampling so each training year has equal expected mass. For
each batch, compute the existing focal loss per sample and average it within
each represented environment. Maintain 15 exponentiated GroupDRO weights with
step size `0.1`; update represented groups from detached group losses, then
minimize their normalized weighted loss. Stabilize the log weights by
subtracting their maximum after every update.

Fine-tune all P02 parameters for exactly 3,000 optimizer steps with AdamW at
`1e-4`, batch size 64, seed 0, and no sweep. Log total loss, the maximum-weight
group, and all 15 weights every 100 steps. P00 and the original P02 checkpoint
remain immutable.

## Inference

P09 uses the existing P03 routing contract. P00 predicts everywhere, while the
GroupDRO-fine-tuned expert replaces logits only inside a known M06/M07 missing
mask. M00 and M01 therefore preserve P00 exactly. There is no stochastic
sampling or additional inference module.

## Evaluation and decision rule

The first job evaluates only 2021 M00, M01, M06, and M07 and compares P09 with
P00 and the original P03 in the same process. P09 advances only if all of the
following hold:

1. absolute M00 and M01 AP differences from P00 are each at most `1e-6`;
2. mean M06/M07 AP is strictly greater than P00;
3. mean M06/M07 AP is no more than `0.005` below P03.

If selected, freeze the P09 checkpoint and evaluate it once on 2022 and once
on 2023. Final temporal success requires mean M06/M07 AP improvement over P00
to be strictly positive in both years. Test results cannot trigger tuning,
weight changes, or another P09 run.

If P09 misses the 2021 gate, stop without reading 2022--2023. If it passes 2021
but fails either fixed test year, GroupDRO is retained as a negative result and
the next design must change the source of invariance rather than tune this run.

## Implementation boundary

Add one focused module containing environment labelling, inverse-year sampling
weights, and the GroupDRO objective; one trainer; one evaluator; and two Nibi
launchers for selection and an authorized fixed-year evaluation. Reuse current
checkpoint loading, controlled datasets, metrics, and spatial router.

Verification is limited to one CPU test file for group assignment, balanced
sampling mass, weight movement toward a higher-loss group, and loss gradients,
plus Python compilation and shell syntax. Training and dataset evaluation run
only through Slurm on a Nibi compute node.

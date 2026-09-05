# T=1 Research Roadmap

## Frozen scope

The active project uses Res18-U-Net with `T=1` on the corrected WSTS+ split:
2016--2020 train, 2021 validation, and 2022--2023 final test. The prediction
target is the next UTC calendar day's VIIRS active-fire proxy. Claims are
limited to controlled missingness and do not describe complete fire perimeter
or operational deployment performance.

The baseline chain is fixed to B0 clean training, B2 FireDrop, B3
FireDrop+BlockDrop, D1-ERM, and D2-STD. The retained methods are D1-KL
predictive consistency and D12-SARP severity-adaptive prompting. Their complete
quantitative record is
[`experiments/quantitative_reliability_ledger.md`](experiments/quantitative_reliability_ledger.md).

## Next experiment

The next compute should confirm the unchanged T=1 recipe rather than search a
new architecture against the test years.

1. Run D2-STD and D12-SARP with predetermined additional seeds using the same
   3,000-step optimizer and corruption policy.
2. Aggregate seed variability for M00, M01, M06, and M07 on 2021.
3. If the D12 module delta remains positive and the clean/M01 guardrails hold,
   perform one fixed 2022--2023 evaluation for the frozen seed ensemble.
4. If 3,000-step variance is inconclusive, repeat D2-STD/D12 at one fixed
   longer budget without changing architecture, loss, threshold, or data.

The central comparison is D12 versus D2-STD on mean M06/M07 AP. D1-ERM versus
D12 remains the total-stack comparison, with the documented 2022 M00 caveat.
D1-KL remains an independent contribution and should be reported with its
matched D1-ERM control.

## Decision rule

For additional seeds, report the seed mean and spread before reading the final
test. Continue to the frozen test only if D12 remains positive over D2-STD on
the 2021 M06/M07 mean and does not reduce either M00 or M01 by more than 0.010.
Do not choose a seed, threshold, checkpoint, or budget using 2022--2023.

## Development boundary

A new direction must target one diagnosed failure, state a distinct mechanism,
and compare against the closest retained baseline. Parameter tuning may support
one contribution but should not replace the method contribution. Screen one
small seed-0 run on 2021; stop immediately when its registered gate fails.

Before submitting a Nibi job, inspect the queue and test candidate resource
slices. If the expected wait exceeds ten minutes, use the faster slice when it
requests no more than twice the minimum resource. Training and evaluation stay
inside Slurm; the login node is limited to source work, metadata inspection,
queue checks, and submission.

## Closed work

All rejected, superseded, and T=5 experiments are consolidated in
[`experiments/rejected_experiments.md`](experiments/rejected_experiments.md).
They are not active implementation branches. Full recovery paths are listed in
that document and in the artifact manifest.

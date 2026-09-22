# Inactive experiment archive

These implementations are preserved, not scheduled research. The active route
is [X22+X17 and necessary controls](../CODE_LIFECYCLE.md). Maintenance selection
does not change historical scientific outcomes: D12 or X8 can remain positive
in their recorded setting while being inactive here.

## Find an exploration and its settings

| Exploration | Implementation | Setting / evidence record |
|---|---|---|
| X-series context transport, consistency, restoration, latent transitions, adapters and auxiliary losses | Shared `cross_history/run.py` and `models.py`; methods other than control/cosine_erm/block_specialist are inactive | [Cross-history ledger](../experiments/t1_t5_innovations.md): distinguish early physical batches, repairs, nearest controls, single-history gains and confirmation |
| Earlier corrected T1 D1–D13 experiments, including D12-SARP | `wsts_fast_track/` optional trainers and evaluators | [Quantitative ledger](../experiments/quantitative_reliability_ledger.md) and [rejected experiments](../experiments/rejected_experiments.md); do not import T1 findings into T5 |
| RF normalization, shallow branches and distillation | `cross_history/run_three_directions.py` and `three_directions.py` | [RF settings/results](../experiments/three_directions_results.md) |
| TD conditional normalization, weight merging and distillation | `reproductions/three_directions/` | [TD settings/results](../experiments/three-directions-results.md); distinct from RF despite similar names |
| Alternative architectures | Noncanonical branches in `cross_history/architectures.py` | [Architecture recipe](../research/architectures.md): original bootstrap and matched continuation; not B3/B5 initialization |
| Belief filters, reconstruction and legacy P00 | `wsts_fast_track/` belief and legacy modules | [Historical limitations](../research/negative-results.md), [P00 recipe](../research/legacy-p00.md); invalid initialization is not valid method evidence |
| Natural observation and target-QA diagnostics | VIIRS evaluation and censored-training modules | [Diagnostic reproduction](../research/diagnostics-reproduction.md); diagnostic populations do not replace forecasting comparisons |

For exact IDs, source commits, original commands, reported outcome and current
maintenance status, use [method-inventory.json](../research/method-inventory.json).
Each file entry points to its setting record. `status` describes evidence;
`lifecycle.state` describes maintenance. Cancelled, unrun, invalid and measured
negative entries must remain distinct.

The complete earlier recipe collection is preserved in [English](method-recipes.md)
and [Chinese](method-recipes-zh.md). Commands retain their original switches,
including the broad `cross_history.run` interface. Some directions need recovered
checkpoints or Git history; an available command is not proof that all assets exist.

## Reopening an archived direction

Specify the hypothesis and original setting being revisited. Use its actual
comparator, initialization, input features, physical batch and source record;
do not force every historical experiment into the current mainline protocol.
Record any changed setting as a new experiment. Obtain the relevant execution
scope and budget before compute, and keep old artifacts and outcomes intact.

Imports and checkpoint interfaces were preserved to support historical recovery.
The two retired confirmation bundles are the exception to runnable compatibility:
their hardcoded job replacement actions stay disabled. Reconstruct a new submission
from an approved experiment instead of reusing historical job IDs.

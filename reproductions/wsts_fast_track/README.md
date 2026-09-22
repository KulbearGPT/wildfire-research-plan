# Corrected foundations and preserved experiments

This package provides the corrected B3/B5 foundations and shared data/evaluation
code for the active [X22+X17 route](../cross_history/README.md). Its many other
trainers preserve historical experiments; the package is not an all-active
research collection. See [code lifecycle](../../docs/CODE_LIFECYCLE.md).

## Active foundation path

Use `train_corrected_baseline.py`, `complete_baseline.py` and
`evaluate_corrected_baseline.py` through the portable research submitter.
[Baseline regeneration](../../docs/research/baselines.md) gives complete commands.
B3 is the T1 foundation; B5 is the T5 foundation. Preserve their completion
receipts, checkpoint provenance, normalization and initialization contract.
B0/B1/B2 remain reference/teaching settings, not the selected contribution.

`compute_stats.py`, `contract.py`, `runtime.py`, `corrected_baselines.py`,
`corruption_training.py`, `missingness.py`, `matrix.py`, `evaluation.py`,
`evaluate_missingness.py` and `processed_reliability.py` support the active path.
Do not delete these modules merely because older experiments also import them.

## Inactive research and legacy execution

D1 consistency, D2 reliability variants, D12 severity-adaptive prompting, D5–D11
extensions, temporal prompting, belief states, legacy P00, natural observations
and target-QA diagnostics remain implemented at their original paths. Their
settings and evidence are in [the inventory](../../docs/research/method-inventory.json)
and its source ledgers. Previously positive D12 evidence remains positive but
is inactive under the current project selection.

The `run_*on_nibi.sh` and `run_d12_heldout_on_nibi.sh` scripts preserve the
original site's paths, modules and resource requests. `complete_corrected_baseline.py`
seals that earlier runner's records. These are legacy interfaces; use the portable
handoff and `complete_baseline.py` for current B3/B5 generation.

Archive runtime qualification and tests remain available. Passing them does not
promote an archived method or require rerunning all preserved experiments.

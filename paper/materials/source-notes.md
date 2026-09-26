# Evidence sources and interpretation

The materials package collects existing experiments. Collection, numerical
normalization, plotting, and packaging use Slurm CPU allocations. No model was
retrained or reevaluated for this collection, and no checkpoint or dataset tensor
is included. The collector is `scripts/paper/collect_evidence.py`.

## Source hierarchy

1. Original run `summary.json` files provide scenario metrics and evaluation
   populations. The collector requires the recorded history, seed, year,
   method, severity (where relevant), and the complete four-scenario AP vector
   to match a retained aggregate. Matching records remain explicit rather than
   being selected by newest modification time or highest score.
2. Completed comparison JSONs provide the historical pairing and composition.
   Canonical results use the exact 18 cells: two configurations, three seeds,
   and three years. Named comparators and routing components are checked
   numerically against the source rows.
3. Run sidecars record available configuration information. In particular,
   `started.json` records intended settings; it does not prove that all planned
   updates completed. Historical files do not contain today's full provenance
   contract. Collection does not retrospectively certify checkpoint hashes,
   training completion, or HDF5 byte identity that those files do not establish.
4. The complete method inventory preserves all 100 recorded directions,
   including unrun, invalid, failed, limited, and archived results. A positive
   entry is not automatically an active contribution.

Every copied JSON retains its original bytes, SHA-256, original path, and a
package-local source path. Historical reports and their original gate flags
are preserved as evidence. Their pass flags do not override later limits on
attribution, test-set exposure, or the owner-selected active scope.

## A consequential route-definition discrepancy

`x22-x17-complete-route-final.json` uses **X22 for both M00 and M01**, the
25% specialist for M06, and the 50% specialist for M07. For example, its
T1 / seed 0 / 2021 M00 AP is `0.5936950362421036`, compared with the fresh
ERM reference `0.5850656354320689`; the recorded clean difference is
`0.008629400810034715`.

The current mainline composition code and later teacher manifest instead use
**ERM for M00**, X22 for M01, and the same two block specialists. Its clean
difference is zero by construction. This collection preserves the historical
route under `X22_X17_HISTORICAL`; the plotting/build stage can construct the
current route with an explicitly different identifier. The primary metric
excludes M00, so its reported gain is unchanged. Clean performance and model
storage/training counts must follow the actual route definition: the historical
composition needs three distinct predictors, while the current ERM-clean route
needs four. Shared frozen initialization is separate from continuation cost.

The X8 archived routes also have different clean branches: its comparison with
global consistency uses the global-consistency model on M00, whereas its
comparison with fresh ERM uses ERM on M00. These are stored separately rather
than relabeled as one identical system.

## Normalized data contract

`evidence.json` contains `rows`, `sources`, `archived_comparisons`, the complete
`inventory`, `repository_evidence`, `gaps`, and collection execution metadata.

Each normalized row is one method / configuration / training seed / evaluation
year / scenario AP. AP uses the original 0–1 scale. `source_id`, `source_rel`,
`source_role`, and, for aggregates, `source_row_index` locate the saved number.
`original_summary_source_ids` links matching evaluation summaries;
`run_metadata_source_ids` links available sidecars. `sample_count` is taken from
original metrics, not filled from a presumed year-to-count rule. Where the
matching original records agree on all metrics, `metrics` also retains F1, IoU,
precision, recall, Brier score, loss, and pixel count. These supplementary
metrics retain their original evaluation definitions.

For a composed row, `component_method` identifies which model supplied that
scenario's metric. This is composition of already evaluated scenario results,
not a new end-to-end inference run, ensemble, or pooled precision–recall curve.

| Identifier | Interpretation |
| --- | --- |
| `ERM` | Fresh matched constant-learning-rate continuation |
| `X22` | Cosine-scheduled continuation, one model |
| `X14_RAW` | Mixed-severity block specialist, all original scenarios |
| `X14_ROUTE` | ERM M00/M01, mixed-severity specialist M06/M07 |
| `X17_MILD`, `X17_SEVERE` | Fixed 25% and 50% block specialists |
| `X17_ROUTE` | ERM M00/M01, fixed-severity specialists M06/M07 |
| `X22_X17_HISTORICAL` | Historical X22 M00/M01 plus fixed-severity experts |
| `FROZEN` | Original B3/B5 references evaluated under the paired contract |
| `HIST_*` | Older corrected T1 baseline/D-series chain, separate provenance |
| `RF_*` | Archived reliability-fusion screen; two-bank batch-stat BN campaign |
| `TD_*` | Distinct archived three-directions screen; four-bank BN/merge/KD campaign |

Frozen references have `seed: null`: each configuration/year uses one shared
source checkpoint. Reusing that reference against three continuation seeds
does not create three independently trained frozen baselines. The frozen audit
uses B3 jobs 21212205 / 21326336 / 21326337 and B5 jobs 21212423 / 21323907 /
21323908. Older `HIST_B3` rows belong to the historical corrected chain and are
not silently substituted into this paired audit.

`archived_comparisons` retains compact complete JSONs for the other available
cross-history comparisons, including failed screens and partial campaigns.
Their original nearest controls must be used when interpreting an improvement;
the word `control` in two different files does not imply the same model.

## Limits relevant to a paper

- T1 and T5 change architecture and feature selection together with history.
  They are two forecasting configurations, not an isolated history ablation.
- AP is pixel-level average precision from each original evaluation. Primary
  is the arithmetic mean of M01/M06/M07; block is the mean of M06/M07. An
  average of APs is not AP computed after pooling predictions.
- The paired evaluation populations are 3,181 / 2,856 / 2,102 samples in
  2021 / 2022 / 2023. Matching counts do not prove byte-identical datasets.
- Seed variability is descriptive variability across three runs. It is not
  an event-bootstrap confidence interval or proof of statistical significance.
- The historical test years were inspected during development. Further
  post-hoc combinations, including X22 plus mixed-severity X14, are retrospective
  analyses rather than independently confirmed new methods.
- X17 has positive total effects against ERM but does not consistently beat
  X14 in the closest-control historical-year comparisons. Preserve those
  adverse cells in figures and tables.
- Controlled missingness and a next-day active-fire proxy do not establish
  natural missingness robustness, complete burned-perimeter prediction, or
  operational deployment performance.
- There are no newly generated qualitative predictions, event-level uncertainty
  estimates, natural-corruption experiments, or same-protocol comparisons with
  every recent external method in this package. Missing evidence is disclosed
  rather than illustrated with invented examples.

For portable regeneration, supply `--runs` pointing to the original run tree,
or rebuild plots from the package's already collected `evidence.json` and copied
sources. Teacher checkpoint hashes in existing manifests are historical recorded
hashes; this collection does not reread model weight files to recompute them.

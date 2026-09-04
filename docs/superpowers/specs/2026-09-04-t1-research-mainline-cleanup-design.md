# T=1 Research Mainline Cleanup Design

## Objective

Reduce the working tree and the active Nibi run directory to the evidence needed
for the current `T=1` paper direction, without destroying the complete historical
record. The cleaned branch must retain the reproducible baseline chain, the two
quantitatively supported methods, and one compact record of rejected work.

This is a research-record cleanup, not a new experiment. It must not submit jobs
or run training/evaluation on a login node.

## Recovery boundary

The complete pre-cleanup Git state is frozen at branch
`archive/pre-t1-cleanup-2026-09-04`, commit `4b843ad`.

Git cannot preserve ignored checkpoints and logs. Obsolete external artifacts
will therefore be moved, not deleted, from
`/project/6085198/kulbear/wildfire/runs` to
`/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs`.
The move remains on the same project filesystem and requires no compression or
large computation. A manifest records every source path, destination path, size,
classification, and reason. Active jobs must be checked before any move.

## Retained scientific chain

### Foundations and published baselines

- Phase 0 inventory, repair, split, field contract, deterministic no-fire and
  persistence baselines.
- Official Res18-U-Net `T=1` reproduction controls and released-weight evidence.
- Nibi configuration and minimal Slurm helpers.
- Related-work and baseline-reproduction pages, revised so that current project
  status points only to the `T=1` mainline.

### Corrected `T=1` experimental chain

- `B0`: corrected-index clean C00 baseline.
- `B2`: FireDrop specialist baseline.
- `B3`: FireDrop + BlockDrop baseline.
- `D1-ERM`: matched continuation control.
- `D1-KL`: predictive-consistency contribution.
- `D2-STD`: matched standard-convolution control.
- `D12-SARP`: severity-adaptive reliability prompting contribution.

Their selected 2021 and fixed 2022--2023 evaluation records, training records,
checkpoints, Slurm provenance, and the source files required to reproduce them
remain in the active tree or active run directory.

## Removed from the active mainline

- All `T=5` implementations and artifacts, including B1, B5, and D13.
- Legacy invalid-resolver C00/C02 runs and P00--P13 implementations and
  artifacts. Their validated insights are represented by B2/B3 or the rejected
  result record.
- Belief filter, attention, and reconstruction probes.
- Natural VIIRS reliability and target-QA prototype implementations and failed
  cohort attempts. Their quantitative observation remains in the rejected
  result record but they are not part of the forecasting method.
- Rejected or gated candidates B4, D2-RNC, D3, and D4--D11 as independent
  experiment entry points, runners, tests, checkpoints, and logs.
- Failed retries, cancelled jobs, OOM runs, stale promotion machinery, obsolete
  per-experiment plans/specifications, and generated test/cache products.

## D12 consolidation

D12 reuses mechanisms first isolated by D4, D10, and D11. Those mechanisms are
not obsolete dependencies and must not be deleted. They will be represented as
parts of the D12 implementation rather than retained as runnable rejected
experiments:

- invalid-region input token;
- multi-scale reliability prompts;
- complete input-to-semantic prompt path;
- observed-severity routing between the input and deeper prompts.

The active trainer, evaluator, checkpoint validator, and Nibi runners expose
only D12. Historical D4/D10/D11 metrics remain in the rejected-results summary
as ablation evidence.

## Documentation layout

The cleaned branch has three kinds of experiment documentation:

1. foundation and official `T=1` reproduction reports;
2. one current `T=1` results document containing the retained baseline and
   contribution tables;
3. one rejected-results archive containing concise metrics, job identifiers,
   rejection reasons, and pointers to the archive branch/artifact directory.

README and the public status page describe only the current scope and link to
these documents. Historical narrative and operational monitoring transcripts
are removed from the active branch because they remain recoverable from the
archive branch.

## Artifact policy

The active run directory retains only the selected runs and their direct
held-out evaluations for B0, B2, B3, D1-ERM, D1-KL, D2-STD, and D12-SARP.
Shared datasets, environments, upstream-code caches, manifests, and official
weights are outside this cleanup and must not be moved.

Obsolete top-level Slurm logs and run directories are moved to the external
archive. Nothing is recursively targeted through an unresolved variable or
glob: the cleanup uses an explicit generated manifest and verifies that each
source is inside the exact run root before moving it.

## Verification and commits

No full test suite, dataset scan, training, or GPU evaluation is required.
Verification is limited to:

- clean Git status before each stage;
- no active Slurm jobs before artifact movement;
- retained-file import/reference checks;
- shell syntax checks for retained runners;
- targeted lightweight tests for the retained baseline/D1/D12 contracts;
- link/path searches for removed identifiers;
- manifest reconciliation between active and archived artifacts.

Commits are split into recovery/design, code-mainline cleanup,
documentation cleanup, and artifact-manifest updates so each stage remains easy
to inspect or revert.

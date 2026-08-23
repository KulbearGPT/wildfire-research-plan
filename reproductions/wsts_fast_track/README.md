# WSTS+ Fast Clean-Baseline Track

This track begins new learned experiments on Alliance Nibi while preserving the
minimum scientific boundaries needed for interpretable screening.

## Frozen screening contract

- Data: 999 active-fixed WSTS+ event HDF5 files.
- Train: 2016–2020.
- Validation: 2021.
- Withheld: 2022–2023; the entrypoint refuses test evaluation.
- Seed: 0.
- Budget: 3,000 optimizer steps per model.
- Hardware: one Nibi H100.

| ID | Model | History | Features | Status | Nibi job | Best 2021 AP |
| --- | --- | ---: | --- | --- | --- | ---: |
| C00 | Res18-UNet | 1 | All (40) | Preparing | — | — |
| C02 | Res18-UTAE | 5 | Multi (33) | Preparing | — | — |

The first runs are screening evidence. A model must be promoted to the frozen
10,000-step budget before the project makes a clean-performance claim.

## Execution controls

- `contract.py` defines the two literal experiment specifications and checks
  the lightweight 999-event directory layout.
- `compute_stats.py` computes normalization and positive-fire statistics from
  the five training years only.
- `entrypoint.py` installs the frozen split into the pinned upstream runtime,
  loads train-only statistics, refuses split/test overrides, and runs the
  upstream Lightning CLI.

Large data, environments, checkpoints, logs, and run evidence remain outside
Git under the Nibi project filesystem.

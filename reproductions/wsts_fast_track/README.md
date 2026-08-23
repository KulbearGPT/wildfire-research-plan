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

## Clean follow-on matrix

Promotion is a fresh training run, not a continuation from a screening
checkpoint. Both controls advance to the matched seed-0 budget after both
screening records pass; the 3,000-step AP ranking does not eliminate either
control.

| Run ID | Model | Seed | Steps | State |
| --- | --- | ---: | ---: | --- |
| `C00-S0-3K` | Res18-UNet | 0 | 3,000 | Active screening |
| `C02-S0-3K` | Res18-UTAE | 0 | 3,000 | Active screening |
| `C00-S0-10K` | Res18-UNet | 0 | 10,000 | Promotable after both 3K records pass |
| `C02-S0-10K` | Res18-UTAE | 0 | 10,000 | Promotable after both 3K records pass |
| `C00-S1-10K` | Res18-UNet | 1 | 10,000 | Gated on both seed-0 10K runs |
| `C00-S2-10K` | Res18-UNet | 2 | 10,000 | Gated on both seed-0 10K runs |
| `C02-S1-10K` | Res18-UTAE | 1 | 10,000 | Gated on both seed-0 10K runs |
| `C02-S2-10K` | Res18-UTAE | 2 | 10,000 | Gated on both seed-0 10K runs |

Seed-0 promotion continues to train on 2016–2020 and select on 2021.
The 2022–2023 test years remain withheld.

## Declared controlled-missingness matrix

These scenarios are frozen identifiers, not implemented or launchable
experiments yet. Later spatial masks must be generated from event/date/scenario
identity without reading the target.

| ID | Condition | Raw inputs | Severity |
| --- | --- | --- | --- |
| `M00` | Clean reference | None | 0 |
| `M01` | Active-fire history absent | 22 | All history |
| `M02` | Active-fire history stale | 22 | One day |
| `M03` | Observed weather absent | 5–11 | Full modality |
| `M04` | Forecast weather absent | 17–21 | Full modality |
| `M05` | Observed and forecast weather absent | 5–11, 17–21 | Both modalities |
| `M06` | Structured spatial blocks | All dynamic inputs | 25% area |
| `M07` | Structured spatial blocks | All dynamic inputs | 50% area |

## Execution controls

- `contract.py` defines the two literal experiment specifications and checks
  the lightweight 999-event directory layout.
- `compute_stats.py` computes normalization and positive-fire statistics from
  the five training years only.
- `entrypoint.py` installs the frozen split into the pinned upstream runtime,
  loads train-only statistics, refuses split/test overrides, and runs the
  upstream Lightning CLI.
- `matrix.py` is the authoritative typed registry for clean follow-on runs and
  declared corruption scenarios.
- `promotion.py` validates immutable screening results and renders reviewed
  seed-0 10K manifests. It never invokes Slurm or retries a run.

Export the canonical matrix once:

```bash
python -m reproductions.wsts_fast_track.promotion matrix \
  --output artifacts/fast-track-matrix.json
```

After both screening jobs have produced reviewed `completed.json` files,
render one seed-0 promotion manifest:

```bash
python -m reproductions.wsts_fast_track.promotion render \
  --run-id C00-S0-10K \
  --result "$c00Completed" \
  --result "$c02Completed" \
  --upstream-root "$upstreamRoot" \
  --data-root "$dataRoot" \
  --run-root "$newRunRoot" \
  --stats-path "$trainingStats" \
  --output artifacts/C00-S0-10K-promotion.json
```

Review the manifest and pass its `command` explicitly to the existing cluster
submission layer. Rendering the manifest is not job submission. Existing
outputs are never overwritten.

Large data, environments, checkpoints, logs, and run evidence remain outside
Git under the Nibi project filesystem.

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

| ID | Model | History | Features | Parameters | Nibi job | Wall | Best step | 2021 AP | 2021 F1 | 2021 loss | CUDA peak |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C00 | Res18-UNet | 1 | All (40) | 14,444,241 | `20346980` | 580 s | 2,343 | 0.547372 | 0.402298 | 0.006614 | 3.88 GB |
| C02 | Res18-UTAE | 5 | Multi (33) | 14,704,785 | `20346981` | 1,988 s | 2,941 | 0.584540 | 0.438810 | 0.006639 | 4.99 GB |

Both jobs completed at exactly 3,000 optimizer steps with Slurm exit `0:0`, one
loadable best checkpoint, finite metrics, and no test action. These runs are
screening evidence. The AP difference does not select a winner, and neither
single-seed result is a clean-performance or held-out-test claim.

## Clean follow-on matrix

Promotion is a fresh training run, not a continuation from a screening
checkpoint. Both controls advance to the matched seed-0 budget after both
screening records pass; the 3,000-step AP ranking does not eliminate either
control.

| Run ID | Model | Seed | Steps | State |
| --- | --- | ---: | ---: | --- |
| `C00-S0-3K` | Res18-UNet | 0 | 3,000 | Pass — job `20346980` |
| `C02-S0-3K` | Res18-UTAE | 0 | 3,000 | Pass — job `20346981` |
| `C00-S0-10K` | Res18-UNet | 0 | 10,000 | Submitted fresh — job `20353582` |
| `C02-S0-10K` | Res18-UTAE | 0 | 10,000 | Submitted fresh — job `20353584` |
| `C00-S1-10K` | Res18-UNet | 1 | 10,000 | Manifest code ready; gated on both seed-0 10K records |
| `C00-S2-10K` | Res18-UNet | 2 | 10,000 | Manifest code ready; gated on both seed-0 10K records |
| `C02-S1-10K` | Res18-UTAE | 1 | 10,000 | Manifest code ready; gated on both seed-0 10K records |
| `C02-S2-10K` | Res18-UTAE | 2 | 10,000 | Manifest code ready; gated on both seed-0 10K records |

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
- `promotion.py` validates immutable prerequisite results and renders reviewed
  seed-0 promotion or seed 1/2 replication manifests. It never invokes Slurm
  or retries a run.
- `completion.py` validates the 10,000-step marker, checkpoint, finite final
  validation metrics, and CUDA observation for every declared seed, then
  creates a non-overwriting completion record with the test lock preserved.

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

After both seed-0 10K completion records pass, the same command renders a
replication by changing `--run-id` to one of `C00-S1-10K`, `C00-S2-10K`,
`C02-S1-10K`, or `C02-S2-10K` and supplying the two seed-0 10K records. A 3K
record or a nonzero-seed prerequisite is rejected.

The post-control execution order and the M00--M07 evaluation boundary are
frozen in
[`docs/superpowers/specs/2026-08-23-wsts-post-control-design.md`](../../docs/superpowers/specs/2026-08-23-wsts-post-control-design.md).
In particular, the upstream validation loader uses training augmentation, so
formal controlled-missingness results require a separate deterministic
`is_train=false` evaluation path. No real 2022--2023 file is opened while that
path is only being implemented and tested.

Large data, environments, checkpoints, logs, and run evidence remain outside
Git under the Nibi project filesystem.

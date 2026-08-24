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
| `C00-S0-10K` | Res18-UNet | 0 | 10,000 | Pass — job `20353582`, 2021 AP 0.583933 |
| `C02-S0-10K` | Res18-UTAE | 0 | 10,000 | Submitted fresh — job `20353584` |
| `C00-S1-10K` | Res18-UNet | 1 | 10,000 | Manifest code ready; gated on both seed-0 10K records |
| `C00-S2-10K` | Res18-UNet | 2 | 10,000 | Manifest code ready; gated on both seed-0 10K records |
| `C02-S1-10K` | Res18-UTAE | 1 | 10,000 | Manifest code ready; gated on both seed-0 10K records |
| `C02-S2-10K` | Res18-UTAE | 2 | 10,000 | Manifest code ready; gated on both seed-0 10K records |

Seed-0 promotion continues to train on 2016–2020 and select on 2021.
The 2022–2023 test years remain withheld.

## Implemented controlled-missingness matrix

These scenarios are deterministic evaluation-only transformations. They share
one effective six-day target population, use an `is_train=false` center crop,
and generate spatial masks from event/date/scenario identity without reading
the target, prediction, or metric.

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
  implemented corruption scenarios.
- `missingness.py` applies the versioned M00--M07 raw-space transformations;
  `evaluation.py` supplies deterministic `is_train=false` datasets with an
  identical target population for C00 and C02. Held-out years require an
  explicit formal-manifest authorization.
- `missingness_manifest.py` renders 2021 engineering tasks from any available
  10K checkpoint, but requires all six clean records for the formal 96-task
  matrix over 2022 and 2023.
- `evaluate_missingness.py` strict-loads a frozen checkpoint and records AP,
  F1, IoU, precision, recall, and loss without retraining or checkpoint
  selection. `aggregate_missingness.py` reports same-year deltas from M00 and
  seed mean/dispersion.
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

`run_replication_on_nibi.sh RUN_ID MANIFEST` is the reviewed Nibi batch
payload for those four runs. It accepts no seed-0 or screening ID, rechecks the
manifest's split/test boundary, snapshots committed project code with
`git archive`, and seals the result through `completion.py`. The payload never
calls `sbatch`; submission remains a separate explicit action after the real
seed-0 gate passes.

An available 10K checkpoint can render a non-scientific 2021 engineering
matrix immediately:

```bash
python -m reproductions.wsts_fast_track.missingness_manifest \
  --mode engineering \
  --record C00-S0-10K="$c00Seed0Completed" \
  --output-root "$engineeringResultRoot" \
  --output "$engineeringManifest"
```

Each declared task is executed by the reviewed Nibi payload
`run_missingness_on_nibi.sh MANIFEST EVALUATION_ID`. The payload snapshots the
committed project tree and never submits another job. Formal manifest rendering
uses `--mode formal` and six `--record RUN_ID=PATH` arguments; an incomplete
clean replication set is rejected before any 2022--2023 dataset can be built.

The post-control execution order and the M00--M07 evaluation boundary are
frozen in
[`docs/superpowers/specs/2026-08-23-wsts-post-control-design.md`](../../docs/superpowers/specs/2026-08-23-wsts-post-control-design.md).
The upstream validation loader uses training augmentation, so every controlled
evaluation instead uses the separate deterministic path above. Until the six
clean records exist, only synthetic fixtures and explicitly non-scientific
2021 engineering tasks are runnable.

Large data, environments, checkpoints, logs, and run evidence remain outside
Git under the Nibi project filesystem.

## Rapid robustness prototypes

P00 (`FireDrop-C00`) trains C00 from scratch for 10,000 steps while dropping
the raw active-fire-history feature for 30% of training samples. Its 2021
controlled AP was 0.585322 on M00 and 0.299465 on M01, versus 0.584368 and
0.045920 for the clean C00 checkpoint. The one-time 2022 and 2023 checks also
showed large positive M01 deltas while keeping clean AP within the frozen
rapid-triage tolerance.

P01 (`FireDropMask-C00`) is the next one-variable experiment. It retains the
same model, split, seed, budget, and dropout probability, and appends one
binary active-fire-validity channel after upstream preprocessing. Its first
evaluation is limited to 2021 M00/M01/M02/M07; there is no probability sweep
or additional-seed launch in this prototype gate.

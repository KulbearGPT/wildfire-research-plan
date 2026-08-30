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

P01 (`FireDropMask-C00`) was the next one-variable experiment. It retained the
same model, split, seed, budget, and dropout probability, and appends one
binary active-fire-validity channel after upstream preprocessing. Its first
evaluation is limited to 2021 M00/M01/M02/M07; there is no probability sweep
or additional-seed launch in this prototype gate.

P01 jobs `20414716`/`20414717` completed but did not beat P00: its AP deltas
were -0.0095 on M00, -0.0111 on M01, -0.0250 on M02, and +0.0002 on M07.
P02 then retained FireDrop and independently applied structured 25%/50%
BlockDrop with probability 0.3. Jobs `20454659`/`20454660` completed; against
the P00 comparison job `20454658`, P02 changed AP by -0.0200 on M00, -0.0322
on M01, +0.0183 on M06, and +0.0303 on M07. P02 failed its preservation gate
despite the M07 gain. P00 remains the accepted rapid baseline; P01/P02 do not
advance to held-out evaluation or additional seeds.

P03 (`SpatialExpertRouter-P00-P02`) reuses those two frozen checkpoints with no
training. It selects P02 logits only at pixels inside the deterministic M06/M07
missing block and P00 logits elsewhere. Job `20458324` completed the 2021
M00/M01/M06/M07 screen. AP was 0.585322, 0.299465, 0.350878, and 0.166982,
respectively: M00/M01 exactly preserve P00, while M06/M07 improve P00 by 0.0339
and 0.0378. P03 was therefore selected as the leading rapid routed prototype
for the one-time held-out comparison reported at the end of this section.

P04 (`FrozenP00-SpatialResidualGate`) freezes P00, reuses its final 16-channel
decoder feature, and trains only a 17-parameter `1x1` residual head inside the
known missing block. Job `20462018` completed 1,000 training steps plus the
2021 screen in 3:37. M00/M01 AP exactly matched P00; M06/M07 AP was 0.332083/
0.139913, improving P00 by 0.0151/0.0107 but trailing P03 by 0.0188/0.0271.
P04 is a useful single-forward compute control, not the promoted prototype.

P05 enlarged the P04 correction to one 145-parameter `3x3` convolution while
holding every other choice fixed. Job `20464221` finished in 3:30. M00/M01
again exactly matched P00, but M06/M07 AP fell slightly to 0.327723/0.138767.
The failure to improve P04 rejects kernel size as the immediate bottleneck.

P06 shared P00 through the first four decoder blocks and trained a copied final
decoder block plus head. Job `20464396` produced M06/M07 AP of 0.330240/
0.141035, again below P03. The cheap capacity escalation therefore stopped.

P03 then received its one-time held-out comparison. Against P00, its M06/M07
AP deltas were -0.0231/-0.0261 in 2022 (job `20465333`) and +0.0135/+0.0096 in
2023 (job `20465334`); M00/M01 were exactly preserved. This mixed result fails
the cross-year robustness requirement. P03 is not promoted, P00 remains the
mainline checkpoint, and 2022--2023 are closed to further prototype tuning.

## Target-QA censored-loss prototype

The fixed 24-sample 2021 target audit found 49,310/393,216 (12.54%) zero-label
pixels without a reliable target-day VIIRS observation. P00 scored 0.382293 AP
on all pixels and 0.398789 AP after retaining all positives plus only reliably
observed zeros. The earlier attention model scored 0.382579 and 0.397911,
respectively, so target QA reverses its otherwise negligible standard-label
advantage. Attention is not the baseline for the next test.

The next experiment changes only the loss mask between two matched P00
fine-tunes. Both use the same deterministic 40-event training cohort (eight per
year in 2016--2020, balanced four positive/four zero targets), P00
initialization, seed 0, sampled crops, AdamW at `1e-4`, batch 64, and 3,000
steps. No reliability channel is added to the model.

| Candidate | Changed variable | Direct baseline | 2021 decision |
| --- | --- | --- | --- |
| Legacy cohort control | Ordinary alpha-disabled focal on every pixel | Frozen P00 | Pending job `20828988` |
| Target-censored cohort | Exclude only unreliable zero target pixels | Matched legacy cohort control | Pending job `20828989` |
| Final comparison | Standard and QA-censored metrics on the fixed 24 samples | Both rows above, with frozen P00 also reported | Pending job `20828992` |

CPU job `20828982` selects the frozen cohort and extracts its 40 target-day QA
fields. It is the only data-processing job. The two training jobs depend on it
and run in parallel on separate H100 MIG compute allocations; evaluation
depends on both. The login node performed only Git, scheduler queries, and
submission. Promotion requires higher QA-censored AP, lower QA Brier and focal
loss, and non-worse standard AP against the matched legacy control. An AP
effect below `1e-3` or conflicting directions is recorded as null/reject; this
screen does not open 2022--2023.

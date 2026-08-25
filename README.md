# Reliable Next-Day Wildfire Spread Forecasting

This repository is the research record and prototype workspace for reliable
next-day wildfire spread forecasting under asynchronous or missing
observations. It hosts the public research-plan website, audited data-contract
code, completed baseline-reproduction controls, and the small amount of cluster
infrastructure needed to run the next experiments.

## Current status

- The eight-year WSTS+ audit covers 999 valid event-level HDF5 files and freezes
  the temporal split to 2016–2020 train, 2021 validation, and 2022–2023 test.
- The fixed no-fire and latest-mask persistence rules have been evaluated under
  that split. See [`docs/experiments/phase0.md`](docs/experiments/phase0.md).
- The official Res18-U-Net, `T=1`, Fold-2 training path has been executed, and
  all twelve official released weights have been evaluated independently. See
  [`docs/experiments/res18_unet_t1_reproduction.md`](docs/experiments/res18_unet_t1_reproduction.md).
- The active-fixed 999-event WSTS+ tree is prepared on Nibi. All six C00/C02
  seed-0/1/2 10,000-step clean runs and the 96-task M00--M07 formal
  missingness matrix have completed. Active-fire-history loss (M01) is the
  dominant diagnosed failure. See
  [`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/).
- P00 (`FireDrop-C00`) remains the accepted single-checkpoint baseline. P01's
  explicit active-fire-validity channel did not improve P00, while P02's
  structured BlockDrop exposed a useful spatial-missingness expert but harmed
  other regimes. P03 now routes frozen P00/P02 logits only inside known missing
  blocks: it exactly preserves P00 on M00/M01 and improves AP by 0.0339 on M06
  and 0.0378 on M07.

## Scientific boundary

The target is a next-calendar-day active-fire proxy, not a complete fire
perimeter. The current public data contract supports prespecified controlled
missingness. It does not establish natural-missingness or
operational-deployment performance because acquisition time, availability time,
QA, coverage, and target-validity provenance are not all recoverable.

The twelve-fold publication supports released-weight executable
reproducibility. Its agreement with the paper reference does not prove
paper-table provenance identity. The local Windows/RTX 3090 evidence remains a
frozen lineage; future cluster runs begin a linked but separate lineage.

## Repository map

- [`index.html`](index.html) — public research plan and status.
- [`related-work/`](related-work/) — Lecture 02, research foundations and
  related work.
- [`baseline-reproduction/`](baseline-reproduction/) — Lecture 03, complete
  baseline-reproduction walkthrough.
- [`src/wildfire_phase0/`](src/wildfire_phase0/) — data audit, repair, split,
  rule-baseline, and reporting code.
- [`reproductions/wsts_res18_unet_t1/`](reproductions/wsts_res18_unet_t1/) —
  pinned official-code reproduction controls and manifests.
- [`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/) — frozen
  WSTS+ split and C00/C02 seed-0 screening controls.
- [`docs/experiments/`](docs/experiments/) — reviewed result summaries and
  claim boundaries.
- [`docs/research-roadmap.md`](docs/research-roadmap.md) — authoritative order
  of future experiments.
- [`docs/cluster-migration.md`](docs/cluster-migration.md) — Slurm migration and
  first-run guide.

## Fast local verification

Use a Python 3.13 environment that satisfies `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Large datasets, weights, environments, caches, checkpoints, and run artifacts
are deliberately excluded from Git.

## Slurm cluster quick start

The current deployment target is **Alliance Nibi**. The cluster layer is a thin
research boundary, not a deployment system. Nibi supplies one ignored profile
containing the user's account, partition, module, and path values. The generic
scripts validate the profile and data manifest, print the exact `sbatch`
command in dry-run mode, and execute one scientific command once. Each job
performs a formal preflight before the scientific command and seals the
production-data, runtime, command, upstream-code, and applicable
official-weight identities into its run record. Follow
[`docs/cluster-migration.md`](docs/cluster-migration.md) after cloning the
reviewed migration commit.

## Active experiment

P03 spatial expert routing completed in job `20458324`. It is a training-free
2021 engineering prototype: P00 is used everywhere except inside the known
M06/M07 missing block, where P02 is used. AP is 0.585322/0.299465/0.350878/
0.166982 on M00/M01/M06/M07. Relative to P00 this is exactly neutral on
M00/M01 and +0.0339/+0.0378 on M06/M07, so P03 is the current leading routed
prototype. No held-out evaluation or additional seed is implied by this
screening result.

P04 tested whether P03 could be compressed to one P00 forward pass plus a
17-parameter `1x1` residual head trained for 1,000 steps. Job `20462018`
preserved M00/M01 exactly and improved P00 AP by 0.0151/0.0107 on M06/M07,
but remained 0.0188/0.0271 below P03. P04 is retained as the minimum-compute
matched control and is not promoted.

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
- A one-step Res18-U-Net `T=1` engineering smoke has completed on an Nibi H100.
  The active Fast Experiment Track now prepares the full 999-event WSTS+ tree
  and runs the first C00/C02 learned controls. See
  [`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/).

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

## Next experiment

The immediate experiments are C00 Res18-U-Net `T=1` All and C02 Res18-UTAE
`T=5` Multi, each with seed 0 and 3,000 optimizer steps on the frozen
2016–2020 train / 2021 validation split. Cluster checkpoint equivalence and the
Fold-2 positive-weight sensitivity study are deferred so they do not block new
model experiments. The 2022–2023 test years remain withheld during screening.

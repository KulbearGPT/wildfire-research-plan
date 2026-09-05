# Nibi execution guide

The project runs on [Alliance Nibi](https://docs.alliancecan.ca/wiki/Nibi).
This is a thin single-researcher Slurm workflow, not a deployment platform.
Training and bulk evaluation must never run on a login node.

## Storage

- Git checkout, WSTS+ HDF5, environments, selected checkpoints, and compact
  evidence live under persistent `/project` storage.
- `$SLURM_TMPDIR` is used for node-local temporary files inside a job.
- `/scratch` is only for regenerable staging and worktrees.
- The active run root is `/project/6085198/kulbear/wildfire/runs`.
- Superseded artifacts are recorded in
  [`experiments/artifact-archive-manifest.tsv`](experiments/artifact-archive-manifest.tsv)
  and stored under
  `/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs`.

## Login-node boundary

Allowed on the login node: source edits, Git operations, small JSON/CSV reads,
queue inspection, dry-run command rendering, and `sbatch`/`scancel`.

Required inside Slurm: training, full dataset traversal, model evaluation,
GPU use, and any CPU- or I/O-heavy batch operation. The cleanup itself only
moves top-level entries within the same project filesystem and does not load
datasets or models.

## Active T=1 jobs

The current execution surface is intentionally small:

| Runner | Purpose |
| --- | --- |
| `run_corrected_baseline_on_nibi.sh` | B0/B2/B3 training |
| `run_predictive_consistency_on_nibi.sh` | D1-ERM/D1-KL training |
| `run_standard_reliability_control_on_nibi.sh` | D2-STD training |
| `run_severity_adaptive_reliability_prompting_on_nibi.sh` | D12-SARP training |
| `run_reliability_evaluation_on_nibi.sh` | fixed-year T=1 evaluation |
| `run_d12_heldout_on_nibi.sh` | paired D2/D12 held-out evaluation |

All are in [`../reproductions/wsts_fast_track/`](../reproductions/wsts_fast_track/).
Site-specific accounts, partitions, modules, and paths remain outside Git.

## Submission rule

1. Confirm the checkout and intended runner.
2. Inspect `squeue`/partition availability before submission.
3. Request the minimum sufficient GPU/CPU/memory slice.
4. If a task waits more than ten minutes, compare alternative slices and use
   an immediately available one when it requests at most twice the minimum
   resource. A blocking prerequisite may exceed that bound when necessary.
5. Record the job ID and immutable output directory in the quantitative
   ledger. Do not silently retry a failed scientific run.

The next planned compute is the predetermined-seed D2-STD/D12-SARP comparison
described in [`research-roadmap.md`](research-roadmap.md). No job is submitted
by repository cleanup.

## Minimal environment and data checks

The tracked cluster utilities remain available for portable preflight and
manifest verification:

```bash
python3 scripts/cluster/clusterctl.py profile validate configs/cluster/profile.env
python3 scripts/cluster/clusterctl.py manifest verify DATA_ROOT \
  manifests/data/wstsplus-hdf5.csv \
  manifests/data/wstsplus-hdf5.summary.json \
  --require-production-contract
```

These checks should be used when the environment or data tree changes, not
repeated before every fast prototype.

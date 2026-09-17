# Prepare the corrected eight-year dataset

These stages reproduce the data path used by the retained experiments: original WSTS years 2018–2021, WSTS+ added years 2016/2017/2022/2023, verified active-fire repair, a 999-event combined inventory, and normalization statistics computed only from 2016–2020. They reuse the tracked conversion, assembly, repair, and statistics implementations. This is a complete preparation recipe; the handoff qualification does not imply that a new full download/conversion has been executed.

Start with the site configuration and fresh environments described in [reproduce.md](reproduce.md). Commit the handoff scripts before submission: the launcher executes an archive of the selected commit. Set absolute site paths, including paths containing spaces, using quoted Bash values:

```bash
export WILDFIRE_SITE_ENV="/path/to/site.env"
# In site.env:
# export WILDFIRE_ROOT="/path/to/research artifacts"
# export WILDFIRE_DATA="$WILDFIRE_ROOT/hdf5/wstsplus-active-fixed"
# export WILDFIRE_STATS="$WILDFIRE_ROOT/hdf5/train-2016-2020-stats.npz"
```

The configured training Python is 3.10 and audit Python is 3.13. `submit.sh cpu` activates training; `submit.sh cpu audit` activates audit. The preparation script checks the selected environment and refuses execution without `SLURM_JOB_ID`. Do not invoke bulk preparation on a login node.

Run the following commands from the committed checkout. **Wait for each job to finish successfully before submitting its dependent next stage.** `submit.sh` returns immediately and prints the job ID and log directory; it does not automatically sequence these commands. Adjust `WILDFIRE_CPU_SBATCH` in the site configuration for download time, conversion memory/CPUs, and repair/audit time. Compute nodes need outbound access for the downloads.

```bash
# 1. Download and verify both source archives; these two jobs are independent.
bash scripts/research/submit.sh cpu bash scripts/research/prepare-data.sh download original
bash scripts/research/submit.sh cpu bash scripts/research/prepare-data.sh download plus

# 2. After both downloads succeed, convert original and added years.
# These conversion jobs are independent of one another.
bash scripts/research/submit.sh cpu bash scripts/research/prepare-data.sh original
bash scripts/research/submit.sh cpu bash scripts/research/prepare-data.sh added

# 3. After added conversion succeeds, repair and verify in the audit environment.
bash scripts/research/submit.sh cpu audit bash scripts/research/prepare-data.sh repair

# 4. After original conversion and repair succeed, assemble and audit all years.
bash scripts/research/submit.sh cpu audit bash scripts/research/prepare-data.sh assemble

# 5. After assembly/audit succeeds, compute and validate training-only statistics.
bash scripts/research/submit.sh cpu bash scripts/research/prepare-data.sh stats
```

The final usable locations are exactly `WILDFIRE_DATA` and `WILDFIRE_STATS`; no source edits or historical worktree paths are needed. `WILDFIRE_DATA/READY` is written only after statistics load successfully and the final inventory passes validation. Stage evidence appears in `WILDFIRE_ROOT/preparation/STAGE-JOBID-SUFFIX`; a `COMPLETED` marker means that stage passed. Scheduler/source/environment provenance remains in the generic launcher's `WILDFIRE_ROOT/jobs` directory.

## Source and validation contracts

The download stage uses the same Zenodo API endpoints and checksums as the existing tutorial:

| Archive | Record | Required MD5 |
| --- | --- | --- |
| `WildfireSpreadTS.zip` | `8006177` | `dc1a04e63ccc70037b277d585b8fe761` |
| `WSTSPlus.zip` | `17584629` | `42da7598cc33a170064e78d8027148c9` |

Downloads resume through `.partial` files and are renamed only after checksum validation. Existing complete archives are checked again; conversion also rechecks the archive before extraction. WSTS+ supplies only the four added years, so it cannot replace the original archive.

Original conversion must produce 176, 74, 201, and 156 events for 2018, 2019, 2020, and 2021 respectively. Added conversion and verified repair must produce 92, 110, 122, and 68 events for 2016, 2017, 2022, and 2023. Assembly validates those per-year counts, totaling 999 HDF5 files.

The repair verifier retains these exact `--expect` contracts:

```text
2016:92:2102:886:303648
2017:110:2490:1481:177972
2022:122:3424:2158:105377
2023:68:2442:1297:167952
```

Repair evidence must be ready and contain exactly these empty-source exclusions:

```text
2022/fire_CA4186812327820220730
2022/fire_ID4570411652620220904
2022/fire_OR4513211711020220825
2022/fire_WA4687912083320220803
2022/fire_WA4796412068520220909
```

The assembly stage runs the existing data gate and checks its exit status. Statistics use `reproductions.wsts_fast_track.compute_stats` with the frozen training years 2016–2020, excluding validation/test years. Their three 23-feature arrays are subsequently checked by `load_training_stats`, and the generated `.npz` receives a SHA-256 record.

## Storage and restarts

Raw extraction persists under `WILDFIRE_ROOT/raw/{original,plus}` so separate Slurm jobs can access the TIFFs; node-local temporary directories are not used. Keep the plus TIFFs through repair and source verification. Intermediate HDF5 directories are `WILDFIRE_ROOT/hdf5/{original,added-source,added-verified}`. The assembler uses hard links, so `WILDFIRE_DATA` must be on the same filesystem as those intermediate HDF5 directories. Do not edit intermediate or assembled files after preparation; hard links share file contents.

Plan space for both archives, extracted TIFFs, and intermediate HDF5 data. Conversion, repair, extraction, and statistics refuse existing targets instead of overwriting results. A failed stage may leave partial output: inspect its Slurm logs and evidence, then explicitly remove only that failed stage's outputs or use a fresh root before rerunning. Downloads alone support automatic byte-range resumption. Never run two instances of the same stage against the same root concurrently. The workflow intentionally does not automatically delete raw inputs or unsuccessful output directories.

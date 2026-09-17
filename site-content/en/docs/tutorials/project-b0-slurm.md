# From Data Download to a Complete ResNet18-U-Net Run: A Step-by-Step Nibi Slurm Tutorial

For students encountering this project for the first time. Commands use **Bash + Alliance Nibi**, starting from an empty personal experiment directory. Prepared: 2026-09-11.

The accompanying scripts are in [`res18/`](res18/) and are already included in the repository; there is no need to search the instructor's personal cache directories. They call the project's existing official reproduction and B0 entry points rather than implementing another model-training pipeline.

## 0. First Choose What You Want to Complete

The model is called **ResNet18-U-Net / Res18-U-Net** throughout: ResNet18 extracts features, and the U-Net decoder outputs per-pixel probabilities of next-day active fire. `T=1` means one day of input observations; All means all features under that configuration. The original HDF5 has 23 bands; processing features such as categories and directions through the model data pipeline produces 40 input channels.

However, the same model does not mean the same experiment.

| Route | Data and split | Training budget | Appropriate comparison |
| --- | --- | --- | --- |
| **Project B0, ultimately joining the main research workflow** | Eight-year WSTS+; 2016–2020 train, 2021 val, 2022/2023 test; corrected sample indexing and training-set statistics | seed 0, 3000 steps; one fixed split, no 12-fold loop | Project B0 ledger |
| **Official Fold 2, learning single-fold paper reproduction** | Original WSTS 2018–2021; 2018/2020 train, 2019 val, 2021 test; retain the pinned official data pipeline | seed 0, 10000 steps | Our existing Fold 2 training and released-weight results |
| **Complete official 12-fold** | The 12 year combinations under official `additional_data=false` | 10000 steps per fold, or test only the 12 released weights | Official Res18-U-Net T=1 All reference `0.460 ± 0.084` from the WSTS+ work, without guaranteeing identical provenance to the paper's table |

**B0 only: follow 1–6 → 9–10; testing in Section 11 is optional. One official fold only: follow 1–7; released-weight evaluation in Section 8 is optional.** The complete 12-fold run in Section 12 is an extension exercise, not a student requirement. Both routes share the original data and model environment.

WSTS+ is the name of the paper/extended benchmark; it does not mean that every released weight set uses an eight-year split. In the pinned code, `additional_data=true` has only **4** eight-year folds; do not set it to true and submit 0–11. Project B0 implements a separate forward temporal split. This tutorial uses explicit parameters to avoid confusing these three protocols.

Paper reference numbers come from the [authors' repository results table](https://github.com/slahrichi/WildfireSpreadTS#benchmark-results-ap--standard-deviation). Fold and code conventions follow the project's [upstream lock](../../reproductions/wsts_res18_unet_t1/upstream.lock.json), [released-weight manifest](../../reproductions/wsts_res18_unet_t1/official_weights_manifest.json), and [B0 contract](../../reproductions/wsts_fast_track/contract.py).

## 1. Log In and Create Your Own Directories

### 1.1 Instructor, Once Only: Provide a Repository Version Containing This Tutorial

Local commits do not automatically appear on GitHub. The instructor can publish the corresponding branch or provide a Git bundle. The following creates a standalone file from **the main branch containing this tutorial**; then place the bundle in a project directory readable by students:

```bash
cd /home/kulbear/scratch/wildfire-research-plan
git bundle create /tmp/wildfire-res18-course.bundle main
git bundle verify /tmp/wildfire-res18-course.bundle
```

Copy this bundle into the shared course directory. Students can enter its absolute path below to `git clone`, or use a Git repository URL where this tutorial has already been published. Students should not directly modify the instructor's worktree while experiments are running.

### 1.2 Students: Log In from a Terminal on Your Own Computer

```bash
ssh YOUR_ALLIANCE_USERNAME@nibi.alliancecan.ca
```

Replace `YOUR_ALLIANCE_USERNAME` with your Alliance username. Run all commands below after logging in to Nibi. Use login nodes only to edit scripts, inspect lightweight results, and submit or inspect jobs; use Slurm for downloads, extraction, data traversal, training, and model evaluation.

### 1.3 Check Available Accounts and GPU Names

```bash
sacctmgr -nP show assoc where user="$USER" format=Account,Partition
sinfo -h -o '%P %G'
```

The instructor's accounts for this project are `def-vislearn_cpu` / `def-vislearn_gpu`; students must use accounts actually associated with their own users. Enter the following once, and do not submit placeholders such as `replace-me` literally.

```bash
read -r -p 'Project group directory, for example /project/6085198: ' WF_GROUP
read -r -p 'Your CPU Slurm account: ' WF_CPU_ACCOUNT
read -r -p 'Your GPU Slurm account: ' WF_GPU_ACCOUNT
read -r -p 'Repository URL or absolute Git bundle path containing this tutorial: ' WF_SOURCE
export WF_ROOT="$WF_GROUP/$USER/wildfire-res18-course"
mkdir -p "$WF_ROOT"/{downloads,hdf5,envs,cache,runs,logs,weights}
export WF_REPO="$WF_ROOT/repo"
git clone --branch main "$WF_SOURCE" "$WF_REPO"
test -f "$WF_REPO/docs/tutorials/res18/official.sh"
cd "$WF_REPO"
git rev-parse HEAD
```

Create your own configuration so paths are not hardcoded in shared scripts:

```bash
export WF_TUTORIAL_ENV="$WF_ROOT/tutorial.env"
{
  printf 'export WF_ROOT=%q\n' "$WF_ROOT"
  printf 'export WF_REPO=%q\n' "$WF_REPO"
  printf 'export WF_CPU_ACCOUNT=%q\n' "$WF_CPU_ACCOUNT"
  printf 'export WF_GPU_ACCOUNT=%q\n' "$WF_GPU_ACCOUNT"
} > "$WF_TUTORIAL_ENV"
export WF_SCRIPTS="$WF_REPO/docs/tutorials/res18"
export WF_GPU_SMOKE='nvidia_h100_80gb_hbm3_1g.10gb:1'
export WF_GPU_TRAIN='nvidia_h100_80gb_hbm3_2g.20gb:1'
{
  printf 'export WF_SCRIPTS=%q\n' "$WF_SCRIPTS"
  printf 'export WF_GPU_SMOKE=%q\n' "$WF_GPU_SMOKE"
  printf 'export WF_GPU_TRAIN=%q\n' "$WF_GPU_TRAIN"
} >> "$WF_TUTORIAL_ENV"
```

This requests one 10GB/20GB MIG slice of an H100, not 10/20 GPUs. The 20GB slice is a conservative starting point for the first full training run and exact AP testing; it can be reduced after memory usage is measured, but do not arbitrarily lower the full-run batch size to fit a smaller slice. If `sinfo` does not list these names, update the configuration using the cluster's actual resource names; other clusters cannot simply reuse Nibi's module and GPU names.

Restore the configuration after logging in again:

```bash
read -r -p 'Absolute path to your tutorial.env: ' WF_TUTORIAL_ENV
export WF_TUTORIAL_ENV
source "$WF_TUTORIAL_ENV"
cd "$WF_REPO"
```

**Storage planning:** The original ZIP is approximately 48.4GB and the extension ZIP approximately 19.9GB; you also need HDF5 data, verification copies for extension years, environments, and checkpoints. First confirm at least approximately 400GB of available project quota, and allow around 200GB of node-local temporary space for extracting the original archive. These are conservative allowances, not measured peaks; concurrent data jobs increase requirements.

```bash
df -h "$WF_ROOT"
quota -s
```

`df` reports free filesystem space, not personal project quota. If the site's `quota` does not report project quota, use the quota-checking method supplied by the course group.

## 2. Understand a Slurm Submission

`sbatch` hands the task to the scheduler and immediately returns a job ID; an ID **does not mean training has finished**. `--cpus-per-task` is the CPU core count, `--mem` is host memory, `--gpus` specifies GPUs, and `--time` is the maximum wall-clock time. Successful script completion requires `COMPLETED` and `ExitCode=0:0`; scientific results also require checking output files.

This tutorial uses `--parsable` to save job IDs and `afterok` to run only after prerequisite jobs succeed, preventing training after a failed download. In log filenames, `%j` expands to the job ID; array jobs use `%A_%a` for array ID and fold ID.

Define this function in your login terminal, then use it to check status at each step:

```bash
jobcheck() {
  squeue -j "$1" -o '%.18i %.24j %.10T %.10M %.25R'
  sacct -j "$1" --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
}
```

An empty `squeue` after a job ends is normal; rely on `sacct` and logs. After each submission, wait for `COMPLETED 0:0` before performing that section's “After success” file checks; logs may not exist before a job starts. `afterok` allows advance queueing, but a queued job does not mean the data are ready. Each scientific run uses a separate output directory. Diagnose failures first; do not overwrite the original directory to present a retry as first-attempt success. After logging in again, redefine `jobcheck` and recover downstream dependency job IDs from your records or `sacct`.

## 3. Create Two Isolated Environments

Training depends on older Lightning/SMP versions and uses the Python 3.10 environment inputs previously used by this project; the data-audit package requires Python 3.13. Keeping them separate prevents installation on one side from changing the other's NumPy/PyTorch versions.

```bash
WF_SETUP_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=res18-setup --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-setup.out" "$WF_SCRIPTS/setup.sh")
WF_SETUP_JOB=${WF_SETUP_JOB%%;*}
printf '%s\n' "$WF_SETUP_JOB"
jobcheck "$WF_SETUP_JOB"
tail -n 40 "$WF_ROOT/logs/$WF_SETUP_JOB-setup.out"
```

The script creates environments, installs [`requirements-training.txt`](res18/requirements-training.txt), pins upstream commit `ed221d4…`, applies the repository's existing Res18 import-pruning patch and removes the unused `T_co` type import, and caches ResNet18 ImageNet initialization. These patches do not change the network, loss, or training-data indexing; the actual diff is saved with each run. Model jobs also use `git archive HEAD` to save and execute a snapshot of this project's committed source, with a separate copy of tutorial scripts. Commit the project code you actually intend to use before submitting, and do not edit the upstream checkout during a run.

**Success criteria:** Slurm `0:0`, with the following files present:

```bash
test -f "$WF_ROOT/envs/READY"
cat "$WF_ROOT/envs/training-freeze.txt"
cat "$WF_ROOT/envs/audit-freeze.txt"
```

Installation uses `--no-index` with the Alliance wheelhouse. If a pinned version is unavailable, preserve the logs and ask the instructor to confirm a compatible version or validated environment; do not simply `pip install -U` the entire dependency set. Scripts refuse to overwrite installed environments. Small downloads of the upstream repository and pretrained encoder also occur in this CPU job.

Note: `encoder_weights=imagenet` is public ImageNet initialization. In project records, “from scratch” means not resuming from our existing wildfire checkpoints; it does not mean that all ResNet18 parameters are randomly initialized.

## 4. Download and Verify Original WSTS

Original WSTS covers 2018–2021 for the official single fold and also forms part of the eight-year WSTS+ assembly. Its release page is [Zenodo 8006177](https://zenodo.org/records/8006177). The script pins `WildfireSpreadTS.zip` with MD5 `dc1a04e63ccc70037b277d585b8fe761`.

```bash
WF_DOWNLOAD_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=wsts-download --cpus-per-task=1 --mem=4G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-download.out" "$WF_SCRIPTS/download.sh" original)
WF_DOWNLOAD_JOB=${WF_DOWNLOAD_JOB%%;*}
jobcheck "$WF_DOWNLOAD_JOB"
tail -n 20 "$WF_ROOT/logs/$WF_DOWNLOAD_JOB-download.out"
```

The script's actual download and verification logic is equivalent to:

```bash
# Explanatory snippet, already executed by the Slurm job above; do not download again on the login node.
curl -fL --retry 8 --retry-delay 5 --continue-at - \
  'https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content' \
  -o WildfireSpreadTS.zip.partial
printf '%s  %s\n' dc1a04e63ccc70037b277d585b8fe761 WildfireSpreadTS.zip.partial | md5sum -c -
```

**Why verify:** A zero download exit code does not necessarily mean the file is complete and the version correct. The script renames `.partial` to the final `.zip` only after verification passes. After a network interruption, keep the partial file and resubmit the download step; failed full training should not be blindly retried in the same way.

## 5. CPU Job: GeoTIFF → HDF5

GeoTIFF uses one file per day and suits geographic raster exchange. Training reads time series frequently, so organizing each event in HDF5 reduces file-opening overhead. Each event is arranged as `data[day, channel, height, width]`, with date, year, event name, and location metadata saved alongside it.

```bash
WF_ORIGINAL_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$WF_SETUP_JOB:$WF_DOWNLOAD_JOB" \
  --job-name=wsts-hdf5 --cpus-per-task=4 --mem=32G --tmp=200G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-original.out" "$WF_SCRIPTS/prepare-original.sh")
WF_ORIGINAL_JOB=${WF_ORIGINAL_JOB%%;*}
jobcheck "$WF_ORIGINAL_JOB"
tail -n 40 "$WF_ROOT/logs/$WF_ORIGINAL_JOB-original.out"
```

The script extracts into `$SLURM_TMPDIR`, calls the pinned upstream `src/preprocess/CreateHDF5Dataset.py`, and writes persistent output to `$WF_ROOT/hdf5/original/`. This official converter handles only 2018–2021 and cannot directly process the extension years.

**Success criteria:** 607 events total, with yearly counts 176, 74, 201, and 156; the log prints `status: pass`, then READY is written:

```bash
test -f "$WF_ROOT/hdf5/original/READY"
```

Output consists of four year directories such as `original/2018/*.hdf5`. The original active-fire band uses HHMM encoding, which this step converts to hours; target generation uses active-fire presence. Extension years use different encoding and are handled separately in Section 9; do not divide them by 100 again.

**Data preparation is now complete. Both routes first run the GPU smoke test in Section 6; after it passes, the B0 route jumps to Section 9, while the official single-fold route continues to Section 7.**

## 6. Official Fold 2: Start with a One-Step Smoke Test

The smoke test is an engineering check: fetch a training batch, backpropagate, and check a validation batch. It runs only 1 step with batch size 4 and one validation batch, with testing disabled; it does not produce an AP suitable for comparison with the paper.

```bash
WF_SMOKE_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_ORIGINAL_JOB" --job-name=res18-smoke \
  --gpus="$WF_GPU_SMOKE" --cpus-per-task=8 --mem=64G --time=00:20:00 \
  --output="$WF_ROOT/logs/%j-smoke.out" "$WF_SCRIPTS/official.sh" smoke 2)
WF_SMOKE_JOB=${WF_SMOKE_JOB%%;*}
jobcheck "$WF_SMOKE_JOB"
tail -n 60 "$WF_ROOT/logs/$WF_SMOKE_JOB-smoke.out"
```

After success:

```bash
export WF_SMOKE_RUN="$WF_ROOT/runs/official-smoke-fold2-$WF_SMOKE_JOB"
test -f "$WF_SMOKE_RUN/SMOKE_PASSED"
cat "$WF_SMOKE_RUN/command.txt"
```

The log should contain the `max_steps=1` completion message and a positive `WSTS_OBSERVER_PEAK_ALLOCATED_BYTES`. Seeing only `CUDA available: True` does not establish that the model ran successfully.

## 7. Official Fold 2: Train for 10000 Steps and Automatically Test the Best Validation Checkpoint

```bash
WF_FOLD_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_SMOKE_JOB" --job-name=res18-fold2 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%j-fold2.out" "$WF_SCRIPTS/official.sh" full 2)
WF_FOLD_JOB=${WF_FOLD_JOB%%;*}
jobcheck "$WF_FOLD_JOB"
tail -n 50 "$WF_ROOT/logs/$WF_FOLD_JOB-fold2.out"
```

The `2` in `full 2` is the fold ID, not the seed. This script explicitly passes:

```text
--data.additional_data=false
--data.data_fold_id=2
--seed_everything=0
--data.n_leading_observations=1
--data.features_to_keep=null
--data.remove_duplicate_features=true
--data.n_leading_observations_test_adjustment=5
--data.batch_size=64
--data.num_workers=8
--trainer.max_steps=10000
--do_train=true --do_test=true --do_predict=false
```

**Why select checkpoints this way:** The optimizer performs all 10000 updates, validation AP determines which checkpoint is saved, and the official `train.py` tests that checkpoint at the end. The best checkpoint's `global_step` can be less than 10000; that does not mean training was shorter. Do not use the test set to choose parameters or the best seed.

Retain the official Focal loss, AdamW optimizer, and learning rate 0.001. The official training entry point recalculates YAML's `pos_class_weight: 236` using the fire rate in the training years; our saved effective Fold 2 value is `608.4653828020165`. Do not manually reset it to 236 to approach a particular score.

Read the results after success:

```bash
export WF_FOLD_RUN="$WF_ROOT/runs/official-full-fold2-$WF_FOLD_JOB"
cat "$WF_FOLD_RUN/result.json"
cat "$WF_FOLD_RUN/command.txt"
```

`result.json` contains AP, F1, IoU, precision, recall, loss, the best checkpoint path and step, and the original log's SHA-256. The script writes it only after checks pass for the training-completion marker, test metrics, and checkpoint.

**Existing references, not results from this rerun:** Historical full Fold 2 training on Windows achieved test AP **0.554664**; measured AP for the released Fold 2 weights was **0.570902**. A single-fold score can exceed the paper's cross-fold mean **0.460** because test years differ. Nibi differs from the historical Windows environment, so bitwise equality is not promised; retain actual results rather than replacing them with target numbers.

## 8. Optional: Test the Official Released Fold 2 Weights

This quickly checks whether the data and evaluation entry point are compatible with the official released assets. It **does not train** and does not evaluate the checkpoint you just trained.

```bash
WF_WEIGHT_DOWNLOAD_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=res18-weight-fetch --cpus-per-task=1 --mem=4G --time=00:30:00 \
  --output="$WF_ROOT/logs/%j-weight-fetch.out" "$WF_SCRIPTS/fetch-weights.sh" 2)
WF_WEIGHT_DOWNLOAD_JOB=${WF_WEIGHT_DOWNLOAD_JOB%%;*}
WF_WEIGHT_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_ORIGINAL_JOB:$WF_WEIGHT_DOWNLOAD_JOB" \
  --job-name=res18-weight2 --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 \
  --mem=96G --time=01:00:00 --output="$WF_ROOT/logs/%j-weight2.out" \
  "$WF_SCRIPTS/official.sh" weight 2)
WF_WEIGHT_JOB=${WF_WEIGHT_JOB%%;*}
jobcheck "$WF_WEIGHT_JOB"
cat "$WF_ROOT/runs/official-weight-fold2-$WF_WEIGHT_JOB/result.json"
```

Downloads use pinned Hub revision `acf70a3…`, checking each file's size and SHA-256. Evaluation checks them again, loads the raw state dict with `strict=True`, and calls only `Trainer.test`. A `.pth` raw state dict cannot be used as a Lightning `.ckpt` for resuming training.

The reference file is `fold2_testAP0.571.pth`; the historical reevaluation `0.570902` agrees with the rounded filename. All weights come from the [authors' release repository](https://huggingface.co/saadlahrichi/WSTSPlus/tree/acf70a37394849f4ec8d108a51d6f4325a554d0a/trained_model_weights/Res18Unet_T1/All).

## 9. Join Project B0: Prepare Extension Years and Recalculate Training Statistics

The WSTS+ download contains **four additional years**, not the complete eight-year dataset. Extension data are released at [Zenodo 17584629](https://zenodo.org/records/17584629); `WSTSPlus.zip` is approximately 19.9GB, with MD5 `42da7598cc33a170064e78d8027148c9`.

```bash
WF_PLUS_DOWNLOAD_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=wstsplus-download --cpus-per-task=1 --mem=4G --time=06:00:00 \
  --output="$WF_ROOT/logs/%j-plus-download.out" "$WF_SCRIPTS/download.sh" plus)
WF_PLUS_DOWNLOAD_JOB=${WF_PLUS_DOWNLOAD_JOB%%;*}
WF_PLUS_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$WF_ORIGINAL_JOB:$WF_PLUS_DOWNLOAD_JOB" \
  --job-name=wstsplus-prepare --cpus-per-task=8 --mem=64G --tmp=100G --time=12:00:00 \
  --output="$WF_ROOT/logs/%j-plus-prepare.out" "$WF_SCRIPTS/prepare-plus.sh")
WF_PLUS_JOB=${WF_PLUS_JOB%%;*}
jobcheck "$WF_PLUS_JOB"
tail -n 60 "$WF_ROOT/logs/$WF_PLUS_JOB-plus-prepare.out"
```

This CPU job performs the following in order:

1. Extract the four extension years and generate event HDF5 files using the converter included with this tutorial. Preserve source active-fire values that are already in hours; record missing locations for events without CRS rather than inventing latitude/longitude.
2. Use the project's `repair-active-fire` to generate normalized labels and evidence in a new directory. Labels already correct after conversion still undergo the same verification; there is no need to deliberately introduce errors first.
3. Use the independent `verify_repair` to check event counts, target-day counts, and positive-pixel counts, verifying the exact five excluded empty events.
4. Hard-link the original four years and verified four years into `hdf5/combined/`. Hard links require the same filesystem, so keep all these directories under your own `WF_ROOT`. Do not modify any hard-linked data in place afterward.
5. Audit the full dataset; calculate normalization statistics using only **2016–2020** training samples. Validation and test years do not participate in this step.

After success:

```bash
test -f "$WF_ROOT/hdf5/combined/READY"
test -f "$WF_ROOT/hdf5/train-stats.npz"
cat "$WF_ROOT/runs/prepare-plus-$WF_PLUS_JOB/data-summary.json"
cat "$WF_ROOT/runs/prepare-plus-$WF_PLUS_JOB/audit/contract_decision.json"
cat "$WF_ROOT/runs/prepare-plus-$WF_PLUS_JOB/audit/phase0_report.md"
```

Actual available event counts are below; do not invent nonexistent events because the paper reports 1005:

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HDF5 events | 92 | 110 | 176 | 74 | 201 | 156 | 122 | 68 | **999** |

Train/validation/test event counts should be **653 / 156 / 190**. The expected audit decision is `continue_controlled`: controlled-missingness experiments may proceed, but the data do not yet support claims about natural missingness or operational deployment. See the [phase0 data report](../experiments/phase0.md) for the reasons.

**Why recalculate statistics:** Input bands have different units, and means/standard deviations are used for normalization; computing them from test years leaks test information into training. Official fold reproduction retains the official statistics procedure, while project B0 uses the frozen training-year statistics computed here. These are not interchangeable.

## 10. Train Project B0: A 3000-Step Clean Baseline

This is B0 in the [roadmap](../research-roadmap.md). `clean` means no artificial FireDrop/BlockDrop during training; missing values in the original inputs are still handled by the existing data pipeline. Do not add reliability hints, consistency loss, or expert routing at this stage.

```bash
WF_B0_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_PLUS_JOB:$WF_SMOKE_JOB" --job-name=project-B0 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=03:00:00 \
  --output="$WF_ROOT/logs/%j-B0.out" "$WF_SCRIPTS/b0.sh")
WF_B0_JOB=${WF_B0_JOB%%;*}
jobcheck "$WF_B0_JOB"
tail -n 60 "$WF_ROOT/logs/$WF_B0_JOB-B0.out"
```

The script's training core is the repository's existing module:

```text
python -m reproductions.wsts_fast_track.train_corrected_baseline
  --baseline-id B0
  --upstream-root "$WF_ROOT/upstream"
  --data-root "$WF_ROOT/hdf5/combined"
  --run-root "THIS_JOB_SEPARATE_DIRECTORY/scientific"
  --stats-path "$WF_ROOT/hdf5/train-stats.npz"
```

This snippet explains the parameters; the actual executable submission is the `sbatch` command above. The entry point fixes seed 0, 3000 steps, batch 64, AdamW 0.001, and 2016–2020 train / 2021 val, and corrects sample indexing across years. It does not implement the split by changing `fold_id`, so B0's fold 0 must not be interpreted as official Fold 0.

After training, the job automatically generates a completion record and evaluates four scenarios on 2021. M00 is the main clean-baseline reference; M01/M06/M07 diagnose robustness without retraining the model.

```bash
export WF_B0_RUN="$WF_ROOT/runs/B0-$WF_B0_JOB"
cat "$WF_B0_RUN/scientific/completed.json"
cat "$WF_B0_RUN/results-2021/summary.json"
```

**B0 acceptance criteria:** Slurm `COMPLETED 0:0`; `completed.json` contains `baseline_id=B0`, `max_steps=3000`, `corrected_index=true`, and `test_enabled=false`; the best checkpoint file exists; and `results-2021/summary.json` contains four scenarios.

Existing B0 ledger (reruns in different environments need not match bitwise):

| Year | M00 clean AP | M01 AP | M06 AP | M07 AP |
| --- | ---: | ---: | ---: | ---: |
| 2021 validation | 0.580988 | 0.039387 | 0.323805 | 0.133672 |
| 2022 fixed test | 0.278852 | 0.006669 | 0.135278 | 0.071779 |
| 2023 fixed test | 0.412674 | 0.010101 | 0.205411 | 0.078691 |

These follow the project's historical B0 evaluation protocol, from the [quantitative ledger](../experiments/quantitative_reliability_ledger.md). Do not directly compare 2021's 0.580988 against the paper's 12-fold 0.460, or mix it into the later main table that aligns target dates across T. The trainer's internal validation AP and external controlled evaluation may use different target-date sets; check loaders and sample counts before comparing, rather than relying only on metric names.

**Students have now completed one B0 run.** The subsequent fixed tests and full official 12-fold run are extensions.

## 11. Optional: Test the Fixed B0 Once on 2022/2023

Run this only after the recipe is frozen. The following explicitly enables `--heldout-authorized` in the existing evaluation entry point, without training, selecting a new checkpoint, or tuning thresholds. Students reproduce historical test results already disclosed to this project; they cannot describe these as their own newly unseen tests.

```bash
export WF_B0_RECORD="$WF_B0_RUN/scientific/completed.json"
for year in 2022 2023; do
  sbatch --account="$WF_GPU_ACCOUNT" --job-name="B0-test-$year" \
    --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=01:00:00 \
    --output="$WF_ROOT/logs/%j-B0-test.out" \
    "$WF_SCRIPTS/evaluate-b0.sh" "$WF_B0_RECORD" "$year"
done
```

Record these two job IDs. Outputs are `runs/B0-test-2022-JOBID/results-2022/summary.json` and `runs/B0-test-2023-JOBID/results-2023/summary.json`, respectively.

## 12. Full Reproduction Appendix: 12 Folds, with Training and Released-Weight Evaluation Summarized Separately

### 12.1 Train All 12 Official Folds from Scratch

Run this after the Section 6 smoke test passes. Each array task trains for 10000 steps and tests the best checkpoint selected by its own validation set. `%2` limits execution to two concurrent folds; it does not divide the batch or data into two parts.

```bash
WF_FULL_ARRAY=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_SMOKE_JOB" --array=0-11%2 --job-name=res18-full12 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%A_%a-full12.out" "$WF_SCRIPTS/official.sh" full)
WF_FULL_ARRAY=${WF_FULL_ARRAY%%;*}
jobcheck "$WF_FULL_ARRAY"
```

If you already completed Fold 2, this independent array trains it again to produce a separate, complete course campaign. With limited resources, do not run both full single-fold training and the full array. An existing Fold 2 can be reused, but the instructor must verify versions and parameters and explicitly select the files; do not automatically mix it into the array results.

Each directory name contains `official-full-foldN-…`. Slurm's `SLURM_JOB_ID` for an array task need not match the main array ID, so use a small listing to locate this campaign's outputs:

```bash
sacct -nP -j "$WF_FULL_ARRAY" --format=JobIDRaw,State,ExitCode
```

After everything completes, select only this campaign's 12 `result.json` files and list their absolute paths in a file, one per line. For example, a small CPU job can match array metadata; Section 12.3 gives a general method.

### 12.2 Evaluate Only All 12 Released Weights

This usually takes less time than training 12 times and is useful for learning to verify a published baseline's evaluation protocol. It demonstrates executable reproduction of released weights; it does not replace training all 12 folds yourself from scratch.

```bash
WF_ALL_WEIGHTS_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=weights12-fetch --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-weights12-fetch.out" "$WF_SCRIPTS/fetch-weights.sh" all)
WF_ALL_WEIGHTS_JOB=${WF_ALL_WEIGHTS_JOB%%;*}
WF_WEIGHT_ARRAY=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_ORIGINAL_JOB:$WF_ALL_WEIGHTS_JOB" \
  --array=0-11%2 --job-name=res18-weight12 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --output="$WF_ROOT/logs/%A_%a-weight12.out" "$WF_SCRIPTS/official.sh" weight)
WF_WEIGHT_ARRAY=${WF_WEIGHT_ARRAY%%;*}
jobcheck "$WF_WEIGHT_ARRAY"
```

### 12.3 Summarize This Array and Check That All 12 Folds Are Present

Each run saves `slurm-job.txt`, which contains `ArrayJobId`. First choose the array type you actually ran:

```bash
# To summarize released weights:
export WF_SUMMARY_ARRAY="$WF_WEIGHT_ARRAY"
export WF_SUMMARY_MODE=weight
# To summarize full training instead, use:
# export WF_SUMMARY_ARRAY="$WF_FULL_ARRAY"
# export WF_SUMMARY_MODE=full
```

Submit a small CPU summary job:

```bash
sbatch --account="$WF_CPU_ACCOUNT" --dependency="afterok:$WF_SUMMARY_ARRAY" \
  --job-name=res18-summary --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --output="$WF_ROOT/logs/%j-summary.out" "$WF_SCRIPTS/summarize-array.sh" \
  "$WF_SUMMARY_MODE" "$WF_SUMMARY_ARRAY"
```

Final file:

```bash
cat "$WF_ROOT/runs/summary-$WF_SUMMARY_MODE-$WF_SUMMARY_ARRAY.json"
```

The summary script requires exactly one result for each fold 0–11, all in the same mode; missing folds, duplicates, or mixed training/weight results fail. It outputs mean AP, population standard deviation `ddof=0`, and sample standard deviation `ddof=1`; state which you report. Do not average yearly AP values first and call that the 12-fold mean.

### 12.4 How to Interpret “Close to the Original Paper's Numbers”

The table below contains existing independent reevaluation records, not new 12-fold results executed for this tutorial:

| Comparison | AP | Explanation |
| --- | ---: | --- |
| Res18-U-Net T=1 All in the official WSTS+ table | 0.460 ± 0.084 | Paper/authors' repository reference |
| Historical per-fold reevaluation of twelve released weights | **0.452764 ± 0.088217** | Population standard deviation; mean differs from the paper reference by about -0.007236 |
| AP labels in the twelve weight filenames | 0.452917 ± 0.088272 | Statistics of rounded filename values only, not a new evaluation |
| Historical Fold 2 trained by us | 0.554664 | Single fold, not a twelve-fold mean |
| Historical released Fold 2 weights | 0.570902 | Single fold, consistent with filename 0.571 |

For complete per-fold metrics, runs, and independent verification records, see the [official reproduction report](../experiments/res18_unet_t1_reproduction.md) and [reproduction README](../../reproductions/wsts_res18_unet_t1/README.md).

“Close” describes existing results; it is not a tuning target or a pass condition for this tutorial. Current records cannot prove that released weights and the paper's table share exactly the same provenance, runs, and aggregation procedure; the intended official positive-weight configuration also remains historically unresolved. Correct delivery reports actual scores and differences, rather than repeatedly changing parameters until reaching 0.460.

## 13. Common Failures: Where to Look and What to Do

| Symptom | Check first | Correct action |
| --- | --- | --- |
| `Invalid account` / GPU type does not exist | `sacctmgr`, `sinfo`, and configuration | Use your own account and actual resource names, not the instructor's account |
| `DependencyNeverSatisfied` | Prerequisite job's `sacct` and logs | Fix the prerequisite failure, save the new job ID, then resubmit downstream jobs; old dependencies do not recover automatically |
| Environment installation reports `No matching distribution` | Specific pinned version in `setup` logs | Ask the instructor to confirm the wheelhouse and existing validated environment; do not arbitrarily upgrade all model dependencies |
| ZIP checksum fails | `.partial`, disk quota, download logs | Do not extract a corrupt archive; preserve evidence and obtain the file again |
| Extension-year targets are almost all zero | Whether the original four-year official converter was used incorrectly | Return to Section 9; convert and verify using the extension years' hour encoding |
| `refusing existing target` / `FileExistsError` | Previous failed directory and logs | Identify the last completed step; use a new course root directory or have the instructor confirm recovery. Do not delete data still used by other steps |
| CUDA OOM | Actual requested slice, peak usage, full testing stage | Request a larger slice and rerun, preserving failure logs; do not arbitrarily change full-training batch size, features, precision, or crop |
| CPU `OUT_OF_MEMORY` | `sacct MaxRSS`, worker count, test AP stage | Increase `--mem`; record worker changes, and do not treat incomplete testing as a result |
| Checkpoint warning about `weights_only` / untrusted deserialization | Whether the file is your own `.ckpt` or an official weight in the pinned manifest | This tutorial accommodates old Lightning for these two trusted sources; do not apply the setting to unknown checkpoints |
| Checkpoint exists but `result.json` is missing | Completion marker, whether testing ended, log-parsing errors | A checkpoint does not mean the whole experiment succeeded; inspect logs first and do not fabricate completion files |
| Queued for more than ten minutes | `squeue --start -j JOBID` and resource probes | Compare `sbatch --test-only` estimates for the same command; change the request only if resources stay within twice the necessary amount and it is clearly faster. Avoid leaving two duplicate sets of scientific jobs |

Inspecting training progress does not start new training:

```bash
squeue -u "$USER"
squeue --start -j "$WF_FOLD_JOB"
tail -f "$WF_ROOT/logs/$WF_FOLD_JOB-fold2.out"
```

`Ctrl-C` stops only `tail`, not the job. Run `scancel JOBID` only when cancellation is actually needed; do not indiscriminately cancel all your jobs.

## 14. What Students Should Submit

- Course repository commit, upstream commit, `tutorial.env`, and both environment freezes.
- Download verification logs and data conversion/audit results; for B0, include training statistics and a description of data years.
- Your job IDs, original training logs, actual commands, and best checkpoint path.
- Single-fold `result.json`, or B0's `scientific/completed.json` and `results-2021/summary.json`.
- Your own results table, clearly distinguishing single fold / twelve folds, your own training / released-weight evaluation, and official protocol / project B0.

A complete submission should explain the inputs, labels, training years, when checkpoints were selected, when testing occurred, and exactly which reference the score can be compared with.

## 15. Validation Scope of This Tutorial

The accompanying data converters come from the project's existing Nibi execution scripts; model calls reuse the existing official/B0 entry points. The new summarizer checks metric validity, the complete fold set, and experiment type. Specific static checks, lightweight tests, and Slurm smoke results are recorded in [validation.md](res18/validation.md).

Writing this tutorial does not automatically rerun full ZIP downloads, conversion of 999 events, 10000-step training, and the complete 12-fold campaign. The historical numbers above are recorded separately from tutorial validation. On their first run with their own account and new environment, students must still verify each step against its success criteria.

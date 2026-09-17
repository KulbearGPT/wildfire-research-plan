# Reproduce the existing research on a new Slurm cluster

The handoff has two parts: [directions with positive signals](positive-signals.md) retain code, results, and run instructions; the [negative-results archive](negative-results.md) records implementation ideas, settings, and results. Improvements from a single seed or setting are included, but are not presented as confirmed contributions. See [method-inventory.json](method-inventory.json) for the complete inventory. Not run, canceled, and invalid comparison are three statuses that do not count as negative results.

**Execution qualification in fresh environments is complete.** With newly created training/audit environments, an independent copy of the official source, and relocated data and weight paths, 260 CPU checks and 91 short-budget GPU cases passed; see the [qualification record](qualification.md). Qualification ran on this cluster's compute nodes. It does not claim execution on a different physical cluster or treat short training runs as full-budget reproductions of every historical score.

## 1. Prepare the code and cluster configuration

This procedure reproduces our research changes, so it requires the source and Git history of this handoff version. A local Git bundle can deliver them without requiring the new server to access our remote repository or original working directory. The submitter uses Git to preserve each job's exact source snapshot, so extracting a `git archive` alone is insufficient to run the submission commands below. The setup job fetches the official model code from its official repository; the standalone instructions for running only the official Res18 baseline remain in the [original tutorial](../tutorials/res18-baseline-slurm.md).

In the source repository with the handoff version committed, generate the bundle and commit ID:

```bash
git bundle create ../handoff.bundle HEAD
git rev-parse HEAD > ../handoff-commit.txt
```

Transfer these two files to the new server, then create a working repository from the local bundle and pin the handoff commit:

```bash
git clone /path/to/transfer/handoff.bundle wildfire-research-plan
cd wildfire-research-plan
git checkout --detach "$(cat /path/to/transfer/handoff-commit.txt)"
```

The bundle contains that commit and its ancestor history; this clone reads a local file and does not connect to our remote repository. Set `WILDFIRE_REPO` below to the absolute path of this new working repository.

Requirements are Linux, Slurm, an NVIDIA GPU, Python 3.10 for training, and Python 3.13 for data auditing. The two Python environments are separate because the original model dependencies and data-audit packages have different version requirements. The training environment does not install the root project package, which requires Python 3.13; it imports the research modules from the archived source instead.

Run the following in the new working repository:

```bash
mkdir -p "$HOME/wildfire-config"
cp configs/research/site.example.env "$HOME/wildfire-config/site.env"
export WILDFIRE_SITE_ENV="$HOME/wildfire-config/site.env"
${EDITOR:-vi} "$WILDFIRE_SITE_ENV"
```

Adapt `WILDFIRE_REPO`, `WILDFIRE_ROOT`, Python commands, module lists, and CPU/GPU Slurm arrays to the new server. `WILDFIRE_ROOT` must be a data-storage directory accessible to compute nodes with sufficient space. Paths may contain spaces. On servers without modules, leave both module variables empty; the local site configuration determines the account, partitions, and GPU model.

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
bash scripts/research/submit.sh cpu setup
```

The submitter prints the job ID and archive directory. Use `squeue -u "$USER"` to view the queue and `sacct -j JOB_ID --format=JobID,State,ExitCode,NodeList` to check the exit status (replace `JOB_ID` with the printed number). Logs are stored in `$WILDFIRE_ROOT/jobs/`; success requires all three: Slurm state `COMPLETED`, exit code `0:0`, and `allocation-JOB_ID/setup-completed.txt`.

Setup installs the pinned official WildfireSpreadTS version, applies the compatibility patches retained in the repository, creates two fresh virtual environments, and downloads pretrained ResNet18 weights within a CPU job. Dependency inputs are listed in the [training environment](../../environments/research-training.txt) and [audit environment](../../environments/research-audit.txt); the job also saves the actual `pip freeze`.

After a setup failure, preserve the logs first. If the [recovery conditions](setup-recovery.md) hold, resume with `setup-finish`; do not delete the environments and retry blindly.

Submit all training, tensor tests, data downloads, and conversions through `submit.sh`. `job.sh` refuses execution without `SLURM_JOB_ID`. The submitter archives the **HEAD commit**, not the uncommitted working tree, so commit changes before submitting experiments.

## 2. Data and weights

See [data preparation](data-preparation.md) for reconstruction from public data. The final directory directly contains the eight years 2016–2023, totaling 999 HDF5 events; normalization statistics use only 2016–2020. Training is fixed to those years, 2021 is used for validation, and 2022/2023 are held-out test years.

See [weight and result bundles](artifacts.md) for transferring existing models. Weights are not stored in Git; verify their digests and include the corresponding results, rather than merely copying the old server's absolute paths into the configuration. Initialization paths within historical checkpoints identify their provenance; do not arbitrarily rewrite those identities when relocating files.

## 3. Retrain the shared B0/B1/B2/B3/B5 baselines

After environment and data preparation succeed, the following command submits a full B3 (T1 FireDrop+BlockDrop) training job:

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id B3 --upstream-root "$WILDFIRE_UPSTREAM" \
  --data-root "$WILDFIRE_DATA" --stats-path "$WILDFIRE_STATS" \
  --run-root "$WILDFIRE_ROOT/runs/B3"
```

Change both `B3` and the output directory to `B0`, `B1`, `B2`, or `B5` to run the corresponding baseline. B0 is T1 clean, B2 is T1 FireDrop, B1 is T5 clean, and B5 is T5 FireDrop+BlockDrop. Each uses seed 0 and 3,000 optimizer steps; `from_scratch` means no continuation from an already trained task model, not that ImageNet encoder initialization is disabled.

After success, follow [baseline export](baselines.md) to generate a verified checkpoint and completion record. Do not manually mark unfinished training as complete.

Baseline training and subsequent continuation are separate stages. T1 continuations start from B3; T5 continuations start from B5. The two configured checkpoints must be the correct Lightning baseline files; wrapped continuation checkpoints cannot substitute for them.

## 4. Cross-history methods

After configuring the B3/B5 weights, first run a T1 smoke test; it checks the computation pipeline and does not provide evidence of paper-level effectiveness:

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method cosine_erm --seed 0 --batch-size 16 --workers 3 \
  --smoke --output "$WILDFIRE_ROOT/runs/smoke-t1-cosine"
```

For a full run, remove `--smoke`, use a new output directory, and retain `--steps 3000`. Below are X22 and its closest control for seed 0, T1:

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method control --seed 0 --steps 3000 --batch-size 64 \
  --workers 3 --output "$WILDFIRE_ROOT/runs/t1-s0-control"
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method cosine_erm --seed 0 --steps 3000 --batch-size 64 \
  --workers 3 --output "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm"
```

Physical batch sizes may differ across experimental campaigns; an effective batch size of 64 does not guarantee identical BatchNorm behavior. Use the physical batch recorded in the original results for comparisons; 16 and 64 are not equivalent reproduction settings. T5 requires `--history 5`, which changes the history length, feature set, and model together; differences between T1/T5 cannot be attributed entirely to history length.

Complete selection on 2021 and freeze the model before evaluating held-out years:

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method cosine_erm --seed 0 --batch-size 64 --workers 3 \
  --evaluate-only "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm/checkpoint.pt" \
  --year 2022 --output "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm-2022"
```

See [method recipes](method-recipes.md) for the X/D methods, closest controls, evaluators, and combination commands; see [teachers and distillation](teachers.md) for teacher reconstruction and RF/TD commands.

## 5. Interpreting reproduction results

M00 denotes complete inputs; the primary metric is mean AP over M01/M06/M07, and block denotes the mean over M06/M07. AP differences are absolute: for example, `+0.005` is an increase of 0.5 percentage points. Compare against the closest control first, then discuss the total gain over the shared baseline.

Two distinct September three-directions implementations are retained: `cross_history.run_three_directions` is the RF family; `three_directions.run` is the TD family. Their initialization, batches, and objectives differ; do not interchange their checkpoints, commands, or result tables.

All 21 original retained result files have been checked: the sample counts for 2021/2022/2023 are consistently 3,181/2,856/2,102. The 2,312 in older quantitative records is a documentation typo; all 84 AP values remain unchanged. See the [sample-count audit](evaluation-population.md). GPU smoke tests, single-seed screens, three-seed confirmations, and held-out results are recorded separately and cannot substitute for one another.

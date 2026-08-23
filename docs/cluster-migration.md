# Slurm cluster migration

## Scope and evidence boundary

This is a thin research-prototype workflow for a Slurm cluster with one H100
per task. It is not a deployment platform and does not add DDP, multi-node
training, automatic retry, or a workflow service. Independent folds and seeds
may later use a bounded array with at most four concurrent tasks.

## Nibi target

The current target is [Alliance Nibi](https://docs.alliancecan.ca/wiki/Nibi).
Connect through `nibi.alliancecan.ca`. Nibi's GPU nodes provide eight 80 GB
H100 SXM GPUs per full node; this project requests one full H100 with
`--gpus=h100:1`. Keep the account and partition in the ignored site profile,
because those values depend on the user's allocation rather than the
repository.

**No cluster scientific job has been run by this repository reorganization.**
The completed Windows/RTX 3090 evidence remains frozen. A cluster run starts a
new lineage and must pass the gates below before it can support a scientific
comparison. In particular, do not run the Windows-fixed controllers on Linux;
use the portable entrypoints shown here.

## Persistent and node-local layout

On Nibi, keep the Git checkout, 999 event HDF5 files, verified weights, compact
run records, and selected checkpoints under persistent `/project` storage.
Use `$SLURM_TMPDIR` for node-local temporary loader and compilation files by
setting `SCRATCH_ENV=SLURM_TMPDIR` in the ignored profile. Use `/scratch` only
for regenerable staging or transfer data: Nibi documents a 1 TB soft quota and
a 60-day grace period, so it is not the authoritative home for datasets,
environments, run evidence, or checkpoints.

The examples below assume these shell variables have been set to site-local
absolute paths:

```bash
export PROJECT_ROOT="/project/group/wildfire-research-plan"
export DATA_ROOT="/project/group/wildfire-data/hdf5"
export RUNS_ROOT="/project/group/wildfire-runs"
export CACHE_ROOT="/project/group/wildfire-cache"
export OFFICIAL_UPSTREAM_ROOT="${CACHE_ROOT}/WildfireSpreadTS"
export DERIVED_UPSTREAM_ROOT="${CACHE_ROOT}/WildfireSpreadTS-res18-runtime"
export WEIGHT_ROOT="${CACHE_ROOT}/WSTSPlus"
cd "${PROJECT_ROOT}"
```

## Clone the reviewed commit

Clone from the login or transfer node, then detach at the reviewed commit that
was pushed to `main`. Record the resulting hash; do not work from an uncommitted
checkout.

```bash
git clone https://github.com/kulbeargpt/wildfire-research-plan.git "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
git checkout --detach "${REVIEWED_COMMIT}"
git status --short
git rev-parse HEAD
```

## Create the site-local profile

Copy the tracked example to the ignored real profile and replace every dummy
account, partition, module, and path. `PROJECT_ROOT` is this checkout; the
other roots are persistent directories. Keep one GPU per task and array
concurrency between one and four.

```bash
cp configs/cluster/profile.example.env configs/cluster/profile.env
${EDITOR:-vi} configs/cluster/profile.env
python3 scripts/cluster/clusterctl.py profile validate configs/cluster/profile.env
```

Validation parses only literal `KEY=VALUE` lines. It does not source the file,
expand variables, or execute substitutions. Never commit the real profile.

Every submitted job runs a formal preflight before it creates run evidence or
executes the scientific command. It automatically verifies the 999-file production manifest,
the clean project checkout, PyTorch, CUDA, and the single visible GPU. When the command names
`--upstream-root` or `--weights-path`, it
also verifies the official upstream commit (or the exact reviewed seven-line
deletion patch) and the official weight filename and size. The resulting data,
runtime, command, code, and weight identities are sealed into `started.json`.
The job exports `WANDB_MODE=disabled` and disables Python bytecode so that a
prototype run does not create external logging or source-tree cache files.

## Qualify the environment

First confirm that the profile's modules expose the intended Python, compiler,
CUDA runtime, driver, and H100. The first target-cluster requirements lock is a
result of qualification, not an assumption copied from Windows.

```bash
module purge
module load <the comma-separated MODULES from profile.env>
python3 --version
python3 -c 'import platform; print(platform.platform())'
nvidia-smi
```

After fetching the pinned upstream tree below, create the candidate training
environment from an explicit reviewed requirements file and the site's wheel
mirror:

```bash
scripts/cluster/bootstrap.sh configs/cluster/profile.env \
  "/project/group/envs/wsts-training" "${OFFICIAL_UPSTREAM_ROOT}/requirements.txt"
```

The bootstrap refuses a nonempty environment directory and never invents
package versions. Freeze a Linux/H100 requirements lock only after the import,
one-batch, and Fold-2 equivalence gates pass.

## Transfer and verify HDF5

Transfer the event-level HDF5 tree as 999 files; do not extract the original
many-small-file raw archive on persistent storage. Root-level conversion JSON
sidecars are not part of the event manifest.

```bash
rsync -a --info=progress2 /transfer/source/hdf5/ "${DATA_ROOT}/"
python3 scripts/cluster/clusterctl.py manifest verify "${DATA_ROOT}" \
  manifests/data/wstsplus-hdf5.csv \
  manifests/data/wstsplus-hdf5.summary.json \
  --require-production-contract
```

The required result is 999 files, 49,816,826,985 bytes, and exact 2016–2023
year counts. Any missing, extra, or resized event stops migration. This is a
fast metadata inventory: it deliberately does not hash the 49.8 GB dataset.

## Fetch pinned upstream code and weights

Fetch the authors' repository at the frozen commit, make a disposable derived
checkout, and apply only the tracked seven-export import-scope patch. The
official model, datamodule, loss, optimizer, callbacks, and metric bodies stay
unchanged.

```bash
git clone https://github.com/slahrichi/WildfireSpreadTS.git "${OFFICIAL_UPSTREAM_ROOT}"
git -C "${OFFICIAL_UPSTREAM_ROOT}" checkout --detach ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad
cp -a "${OFFICIAL_UPSTREAM_ROOT}" "${DERIVED_UPSTREAM_ROOT}"
git -C "${DERIVED_UPSTREAM_ROOT}" apply \
  "${PROJECT_ROOT}/reproductions/wsts_res18_unet_t1/patches/res18_import_scope.patch"
git -C "${OFFICIAL_UPSTREAM_ROOT}" status --short
git -C "${DERIVED_UPSTREAM_ROOT}" diff --check
```

Fetch only the Fold-2 weight needed for qualification and verify its filename
and byte size against the canonical manifest. Fetch other folds later only when
an experiment needs them.

```bash
mkdir -p "${WEIGHT_ROOT}/trained_model_weights/Res18Unet_T1/All"
curl --fail --location \
  "https://huggingface.co/saadlahrichi/WSTSPlus/resolve/acf70a37394849f4ec8d108a51d6f4325a554d0a/trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth?download=true" \
  --output "${WEIGHT_ROOT}/trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth"
ls -lh "${WEIGHT_ROOT}/trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth"
```

The formal job preflight checks that the filename and byte size match the
tracked official manifest; it does not hash the weight.

## Run the one-batch smoke

The smoke loads exactly one seeded Fold-2 training batch and one validation
batch; it does not optimize and does not call the test loader. First print the
complete Slurm command:

```bash
scripts/cluster/submit.sh configs/cluster/profile.env \
  --job-name fold2-smoke --time-class smoke --dry-run -- \
  python "${PROJECT_ROOT}/reproductions/wsts_res18_unet_t1/scripts/smoke_fold2.py" \
  --upstream-root "${OFFICIAL_UPSTREAM_ROOT}" --data-root "${DATA_ROOT}" \
  --output-dir . --num-workers 8 --include-validation
```

Inspect the printed command, then remove only `--dry-run` and submit once. A
pass requires the expected `[B,1,40,128,128]` input, binary next-day target,
finite values, distinct train/validation batches, `training_started=false`,
and `test_loader_called=false`.

## Run Fold-2 test-only equivalence

Only after the smoke passes, dry-run the official released weight through the
portable strict-load/test-only entrypoint:

```bash
scripts/cluster/submit.sh configs/cluster/profile.env \
  --job-name fold2-equivalence --time-class calibration --dry-run -- \
  python "${PROJECT_ROOT}/reproductions/wsts_res18_unet_t1/scripts/official_weight_entrypoint.py" \
  --upstream-root "${DERIVED_UPSTREAM_ROOT}" \
  --weights-path "${WEIGHT_ROOT}/trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth" \
  --config="${DERIVED_UPSTREAM_ROOT}/cfgs/unet/res18_monotemporal.yaml" \
  --trainer="${DERIVED_UPSTREAM_ROOT}/cfgs/trainer_single_gpu.yaml" \
  --data="${DERIVED_UPSTREAM_ROOT}/cfgs/data_monotemporal_full_features.yaml" \
  --data.data_dir="${DATA_ROOT}" --data.data_fold_id=2 \
  --data.features_to_keep=null --data.n_leading_observations=1 \
  --data.remove_duplicate_features=true --data.num_workers=8 \
  --trainer.default_root_dir=. --do_train=false --do_test=false
```

After inspection, remove only `--dry-run` and submit once. Require strict load
of 182 tensors, no fit/validation/predict path, complete 3,337 test batches,
and Fold-2 metrics within a tolerance declared before reading the cluster
result. The local reference AP is `0.5709022879600525`. This gate establishes
numerical executable equivalence, not identical floating-point bytes.

## Run the 500-step calibration

Only after equivalence passes, dry-run the same official model/data command
with training enabled, test disabled, and exact `max_steps=500`:

```bash
scripts/cluster/submit.sh configs/cluster/profile.env \
  --job-name fold2-calibration --time-class calibration --dry-run -- \
  python "${PROJECT_ROOT}/reproductions/wsts_res18_unet_t1/scripts/official_entrypoint.py" \
  --upstream-root "${DERIVED_UPSTREAM_ROOT}" \
  --config="${DERIVED_UPSTREAM_ROOT}/cfgs/unet/res18_monotemporal.yaml" \
  --trainer="${DERIVED_UPSTREAM_ROOT}/cfgs/trainer_single_gpu.yaml" \
  --data="${DERIVED_UPSTREAM_ROOT}/cfgs/data_monotemporal_full_features.yaml" \
  --data.data_dir="${DATA_ROOT}" --data.data_fold_id=2 \
  --data.features_to_keep=null --data.n_leading_observations=1 \
  --data.remove_duplicate_features=true --data.num_workers=8 \
  --trainer.max_steps=500 --trainer.default_root_dir=. --do_test=false
```

Remove only `--dry-run` after review. This is a timing calibration, not a paper
result; it must not be used to choose a model or claim test AP.

## Choose time limits from measured runtime

Use the one-batch and 500-step H100 evidence to select site time classes. Do
not copy the RTX 3090 wall-time estimate as an H100 guarantee. Keep at least a
small scheduler margin, then update only the ignored site profile.

## Monitor, stop, and collect

Use `squeue`, `sacct`, and the profile's log root. Stop a defective run with
`scancel JOB_ID`; never hide it by automatic retry. Each job writes immutable
`started.json` and exactly one `completed.json` or `failure.json`.

Archive only declared small JSON/CSV outputs:

```bash
scripts/cluster/collect.sh "${RUN_DIR}" "${RUN_DIR}/collection.json" \
  "${RUN_DIR}/work/smoke.json" "${RUN_DIR}/work/metrics.csv"
```

Scientific commands run under `${RUN_DIR}/work`; immutable lifecycle evidence
stays in `${RUN_DIR}`. This separation prevents an upstream script with fixed
relative output names from colliding with the evidence markers.

## Inode hygiene

- Keep WSTS+ as 999 event HDF5 files; do not unpack raw imagery trees.
- Download only required weights and retain only the declared best/final
  checkpoint per scientific run.
- Put caches and temporary compilation files under `CACHE_ROOT` or node-local
  scratch, not beside source.
- Consolidate small result JSON/CSV files with `collect.sh`; never archive
  checkpoints through that command.
- Delete or compress superseded logs only after a reviewed result is sealed.

## Failure handling

A failed preflight, manifest check, smoke, equivalence run, or calibration stops
the sequence. Preserve the run directory and Slurm logs, diagnose the exact
cause, and create a new run identity only after review. There is no auto-resume
or automatic scientific retry. Migration success does not broaden the
controlled-missingness claim or establish operational-deployment performance.

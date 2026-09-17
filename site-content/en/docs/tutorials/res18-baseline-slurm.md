# Running Res18-U-Net from the Official Codebase on a New Cluster: A Standalone Slurm Tutorial

This version requires only the authors' official repository, public data, and public weights. **Do not clone `wildfire-research-plan`, call our modules, or read the instructor's caches, environments, or checkpoints.** All small scripts needed are included here for students to copy into their own experiment directory.

As specified, the new cluster uses the same Alliance/Nibi software stack and Slurm as the existing server. Module names remain `StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2`; accounts, storage directories, and available GPUs must use the new server's own values. This tutorial does not promise compatibility with arbitrary unknown clusters without configuration.

## 0. Exactly Which Baseline Does This Tutorial Reproduce?

The model is **ResNet18-U-Net, T=1, All features** from the authors' WSTS+ work. The ResNet18 encoder extracts features, and the U-Net decoder outputs a per-pixel probability of active fire on the next day. One day of observations is supplied; official feature processing transforms the original 23 bands into 40 channels. ResNet18 uses ImageNet initialization, not our previously trained wildfire checkpoint.

Students only need to train **Fold 2**: train on 2018/2020, validate on 2019, and test on 2021, with 10000 optimizer updates. Full 12-fold training and released-weight evaluation are optional steps at the end.

**This official baseline is different from corrected B0 in the project roadmap.** Project B0 has its own eight-year temporal split, index correction, training statistics, and 3000-step budget; unchanged official code does not automatically implement those behaviors. This tutorial prioritizes the requirement to avoid depending on our repository by reproducing the official baseline.

WSTS+ is the name of the authors' work, but the twelve official weights pinned here use the original four-year WSTS splits, so download the original `WildfireSpreadTS.zip`. Do not change to `additional_data=true`: the pinned code's eight-year setting has only four folds, cannot use 0–11, and cannot be compared directly with the twelve-fold references below.

Official sources: [authors' code](https://github.com/slahrichi/WildfireSpreadTS), [original WSTS data](https://zenodo.org/records/8006177), and [authors' released weights](https://huggingface.co/saadlahrichi/WSTSPlus). Code is pinned to `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`, and weights to `acf70a37394849f4ec8d108a51d6f4325a554d0a`.

## 1. Log In to the New Cluster and Configure Your Own Directories

Run this in a terminal on your own computer, replacing the username and new server address:

```bash
ssh YOUR_USERNAME@YOUR_NEW_CLUSTER
```

Run all subsequent commands in a Bash terminal on that cluster. First check the site's software and accounts:

```bash
module avail python cuda
sacctmgr -nP show assoc where user="$USER" format=Account,Partition
sinfo -h -o '%P %G'
```

Obtain your CPU/GPU accounts from the output or an administrator. Do not copy the instructor's accounts. Your experiment directory should be on writable shared storage visible to compute nodes, not in the instructor's directory. Allow enough quota for the approximately 48.4GB ZIP, HDF5 data, environment, and checkpoints; around 250GB is recommended, with around 200GB of temporary disk requested for conversion. These are conservative allowances, not measured peaks.

```bash
read -r -p 'Absolute path to your experiment root directory: ' WF_ROOT
read -r -p 'CPU account: ' WF_CPU_ACCOUNT
read -r -p 'GPU account: ' WF_GPU_ACCOUNT
read -r -p 'GPU type and count, for example h100:1: ' WF_GPU
export WF_ROOT WF_CPU_ACCOUNT WF_GPU_ACCOUNT WF_GPU
mkdir -p "$WF_ROOT"/{scripts,downloads,data,runs,logs,cache,weights}
{
  printf 'export WF_ROOT=%q\n' "$WF_ROOT"
  printf 'export WF_CPU_ACCOUNT=%q\n' "$WF_CPU_ACCOUNT"
  printf 'export WF_GPU_ACCOUNT=%q\n' "$WF_GPU_ACCOUNT"
  printf 'export WF_GPU=%q\n' "$WF_GPU"
} > "$WF_ROOT/config.env"
df -h "$WF_ROOT"
```

If the cluster requires an explicit partition, add `--partition=ACTUAL_PARTITION` to the CPU/GPU `sbatch` commands below. A GPU may be a whole card or a suitable slice with at least approximately 20GB; use names reported by `sinfo`. Request 1 device and retain the full-run batch size of 64; first verify driver and environment compatibility with the smoke test. MIG names on the instructor's current server are not universal across servers.

Restore your configuration after logging in again:

```bash
read -r -p 'Absolute path to your config.env: ' WF_CONFIG
source "$WF_CONFIG"
```

**Slurm usage:** A job ID returned by `sbatch` does not mean completion. Check `squeue -j JOBID` and `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS`, and inspect results only after `COMPLETED 0:0`. You may submit downstream jobs with `afterok` in advance at each section, but a queued job is not a successful job. After logging in again, recover the job IDs needed for downstream dependencies from your notes.

Use Slurm for downloads, installation, data conversion, and model execution. The installation from scratch and download jobs below require outbound network access. If compute nodes on the new cluster cannot access the network, first ask the administrator for a network-enabled CPU/data-transfer node or a mirror; do not quietly perform bulk work on an ordinary login node.

## 2. Create Self-Contained Script Files

Copy each `cat … EOF` block below in full, including the final `EOF`. These commands only write small files: they do not train models, download large files, or require cloning another repository.

### 2.1 Training Environment Dependencies

Retain the Python 3.10 / PyTorch 2.6 / Lightning 2.0.2 environment inputs previously used by the project. This single-official-fold tutorial **does not require the project's Python 3.13 audit environment**.

```bash
cat > "$WF_ROOT/scripts/requirements.txt" <<'EOF'
h5py==3.10.0
numpy==1.26.4
pandas==1.5.3
einops==0.6.1
torch==2.6.0
torchmetrics==1.4.0.post0
torchvision==0.21.0
tqdm==4.65.0
wandb==0.15.3
xarray==2023.4.2
rasterio==1.3.10
geopandas==0.12.2
matplotlib==3.7.2
imageio==2.27.0
pytorch-lightning==2.0.2
scikit-learn==1.2.1
segmentation-models-pytorch==0.3.3
jsonargparse[signatures]==4.40.2
psutil==7.2.2
setuptools==80.9.0
EOF
```
Installation uses the Alliance wheelhouse available on this type of cluster (`--no-index`). If the new site lacks that software source, stop and confirm an installation channel; do not assume the same wheels automatically exist on another server. The first installation freeze and GPU smoke test provide actual evidence for the new environment.

### 2.2 Official Weight Manifest (Pinned Version)

Each row contains the fold, filename, SHA-256, and byte count, in that order. The download script verifies these values rather than assuming that a matching filename means a correct download.

```bash
cat > "$WF_ROOT/scripts/weights.tsv" <<'EOF'
0 fold0_testAP0.528.pth be33d244916d71158dab42b3aa607395a49989f862165529f38cabd735dd69eb 57889221
1 fold1_testAP0.426.pth 367cdec7a9a0f5729d2c01db1c195345aaf78768976293277b5386e7bcc0e36f 57889221
2 fold2_testAP0.571.pth e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec 57889221
3 fold3_testAP0.307.pth 41e24f55dd0270566351fa5b52d50c77a45d01074805f18a591791b29860ab81 57889221
4 fold4_testAP0.483.pth aa8e512a63bdcc229dedb358a0e8cf9e40c6559ae3b2504adbd3e30ccc97fd3d 57889221
5 fold5_testAP0.322.pth 0b31b3749c239123de159171fc0f60390a14ae25bd986b713135b73f5faecda9 57889221
6 fold6_testAP0.577.pth e83c49de0bf7524c92946a205e3eadd2ffce76d04502e23ffcab9a3593e8afd9 57889221
7 fold7_testAP0.474.pth 5ed6e8c59a57d6d948522efd4f90c64596f4292f16a18f96610b310faa30d41a 57889221
8 fold8_testAP0.478.pth 6004ab1060a6cfed1ffa70174dcf0c68717e75ea5b96ebde0e60a63297f45ab4 57889221
9 fold9_testAP0.471.pth 7237244ce4a5ee5f3084985fc074275e69caca853b9bde2e1ada7000c5f5d3d0 57889221
10 fold10_testAP0.324.pth 2c9247afccdab53cda6bb3da00b4ba414ce963e6ad0e3e43bcf4623e431d3e96 57889405
11 fold11_testAP0.474.pth 9ecc0ee9e5908dff4f242a6b6a58060c4bfdd9ceb6857c4a12e13b633fa4c709 57889405
EOF
```
### 2.3 Download and Verify Official Weights

```bash
cat > "$WF_ROOT/scripts/weights.py" <<'EOF'
import hashlib
from pathlib import Path
import sys
import urllib.request
import shutil
root, fold = Path(sys.argv[1]), sys.argv[2]
verify_only = '--verify-only' in sys.argv[3:]
assert fold in ['all'] + [str(i) for i in range(12)]
revision = 'acf70a37394849f4ec8d108a51d6f4325a554d0a'
for line in (root / 'scripts/weights.tsv').read_text().splitlines():
    index, name, expected_hash, size = line.split()
    if fold != 'all' and fold != index:
        continue
    path = root / 'weights' / name
    if not path.exists() and not verify_only:
        url = f'https://huggingface.co/saadlahrichi/WSTSPlus/resolve/{revision}/trained_model_weights/Res18Unet_T1/All/{name}'
        with urllib.request.urlopen(url, timeout=180) as response, path.with_suffix('.partial').open('wb') as out:
            shutil.copyfileobj(response, out)
        candidate = path.with_suffix('.partial')
    else:
        candidate = path
    assert candidate.stat().st_size == int(size), candidate
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_hash, candidate
    if candidate != path:
        candidate.rename(path)
    print('VERIFIED', index, name, expected_hash)
EOF
```
### 2.4 Entry Point for Evaluating Released Weights Only

This small script uses the model classes, LightningCLI, and DataModule from the authors' `train.py`, strictly loads their raw state dict, and calls only `Trainer.test`. It does not reimplement the network, loss, or dataset.

```bash
cat > "$WF_ROOT/scripts/evaluate_weight.py" <<'EOF'
import runpy
import sys
from pathlib import Path
import torch
code, weight = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(code / 'src'))
sys.argv = [str(code / 'src/train.py'), *sys.argv[3:]]
ns = runpy.run_path(str(code / 'src/train.py'), run_name='official_weight_evaluation')
cli = ns['MyLightningCLI'](ns['BaseModel'], ns['FireSpreadDataModule'],
    subclass_mode_model=True, save_config_kwargs={'overwrite': True},
    parser_kwargs={'parser_mode': 'yaml'}, run=False)
state = torch.load(weight, map_location='cpu', weights_only=True)
assert isinstance(state, dict) and 'state_dict' not in state
cli.model.load_state_dict(state, strict=True)
print('OFFICIAL_WEIGHT_STRICT_LOAD=1', flush=True)
cli.trainer.test(cli.model, cli.datamodule)
EOF
```
### 2.5 Save Results and Check the Complete 12-Fold Set

Every metric must exist and be a valid finite value. A complete summary requires exactly one result for each of the twelve folds; do not mix your own training results with released-weight evaluations.

```bash
cat > "$WF_ROOT/scripts/results.py" <<'EOF'
import json
import math
from pathlib import Path
import re
import statistics
import sys
if sys.argv[1] == 'record':
    run, mode, fold = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', (run / 'output.log').read_text(errors='replace'))
    marker = ('`Trainer.fit` stopped: `max_steps=10000` reached.' if mode == 'train' else 'OFFICIAL_WEIGHT_STRICT_LOAD=1')
    assert marker in text, 'Incomplete training or strict loading'
    metrics = {}
    for name in ('AP','f1','iou','precision','recall','loss'):
        matches = re.findall(r'\btest_' + name + r'\b[ \t│┃|:=]*([^\s│┃|]+)', text)
        assert matches, name
        value = float(matches[-1])
        assert math.isfinite(value) and value >= 0 and (name == 'loss' or value <= 1), name
        metrics[name] = value
    result = {'mode':mode, 'fold':fold, 'metrics':metrics, 'run':str(run)}
    if mode == 'train':
        checkpoints = list((run / 'work').rglob('*.ckpt'))
        assert len(checkpoints) == 1, checkpoints
        result['checkpoint'] = str(checkpoints[0])
    with (run / 'result.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
elif sys.argv[1] == 'aggregate':
    root, array = Path(sys.argv[2]), sys.argv[3]
    assert array.isdigit()
    rows = []
    for meta in (root / 'runs').glob('*/slurm-job.txt'):
        if re.search(r'\bArrayJobId=' + array + r'\b', meta.read_text()):
            rows.append(json.loads((meta.parent / 'result.json').read_text()))
    assert len(rows) == 12 and sorted(r['fold'] for r in rows) == list(range(12)), 'Missing/duplicate folds'
    assert len({r['mode'] for r in rows}) == 1, 'Do not mix trained and released-weight results'
    values = [r['metrics']['AP'] for r in rows]
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
    result = {'mean_AP':statistics.mean(values), 'population_std_AP':statistics.pstdev(values),
              'sample_std_AP':statistics.stdev(values), 'rows':sorted(rows,key=lambda r:r['fold'])}
    with (root / f'summary-{array}.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
else:
    raise ValueError('Use record or aggregate')
EOF
```
### 2.6 Unified Slurm Job Script

The only `git clone` in this script points to the authors' repository. To make the pinned version importable in this PyTorch environment, setup only prunes package imports for unused architectures and removes the unused `T_co` type import; the complete diff is saved as `setup-runtime.patch`. It does not correct official sample indexing, replace the data split, or change the loss.

```bash
cat > "$WF_ROOT/scripts/job.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch to run this script}"
source "${1:?Pass config.env}"
action=${2:?Pass setup/download/convert/smoke/train/fetch-weight/test-weight/aggregate}
fold=${3:-${SLURM_ARRAY_TASK_ID:-2}}
case "$action" in setup|download|convert|smoke|train|fetch-weight|test-weight|aggregate) ;; *) exit 2;; esac
if [[ "$action" != aggregate && "$fold" != all ]]; then
  [[ "$fold" =~ ^([0-9]|1[01])$ ]] || exit 2
fi
export TORCH_HOME="$WF_ROOT/cache/torch"
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Only our own checkpoints and the checksum-verified official weights are loaded.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
code="$WF_ROOT/official-code"
env_dir="$WF_ROOT/env"
if [[ "$action" == setup ]]; then
  test ! -e "$code"
  test ! -e "$env_dir"
  virtualenv --no-download "$env_dir"
  source "$env_dir/bin/activate"
  python -m pip install --no-index -r "$WF_ROOT/scripts/requirements.txt"
  git clone https://github.com/slahrichi/WildfireSpreadTS.git "$code"
  git -C "$code" checkout --detach ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad
  python - "$code" <<'PATCH'
from pathlib import Path
import sys
root = Path(sys.argv[1])
p = root / 'src/models/__init__.py'
keep = ['from .BaseModel import BaseModel',
        'from .ConvLSTMLightning import ConvLSTMLightning',
        'from .LogisticRegression import LogisticRegression',
        'from .SMPModel import SMPModel']
assert all(p.read_text().splitlines().count(line) == 1 for line in keep)
p.write_text('\n'.join(keep) + '\n')
p = root / 'src/dataloader/FireSpreadDataset.py'
s = p.read_text()
assert s.count('from torch.utils.data.dataset import T_co\n') == 1
p.write_text(s.replace('from torch.utils.data.dataset import T_co\n', ''))
PATCH
  git -C "$code" diff > "$WF_ROOT/setup-runtime.patch"
  python - <<'ENCODER'
import segmentation_models_pytorch as smp
smp.encoders.get_encoder('resnet18', in_channels=40, weights='imagenet')
print('ImageNet encoder cached')
ENCODER
  python -m pip freeze > "$WF_ROOT/environment-freeze.txt"
  touch "$WF_ROOT/SETUP_READY"
  exit 0
fi
if [[ "$action" == download ]]; then
  cd "$WF_ROOT/downloads"
  if [[ ! -f WildfireSpreadTS.zip ]]; then
    curl -fL --retry 8 --retry-delay 5 --continue-at - \
      https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content \
      -o WildfireSpreadTS.zip.partial
    printf '%s  %s\n' dc1a04e63ccc70037b277d585b8fe761 WildfireSpreadTS.zip.partial | md5sum -c -
    mv WildfireSpreadTS.zip.partial WildfireSpreadTS.zip
  fi
  printf '%s  %s\n' dc1a04e63ccc70037b277d585b8fe761 WildfireSpreadTS.zip | md5sum -c -
  exit 0
fi
test -f "$WF_ROOT/SETUP_READY"
source "$env_dir/bin/activate"
export PYTHONPATH="$code/src"
if [[ "$action" == fetch-weight ]]; then
  python "$WF_ROOT/scripts/weights.py" "$WF_ROOT" "$fold"
  exit 0
fi
if [[ "$action" == aggregate ]]; then
  python "$WF_ROOT/scripts/results.py" aggregate "$WF_ROOT" "$fold"
  exit 0
fi
if [[ "$action" == convert ]]; then
  test ! -e "$WF_ROOT/data/hdf5"
  stage="${SLURM_TMPDIR:?}/wsts-raw"
  mkdir "$stage"
  unzip -q "$WF_ROOT/downloads/WildfireSpreadTS.zip" -d "$stage"
  mapfile -t found < <(find "$stage" -type d -name 2018 -print)
  [[ ${#found[@]} -eq 1 ]]
  raw=$(dirname "${found[0]}")
  python "$code/src/preprocess/CreateHDF5Dataset.py" \
    --data_dir "$raw" --target_dir "$WF_ROOT/data/hdf5"
  python - "$WF_ROOT/data/hdf5" <<'COUNTS'
from pathlib import Path
import sys
p = Path(sys.argv[1])
expected = {2018:176, 2019:74, 2020:201, 2021:156}
actual = {y:len(list((p / str(y)).glob('*.hdf5'))) for y in expected}
assert actual == expected, actual
print('PASS: 607 events', actual)
(p / 'READY').touch()
COUNTS
  exit 0
fi
test -f "$WF_ROOT/data/hdf5/READY"
[[ "$fold" =~ ^([0-9]|1[01])$ ]] || exit 2
run="$WF_ROOT/runs/$action-fold$fold-${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
mkdir "$run"
mkdir "$run/work"
cp "$WF_ROOT/config.env" "$run/config.env"
cp -r "$WF_ROOT/scripts" "$run/scripts"
git -C "$code" rev-parse HEAD > "$run/upstream-commit.txt"
git -C "$code" diff > "$run/runtime.patch"
python -m pip freeze > "$run/pip-freeze.txt"
module list > "$run/modules.txt" 2>&1
nvidia-smi -q > "$run/nvidia-smi.txt"
scontrol show job "$SLURM_JOB_ID" > "$run/slurm-job.txt"
export WANDB_DIR="$run/work"
cd "$run/work"
batch=64
steps=10000
extra=()
if [[ "$action" == smoke ]]; then
  batch=4
  steps=1
  extra+=(--trainer.num_sanity_val_steps=1 --trainer.limit_val_batches=1 --do_test=false)
fi
args=(
  "--config=$code/cfgs/unet/res18_monotemporal.yaml"
  "--trainer=$code/cfgs/trainer_single_gpu.yaml"
  "--data=$code/cfgs/data_monotemporal_full_features.yaml"
  --seed_everything=0 "--data.data_dir=$WF_ROOT/data/hdf5"
  --data.additional_data=false "--data.data_fold_id=$fold"
  --data.n_leading_observations=1 --data.features_to_keep=null
  --data.remove_duplicate_features=true --data.n_leading_observations_test_adjustment=5
  "--data.batch_size=$batch" --data.num_workers=8 "--trainer.max_steps=$steps"
  "--trainer.default_root_dir=$run/work" --trainer.logger.init_args.log_model=false
  --do_train=true --do_validate=false --do_test=true --do_predict=false "${extra[@]}"
)
if [[ "$action" == test-weight ]]; then
  python "$run/scripts/weights.py" "$WF_ROOT" "$fold" --verify-only
  weight=$(awk -v f="$fold" '$1==f {print $2}' "$WF_ROOT/scripts/weights.tsv")
  command=(python "$run/scripts/evaluate_weight.py" "$code" "$WF_ROOT/weights/$weight" "${args[@]}" --do_train=false)
else
  command=(python "$code/src/train.py" "${args[@]}")
fi
printf '%q ' "${command[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
"${command[@]}" 2>&1 | tee "$run/output.log"
if [[ "$action" == smoke ]]; then
  grep -F '`Trainer.fit` stopped: `max_steps=1` reached.' "$run/output.log"
  touch "$run/SMOKE_PASSED"
else
  python "$run/scripts/results.py" record "$run" "$action" "$fold"
fi
EOF
```
## 3. Create the Environment and Obtain the Official Code

```bash
SETUP_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-setup.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" setup)
SETUP_JOB=${SETUP_JOB%%;*}
printf '%s\n' "$SETUP_JOB"
```

After success:

```bash
test -f "$WF_ROOT/SETUP_READY"
cat "$WF_ROOT/environment-freeze.txt"
cat "$WF_ROOT/setup-runtime.patch"
git -C "$WF_ROOT/official-code" remote -v
```

The final command should show only `slahrichi/WildfireSpreadTS`. Setup caches the ImageNet encoder to avoid network access during training. Existing environment/code directories cannot be overwritten; if installation fails, inspect the logs before repeatedly overwriting files or upgrading dependencies.

## 4. Download the Original Four-Year WSTS Dataset

```bash
DOWNLOAD_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --cpus-per-task=1 --mem=4G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-download.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" download)
DOWNLOAD_JOB=${DOWNLOAD_JOB%%;*}
```

The actual URL is `https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content`. The MD5 should be `dc1a04e63ccc70037b277d585b8fe761`; rename `.partial` to `.zip` only after verification passes. The log should show checksum status `OK`. Network downloads can resume, but this does not imply that failed training should automatically retry.

## 5. CPU Conversion: GeoTIFF → HDF5

```bash
DATA_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$SETUP_JOB:$DOWNLOAD_JOB" \
  --cpus-per-task=4 --mem=32G --tmp=200G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-convert.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" convert)
DATA_JOB=${DATA_JOB%%;*}
```

GeoTIFF contains daily geographic rasters. HDF5 organizes each event as `day × channel × height × width`, reducing the overhead of repeatedly opening small files during training. Conversion directly calls the authors' `src/preprocess/CreateHDF5Dataset.py` and converts original active-fire HHMM values to hours. The prediction target is a proxy for next-day active fire, not the complete fire perimeter.

Success criteria: **176/74/201/156** HDF5 files for 2018/2019/2020/2021 respectively, totaling **607** events. The script prints the counts and writes READY:

```bash
test -f "$WF_ROOT/data/hdf5/READY"
tail -n 30 "$WF_ROOT/logs/$DATA_JOB-convert.out"
```

Do not feed the additional four-year `WSTSPlus.zip` directly to this converter: the extension's dates, encodings, and missing geographic information require separate handling. It is not the input for these official twelve folds.

## 6. One Validation Batch + One-Step GPU Smoke Test

```bash
SMOKE_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$DATA_JOB" --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=64G --time=00:20:00 \
  --output="$WF_ROOT/logs/%j-smoke.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" smoke 2)
SMOKE_JOB=${SMOKE_JOB%%;*}
```

After success:

```bash
test -f "$WF_ROOT/runs/smoke-fold2-$SMOKE_JOB/SMOKE_PASSED"
tail -n 40 "$WF_ROOT/logs/$SMOKE_JOB-smoke.out"
```

The smoke test trains for one step with batch size 4, validates one batch, and does not test. A message about the absence of a best validation checkpoint may appear in this one-step check; it neither indicates completed full training nor provides an AP for reporting in a paper. The full run restores batch size 64 and the complete validation set.

## 7. Full Training of Official Fold 2

```bash
TRAIN_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$SMOKE_JOB" --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%j-train.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" train 2)
TRAIN_JOB=${TRAIN_JOB%%;*}
```

The job runs **the authors' `official-code/src/train.py`**. Key settings:

| Parameter | Value and meaning |
| --- | --- |
| fold / seed | Fold 2 and seed 0 are different concepts |
| Data | `additional_data=false`; 2018/2020 train, 2019 val, 2021 test |
| Model | Res18-U-Net, T=1, All, ImageNet encoder initialization |
| Update budget | 10000 steps, batch 64, AdamW, lr 0.001, official Focal loss |
| Selection and evaluation | Select the best checkpoint by validation AP; the official entry point automatically tests it after training |
| Temporal alignment | Retain `n_leading_observations_test_adjustment=5` |

The official entry point recalculates the effective positive class weight from the training years. Do not manually reset it to 236 just because that number appears in YAML. The historical effective Fold 2 value was 608.4653828020165.

After success:

```bash
export TRAIN_RUN="$WF_ROOT/runs/train-fold2-$TRAIN_JOB"
cat "$TRAIN_RUN/result.json"
cat "$TRAIN_RUN/command.txt"
sacct -j "$TRAIN_JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Acceptance requires `COMPLETED 0:0`, a log confirming completion of 10000 steps, six valid test metrics, one best checkpoint, and `result.json`. The best checkpoint may have been saved before step 10000: training uses the full budget, while testing selects the best validation performer rather than forcing selection of the final step.

**Historical references:** Our own official Fold 2 training achieved test AP **0.554664**; the authors' released weights measured **0.570902** on the same fold. These are historical records, not a promise of bitwise reproduction on the new server. Record your actual measured AP; do not tune on the test set to approach these numbers.

At this point, students have completed one official baseline fold.

## 8. Optional: Evaluate Only the Official Fold 2 Weights

```bash
FETCH_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$SETUP_JOB" --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-weight-download.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" fetch-weight 2)
FETCH_JOB=${FETCH_JOB%%;*}
WEIGHT_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$DATA_JOB:$FETCH_JOB" --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-weight-test.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" test-weight 2)
WEIGHT_JOB=${WEIGHT_JOB%%;*}
```

After completion, read `runs/test-weight-fold2-JOBID/result.json`. The 0.571 in the released filename `fold2_testAP0.571.pth` is a rounded label, not the student's own training result. This evaluation does not train; if you only want to verify official weights, you can skip the 10000-step training in Section 7.

## 9. Optional: Complete 12-Fold Runs

### 9.1 Train All Twelve Folds from Scratch

```bash
FULL_ARRAY=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$SMOKE_JOB" --array=0-11%2 --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%A_%a-train12.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" train)
FULL_ARRAY=${FULL_ARRAY%%;*}
```

`%2` limits execution to two concurrent folds; each fold is a complete independent training run. This separate campaign includes Fold 2. To avoid duplicate training, choose either the single-fold tutorial or the full array; do not implicitly mix in earlier runs.

### 9.2 Alternatively, Evaluate Only the Twelve Released Weights

```bash
FETCH_ALL=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$SETUP_JOB" --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-fetch12.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" fetch-weight all)
FETCH_ALL=${FETCH_ALL%%;*}
WEIGHT_ARRAY=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$DATA_JOB:$FETCH_ALL" --array=0-11%2 --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --output="$WF_ROOT/logs/%A_%a-weights12.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" test-weight)
WEIGHT_ARRAY=${WEIGHT_ARRAY%%;*}
```

### 9.3 Summarize the Twelve Folds Actually Completed

Choose one submitted array:

```bash
# Use this line for released-weight evaluation; for training, use ARRAY_ID="$FULL_ARRAY".
ARRAY_ID="$WEIGHT_ARRAY"
sbatch --account="$WF_CPU_ACCOUNT" --dependency="afterok:$ARRAY_ID" \
  --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --output="$WF_ROOT/logs/%j-summary.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" aggregate "$ARRAY_ID"
```

The summary file is `$WF_ROOT/summary-ARRAY_ID.json`, containing per-fold AP, the mean, population standard deviation, and sample standard deviation. A missing fold, duplicate fold, or mixture of training and released-weight results causes an error.

### 9.4 How to Compare with the Paper's Numbers

| Reference | AP | What it supports |
| --- | ---: | --- |
| Res18-U-Net T=1 All results table in the authors' WSTS+ work | 0.460 ± 0.084 | Paper/authors' table reference |
| Our historical reevaluation of the twelve released weights | 0.452764 ± 0.088217 | Population standard deviation; supports executable reproduction of released weights |
| Our historical full Fold 2 training | 0.554664 | Single-fold result for one test year |

See the authors' table in the [official README](https://github.com/slahrichi/WildfireSpreadTS#benchmark-results-ap--standard-deviation). The historical twelve-weight mean differs from the reference by about -0.007236. The values are close, but this does not prove that the paper's table and released assets have exactly the same run provenance. Statistics from filename scores cannot replace actual evaluation either. A single fold exceeding the cross-fold mean is normal.

Report the new server's current results separately; do not copy the historical numbers above into your own experiment results.

## 10. Troubleshooting and Deliverables

| Issue | Action |
| --- | --- |
| Module, GPU name, or account does not exist | Check the new cluster's `module avail`, `sinfo`, and accounts; do not copy values from the old server |
| `--no-index` cannot find a dependency | Confirm the cluster's software source and availability of the pinned wheels; do not blindly upgrade all dependencies |
| Download/installation connection failure | Confirm outbound connectivity on compute nodes; request an allowed network-enabled node or mirror |
| `SLURM_TMPDIR` is undefined or temporary space is insufficient | Conversion requires node-local temporary storage provided by the administrator; follow this cluster's conventions, not the instructor's directory |
| `DependencyNeverSatisfied` | Diagnose the failed prerequisite; after fixing it, resubmit downstream jobs with the new job ID |
| Directory already exists | The script refuses to overwrite existing environments and scientific runs; preserve logs and identify the failed stage first |
| CUDA/CPU OOM | Increase GPU/host memory based on `nvidia-smi` and MaxRSS; do not arbitrarily change the full-run batch size, features, or precision |
| Checkpoint exists but result.json is missing | Check whether training finished its budget, testing ended, and all metrics exist; do not handwrite a success file |
| Scores differ | First check data, fold, parameters, software versions, and evaluation protocol; do not select configurations using test AP |

Deliver: configuration, official commit and patch, environment freeze, download checksums, conversion counts, job IDs, actual commands, logs, checkpoint paths, and result JSON. Keep large data and models in your personal experiment storage; do not concurrently change upstream code or install into the environment in the same directory.

## 11. Independence and Validation Boundaries

All scripts in this tutorial are created in Section 2. Runtime dependencies are limited to the student's own directories, the authors' repository, official data and weights, and the software environment supplied by this type of cluster. Nothing here requires obtaining `wildfire-research-plan` code or calls its Python package.

The previous corrected B0 route, which depends on this project, is saved separately as `project-b0-slurm.md` and is not part of this standalone tutorial's execution steps.

The new server has not yet been connected to or validated. Historical scores and the earlier Nibi smoke test cannot be presented as full training results from this standalone version on the new server. The exact scope of local checks for the newly added scripts is recorded in `official-standalone-validation.md` in the same directory; that record is not a dependency for student execution.

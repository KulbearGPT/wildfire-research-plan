# 在新集群从官方 codebase 跑通 Res18-U-Net：独立 Slurm 教程

本版只需要作者的官方仓库、公开数据和公开权重，**不 clone `wildfire-research-plan`，不调用我们的模块，不读取老师的缓存、环境或检查点**。所需的小型脚本全部在本文给出，学生复制后保存在自己的实验目录。

按你的说明，新集群采用与现有服务器相同的 Alliance/Nibi 软件栈与 Slurm。模块名沿用 `StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2`；账户、存储目录和可申请 GPU 必须使用新服务器自己的值。本文不是对任意未知集群免配置兼容的承诺。

## 0. 本教程到底复现哪一个 baseline

模型是作者 WSTS+ 工作中的 **ResNet18-U-Net、T=1、All features**。ResNet18 编码器提取特征，U-Net 解码器输出下一天活跃火点的逐像素概率。输入一天观测，原始 23 波段经过官方特征处理后成为 40 通道。ResNet18 用 ImageNet 初始化，不使用我们已经训练好的野火 checkpoint。

学生只需训练 **Fold 2**：2018/2020 训练、2019 验证、2021 测试，10000 个优化器更新。完整 12-fold 和发布权重评价在文末作为可选步骤。

**这里的官方 baseline 不等于项目 roadmap 的 corrected B0。** 项目 B0 另有八年时间划分、索引修正、训练统计量及 3000 步预算；官方原样代码不会自动产生那些行为。你的“不依赖我们仓库”要求在本教程中优先落实为官方 baseline 复现。

WSTS+ 是作者工作的名字，但这里固定的十二个官方权重使用原始四年 WSTS 划分，所以下载原始 `WildfireSpreadTS.zip` 即可。不要改成 `additional_data=true`：锁定代码的八年设置只有四个 fold，既不能套用 0–11，也不能直接比较下面的十二 fold 参照。

官方来源：[作者代码](https://github.com/slahrichi/WildfireSpreadTS)、[原始 WSTS 数据](https://zenodo.org/records/8006177)、[作者发布权重](https://huggingface.co/saadlahrichi/WSTSPlus)。代码固定为 `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`，权重固定为 `acf70a37394849f4ec8d108a51d6f4325a554d0a`。

## 1. 登录新集群并配置自己的目录

在自己电脑终端执行，替换用户名和新服务器地址：

```bash
ssh YOUR_USERNAME@YOUR_NEW_CLUSTER
```

后续命令均在该集群的 Bash 终端执行。先检查站点软件和账户：

```bash
module avail python cuda
sacctmgr -nP show assoc where user="$USER" format=Account,Partition
sinfo -h -o '%P %G'
```

从输出或管理员处取得 CPU/GPU account。不要复制教师的 account。实验目录应位于计算节点可见、可写的共享存储，而不是教师目录。为约 48.4GB 的 ZIP、HDF5、环境和 checkpoint 准备充足配额；建议预留约 250GB，并在转换作业申请约 200GB 临时磁盘。这是保守准备量，不是实测峰值。

```bash
read -r -p '你的实验根目录绝对路径：' WF_ROOT
read -r -p 'CPU account：' WF_CPU_ACCOUNT
read -r -p 'GPU account：' WF_GPU_ACCOUNT
read -r -p 'GPU 类型及数量，例如 h100:1：' WF_GPU
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

如集群要求显式 partition，请在下面 CPU/GPU 的 `sbatch` 命令中分别加上 `--partition=实际分区`。GPU 可以是整卡或至少约 20GB 的适用分片，名称以 `sinfo` 为准。申请 1 块设备，保持正式 batch 64；先通过 smoke 验证驱动与环境兼容。教师当前服务器的 MIG 名称不是跨服务器通用名称。

重新登录后恢复：

```bash
read -r -p '你的 config.env 绝对路径：' WF_CONFIG
source "$WF_CONFIG"
```

**Slurm 使用原则：** `sbatch` 返回 job ID，不等于完成。查看 `squeue -j JOBID` 和 `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS`，只有 `COMPLETED 0:0` 后再检查结果。每一节都可提前提交带 `afterok` 的后续任务，但不要把还在排队的任务当作成功。重新登录还需要从笔记恢复后续依赖所用的 job ID。

下载、安装、数据转换和模型运行都通过 Slurm。下面的从零安装与下载任务需要出站网络；若新集群计算节点禁止联网，应先由管理员提供可联网的 CPU/数据传输节点或镜像渠道，不能在普通登录节点偷偷执行批量工作。

## 2. 创建自包含的脚本文件

下面各个 `cat … EOF` 块应整段复制，包括最后的 `EOF`。它们只是写小文件，不训练、不下载大文件，也不需要 clone 任何额外仓库。

### 2.1 训练环境依赖

保留项目以前使用过的 Python 3.10 / PyTorch 2.6 / Lightning 2.0.2 环境输入。在这个单官方 fold 教程中**不需要 Python 3.13 的项目审计环境**。

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
安装使用同类型集群的 Alliance wheelhouse（`--no-index`）。如果新站点没有该软件源，应停止并确认安装渠道；不能认为换个服务器仍自动存在相同 wheel。第一次安装的 freeze 和 GPU smoke 才是新环境的实际证据。

### 2.2 官方权重清单（固定版本）

每行依次为 fold、文件名、SHA-256、字节数。下载脚本会核验这些值，不凭文件名认定下载正确。

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
### 2.3 下载和验证官方权重

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
### 2.4 只评价发布权重的入口

此小脚本使用作者 `train.py` 中的模型类、LightningCLI 和 DataModule，严格加载作者 raw state dict，然后只调用 `Trainer.test`。不重新实现网络、损失或数据集。

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
### 2.5 保存结果及检查完整 12-fold

每项指标必须存在且为合法有限值；完整汇总要求十二个 fold 恰好各一个，禁止混用自己训练和发布权重评价结果。

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
### 2.6 统一 Slurm 作业脚本

这个脚本里的唯一 `git clone` 指向作者仓库。为让锁定版本在该 PyTorch 环境可导入，setup 只裁剪未用架构的包导入，并删除未使用的 `T_co` 类型导入；完整差异保存为 `setup-runtime.patch`。不修正官方样本索引、不替换数据划分、不改损失。

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
## 3. 创建环境并取得官方代码

```bash
SETUP_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-setup.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" setup)
SETUP_JOB=${SETUP_JOB%%;*}
printf '%s\n' "$SETUP_JOB"
```

成功后：

```bash
test -f "$WF_ROOT/SETUP_READY"
cat "$WF_ROOT/environment-freeze.txt"
cat "$WF_ROOT/setup-runtime.patch"
git -C "$WF_ROOT/official-code" remote -v
```

最后一条应只显示 `slahrichi/WildfireSpreadTS`。setup 会缓存 ImageNet 编码器，避免训练时临时联网。已存在的环境/代码目录拒绝被覆盖；安装失败先检查日志，不要直接反复覆盖或升级依赖。

## 4. 下载原始四年 WSTS

```bash
DOWNLOAD_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --cpus-per-task=1 --mem=4G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-download.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" download)
DOWNLOAD_JOB=${DOWNLOAD_JOB%%;*}
```

实际地址是 `https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content`。MD5 应为 `dc1a04e63ccc70037b277d585b8fe761`；只在校验通过后把 `.partial` 改为 `.zip`。日志应显示校验 `OK`。网络下载可续传，训练失败不能因此被默认自动重试。

## 5. CPU 转换：GeoTIFF → HDF5

```bash
DATA_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$SETUP_JOB:$DOWNLOAD_JOB" \
  --cpus-per-task=4 --mem=32G --tmp=200G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-convert.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" convert)
DATA_JOB=${DATA_JOB%%;*}
```

GeoTIFF 是逐日地理栅格，HDF5 将同一事件整理成 `day × channel × height × width`，减少训练时频繁打开小文件的开销。转换直接调用作者 `src/preprocess/CreateHDF5Dataset.py`，原始活跃火点 HHMM 值转为小时；预测目标是次日活跃火点代理，不是完整火场边界。

成功标准：2018/2019/2020/2021 分别为 **176/74/201/156** 个 HDF5，共 **607** 个事件。脚本打印数量并写 READY：

```bash
test -f "$WF_ROOT/data/hdf5/READY"
tail -n 30 "$WF_ROOT/logs/$DATA_JOB-convert.out"
```

不要把新增四年 `WSTSPlus.zip` 直接塞给此转换器：扩展包的日期/编码/缺失地理信息需要另做处理；这不是本次官方十二 fold 的输入。

## 6. 一个验证 batch + 一步 GPU smoke

```bash
SMOKE_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$DATA_JOB" --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=64G --time=00:20:00 \
  --output="$WF_ROOT/logs/%j-smoke.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" smoke 2)
SMOKE_JOB=${SMOKE_JOB%%;*}
```

成功后：

```bash
test -f "$WF_ROOT/runs/smoke-fold2-$SMOKE_JOB/SMOKE_PASSED"
tail -n 40 "$WF_ROOT/logs/$SMOKE_JOB-smoke.out"
```

smoke 只训练一步，batch 4，验证一个 batch，不测试。没有最佳验证 checkpoint 的提示在该一步检查中可能出现；它不代表完整训练已完成，也不能用来报告论文 AP。正式任务恢复 batch 64 和完整验证集。

## 7. 正式训练官方 Fold 2

```bash
TRAIN_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$SMOKE_JOB" --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%j-train.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" train 2)
TRAIN_JOB=${TRAIN_JOB%%;*}
```

任务调用的训练程序是**作者的 `official-code/src/train.py`**。关键设置：

| 参数 | 值与含义 |
| --- | --- |
| fold / seed | Fold 2、seed 0，是两个不同概念 |
| 数据 | `additional_data=false`；2018/2020 train、2019 val、2021 test |
| 模型 | Res18-U-Net、T=1、All、ImageNet 编码器初始化 |
| 更新预算 | 10000 步、batch 64、AdamW、lr 0.001、官方 Focal loss |
| 选择与评价 | 用 validation AP 选最佳 checkpoint，训练结束后官方入口自动测试它 |
| 时间对齐 | 保留 `n_leading_observations_test_adjustment=5` |

有效 positive class weight 由官方入口按训练年份重算，不能因为 YAML 中看到 236 就手动改回 236。历史 Fold 2 有效值为 608.4653828020165。

成功后：

```bash
export TRAIN_RUN="$WF_ROOT/runs/train-fold2-$TRAIN_JOB"
cat "$TRAIN_RUN/result.json"
cat "$TRAIN_RUN/command.txt"
sacct -j "$TRAIN_JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

验收要求：`COMPLETED 0:0`，日志确认 10000 步完成，六项测试指标合法，一个最佳 checkpoint，以及 `result.json`。最佳 checkpoint 的保存步数可以早于 10000：训练做满预算，测试选择验证集表现最好的那个，而不是强行选最后一步。

**历史参照：** 自己训练的官方 Fold 2 测试 AP **0.554664**；作者发布权重同 fold 实测 **0.570902**。这些是历史记录，不是承诺新服务器逐位复现的数值。记录自己实际测得的 AP，不为了接近它们使用测试集调参。

学生到这里就完成了一个官方 baseline fold。

## 8. 可选：仅评价官方 Fold 2 权重

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

结束后读取 `runs/test-weight-fold2-JOBID/result.json`。发布文件 `fold2_testAP0.571.pth` 的 0.571 是四舍五入标签，不是学生自己训练的成绩。此评价不进行训练；如果只想验证官方权重，可以跳过第 7 节的 10000 步训练。

## 9. 可选：完整 12-fold

### 9.1 从头训练十二个 fold

```bash
FULL_ARRAY=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$SMOKE_JOB" --array=0-11%2 --gpus="$WF_GPU" \
  --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%A_%a-train12.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" train)
FULL_ARRAY=${FULL_ARRAY%%;*}
```

`%2` 限制同时运行两个 fold；每个 fold 都是完整独立训练。这个独立 campaign 会包含 Fold 2；若不想重复训练，选择单 fold 教学或全数组之一，不默认混入旧运行。

### 9.2 或者只评价十二个发布权重

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

### 9.3 汇总实际完成的十二个 fold

选择一种已提交的数组：

```bash
# 发布权重评价用此行；如果做的是训练，改成 ARRAY_ID="$FULL_ARRAY"。
ARRAY_ID="$WEIGHT_ARRAY"
sbatch --account="$WF_CPU_ACCOUNT" --dependency="afterok:$ARRAY_ID" \
  --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --output="$WF_ROOT/logs/%j-summary.out" \
  "$WF_ROOT/scripts/job.sh" "$WF_ROOT/config.env" aggregate "$ARRAY_ID"
```

汇总文件为 `$WF_ROOT/summary-ARRAY_ID.json`，包含每 fold AP、均值、总体标准差和样本标准差。缺一个 fold、重复 fold 或混合训练与发布权重结果都会报错。

### 9.4 与论文数字如何比较

| 参照 | AP | 能说明什么 |
| --- | ---: | --- |
| 作者 WSTS+ 工作的 Res18-U-Net T=1 All 结果表 | 0.460 ± 0.084 | 论文/作者表格参照 |
| 我们历史上重算十二个发布权重 | 0.452764 ± 0.088217 | 总体标准差；支持发布权重可执行复现 |
| 我们历史完整训练 Fold 2 | 0.554664 | 一个测试年份的单 fold 结果 |

作者表格见[官方 README](https://github.com/slahrichi/WildfireSpreadTS#benchmark-results-ap--standard-deviation)。历史十二权重均值与参照相差约 -0.007236，数值接近，但不能据此证明论文表格与发布资产具有完全相同的运行来源。文件名分数统计也不能替代真实评价。单 fold 高于跨 fold 均值很正常。

新服务器的本次结果应单独报告；不要把以上历史数字直接填成自己的实验结果。

## 10. 常见问题与交付清单

| 问题 | 处理 |
| --- | --- |
| module、GPU 名称或 account 不存在 | 对照新集群的 `module avail`、`sinfo` 和账户；不能直接复制旧服务器值 |
| `--no-index` 找不到依赖 | 确认同类型集群的软件源和固定 wheel 可用性；不要盲目升级全部依赖 |
| 下载/安装连接失败 | 确认计算节点出站网络；申请允许联网的节点/镜像渠道 |
| `SLURM_TMPDIR` 未定义、临时空间不足 | 转换需要管理员提供节点临时目录；按本集群约定设置，不使用教师目录 |
| `DependencyNeverSatisfied` | 检查前置任务失败原因；修复后用新的 job ID 重新提交下游 |
| 目录已存在 | 脚本拒绝覆盖已有环境和科学运行，先保留日志并定位失败阶段 |
| CUDA/CPU OOM | 根据 `nvidia-smi`、MaxRSS 增加显存/内存；不擅改正式 batch、特征或精度 |
| 已有 checkpoint 却没有 result.json | 检查训练是否做满、测试是否结束、指标是否完整；不能手写成功文件 |
| 分数不同 | 先核对数据、fold、参数、软件版本和测试口径，不使用测试 AP 挑配置 |

交付：config、官方 commit 和补丁、环境 freeze、下载校验、转换数量、job ID、真实命令、日志、checkpoint 路径和结果 JSON。大数据和模型留在个人实验存储中；同目录内不要并行改动上游代码或安装环境。

## 11. 本版独立性与验证边界

本文所有脚本都由第 2 节现场创建。运行时只依赖学生自己的目录、作者仓库、官方数据与权重，以及同类型集群提供的软件环境。全文没有要求取得 `wildfire-research-plan` 代码，也不调用其 Python 包。

上一版依赖本项目的 corrected B0 路线另存为 `project-b0-slurm.md`，不属于本独立教程的执行步骤。

新服务器尚未连接验证。文档中的历史成绩与此前 Nibi smoke 不能冒充此独立版本在新服务器的完整训练结果；本版新增脚本的具体本地检查范围记录在同目录 `official-standalone-validation.md` 中，该记录不是学生执行依赖。

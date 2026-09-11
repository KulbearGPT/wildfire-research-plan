# 从下载数据到跑完 ResNet18-U-Net：Nibi Slurm 手把手教程

面向第一次接触本项目的学生。命令使用 **Bash + Alliance Nibi**，从一个空的个人实验目录开始。整理日期：2026-09-11。

本教程配套脚本位于 [`res18/`](res18/)，脚本已经随仓库保存；无需从老师的个人缓存目录寻找文件。它们调用本项目已有的官方复现入口和 B0 入口，不另写一套模型训练逻辑。

## 0. 先选定你要完成哪件事

模型统一叫 **ResNet18-U-Net / Res18-U-Net**：ResNet18 提取特征，U-Net 解码器逐像素输出下一日活跃火点的概率。`T=1` 表示输入一天的观测；All 表示采用该配置下的全部特征。原始 HDF5 有 23 个波段，类别与方向等特征经过模型数据管线处理后形成 40 个输入通道。

但“同一个模型”不意味着“同一个实验”。

| 路径 | 数据与划分 | 训练量 | 结果应该与谁对照 |
| --- | --- | --- | --- |
| **本项目 B0，最终接入研究主线** | WSTS+ 八年；2016–2020 train，2021 val，2022/2023 test；修正样本索引，训练集统计量 | seed 0、3000 步；一套固定划分，没有 12-fold 循环 | 本项目 B0 台账 |
| **官方 Fold 2，学习单 fold 论文复现** | 原始 WSTS 2018–2021；2018/2020 train，2019 val，2021 test；保留锁定官方数据管线 | seed 0、10000 步 | 我们已有的 Fold 2 训练与发布权重结果 |
| **官方完整 12-fold** | 官方 `additional_data=false` 的 12 个年份组合 | 每 fold 10000 步，或仅测试 12 个发布权重 | WSTS+ 工作的官方 Res18-U-Net T=1 All 参照 `0.460 ± 0.084`，但不保证论文表格来源完全相同 |

**只做 B0：按 1–6 → 9–10 执行，第 11 节测试可选。只学一个官方 fold：按 1–7 执行，第 8 节发布权重评价可选。** 第 12 节的完整 12-fold 为扩展练习，不是学生必做项。两条路径共用原始数据和模型环境。

WSTS+ 是论文/扩展基准的名字，不代表它发布的每组权重都采用八年划分。锁定代码中 `additional_data=true` 只有 **4** 个八年 fold；不能把它设成 true 后提交 0–11。项目 B0 又是单独实现的前向时间划分。本教程用显式参数避免这三个协议混淆。

论文参考数来自[作者仓库结果表](https://github.com/slahrichi/WildfireSpreadTS#benchmark-results-ap--standard-deviation)。Fold 与代码口径以项目的 [upstream lock](../../reproductions/wsts_res18_unet_t1/upstream.lock.json)、[发布权重清单](../../reproductions/wsts_res18_unet_t1/official_weights_manifest.json)和 [B0 contract](../../reproductions/wsts_fast_track/contract.py)为准。

## 1. 登录并建立自己的目录

### 1.1 教师只做一次：给学生一个包含本教程的仓库版本

本地 commit 不会自动出现在 GitHub。教师可以发布相应分支，也可以提供 Git bundle。下面从**含本教程的 main 分支**创建独立文件，再把 bundle 放到学生可读的项目目录：

```bash
cd /home/kulbear/scratch/wildfire-research-plan
git bundle create /tmp/wildfire-res18-course.bundle main
git bundle verify /tmp/wildfire-res18-course.bundle
```

将这个 bundle 复制到课程共享目录。学生下面填入该文件的绝对路径即可 `git clone`；也可以填入已经发布了本教程的 Git 仓库 URL。不要让学生直接修改老师正在跑实验的 worktree。

### 1.2 学生在自己电脑的终端登录

```bash
ssh YOUR_ALLIANCE_USERNAME@nibi.alliancecan.ca
```

`YOUR_ALLIANCE_USERNAME` 换成你的 Alliance 用户名。以下所有命令都在登录 Nibi 后执行。登录节点只编辑脚本、查看轻量结果、提交和查看作业；下载、解压、遍历数据、训练和模型评价交给 Slurm。

### 1.3 查看可用账户和 GPU 名称

```bash
sacctmgr -nP show assoc where user="$USER" format=Account,Partition
sinfo -h -o '%P %G'
```

本项目教师账户是 `def-vislearn_cpu` / `def-vislearn_gpu`，学生应以自己实际关联的账户为准。下列输入只做一次，不要把 `replace-me` 一类占位符原样提交。

```bash
read -r -p '项目组目录，例如 /project/6085198：' WF_GROUP
read -r -p '你的 CPU Slurm account：' WF_CPU_ACCOUNT
read -r -p '你的 GPU Slurm account：' WF_GPU_ACCOUNT
read -r -p '包含本教程的仓库 URL 或 Git bundle 绝对路径：' WF_SOURCE
export WF_ROOT="$WF_GROUP/$USER/wildfire-res18-course"
mkdir -p "$WF_ROOT"/{downloads,hdf5,envs,cache,runs,logs,weights}
export WF_REPO="$WF_ROOT/repo"
git clone --branch main "$WF_SOURCE" "$WF_REPO"
test -f "$WF_REPO/docs/tutorials/res18/official.sh"
cd "$WF_REPO"
git rev-parse HEAD
```

创建自己的配置，路径不会写死在公共脚本中：

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

这里申请一个 H100 的 10GB/20GB MIG 分片，不是 10/20 块 GPU。20GB 用于首次完整训练和精确 AP 测试的保守起点；测得显存占用后可缩小，但不要为塞进小分片随意降低正式 batch size。若 `sinfo` 未列出这些名称，按该集群实际资源名称修改配置；其他集群不能直接照搬 Nibi 的 module 和 GPU 名称。

重新登录后恢复配置：

```bash
read -r -p '你的 tutorial.env 绝对路径：' WF_TUTORIAL_ENV
export WF_TUTORIAL_ENV
source "$WF_TUTORIAL_ENV"
cd "$WF_REPO"
```

**空间规划：** 原始 ZIP 约 48.4GB、扩展 ZIP 约 19.9GB；还要保留 HDF5、扩展年验证副本、环境和 checkpoint。建议先确认至少约 400GB 项目可用配额，并为原始解压预留约 200GB 节点临时空间。这是保守准备量，不是测得峰值；并发数据作业会增加需求。

```bash
df -h "$WF_ROOT"
quota -s
```

`df` 是文件系统剩余量，不是个人项目配额；如果站点 `quota` 不报告项目配额，按课程组提供的配额查询方式确认。

## 2. 看懂一次 Slurm 提交

`sbatch` 把任务交给调度器并立刻返回 job ID；返回 ID **不代表训练完成**。`--cpus-per-task` 是 CPU 核数，`--mem` 是主机内存，`--gpus` 是 GPU，`--time` 是最长墙钟时间。脚本成功退出要求 `COMPLETED` 且 `ExitCode=0:0`；科研结果还要检查输出文件。

本教程用 `--parsable` 保存 job ID，用 `afterok` 表示前置作业成功后才运行，防止下载失败仍继续训练。`%j` 在日志文件名中展开成 job ID；数组作业用 `%A_%a` 展开成数组 ID 和 fold ID。

下面定义一个登录终端函数，之后每一步用它检查状态：

```bash
jobcheck() {
  squeue -j "$1" -o '%.18i %.24j %.10T %.10M %.25R'
  sacct -j "$1" --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
}
```

作业结束后 `squeue` 为空是正常的，以 `sacct` 和日志为准。每节提交后，等待 `COMPLETED 0:0` 再执行该节的“成功后”文件检查；作业尚未启动时日志文件不存在是正常的。`afterok` 可用于提前排队，但不能把任务仍在排队理解为数据已准备好。每个科学运行使用独立输出目录；失败后先诊断，不能覆盖原目录伪装成第一次成功运行。重新登录需要重新定义 `jobcheck`，并从记录或 `sacct` 恢复后续依赖用的 job ID。

## 3. 创建两个隔离环境

训练依赖较旧的 Lightning/SMP，采用本项目已使用过的 Python 3.10 环境输入；数据审计包要求 Python 3.13。把二者分开可以避免一次安装改变另一侧的 NumPy/PyTorch 版本。

```bash
WF_SETUP_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=res18-setup --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$WF_ROOT/logs/%j-setup.out" "$WF_SCRIPTS/setup.sh")
WF_SETUP_JOB=${WF_SETUP_JOB%%;*}
printf '%s\n' "$WF_SETUP_JOB"
jobcheck "$WF_SETUP_JOB"
tail -n 40 "$WF_ROOT/logs/$WF_SETUP_JOB-setup.out"
```

脚本完成的工作：创建环境；安装 [`requirements-training.txt`](res18/requirements-training.txt)；锁定上游 commit `ed221d4…`；应用仓库已有的 Res18 导入裁剪补丁和移除未使用的 `T_co` 类型导入；缓存 ResNet18 ImageNet 初始化。补丁不改变网络、损失或训练数据索引，实际差异会随运行保存。模型作业还会用 `git archive HEAD` 保存并执行本项目已提交的源码快照；教程脚本另存一份，所以提交前先 commit 你确实要使用的项目代码，运行中不要编辑上游 checkout。

**成功标准：** Slurm `0:0`，且以下文件存在：

```bash
test -f "$WF_ROOT/envs/READY"
cat "$WF_ROOT/envs/training-freeze.txt"
cat "$WF_ROOT/envs/audit-freeze.txt"
```

安装使用 Alliance wheelhouse 的 `--no-index`。如果某个固定版本不可用，保留日志并请教师确认兼容版本/已验证环境；不要直接 `pip install -U` 整套依赖。已安装环境拒绝被脚本覆盖。上游仓库与预训练编码器的小型下载也在该 CPU 作业完成。

注意：`encoder_weights=imagenet` 是公开的 ImageNet 初始化。项目记录中的“from scratch”表示不从我们已有的野火模型 checkpoint 续训，不代表 ResNet18 所有参数都随机初始化。

## 4. 下载原始 WSTS 并校验

原始 WSTS 覆盖 2018–2021，用于官方单 fold；它也是组装八年 WSTS+ 的一部分。原始数据发布入口是 [Zenodo 8006177](https://zenodo.org/records/8006177)。脚本锁定文件 `WildfireSpreadTS.zip`，MD5 为 `dc1a04e63ccc70037b277d585b8fe761`。

```bash
WF_DOWNLOAD_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --job-name=wsts-download --cpus-per-task=1 --mem=4G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-download.out" "$WF_SCRIPTS/download.sh" original)
WF_DOWNLOAD_JOB=${WF_DOWNLOAD_JOB%%;*}
jobcheck "$WF_DOWNLOAD_JOB"
tail -n 20 "$WF_ROOT/logs/$WF_DOWNLOAD_JOB-download.out"
```

脚本中的实际下载与校验逻辑相当于：

```bash
# 以下是解释用片段，已由上面的 Slurm 作业执行；不要在登录节点重复下载。
curl -fL --retry 8 --retry-delay 5 --continue-at - \
  'https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content' \
  -o WildfireSpreadTS.zip.partial
printf '%s  %s\n' dc1a04e63ccc70037b277d585b8fe761 WildfireSpreadTS.zip.partial | md5sum -c -
```

**为什么校验：** 下载退出码为零不一定代表拿到了完整、正确的版本。脚本只有校验通过才把 `.partial` 改成正式 `.zip`。网络中断可保留 partial 后重新提交下载步骤；正式训练的失败不能按同样方式盲目自动重试。

## 5. CPU 作业：GeoTIFF → HDF5

GeoTIFF 每天一个文件，适合地理栅格交换；训练频繁读取时间序列，把同一事件整理为 HDF5 能减少文件打开开销。每个事件组织为 `data[day, channel, height, width]`，并保存日期、年份、事件名及位置元数据。

```bash
WF_ORIGINAL_JOB=$(sbatch --parsable --account="$WF_CPU_ACCOUNT" \
  --dependency="afterok:$WF_SETUP_JOB:$WF_DOWNLOAD_JOB" \
  --job-name=wsts-hdf5 --cpus-per-task=4 --mem=32G --tmp=200G --time=08:00:00 \
  --output="$WF_ROOT/logs/%j-original.out" "$WF_SCRIPTS/prepare-original.sh")
WF_ORIGINAL_JOB=${WF_ORIGINAL_JOB%%;*}
jobcheck "$WF_ORIGINAL_JOB"
tail -n 40 "$WF_ROOT/logs/$WF_ORIGINAL_JOB-original.out"
```

脚本在 `$SLURM_TMPDIR` 解压，调用锁定上游的 `src/preprocess/CreateHDF5Dataset.py`，把持久输出写到 `$WF_ROOT/hdf5/original/`。该官方转换器只处理 2018–2021，不能拿它直接完成扩展年份。

**成功标准：** 总计 607 个事件，逐年为 176、74、201、156；日志输出 `status: pass`，然后写入 READY：

```bash
test -f "$WF_ROOT/hdf5/original/READY"
```

输出目录是 `original/2018/*.hdf5` 等四个年份目录。原始活跃火点波段使用 HHMM 编码，这一步将其变成小时；目标生成时使用是否有活跃火点。扩展年份的编码不同，第 9 节有单独处理，不能重复除以 100。

**到这里数据准备完成。两条路径都先做第 6 节 GPU smoke；通过后，B0 路径跳到第 9 节，官方单 fold 路径继续第 7 节。**

## 6. 官方 Fold 2：先运行一步 smoke

smoke 是工程检查：取训练批次、反向传播，并检查验证批次。它只做 1 步，batch 4，仅一个验证批次，不启用测试；不会产生可以与论文比较的 AP。

```bash
WF_SMOKE_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_ORIGINAL_JOB" --job-name=res18-smoke \
  --gpus="$WF_GPU_SMOKE" --cpus-per-task=8 --mem=64G --time=00:20:00 \
  --output="$WF_ROOT/logs/%j-smoke.out" "$WF_SCRIPTS/official.sh" smoke 2)
WF_SMOKE_JOB=${WF_SMOKE_JOB%%;*}
jobcheck "$WF_SMOKE_JOB"
tail -n 60 "$WF_ROOT/logs/$WF_SMOKE_JOB-smoke.out"
```

成功后：

```bash
export WF_SMOKE_RUN="$WF_ROOT/runs/official-smoke-fold2-$WF_SMOKE_JOB"
test -f "$WF_SMOKE_RUN/SMOKE_PASSED"
cat "$WF_SMOKE_RUN/command.txt"
```

日志应包含 `max_steps=1` 完成提示和正的 `WSTS_OBSERVER_PEAK_ALLOCATED_BYTES`。只看到 `CUDA available: True` 不算模型跑通。

## 7. 官方 Fold 2：10000 步训练，自动测试最佳验证 checkpoint

```bash
WF_FOLD_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_SMOKE_JOB" --job-name=res18-fold2 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%j-fold2.out" "$WF_SCRIPTS/official.sh" full 2)
WF_FOLD_JOB=${WF_FOLD_JOB%%;*}
jobcheck "$WF_FOLD_JOB"
tail -n 50 "$WF_ROOT/logs/$WF_FOLD_JOB-fold2.out"
```

`full 2` 中的 `2` 是 fold 编号，不是 seed。这个脚本明确传入：

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

**为什么这样选 checkpoint：** 优化器做满 10000 次更新，validation AP 决定保存哪一个 checkpoint，官方 `train.py` 最后用它测试。最佳 checkpoint 的 `global_step` 可以小于 10000；这并不等于少训练了。测试集不用于选择参数或最好 seed。

损失保持官方 Focal、优化器保持 AdamW、学习率 0.001。YAML 中的 `pos_class_weight: 236` 会被官方训练入口依据训练年份的 fire rate 重算；我们保存的 Fold 2 有效值是 `608.4653828020165`。不要为了接近某个分数手动把它改回 236。

成功后阅读结果：

```bash
export WF_FOLD_RUN="$WF_ROOT/runs/official-full-fold2-$WF_FOLD_JOB"
cat "$WF_FOLD_RUN/result.json"
cat "$WF_FOLD_RUN/command.txt"
```

`result.json` 含 AP、F1、IoU、precision、recall、loss、最佳 checkpoint 路径及步数、原始日志 SHA-256。脚本只在训练完成标志、测试指标和 checkpoint 检查通过后写此文件。

**已有参照，不是本次重跑结果：** 历史 Windows 完整 Fold 2 训练测试 AP 为 **0.554664**；发布权重 Fold 2 实测 AP 为 **0.570902**。单 fold 分数可以高于论文跨 fold 均值 **0.460**，因为测试年份不同。Nibi 环境与历史 Windows 环境不同，不承诺逐位相等；应保留实际结果，不能用目标数替换。

## 8. 可选：测试官方 Fold 2 发布权重

这一步快速检查数据与评价入口是否和官方发布资产相容。它**不训练**，也不是你刚训练出来的 checkpoint。

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

下载使用固定 Hub revision `acf70a3…`，逐文件检查大小和 SHA-256；评价再次校验后 `strict=True` 加载 raw state dict，只调用 `Trainer.test`。不能把 `.pth` raw state dict 当成 Lightning `.ckpt` 用来 resume。

参考文件是 `fold2_testAP0.571.pth`，历史重算 `0.570902` 与文件名四舍五入结果一致。所有权重来自[作者发布仓库](https://huggingface.co/saadlahrichi/WSTSPlus/tree/acf70a37394849f4ec8d108a51d6f4325a554d0a/trained_model_weights/Res18Unet_T1/All)。

## 9. 接入项目 B0：准备扩展年份并重新计算训练统计量

WSTS+ 的下载包是**新增四年**，不是完整八年包。扩展数据发布于 [Zenodo 17584629](https://zenodo.org/records/17584629)，文件 `WSTSPlus.zip` 约 19.9GB，MD5 `42da7598cc33a170064e78d8027148c9`。

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

该 CPU 作业按顺序完成：

1. 解压扩展四年，用随教程保存的转换器生成事件 HDF5。保留源数据中已经是小时的活跃火点值；没有 CRS 的事件记录位置缺失，不能捏造经纬度。
2. 用项目 `repair-active-fire` 在新目录生成规范化标签及证据。转换后已正确的标签仍经过同一验证流程，不需要先故意制造错误。
3. 用独立 `verify_repair` 检查事件数、目标天数与正像素计数，验证确切的五个空事件排除项。
4. 把原始四年与验证后的四年硬链接为 `hdf5/combined/`；硬链接要求同一文件系统，因此这些目录统一放在自己的 `WF_ROOT` 下。之后不要就地修改任何硬链接数据。
5. 执行全数据审计；只用 **2016–2020** 的训练样本计算 normalization statistics。验证年和测试年不参与这一步。

成功后：

```bash
test -f "$WF_ROOT/hdf5/combined/READY"
test -f "$WF_ROOT/hdf5/train-stats.npz"
cat "$WF_ROOT/runs/prepare-plus-$WF_PLUS_JOB/data-summary.json"
cat "$WF_ROOT/runs/prepare-plus-$WF_PLUS_JOB/audit/contract_decision.json"
cat "$WF_ROOT/runs/prepare-plus-$WF_PLUS_JOB/audit/phase0_report.md"
```

实际可用事件数如下；不能因为论文写了 1005 就补造不存在的事件：

| 年份 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HDF5 事件 | 92 | 110 | 176 | 74 | 201 | 156 | 122 | 68 | **999** |

训练/验证/测试事件数应为 **653 / 156 / 190**。审计预期为 `continue_controlled`：可以继续受控缺失实验，但数据尚不支持自然缺失和实际业务部署声明。具体原因见 [phase0 数据报告](../experiments/phase0.md)。

**为什么重新计算统计量：** 每个输入波段的量纲不同，均值/标准差用于标准化；如果从测试年计算这些量就让测试信息进入了训练。官方 fold 复现继续用官方统计方式，项目 B0 则使用此处冻结训练年的统计，两者不能互换。

## 10. 训练项目 B0：3000 步 clean baseline

这才是 [roadmap](../research-roadmap.md) 中的 B0。`clean` 表示训练时不人工加入 FireDrop/BlockDrop；原始输入里的缺失仍按现有数据管线处理。此时不添加可靠性提示、一致性损失或专家路由。

```bash
WF_B0_JOB=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_PLUS_JOB:$WF_SMOKE_JOB" --job-name=project-B0 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=03:00:00 \
  --output="$WF_ROOT/logs/%j-B0.out" "$WF_SCRIPTS/b0.sh")
WF_B0_JOB=${WF_B0_JOB%%;*}
jobcheck "$WF_B0_JOB"
tail -n 60 "$WF_ROOT/logs/$WF_B0_JOB-B0.out"
```

脚本的训练核心就是仓库已有模块：

```text
python -m reproductions.wsts_fast_track.train_corrected_baseline
  --baseline-id B0
  --upstream-root "$WF_ROOT/upstream"
  --data-root "$WF_ROOT/hdf5/combined"
  --run-root "本次作业独立目录/scientific"
  --stats-path "$WF_ROOT/hdf5/train-stats.npz"
```

这段用于解释参数；真正逐行执行的是上面的 `sbatch`。入口固定 seed 0、3000 步、batch 64、AdamW 0.001、2016–2020 train / 2021 val，并修正跨年样本索引。它不是通过改 `fold_id` 来实现划分，因此不能把 B0 的 fold 0 解读为官方 Fold 0。

任务训练完会自动生成完成记录并评价 2021 四个场景。M00 是 clean baseline 主要参照；M01/M06/M07 是不重新训练模型的鲁棒性诊断。

```bash
export WF_B0_RUN="$WF_ROOT/runs/B0-$WF_B0_JOB"
cat "$WF_B0_RUN/scientific/completed.json"
cat "$WF_B0_RUN/results-2021/summary.json"
```

**B0 完成验收：** Slurm `COMPLETED 0:0`；`completed.json` 中 `baseline_id=B0`、`max_steps=3000`、`corrected_index=true`、`test_enabled=false`；最佳 checkpoint 文件存在；`results-2021/summary.json` 有四个场景。

已有 B0 台账（不同环境重跑不要求逐位相等）：

| 年份 | M00 clean AP | M01 AP | M06 AP | M07 AP |
| --- | ---: | ---: | ---: | ---: |
| 2021 验证 | 0.580988 | 0.039387 | 0.323805 | 0.133672 |
| 2022 固定测试 | 0.278852 | 0.006669 | 0.135278 | 0.071779 |
| 2023 固定测试 | 0.412674 | 0.010101 | 0.205411 | 0.078691 |

这些是本项目历史 B0 评价口径，来自[量化台账](../experiments/quantitative_reliability_ledger.md)。不能把 2021 的 0.580988 直接和论文 12-fold 的 0.460 比大小，也不能混入后来跨 T 对齐目标日期的主表。训练器内部验证 AP 与外部受控评价可能使用不同目标日期集合；比较时先确认 loader 和样本数，不只看指标名字。

**学生到这里就已经跑完一个 B0。** 后续固定测试和完整官方 12-fold 都是扩展步骤。

## 11. 可选：对固定 B0 一次性测试 2022/2023

仅在配方已经冻结后执行。下面明确打开已有评价入口的 `--heldout-authorized`，不训练、不选新 checkpoint、不调阈值。学生复现的是已公开给本项目的历史测试结果，不能称作自己的全新未见测试。

```bash
export WF_B0_RECORD="$WF_B0_RUN/scientific/completed.json"
for year in 2022 2023; do
  sbatch --account="$WF_GPU_ACCOUNT" --job-name="B0-test-$year" \
    --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=01:00:00 \
    --output="$WF_ROOT/logs/%j-B0-test.out" \
    "$WF_SCRIPTS/evaluate-b0.sh" "$WF_B0_RECORD" "$year"
done
```

记录这两个 job ID。输出分别位于 `runs/B0-test-2022-JOBID/results-2022/summary.json` 和 `runs/B0-test-2023-JOBID/results-2023/summary.json`。

## 12. 完整复现附录：12-fold，训练和发布权重评价分开汇总

### 12.1 从头训练全部 12 个官方 fold

这一步在第 6 节 smoke 通过后执行。每个数组任务训练 10000 步并测试各自验证集选择的最佳 checkpoint；`%2` 限制最多两个 fold 同时运行，不是把 batch 或数据分成两份。

```bash
WF_FULL_ARRAY=$(sbatch --parsable --account="$WF_GPU_ACCOUNT" \
  --dependency="afterok:$WF_SMOKE_JOB" --array=0-11%2 --job-name=res18-full12 \
  --gpus="$WF_GPU_TRAIN" --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output="$WF_ROOT/logs/%A_%a-full12.out" "$WF_SCRIPTS/official.sh" full)
WF_FULL_ARRAY=${WF_FULL_ARRAY%%;*}
jobcheck "$WF_FULL_ARRAY"
```

如果之前已经完成 Fold 2，这个独立数组会再次训练 Fold 2，以获得单独、完整的课程 campaign；资源有限时不要同时做“单 fold 完整训练”和“全数组”。已有 Fold 2 也可以复用，但必须由教师核对版本与参数后显式选取文件，不能自动混进数组结果。

每个目录名含 `official-full-foldN-…`。Slurm 对数组的 `SLURM_JOB_ID` 与主数组 ID 不必一致，所以用一个小清单定位本次 campaign 的输出：

```bash
sacct -nP -j "$WF_FULL_ARRAY" --format=JobIDRaw,State,ExitCode
```

全部完成后，只选本次 12 个 `result.json`，写进文件，每行一个绝对路径。例如可以在 CPU 小作业中通过数组元数据匹配，下面第 12.3 节给出通用方法。

### 12.2 只评价全部 12 个发布权重

这通常比训练 12 次省时，适合学习如何核验已发表 baseline 的评价口径。它证明的是发布权重可执行复现，不能替代“自己从头训练了 12 个 fold”。

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

### 12.3 汇总本次数组，检查 12 个 fold 一个不缺

每个运行保存 `slurm-job.txt`，其中包含 `ArrayJobId`。先选择你实际运行的数组类型：

```bash
# 如果汇总发布权重：
export WF_SUMMARY_ARRAY="$WF_WEIGHT_ARRAY"
export WF_SUMMARY_MODE=weight
# 如果汇总完整训练，则改成：
# export WF_SUMMARY_ARRAY="$WF_FULL_ARRAY"
# export WF_SUMMARY_MODE=full
```

提交小型 CPU 汇总作业：

```bash
sbatch --account="$WF_CPU_ACCOUNT" --dependency="afterok:$WF_SUMMARY_ARRAY" \
  --job-name=res18-summary --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --output="$WF_ROOT/logs/%j-summary.out" "$WF_SCRIPTS/summarize-array.sh" \
  "$WF_SUMMARY_MODE" "$WF_SUMMARY_ARRAY"
```

最终文件：

```bash
cat "$WF_ROOT/runs/summary-$WF_SUMMARY_MODE-$WF_SUMMARY_ARRAY.json"
```

汇总脚本要求 fold 0–11 恰好各一个，且模式相同；缺失、重复、训练与权重混用均会失败。它同时输出 AP 均值、总体标准差 `ddof=0` 和样本标准差 `ddof=1`，报告时说清选择哪一个。不能先平均每年的 AP 再把它当成 12-fold 均值。

### 12.4 怎样理解“接近原论文数字”

下表是已有独立复算记录，不是本教程新执行的 12-fold 结果：

| 比较对象 | AP | 说明 |
| --- | ---: | --- |
| WSTS+ 工作官方表中 Res18-U-Net T=1 All | 0.460 ± 0.084 | 论文/作者仓库参照 |
| 十二个发布权重历史逐 fold 复算 | **0.452764 ± 0.088217** | 标准差为总体标准差；均值与论文参照相差约 -0.007236 |
| 十二个权重文件名 AP 标签 | 0.452917 ± 0.088272 | 仅文件名四舍五入值的统计，不是新评价 |
| 历史自己训练的 Fold 2 | 0.554664 | 单 fold，不能作为十二 fold 均值 |
| 历史发布权重 Fold 2 | 0.570902 | 单 fold，和文件名 0.571 一致 |

完整逐 fold 指标、运行与独立验证记录见[官方复现报告](../experiments/res18_unet_t1_reproduction.md)及[复现 README](../../reproductions/wsts_res18_unet_t1/README.md)。

“接近”是已有结果的描述，不是调参目标或本教程的通过条件。当前记录不能证明发布权重与论文表格采用了完全同一来源、运行与聚合过程；官方 positive-weight 配置的意图也存在历史未解边界。正确交付是报告真实分数与差异，而不是不断改参数直到撞上 0.460。

## 13. 常见故障：看哪里，怎么处理

| 现象 | 先检查 | 正确处理 |
| --- | --- | --- |
| `Invalid account` / GPU 类型不存在 | `sacctmgr`、`sinfo` 与配置 | 改为自己的 account / 实际资源名，不照抄教师账号 |
| `DependencyNeverSatisfied` | 前置 job 的 `sacct` 和日志 | 修复前置失败，保存新 job ID，再重新提交下游；旧依赖不会自动恢复 |
| 环境安装 `No matching distribution` | `setup` 日志中的具体固定版本 | 请教师确认 wheelhouse 与已有验证环境，不随意升级整套模型依赖 |
| ZIP 校验失败 | `.partial`、磁盘配额、下载日志 | 不解压坏包；保留证据后重新获取文件 |
| 扩展年目标几乎全零 | 是否错误使用官方原始四年转换器 | 回到第 9 节，按扩展年的小时编码转换和验证 |
| `refusing existing target` / `FileExistsError` | 上一次失败的目录和日志 | 先定位完成到哪一步；使用新的课程根目录或由教师确认恢复，别删除仍被其他步骤使用的数据 |
| CUDA OOM | 实际申请分片、峰值、完整测试阶段 | 申请更大分片重跑并保留失败日志；正式训练不擅改 batch、特征、精度或 crop |
| CPU `OUT_OF_MEMORY` | `sacct MaxRSS`、worker 数、测试 AP 阶段 | 增加 `--mem`；记录 worker 改动，不能把未完成测试当结果 |
| checkpoint 提示 `weights_only` / 不可信反序列化 | 是否加载自己的 `.ckpt` 或固定清单的官方权重 | 本教程已为这两种可信来源兼容旧 Lightning；不要把该设置用于陌生 checkpoint |
| 有 checkpoint 却没有 `result.json` | 完成标志、测试是否结束、日志解析错误 | checkpoint 不等于整个实验成功，先检查日志；不要伪造完成文件 |
| 排队超过十分钟 | `squeue --start -j JOBID` 与资源试探 | 比较同命令的 `sbatch --test-only` 估计；资源不超过必要量两倍且明显更快时再改请求，避免同时留下两套重复科学任务 |

查看训练进度不会触发新的训练：

```bash
squeue -u "$USER"
squeue --start -j "$WF_FOLD_JOB"
tail -f "$WF_ROOT/logs/$WF_FOLD_JOB-fold2.out"
```

`Ctrl-C` 只结束 `tail`，不会取消作业。确实需要取消时才运行 `scancel JOBID`；不要不加区分地取消自己全部任务。

## 14. 学生最终提交什么

- 使用的课程仓库 commit、上游 commit、`tutorial.env`、两个环境 freeze。
- 下载校验日志、数据转换/审计结果；B0 还包括训练统计量及数据年份说明。
- 自己的 job ID、原始训练日志、实际命令、最佳 checkpoint 路径。
- 单 fold 的 `result.json`，或 B0 的 `scientific/completed.json` 和 `results-2021/summary.json`。
- 一张自己的结果表；明确标注“单 fold / 十二 fold”“自己训练 / 发布权重评价”“官方协议 / 项目 B0”。

一次完整交付应能回答：输入是什么、标签是什么、哪些年份用于训练、何时选择 checkpoint、测试发生在何时、这个分数究竟与哪个参照可比。

## 15. 本教程的验证范围

配套数据转换器来自项目已有 Nibi 执行脚本；模型调用复用现有官方/B0 入口。新增汇总器检查指标合法性、完整 fold 集合与实验类型。具体静态检查、轻量测试与 Slurm smoke 的结果记录在 [validation.md](res18/validation.md)。

全量 ZIP 下载、999 事件转换、10000 步训练和完整 12-fold 不会因撰写教程自动全部重跑；上述历史数字与本次教程验证分开记录。学生首次在自己的账户与新环境执行，仍须按各节成功标准逐步核验。

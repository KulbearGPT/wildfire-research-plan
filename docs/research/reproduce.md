# 在新 Slurm 集群复现现有研究

交付分成两部分：[发现过正向信号的方向](positive-signals.md)保留代码、结果和运行方法；[负结果档案](negative-results.md)记录实现思路、设置及结果。单 seed、单场景的改善也收录，但不会写成已确认的贡献。完整清单见 [method-inventory.json](method-inventory.json)。未运行、取消、比较无效三种状态不算负结果。

**当前资格验证尚未完成。** 下列环境入口已实现；各方法的实数据运行记录将随 Slurm 验证补齐。不要把代码存在或单元测试通过理解为所有历史数字已经重新复现。

## 1. 准备代码和集群配置

这里复现的是我们的研究改动，因此需要本交付版本的源码和 Git 提交记录。可以通过本地 Git bundle 交付，不要求新服务器访问我们的远程仓库或原工作目录。提交器使用 Git 保存每个作业的精确源码快照，所以单独解压 `git archive` 不足以运行下面的提交命令。官方模型代码由安装作业从官方仓库获取；只跑官方 Res18 基线的独立教程仍见 [原教程](../tutorials/res18-baseline-slurm.md)。

在交付版本已提交的源仓库目录生成 bundle 和提交号：

```bash
git bundle create ../handoff.bundle HEAD
git rev-parse HEAD > ../handoff-commit.txt
```

把这两个文件传到新服务器，然后从本地 bundle 建立工作仓库并固定交付提交：

```bash
git clone /path/to/transfer/handoff.bundle wildfire-research-plan
cd wildfire-research-plan
git checkout --detach "$(cat /path/to/transfer/handoff-commit.txt)"
```

bundle 携带该提交及其祖先历史；这里的 clone 读取本地文件，不连接我们的远程仓库。把后面的 `WILDFIRE_REPO` 配置为这个新工作仓库的绝对路径。

需要 Linux、Slurm、NVIDIA GPU，以及 Python 3.10（训练）和 Python 3.13（数据审计）。两套 Python 分开是因为原模型依赖与数据审计包的版本要求不同。训练环境不安装要求 Python 3.13 的根项目包，而从归档源码导入研究模块。

在上述新工作仓库目录执行：

```bash
mkdir -p "$HOME/wildfire-config"
cp configs/research/site.example.env "$HOME/wildfire-config/site.env"
export WILDFIRE_SITE_ENV="$HOME/wildfire-config/site.env"
${EDITOR:-vi} "$WILDFIRE_SITE_ENV"
```

按新服务器修改 `WILDFIRE_REPO`、`WILDFIRE_ROOT`、Python 命令、module 列表及 CPU/GPU 的 Slurm 数组。`WILDFIRE_ROOT` 必须是计算节点可访问、有足够空间的数据盘目录。路径可以包含空格。没有 module 的服务器把两个 module 变量留空；账户、分区、GPU 型号由本站配置决定。

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
bash scripts/research/submit.sh cpu setup
```

提交器会打印 job ID 与归档目录。用 `squeue -u "$USER"` 看队列，用 `sacct -j JOB_ID --format=JobID,State,ExitCode,NodeList` 检查退出状态（把 `JOB_ID` 换成输出的数字）。日志保存在 `$WILDFIRE_ROOT/jobs/`；成功必须同时有 Slurm 状态 `COMPLETED`、退出码 `0:0` 和 `allocation-作业号/setup-completed.txt`。

安装固定版本的官方 WildfireSpreadTS，应用仓库内保留的兼容性补丁，创建全新的两个虚拟环境，并在 CPU 作业内下载 ResNet18 预训练权重。依赖输入见 [训练环境](../../environments/research-training.txt) 和 [审计环境](../../environments/research-audit.txt)，作业同时保存实际 `pip freeze`。

安装失败后先保留日志；符合 [恢复条件](setup-recovery.md) 时可以用 `setup-finish` 继续，不能删除环境后盲目重试。

所有训练、张量测试、数据下载和转换都通过 `submit.sh` 提交。`job.sh` 在缺少 `SLURM_JOB_ID` 时拒绝执行。提交器归档的是 **HEAD 提交**，不是未提交的工作区，因此修改后需要先 commit 再提交实验。

## 2. 数据与权重

从公开数据重建的步骤见 [数据准备](data-preparation.md)。最终目录直接包含 2016–2023 八个年份，共 999 个 HDF5 事件；归一化统计只使用 2016–2020。训练固定使用这些年份，2021 用于验证，2022/2023 是保留测试年份。

已有模型的迁移见 [权重与结果打包](artifacts.md)。权重不放入 Git；需要检查摘要并携带对应结果，不能只把旧服务器绝对路径抄进配置。历史 checkpoint 内的初始化路径是来源标识，迁移文件时不能随意改写该标识。

## 3. 重新训练公共基础 B0/B1/B2/B3/B5

环境与数据准备成功后，以下命令提交一个完整 B3（T1 FireDrop+BlockDrop）训练作业：

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id B3 --upstream-root "$WILDFIRE_UPSTREAM" \
  --data-root "$WILDFIRE_DATA" --stats-path "$WILDFIRE_STATS" \
  --run-root "$WILDFIRE_ROOT/runs/B3"
```

将 `B3` 和输出目录同时改成 `B0`、`B1`、`B2` 或 `B5` 可运行对应基础模型。B0 是 T1 clean，B2 是 T1 FireDrop，B1 是 T5 clean，B5 是 T5 FireDrop+BlockDrop。每项都是 seed 0、3,000 个 optimizer steps；`from_scratch` 指没有从已训练任务模型继续，并不表示禁用 ImageNet 编码器初始化。

成功后按照 [基础模型导出](baselines.md) 生成经过检查的 checkpoint 和完成记录。不能手工把未完成的训练标记为完成。

基础训练和后续 continuation 是不同阶段。T1 continuation 从 B3 开始；T5 continuation 从 B5 开始。配置中的两个 checkpoint 必须是正确的 Lightning 基础模型文件，不能用 continuation 的包装 checkpoint 冒充。

## 4. Cross-history 方法

设置好 B3/B5 权重后，先跑一个 T1 smoke；它检查计算链路，不提供论文效果证据：

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method cosine_erm --seed 0 --batch-size 16 --workers 3 \
  --smoke --output "$WILDFIRE_ROOT/runs/smoke-t1-cosine"
```

完整运行移除 `--smoke`，改用新输出目录，保留 `--steps 3000`。下面是 seed 0、T1 的 X22 及其最近对照：

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method control --seed 0 --steps 3000 --batch-size 64 \
  --workers 3 --output "$WILDFIRE_ROOT/runs/t1-s0-control"
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method cosine_erm --seed 0 --steps 3000 --batch-size 64 \
  --workers 3 --output "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm"
```

不同实验批次的 physical batch 可能不同；effective batch 64 并不保证 BatchNorm 行为相同。比较时使用原结果记录的 physical batch，不能把 16 和 64 当作相同复现设置。T5 必须使用 `--history 5`，它同时更换时间长度、特征集合和模型，不能把 T1/T5 差异全部归因于时间长度。

先在 2021 完成选择并固定模型，再评估保留年份：

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --history 1 --method cosine_erm --seed 0 --batch-size 64 --workers 3 \
  --evaluate-only "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm/checkpoint.pt" \
  --year 2022 --output "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm-2022"
```

各 X/D 方法、最近对照、评估器和组合命令见 [方法配方](method-recipes.md)，教师重建及 RF/TD 命令见 [教师与蒸馏](teachers.md)。

## 5. 如何理解复现结果

M00 为完整输入；主指标是 M01/M06/M07 的 AP 均值，block 指 M06/M07 的均值。AP 差值是绝对值，例如 `+0.005` 是增加 0.5 个百分点。先对最近对照比较，再讨论相对公共基础模型的总收益。

保留两套不同的 September three-directions 实现：`cross_history.run_three_directions` 是 RF 系列；`three_directions.run` 是 TD 系列。它们的初始化、batch 和目标不同，不可混用 checkpoint、命令或结果表。

已核对全部 21 份原始 retained 结果：2021/2022/2023 样本数均为 3,181/2,856/2,102；旧量化记录中的 2,312 是文档笔误，84 个 AP 数值不变，见 [样本量核对](evaluation-population.md)。GPU smoke、单 seed screen、三 seed 确认和 heldout 结果分别记录，不能互相替代。

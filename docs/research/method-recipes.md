# 正向信号的具体方法入口

先完成 [环境和数据](reproduce.md)、[B3/B5 基础模型](baselines.md)。这里的命令只提交作业，不等待完成；有依赖时先检查上一步退出状态。每次使用新输出目录。完整结果、对照和哪些设置真正跑过见 [正向信号清单](positive-signals.md)，不能把可运行的设置写成已有实验结果。

## X 系列：同一入口，保留各自机制

以下函数在登录 shell 中仅提交 Slurm 作业。`history=1` 和 `seed=0` 是首次复现的选择；改为 T5 或其他 seed 前先检查该方向是否有对应历史证据。`physical_batch=64` 对应后期主干的记录；较早实验应采用其原始结果里的 batch，effective batch 都是 64。

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
history=1
seed=0
physical_batch=64
ch() {
  local method=$1
  shift
  bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
    --history "$history" --seed "$seed" --steps 3000 \
    --batch-size "$physical_batch" --workers 3 --method "$method" \
    --output "$WILDFIRE_ROOT/runs/t${history}-s${seed}-${method}" "$@"
}
```

按需选择一项及其最近对照，不必把下表全部训练：

| 方向 | 提交命令 | 需要同时保留的最近对照 |
|---|---|---|
| 公共 continuation | `ch control` | 冻结 B3/B5，另与新训练 control 区分 |
| X1 | `ch context` | control |
| X1 adapter | `ch context_adapter` | 冻结基础模型和 control |
| X3 | `ch transition` | control |
| X4 | `ch risk` | control |
| X6 | `ch balanced_corruption` | control |
| GLOBAL-D1 | `ch global_consistency` | control |
| X8 | `ch impact_consistency` | global_consistency、control |
| X10 | `ch dynamic_inpaint` | control |
| X11 对 ERM 的局部信号 | `ch dynamic_inpaint_reconstruct` | dynamic_inpaint、control |
| X12 | `ch fire_specialist_impact` | fire_specialist、control |
| X14 | `ch block_specialist` | control |
| X16 | `ch block_specialist_dynamic` | block_specialist、control |
| X18 | `ch block_specialist_context` | block_specialist、control |
| X19 | `ch block_specialist_severity_adapter` | block_specialist、control |
| X22 | `ch cosine_erm` | control |
| X23 | `ch block_specialist_impact` | block_specialist、control |
| X25 | `ch block_specialist_reliability_prompt` | block_specialist、control |
| X26 | `ch cosine_fire_impact` | cosine_fire_global、cosine_erm |
| X27 | `ch cosine_block_hard` | cosine_block_specialist、cosine_erm |
| X29 | `ch cosine_block_memory` | cosine_block_specialist、cosine_erm |
| X1+X3 | `ch context_transition` | context、transition、control |
| X2 block 局部信号 | `ch distill` | control；不要替换成修订的 distill_block |

对照名称同样可传给 `ch`，例如 `ch cosine_fire_global`。这些是原实现的开关，不是用某个“类似模块”替代被恢复的旧方法。

## D 系列：历史独立实现

D 系列和 X 系列不是相同训练配方。D 的基础完成记录由 [基础模型导出](baselines.md) 生成；不能只传 checkpoint。以下辅助函数显式传入新环境路径：

```bash
dtrain() {
  local label=$1 module=$2 batch=$3
  shift 3
  bash scripts/research/submit.sh gpu python -m "reproductions.wsts_fast_track.$module" \
    --b3-record "$WILDFIRE_ROOT/checkpoints/B3-completed.json" \
    --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA" \
    --stats-path "$WILDFIRE_STATS" --batch-size "$batch" --num-workers 8 \
    --device cuda --output-path "$WILDFIRE_ROOT/runs/$label/model.ckpt" "$@"
}
```

| 方向 | 提交命令 |
|---|---|
| D1 ERM 对照 | `dtrain D1-ERM train_predictive_consistency 64 --lambda-consistency 0.0` |
| D1 KL | `dtrain D1-KL train_predictive_consistency 64 --lambda-consistency 0.1` |
| D2 STD 对照 | `dtrain D2-STD train_reliability_normalized 64 --variant standard` |
| D2 RNC | `dtrain D2-RNC train_reliability_normalized 64 --variant rnc` |
| D4 token | `dtrain D4 train_reliability_normalized 64 --variant token` |
| D5 CIWC | `dtrain D5 train_counterfactual_impact_consistency 32` |
| D6 CIWC+rank | `dtrain D6 train_counterfactual_rank_consistency 32` |
| D7 adapter | `dtrain D7 train_counterfactual_reliability_adapter 32 --adapter-scope all` |
| D8 factorized adapter | `dtrain D8 train_counterfactual_reliability_adapter 32 --adapter-scope block` |
| D10 prompt pyramid | `dtrain D10 train_reliability_prompt_pyramid 64 --variant prompt-pyramid` |
| D11 complete prompts | `dtrain D11 train_reliability_prompt_pyramid 64 --variant complete-prompt-pyramid` |
| D12 SARP | `dtrain D12 train_severity_adaptive_reliability_prompting 64` |

这些训练器固定 seed 0、3,000 steps。最近对照要沿用对应实验的 batch 和数据配对设置：D5/D6 的 paired ERM 对照应使用 `train_predictive_consistency --lambda-consistency 0.0 --batch-size 32`，不要直接拿上表 batch 64 的 D1-ERM 替代。其他匹配对照和门槛以原 [量化记录](../experiments/quantitative_reliability_ledger.md) 与 [弃用记录](../experiments/rejected_experiments.md) 为准。

D13 使用 T5 基础模型，standard 和 sarp 两个版本分别提交：

```bash
for variant in standard sarp; do
  bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.train_temporal_reliability_prompting \
    --b5-record "$WILDFIRE_ROOT/checkpoints/B5-completed.json" \
    --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA" \
    --stats-path "$WILDFIRE_STATS" --variant "$variant" \
    --batch-size 64 --num-workers 8 --device cuda \
    --output-path "$WILDFIRE_ROOT/runs/D13-$variant/model.ckpt"
done
```

训练结束后，用对应 evaluator 评估 2021；例如 D4：

```bash
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.evaluate_reliability_normalized \
  --checkpoint "$WILDFIRE_ROOT/runs/D4/model.ckpt" \
  --output-root "$WILDFIRE_ROOT/runs/D4/eval-2021" \
  --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA" \
  --stats-path "$WILDFIRE_STATS" --year 2021 --device cuda
```

| 模型 | evaluator 后缀（前缀都是 `reproductions.wsts_fast_track.`） |
|---|---|
| D1 | `evaluate_predictive_consistency` |
| D2 / D4 | `evaluate_reliability_normalized` |
| D5 | `evaluate_counterfactual_impact_consistency` |
| D6 | `evaluate_counterfactual_rank_consistency` |
| D7 / D8 | `evaluate_counterfactual_reliability_adapter` |
| D10 / D11 | `evaluate_reliability_prompt_pyramid` |
| D12 | `evaluate_severity_adaptive_reliability_prompting` |
| D13 | `evaluate_temporal_reliability_prompting` |

评估保留年份时同时更改 `--year 2022` 或 `2023`、输出目录，并加 `--heldout-authorized`。不要为未通过 screen 的方向新增 heldout 结论。

## 组合、教师和两套 September 实验

教师训练及 RF/TD 命令见 [教师与蒸馏](teachers.md)。X8+X17 与 X22+X17 使用同一 complete-route 汇总器，区别是 FireDrop 分支来自 impact_consistency 或 cosine_erm。先训练并评估匹配的 control、fire、固定 25% 和 50% BlockDrop 专家，再汇总。例如 X22+X17 的 seed 0、T1、2021：

```bash
bash scripts/research/submit.sh cpu python -m reproductions.cross_history.compose_complete_routes \
  --control "$WILDFIRE_ROOT/runs/t1-s0-control/summary.json" \
  --fire "$WILDFIRE_ROOT/runs/t1-s0-cosine_erm/summary.json" \
  --mild "$WILDFIRE_ROOT/runs/t1-s0-block025/summary.json" \
  --severe "$WILDFIRE_ROOT/runs/t1-s0-block050/summary.json" \
  --output "$WILDFIRE_ROOT/runs/t1-s0-X22-X17.json"
```

X8+X10 使用 `compose_routes --control ... --fire ... --spatial ...`，其中 fire 是 impact_consistency，spatial 是 dynamic_inpaint。汇总器读取已有场景结果，不会训练新网络；输出的路由结果适用于约定的可观测缺失场景，不是额外训练出来的单模型。

自然 VIIRS 和 target-QA 两个诊断性正向信号见 [诊断复现](diagnostics-reproduction.md)。它们依赖历史 P00，不作为 corrected baseline 的科学贡献。架构迁移必须使用对应 pretrained 资产和独立 bootstrap，不能把 Res18 的 B3/B5 权重载入 Swin/SegFormer。

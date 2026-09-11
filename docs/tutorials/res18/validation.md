# 教程验证记录（2026-09-11）

本记录区分新做的工程验证和文档引用的历史科研结果。机器可读摘要见 [validation.json](validation.json)。

| 检查 | 实际结果 |
| --- | --- |
| 教程 Bash 代码块与 shell 脚本 | 31 个 Bash 代码块及全部 `.sh` 通过 `bash -n` |
| Python 源文件 | 全部通过 AST 解析 |
| 结果汇总单元测试 | `python docs/tutorials/res18/test_results.py`：6 passed；覆盖表格解析、非有限最终结果、缺/重复 fold、混合模式与标准差 |
| 首次 Slurm GPU smoke | job `21746988`，`COMPLETED 0:0`，32 秒 |
| 加入已提交项目源码快照后的最终 GPU smoke | job `21747240`，`COMPLETED 0:0`，36 秒；Fold 2、1 个验证批次、1 步训练，batch 4 |
| 最终 GPU 峰值分配 | 308364288 bytes；10GB H100 MIG 分片 |
| CPU 数据转换往返 | job `21747243`，`COMPLETED 0:0`；小时值 14/23 保留，NaN 火点置零，缺 CRS 保留 NaN 经纬度，拒绝覆盖文件 |
| 现有 B0 与独立标签验证测试 | 同一 CPU job：`tests/test_wsts_fast_track_corrected_baselines.py`、`tests/test_verify_repair.py`，23 passed |
| Slurm 资源请求校验 | CPU 转换 `--tmp=200G`、GPU 正式训练 20GB MIG 请求被 `sbatch --test-only` 接受；没有启动这两个完整任务 |

最终 GPU smoke 使用现有的已验证训练环境和原始四年 HDF5，只在独立的验证目录创建输出。打印的划分为 2018/2020 train、2019 val、2021 test；**没有执行测试集**。一次验证批次出现无正样本提示，以及一步训练没有验证最佳 checkpoint 的提示，均未被解释为科研结果；smoke 成功只表示这个工程路径可执行。

本次没有重新下载完整 ZIP、从空环境完成安装、转换全部 999 个事件、训练 3000/10000 步、评价所有 12 个权重或验证这些新包装脚本的所有完整分支。新环境安装输入来自既有 Nibi 配置；数据准备流程来自既有脚本。学生仍须按教程每一步的成功标志验收，不能将上述 smoke 推广成全流程已经重跑。

原始日志、实际命令、Slurm 记录、环境、上游补丁以及执行时的教程脚本快照归档于：

```text
/project/6085198/kulbear/wildfire/runs/res18-tutorial-validation-20260911/
```

该目录的 `sha256.json` 记录归档文件校验和。完整训练及发布权重的历史分数另见 [官方复现记录](../../experiments/res18_unet_t1_reproduction.md)，本次 smoke 不新增 AP 复现结论。

配套 `convert-wstsplus-added.py`、`assemble-wstsplus.py` 原样来自项目先前在 `/project/6085198/kulbear/wildfire/cache/` 使用的数据转换脚本。本次将它们纳入 Git，解决新学生无法从仓库独立取得这些步骤的问题。其他 wrapper 只参数化路径、Slurm 环境、调用顺序和结果记录。

# 官方 codebase 独立教程检查记录

日期：2026-09-11。对应 `res18-baseline-slurm.md` 的独立版本，不沿用上一版项目 wrapper 的验证结论。

## 独立性

从 Markdown 的 `cat > ... <<'EOF'` 代码块实际提取出六个文件：requirements.txt、weights.tsv、weights.py、evaluate_weight.py、results.py、job.sh。

- 脚本唯一的 `git clone` 地址是 `https://github.com/slahrichi/WildfireSpreadTS.git`。
- 提取后的文件没有 `wildfire-research-plan` 路径，也不导入 `reproductions` 或 `wildfire_phase0`。
- 训练直接调用作者 `src/train.py`；权重评价使用作者的 CLI、模型和 DataModule。
- 十二个固定权重的文件名、SHA-256 和大小逐项核对一致。

## 本次实际检查

| 检查 | 结果 |
| --- | --- |
| Markdown 中 23 个 Bash 块及提取后的 job.sh | `bash -n` 通过 |
| 提取后的三个 Python 文件 | AST 解析通过 |
| 结果记录器 4 种情况 | 完整合法结果通过；最终 NaN、缺指标、缺完成标志均拒绝 |
| 汇总器 4 种情况 | 完整十二 fold 通过并校验均值；缺 fold、重复 fold、混合训练/权重模式均拒绝 |
| 运行提取出来的独立 job.sh | Slurm job `21748673`：`COMPLETED 0:0`，36 秒 |
| 实际执行内容 | 官方 Fold 2，一个验证 batch、一步训练，batch 4，不启用测试 |

原始日志、实际命令、环境、上游补丁及六个提取文件保存在：

```text
/project/6085198/kulbear/wildfire/runs/official-standalone-tutorial-validation-20260911/
```

`sha256.json` 记录归档文件校验和。这个归档只是教师的验证证据，不是学生运行教程的输入。

## 尚未验证的范围

上述 smoke 在当前 Nibi 上复用已有 Python 环境和原始四年 HDF5，通过独立目录引用它们；这是节省重复下载与转换的验证安排，学生教程仍提供从零安装和转换步骤。

本次没有登录学生的新服务器，没有重新安装空环境、下载完整 ZIP、转换全部数据、训练 10000 步或运行完整十二 fold。不能将本次 smoke 写成“新服务器全流程已验证”或新的 AP 复现结果。新服务器按用户说明具有相同类型的软件栈；仍须核对其账户、GPU 资源、module、wheelhouse、临时存储和计算节点网络权限。

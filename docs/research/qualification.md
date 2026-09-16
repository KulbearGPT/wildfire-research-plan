# 交付资格验证记录

验证对象是一个全新虚拟环境、独立的官方源码 checkout、每作业 Git 归档，以及迁移后的数据和权重路径。运行在现有集群的计算节点上，**没有声称已在另一套物理集群执行过**。新集群通过 site 配置选择账户、GPU、Python/module 和存储路径。

尚在进行中；本页不代表全部方法已通过。所有 tensor/model、依赖安装、下载和大文件校验均在 Slurm。登录节点只做源码/小型元数据/Git/提交检查。

## 已核实的环境与 CPU 证据

| 作业 | 源提交 | 节点 | 结果与适用范围 |
|---|---|---|---|
| 22101442 | 98d12f0 | c142 | 66 tests pass；旧环境，只作早期回归 |
| 22101813 | 58cf86a | c56 | 106 tests pass，实际官方 T5 类及驱动导入通过；旧环境 |
| 22102010 | 7416004 | c376 | 133 tests pass，实际官方 T5 类及驱动导入通过；旧环境 |
| 22101774 | 454f192 | c14 | 数据改放独立目录，B3/B5 与两套 seed 0 教师权重迁移；教师逐文件摘要校验 |
| 22102009 | 7416004 | c350 | P00、attention、48 个 QA map 迁移与摘要记录 |
| 22101441 | 98d12f0 | c83 | 安装失败：官方文件末尾缺换行导致旧补丁不适用；另发现 module 重设 pip wheelhouse |
| 22101770 | 454f192 | c56 | 全新公开 PyPI 训练环境已安装；在同一个旧补丁处停止 |
| 22102098 | c617100 | c3 | 从上述公共环境继续，精确修正官方补丁，训练/审计 `pip check` 均通过，setup 完成 |
| 22102101 | c617100 | c38 | 从公开源下载 Swin 与 MiT-B2 三个资产，全部匹配固定 SHA256 |
| 22102100 | c617100 | c3 | 新环境 134 tests pass、2 fail；失败是旧测试断言原服务器路径，已改为显式临时目录，由 22102394 重跑通过 |
| 22102394 | 19132b0 | c333 | 新训练环境 143 tests pass，实际官方 T5 类和保留驱动导入通过 |
| 22102400 | 19132b0 | c332 | 新审计环境 117 tests pass，覆盖修复/验证、schema、inventory、target gate 和 split |
| 22102613 | eb702c2 | c422 | 新训练环境中，官方原始年份 converter CLI 与新增年份 converter 均通过 23 波段 GeoTIFF → HDF5 合成样例检查；覆盖数值、日期、坐标及 active-fire 转换语义 |

两次安装失败的日志保留，没有把失败作业写成成功。公共环境来自 `22101770` 创建的空目录，没有复用旧训练 venv；`22102098` 重用的是这次新安装的公共环境。模块配置之后重新隔离 pip 变量，安装记录不依赖本站 wheelhouse。实际版本记录为 [训练 freeze](environment/train-pip-freeze.txt)、[审计 freeze](environment/audit-pip-freeze.txt)，官方改动见 [upstream.diff](environment/upstream.diff)。这些是观察记录，安装输入仍是 `environments/research-*.txt`。

完整数据来自已审计的既有 999 事件数据，迁移时使用同文件系统 hard link；因此这里验证新路径和新软件环境，没有重复完整公开下载/转换。权重实际复制并校验。公开数据的完整重建命令、逐阶段检查与原始校验值见 [数据准备](data-preparation.md)。

## GPU 验证的边界

Cross-history 和两套 September smoke 使用实数据、一轮短训练或校准、checkpoint 保存/重载，以及有限样本评估。历史 D 系列和基础模型使用 [qualification driver](../../scripts/research/README-qualification.md)；短预算输出明确标记为 qualification，不能作为 3,000/10,000-step 结果、教师或完整基线完成记录。

D 系列正式 evaluator 要求固定训练预算。资格验证只在内存副本中替换其预算/status 字段以调用原模型构造器，其余方法元数据校验和严格 state_dict 加载保留；磁盘上的短预算权重仍被正式校验器拒绝。它证明加载/计算链路，不证明方法效果或完整训练的数值一致性。

GPU 作业 `22102452`（`e13b9fb5db55`，`g36`）已完成两项实际 24 样本诊断；[重放结果](environment/diagnostics-replay.json)与历史记录的比较方向相同，数值最大绝对差约 `4.3e-6`。这验证迁移权重后的推理链路，不代表重新训练。

首批 6 个训练/校准 GPU 作业已成功退出，合计 19 个短预算 case 的报告见[原始记录](environment/gpu-qualification-first-batch.json)：

| 作业 | 节点 | 已通过的范围 |
|---|---|---|
| 22102102 / 22102104 | g32 / g35 | T1 / T5：control、cosine_erm、context、transition；每项实际 1 step，strict reload，预测重载差为 0 |
| 22102396 | g36 | B5：实际 Lightning 2 steps、最佳权重选择、strict reload、M00/M06 各 2 样本评估 |
| 22102397 | g32 | RNC：实际 1 step、正式 validator 拒绝短权重、strict reload、M00/M06 各 2 样本评估 |
| 22102787 | g36 | RF T1：control、mixed、typed、distill、BN batch_stats；重载差为 0 |
| 22102798 | g35 | Routed T1：student_control、student_distill、merge、bn_shared；重载 state 严格相等后评估 |

这些短预算样本不用于效果判断，AP 为 0 的小样本也不能解释为方法失效。后续仍需检查全部保留训练执行族、完整数据审计、独立最终审查及主工作区交付；不能据此页提前声明目标完成。

转换作业的[原始小型报告](environment/conversion-qualification.json)保留逐年份检查结果。它使用八个合成事件，未替代全量公开下载、999 事件转换或修复阶段验证。

## 后续实际验证

第二批 9 个 GPU 作业（`22102808/09/10/82/84/88/93/94/95`，源 `a7e8e0a`）全部正常退出，新增 50 个通过的 case，见[逐项报告](environment/gpu-qualification-second-batch.json)。包括其余列入正向清单的 cross-history 方法在 T1/T5 的短训练及重载、RF 与 routed 的 T5 分支，以及 B1、token 和 CRA。仍需补齐架构、剩余 D 系列和 attention 训练等验证，不能把这个批次当成整体完成。

CPU 作业 `22102402`（源 `19132b0`，`c333`）完成全量 999 事件审计：[报告](environment/data-audit-phase0_report.md)显示无无效文件错误，门控结论为 `continue_controlled`。原始数据缺少 observation/availability 时间、QA、coverage 和 target validity 等字段，因此支持既定受控缺失实验，不支持自然缺失或 operational 声明。报告的 target days 是原始事件的逐日统计，与模型按历史窗口构造的评估样本数不是同一口径。

历史入口的下一批 10 个 GPU case 也已通过，见[逐项报告](environment/gpu-qualification-legacy-batch.json)：D1-KL / paired0、原始 D12-SARP、B2、B3、legacy-P00、CIWC、rank、FFCA、prompt-pyramid。B2/B3/P00 使用实际 Lightning 2 steps，其余 1 step；报告均保留 qualification 标记。另有 CPU 作业 `22103003`（`c86`）在迁移后的历史 summary 上执行 T1/T5 的 complete-route 与 severity-route CLI，见[输出](environment/composition-qualification.json)；它验证汇总命令，不是重新评估模型。

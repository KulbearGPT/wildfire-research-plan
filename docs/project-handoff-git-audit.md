# 教学交接 Git 核对

核对日期：2026-09-11，以下为整理教学文档之前的本地快照。

用户要求确认从项目开始至今的改动均已 commit。检查全部六个注册 worktree 的 tracked/untracked 状态：均为干净，既有改动已保存在各自分支，不需要制造空提交。检查范围为该本地 Git 仓库；未查询远端集群工作目录，也未推送远端。

| 分支 | 整理前 HEAD | 状态 |
| --- | --- | --- |
| `main` | `8c9c072449d9` | 干净，无未提交/未跟踪文件 |
| `codex/belief-attention` | `f45193465652` | 干净，无未提交/未跟踪文件 |
| `codex/belief-filter` | `fac140f05e65` | 干净，无未提交/未跟踪文件 |
| `codex/belief-reconstruction` | `12cdc2dda135` | 干净，无未提交/未跟踪文件 |
| `codex/belief-submit` | `86a8abf953eb` | 干净，无未提交/未跟踪文件 |
| `research/t1-t5-innovations` | `6111d9e55587` | 干净，无未提交/未跟踪文件 |

历史从 `3f22e30` 初始提交开始；早期教学页面、数据审计、基线复现、方法与实验记录均有提交历史。`archive/pre-t1-cleanup-2026-09-04` 在 `4b843ad` 保留清理前快照。其他 belief worktree 的提交保留为历史，不纳入当前教学实验路线。

“已提交”不等于“全部合并 main”。本次不合并历史实验分支，不改写已有提交。最新研究源仍在 `research/t1-t5-innovations`，主分支与该研究分支均补充同一教学 roadmap 和本核对说明，并在 README 提供入口。用各分支最新 `git log -1` 查看本次文档提交。

核对命令：

```bash
git worktree list --porcelain
git log --all --oneline --decorate
# 对每个 worktree 分别执行；包含普通未跟踪文件：
git status --porcelain --untracked-files=all
git diff --check
```

Git 忽略的数据、虚拟环境、缓存、检查点、日志和本机配置不属于待提交源码。大型实验产物按现有 manifests、实验台账和集群归档路径管理；本次没有把这些文件加入 Git，也没有声称重新校验其远端完整性。

本次变更仅为文档：保存早期 roadmap，新增当前教学路线和 Git 核对说明，更新 README 入口。验收检查文档路径、关键数值与原台账一致性、差异格式，以及提交后所有 worktree 清洁状态；没有重新训练或重新验证历史科研分数。

# 配套教学网站维护

入口是仓库根目录的 [index.html](../index.html)，配套 [研究基础](../related-work/index.html) 与 [基线复现](../baseline-reproduction/index.html) 两份课程。页面可直接从本地打开；静态服务器也可以原样托管仓库根目录。

## 内容来源

- 网站首页：`index.html`；公共样式：`assets/course.css`。
- 两份 HTML 课程保留已有图示、导航、搜索与打印功能。文献部分保留原有的文献核对日期，项目状态单独更新。
- 操作指南以 `docs/` 下的 Markdown 为源，`guides/` 下的 HTML 为生成物。不要直接编辑生成页。

修改入门、Roadmap、两份 Slurm 教程或 `docs/research/*.md` 后，在仓库目录执行：

```bash
python scripts/build-course-guides.py
```

需要 Pandoc；本次生成使用 Pandoc 2.18。生成过程不导入模型、不访问数据或网络，只转换文档。脚本把文档之间的链接改为网页路径，代码块保持原样；网页和生成脚本一起提交。页面不需要在线 Markdown 渲染服务。

## 发布边界

此前课件内写入的 GitHub Pages 地址检查时返回 404，且仓库没有 Pages workflow。这里使用相对链接，不假定公网部署已经可用。

将完整静态目录部署到可用的静态站点即可，必须同时保留 `related-work/assets/`、`guides/`、`assets/` 和页面所链接的源文件。若使用 GitHub Pages，可在有仓库管理权限的会话中配置从 `main` 根目录发布；是否公开仓库或站点由维护者决定。仅本地 commit 不等于网站已发布。

## 更新时核对

- 官方单 fold、项目 corrected baseline 和全量发布权重评估保持不同协议。
- 新的效果结论必须链接原始实验记录；运行资格验证不能当作效果复现。
- 首页、课程中的项目定位与最新交付清单一致，历史结果标明时期。
- 检查本地链接、图片路径、页内锚点与教程代码块；实际模型检查仍通过 Slurm。

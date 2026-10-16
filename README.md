# VISTA–WAM

该仓库把冻结的 LingBot-VA `posttrain-robotwin`、RMBench 仿真、外部视觉/事件记忆和可审计动作执行整理成一套可迁移代码。项目不新增 RMBench 参数训练；外部模块负责保存证据、形成目标条件或构建明确标注的解析前缀，WAM 负责正式后缀的动作生成。

迁移与验收按 [完整迁移文档](docs/MIGRATION.md) 执行。模块边界见 [架构说明](docs/ARCHITECTURE.md)，保留结果的口径见 [结果说明](docs/RESULTS.md)。

2026-10-16 已在原服务器用整理后的新目录完成一次独立迁移验收：仓库/权重检查通过，单元测试 5/5，通过冻结 WAM 生成并执行 16 条动作，随后提交 4 个真实反馈帧。详见 [迁移整理与验证报告](docs/迁移整理与验证报告_20261016.md)。

```bash
bash data_processing/setup_repositories.sh
python -m pip install huggingface_hub
python data_processing/download_models.py lingbot
bash data_processing/download_rmbench.sh
bash scripts/install_environments.sh
```

代码全部位于 `src/vista_wam`；第三方仓库、权重、数据和运行产物由 `.gitignore` 排除，适合直接初始化 Git 仓库。


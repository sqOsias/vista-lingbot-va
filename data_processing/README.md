# 数据与权重

Git 仓库不保存模型权重、RMBench assets、完整数据集或第三方源码。迁移后依次运行：

```bash
bash data_processing/setup_repositories.sh
python -m pip install huggingface_hub
python data_processing/download_models.py lingbot
bash data_processing/download_rmbench.sh
```

默认只下载运行小实验所需的 RMBench assets。需要完整公开数据集时运行
`WITH_FULL_DATASET=1 bash data_processing/download_rmbench.sh`。外部 VLM 是可选组件，运行
`python data_processing/download_models.py qwen` 下载 Qwen3-VL-8B-Instruct。

如果已有共享存储，不必复制：设置 `VISTA_WAM_LINGBOT_REPO`、`VISTA_WAM_RMBENCH_REPO`、`VISTA_WAM_LINGBOT_CHECKPOINT` 和可选的 `VISTA_WAM_QWEN_CHECKPOINT` 即可。


# 系统与硬件要求

- 已验证系统：Ubuntu 22.04、NVIDIA Driver 570.133、4×RTX 3090 24 GB、248 GB RAM。
- 推荐：2 张 24 GB GPU，分别运行 LingBot-VA 服务和 RMBench 仿真；64 GB RAM；至少 120 GB 可用磁盘。加入可选 Qwen3-VL、完整 RMBench 数据和调试产物时建议 180 GB。
- Python 固定为 3.10。模型环境使用 PyTorch 2.9.0/CUDA 12.6；仿真环境按 RMBench 官方安装脚本构建。两个环境分开，避免 Sapien/MPLib 与新版本 Torch 的依赖冲突。
- 单卡顺序运行可能可行，但本项目没有完成单卡显存与进程隔离验证，不能作为已验证配置。


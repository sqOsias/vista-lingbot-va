# VISTA–WAM 完整迁移说明

## 1. 迁移内容与边界

新仓库只保留当前研究主线：冻结 `robbyant/lingbot-va-posttrain-robotwin`，由外部视觉/事件记忆形成任务条件，在 RMBench 中经发布数值合同解码和 MPLib 执行 WAM 动作。代码已合并到 `src/vista_wam`，运行不再引用旧项目的 `phase1`—`phase10` 目录。

仓库内包含：

- 模型 HTTP 服务、客户端、动作编解码、仿真环境、外部证据库、视觉/RGB-D 接地、解析执行器、目标接口诊断和迁移验收；
- 固定版本的第三方仓库与模型下载脚本；
- 第 7—10 阶段支撑研究结论的汇总、CSV、审计和少量真实媒体；
- 两套隔离环境的安装要求和自动检查。

仓库不包含：23 GB LingBot 权重、17 GB 可选 Qwen3-VL、约 1.3 GB RMBench assets、约 24 GB 完整 RMBench 数据、第三方 Git 工作树、旧 FastWAM、纯 base、阶段逐帧图像/NPZ、重复日志和历史训练检查点。这些内容会显著增大 Git，并非复现当前主线所必需。

## 2. 在原服务器生成迁移源

当前交付目录为 `vista_wam_portable_20261016`。上传 GitHub 前在目录中执行：

```bash
cd /root/wyc/code/vista_wam_portable_20261016
git init
git add .
git status --short
git commit -m "Initial portable VISTA-WAM harness"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY>
git push -u origin main
```

`git status` 中不能出现 `third_party/`、`checkpoints/`、`datasets/`、`outputs/` 或大模型文件。项目自研代码未擅自附加开源许可证；公开仓库前由项目负责人选择许可证，第三方代码和模型继续遵守各自许可证。

## 3. 新服务器要求

已验证配置为 Ubuntu 22.04、Driver 570.133、4×RTX 3090 24 GB、248 GB RAM。建议至少 2×24 GB GPU、64 GB RAM 和 120 GB 空闲磁盘；下载可选 VLM 与完整数据集时建议 180 GB。Python 固定 3.10。模型和仿真分别使用 `vista-wam-model` 与 `vista-wam-sim` 环境。

先确认：

```bash
nvidia-smi
git --version
conda --version
df -h
```

## 4. 获取代码、第三方仓库和数据

```bash
git clone <YOUR_GITHUB_REPOSITORY> vista-wam
cd vista-wam
bash data_processing/setup_repositories.sh
```

脚本固定：

- LingBot-VA `a85cc1012db8142f8ced2723968f4370708751c0`；
- RMBench `6d36e28ccce86710627edac26b5899bd5b6994c2`；
- 可选 RoboTwin `2eeec322`，仅在 `WITH_ROBOTWIN=1` 时克隆。

下载当前主线权重和仿真 assets：

```bash
python -m pip install huggingface_hub
python data_processing/download_models.py lingbot
bash data_processing/download_rmbench.sh
```

迁移验收和 `put_back_block` 不需要公开演示数据。开展完整数据评测时再运行：

```bash
WITH_FULL_DATASET=1 bash data_processing/download_rmbench.sh
```

需要外部 Qwen3-VL 规划/记忆时：

```bash
python data_processing/download_models.py qwen
```

如果新服务器能访问旧服务器或共享盘，也可直接复制/挂载权重和 assets，然后通过环境变量指定位置：

```bash
export VISTA_WAM_LINGBOT_REPO=/shared/lingbot-va
export VISTA_WAM_RMBENCH_REPO=/shared/RMBench
export VISTA_WAM_LINGBOT_CHECKPOINT=/shared/checkpoints/lingbot-va-posttrain-robotwin
export VISTA_WAM_QWEN_CHECKPOINT=/shared/checkpoints/Qwen3-VL-8B-Instruct
```

不要用目录名判断权重身份。验收脚本会检查权重文件是否存在，正式实验还应把下载来源、文件清单和 SHA256 写入运行收据。

## 5. 安装环境

```bash
bash scripts/install_environments.sh
```

该脚本执行以下工作：

1. 建立 `vista-wam-model`，安装 PyTorch 2.9.0/CUDA 12.6、LingBot-VA 依赖、FlashAttention 和本项目；
2. 建立 `vista-wam-sim`，调用固定 RMBench 的官方 `_install.sh` 安装 Sapien/MPLib/Curobo，再安装本项目和测试依赖。

需要本地 Qwen3-VL 时使用 `WITH_VLM=1 bash scripts/install_environments.sh`，额外建立 `vista-wam-vlm`；默认不安装它，以节省磁盘和安装时间。

若官方 RMBench 安装脚本修改其虚拟环境内的 Sapien/MPLib 文件，这是上游安装步骤；不得修改固定的 RMBench Git 源码来迁就实验。

安装后先做不加载大模型的检查：

```bash
conda run -n vista-wam-sim python scripts/verify_install.py
conda run -n vista-wam-sim pytest -q
```

## 6. 启动冻结 WAM

选择一张空闲 GPU，例如物理 GPU 1：

```bash
conda activate vista-wam-model
export VISTA_WAM_MODEL_GPU=1
export VISTA_WAM_PORT=18774
bash scripts/start_model_server.sh
```

另开终端检查：

```bash
curl http://127.0.0.1:18774 | python -m json.tool
```

必须看到 `release=robbyant/lingbot-va-posttrain-robotwin`、`trainable_parameters=0`、`memorywam_enabled=false`。这证明加载的是冻结的后训练 RoboTwin 权重，不是纯 base，也没有启用模型内部 MemoryWAM。

## 7. 小规模迁移验收实验

模型服务运行后，在另一张空闲 GPU 上执行：

```bash
conda activate vista-wam-sim
export CUDA_VISIBLE_DEVICES=0
export VISTA_WAM_PORT=18774
bash scripts/run_smoke_test.sh
```

测试固定使用 `put_back_block`、seed `100065`、D435 三相机和 `front` 目标。外部解析执行器先构建“方块在中央且按钮按一次”的共同后缀初态；该前缀有独立收据，不计入 WAM 成功。随后服务完整重置，冻结 WAM 读取真实三相机观察与文字指令，生成第一块 `(16,2,16)` 原生动作；数值合同解码其中 16 条动作，RMBench 实际执行，每 4 步保存真实观察，最后把四帧和动作张量提交为历史。

迁移通过以 `outputs/migration_smoke_*/smoke_result.json` 中以下条件为准：

- 权重发布身份正确、训练参数为 0、内部 MemoryWAM 关闭；
- 三相机形状均为 `240×320×3`；
- 生成动作形状为 `(16,2,16)`，全部有限；
- 16 条解码命令均被环境恰好接受一次；
- 四个真实反馈帧成功提交给模型历史缓存；
- 无基础设施错误或未处理的规划异常。

这个测试证明代码、权重、相机、模型服务、动作解码、仿真执行和真实历史回灌在新服务器连通。16 步不足以把目标任务成功作为验收条件，因此不能把 smoke 的 `passed` 写成完整任务成功或记忆收益。需要验证已知阳性时，应另建输出目录，将动作预算扩展到正式协议，并继续区分外部前缀、WAM 后缀和完整任务。

## 8. 结果与故障定位

每次运行新建 `outputs/<run_id>`，不要覆盖旧目录。最小证据包括 `environment_receipt.json`、`auxiliary_prefix_receipt.json`、`continuation_adapter.json`、`generated_block.npz`、`actual_actions.jsonl`、三相机 PNG 和 `smoke_result.json`。

常见失败按以下顺序定位：

1. `verify_install.py` 失败：仓库 commit、assets 或权重路径错误；
2. 模型健康检查失败：模型环境、端口、GPU 显存或权重不正确；
3. 相机形状错误：RMBench assets/相机配置不匹配；
4. 动作 shape/有限值失败：LingBot 配置或权重版本不匹配；
5. 命令未被接受：MPLib、数值合同或任务驱动异常；
6. commit 失败：真实帧数量与首块 16 步历史对齐错误。

只有重复出现且定位后的基础设施问题才修改迁移代码；目标选择失败属于当前研究瓶颈，不能通过暗改任务成功判定、真值输入或动作覆盖来“修复”。

## 9. 旧项目与新仓库的关系

旧目录继续作为不可变历史证据，不从中直接运行新实验。新仓库中的 `results/` 是审阅子集，路径和口径见 `docs/RESULTS.md`。若论文审计需要逐动作 NPZ、所有图像或完整日志，再从旧项目按运行 ID 恢复；不要把这些大文件推入 GitHub。


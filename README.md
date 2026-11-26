# ML-VLA（机器学习课程大作业 · B 类：工程实践型）

本项目面向课程 **B 类（工程实践型）** 要求，基于公开视觉-语言-动作模型（VLA）与 BridgeData V2（WidowX）数据，完成环境搭建、数据核查与完整轨迹划分、轻量微调（动作头）以及 SimplerEnv WidowX 闭环评测对比。

---

## 1. 选型与固定版本清单

| 项目 | 选定内容 | 固定版本 / Commit / Revision |
|---|---|---|
| Base VLA 模型 | `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` | `8ddb8116d5bd2953931619cfa9d60f278aeebceb` |
| Base Checkpoint | `checkpoints/steps_20000_pytorch_model.pt` | SHA-256: `769d6c400d582a86ae8df8b0b445240ab679dbe77eeb72a4db71e43cd129c7c3` |
| 基础视觉语言模型 | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 训练 / 推理框架 | `starVLA` (`QwenGR00T`) | `3422b9f2387b6f682cf02802904a77b23ab13afd` |
| 仿真环境 | `SimplerEnv` + `ManiSkill2_real2sim` | `eddb5690749a1e34267807c0f7b5eede323918d7` / `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` |
| 微调数据集 | `IPEC-COMMUNITY/bridge_orig_lerobot`（LeRobot v2.0） | `0e9d76d07e9df3ea3eba257b2520d4913833fad2`（本次使用 `chunk-000` 前 1,000 条完整 WidowX 轨迹、`image_0` 单视角，共 36,027 帧） |
| 轻量微调模块 | 冻结 `qwen_vl_interface`，仅微调 Action Head（DiT-B） | 见后续 `train` 阶段 |
| B 类评测任务 | T1：`widowx_spoon_on_towel`；T2：`widowx_put_eggplant_in_basket` | Base / Final 各任务 10 次，共 40 次 |

- **WidowX 训练依据**：[StarVLA 官方模型卡](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/tree/8ddb8116d5bd2953931619cfa9d60f278aeebceb) 与 [StarVLA SimplerEnv 文档](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/examples/SimplerEnv/README.md) 明确记录该 20K checkpoint 使用 Bridge / WidowX 数据训练，并给出 SimplerEnv WidowX Visual Matching 评测接口。

---

## 2. 当前进度

- [x] **方案定型与版本锁定**：见 `ML-VLA-操作手册.md` 与 `configs/env_versions.json`
- [x] **双环境搭建与依赖导出**：`configs/sim-env.freeze.txt`（`mlvla_sim`）与 `configs/star-env.freeze.txt`（`mlvla_star`），已在 NVIDIA Tesla V100-PCIE-32GB 上通过 Vulkan 离屏渲染与 `strict=True` 权重加载验证（V100 不支持原生 bf16，推理自动使用 `use_bf16=False` / fp32）
- [x] **Base 开发闭环跑通（episode 0，仅用于开发检查）**：T1 与 T2 均完成 120 步闭环且成功（`success: true`），视频与动作曲线见 `results/dev_sanity/`
- [x] **数据下载、接口验证、样本展示与完整轨迹划分**：
  - 使用 `chunk-000`（1,000 条完整 WidowX 轨迹、36,027 帧、5 Hz、`image_0` 单视角），已通过 `starVLA/dataloader/lerobot_datasets.py` 接口验证
  - 真实样本展示：`data_prep/sample_ep0_frame0.png` 与 `data_prep/sample_inspection.json`
  - 完整轨迹级划分（`seed=42`，90%/10%）：`data_prep/splits/`（900 条训练 / 100 条验证，交集为 0）
- [ ] **动作头轻量微调与 Final Checkpoint 选择**：待执行（使用 `data_prep/splits/`，保存 `train_log.txt` 与 `checkpoint_link.txt`）
- [ ] **B 类 40 次正式闭环评测**：待执行（冻结正式初始状态清单，避开开发用 episode 0，导出 `episodes.csv`、`results.csv` 与代表性视频）
- [ ] **实验报告（≤10 页 PDF）**：待完成

---

## 3. Base 开发闭环验证（Development Only）

> 以下结果仅使用开发用 `episode 0` 验证模型加载、动作反归一化（`unnorm_key=oxe_bridge`）与 SimplerEnv 闭环渲染，**不计入正式 40 次评测**。

| 任务 | SimplerEnv 环境标识 | 开发 Episode | 状态 | 产物 |
|---|---|---|---|---|
| T1 | `PutSpoonOnTableClothInScene-v0` | `episode 0` | `success: true` | [视频](results/dev_sanity/T1_dev_ep0_success.mp4) / [动作图](results/dev_sanity/T1_dev_ep0_actions.png) / [摘要](results/dev_sanity/T1_dev_ep0_summary.json) |
| T2 | `PutEggplantInBasketScene-v0` | `episode 0` | `success: true` | [视频](results/dev_sanity/T2_dev_ep0_success.mp4) / [动作图](results/dev_sanity/T2_dev_ep0_actions.png) / [摘要](results/dev_sanity/T2_dev_ep0_summary.json) |

---

## 4. 数据集与真实样本核查

- **数据来源**：`IPEC-COMMUNITY/bridge_orig_lerobot`（BridgeData V2 的 LeRobot v2.0 导出，WidowX 机器人，5 Hz）
- **本次使用子集**：`chunk-000` 前 1,000 条完整轨迹（共 36,027 帧），仅下载模型所需的 `observation.images.image_0` 单视角视频
- **输入输出与归一化约定**：
  - 图像：`image_0` 单视角，模型输入 `224×224` RGB
  - 动作：7 维末端增量动作（`x, y, z, roll, pitch, yaw, gripper`），预测未来 16 步 `(16, 7)`；前 6 维连续动作采用 `q99` 归一化，夹爪维度采用 `binary`（`0.0 / 1.0`）约定，推理反归一化 key 为 `oxe_bridge`
  - 状态：`observation.state` 原始为 8 维，按官方配置设 `include_state: false`，不送入模型
- **真实样本（Episode 0）**：
  - 语言指令：`"put small spoon from basket to tray"`（轨迹长度 26 步）
  - 首帧图像：`data_prep/sample_ep0_frame0.png`（256×256 RGB）
  - 数值范围与详情：`data_prep/sample_inspection.json`
- **训练/验证划分**：按完整轨迹（`episode_index`）以 `seed=42` 按 90%/10% 划分，得到 **900 条训练轨迹**（`data_prep/splits/train_episodes.json`）与 **100 条验证轨迹**（`data_prep/splits/val_episodes.json`），交集为 0。

---

## 5. 仓库目录结构

```text
ML-VLA/
├── README.md                              # 项目说明、版本清单与当前进度
├── ML-VLA-操作手册.md                      # 完整选型与首次闭环操作手册
├── data_load_README.md                    # 数据集下载说明
├── download_bridge.py                     # 完整数据集下载脚本
├── requirements.txt                       # 下载脚本依赖
├── configs/
│   ├── env_versions.json                  # 固定代码/模型/数据版本与硬件验证记录
│   ├── sim-env.freeze.txt                 # SimplerEnv (mlvla_sim) 实际依赖快照
│   ├── star-env.freeze.txt                # StarVLA (mlvla_star) 实际依赖快照
│   ├── base_runtime_template.yaml         # Base 推理配置模板（不含本机绝对路径）
│   └── data_check_template.yaml           # 数据接口检查配置模板
├── data_prep/
│   ├── sample_ep0_frame0.png              # 真实样本首帧图像（Episode 0）
│   ├── sample_inspection.json             # 真实样本动作/指令/归一化检查报告
│   └── splits/
│       ├── split_summary.json             # 90/10 完整轨迹划分摘要
│       ├── train_episodes.json            # 900 条训练轨迹 ID
│       └── val_episodes.json              # 100 条验证轨迹 ID
├── results/
│   └── dev_sanity/                        # Base 开发闭环（episode 0）验证视频与日志
└── scripts/
    ├── eval_dev.py                        # 开发用单 episode 闭环验证脚本
    └── inspect_and_split_bridge.py        # 样本导出与完整轨迹级划分复现脚本
```

---

## 6. 复现说明

完整环境搭建、模型下载与 Base 闭环步骤请参见 [`ML-VLA-操作手册.md`](ML-VLA-操作手册.md)。数据样本检查与 90/10 完整轨迹划分可通过以下命令复现：

```bash
python scripts/inspect_and_split_bridge.py \
  --bridge-dir <PATH_TO_BRIDGE_ORIG_LEROBOT> \
  --output-dir <OUTPUT_DIR> \
  --seed 42 --val-ratio 0.1
```

> 后续微调训练命令（`train.sh`）、训练日志（`train_log.txt`）、Final checkpoint 下载链接（`checkpoint_link.txt`）及 40 次正式评测脚本与结果（`evaluate.py`、`results.csv`、`episodes.csv`）将在下一阶段补充提交。

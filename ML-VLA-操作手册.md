# ML-VLA：模型定版、下载与首次闭环操作手册

适用：课程 B 类；1 号负责方案确认，2 号负责下载与 Base 跑通，3 号负责数据，4 号负责微调，5 号负责正式评测，6 号协助数据验证和结果分析。

核对日期：2026-10-22。命令以 Linux / Bash、Conda 为例。本手册先确定软件、模型和数据，不根据 2 号电脑配置选型。

## 1. 本次直接确定的方案

| 项目 | 确定选择 |
|---|---|
| VLA 模型 | `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` |
| 具体 Base 权重 | `checkpoints/steps_20000_pytorch_model.pt` |
| 基础视觉语言模型 | `Qwen/Qwen3-VL-4B-Instruct` |
| 加载框架 | StarVLA，`QwenGR00T` |
| 代码版本 | 官方 `starVLA` 稳定分支，固定到下表 commit |
| 本次微调数据 | `IPEC-COMMUNITY/bridge_orig_lerobot`，LeRobot v2.0 |
| 微调方式 | 冻结 Qwen 视觉语言骨干，训练动作头；具体训练批量后续再定 |
| 仿真 | SimplerEnv 的 ManiSkill2 路线 |
| B 类任务 | T1：勺子放毛巾；T2：茄子放篮子 |

**这次将此前暂定的 `Qwen-GR00T-Bridge` 调整为上述 Qwen3VL 版本。** 两个名字代表不同权重，不能混用其基础 VLM、配置或下载命令。选择本版本的理由是：官方固定版本的 SimplerEnv 文档明确列出了这个 checkpoint，模型卡明确记录其使用 Bridge/WidowX 数据训练，并给出了两个目标任务的已有评测记录。

你已经准备的 Bridge 数据可以继续使用。权重名称中的 `RT-1` 是原作者训练数据的一部分，不代表本次 Bridge-only 微调还必须下载 RT-1/Fractal 数据，更不需要另外下载一个 RT-1 模型。

### 固定版本清单

| 对象 | revision / commit |
|---|---|
| StarVLA | `3422b9f2387b6f682cf02802904a77b23ab13afd` |
| VLA 权重仓库 | `8ddb8116d5bd2953931619cfa9d60f278aeebceb` |
| Qwen 基础 VLM | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Bridge 数据仓库 | `0e9d76d07e9df3ea3eba257b2520d4913833fad2` |
| SimplerEnv | `eddb5690749a1e34267807c0f7b5eede323918d7` |

**核实范围：** 已查阅官方文档、上述代码中的加载器和参数解析、公开模型配置、数据元信息及接口。没有在 2 号机器上安装环境、下载全部大文件、执行 GPU 加载或闭环实验。因此这里是有官方配套依据、固定版本的执行方案，不是“已在你们电脑上跑通”的声明。2 号需完成第 6～8 节的实际验收，不需要再等 1 号重新选模型。

## 2. 现在各人做什么

| 人员 | 现在的工作 | 交付标准 |
|---|---|---|
| 你 / 3 号 | 分享本手册；提供已有数据的目录、repo ID 和下载 revision | 2 号能找到数据；已有数据不重复搬运 |
| 1 号 | 将本方案登记到团队仓库；协助分析加载报错 | 记录模型、权重、代码版本和来源；不再泛泛比较模型 |
| 2 号 | 下载模型；安装两个环境；跑通两个开发 episode | 严格加载成功、输出合法动作、T1/T2 各完成一次闭环，保存日志和视频 |
| 4 号 | Base 跑通后，接入训练集划分并做动作头微调 | 确认只更新指定模块，保存 Final 和训练记录 |
| 5、6 号 | 冻结开发/正式评测划分，整理数据检查和结果格式 | 最终 Base/Final 各任务 10 次，共 40 次，保留每次结果 |

2 号若仅做 Base 仿真，不需要先下载完整 Bridge 数据集。模型和仿真可先跑；本地微调才需要训练数据。若数据已经在 2 号可访问的盘上，直接使用；若只在你的电脑上，可传输过去或让 2 号按固定 revision 下载，二选一。

## 3. 建立统一目录并固定代码

以下在 2 号执行机器终端运行。项目根目录可以改，但之后所有终端保持一致。

```bash
export VLA_WORK="$HOME/ml-vla-work"
mkdir -p "$VLA_WORK"/{src,models,data,runs,scripts}
cd "$VLA_WORK/src"
git clone --branch starVLA https://github.com/starVLA/starVLA.git
git -C starVLA checkout --detach 3422b9f2387b6f682cf02802904a77b23ab13afd
git clone https://github.com/simpler-env/SimplerEnv.git
git -C SimplerEnv checkout --detach eddb5690749a1e34267807c0f7b5eede323918d7
git -C SimplerEnv submodule update --init --recursive
```

新开终端都先设置：

```bash
export VLA_WORK="$HOME/ml-vla-work"
export STAR_REPO="$VLA_WORK/src/starVLA"
export SIMPLER_REPO="$VLA_WORK/src/SimplerEnv"
export MODEL_DIR="$VLA_WORK/models/Qwen3VL-GR00T-Bridge-RT-1"
export BASE_VLM_DIR="$VLA_WORK/models/Qwen3-VL-4B-Instruct"
export BASE_RUN="$VLA_WORK/runs/base"
export CKPT="$BASE_RUN/checkpoints/steps_20000_pytorch_model.pt"
export PYTHONPATH="$STAR_REPO${PYTHONPATH:+:$PYTHONPATH}"
```

`src` 放代码，`models` 放下载的原件，`data` 放数据，`runs` 放本次运行配置/日志。不要把模型和数据大文件提交到团队 GitHub 仓库。

注意：本手册路径是 `examples/SimplerEnv/`。不要混入开发分支的 `examples/simBenchmarks/SimplerEnv/` 命令，也不要直接运行带作者绝对路径的自动评测脚本。

## 4. 安装 StarVLA 软件环境

```bash
conda create -n mlvla_star python=3.10 -y
conda activate mlvla_star
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
cd "$STAR_REPO"
python -m pip install -r requirements.txt
python -m pip install 'huggingface_hub>=0.34,<1.0' msgpack 'opencv-python==4.10.0.84'
python -m pip install -e .
python -m pip check
python -m pip freeze > "$VLA_WORK/runs/star-env.freeze.txt"
```

这版 `QWen3.py` 实际使用 PyTorch 的 `sdpa` 注意力实现，故本路线不把编译 flash-attn 作为必做安装步骤。运行配置中也显式记为 `sdpa`。这是与固定代码要求相对应的参考软件组合。CUDA 驱动、编译工具链与显存适配按你的要求留给实际安装时处理；不能据此推定任意电脑都能运行。官方 requirements 仍有未锁版本的依赖，首次成功安装和运行后保留 `pip freeze`，作为小组实际环境记录。

## 5. 下载模型、基础 VLM，以及需要时下载数据

在 `mlvla_star` 环境运行。以下只下载所选 VLA 权重及配置，不下载它的全部历史评测视频。基础 VLM 单独下载，因为这版框架构建模型时还会从 `base_vlm` 读取骨干权重、处理器和 tokenizer，然后再装载完整 VLA 权重。

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['VLA_WORK'])
items = [
    ('StarVLA/Qwen3VL-GR00T-Bridge-RT-1',
     '8ddb8116d5bd2953931619cfa9d60f278aeebceb',
     root/'models/Qwen3VL-GR00T-Bridge-RT-1',
     ['checkpoints/steps_20000_pytorch_model.pt', 'config.yaml',
      'config.json', 'dataset_statistics.json', 'README.md']),
    ('Qwen/Qwen3-VL-4B-Instruct',
     'ebb281ec70b05090aa6165b016eac8ec08e71b17',
     root/'models/Qwen3-VL-4B-Instruct', None),
]
for repo, revision, folder, allow in items:
    snapshot_download(repo_id=repo, revision=revision,
                      local_dir=str(folder), allow_patterns=allow)
    (folder/'download_manifest.json').write_text(json.dumps(
        {'repo_id': repo, 'revision': revision}, indent=2))
print('Model downloads complete')
PY
```

核验 VLA 权重文件的 SHA-256；计算大文件摘要需要一些时间：

```bash
python - <<'PY'
import hashlib, os
from pathlib import Path
p = Path(os.environ['MODEL_DIR'])/'checkpoints/steps_20000_pytorch_model.pt'
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(8*1024*1024), b''):
        h.update(block)
expected = '769d6c400d582a86ae8df8b0b445240ab679dbe77eeb72a4db71e43cd129c7c3'
assert h.hexdigest() == expected, '权重校验失败，请检查下载'
print('Checkpoint SHA256 OK:', h.hexdigest())
PY
```

### 数据已经准备好的情况

直接将 `BRIDGE_DIR` 设为已有 LeRobot 数据根目录，例如：

```bash
export BRIDGE_DIR="$VLA_WORK/data/bridge_orig_lerobot"
```

路径必须包含 `meta/info.json`、`data/`、`videos/`，不是 ZIP 文件，也不是数据集上一级目录。把现有下载 manifest 的 revision 与第 1 节比对；不一致时先记录差异，不要无故删除或覆盖已加工数据。

### 只有需要下载时才运行

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
p = Path(os.environ['BRIDGE_DIR'])
repo = 'IPEC-COMMUNITY/bridge_orig_lerobot'
rev = '0e9d76d07e9df3ea3eba257b2520d4913833fad2'
snapshot_download(repo_id=repo, repo_type='dataset', revision=rev, local_dir=str(p))
(p/'download_manifest.json').write_text(json.dumps(
    {'repo_id': repo, 'repo_type': 'dataset', 'revision': rev}, indent=2))
PY
```

这是完整数据下载，不要求在跑 Base 前完成。下载中断后重跑同一段，已完成文件可复用。无需下载 Fractal 数据，因为本次微调确定使用 `data_mix: bridge`。

## 6. 创建可加载的 Base 运行目录

固定版加载器会从权重的上两级目录读取 **`config.yaml` 和 `dataset_statistics.json`**。只有一个 `.pt` 文件是不够的。创建运行副本，设置基础 VLM 本机路径，并将注意力配置记为源码实际使用的 sdpa，保留下载原件：

```bash
python - <<'PY'
import os, shutil
from pathlib import Path
from omegaconf import OmegaConf
src = Path(os.environ['MODEL_DIR'])
run = Path(os.environ['BASE_RUN'])
(run/'checkpoints').mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(src/'config.yaml')
cfg.framework.qwenvl.base_vlm = os.environ['BASE_VLM_DIR']
cfg.framework.qwenvl.attn_implementation = 'sdpa'
OmegaConf.save(cfg, run/'config.yaml')
shutil.copy2(src/'dataset_statistics.json', run/'dataset_statistics.json')
name = 'steps_20000_pytorch_model.pt'
dst = run/'checkpoints'/name
if not dst.exists():
    dst.symlink_to((src/'checkpoints'/name).resolve())
print('Runtime checkpoint:', dst)
PY
```

本版本读取 YAML，不能只修改 `config.json`。也不要照抄新模型卡上的 `--config_override` 参数：本手册固定的服务器版本没有该 CLI 参数，所以在上面直接生成本地 YAML。

基础 VLM 目录名称保留 `Qwen3-VL-4B-Instruct`，因为框架根据路径中的 `Qwen3-VL` 字符串选择 VLM 实现。

### 首次严格加载与动作检查

```bash
cd "$STAR_REPO"
python - <<'PY'
import os
import numpy as np
from PIL import Image
from deployment.model_server.policy_wrapper import PolicyServerWrapper
policy = PolicyServerWrapper(os.environ['CKPT'], device='cuda', use_bf16=True)
print('metadata:', policy.metadata)
result = policy.predict_action(
    examples=[{'image': [Image.new('RGB', (224, 224))],
               'lang': 'put spoon on towel'}],
    unnorm_key='oxe_bridge')
a = np.asarray(result['actions'])
assert a.shape == (1, 16, 7), a.shape
assert np.isfinite(a).all(), '动作含 NaN/Inf'
print('STRICT LOAD + ACTION SHAPE OK:', a.shape)
PY
```

底层 `from_pretrained` 使用 `strict=True` 装载权重。若发生 missing/unexpected keys 或 shape mismatch，记录完整报错并核对版本，不要改为 `strict=False` 把问题跳过去。

这个黑图测试只证明加载和动作接口可用，不能说明任务成功。动作由服务端反归一化，客户端不要再做一次。Bridge 的统计 key 固定为 `oxe_bridge`。

## 7. 安装 SimplerEnv 环境

保留 StarVLA 环境，在另一个 Conda 环境安装仿真：

```bash
conda create -n mlvla_sim python=3.10 -y
conda activate mlvla_sim
python -m pip install numpy==1.24.4
python -m pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim"
python -m pip install -e "$SIMPLER_REPO"
python -m pip install tyro matplotlib mediapy websockets msgpack transforms3d 'opencv-python==4.10.0.84' numpy==1.24.4
python -m pip check
cd "$STAR_REPO"
python examples/SimplerEnv/eval_files/test_your_simplerEnv.py
python -m pip freeze > "$VLA_WORK/runs/sim-env.freeze.txt"
git -C "$SIMPLER_REPO" submodule status > "$VLA_WORK/runs/sim-submodules.txt"
```

这里修正了官方说明里测试脚本的路径。我们使用现成 WidowX 仿真与自己的策略服务，不需要按 SimplerEnv 的 RT-1/Octo 推理章节另装这些模型。若报 Vulkan 或 SAPIEN 渲染错误，属于仿真环境安装问题，保留错误日志处理。

## 8. 跑通 T1 / T2 开发 episode

### 终端 A：启动模型服务

重新设置第 3 节变量，激活 StarVLA 环境：

```bash
conda activate mlvla_star
cd "$STAR_REPO"
set -o pipefail
python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 10093 --use_bf16 \
  2>&1 | tee "$BASE_RUN/server.log"
```

保持终端 A 运行。出现服务就绪信息后再开始终端 B。端口已被占用时先查明对应进程，不要直接运行官方自动脚本去杀占用端口的进程。

### 终端 B：生成一个小型开发评测入口

这个入口沿用官方 evaluator 和 ModelClient，仅显式设置本机地址并把每次结果保存为 JSON。固定版原入口没有传递 `--host` 给 ModelClient；这里避免依赖其 `0.0.0.0` 客户端默认值。它是开发检查脚本，不是已完成随机种子控制的正式评测器。

```bash
conda activate mlvla_sim
cat > "$VLA_WORK/scripts/eval_dev.py" <<'PY'
import json
from pathlib import Path
import numpy as np
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient
args = get_args()
model = ModelClient(policy_ckpt_path=args.ckpt_path,
                    policy_setup=args.policy_setup, host=args.host,
                    port=args.port, action_scale=args.action_scale)
results = maniskill2_evaluator(model, args)
assert len(results) == args.obj_episode_range[1] - args.obj_episode_range[0]
folder = Path(args.logging_dir)
folder.mkdir(parents=True, exist_ok=True)
record = {'task': args.env_name, 'checkpoint': args.ckpt_path,
          'episode_range': args.obj_episode_range,
          'success': [bool(x) for x in results],
          'success_rate': float(np.mean(results)),
          'purpose': 'development_only'}
(folder/'summary.json').write_text(json.dumps(record, indent=2))
print(record)
PY
```

运行两个任务各 1 次，开发 episode ID 固定用 `0`。每次执行创建新日志目录：

```bash
cd "$STAR_REPO"
export DISPLAY=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
set -o pipefail
DEV_RUN="$BASE_RUN/dev-$(date +%Y%m%d-%H%M%S)"
for TASK in T1 T2; do
  if [ "$TASK" = T1 ]; then
    ENV_NAME=PutSpoonOnTableClothInScene-v0
    ROBOT=widowx
    SCENE=bridge_table_1_v1
    OVERLAY=bridge_real_eval_1.png
    ROBOT_X=0.147
    ROBOT_Y=0.028
  else
    ENV_NAME=PutEggplantInBasketScene-v0
    ROBOT=widowx_sink_camera_setup
    SCENE=bridge_table_1_v2
    OVERLAY=bridge_sink.png
    ROBOT_X=0.127
    ROBOT_Y=0.06
  fi
  mkdir -p "$DEV_RUN/$TASK"
  python "$VLA_WORK/scripts/eval_dev.py" \
    --host 127.0.0.1 --port 10093 --ckpt-path "$CKPT" \
    --policy-setup widowx_bridge --robot "$ROBOT" \
    --env-name "$ENV_NAME" --scene-name "$SCENE" \
    --rgb-overlay-path "$SIMPLER_REPO/ManiSkill2_real2sim/data/real_inpainting/$OVERLAY" \
    --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
    --robot-init-x-range "$ROBOT_X" "$ROBOT_X" 1 \
    --robot-init-y-range "$ROBOT_Y" "$ROBOT_Y" 1 \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --obj-variation-mode episode --obj-episode-range 0 1 \
    --logging-dir "$DEV_RUN/$TASK" \
    2>&1 | tee "$DEV_RUN/$TASK/eval.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then break; fi
done
```

注意 T2 的 robot、scene 和背景图与 T1 不同，上述参数来自固定版本的官方 Bridge 脚本。不要只换任务名而保留 T1 全部配置。

**2 号完成标准：** 两任务均有完整 episode、`summary.json`、日志和视频；动作无维度/数值错误。成功率可以为 0，跑通不等于成功完成任务。两个开发 episode 不计入课程要求的正式 40 次。

## 9. 将现有 Bridge 数据接入训练代码

这一节在数据可访问后执行，不阻塞 Base 的闭环检查。

### 9.1 对齐目录别名与 modality

官方 `data_mix: bridge` 实际查找子目录 `bridge_orig_1.0.0_lerobot`，HF 名称却是 `bridge_orig_lerobot`。使用软链接对齐，不复制整份数据：

```bash
conda activate mlvla_star
mkdir -p "$VLA_WORK/data/OXE_LEROBOT"
python - <<'PY'
import json, os, shutil
from pathlib import Path
root = Path(os.environ['VLA_WORK'])
data = Path(os.environ['BRIDGE_DIR']).resolve()
info = json.loads((data/'meta/info.json').read_text())
assert info['robot_type'] == 'widowx'
assert info['features']['action']['shape'] == [7]
assert info['features']['observation.state']['shape'] == [8]
assert (data/'data').is_dir() and (data/'videos').is_dir()
src = Path(os.environ['STAR_REPO'])/'examples/SimplerEnv/train_files/modality.json'
dst = data/'meta/modality.json'
if not dst.exists():
    shutil.copy2(src, dst)
else:
    assert json.loads(dst.read_text()) == json.loads(src.read_text()), '已有 modality 不同，先核对再继续'
alias = root/'data/OXE_LEROBOT/bridge_orig_1.0.0_lerobot'
if not alias.exists():
    alias.symlink_to(data, target_is_directory=True)
else:
    assert alias.resolve() == data, '已有别名指向其他数据，请核对'
print('Dataset metadata and alias OK:', info['total_episodes'])
PY
```

### 9.2 生成数据接口检查配置

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
root = Path(os.environ['VLA_WORK'])
cfg = OmegaConf.load(Path(os.environ['BASE_RUN'])/'config.yaml')
vla = cfg.datasets.vla_data
vla.data_mix = 'bridge'
vla.data_root_dir = str(root/'data/OXE_LEROBOT')
vla.video_backend = 'torchvision_av'
vla.include_state = False
OmegaConf.save(cfg, root/'runs/data-check.yaml')
PY
cd "$STAR_REPO"
python starVLA/dataloader/lerobot_datasets.py --config_yaml "$VLA_WORK/runs/data-check.yaml"
```

首次加载可能需要扫描数据和统计信息，不代表卡死。视频是 AV1，使用源码标注支持 AV1 的 `torchvision_av` 后端。上面是接口检查配置，**不能直接当作已完成训练/验证划分的正式训练配置**。

模型所需接口：

| 内容 | 本路线约定 |
|---|---|
| 图像 | `image_0` 单视角，模型输入 224×224 |
| 指令 | 数据对应的自然语言任务描述 |
| 动作 | 7 维末端增量动作；未来 16 步，即 `(16, 7)` |
| state | 原数据 8 维；`include_state: false`，不送入模型 |
| 归一化 | 复用官方 Bridge transform，连续动作 q99，夹爪 binary |
| Base 反归一化 | 使用 Base 随附统计，key 为 `oxe_bridge` |

不要把原始 8 维 state 直接塞给声明 7 维 state 的动作头；也不要为了“对齐”随意删列、改模型参数。固定代码的数据打包默认就可以不包含 state，与 SimplerEnv 图像+语言输入一致。

3 号/6 号还需保存一个真实样本的图像、语言、动作尺寸和数值范围；数据目录检查不能替代样本展示。

## 10. 微调与正式评测如何接下去

选型和 Base 运行步骤已确定，以下是后续成员的交接要求，不宣称本手册已经实现完整训练流水线。

1. **按完整轨迹划分训练/验证集。** 建议固定随机种子 42，90%/10%，保存 episode ID 清单；训练集与验证集交集必须为空。既有划分符合要求就沿用。不能随机拆帧；也不能只生成两个清单却仍让 dataloader 读取全量数据。4 号需要在采样入口实际应用划分，并确认统计量没有利用本次验证集重新拟合。
2. **选定轻量方案：只训练动作头。** 训练配置从 Base 配置复制，`trainer.pretrained_checkpoint` 指向 Base `.pt`，完整加载全部模块后冻结 `qwen_vl_interface`；`trainer.freeze_modules` 写字符串 `"qwen_vl_interface"`，不能照抄模型 YAML 的布尔值 `true`。`reload_modules` 留空表示全模型加载。打印可训练参数清单，确认骨干不更新，且所有原始权重正确载入。
3. **数据只用 Bridge。** 保持 `data_mix: bridge`、`include_state: false`、action horizon 16、action dim 7。不要继承原作者的额外 VLM 联合训练数据列表；本课程动作头微调不需要另下这些视觉语言数据。固定版从 `framework.action_model.repeated_diffusion_steps` 读取 flow 重复数，若设为 4，应明确记录这是本次微调设置，而非原 YAML 的 8。
4. **先做一次真实 batch 的 forward/backward。** 检查 loss 有限、骨干无梯度、动作头有梯度，再决定正式训练步数。基于训练/验证数据选择 Final，不能用正式仿真成功率挑权重。当前手册不提供绕过划分的“一键训练”命令。
5. **Final 必须带自己的配置与匹配的统计文件。** 保存加载所需的完整状态，或明确可复原的 Base+增量方案；不要拿 Final 权重搭配不对应的统计。Base 的原始统计不被数据检查生成的统计覆盖。
6. **5 号在正式评测前冻结初始状态。** 开发目前只用 episode 0。可将 episode 1～10 作为正式候选；确认未用于调参后冻结，并记录机器人位置/朝向、场景、物体配置与实际 reset 种子。对象 episode ID 不等同于完整随机种子控制：官方 evaluator 的 `env.reset` 未显式传 seed，正式版本需补充并记录；策略服务端也应固定/记录随机种子。不要虚构本版不存在的 `--seed` 参数。
7. **确定 Final 后再做正式对比。** Base 和 Final 在独立进程分别加载，两者使用同一套预先确定的正式初始状态。T1/T2 每模型每任务 10 次，共 40 次，保存逐 episode success、版本、种子/初始化信息、日志和视频。成功率由你们实际运行结果计算，不抄模型卡。

## 11. 如何放到团队 GitHub

本手册可以直接存为 `docs/model_and_base_setup.md`。它是说明文件，不包含模型或数据，也不会自动下载；2 号需要在运行机器上执行其中命令。

推荐创建一个普通工作分支，方便大家审核文档和脚本变更；模型下载本身不要求创建分支。在你已有的 `ML-VLA` 克隆目录中执行：

```bash
git status
git switch -c docs/model-base-setup
mkdir -p docs
```

把本手册复制到 `docs/model_and_base_setup.md`，然后：

```bash
git add docs/model_and_base_setup.md
git commit -m "docs: pin StarVLA model, checkpoint and dataset setup"
git push -u origin docs/model-base-setup
```

在 GitHub 打开该分支发起 Pull Request，由 1 号核对记录后合并。已有未提交修改时先确认归属，不要执行 reset 或覆盖。仓库里已有同名分支则切换到它继续，不必重复创建。

后续提交小文件：版本清单、真实配置模板、数据划分清单、启动脚本、评测结果。数据、权重、下载缓存和机器绝对路径留在执行机器上；公开 checkpoint 链接另写入 README。

## 12. 常见问题快速定位

| 现象 | 首先检查 |
|---|---|
| 找不到 `config.yaml` | `.pt` 是否位于 `运行目录/checkpoints/`，YAML 是否在运行目录根部 |
| `--config_override` 不认识 | 本固定版无此参数，按第 6 节改运行 YAML |
| VLM 无法选择或路径不存在 | 基础 VLM 是否下载；绝对路径是否包含 `Qwen3-VL` |
| 权重 missing/unexpected keys | 代码 commit、型号和 checkpoint 是否混用；保留 strict 检查 |
| 找不到 Bridge 数据 | 是否建立了 `bridge_orig_1.0.0_lerobot` 别名，根目录是否正确 |
| 要求下载 RT-1 数据 | 本次训练配置是否仍为 `bridge_rt_1`，应改为 `bridge` |
| 视频无法解码 | 后端是否 `torchvision_av`，对应视频是否完整 |
| state 维度不符 | 是否误开了 `include_state` |
| WebSocket 连接失败 | 服务是否加载完成；两个终端是否同机、同端口；显式用 127.0.0.1 |
| 能推理但动作明显异常 | 是否使用 `oxe_bridge`，是否重复反归一化，夹爪/旋转是否绕过官方 adapter |
| T2 场景不对 | 是否使用 sink robot、`bridge_table_1_v2` 和 `bridge_sink.png` |

## 13. 核对依据与原始链接

- [StarVLA 固定版本 SimplerEnv 文档](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/examples/SimplerEnv/README.md)：明确列出所选权重与 Bridge LeRobot 数据路线。
- [所选模型卡及文件](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/tree/8ddb8116d5bd2953931619cfa9d60f278aeebceb)：20K checkpoint、基础 VLM、动作接口及 WidowX 训练依据。
- [Qwen 基础 VLM](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)。
- [Bridge 数据及元信息](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot/tree/0e9d76d07e9df3ea3eba257b2520d4913833fad2)：WidowX、LeRobot v2.0、7 维动作、8 维 state。
- [严格权重加载源码](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/starVLA/model/framework/base_framework.py)；[YAML 读取约定](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/starVLA/model/framework/share_tools.py)。
- [Bridge 数据注册](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/examples/SimplerEnv/train_files/data_registry/data_config.py)；[可选 state 的打包逻辑](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/starVLA/dataloader/gr00t_lerobot/datasets.py)。
- [官方 Bridge 场景参数脚本](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/examples/SimplerEnv/eval_files/auto_eval_scripts/star_bridge.sh)：本手册参考其参数，未直接调用作者机器配置。
- [SimplerEnv 固定版本](https://github.com/simpler-env/SimplerEnv/tree/eddb5690749a1e34267807c0f7b5eede323918d7)。
- 课程验收依据：你提供的《26ml_project.md》，B 类完整流程及评测要求。

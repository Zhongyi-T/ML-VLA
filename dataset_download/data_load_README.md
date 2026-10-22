# 数据集下载说明

本文件夹只用于下载数据集。将整个 `dataset_download` 文件夹放到你的电脑或服务器上，打开终端并进入这个文件夹再执行命令。

## 下载哪个数据集

- 数据集：`IPEC-COMMUNITY/bridge_orig_lerobot`
- 地址：https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot
- 内容：Bridge 机器人数据的 LeRobot 导出，包含数据表、视频和元数据。
- 下载不需要 GPU。真正的数据从上述网站下载，不包含在这个文件包里。

## 第一步 准备 Python

使用 Python 3.10 或更新版本。已有合适的 Python 环境可直接使用。若使用 Conda 且想单独建环境，可执行：

```bash
conda create -n bridge-download python=3.10 -y
conda activate bridge-download
```

## 第二步 安装下载工具

在本文件夹内执行：

```bash
python -m pip install -r requirements.txt
```

## 第三步 下载完整数据集

按你的系统选择一条命令。下载路径可以修改，请放在 GitHub 项目目录之外，并先确认磁盘空间充足。

Linux 或 WSL：

```bash
python download_bridge.py --dest ~/datasets/ml-vla/bridge_orig_1.0.0_lerobot
```

Windows PowerShell（示例使用 D 盘，若无 D 盘请换成实际目录）：

```powershell
python download_bridge.py --dest "D:\datasets\ml-vla\bridge_orig_1.0.0_lerobot"
```

脚本下载整个数据仓库，包含 `meta`、`data` 和 `videos`。第一次使用空目标文件夹；目录不存在时脚本会自动创建。脚本记录数据版本，后续重跑使用相同版本。

下载中断后，重新激活同一 Python 环境，在本文件夹里再次执行完全相同的下载命令。下载工具会复用已完成的文件；不要删除下载目录和其中的缓存。

## 第四步 确认下载调用完成

终端出现“数据集下载完成”后，下载目录应有 `meta`、`data`、`videos` 和 `download_manifest.json`。将以下内容告诉 3 号即可：

1. 数据存放在哪台机器、具体目录是什么。
2. 下载是否正常结束；若失败，发报错原文。
3. 下载目录中的 `download_manifest.json`。

“下载完成”仅表示下载工具正常返回，不代表样本质量或模型兼容性已检查。本包只负责下载。

## 常见问题

- 提示找不到 `requirements.txt` 或 `download_bridge.py`：终端没有进入本文件夹。
- 提示目标目录已有其他文件：换一个新的空目录，避免不同来源或版本混在一起。
- 网络超时：确认该机器能访问 Hugging Face，然后重跑原命令。
- 匿名下载出现限流：可在该机器运行 `hf auth login` 后重试；不要把 Token 写进代码或上传 GitHub。
- 空间不足：换到空间充足的磁盘，修改 `--dest`。不要把一个中断目录当作完整数据交付。

## 文件作用

| 文件 | 作用 |
|---|---|
| README.md | 本下载操作说明 |
| requirements.txt | 唯一下载依赖 huggingface_hub |
| download_bridge.py | 下载完整数据集，并记录版本和完成状态 |

上传 GitHub 时只上传这个小文件夹。真正下载得到的数据目录留在运行机器上。

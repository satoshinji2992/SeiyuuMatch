# SeiyuuMatch

当前版本：`2.0.0`

> 拍张照，测测你长得最像哪个女声优

[![Version](https://img.shields.io/badge/version-2.0.0-ff6b9d)](./CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-online-c44dff)](#)
[![Dataset](https://img.shields.io/badge/dataset-119_people-6c5ce7)](#)
[![Privacy](https://img.shields.io/badge/privacy-upload%20notice-2d3436)](#隐私说明)

## 运行界面

![SeiyuuMatch 页面预览](./example/page.png)

![识别结果预览](./example/recognize.png)

## 功能一览

| 功能           | 说明                                                       |
| -------------- | ---------------------------------------------------------- |
| 上传照片识别   | 自动检测人脸，告诉你最像谁                                 |
| 多人合照       | 一张图里有几个人就出几份结果                               |
| Top 5 候选排行 | 不只看第一名，展开看看其他候选人                           |
| 双企划标签页   | bangdream（粉）/ lovelive（金）标签切换，可跨企划多选      |
| 二挡模式       | 标准模式没识别到脸？降低阈值再来一次                       |
| 声优头像展示   | 结果卡片直接显示匹配声优的头像                             |
| 换脸娱乐功能   | 可把上传照片中的脸替换为最像的声优，并支持 CodeFormer 修复 |
| 跨团声优合并   | 同一人在多个团的结果自动去重合并                           |
| 数据集贡献     | 可以上传公开清晰的声优照片，帮我们补全数据                 |
| 反馈意见       | 页面内直接提交，方便后续改进                               |
| 特殊结果卡     | 支持隐藏候选触发专属展示，不进入普通候选排行               |

## 数据概览

| 项目         | 数量   | 详情                                                                                                                   |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| 声优条目     | 119 人 | bangdream 64 + lovelive 54 + hidden 1                                                                                  |
| bangdream    | 13 团  | ppp / roselia / afterglow / pastel / hhw / ras / morfonica / mygo / sumimi / avemujica / mewtype / millsage / dumbrock |
| lovelive     | 5 团   | μ's / 虹咲 / Aqours / Liella! / 莲之空                                                                                 |
| 默认检测范围 | 2 团   | MyGO!!!!! / Ave Mujica                                                                                                 |
| 头像展示     | 89 人  | 独立存放在 `avatar/`                                                                                                   |
| 乐队图标     | 13 个  | 独立存放在 `icon/`                                                                                                     |

## 在线访问

正式站点：

```
https://seiyuumatch.org
```

## 使用提醒

- 请上传清晰、正脸或半侧脸的照片，遮挡太多会影响结果。
- 识别结果纯属娱乐，不代表真实身份或任何评价。
- 每次刷新会弹出隐私说明，确认后才能开始。
- 数据集贡献适合上传公开照片，请不要上传敏感或他人隐私图片。

## 隐私说明

识别功能会把照片上传到服务器处理，并保存一份压缩后的历史记录。数据集贡献入口会把照片存到 `faces_upload/` 待审核目录。请不要上传敏感照片、他人隐私照片，或没有权利处理的图片。

---
\

## 本地部署

```bash
# 1. 创建环境
conda create -n seiyumatch python=3.10
conda activate seiyumatch
pip install opencv-python numpy insightface onnxruntime requests

# 2. 注册人脸特征
python3 register.py

# 3. 启动服务
python3 -u server.py --host 127.0.0.1 --port 3724
```

本地开发访问：

```
http://localhost:3724
```

### 注册工具

全量注册（需要较大内存）：

```bash
python3 register.py
```

逐团注册（节省内存，推荐）：

```bash
python3 register.py --by-group
```

只更新某个企划：

```bash
python3 register.py --project bangdream
```

只更新某个团：

```bash
python3 register.py --project bangdream --group mygo
```

### 换脸与 CodeFormer

换脸默认使用 `inswapper_128.onnx`。为了改善 128 脸块贴回后的发糊问题，可以接入 CodeFormer 做人脸修复。

准备 CodeFormer：

```bash
cd /path/to
git clone https://github.com/sczhou/CodeFormer.git
cd CodeFormer
pip install -r requirements.txt

PYTHONPATH=/path/to/CodeFormer python3 scripts/download_pretrained_models.py facelib
PYTHONPATH=/path/to/CodeFormer python3 scripts/download_pretrained_models.py CodeFormer
```

> 如果 `python3` 没有 `torch`，请使用装有 PyTorch 的解释器，例如 conda 环境里的 `python`。本项目调用 CodeFormer 时会自动设置 `PYTHONPATH`，所以不需要额外执行 `python basicsr/setup.py develop`。

单独测试换脸：

```bash
python3 face_swap.py tests/青木阳菜_1.jpg 羊宮妃那 -o swap_test_plain
```

单独测试换脸 + CodeFormer：

```bash
python3 face_swap.py tests/青木阳菜_1.jpg 羊宮妃那 \
  --restore-backend codeformer \
  --codeformer-dir /path/to/CodeFormer \
  --codeformer-weight 0.7 \
  -o swap_test_output
```

服务端启用 CodeFormer：

```bash
SWAP_RESTORE_BACKEND=codeformer \
CODEFORMER_DIR=/path/to/CodeFormer \
CODEFORMER_WEIGHT=0.5 \
MAX_SWAP_IMAGE_DIM=0 \
python3 -u server.py --host 127.0.0.1 --port 3724
```

`MAX_SWAP_IMAGE_DIM` 控制换脸输入图的最长边。默认 `0`，表示换脸时不主动缩小整图，优先保清晰度。服务器还会保留 `MAX_SWAP_UPLOAD_BYTES` 作为安全上限。

`SWAP_IDENTITY_BLEND` 控制换脸时保留多少目标身份特征。默认 `1.0`，表示完全使用目标身份特征；如果想保留一点原脸感，可以调到 `0.8` 或 `0.6`。

## 公开部署

公开推广时推荐使用双进程部署，可以缓解多人同时上传时的排队：

```
Cloudflare Tunnel
        ↓
nginx 8080
        ↓
SeiyuuMatch 3724 / 3725
```

服务器上准备好环境后，可以使用仓库里的部署模板：

```bash
sudo apt install -y nginx
sudo cp deploy/seiyuumatch@.service /etc/systemd/system/seiyuumatch@.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuumatch@3724 seiyuumatch@3725

sudo cp deploy/nginx-seiyuumatch.conf /etc/nginx/conf.d/seiyuumatch.conf
sudo nginx -t
sudo systemctl reload nginx
```

部署模板默认启用 CodeFormer，并打开脸部上采样：

```text
SWAP_RESTORE_BACKEND=codeformer
CODEFORMER_DIR=/root/CodeFormer
CODEFORMER_WEIGHT=0.5
CODEFORMER_FACE_UPSAMPLE=1
MAX_SWAP_IMAGE_DIM=0
SWAP_IDENTITY_BLEND=1.0
```

如果服务器暂时不启用 CodeFormer，请在 `/etc/systemd/system/seiyuumatch@.service` 里把 `SWAP_RESTORE_BACKEND` 清空，或删除相关 `Environment=` 行。

健康检查：

```bash
curl http://127.0.0.1:3724/health
curl http://127.0.0.1:3725/health
curl http://127.0.0.1:8080/health
```

Cloudflare Tunnel 指向：

```
Type: HTTP
URL: http://127.0.0.1:8080
```

双进程会提升并发能力和排队体验，但不会让单张照片的识别时间减半。

## 维护数据集

正式数据放在 `faces/`，用户贡献的照片会进入 `faces_upload/` 等待审核。

审核通过后，把照片移动到正式数据目录，然后重新注册特征：

```bash
python3 register.py --by-group
sudo systemctl restart 'seiyuumatch@*'
```

只更新某个企划：

```bash
python3 register.py --project bangdream
sudo systemctl restart 'seiyuumatch@*'
```

只更新某个团：

```bash
python3 register.py --project bangdream --group mygo
sudo systemctl restart 'seiyuumatch@*'
```

如果只改了已经生成好的 `features.npz`，只需要刷新服务：

```bash
sudo systemctl restart seiyuumatch@3724 seiyuumatch@3725
```

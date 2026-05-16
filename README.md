# SeiyuuMatch

当前版本：`1.2.0`

> 拍张照，测测你长得最像邦多利哪位女声优 🎸

[![Version](https://img.shields.io/badge/version-1.2.0-ff6b9d)](./CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-online-c44dff)](#)
[![Dataset](https://img.shields.io/badge/dataset-expandable-6c5ce7)](#)
[![Privacy](https://img.shields.io/badge/privacy-upload%20notice-2d3436)](#隐私说明)

## 运行界面

![SeiyuuMatch 页面预览](./example/page.png)

![识别结果预览](./example/recognize.png)

## 功能一览

| 功能           | 说明                                             |
| -------------- | ------------------------------------------------ |
| 上传照片识别   | 自动检测人脸，告诉你最像谁                       |
| 多人合照       | 一张图里有几个人就出几份结果                     |
| Top 5 候选排行 | 不只看第一名，展开看看其他候选人                 |
| 乐队范围筛选   | 可选择 MyGO!!!!!、Ave Mujica、sumimi 等多个团    |
| 二挡模式       | 标准模式没识别到脸？降低阈值再来一次             |
| 声优头像展示   | 结果卡片直接显示匹配声优的头像                   |
| 数据集贡献     | 可以上传公开清晰的声优照片，帮我们补全数据       |
| 反馈意见       | 页面内直接提交，方便后续改进                     |
| 特殊结果卡     | 支持隐藏候选触发专属展示，不进入普通候选排行     |

## 数据概览

| 项目         | 状态                             |
| ------------ | -------------------------------- |
| 声优条目     | 随 `features.npz` 和正式数据集更新 |
| 默认检测范围 | MyGO!!!!! / Ave Mujica / sumimi  |
| 支持团体     | 已包含新旧共 13 个团体入口        |
| 头像展示     | 独立存放在 `avatar/`             |
| 乐队图标     | 独立存放在 `icon/`               |
| 数据集上传   | 进入 `faces_upload/`，需人工审核 |

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

## 本地部署

```bash
# 1. 创建环境
conda create -n seiyumatch python=3.10
conda activate seiyumatch
pip install opencv-python numpy torch pytorch-lightning pillow requests

# 2. 注册人脸特征
python3 register.py

# 3. 启动服务
python3 -u server.py --host 127.0.0.1 --port 3724
```

本地开发访问：

```
http://localhost:3724
```

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
python3 register.py
sudo systemctl restart 'seiyuumatch@*'
```

只更新某个团：

```bash
python3 register.py --band mygo
sudo systemctl restart 'seiyuumatch@*'
```

如果只改了已经生成好的 `features.npz`，只需要刷新服务：

```bash
sudo systemctl restart seiyuumatch@3724 seiyuumatch@3725
```

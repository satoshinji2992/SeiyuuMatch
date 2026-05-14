# SeiyuuMatch

当前版本：`1.0.0`

> 拍张照，测测你长得最像邦多利哪位女声优 🎸

[![Version](https://img.shields.io/badge/version-1.0.0-ff6b9d)](./CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-online-c44dff)](#)
[![Dataset](https://img.shields.io/badge/dataset-48%20seiyuu-6c5ce7)](#)
[![Privacy](https://img.shields.io/badge/privacy-upload%20notice-2d3436)](#隐私说明)

## 功能一览

| 功能           | 说明                                             |
| -------------- | ------------------------------------------------ |
| 上传照片识别   | 自动检测人脸，告诉你最像谁                       |
| 多人合照       | 一张图里有几个人就出几份结果                     |
| Top 5 候选排行 | 不只看第一名，展开看看其他候选人                 |
| 乐队范围筛选   | 只测 MyGO!!!!!、Ave Mujica、sumimi，或者全员检测 |
| 二挡模式       | 标准模式没识别到脸？降低阈值再来一次             |
| 声优头像展示   | 结果卡片直接显示匹配声优的头像                   |
| 数据集贡献     | 可以上传公开清晰的声优照片，帮我们补全数据       |
| 反馈意见       | 页面内直接提交，方便后续改进                     |

## 数据概览

| 项目         | 状态                             |
| ------------ | -------------------------------- |
| 声优条目     | 48 人                            |
| 默认检测范围 | MyGO!!!!! / Ave Mujica / sumimi  |
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

## API

#### GET `/`

返回 Web 页面。

#### GET `/health`

健康检查：

```json
{
  "ok": true,
  "people": 47
}
```

#### GET `/face_groups`

返回 `faces/` 中的团、声优列表和照片数量，用于前端选择识别范围与数据集上传。

#### POST `/`

上传图片并识别。支持查询参数：

- `bands=mygo,avemujica`：限制识别候选团
- `mode=relaxed`：使用低阈值检测人脸

示例：

```bash
curl --noproxy localhost \
  -X POST 'http://localhost:3724/?bands=mygo,avemujica' \
  --data-binary @photo.jpg
```

响应包含最相似声优、bbox、当前相似度与 Top 5：

```json
{
  "faces": ["羊宮妃那"],
  "details": [
    {
      "name": "羊宮妃那",
      "band": "mygo",
      "similarity": 0.7812,
      "top5": [
        {"name": "羊宮妃那", "band": "mygo", "similarity": 0.7812}
      ],
      "bbox": [0.14, 0.21, 0.44, 0.78]
    }
  ],
  "mode": "default",
  "bands": ["avemujica", "mygo"]
}
```

#### POST `/upload_faces`

数据集贡献上传接口。前端会保存到：

```
faces_upload/<团>/<声优>/
```

这些照片不会自动进入正式识别库，需要人工审核后移动到 `faces/`，再重新注册。

## 人脸注册

正式数据放在：

```
faces/<团>/<声优>/
```

例如：

```
faces/
├── mygo/
│   ├── 羊宮妃那/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── 立石凛/
└── avemujica/
    └── 渡瀬結月/
```

前端识别结果展示的头像单独放在：

```
avatar/<声优>/1.jpg
```

服务端收到 `/avatar/<声优>` 请求时，会优先读取 `avatar/` 目录；如果没有找到头像，会临时回退到 `faces/` 中对应声优目录下的 `1.jpg`。

添加或修改正式照片后，重新注册并重启服务：

```bash
python3 register.py
# 重启 server.py 即可加载新的 features.npz
```

`features.npz` 不是热更新，服务启动时只加载一次。

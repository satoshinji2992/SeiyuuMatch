# SeiyuuMatch

基于 MTCNN 人脸检测对齐 + AdaFace 人脸识别的邦多利女声优相似度识别 Web 应用。

当前功能：

- 上传照片后识别最相似的声优，并展示上传裁脸、匹配声优头像与 Top 5 相似度。
- 支持按团选择识别范围，默认只启用 `mygo` 与 `avemujica`，这两组当前数据最多、稳定性最高。
- 支持默认阈值识别与“降低阈值再试一次”二挡识别。
- 提供隐私提示与数据集贡献入口，用户上传的数据集照片会先进入待审核目录。
- 支持 Cloudflare Tunnel 暴露到固定域名或临时公网地址。

## 目录结构

```
├── faces/                  # 正式人脸库：faces/<团>/<声优>/
├── faces_upload/           # 用户上传的待审核数据集照片，不提交 Git
├── uploads/                # 识别历史压缩图与 history.json，不提交 Git
├── AdaFace/                # AdaFace 模型仓库
│   └── pretrained/
│       └── adaface_ir50_ms1mv2.ckpt
├── register.py             # 注册正式人脸库并生成 features.npz
├── server.py               # HTTP 识别服务与上传接口
├── features.npz            # 注册生成的特征文件，包含 names/bands/features
├── index.html              # 前端页面
├── start_register.zsh      # 注册启动脚本
├── start_server.zsh        # 服务启动脚本
└── start_tunnel.zsh        # Cloudflare Quick Tunnel 临时公网脚本
```

## 快速开始

```bash
# 1. 创建环境
conda create -n seiyumatch python=3.10
conda activate seiyumatch
pip install opencv-python numpy torch pytorch-lightning pillow requests

# 2. 注册人脸特征
./start_register.zsh

# 3. 启动服务
./start_server.zsh
```

浏览器访问：

```text
http://localhost:3724
```

## Cloudflare Tunnel

### 固定域名

在 Cloudflare Zero Trust 中创建 Tunnel 后，将 Public Hostname 指向：

```text
Type: HTTP
URL: 127.0.0.1:3724
```

本机保持以下服务运行：

```bash
./start_server.zsh
cloudflared tunnel run --token <Cloudflare 给你的 token>
```

如果已经用 `cloudflared service install ...` 安装成系统服务，通常只需要启动 Python 服务。

### 临时地址

开发测试时也可以用 Quick Tunnel，不需要域名：

```bash
brew install cloudflared
./start_server.zsh
./start_tunnel.zsh
```

终端会打印一个 `https://*.trycloudflare.com` 临时地址。

## API

### GET `/`

返回 Web 页面。

### GET `/health`

健康检查：

```json
{
  "ok": true,
  "people": 47
}
```

### GET `/face_groups`

返回 `faces/` 中的团、声优列表和照片数量，用于前端选择识别范围与数据集上传。

### POST `/`

上传图片并识别。支持查询参数：

- `bands=mygo,avemujica`：限制识别候选团。
- `mode=relaxed`：使用低阈值检测人脸。

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

### POST `/upload_faces`

数据集贡献上传接口。前端会保存到：

```text
faces_upload/<团>/<声优>/
```

这些照片不会自动进入正式识别库，需要人工审核后移动到 `faces/`，再重新注册。

## 人脸注册

正式数据放在：

```text
faces/<团>/<声优>/
```

例如：

```text
faces/
├── mygo/
│   ├── 羊宮妃那/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── 立石凛/
└── avemujica/
    └── 渡瀬結月/
```

添加或修改正式照片后：

```bash
./start_register.zsh
./start_server.zsh
```

`features.npz` 不是热更新，服务启动时只加载一次。

## 隐私说明

识别功能会把用户选择的图片上传到服务器进行处理，并保存一份压缩后的历史记录。数据集贡献入口会把照片保存到 `faces_upload/` 待审核目录。请不要上传敏感照片、他人隐私照片，或没有权利处理的图片。

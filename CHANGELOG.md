# Changelog

## 1.0.0 - 2026-05-14

SeiyuuMatch 第一个正式版本。

### Features

- 基于 MTCNN + AdaFace 的声优相似度识别 Web 应用。
- 支持上传图片识别多张人脸，并展示匹配声优头像、相似度和 Top 5 候选。
- 支持按团限制识别范围，默认启用 `mygo`、`avemujica`、`sumimi`。
- 支持默认阈值识别，以及“识别不到？降低阈值再试一次”的二挡检测。
- 支持独立头像目录 `avatar/<声优>/1.jpg` 和团标目录 `icon/<团>.png`。
- 支持数据集贡献上传，保存到 `faces_upload/` 等待人工审核。
- 支持反馈意见提交，保存到 `feedback/feedback.jsonl`。
- 支持隐私提示确认、左下角隐私提示入口、页面缩放控制和 GitHub 跳转。
- 支持 Cloudflare Tunnel 固定域名部署。

### Data

- 当前特征库包含 48 个声优条目。
- 修正 `志崎樺音` 名称。
- 支持 `佐々木李子` 同时归属 `avemujica` 和 `sumimi`。

### Operations

- 新增 `bench_env.py` 环境检测与性能测试脚本。
- README 增加服务器部署、systemd、Cloudflare Tunnel、性能测试说明。
- 上传识别图片限制为 6MB。
- 数据集上传单次限制为 80MB，`faces_upload/` 总量限制为 500MB。

### Notes

- `features.npz` 不是热更新，修改正式 `faces/` 数据后需要重新运行 `register.py` 并重启服务。
- 生产环境推荐让 Python 服务仅监听 `127.0.0.1:3724`，由 Cloudflare Tunnel 对外提供 HTTPS。

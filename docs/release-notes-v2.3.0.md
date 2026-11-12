## v2.3.0 — 安全、性能、结构与体验的全面升级

本地语音识别字幕工具九批改进合集。安装包为目录版（EXE + `_internal/`），首次运行按需下载模型。

### 🔐 下载安全
- 模型与可执行件下载补 **SHA256 清单校验**（27 项，含 ffmpeg.zip）
- TLS 证书验证降级**只允许固定可信主机**（HF / GitHub / 镜像我方白名单）
- CrispASR 核心 zip 首次下载记录 **TOFU pin**，防止同 URL 内容被静默替换

### ⚡ 性能
- 模型加载**四路并行**（VAD / 说话者分离 / ASR 编译 / 对齐器），启动约 20s
- 推理锁粒度缩小（前处理/解码移出临界区）
- 切点吸附向量化（与朴素实现对拍等价）
- 字幕列表 **>400 段自动窗口化渲染** + 事件委托 + 编辑脏区更新
- **卡拉OK游标式增量渲染**（多数帧零 DOM 写）

### 🧱 结构
- `webview_backend` 1982 → 1343 行，拆出 `slicing` / `model_catalog` / `transcription_service` / `transcription_queue` / `engine_types`
- 前端迁移 **ES modules**（`main` 入口 + `state` 容器 + `views/modelView` + `appCore`）
- CLI 与 GUI 字幕写出统一（补回说话者前缀）

### ✍️ 体验
- 字幕**撤销/重做**（Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z）
- 字幕**搜索替换**（大小写不敏感、逐处/全部替换、Enter 循环）
- 转录**取消按钮**
- 录制页**实时延迟提示**（按当前引擎给出出字延迟预期）
- i18n 存量清理（30+ 新 key，简中/English 全备）

### 🧰 功能
- **批量转录任务队列**（`/api/queue`，串行执行 + SSE 事件推送）
- **ASS / VTT 导出**：CLI `-f ass|vtt`（有字级数据时 ASS 自动带 `\k` 卡拉OK）
- 服务端加固：坏 JSON 回 400、上传上限、临时目录全路径清理、SSE 丢事件告警
- 字幕落盘**原子写**（中断不留半截 SRT）
- 说话者指派改为**重叠最长段**（取代时间中点，跨界行更稳）

### 🧪 工程
- GitHub Actions CI（unittest + ruff + mypy 分级门禁）
- `requirements-lock.txt` 依赖锁定
- 测试 98 → **151 例**（新增下载断点续传/TOFU 拦截、转录编排、任务队列、semver 预发布比较等；真机全绿）

### 📋 修复
- CLI 丢弃说话者前缀
- semver 预发布版误判为更新（`2.3.0-rc.1` 不再比 `2.3.0` 新）
- 模型页「下载并加载模型」按钮与加速器提示文案矛盾
- 连开两卡编辑卡死、模型加载期间改选误报、对齐失败静默降级等

### 安装
1. 解压 `STTMiniTool-v2.3.0-windows.zip` 到**全英文路径**
2. 运行 `STTMiniTool.exe`（首次按需下载模型；NVIDIA 用户建议在「模型」页选择 CUDA 加速版本）
3. Windows 10 需预装 Edge WebView2 Runtime

**完整变更**：https://github.com/Superce1l/STTMiniTool/compare/v2.2.10...v2.3.0

# webview/ — 共用前端（语音识别）

WebView 版（本分支唯一主线）与纯设计预览共用**同一份**前端界面。
设计方向：浅色清爽编辑室风，图标同款深蓝 #2859C5 强调色
（Streamline「语音转文字」图标）。

## 结构

```
webview/
├── index.html        壳 + 视图（音频 / 录制 / 模型 / 设置）
├── css/app.css       设计系统（色彩 token、组件、响应式）
├── favicon.png       标签页图标（64px，由 assets/make_icon.py 生成）
└── js/
    ├── bridge.js     window.QwenAPI 统一桥接层（传输抽象 + event bus + 请求超时）
    ├── i18n.js       界面语言（简体中文 / English）
    ├── main.js       ESM 入口：import 编排 + boot()
    ├── state.js      跨视图共享状态容器（curSegs / speakerNames / REC / 搜索 …）
    ├── util.js       纯工具（DOM 简写 / 字幕时间格式 / 撤销栈 / 虚拟窗口 / 卡拉OK游标）
    ├── appCore.js    应用核心（音频页 / 字幕编辑与渲染 / 波形 / 录制 / 设置 / 历史）
    └── views/
        └── modelView.js   模型页视图（自检 / 引擎模型选择 / 加速版本 / 加载协调）
```

> `bridge.js` / `i18n.js` 保持经典脚本（挂全局 `window.QwenAPI` / `window.I18N`），
> 其余为 **ES modules**（`<script type="module" src="js/main.js">`）。
> import 图无环单向：`main → state → util → views/modelView → appCore`，
> 视图模块对 appCore 的反向导航经 `state.gotoView` 回调（零反向 import）。

## 架构：本机服务器 + 浏览器渲染（同 Eel / flaskwebgui，但纯标准库）

不直接依赖 pywebview 的 js_api（其 Windows pythonnet 后端有序列化递归 bug，
且 EXE 内要打包 .NET）。前端完全通过 HTTP/SSE 与 server 沟通。

| 后端文件 | 角色 |
|---|---|
| `webview_backend.py` | 与窗口/传输无关的业务逻辑（status/settings/devices/accel/transcribe），重用 app.py 的 OpenVINO ASREngine |
| `webview_server.py`  | stdlib HTTP：serve 本文件夹 + `/api/*` + SSE `/api/events`；只绑 127.0.0.1 |
| `app_webview.py`     | 起 server → 后台加载模型 → 原生 WebView2 窗口（pywebview 只加载网址；WebView2 不可用时回退 Edge `--app` → 默认浏览器）→ 等关闭收 server |

## 主要视图

| 视图 | 说明 |
|---|---|
| 音频转字幕 | 拖入音频/视频 → 转录 → 波形 + 字幕卡（卡拉OK逐字模式）|
| 录制转换 | 浏览器 MediaRecorder + 停顿检测，分段实时识别 |
| 模型与设备 | 系统自检、基础/高级双模式选模型、加速版本（CUDA/Vulkan）|
| 设置 | 界面缩放（下拉框）、输出格式、VAD 灵敏度、每段最长秒数、HF 镜像、FFmpeg 路径、主题、界面语言 |

> 2.1.0 起：批次识别与端点服务（LAN API / QR / Cloudflare）已移除；
> 简繁词汇转换设置已移除（引擎固定输出简体）。

## bridge.js 传输检测

| 条件 | 模式 | 传输 |
|---|---|---|
| HTTP 且 `/health` 回 200 | `web` | `/api/*`（本机 server）；进度经 SSE `/api/events` |
| 其余（含纯静态服务器 / file://）| `mock` | 内置范例数据（设计预览） |

UI 只调用 `QwenAPI.transcribe()` 等抽象方法。文件选择走浏览器原生
`<input type=file>` → 以 multipart 上传到本机 server。

## 本地预览（纯设计，无后端）

```bash
python -m http.server 8777 --bind 127.0.0.1   # 于 webview/ 下执行
# 开 http://127.0.0.1:8777/index.html → /health 探测失败 → 自动 mock 模式
```

## 实机运行

```bash
python app_webview.py          # 起本机 server + 加载模型 + WebView2 窗口
# 调试：python webview_server.py 固定起在 :8765，再用浏览器开
```

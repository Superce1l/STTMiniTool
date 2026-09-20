"""version.py — 应用程序版本与更新来源设置（单一事实来源）

此档被 app.py / app_webview.py / webview_backend.py 引用。

版本规则：
    语义化版本 MAJOR.MINOR.PATCH。
    每次发布新编译版时，先把 __version__ 往上加，
    再到 GitHub 建立同名 tag 的 Release，并上传整包 ZIP 资产。
"""
from __future__ import annotations

__version__ = "2.1.0"

# WebView 版独立版本字符串（与 __version__ 同值）。
# 显示于 WebView 设置页的版本徽章；保留常数名是为了不动既有引用
# （webview_backend._app_version / setting.py 版本徽章）。
WEBVIEW_VERSION = __version__

# ── 2.1.0 更新汇总（本地改进版，基于上游 2.0.0）────────────────────────────
#   界面：全面简体中文——界面、注释、文档统一简体；字库换用 Noto Sans SC；
#         识别输出默认简体（可在设置切换简繁词汇转换）。
#   功能：移除「批次识别」与「端点服务」（LAN API／QR／Cloudflare 通道）。
#   模型：移除 TEA-ASR-1.1 与 Whisper Breeze-ASR-26，
#         新增 OpenAI Whisper 官方模型（Base / Small / Medium / Large / Large-Turbo，
#         ggerganov/whisper.cpp GGML，CrispASR whisper 后端原生支持）。
#   核心：CrispASR 升级 v0.8.33；OpenVINO 依赖升级至 2026.4；chatllm
#         向下兼容层对齐 v24。
#   加速：NVIDIA 显卡用户默认推荐 CUDA（依驱动版本自动选 CUDA 13 / 12，
#         自带 runtime 免装 Toolkit）；AMD/Intel 仍为 Vulkan。
#   长音频：新增 FFmpeg 切片转录——超过 2 小时的音频自动按 30 分钟切片（15 秒
#         重叠防切点截字）、逐片转录、时间轴平移合并，内存峰值大幅降低。
#   图标：全新 teal 声波麦克风设计（assets/make_icon.py 程序化生成）。
#   修复：删除端点模块时误删 _persist_setting 导致基础模式页 500；语言下拉
#         title 误绑「说话者分离」；uiLang 默认值改为简体中文。

# 自动更新来源：GitHub repo（owner/name）+ 发行页
GITHUB_REPO = "Superce1l/STTMiniTool"
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"

"""version.py — 应用程序版本与更新来源设置（单一事实来源）

此档被 app.py / app_webview.py / webview_backend.py 引用。

版本规则：
    语义化版本 MAJOR.MINOR.PATCH。
    每次发布新编译版时，先把 __version__ 往上加，
    再到 GitHub 建立同名 tag 的 Release，并上传整包 ZIP 资产。
"""
from __future__ import annotations

__version__ = "2.1.1"

# WebView 版独立版本字符串（与 __version__ 同值）。
# 显示于 WebView 设置页的版本徽章；保留常数名是为了不动既有引用
# （webview_backend._app_version / setting.py 版本徽章）。
WEBVIEW_VERSION = __version__

# ── 2.1.1 更新汇总（本地改进版，基于上游 2.0.0）────────────────────────────
#   界面自适应：主内容区随窗口宽度伸缩（窄窗撑满、宽窗居中封顶）；设置页
#         输入框、录制控件等不再写死像素宽度，窄窗口自动换行；系统自检
#         面板、批次列表在窄视口改单栏／堆叠，不再溢出。
#   关闭流程：关窗时有任务进行中（识别音频／下载加载模型／录音）→ 弹原生
#         确认框提醒，确认后中断任务并退出；无任务直接关闭。关闭后所有
#         后台子进程随 Job Object 一并终止，不留残余进程／显存占用。
#   窗口适配：启动时按屏幕工作区（SPI_GETWORKAREA）收敛窗口尺寸并居中，
#         任务栏各种形态（贴底／靠侧／自动隐藏）均正确避开，小屏幕不
#         溢出边缘。

# 自动更新来源：GitHub repo（owner/name）+ 发行页
GITHUB_REPO = "Superce1l/STTMiniTool"
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"

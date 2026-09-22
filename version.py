"""version.py — 应用程序版本与更新来源设置（单一事实来源）

此档被 app.py / app_webview.py / webview_backend.py 引用。

版本规则：
    语义化版本 MAJOR.MINOR.PATCH。
    每次发布新编译版时，先把 __version__ 往上加，
    再到 GitHub 建立同名 tag 的 Release，并上传整包 ZIP 资产。
"""
from __future__ import annotations

__version__ = "2.2.0"

# WebView 版独立版本字符串（与 __version__ 同值）。
# 显示于 WebView 设置页的版本徽章；保留常数名是为了不动既有引用
# （webview_backend._app_version / setting.py 版本徽章）。
WEBVIEW_VERSION = __version__

# ── 2.2.0 更新汇总（本地改进版，基于上游 2.0.0）────────────────────────────
#   热切换：引擎／模型／加速版本切换不再需要重启——选好后点「下载并加载
#         模型」即就地热切换（转录进行中禁切换）；加载失败自动回退原引擎。
#   字幕：输出文件名拼接所用引擎与模型（如「唯一 [CrispASR · Whisper
#         Base].srt」），多模型对比不再混淆。
#   日志：新增运行日志（logs/app-年月.log）——转录开始/完成（含耗时、
#         段数）、模型加载、下载完成与完整性失败全程留痕。
#   可靠：下载完成做 Content-Length 对账，中断残档即删即重下；GGUF 模型
#         加 100MB 大小下限，杜绝残档被当完整模型（此前会静默加载失败）。
#   修复：CrispASR 引擎打开含中文／非 ASCII 字符的音频路径报
#         「input file not found」——自动复制 ASCII 文件名临时副本转录；
#         OpenVINO 路径 VAD 崩溃「Unable to handle object of type int」
#         —— silero v4 的 sr 输入改用 0-d int64 张量；
#         crispasr.exe 异常退出时错误消息带上子进程末尾输出（真实原因
#         不再被吞）；打包补齐缺失的 scipy 依赖（说话者分离必需）。
#   智能：长音频切片时长依可用物理内存自适应（5–30 分钟），小内存
#         机器不再有转录中 OOM 风险；内存充裕行为不变。
#   下载：CrispASR 核心不再固定 v0.8.33，自动获取 GitHub 最新 release
#         （离线回退 0.8.33）；旧版安装下次加载时自动升级。
#   精简：模型页移除「基础/高级」双模式，改为 引擎（OpenVINO/CrispASR
#         两列并排）→ 选择模型 直选；说话者分离归入自检「共享组件」；
#         删除闲置的 OpenCC 简繁转换链路。

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

# ── 2.1.0 更新汇总（本地改进版，基于上游 2.0.0）────────────────────────────
#   界面：全面简体中文——界面、注释、文档统一简体；字库换用 Noto Sans SC；
#         识别输出默认简体（可在设置切换简繁词汇转换）。
#   功能：移除「批次识别」与「端点服务」（LAN API／QR／Cloudflare 通道）。
#   模型：移除 TEA-ASR-1.1 与 Whisper Breeze-ASR-26，
#         新增 OpenAI Whisper 官方模型（Base / Small / Medium / Large / Large-Turbo，
#         ggerganov/whisper.cpp GGML，CrispASR whisper 后端原生支持）。
#   核心：CrispASR 升级 v0.8.33；OpenVINO 依赖升级至 2026.4。
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

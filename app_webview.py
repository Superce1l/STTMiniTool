"""app_webview.py — 桌面 WebView 启动器（本机 HTTP server + 原生 WebView2 窗口）

窗口用 pywebview 开「只加载本机网址」的原生 WebView2 窗口：系统内置
WebView2 runtime（不打包 Chromium → EXE 小）、原生标题栏、无浏览器感、
无登入提示。前端完全通过 HTTP/SSE 与 server 沟通，**不使用 pywebview 的
js_api** —— 那一层（pythonnet 序列化 .NET 对象）正是先前踩到无限递回
死锁的元凶；改成纯加载网址后窗口稳定（实测无递回、窗口持续存活）。

启动顺序：起本机 server(只绑 127.0.0.1) → 背景加载模型 → 开原生窗口。
WebView2 不可用时 fallback：系统 Edge --app 无痕窗口 → 再不行开默认浏览器。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path

from proc_guard import setup_kill_on_close_job
from webview_server import WebViewServer

APP_NAME = "语音识别小工具"
WIN_W, WIN_H = 1180, 820


# ════════════════════════════════════════════════════════
# 标题栏配色：配合浅色页面主题 → 白底黑字
# ════════════════════════════════════════════════════════
#   原生窗口（WebView2）与 Edge --app 窗口的标题栏由 OS（DWM）绘制，默认会
#   跟随系统的深／浅色设置 —— 系统若为深色，标题栏就是深底白字，与本程序
#   固定的浅色页面主题不搭。Windows 11（build 22000+）提供 DWM 属性可直接
#   指定标题栏配色，故在窗口显示后调用 DwmSetWindowAttribute 强制：
#     • DWMWA_USE_IMMERSIVE_DARK_MODE(20)=0 → 关闭深色模式（浅色标题栏）
#     • DWMWA_CAPTION_COLOR(35)=白          → 标题栏底色（COLORREF 0x00BBGGRR）
#     • DWMWA_TEXT_COLOR(36)=黑             → 标题栏文字／图标色
#   两种窗口（原生 / Edge fallback）的标题都是页面 <title>「声音识别」，故可用
#   同一支以窗口标题查找 HWND 的辅助函数套用。失败（旧版 Windows / 非 win32）
#   一律静默略过，不影响窗口开启。
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36
_COLOR_WHITE = 0x00FFFFFF        # COLORREF：白底（浅色标题栏底）
_COLOR_BLACK = 0x00000000        # COLORREF：黑字
_COLOR_DARK_BG = 0x0026201E      # COLORREF 0x00BBGGRR ← #1E2026（深色面板底，与 CSS 一致）

_initial_theme = "light"         # main() 启动时由设置填入，供窗口装饰 worker 取用初始深浅


def _os_dark() -> bool:
    """读 Windows 个人化设置判断系统是否为深色（AppsUseLightTheme==0）。"""
    if sys.platform != "win32":
        return False
    try:
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            v, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
            return v == 0
    except Exception:
        return False


def _resolve_dark(theme: str) -> bool:
    """外观偏好（light/dark/system）→ 是否深色。"""
    if theme == "dark":
        return True
    if theme == "light":
        return False
    return _os_dark()              # system → 跟随 OS


def _set_titlebar(hwnd: int, dark: bool) -> bool:
    """对 HWND 套用标题栏配色：dark→深底白字、light→白底黑字。返回是否设成底色。"""
    if not hwnd or sys.platform != "win32":
        return False
    try:
        dwm = ctypes.windll.dwmapi
    except Exception:
        return False

    def _set(attr: int, value: int) -> int:
        v = ctypes.c_int(value)
        return dwm.DwmSetWindowAttribute(
            wintypes_hwnd(hwnd), attr, ctypes.byref(v), ctypes.sizeof(v))

    cap = _COLOR_DARK_BG if dark else _COLOR_WHITE
    txt = _COLOR_WHITE if dark else _COLOR_BLACK
    ok = False
    try:
        _set(_DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
    except Exception:
        pass
    try:
        if _set(_DWMWA_CAPTION_COLOR, cap) == 0:        # S_OK
            ok = True
        _set(_DWMWA_TEXT_COLOR, txt)
    except Exception:
        pass
    return ok


def apply_titlebar_theme(theme: str):
    """供 webview_backend 回调：依新外观设置实时切换窗口标题栏深浅（两种窗口皆套）。"""
    dark = _resolve_dark(theme or "light")
    hwnd = _find_app_hwnd()
    if hwnd:
        _set_titlebar(hwnd, dark)


def wintypes_hwnd(hwnd: int):
    """把 Python int 包成 DWM API 接受的 HWND（c_void_p）。"""
    return ctypes.c_void_p(int(hwnd))


def _find_app_hwnd() -> int:
    """以窗口标题（APP_NAME）查找顶层窗口 HWND；找不到回 0。"""
    if sys.platform != "win32":
        return 0
    try:
        return int(ctypes.windll.user32.FindWindowW(None, APP_NAME) or 0)
    except Exception:
        return 0


# ────────────────────────────────────────────────────────
# 窗口 / 任务栏图标（猫耳毛绒小耳机）
# ────────────────────────────────────────────────────────
#   为何需要程序化设置：PyInstaller 的 --icon 只内嵌进「打包后的 EXE」，
#   开发模式 `python app_webview.py` 的宿主进程是 python.exe → 任务栏显示
#   python 默认图标。pywebview 6.x 也没有可靠的 per-window 图标 API。故改用
#   Win32 `WM_SETICON` 在窗口建立后把 icon.ico 套上去（标题栏小图 + 任务栏大图），
#   并设置独立的 AppUserModelID，让任务栏不与 python.exe 共享同一颗按钮／图标。
_APP_USER_MODEL_ID = "Superce1l.STTMiniTool"
_WM_SETICON = 0x0080
_ICON_SMALL, _ICON_BIG = 0, 1
_IMAGE_ICON = 1
_LR_LOADFROMFILE, _LR_DEFAULTSIZE = 0x00000010, 0x00000040


def set_app_user_model_id():
    """让 Windows 把本进程视为独立 App（任务栏图标不跟 python.exe 绑在一起）。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except Exception:
        pass


def _resolve_icon_path() -> Path | None:
    """找出 icon.ico：开发模式在 assets/；冻结模式在 _MEIPASS／exe 旁。"""
    cands = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cands.append(Path(meipass) / "icon.ico")
        cands.append(Path(sys.executable).parent / "icon.ico")
    cands.append(Path(__file__).resolve().parent / "assets" / "icon.ico")
    for c in cands:
        if c and c.exists():
            return c
    return None


def _set_window_icon(hwnd: int, ico: Path) -> bool:
    """以 WM_SETICON 把 icon.ico 套到窗口（同时更新标题栏小图与任务栏大图）。"""
    if not hwnd or sys.platform != "win32":
        return False
    try:
        u = ctypes.windll.user32
        # 设置 restype/argtypes：HICON 是 64 位 handle，未指定会被 ctypes 截成 int
        # → handle 失效。LoadImageW 回 HANDLE、SendMessageW 收 HWND/WPARAM/LPARAM。
        u.LoadImageW.restype = wintypes.HANDLE
        u.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                 ctypes.c_int, ctypes.c_int, wintypes.UINT]
        u.SendMessageW.restype = ctypes.c_ssize_t
        u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                   wintypes.WPARAM, wintypes.LPARAM]
        path = str(ico)
        big = u.LoadImageW(None, path, _IMAGE_ICON, 0, 0,
                           _LR_LOADFROMFILE | _LR_DEFAULTSIZE)        # 大图（任务栏）
        small = u.LoadImageW(None, path, _IMAGE_ICON, 16, 16, _LR_LOADFROMFILE)  # 小图（标题栏）
        h = wintypes_hwnd(hwnd)
        if big:
            u.SendMessageW(h, _WM_SETICON, _ICON_BIG, big)
        if small:
            u.SendMessageW(h, _WM_SETICON, _ICON_SMALL, small)
        return bool(big or small)
    except Exception:
        return False


def _decorate_window_async():
    """背景轮询，待窗口建立后套用浅色标题栏 + 程序图标（原生 / Edge 皆适用）。

    窗口建立有延迟，且 DWM／WM_SETICON 都需 HWND 存在后才能套 —— 故短轮询，
    两者皆完成（或逾时）即停。独立 daemon 绪执行，不阻塞主流程。
    """
    ico = _resolve_icon_path()
    dark = _resolve_dark(_initial_theme)

    def worker():
        themed = False
        iconed = ico is None          # 无 icon 档则视为「已处理」（只做标题栏）
        for _ in range(40):           # 最多 ~6 秒（40 × 0.15s）
            hwnd = _find_app_hwnd()
            if hwnd:
                if not themed:
                    themed = _set_titlebar(hwnd, dark)
                if not iconed:
                    iconed = _set_window_icon(hwnd, ico)
                if themed and iconed:
                    return
            time.sleep(0.15)
    threading.Thread(target=worker, name="window-decorate", daemon=True).start()


# ════════════════════════════════════════════════════════
# 前置检测：系统是否具备 WebView2 Runtime（原生窗口的前提）
# ════════════════════════════════════════════════════════
#   为何需要：缺 WebView2 Runtime 时，pywebview 仍会「开出窗口」但 WebView2
#   控制项初始化失败 —— edgechromium 只记 log 后 return，**不丢例外、不关窗**，
#   于是 webview.start() 既不返回也不报错 → 我们的 Edge fallback 永远不会触发，
#   使用者只看到一个空白窗口（像整个程序坏掉）。故启动前先用官方注册表键判定，
#   不具备就直接走 Edge --app（纯 Chromium，不需 WebView2/.NET）。
#   （缺 .NET 的情况则由 run_native_window 内的例外处理干净退回 Edge。）
_WEBVIEW2_CLIENT = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def webview2_available() -> bool:
    """检测系统是否已安装 WebView2 Runtime（Evergreen 或 per-user）。"""
    if sys.platform != "win32":
        return False
    import winreg
    for hive, path in (
        (winreg.HKEY_LOCAL_MACHINE,
         rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT}"),
        (winreg.HKEY_LOCAL_MACHINE,
         rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT}"),
        (winreg.HKEY_CURRENT_USER,
         rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT}"),
    ):
        try:
            with winreg.OpenKey(hive, path) as k:
                pv, _ = winreg.QueryValueEx(k, "pv")
                if pv and pv not in ("", "0.0.0.0"):
                    return True
        except OSError:
            continue
    return False


# ════════════════════════════════════════════════════════
# 主要：原生 WebView2 窗口（pywebview，只加载网址、无 js_api）
# ════════════════════════════════════════════════════════
def run_native_window(url: str) -> bool:
    """以原生 WebView2 窗口加载 url，阻塞至关窗回 True；不可用/失败回 False。"""
    try:
        import webview
    except Exception:
        return False
    try:
        # 允许下载：原生 WebView2 默认 ALLOW_DOWNLOADS=False 会「静默取消」<a download>，
        # 导致「字幕存档」按钮没有任何反应、也不跳存档窗口。开启后 edgechromium 会弹
        # 原生 Save 窗口让使用者选位置。
        try:
            webview.settings['ALLOW_DOWNLOADS'] = True
        except Exception:
            pass
        webview.create_window(
            APP_NAME, url=url,
            width=WIN_W, height=WIN_H, min_size=(960, 680),
        )
        _decorate_window_async()   # 窗口显示后背景套用白底黑字标题栏 + 程序图标
        webview.start()            # 阻塞至窗口关闭（在主线程）
        return True
    except Exception as e:
        print(f"[{APP_NAME}] 原生窗口失败，改用 Edge：{e}")
        return False


# ════════════════════════════════════════════════════════
# Fallback：系统 Microsoft Edge --app 无痕窗口
# ════════════════════════════════════════════════════════
def _find_edge() -> str | None:
    for c in [
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]:
        if c and Path(c).is_file():
            return str(c)
    return None


def _profile_dir() -> str:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "edge-profile"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def open_edge_app(url: str) -> subprocess.Popen | None:
    edge = _find_edge()
    if edge:
        args = [
            edge, f"--app={url}", "--inprivate",
            f"--user-data-dir={_profile_dir()}",
            f"--window-size={WIN_W},{WIN_H}",
            "--no-first-run", "--no-default-browser-check", "--disable-sync",
            "--disable-background-networking",
            "--disable-features=msImplicitSignin,msEdgeSyncEnabled,msEdgeWelcomePage,EdgeFollowEnabled",
        ]
        flags = 0x08000000 if sys.platform == "win32" else 0
        try:
            proc = subprocess.Popen(args, creationflags=flags)
            _decorate_window_async()   # Edge --app 窗口同样套白底黑字标题栏 + 图标
            return proc
        except Exception:
            pass
    webbrowser.open(url)
    return None


def _cli_argv() -> list[str] | None:
    """有带已知子命令（或 --help/--version）→ 返回要交给 CLI 的 argv；否则 None。

    判断刻意保守：只认 cli_mode.SUBCOMMANDS 里的字，其余一律当作「正常启动
    GUI」。这样就算使用者的捷径或文件关联打开时带了奇怪参数，也不会意外变成无头模式。
    """
    argv = sys.argv[1:]
    if not argv:
        return None
    from cli_mode import SUBCOMMANDS
    for a in argv:
        if a in SUBCOMMANDS:
            return argv
        if a in ("-h", "--help", "--version"):
            return argv
        if not a.startswith("-"):        # 第一个非标志不是子命令 → 不是 CLI
            break
    return None


def main():
    # ── 无头 CLI 模式（给外部 Agent／脚本）──────────────────────────────
    #   带子命令就完全不开窗口、不起 server，执行完即退出并返回退出码。
    cli = _cli_argv()
    if cli is not None:
        setup_kill_on_close_job()        # 子程序（crispasr.exe 等）仍要连带回收
        from cli_mode import main as cli_main
        sys.exit(cli_main(cli))

    # 先把自己绑进 Job Object：此后 start_load()/转录派生的 crispasr.exe、
    # chatllm main.exe(FA) 等子程序都会自动纳入同一 Job，关窗时连带被杀。
    # handle 由 proc_guard 模块级变量持有（生命周期 == 进程），无须在此保管。
    setup_kill_on_close_job()

    # 设置独立 AppUserModelID（要在开任何窗口前）→ 任务栏图标不跟 python.exe 共享。
    set_app_user_model_id()

    srv = WebViewServer(host="127.0.0.1", port=0)   # 随机空闲端口，只绑回环
    srv.start()
    # 窗口外观：读持久化外观偏好决定初始标题栏深浅；注册回调，让 UI 切换主题时
    # 窗口标题栏实时跟着深/浅。
    global _initial_theme
    try:
        _initial_theme = srv.backend.persisted_theme()
        srv.backend.set_theme_callback(apply_titlebar_theme)
    except Exception:
        pass
    # 自动加载策略：只有「目前选择的模型已下载」时才自动加载（快速、无下载）。
    # 首次／模型未下载 → 不自动触发下载，停在模型页等使用者选好（可改 Whisper 等）
    # 并按「下载并加载模型」才开始下载。避免一开启就硬抓默认模型。
    try:
        if srv.backend.selected_model_present():
            srv.backend.start_load()
        else:
            print(f"[{APP_NAME}] 选择的模型尚未下载 → 等使用者于模型页确认后再下载。")
    except Exception:
        srv.backend.start_load()                     # 保底：判断失败仍尝试加载
    url = srv.url
    print(f"[{APP_NAME}] {url}")

    # 只有在系统具备 WebView2 Runtime 时才试原生窗口（否则会卡在空白窗，
    # 见 webview2_available 注释）；缺则直接走 Edge --app，不需 WebView2/.NET。
    has_wv2 = webview2_available()
    if not has_wv2:
        print(f"[{APP_NAME}] 未检测到 WebView2 Runtime，改用 Edge --app 窗口。")
    native_ok = has_wv2 and run_native_window(url)
    try:
        if not native_ok:                           # fallback：Edge --app 无痕 → 默认浏览器
            proc = open_edge_app(url)
            if proc is not None:
                proc.wait()
            else:
                threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        # 窗口已关 → 立即收网。先停 HTTP server（释放端口），再硬退出进程。
        # 为什么用 os._exit 而非正常 return：chatllm 走常驻 libchatllm.dll，
        # Vulkan context 刻意永不释放（见 vulkan-dual-context-crash）；正常
        # 直译器关闭会去清理这颗 DLL → 可能卡死成僵尸进程、持续占显存。
        # os._exit 跳过 atexit/GC/DLL 卸载，由 OS 直接回收进程与 GPU 资源；
        # 同一瞬间 Job handle 关闭 → 残余子程序一并被终止。
        try:
            srv.stop()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()

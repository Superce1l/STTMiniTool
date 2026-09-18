"""ffmpeg_utils.py — ffmpeg 检测、音轨提取、一键下载

设计原则：
  1. 零磁碟开销的 pipe 提取（f32le → numpy），仅用于需要 ndarray 的场合
  2. 引擎层需要 Path 的场合，改写临时 WAV 再传入
  3. 下载 dialog 使用 after() 轮询，不阻塞 Tkinter 事件循环
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ── 视频副文件名集合 ──────────────────────────────────────────────────
VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".ts", ".m2ts", ".mpg", ".mpeg", ".m4v",
    ".vob", ".3gp", ".f4v", ".mxf",
}

# ── ffmpeg Windows 下载来源（BtbN essentials，约 55 MB）────────────
_FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/"
    "ffmpeg-master-latest-win64-gpl-essentials.zip"
)
# ZIP 内 ffmpeg.exe 的路径前缀（版本号不固定，只比对后缀）
_FFMPEG_ZIP_SUFFIX = "bin/ffmpeg.exe"


def is_video(path: Path) -> bool:
    """返回 True 代表此文件是视频（需要 ffmpeg 提取音轨）。"""
    return path.suffix.lower() in VIDEO_EXTS


def find_ffmpeg() -> Path | None:
    """按顺序搜索 ffmpeg：系统 PATH → App 目录 → 常见安装路径。"""
    # 1. 系统 PATH
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)

    # 2. App 目录下的 ffmpeg/ 子目录（EXE 模式 or 源码模式）
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent
    local = base / "ffmpeg" / "ffmpeg.exe"
    if local.exists():
        return local

    # 3. 常见 Windows 安装路径
    for candidate in [
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"),
    ]:
        if candidate.exists():
            return candidate

    return None


def get_default_ffmpeg_dest() -> Path:
    """返回下载后的存储目录（<app_dir>/ffmpeg/ffmpeg.exe）。"""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent
    return base / "ffmpeg"


# ── 音频提取 ──────────────────────────────────────────────────────────

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def extract_audio_to_wav(
    video_path: Path,
    out_wav: Path,
    ffmpeg_exe: Path,
    sr: int = 16000,
) -> None:
    """用 ffmpeg 把视频音轨提取成 16kHz mono PCM WAV。

    出现 ffmpeg 错误时抛出 RuntimeError（含 stderr 的最后一行）。
    """
    cmd = [
        str(ffmpeg_exe), "-y",
        "-i", str(video_path),
        "-vn",               # 丢弃图像流
        "-ar", str(sr),      # 取样率
        "-ac", "1",          # 单声道
        "-f", "wav",
        str(out_wav),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")
        last_line = next(
            (l.strip() for l in reversed(err.splitlines()) if l.strip()), "未知错误"
        )
        raise RuntimeError(f"ffmpeg 提取失败：{last_line}")


# ── 下载对话框 ────────────────────────────────────────────────────────

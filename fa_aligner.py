"""fa_aligner.py — chatllm 原生 ForcedAligner（后端无关，无需 PyTorch）

封装「字级时间轴对齐」：以 subprocess 调用 chatllm `main.exe`，给定
「参考文字 + 音频」即可输出字级毫秒时间轴（format json）。

设计重点：
  • 对齐本身与 ASR 后端无关——OpenVINO（CPU）与 chatllm（Vulkan GPU）
    两种引擎共享同一份逻辑，避免 _align_chunk 在两处分歧。
  • CPU 用 `-ngl 0`，GPU 用 `-ngl {device_id}:all`。
  • 失败、无输出时一律返回 []（调用端据此退回比例估算）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

# chatllm 来源文件名沿用上游拼字（foced 非 forced），与 downloader 一致。
FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

# Windows：隐藏子程序控制台窗口
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _startup_info():
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        return si
    return None


def _base_dir() -> Path:
    """App 根目录（冻结时为 EXE 旁，否则为源码目录）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def find_chatllm_dir() -> Path | None:
    """定位含 main.exe 的 chatllm 目录。

    与 app.py 的解析顺序一致：<app>/chatllm → chatllmtest 备援。
    """
    base = _base_dir()
    for cand in (
        base / "chatllm",
        base / "chatllmtest" / "chatllm_win_x64" / "bin",
        base,
    ):
        if (cand / "main.exe").exists():
            return cand
    return None


class ChatLLMAligner:
    """chatllm 原生 ForcedAligner 的轻量包装。

    用法
    ----
    fa = ChatLLMAligner(fa_dir=model_dir)        # CPU（-ngl 0）
    if fa.load(cb=status):                        # 验证 .bin 可用
        items = fa.align_chunk(wav, ref_text)     # [(word, start_s, end_s), ...]
    """

    FA_BIN_NAME = FA_BIN_NAME

    def __init__(
        self,
        fa_dir: str | Path | None,
        chatllm_dir: str | Path | None = None,
        n_gpu_layers: int = 0,
        device_id: int = 0,
    ):
        self._fa_dir = Path(fa_dir) if fa_dir else None
        self._chatllm_dir = (
            Path(chatllm_dir) if chatllm_dir else find_chatllm_dir()
        )
        self._n_gpu_layers = n_gpu_layers
        self._device_id = device_id
        self._fa_bin: Path | None = None
        self.ready = False

    # ── 检测 / 验证 ────────────────────────────────────────────────────
    def fa_bin_path(self) -> Path | None:
        """FA .bin 应在的位置（fa_dir / FA_BIN_NAME）。"""
        if self._fa_dir is None:
            return None
        return self._fa_dir / self.FA_BIN_NAME

    def load(self, cb: Callable[[str], None] | None = None) -> bool:
        """验证 chatllm FA .bin 可用。成功返回 True 并设置 ready/_fa_bin。

        实际对齐在 align_chunk() 以 subprocess 完成；此处仅以 `--show`
        确认模型确为 ForcedAligner，避免日后拿到错误模型。
        """
        def _s(msg: str):
            if cb:
                cb(msg)

        self.ready = False
        self._fa_bin = None

        fa_path = self.fa_bin_path()
        if fa_path is None or not fa_path.exists():
            return False
        if self._chatllm_dir is None:
            _s("⚠ 找不到 chatllm（main.exe），时间轴对齐改用比例估算")
            return False

        try:
            _s("验证时间轴对齐模型（chatllm ForcedAligner）…")
            exe = self._chatllm_dir / "main.exe"
            r = subprocess.run(
                [str(exe), "-m", str(fa_path), "-ngl", "0",
                 "--hide_banner", "--show"],
                capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=30, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW, startupinfo=_startup_info(),
            )
            out = r.stdout + r.stderr
            if "ForcedAligner" not in out:
                _s("⚠ FA 模型验证失败，改用比例估算")
                return False
            self._fa_bin = fa_path
            self.ready = True
            _s("时间轴对齐模型就绪（chatllm，无需 PyTorch）")
            return True
        except Exception as _e:
            _s(f"⚠ ForcedAligner 加载失败（{_e}），改用比例估算")
            return False

    # ── 对齐 ──────────────────────────────────────────────────────────
    def align_chunk(
        self,
        wav_path: str,
        ref_text: str,
        language: str = "Chinese",
    ) -> list[tuple[str, float, float]]:
        """对齐「参考文字 + 音频」，返回字级 [(word, start_s, end_s), ...]。

        失败或无输出时返回 []。
        """
        if not self._fa_bin or self._chatllm_dir is None:
            return []

        gpu_args = (["-ngl", f"{self._device_id}:all"]
                    if self._n_gpu_layers > 0 else ["-ngl", "0"])
        exe = self._chatllm_dir / "main.exe"
        prompt = f"{ref_text}{{{{audio:{wav_path}}}}}"
        cmd = [
            str(exe), "-m", str(self._fa_bin), *gpu_args, "--hide_banner",
            "--multimedia_file_tags", "{{", "}}",
            "-p", prompt,
            "--set", "format", "json",
            "--set", "language", language,
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=120, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW, startupinfo=_startup_info(),
            )
        except Exception:
            return []

        out = r.stdout + r.stderr
        # main.exe 末尾会附 timings 文字，需先切出 JSON 数组 [...]
        i = out.find("[")
        j = out.rfind("]")
        if i < 0 or j <= i:
            return []
        try:
            data = json.loads(out[i:j + 1])
        except (ValueError, json.JSONDecodeError):
            return []

        items: list[tuple[str, float, float]] = []
        for d in data:
            try:
                w = str(d["text"])
                s = float(d["start"]) / 1000.0   # 毫秒 → 秒
                e = float(d["end"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            if w.strip():
                items.append((w, s, e))
        return items

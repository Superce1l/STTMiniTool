"""
chatllm_engine.py — ChatLLM.cpp + Vulkan 推理后端

两种执行模式：
  1. DLL 模式（优先）：ctypes 直接调用 libchatllm.dll，模型常驻内存
     - 每 chunk 约 0.23s（GPU shader 暖机后），免去 subprocess 启动 overhead
  2. Subprocess 模式（后备）：每 chunk 启动 main.exe 子程序
     - 模型每次重载，但不需要 DLL

输出格式：language {lang}<asr_text>{transcription}
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ── 输出语言标志（由 app.py / app-gpu.py 切换时同步设置）──────────────
# True = 直接输出模型原始简体；False = 经 OpenCC 转为繁体
_output_simplified: bool = False

# 繁体输出时是否启用「简繁词汇转换」：
#   True  → OpenCC "s2twp"（含台湾惯用词）；False → "s2t"（仅字形）
_vocab_convert: bool = True


def _opencc_config() -> str:
    return "s2twp" if _vocab_convert else "s2t"

# ── 共享常数（与 app.py 保持同步）─────────────────────────────────────
SAMPLE_RATE   = 16000
VAD_CHUNK     = 512
VAD_THRESHOLD = 0.5
MAX_GROUP_SEC = 20
MAX_CHARS     = 20
MIN_SUB_SEC   = 0.6
GAP_SEC       = 0.08

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Windows 标志：建立子程序时不弹出控制台窗口（防止识别时画面闪烁）
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# STARTUPINFO：额外强制隐藏子程序窗口（搭配 CREATE_NO_WINDOW 双重保护）
# CREATE_NO_WINDOW 阻止 console 分配，STARTF_USESHOWWINDOW+SW_HIDE 隐藏主窗口
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = None
if sys.platform == "win32":
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUP_INFO.wShowWindow = 0  # SW_HIDE


def _to_path_bytes(path: "str | Path") -> bytes:
    """路径转 bytes，适合传给 Windows C DLL（ANSI API 兼容）。

    Windows C 函数库的 fopen() / LoadLibrary() 等 ANSI 函数期望系统码页
    （CP936/CP950）编码的路径，而 Python 默认 .encode() 是 UTF-8，
    两者在中文路径上不兼容。

    解法优先顺序：
      1. GetShortPathNameW → 8.3 短路径（纯 ASCII，任何 C API 都能处理）
      2. 若 8.3 短路径仍含非 ASCII → 改用 GetACP() 系统码页编码
      3. 最后回退 UTF-8
    """
    p = str(path)
    if sys.platform != "win32":
        return p.encode("utf-8")
    # 尝试 GetShortPathNameW 取得 ASCII 8.3 短路径
    try:
        n = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, n) > 0:
                try:
                    return buf.value.encode("ascii")
                except UnicodeEncodeError:
                    p = buf.value   # 短路径仍有非 ASCII → 继续往下
    except Exception:
        pass
    # 回退：系统 ANSI 码页（C fopen/CreateFileA 期望的编码）
    try:
        cp = ctypes.windll.kernel32.GetACP()
        return p.encode(f"cp{cp}")
    except (UnicodeEncodeError, LookupError):
        return p.encode("utf-8")


def _short_path_str(path: "str | Path") -> str:
    """返回 8.3 短路径字符串（尽量 ASCII，用于嵌入 DLL 消息字符串中）。"""
    p = str(path)
    if sys.platform != "win32":
        return p
    try:
        n = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, n) > 0:
                return buf.value
    except Exception:
        pass
    return p

# 语言名称 → ISO 639-1 语言代码（Qwen3-ASR 输出格式 "language {code}<asr_text>..."）
_LANG_CODE: dict[str, str] = {
    "Chinese":    "zh",
    "English":    "en",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Cantonese":  "yue",
    "French":     "fr",
    "German":     "de",
    "Spanish":    "es",
    "Portuguese": "pt",
    "Russian":    "ru",
    "Arabic":     "ar",
    "Thai":       "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Malay":      "ms",
    # 中文 UI 标签（OpenVINO 路线带来的标签兼容）
    "中文":  "zh",
    "英文":  "en",
    "日文":  "ja",
    "韩文":  "ko",
    "法文":  "fr",
    "德文":  "de",
    "西班牙文": "es",
    "葡萄牙文": "pt",
    "俄文":  "ru",
    "阿拉伯文": "ar",
    "泰文":  "th",
    "越南文": "vi",
}

SRT_DIR = BASE_DIR / "subtitles"


# ══════════════════════════════════════════════════════
# Vulkan 设备检测
# ══════════════════════════════════════════════════════

def probe_vulkan_devices(chatllm_dir: str | Path) -> dict:
    """执行 main.exe --show_devices 并返回「完整诊断」，供 UI 明确告知使用者。

    与 detect_vulkan_devices() 不同：本函数不吞错误，而是把整个探测过程
    的结果都带回来，让上层能分辨「真的没有 GPU」与「chatllm 列举失败
    （超时 / 崩溃 / 卡在核显）」这两种完全不同的情况。

    输出格式（每设备两行）：
      0: Vulkan - VulkanO (AMD Radeon(TM) Graphics)
         type: ACCEL
         memory free: 7957908736 B
      1: CPU - CPU (AMD Ryzen 5 9600X 6-Core Processor)
         type: CPU

    返回 dict：
      {
        "devices":     [...],     # 非 CPU 计算设备（供推理选用）
        "all_devices": [...],     # 全部列举到的设备（含 CPU/iGPU，供诊断）
        "raw":         str,       # --show_devices 原始输出（stdout+stderr）
        "error":       str|None,  # 例外/超时消息（None 表示正常结束）
        "exit_code":   int|None,  # main.exe 结束码
        "exe_found":   bool,      # 是否找得到 main.exe
      }
    每个设备 dict：{'id', 'name', 'vram_free', 'backend', 'is_cpu'}
    """
    exe = Path(chatllm_dir) / "main.exe"
    diag: dict = {
        "devices": [], "all_devices": [], "raw": "",
        "error": None, "exit_code": None, "exe_found": exe.exists(),
    }
    if not exe.exists():
        diag["error"] = f"找不到 main.exe：{exe}"
        return diag
    try:
        result = subprocess.run(
            [str(exe), "--show_devices"],
            capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=10,
            cwd=str(chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = result.stdout + result.stderr
        diag["raw"] = output
        diag["exit_code"] = result.returncode

        pending: list[dict] = []
        current: dict | None = None
        for line in output.splitlines():
            # 设备标头行：「0: Vulkan - VulkanO (AMD Radeon(TM) Graphics)」
            m = re.match(r"\s*(\d+):\s*(\S+)\s+-\s+\S+\s+\((.+)\)", line)
            if m:
                backend = m.group(2).upper()   # "VULKAN", "CPU", "METAL" …
                current = {
                    "id":        int(m.group(1)),
                    "name":      m.group(3).strip(),
                    "vram_free": 0,
                    "backend":   backend,
                    "is_cpu":    backend == "CPU",
                }
                pending.append(current)
            elif "memory free" in line and current is not None:
                mf = re.search(r"(\d+)\s*B", line)
                if mf:
                    current["vram_free"] = int(mf.group(1))

        diag["all_devices"] = pending
        diag["devices"] = [d for d in pending if not d["is_cpu"]]
    except subprocess.TimeoutExpired:
        diag["error"] = (
            "main.exe --show_devices 逾时（10 秒）。"
            "常见于笔电多 GPU 环境，Vulkan 初始化卡在内置显卡。"
        )
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {e}"
    return diag


def detect_vulkan_devices(chatllm_dir: str | Path) -> list[dict]:
    """执行 main.exe --show_devices，解析所有非 CPU 的计算设备（薄包装）。

    保留旧签名以维持向后兼容；完整诊断请改用 probe_vulkan_devices()。
    返回: [{'id', 'name', 'vram_free'}, ...]，失败时返回空清单。
    """
    diag = probe_vulkan_devices(chatllm_dir)
    return [
        {"id": d["id"], "name": d["name"], "vram_free": d["vram_free"]}
        for d in diag["devices"]
    ]


# ══════════════════════════════════════════════════════
# main.exe 子程序包装
# ══════════════════════════════════════════════════════

class _ChatLLMRunner:
    """
    以一次性模式执行 main.exe（每个音频 chunk 一次调用）。

    使用 `-mgl main N` 而非 `-ngl N`：
      - Transformer 放 GPU（Vulkan 加速）
      - 音频 encoder（FFmpeg + GGML audio）留在 CPU（Vulkan 不支持 audio encoder）

    输出格式：language {lang}<asr_text>{transcription}
    """

    def __init__(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        device_id:    int = 0,
    ):
        self._model_path   = Path(model_path).resolve()   # 必须解析为绝对路径
        self._chatllm_dir  = Path(chatllm_dir).resolve()
        self._n_gpu_layers = n_gpu_layers
        self._device_id    = device_id
        self._lock         = threading.Lock()

        exe = self._chatllm_dir / "main.exe"
        if not exe.exists():
            raise FileNotFoundError(f"main.exe 不存在：{exe}")
        self._exe = exe

        # 验证：执行 --show 确认模型可加载
        # 注意：用 -ngl 0 验证（不上 GPU），避免验证步骤占用显存
        r = subprocess.run(
            [str(exe), "-m", str(self._model_path), "-ngl", "0",
             "--hide_banner", "--show"],
            capture_output=True, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            timeout=30, cwd=str(self._chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = r.stdout + r.stderr
        if "Qwen3-ASR" not in output:
            raise RuntimeError(f"模型验证失败（rc={r.returncode}）：{output[:300]}")

    def transcribe(self, wav_path: str, sys_prompt: str | None = None) -> str:
        """送入 WAV 路径（绝对路径），返回转录文字。"""
        # -ngl {id}:all = 指定设备 id + 全部 layer（含 audio encoder Conv2D）放 GPU
        # 比 -mgl main N 快 2.7×（GPU 加速 audio encoder + Transformer 两段）
        gpu_args = ["-ngl", f"{self._device_id}:all"] if self._n_gpu_layers > 0 else ["-ngl", "0"]
        cmd = [
            str(self._exe),
            "-m",    str(self._model_path),
            *gpu_args,
            "--hide_banner",
            "-p",    wav_path,
        ]
        if sys_prompt:
            cmd += ["-s", sys_prompt]

        with self._lock:
            r = subprocess.run(
                cmd,
                capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=120, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUP_INFO,
            )
        output = r.stdout + r.stderr

        # 正常输出必含 <asr_text>；若缺失代表设备错误，立即中止，不返回垃圾字幕
        if "<asr_text>" not in output:
            preview = output.strip()[:300] or "(无输出)"
            raise RuntimeError(
                f"GPU 推理失败，未取得语音输出。\n"
                f"可能原因：设备不兼容、模型错误或内存不足。\n"
                f"chatllm 输出：{preview}"
            )
        text = output.split("<asr_text>", 1)[1].strip()
        deg = detect_degenerate_asr(text)
        if deg:
            raise RuntimeError(
                f"GPU 推理退化：{deg}。\n"
                f"此设备／核心可能不兼容，建议改用 CPU(OpenVINO) 核心。\n"
                f"模型输出：{text[:200]}"
            )
        return text


# ══════════════════════════════════════════════════════
# DLL 模式包装（ctypes，模型常驻内存）
# ══════════════════════════════════════════════════════

class _DLLASRRunner:
    """
    libchatllm.dll ctypes 包装，模型常驻 GPU 内存。

    每 chunk 调用 transcribe()：
      chatllm_restart → 写 WAV → chatllm_user_input("{{audio:path}}")
      第一次因 Vulkan shader 编译约 8s；后续 ~0.23s（43× 实时）
    """

    def __init__(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        device_id:    int = 0,
        cb=None,
    ):
        self._chatllm_dir = Path(chatllm_dir).resolve()
        self._lock        = threading.Lock()

        # ── 冻结窗口模式：预配置隐藏控制台（防止 DLL 闪出黑色窗口）────
        # 问题根源：--windowed PyInstaller EXE 没有控制台，
        # libchatllm.dll 的 MSVC C runtime 每次调用 chatllm_restart() /
        # chatllm_user_input() 写 stderr/stdout 时，发现 handle 无效，
        # 就会自行调用 AllocConsole() 建立控制台窗口（黑色窗口闪烁）。
        # 解法：在 LoadLibrary 前抢先 AllocConsole() 并立即隐藏，
        # 让 DLL C runtime 找到合法 handle，不再自行建立可见窗口。
        # source 模式（python app.py）从 cmd.exe 继承控制台，不触发此问题。
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            _k32 = ctypes.windll.kernel32
            _u32 = ctypes.windll.user32
            if not _k32.GetConsoleWindow():          # 目前无控制台
                if _k32.AllocConsole():              # 分配一个
                    _hwnd = _k32.GetConsoleWindow()
                    if _hwnd:
                        _u32.ShowWindow(_hwnd, 0)    # SW_HIDE：立即隐藏

        dll_path = self._chatllm_dir / "libchatllm.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"libchatllm.dll 不存在：{dll_path}")

        # ── DLL 相依解析修复（PyInstaller EXE 关键）──────────────
        # libchatllm.dll 内部用 plain LoadLibrary("ggml-vulkan.dll")
        # （不带 LOAD_LIBRARY_SEARCH_* 标志），走传统 DLL 搜索顺序：
        #   模块目录 → CWD → System32 → PATH
        # AddDllDirectory()（os.add_dll_directory）只影响有标志的 LoadLibraryEx，
        # 对传统搜索无效。EXE 的 CWD ≠ chatllm/，PATH 也不含 chatllm/，
        # 所以 ggml-vulkan.dll 等找不到 → DLL 初始化失败 → fallback subprocess。
        # 解法：暂时把 chatllm_dir 插到 PATH 最前面，chatllm_start() 后还原。
        _saved_path = os.environ.get("PATH", "")
        _chatllm_dir_str = str(self._chatllm_dir)
        os.environ["PATH"] = _chatllm_dir_str + os.pathsep + _saved_path

        os.add_dll_directory(_chatllm_dir_str)
        lib = ctypes.windll.LoadLibrary(str(dll_path))

        # ── 函数原型 ─────────────────────────────────────────────
        PRINTFUNC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
        ENDFUNC   = ctypes.WINFUNCTYPE(None, ctypes.c_void_p)

        lib.chatllm_append_init_param.argtypes = [ctypes.c_char_p]
        lib.chatllm_append_init_param.restype  = None
        lib.chatllm_init.argtypes              = []
        lib.chatllm_init.restype               = ctypes.c_int
        lib.chatllm_create.argtypes            = []
        lib.chatllm_create.restype             = ctypes.c_void_p
        lib.chatllm_append_param.argtypes      = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_append_param.restype       = None
        lib.chatllm_start.argtypes             = [ctypes.c_void_p, PRINTFUNC, ENDFUNC, ctypes.c_void_p]
        lib.chatllm_start.restype              = ctypes.c_int
        lib.chatllm_restart.argtypes           = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_restart.restype            = None
        lib.chatllm_user_input.argtypes        = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_user_input.restype         = ctypes.c_int

        self._lib       = lib
        self._PRINTFUNC = PRINTFUNC
        self._ENDFUNC   = ENDFUNC

        # ── chatllm 全域初始化（--ggml_dir 告知后端 DLL 位置）────
        # 必须用 ANSI/ASCII 兼容的路径；DLL 的 C 函数库用 ANSI fopen/LoadLibrary，
        # 若传 UTF-8 中文路径会找不到 ggml-*.dll，使用 _to_path_bytes() 解决此问题。
        lib.chatllm_append_init_param(b"--ggml_dir")
        lib.chatllm_append_init_param(_to_path_bytes(self._chatllm_dir))
        r = lib.chatllm_init()
        if r != 0:
            raise RuntimeError(f"chatllm_init() failed: {r}")

        # ── 建立 LLM object ───────────────────────────────────────
        chat = lib.chatllm_create()
        if not chat:
            raise RuntimeError("chatllm_create() returned NULL")
        self._chat = chat

        # ── 模型参数：必须加 --multimedia_file_tags {{ }} ─────────
        # 若缺少此参数，chat->history 的 mm_opening/closing 为空字符串，
        # Content::push_back() 会把 {{audio:path}} 当纯文字存储，不做音频解析。
        # 模型路径同样需要 ANSI/ASCII 兼容编码（GetShortPathNameW）。
        # 带入 device_id：「1:all」= 设备 1 全层 GPU，「0:all」= 默认设备 0
        # chatllm.cpp -ngl 语法：one_spec ::= [id:]spec
        gpu_arg = f"{device_id}:all" if n_gpu_layers > 0 else "0"
        model_path_bytes = _to_path_bytes(Path(model_path).resolve())
        for p_b in [
            b"-m", model_path_bytes,
            b"-ngl", gpu_arg.encode(),
            b"--multimedia_file_tags", b"{{", b"}}",
        ]:
            lib.chatllm_append_param(chat, p_b)

        # ── 回调（必须存为 instance attribute 防止 GC 回收）────────
        self._chunks: list[str] = []
        self._error:  str | None = None

        @PRINTFUNC
        def on_print(user_data, print_type, s_ptr):
            text = s_ptr.decode("utf-8", errors="replace") if s_ptr else ""
            if print_type == 0:       # PRINT_CHAT_CHUNK
                self._chunks.append(text)
            elif print_type == 2:     # PRINTLN_ERROR
                self._error = text

        @ENDFUNC
        def on_end(user_data):
            pass

        self._on_print = on_print
        self._on_end   = on_end

        # ── 加载模型（Vulkan 全层 GPU）───────────────────────────
        if cb:
            cb("加载 chatllm 模型（Vulkan GPU，-ngl all）…")
        r = lib.chatllm_start(chat, on_print, on_end, ctypes.c_void_p(0))
        # chatllm_start() 后 DLL 已完全初始化，相依 DLL 也已加载内存，可还原 PATH
        os.environ["PATH"] = _saved_path
        if r != 0:
            raise RuntimeError(f"chatllm_start() failed: {r}")

    def transcribe(self, wav_path: str, sys_prompt: str | None = None) -> str:
        """送入 WAV 路径（绝对路径），返回转录文字。"""
        # 取得 8.3 短路径（ASCII），避免中文路径无法被 DLL 的 C fopen 开启
        # 例：C:\Users\陈小明\AppData\Local\Temp\xxx.wav
        #   → C:\Users\CHEN~1\AppData\Local\Temp\xxx.wav（纯 ASCII）
        safe_path = _short_path_str(str(Path(wav_path).resolve()))
        fwd = safe_path.replace("\\", "/")
        # 若 _short_path_str 仍有非 ASCII（8.3 名称禁用），改用 ANSI 码页
        try:
            path_b = fwd.encode("ascii")
        except UnicodeEncodeError:
            cp = ctypes.windll.kernel32.GetACP() if sys.platform == "win32" else 65001
            path_b = fwd.encode(f"cp{cp}", errors="replace")
        msg = b"{{audio:" + path_b + b"}}"
        sys_bytes = sys_prompt.encode("utf-8") if sys_prompt else None

        with self._lock:
            self._lib.chatllm_restart(
                self._chat,
                ctypes.c_char_p(sys_bytes) if sys_bytes else ctypes.c_char_p(None),
            )
            self._chunks.clear()
            self._error = None

            r = self._lib.chatllm_user_input(self._chat, msg)

        if r != 0:
            raise RuntimeError(f"chatllm_user_input() failed: {r}")
        if self._error:
            raise RuntimeError(f"DLL 错误：{self._error}")

        full = "".join(self._chunks)
        if "<asr_text>" not in full:
            preview = full.strip()[:300] or "(无输出)"
            raise RuntimeError(
                f"GPU 推理失败，未取得语音输出。\n"
                f"可能原因：设备不兼容、模型错误或内存不足。\n"
                f"DLL 输出：{preview}"
            )
        text = full.split("<asr_text>", 1)[1].strip()
        deg = detect_degenerate_asr(text)
        if deg:
            raise RuntimeError(
                f"GPU 推理退化：{deg}。\n"
                f"此设备／核心可能不兼容，建议改用 CPU(OpenVINO) 核心。\n"
                f"模型输出：{text[:200]}"
            )
        return text


# ══════════════════════════════════════════════════════
# 辅助函数（从 app.py 复制，避免循环 import）
# ══════════════════════════════════════════════════════

def _detect_speech_groups(audio: np.ndarray, vad_sess, max_group_sec: int = MAX_GROUP_SEC,
                          stats: dict | None = None):
    h  = np.zeros((2, 1, 64), dtype=np.float32)
    c  = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    n  = len(audio) // VAD_CHUNK
    probs = []
    for i in range(n):
        chunk = audio[i*VAD_CHUNK:(i+1)*VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c, "sr": sr})
        probs.append(float(out[0, 0]))
    if stats is not None:
        stats["n_chunks"]  = len(probs)
        stats["max_prob"]  = max(probs) if probs else 0.0
        stats["mean_prob"] = (sum(probs) / len(probs)) if probs else 0.0
        stats["threshold"] = VAD_THRESHOLD
    if not probs:
        return [(0.0, len(audio) / SAMPLE_RATE, audio)]

    MIN_CH = 16; PAD = 5; MERGE = 16
    raw: list[tuple[int, int]] = []
    in_sp = False; s0 = 0
    for i, p in enumerate(probs):
        if p >= VAD_THRESHOLD and not in_sp:
            s0 = i; in_sp = True
        elif p < VAD_THRESHOLD and in_sp:
            if i - s0 >= MIN_CH:
                raw.append((max(0, s0-PAD), min(n, i+PAD)))
            in_sp = False
    if in_sp and n - s0 >= MIN_CH:
        raw.append((max(0, s0-PAD), n))
    if stats is not None:
        stats["n_segments"] = len(raw)
    if not raw:
        return []

    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= MERGE:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    mx_samp = max_group_sec * SAMPLE_RATE
    groups: list[tuple[int, int]] = []
    gs = merged[0][0] * VAD_CHUNK
    ge = merged[0][1] * VAD_CHUNK
    for seg in merged[1:]:
        s = seg[0] * VAD_CHUNK; e = seg[1] * VAD_CHUNK
        if e - gs > mx_samp:
            groups.append((gs, ge)); gs = s
        ge = e
    groups.append((gs, ge))

    result = []
    _tail = int(0.35 * SAMPLE_RATE)   # 多含尾段 0.35s：补捉 Silero 漏掉的句尾轻声字（的/了/吗）
    for gs, ge in groups:
        # 取完整语音段并多含一点尾巴（mel 会补零到固定长度）；旧版 floor 会丢尾段
        # 近 1 秒，句尾轻声字也常被 Silero 的 ge 排除，故补 0.35s 尾段。
        ch = audio[gs: min(len(audio), ge + _tail)].astype(np.float32)
        if len(ch) < SAMPLE_RATE // 2:      # 最小 0.5 秒
            continue
        result.append((gs / SAMPLE_RATE, ge / SAMPLE_RATE, ch))
    return result


def format_vad_diag(stats: dict) -> str:
    """把 _detect_speech_groups 的 stats 转成「为什么没检测到人声」的明确说明。

    区分三种情况：音频过短 / 全段机率过低（纯背景音或门槛太高）/ 有讯号但段落太短。
    """
    n    = stats.get("n_chunks", 0)
    mp   = stats.get("max_prob", 0.0)
    th   = stats.get("threshold", VAD_THRESHOLD)
    nseg = stats.get("n_segments", 0)
    if n == 0:
        return "⚠ 未检测到人声：音频过短或为空档。"
    if mp < th:
        return (f"⚠ 未检测到人声：全段人声机率偏低（最高 {mp:.2f} < 门槛 {th:.2f}）。"
                f"可能为纯背景音／音乐，或门槛过高 —— 可至『设置 → VAD 灵敏度』调低后重试。")
    if nseg == 0:
        return (f"⚠ 未检测到人声：检测到语音频号（最高 {mp:.2f}）但每段都过短（< 0.5 秒）未成段。")
    return "⚠ 未检测到人声：无有效语音分段。"


def detect_degenerate_asr(text: str) -> str | None:
    """检测 ASR 退化输出（模型复诵提示范本 / 高度重复），返回原因字符串；正常回 None。

    chatllm 在部分不兼容 GPU（常见 AMD Vulkan）上会吐出复诵 system prompt 的垃圾，
    例如 'language en<asr_text>[transcription]' 或同字无限重复。这类内容不该被当成
    字幕写出，需明确判为推理失败（issue #30）。
    """
    t = (text or "").strip()
    if not t:
        return None   # 空字符串交由上层「未检测到人声／跳过」逻辑处理
    low = t.lower()
    # 1) 残留 prompt 范本 token（最明确的退化讯号）
    for tok in ("<asr_text", "asr_text>", "[transcription]", "[asr_text]"):
        if tok in low:
            return "模型输出残留提示范本（疑似未正常识别）"
    # 2) 整段就是语言标记 'language xx'（exact，避免误判 'Language is power'）
    if re.fullmatch(r"language\s+[a-z]{2}", low):
        return "模型仅复诵语言标记，未产生转录内容"
    if low.count("language ") >= 3:
        return "模型反复复诵语言标记（疑似推理退化）"
    # 3) 单一字符高度重复
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 12 and max(Counter(compact).values()) / len(compact) >= 0.85:
        return "模型输出高度重复（疑似推理退化）"
    return None


def _split_to_lines(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"[。！？，、；：…—,.!?;:]+", text)
    lines = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > MAX_CHARS:
            lines.append(p[:MAX_CHARS]); p = p[MAX_CHARS:]
        lines.append(p)
    return [l for l in lines if l.strip()]


def _srt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _assign_ts(lines: list[str], g0: float, g1: float) -> list[tuple[float, float, str]]:
    if not lines:
        return []
    total = sum(len(l) for l in lines)
    if total == 0:
        return []
    dur = g1 - g0; res = []; cur = g0
    for i, line in enumerate(lines):
        end = cur + max(MIN_SUB_SEC, dur * len(line) / total)
        if i == len(lines) - 1:
            end = max(end, g1)
        res.append((cur, end, line))
        cur = end + GAP_SEC
    return res


# ══════════════════════════════════════════════════════
# ChatLLMASREngine
# ══════════════════════════════════════════════════════

class ChatLLMASREngine:
    """
    chatllm.cpp + Vulkan 推理后端。

    优先使用 DLL 模式（_DLLASRRunner）：
      - 模型常驻 GPU 内存，每 chunk ~0.23s（vs subprocess ~2.5s）
      - 若 libchatllm.dll 不存在，自动回退到 subprocess 模式（_ChatLLMRunner）

    与 ASREngine / ASREngine1p7B 界面兼容：
      - max_chunk_secs = 30
      - processor = None（chatllm 不使用 LightProcessor）
      - diar_engine    ← CPU ONNX，与后端无关，照常初始化
      - vad_sess       ← CPU ONNX
      - cc             ← opencc 简→繁转换
      - ready
    """

    max_chunk_secs = 30
    processor      = None   # chatllm 不用 LightProcessor，UI 检测此为 None

    def __init__(self):
        self.ready       = False
        self.vad_sess    = None
        self.diar_engine = None
        self.cc          = None
        self._runner: _DLLASRRunner | _ChatLLMRunner | None = None
        self._use_dll    = False   # 记录目前使用哪种模式
        self.aligner     = None   # 兼容标志（chatllm FA 就绪时为 True）
        self.use_aligner = False  # 是否启用时间轴对齐
        self._fa         = None   # ChatLLMAligner（与 OpenVINO 引擎共享）
        self._fa_bin     = None   # FA .bin 路径（就绪时非 None）

    # ── 加载 ──────────────────────────────────────────────────────────

    def load(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        device_id:    int = 0,
        cb=None,
    ):
        """从背景线程调用。cb(msg) 更新 UI 状态。"""
        import onnxruntime as ort

        def _s(msg):
            if cb:
                cb(msg)

        self._model_path   = Path(model_path)
        self._chatllm_dir  = Path(chatllm_dir)
        self._n_gpu_layers = n_gpu_layers
        self._device_id    = device_id   # FA 对齐的 -ngl {id}:all 需要

        # ── VAD ──────────────────────────────────────────────────────
        _s("加载 VAD 模型…")
        vad_candidates = [
            BASE_DIR / "ov_models" / "silero_vad_v4.onnx",
            BASE_DIR / "GPUModel"  / "silero_vad_v4.onnx",
        ]
        # PyInstaller onedir 模式：bundled 资源在 _internal/（sys._MEIPASS）
        import sys as _sys
        if getattr(_sys, "frozen", False) and hasattr(_sys, "_MEIPASS"):
            vad_candidates.insert(0, Path(_sys._MEIPASS) / "ov_models" / "silero_vad_v4.onnx")
        vad_path = next((p for p in vad_candidates if p.exists()), None)
        if vad_path is None:
            raise FileNotFoundError("找不到 silero_vad_v4.onnx")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

        # ── 说话者分离（CPU ONNX，与后端无关）───────────────────────
        _s("加载说话者分离模型…")
        try:
            from diarize import DiarizationEngine
            diar_candidates = [
                BASE_DIR / "ov_models" / "diarization",
                BASE_DIR / "GPUModel"  / "diarization",
            ]
            diar_dir = next((p for p in diar_candidates if p.exists()), None)
            if diar_dir:
                eng = DiarizationEngine(diar_dir)
                self.diar_engine = eng if eng.ready else None
        except Exception:
            self.diar_engine = None

        # ── OpenCC 简→繁转换 ──────────────────────────────────────
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            self.cc = None

        # ── 验证路径 ─────────────────────────────────────────────────
        if not self._model_path.exists():
            raise FileNotFoundError(f"模型不存在：{self._model_path}")

        # ── 建立 Runner：优先 DLL，后备 subprocess ───────────────
        dll_path = self._chatllm_dir / "libchatllm.dll"
        if dll_path.exists():
            try:
                _s("加载 chatllm 模型（DLL 模式，Vulkan 全层 GPU）…")
                self._runner = _DLLASRRunner(
                    model_path   = model_path,
                    chatllm_dir  = chatllm_dir,
                    n_gpu_layers = n_gpu_layers,
                    device_id    = device_id,
                    cb           = cb,
                )
                self._use_dll = True
                self.ready = True
                _s("ChatLLM DLL 加载完成（模型常驻 GPU，每 chunk ~0.23s）")

                # ── ForcedAligner（可选，CPU PyTorch，不需 CUDA）────────
                self._load_aligner(cb=cb)
                return
            except Exception as e:
                _s(f"DLL 模式失败（{e}），改用 subprocess 模式…")

        _s("验证 chatllm 模型（subprocess 模式）…")
        self._runner = _ChatLLMRunner(
            model_path   = model_path,
            chatllm_dir  = chatllm_dir,
            n_gpu_layers = n_gpu_layers,
            device_id    = device_id,
        )
        self._use_dll = False
        self.ready = True
        _s("ChatLLM 加载完成（subprocess 模式，Vulkan GPU）")

        # ── ForcedAligner（可选，CPU PyTorch，不需 CUDA）──────────────
        self._load_aligner(cb=cb)

    # ── 单段转录 ──────────────────────────────────────────────────────

    def rebuild_cc(self):
        """依目前的词汇转换标志重建 OpenCC 转换器（免重新加载模型）。"""
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            pass

    def transcribe(
        self,
        audio:      np.ndarray,
        sr:         int = SAMPLE_RATE,
        language:   str | None = None,
        context:    str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """16kHz float32 → 转录文字。"""
        import soundfile as sf

        # 语言 → system prompt
        # Qwen3-ASR 输出格式：language {code}<asr_text>{text}
        # 通过 sys_prompt 明确指定语言代码，引导模型用正确语言输出。
        sys_prompt: str | None = None
        if language and language != "自动检测":
            code = _LANG_CODE.get(language, language.lower()[:2])
            sys_prompt = (
                f"The audio language is {language}. "
                f"Transcribe it and output strictly in this format: "
                f"language {code}<asr_text>[transcription]. "
                f"Output only {language} text after <asr_text>, no translation."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
            text = self._runner.transcribe(tmp_path, sys_prompt=sys_prompt)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # OpenCC 简→繁转换（模型默认输出简体中文；简体模式则跳过）
        if self.cc and text and not _output_simplified:
            text = self.cc.convert(text)

        return text

    # ── ForcedAligner 加载 ────────────────────────────────────────────

    # FA chatllm 模型文件名（与 ASR .bin 同文件夹）
    FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

    def _load_aligner(self, cb=None):
        """检测 chatllm 版 ForcedAligner .bin（委派共享 fa_aligner，无需 torch）。

        与 OpenVINO 引擎共享 fa_aligner.ChatLLMAligner：给「参考文字 + 音频」
        即可输出字级毫秒时间轴。GPU 用 -ngl {id}:all、CPU 用 -ngl 0。
        缺 .bin 或验证失败时静默退回比例估算。
        """
        from fa_aligner import ChatLLMAligner
        self.aligner     = None
        self.use_aligner = False
        self._fa_bin     = None
        # FA .bin 与 ASR .bin 放同一文件夹（尊重使用者指定的 model_dir）
        # ── FA 强制 CPU 子程序（n_gpu_layers=0）──────────────────────────
        # 致命关键：ASR 走常驻 libchatllm.dll，Vulkan context 永不释放。
        # 若 FA 也用 -ngl {id}:all 上 GPU，main.exe 子程序会在 DLL 仍持有
        # context 时建立「第二个 Vulkan context」→ 同卡双 context 并存 →
        # 驱动 TDR/reset → 整机死机（1.0.8 起的潜伏 bug，见 vulkan-dual-context-crash）。
        # FA 是 0.6B，CPU 对齐本就够快；改 CPU 后全程仅一个 GPU context（DLL）。
        self._fa = ChatLLMAligner(
            fa_dir       = self._model_path.parent,
            chatllm_dir  = self._chatllm_dir,
            n_gpu_layers = 0,
            device_id    = self._device_id,
        )
        if self._fa.load(cb=cb):
            self._fa_bin     = self._fa._fa_bin
            self.aligner     = True   # 兼容标志（沿用 use_aligner 判断流程）
            self.use_aligner = True

    def _align_chunk(
        self,
        wav_path:  str,
        ref_text:  str,
        language:  str = "Chinese",
    ) -> list[tuple[str, float, float]]:
        """委派 chatllm FA 对齐，返回字级 [(word, start_s, end_s), ...]（失败回 []）。"""
        if self._fa is None or not self._fa_bin:
            return []
        return self._fa.align_chunk(wav_path, ref_text, language)

    # ── chunk 长度限制 ─────────────────────────────────────────────────

    def _enforce_chunk_limit(
        self,
        groups: list[tuple[float, float, np.ndarray, "str | None"]],
    ) -> list[tuple[float, float, np.ndarray, "str | None"]]:
        max_samples = self.max_chunk_secs * SAMPLE_RATE
        result = []
        for t0, t1, chunk, spk in groups:
            if len(chunk) <= max_samples:
                result.append((t0, t1, chunk, spk))
            else:
                pos = 0
                while pos < len(chunk):
                    piece = chunk[pos: pos + max_samples]
                    if len(piece) < SAMPLE_RATE:
                        break
                    piece_t0 = t0 + pos / SAMPLE_RATE
                    piece_t1 = min(t1, piece_t0 + len(piece) / SAMPLE_RATE)
                    result.append((piece_t0, piece_t1, piece, spk))
                    pos += max_samples
        return result

    # ── 音频转 SRT ─────────────────────────────────────────────────────

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language:   str | None = None,
        context:    str | None = None,
        diarize:    bool = False,
        n_speakers: int | None = None,
        original_path: Path | None = None,
        out_format: str | None = None,
    ) -> Path | None:
        from audio_io import load_audio_16k_mono
        from subtitle_lines import write_transcript

        audio, _ = load_audio_16k_mono(audio_path, SAMPLE_RATE)
        self._last_segments_rich = None   # 本次字级结果（卡拉OK侧通道）

        # ── 分段策略：说话者分离 vs 传统 VAD（与 ASREngine 一致）────
        use_diar = diarize and self.diar_engine is not None and self.diar_engine.ready
        if use_diar:
            diar_segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
            if not diar_segs:
                return None
            groups_spk = [
                (t0, t1,
                 audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)],
                 spk)
                for t0, t1, spk in diar_segs
            ]
        else:
            if self.vad_sess is None:
                self._last_vad_diag = "⚠ VAD 模型（silero_vad）未加载，无法分段"
                if progress_cb:
                    progress_cb(0, 1, self._last_vad_diag)
                return None
            _vad_stats: dict = {}
            vad_groups = _detect_speech_groups(
                audio, self.vad_sess, self.max_chunk_secs, stats=_vad_stats)
            if not vad_groups:
                self._last_vad_diag = format_vad_diag(_vad_stats)
                if progress_cb:
                    progress_cb(0, 1, self._last_vad_diag)
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        groups_spk = self._enforce_chunk_limit(groups_spk)

        # ── 导入 chatllm 版断句函数（避免循环 import，延迟导入）─────────
        _ts_fn = None
        if self.use_aligner and self._fa_bin is not None:
            try:
                from app import _ts_chatllm_to_subtitle_lines
                _ts_fn = _ts_chatllm_to_subtitle_lines
            except ImportError:
                pass

        all_subs: list[tuple[float, float, str, str | None]] = []
        all_rich: list[dict] = []    # 与 all_subs 对应的字级结构（卡拉OK用）
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total, f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")

            # ── 转录（取原始简体输出，对齐后再繁化）─────────────────────
            import soundfile as sf
            sys_prompt: str | None = None
            if language and language != "自动检测":
                code = _LANG_CODE.get(language, language.lower()[:2])
                sys_prompt = (
                    f"The audio language is {language}. "
                    f"Transcribe it and output strictly in this format: "
                    f"language {code}<asr_text>[transcription]. "
                    f"Output only {language} text after <asr_text>, no translation."
                )

            import tempfile as _tempfile
            with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            aligned  = False
            raw_text = ""
            try:
                sf.write(tmp_path, chunk, SAMPLE_RATE, subtype="PCM_16")
                raw_text = self._runner.transcribe(tmp_path, sys_prompt=sys_prompt)

                if not raw_text:
                    continue

                # ── ForcedAligner 精确时间轴对齐（chatllm .bin，无需 torch）──
                #    重用同一个临时 wav，避免重复写档。
                if self.use_aligner and self._fa_bin is not None and _ts_fn is not None:
                    try:
                        align_lang = (language
                                      if (language and language != "自动检测")
                                      else "Chinese")
                        ts_items = self._align_chunk(tmp_path, raw_text, align_lang)
                        if ts_items:
                            subs = _ts_fn(
                                ts_items, raw_text, g0, spk,
                                self.cc, _output_simplified, with_words=True,
                            )
                            if subs:
                                for (s, e, t, sp, words) in subs:
                                    all_subs.append((s, e, t, sp))
                                    all_rich.append({"start": s, "end": e, "text": t,
                                                     "speaker": sp, "words": words})
                                aligned = True
                    except Exception:
                        aligned = False
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if not raw_text:
                continue

            if not aligned:
                # ── 比例估算 Fallback ──────────────────────────────────────
                text = raw_text
                if self.cc and not _output_simplified:
                    text = self.cc.convert(raw_text)
                lines = _split_to_lines(text)
                for s, e, line in _assign_ts(lines, g0, g1):
                    all_subs.append((s, e, line, spk))
                    all_rich.append({"start": s, "end": e, "text": line,
                                     "speaker": spk, "words": []})   # 无对齐 → 前端内插

        if not all_subs:
            return None

        # 字级结果挂上实例（卡拉OK逐字高亮）
        self._last_segments_rich = all_rich

        if progress_cb:
            progress_cb(total, total, "写入字幕…")

        # 以原始文件的目录与文件名输出（视频抽音轨时 audio_path 是临时路径）
        # 共享写出层：依全域设置（或 out_format 覆盖）产出 .srt 或 .txt。
        ref = original_path if original_path is not None else audio_path
        return write_transcript(ref, all_subs, out_format)

    def __del__(self):
        pass   # DLL runner 由 GC 自然回收（ctypes callback 会被 GC 清理）


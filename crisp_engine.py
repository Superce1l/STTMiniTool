"""
crisp_engine.py — CrispASR (ggml / Vulkan) Whisper 推理后端

定位：把 CrispASR 的 whisper backend 当作本项目的第三个推理引擎（与
ASREngine[OpenVINO] 并列），用来加载 Whisper
类模型（默认 OpenAI Whisper 官方 GGML）。

设计重点：
  • CrispASR 是 whisper.cpp fork，crispasr.exe 自己会输出 SRT（含分段时间轴），
    因此本引擎不需重造 VAD/chunk/FA，本质是「调用 crispasr.exe → 读 SRT →
    写回」。
  • GPU 后端由 load(gpu_backend=) 决定，对应「使用者选的 CrispASR 加速版本」
    （downloader.crispasr_gpu_backend）：
      - vulkan（默认）：Windows 通吃 Intel/AMD 核显与 NVIDIA 独显，体积最小。
      - cuda：NVIDIA 专用，需下载 CUDA build（0.8.32 起在 Blackwell 实测正常；
        早期 0.7.1 的 fattn crash 为 nemotron 后端专属问题）。
      - None：CPU build，改挂 -ng（该 build 没有 ggml-vulkan/cuda DLL）。
  • 界面对齐 app.py 既有引擎契约：load / ready / transcribe / process_file /
    processor / diar_engine / use_aligner / aligner / _lock。

对外界面（app.py 的 _load_models[crispasr 分支] 会这样调用）：
    eng = CrispWhisperEngine()
    eng.load(model_path=..., crispasr_dir=..., device_id=0, cb=set_status)
    srt = eng.process_file(audio_path, progress_cb=..., language=..., ...)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path

import numpy as np

# 与 Qwen 路径统一的字幕分行（字级时间轴 → 字幕行，全引擎共享）
from subtitle_lines import (_ts_chatllm_to_subtitle_lines, _srt_ts, write_transcript,
                            _ZH_CLAUSE_END, _EN_SENT_END)
from slicing import assign_speaker_by_overlap as _assign_speaker_by_overlap
from engine_types import (TranscribeResult, SAMPLE_RATE, CREATE_NO_WINDOW,
                          base_dir, startup_info)

# 中文/英文标点集合（与断句器一致）：Qwen 模式下标点只当切点、不进 items
_PUNCT = _ZH_CLAUSE_END | _EN_SENT_END

# ── 共享常数（SAMPLE_RATE 等收敛于 common.py；保留模块级别名供旧引用）──
BASE_DIR = base_dir()

# Windows：隐藏子程序控制台窗口（避免识别时画面闪烁）
_CREATE_NO_WINDOW = CREATE_NO_WINDOW
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = startup_info()

# 实时/单段转录子进程超时：单段音频通常 <30s，crispasr 处理应在几分钟内完成。
# 超时即认为挂死（Vulkan 初始化卡死等），杀进程并抛错，避免永久持有 self._lock。
_RT_PROC_TIMEOUT_SECS = 600
# 流式模式收尾宽限：正常退出等待上限（秒）；超时强杀。
_PROC_SHUTDOWN_GRACE_SECS = 30
# 实时分段：常驻 server（crispasr --server）的三条时间线。
# 启动上限给足：首次要加载模型上显存（qwen3 Q8 2.5GB），慢盘/小显存机器
# 可能十几秒；超时即判失败，本进程后续改走一次性 spawn，不再反复等待。
_SERVER_START_TIMEOUT_SECS = 180.0
# 单段请求上限（录音段通常 1–30s；推理本身 ~0.2s，余量给解码/排队）。
_SERVER_REQ_TIMEOUT_SECS = 300.0
# 停服务宽限：先 terminate，超时才 kill（Vulkan 下 take 一点时间收尾）。
_SERVER_STOP_GRACE_SECS = 5.0

# 流式模式输出停滞上限（秒）：超过此时长没有任何新输出行才判挂死。
# 注意不能太小：qwen3 后端可能先做 LID 初始化与音频扫描（几十秒无输出），
# 且上游明确不出 progress 行（-pp 仅 whisper 后端有效），进度只能靠
# 「行活跃度」近似回报，停滞看门狗是唯一的挂死防线。
_PROC_STALL_TIMEOUT_SECS = 180

# 默认 crispasr 目录与模型文件名
_DEFAULT_CRISPASR_DIR = BASE_DIR / "crispasr"
_DEFAULT_MODEL_NAME   = "ggml-base.bin"   # OpenAI Whisper Base（ggerganov/whisper.cpp）
# qwen3 ForcedAligner GGUF（FA 对齐器）文件名样式；与 app.py 的 _DEFAULT_MODEL_DIR
# 同口径放在模型目录，crispasr_dir 只作旧版兼容备援（issue #7）。
_ALIGNER_GLOB = "qwen3-forced-aligner-0.6b-*.gguf"
_DEFAULT_OV_MODEL_DIR = BASE_DIR / "ov_models"


def _find_aligner_gguf(*dirs: Path) -> Path | None:
    """依序在给定目录查找 qwen3 ForcedAligner GGUF（取第一个符合者）。

    调用端给的顺序是「模型目录 → crispasr 目录」：新版下载到模型目录，
    旧版留在核心目录的档仍能找到，不必重新下载。
    """
    seen: set[str] = set()
    for d in dirs:
        if not d:
            continue
        d = Path(d)
        try:
            key = str(d.resolve())
        except OSError:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        for g in sorted(d.glob(_ALIGNER_GLOB)):
            if g.is_file():
                return g
    return None


def _infer_backend(model_path: Path | str) -> str:
    """依模型文件名推断 crispasr 后端（--backend）。

    CrispASR 是多后端 runtime（whisper / qwen3 / parakeet …），同一支
    crispasr.exe 靠 `--backend` 切换模型架构。OpenAI Whisper GGML 属 whisper
    架构（默认，免标志）；Qwen3-ASR GGUF 文件名含 'qwen3' → 走 qwen3 后端，1.7B
    用 'qwen3-1.7b'。判断错误时 crispasr 会自报模型不符，不致默默产生垃圾。
    """
    name = Path(model_path).name.lower()
    if "qwen3" in name:
        return "qwen3-1.7b" if "1.7b" in name else "qwen3"
    return "whisper"

# app.py 传入的 language（如 "Chinese" / "自动检测"）→ whisper 语言代码
_LANG_MAP = {
    "自动检测": None, "auto": None, "": None,
    "Chinese": "zh", "中文": "zh", "Mandarin": "zh",
    "English": "en", "Japanese": "ja", "Korean": "ko",
    "Cantonese": "yue", "French": "fr", "German": "de",
    "Spanish": "es", "Portuguese": "pt", "Russian": "ru",
    "Arabic": "ar", "Thai": "th", "Vietnamese": "vi",
    "Indonesian": "id", "Malay": "ms",
}

# 本引擎对外宣告的常用语言清单（app.py 在 processor 为 None 时改用此清单）
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "Cantonese",
    "French", "German", "Spanish", "Portuguese", "Russian",
    "Arabic", "Thai", "Vietnamese", "Indonesian", "Malay",
]

# 以空白分词、且分行器能以空格拼回词界的语言：whisper 挂 -sow 按词切段
# 才安全（见 _build_cmd）。韩国语同样以空格分词（上游 2.1.0 修正），故纳入；
# 中文／粤语／日文／泰文不以空白分词，不在此列。
_SPACE_DELIMITED = frozenset({"en", "fr", "de", "es", "pt", "id", "ms", "vi", "ko"})


def _default_threads() -> int:
    """crispasr `-t` 的预设值。

    不给 -t 时 crispasr 只用 min(4, 核心数) 个线程，CPU 推理（CPU 版，或 GPU 后端
    载入失败而静默退回 CPU）明显吃亏；GPU 推理几乎不受线程数影响。实测（CPU
    推理 230s 音频）：4 核 8 线程笔记本 4→8 线程 20.9s→16.2s（超线程有益）；
    双路 128 线程服务器 16 线程最快，32／64 线程反而退回（同步开销）。故取逻辑
    核心数、上限 16。
    """
    return max(1, min(16, os.cpu_count() or 4))


def _find_exe(crispasr_dir: Path) -> Path | None:
    """在 crispasr 目录（含一层子文件夹）查找 crispasr.exe。"""
    direct = crispasr_dir / "crispasr.exe"
    if direct.exists():
        return direct
    for sub in crispasr_dir.glob("**/crispasr.exe"):
        return sub
    return None


def probe_crispasr_devices(crispasr_dir: str | Path) -> dict:
    """执行 `crispasr.exe --diagnostics` 并解析计算设备清单（含 GPU 内存）。

    走 CrispASR 自带的 `--diagnostics`（会列举 ggml 后端与设备后立即退出，
    不需模型/音频）。

    解析目标是 diagnostics 输出中的「registered devices」区块，例如：
        registered devices : 2
          [0] gpu    name=Vulkan0 desc=NVIDIA GeForce RTX 5090 mem=31418/32186 MiB id=0000:01:00.0
          [1] cpu    name=CPU desc=13th Gen Intel(R) Core(TM) i5-13500 mem=79978/98045 MiB id=?
    每行 → {'id','name','vram_free'(bytes),'backend','is_cpu'}；mem 为 free/total MiB。

    返回设备列表 dict：
      {devices(非CPU), all_devices(全部), raw, error, exit_code, exe_found}
    """
    crispasr_dir = Path(crispasr_dir)
    exe = _find_exe(crispasr_dir)
    diag: dict = {
        "devices": [], "all_devices": [], "raw": "",
        "error": None, "exit_code": None, "exe_found": bool(exe),
    }
    if not exe:
        diag["error"] = f"找不到 crispasr.exe：{crispasr_dir}"
        return diag
    try:
        result = subprocess.run(
            [str(exe), "--diagnostics"],
            capture_output=True, stdin=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace", timeout=15,
            cwd=str(exe.parent),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = (result.stdout or "") + (result.stderr or "")
        diag["raw"] = output
        diag["exit_code"] = result.returncode

        # 「registered devices」逐行：[idx] <type> name=.. desc=<名称> mem=free/total MiB id=..
        dev_re = re.compile(
            r"^\s*\[(\d+)\]\s+(\w+)\s+name=(\S+)\s+desc=(.+?)\s+mem=(\d+)/(\d+)\s*MiB",
            re.IGNORECASE)
        pending: list[dict] = []
        for line in output.splitlines():
            m = dev_re.match(line)
            if not m:
                continue
            kind = m.group(2).lower()            # gpu / cpu
            is_cpu = kind == "cpu"
            # 后端标签依设备名（name=CUDA0 → CUDA、Vulkan0 → VULKAN）推导，
            # 不再硬编码 VULKAN——CUDA 构建的 GPU 也曾被误标。
            tag = re.sub(r"[^A-Za-z]", "", m.group(3)).upper()
            pending.append({
                "id":        int(m.group(1)),
                "name":      m.group(4).strip(),                 # 人类可读名称
                "vram_free": int(m.group(5)) * 1024 * 1024,      # MiB → bytes
                "backend":   tag or ("CPU" if is_cpu else "GPU"),
                "is_cpu":    is_cpu,
            })

        # 备援：若无 registered devices 区块，改解析 ggml_vulkan 列举行
        #   「ggml_vulkan: 0 = NVIDIA GeForce RTX 5090 (NVIDIA) | uma: 0 | ...」
        if not pending:
            vk_re = re.compile(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\((.+?)\)\s*\|")
            for line in output.splitlines():
                m = vk_re.match(line)
                if m:
                    pending.append({
                        "id": int(m.group(1)), "name": m.group(2).strip(),
                        "vram_free": 0, "backend": "VULKAN", "is_cpu": False,
                    })

        diag["all_devices"] = pending
        diag["devices"] = [d for d in pending if not d["is_cpu"]]
    except subprocess.TimeoutExpired:
        diag["error"] = ("crispasr.exe --diagnostics 逾时（15 秒）。"
                         "Vulkan 初始化可能卡在内置显卡。")
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {e}"
    return diag


# ── 与 CTC 对齐器不兼容的模型 ──────────────────────────────────────────
#   `qwen3-forced-aligner-0.6b` 是「把 vocabulary head 换成 5000 类 timestamp
#   head」的对齐器，其类别表是配著 Qwen3-ASR 的**简体**输出建立的。我们的管线
#   一直没事，是因为既有模型都吐简体、繁化发生在对齐**之后** —— 对齐器
#   永远只看到简体。
#
#   TEA-ASR 打破了这个隐含假设：它**原生输出繁体**，`询`／`会`／`湾` 这类字落在
#   对齐器的类别表外，对齐器就输出未映射的类别 ID，crispasr 直接把它渲染成私用
#   区字符（实测 `U+E000 + class_id`）→ 字幕出现乱码，时间轴也跟着对歪。
#   故这类模型一律不挂 `-falign`，改用后端自带的时间戳（较粗但正确）。
_FA_INCOMPATIBLE = ("tea-asr", "tea_asr")


def _fa_incompatible(model_path) -> bool:
    name = Path(model_path).name.lower()
    return any(k in name for k in _FA_INCOMPATIBLE)


def _lid_model_path(crispasr_dir: Path | None = None) -> Path | None:
    """找本地的 whisper ggml 给 qwen3 后端当 LID 模型（--lid-model）。

    qwen3 后端启动要做语言识别，默认找 ggml-tiny.bin；本目录没有时 crispasr
    会联网下载（失败要等 ~44 秒超时才继续，离线机器更久）。crispasr 目录里
    通常已有 whisper ggml（base 随包、medium 等按需下载），任选其一即可。
    找不到回 None（不挂标志，维持 crispasr 默认行为）。
    """
    d = Path(crispasr_dir) if crispasr_dir else _DEFAULT_CRISPASR_DIR
    # 优先 tiny（体积最小、加载最快），否则退回任何 ggml whisper 模型
    for cand in sorted(d.glob("ggml-*.bin"),
                       key=lambda p: (("tiny" not in p.name), p.stat().st_size)):
        if cand.is_file() and cand.stat().st_size > 10 * 1024 * 1024:
            return cand
    return None


# ── 实时分段：常驻 server 的薄客户端 ──────────────────────────────────
def _free_local_port() -> int:
    """取一个当前空闲的本地端口（crispasr --server 绑定用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# 句末标点后切行（录音页一句一行追加显示）。不切逗号，避免碎片化。
_RT_SENT_END = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+")


def _split_realtime_lines(text: str) -> list[str]:
    """整段识别文字 → 一句一行（空行丢弃）。"""
    t = (text or "").strip()
    if not t:
        return []
    return [p.strip() for p in _RT_SENT_END.split(t) if p and p.strip()]


def _realtime_segments_from_api(obj: dict) -> list[dict]:
    """server 的 JSON → segments_rich（[{'start','end','text','speaker','words'}]）。

    server 的 verbose_json 已带分段时间轴；没有 segments 时退回把 text 切句，
    时间轴留 0（录音页本来就用录音时钟盖时间，不依赖后端时间轴）。
    """
    out: list[dict] = []
    for s in (obj.get("segments") or []):
        try:
            start = float(s.get("start") or 0.0)
            end = float(s.get("end") or 0.0)
        except (TypeError, ValueError):
            start = end = 0.0
        for line in _split_realtime_lines(s.get("text") or ""):
            out.append({"start": start, "end": end, "text": line,
                        "speaker": None, "words": []})
    if out:
        return out
    return [{"start": 0.0, "end": 0.0, "text": line, "speaker": None, "words": []}
            for line in _split_realtime_lines(obj.get("text") or "")]


def _warn(msg: str) -> None:
    """尽力记日志：GUI 里走 applog；单测/CLI 无 applog 时退 stderr。"""
    try:
        from applog import log_warn
        log_warn(msg)
    except Exception:
        print(f"[crisp_engine] {msg}", file=sys.stderr)


class _CrispServer:
    """crispasr.exe `--server` 常驻进程的托管（实时分段专用）。

    为什么需要：录音页每说完一句就切一段上传，旧路径每段 spawn 一次
    crispasr.exe、模型每次重新上显存。实测（GTX 1660 SUPER / 核心 0.8.54）
    固定开销 whisper base ≈0.6s、qwen3 Q8 ≈2.35s，占出字延迟 ~90%；
    `--server` 只在启动时加载一次模型，POST /v1/audio/transcriptions 复用，
    单段回落到 ~0.2s（实测 0.17–0.23s）。

    生命周期归引擎：录音开始按需起，录音结束宽限若干秒再停（收尾段还要
    上传），换模型／批量转录前立即停——否则常驻模型与批量转录的一次性进程
    会各占一份显存（4GB 卡直接 OOM）。
    """

    def __init__(self, argv: list[str], cwd: str | Path):
        self._argv = list(argv)
        self._cwd = str(cwd)
        # RLock：start() 持锁等 /health，失败路径要调 stop()（同一把锁）——
        # 用普通 Lock 会在「起不来」这条路上自锁死，恰好让退路永远走不到。
        self._lock = threading.RLock()
        self._proc: "subprocess.Popen[str] | None" = None
        self._port = 0
        self._tail: deque = deque(maxlen=80)

    # ── 状态 ──
    @property
    def port(self) -> int:
        return self._port

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def tail(self, n: int = 6) -> str:
        return "　|　".join(list(self._tail)[-n:]) or "（无输出）"

    # ── 子进程输出泵：只留最近几行，失败时拿来报原因 ──
    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                t = line.rstrip()
                if t:
                    self._tail.append(t)
        except Exception:
            pass

    # ── 启动 / 停止 ──
    def start(self, timeout: float = _SERVER_START_TIMEOUT_SECS) -> None:
        with self._lock:
            if self.alive():
                return
            port = _free_local_port()
            argv = self._argv + ["--server", "--host", "127.0.0.1", "--port", str(port)]
            self._proc = subprocess.Popen(
                argv, cwd=self._cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
            )
            self._port = port
            threading.Thread(target=self._pump, name="crispasr-server-log",
                             daemon=True).start()
            try:
                self._wait_healthy(timeout)
            except BaseException:
                self.stop()
                raise

    def _wait_healthy(self, timeout: float) -> None:
        """轮询 /health 直到就绪；进程提前退出/超时都带上输出尾巴报错。"""
        deadline = time.monotonic() + max(1.0, float(timeout))
        url = f"http://127.0.0.1:{self._port}/health"
        while time.monotonic() < deadline:
            if not self.alive():
                raise RuntimeError(
                    "crispasr --server 启动即退出（这份核心可能不带 --server，"
                    f"或显存不足）。最后输出：{self.tail()}")
            try:
                with urllib.request.urlopen(url, timeout=2.0) as r:
                    body = json.loads(r.read().decode("utf-8", "replace"))
                if str((body or {}).get("status", "")).lower() in ("ok", "ready", "loaded"):
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(
            f"crispasr --server 在 {timeout:.0f} 秒内未就绪"
            f"（模型可能仍在加载或卡在初始化）。最后输出：{self.tail()}")

    def transcribe_file(self, path: Path | str, fields: dict | None = None,
                        timeout: float = _SERVER_REQ_TIMEOUT_SECS) -> dict:
        """POST /v1/audio/transcriptions（multipart）→ 解析后的 JSON。"""
        if not self.alive():
            raise RuntimeError("crispasr 常驻服务已退出。")
        path = Path(path)
        boundary = "----sttcrisp" + uuid.uuid4().hex
        chunks: list[bytes] = []
        for k, v in (fields or {}).items():
            if v is None or v == "":
                continue
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            chunks.append(str(v).encode("utf-8") + b"\r\n")
        chunks.append(f"--{boundary}\r\n".encode())
        # 文件名保留原扩展名（.webm 等容器由 crispasr 内部解码器判定）
        fname = f"seg{path.suffix or '.wav'}"
        chunks.append(
            f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            .encode())
        chunks.append(b"Content-Type: application/octet-stream\r\n\r\n")
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/audio/transcriptions",
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = (e.read() or b"").decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise RuntimeError(
                f"crispasr 常驻服务返回 HTTP {e.code}：{detail or e.reason}") from None

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_SERVER_STOP_GRACE_SECS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:
            pass


class CrispWhisperEngine:
    """CrispASR(whisper backend) 推理引擎。

    文件转录：一次性子进程（`crispasr.exe -osrt -ml 1`，要字级时间轴与进度）。
    实时分段：常驻 `crispasr.exe --server`（见 _CrispServer），免去每段重新
    加载模型的固定开销；起不来时自动退回一次性子进程。
    """

    # whisper_fa 做成 property：三处判定收敛一处（原先两处内联重复，改一处
    # 漏一处即「字级时间轴走没走 -am 对齐」行为分叉）。
    @property
    def whisper_fa(self) -> bool:
        """whisper 后端且对齐器就绪 → 字级时间轴走 -am 对齐链路。"""
        return (self._backend == "whisper" and self.use_aligner
                and bool(self._aligner_path))

    # use_aligner 做成 property：对齐器与模型不兼容时，**任何**调用端把它设成
    # True 都会被挡下（webview_backend.transcribe 每次转录都会依 UI 勾选重置，
    # 只在 load() 关掉是拦不住的）。
    @property
    def use_aligner(self) -> bool:
        return self._use_aligner

    @use_aligner.setter
    def use_aligner(self, v: bool):
        self._use_aligner = bool(v) and not getattr(self, "_fa_blocked", False)

    def __init__(self):
        self._use_aligner = False
        self._fa_blocked  = False
        self.ready       = False
        self._lock       = threading.Lock()
        # 取消事件：webview_backend.cancel() 置位 → _run_streaming 杀子进程。
        # 每次转录前由调用端 clear（见 webview_backend.transcribe）。
        self.cancel_event = threading.Event()
        # 界面兼容标志
        self.processor   = None    # None → app.py 改用 SUPPORTED_LANGUAGES
        self.diar_engine = None
        # 时间轴对齐（FA）：用 qwen3 ForcedAligner GGUF + crispasr `-am <gguf> -falign`，
        # 以 CTC 对齐器的字级时间轴覆盖 whisper 自带（较粗）的时间戳。gguf 存在才启用；
        # 否则退回 whisper native 时间轴（仍可用，仅较不精确）。
        # 标志真伪性供 app.py 共享 FA 流程判断（use_aligner=勾选、_fa_bin=模型就绪）。
        self.use_aligner   = False
        self.aligner       = False
        self._fa           = False
        self._fa_bin       = None   # FA gguf 路径（真值＝就绪）
        self._aligner_path = None   # Path: qwen3-forced-aligner-*.gguf
        # 执行状态
        self._exe        = None    # Path: crispasr.exe
        self._model_path = None    # Path: Whisper .bin / Qwen3-ASR .gguf
        self._backend    = "whisper"  # crispasr 后端：whisper / qwen3 / qwen3-1.7b …
        self._device_id  = 0       # GPU device id
        self._gpu_backend = "vulkan"  # --gpu-backend；None＝纯 CPU（挂 -ng）
        self._threads    = _default_threads()  # crispasr -t
        # 实时分段常驻服务（见 _CrispServer）：录音期间按需起、之后复用。
        # server_mode=False 或核心不支持 --server 时退回一次性 spawn。
        self.server_mode = True
        self._srv: "_CrispServer | None" = None
        self._srv_failed = False    # 起过一次失败就不再重试（旧核心没有 --server）
        self._srv_lock   = threading.Lock()

    # ══ 加载 ══════════════════════════════════════════════════════════

    def load(self, model_path: Path = None, crispasr_dir: Path = None,
             device_id: int = 0, cb=None, aligner_path: Path = None,
             backend: str | None = None, gpu_backend: str | None = "vulkan",
             cpu_threads: int = 0):
        """验证 crispasr.exe 与模型存在即可（子程序模式无常驻加载）。

        aligner_path：qwen3 ForcedAligner GGUF；None 时自动在模型目录（ov_models）
        与 crispasr_dir 内查找（模型目录优先）。
        找到 → 启用 FA（process_file 加 `-am <gguf> -falign`）；找不到 → 退回
        whisper native 时间轴。

        backend：crispasr 后端名（whisper / qwen3 / qwen3-1.7b …）。None 时依
        模型文件名自动推断（whisper GGML→whisper、qwen3*.gguf→qwen3-1.7b）。

        gpu_backend：crispasr.exe 的 `--gpu-backend`（vulkan / cuda）。传 None
        代表这份核心是 CPU 版或使用者选了纯 CPU 推理 → 改挂 `-ng`。调用端应
        依「实际安装的加速版本」给值（downloader.crispasr_gpu_backend）。

        cpu_threads：crispasr 的 `-t`；0＝自动（_default_threads），>0＝指定。
        与 OpenVINO 引擎共用 settings 的 cpu_threads 键。
        """
        def _s(msg):
            if cb:
                cb(msg)

        crispasr_dir = Path(crispasr_dir) if crispasr_dir else _DEFAULT_CRISPASR_DIR
        _s("查找 CrispASR 可执行文件…")
        exe = _find_exe(crispasr_dir)
        if exe is None:
            raise FileNotFoundError(
                f"找不到 crispasr.exe（已搜索 {crispasr_dir}）。\n"
                "请将 CrispASR Vulkan 版解压到 crispasr/ 目录。"
            )
        self._exe = exe

        model_path = Path(model_path) if model_path else (_DEFAULT_CRISPASR_DIR / _DEFAULT_MODEL_NAME)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到 Whisper 模型：{model_path}\n"
                f"请放入 {_DEFAULT_MODEL_NAME}（OpenAI Whisper Base GGML）。"
            )
        self._model_path = model_path
        self._device_id  = int(device_id)
        self._gpu_backend = gpu_backend or None
        self._threads    = int(cpu_threads) if cpu_threads and cpu_threads > 0 \
            else _default_threads()
        # 后端：显式指定优先，否则依模型文件名自动推断
        self._backend    = backend or _infer_backend(model_path)

        # ── 检测 FA 对齐器 GGUF（有则默认启用）──────────────────────────
        #   先判断这颗模型能不能用对齐器（见 _FA_INCOMPATIBLE 的说明）；不能用
        #   就整条 FA 关掉，避免对齐器吐出私用区乱码并把时间轴对歪。
        self._fa_blocked = _fa_incompatible(model_path)
        if self._fa_blocked:
            _s("此模型原生输出繁体，与 CTC 对齐器不兼容 → 禁用字级对齐，"
               "改用后端自带时间轴")
        ap = (Path(aligner_path) if aligner_path
              else _find_aligner_gguf(_DEFAULT_OV_MODEL_DIR, crispasr_dir))
        if ap and ap.exists() and not self._fa_blocked:
            self._aligner_path = ap
            self._fa_bin   = ap          # 真值＝FA 就绪
            self.use_aligner = True      # 默认开（缺文件则维持 False，退回 native）
            self.aligner   = True
            self._fa       = True
            _s(f"FA 对齐器就绪（{ap.name}）")
        else:
            self._aligner_path = None
            self._fa_bin   = None
            self.use_aligner = False
            self.aligner   = False
            self._fa       = False

        self.ready = True
        _fa_tag = "  + FA" if self._aligner_path else ""
        _gpu_tag = (self._gpu_backend or "cpu").upper()
        _s(f"就绪（CrispASR / {_gpu_tag} / {self._backend}  {model_path.name}{_fa_tag}）")

    # ══ 实时分段：常驻 --server（免每段重新加载模型）════════════════════

    def _server_argv(self) -> list[str]:
        """常驻 server 的启动参数（模型/后端/设备/调优与 _build_cmd 同口径）。

        `--server` 后的 --host/--port 由 _CrispServer.start() 追加（端口动态取，
        避免与其它实例/端口占用冲突）。-fa/-nf/-bo 1 是启动期解码默认值，
        与文件转录那条命令保持同一套速度/质量取舍。
        """
        cmd = [str(self._exe), "-m", str(self._model_path)]
        if self._backend and self._backend != "whisper":
            cmd += ["--backend", self._backend]
            # qwen3 后端的 LID 模型：显式指到本地 whisper ggml，免联网下载
            # （详见 _build_cmd 同一位置的说明）。
            lid = _lid_model_path(self._exe.parent) if self._exe else None
            if lid:
                cmd += ["--lid-model", str(lid)]
        if self._gpu_backend:
            cmd += ["--gpu-backend", self._gpu_backend, "-dev", str(self._device_id)]
        else:
            cmd += ["-ng"]
        cmd += ["-t", str(self._threads), "-fa", "-nf", "-bo", "1"]
        return cmd

    def _ensure_server(self, timeout: float | None = None) -> "_CrispServer | None":
        """取常驻 server（必要时启动）。失败→记一次并永久退回 spawn，返回 None。"""
        if not getattr(self, "server_mode", True) or not self._exe or not self._model_path:
            return None
        with self._srv_lock:
            if self._srv is not None and self._srv.alive():
                return self._srv
            if self._srv_failed:
                return None
            srv = _CrispServer(self._server_argv(), self._exe.parent)
            try:
                srv.start(timeout if timeout is not None else _SERVER_START_TIMEOUT_SECS)
            except Exception as e:
                self._srv_failed = True
                _warn(f"crispasr 常驻服务启动失败，实时分段改用一次性进程：{e}")
                return None
            self._srv = srv
            return srv

    def prepare_realtime(self) -> None:
        """录音开始：后台预热常驻 server（失败静默，分段时自然退回 spawn）。"""
        if not getattr(self, "server_mode", True) or self._srv_failed:
            return
        threading.Thread(target=self._ensure_server, name="crispasr-server-warm",
                         daemon=True).start()

    def release_realtime(self) -> None:
        """停掉常驻 server（录音结束／换模型／开始批量转录前调用，给显存让位）。"""
        with self._srv_lock:
            srv, self._srv = getattr(self, "_srv", None), None
        if srv is not None:
            srv.stop()

    def transcribe_realtime(self, audio_path, language: str | None = None,
                            context: str | None = None) -> TranscribeResult:
        """单段音频 → 文字（录音页实时出字；不产字幕文件、不做字级对齐）。

        ── 为什么改走常驻服务（2026-11 实测，GTX 1660 SUPER / 核心 0.8.54）──
        旧路径每段 spawn 一次 crispasr.exe，模型每次都重新上显存：whisper base
        固定开销 ≈0.6s、qwen3 Q8 ≈2.35s，占实时出字延迟 ~90%。`--server`
        只加载一次模型，同样的 2s 音频单段实测 0.17–0.23s。
        服务起不来（旧核心无 --server／显存不足）时自动退回一次性 spawn，
        行为与改动前一致，不会因为常驻失败而丢字。
        """
        srv = self._ensure_server()
        if srv is not None:
            try:
                obj = srv.transcribe_file(audio_path, {
                    # 语言名 → whisper 语言代码；None = 自动检测（沿用服务端设置）
                    "language": _LANG_MAP.get(language) if language else None,
                    "prompt": ((context or "").strip()[:500] or None),
                    "response_format": "verbose_json",
                })
                segs = _realtime_segments_from_api(obj)
                return TranscribeResult(
                    segments_rich=segs,
                    diag=None if segs else "未识别到语音内容。")
            except Exception as e:
                # 单次失败不永久禁用：停掉可惜的坏进程，本段退回 spawn，
                # 下一段再试常驻（服务器崩溃/显存回收等情况能自愈）。
                _warn(f"常驻服务转录失败，本段退回一次性进程：{e}")
                with self._srv_lock:
                    bad, self._srv = self._srv, None
                if bad is not None:
                    bad.stop()
        text = self._realtime_spawn_text(audio_path, language, context)
        segs = _realtime_segments_from_api({"text": text})
        return TranscribeResult(segments_rich=segs,
                                diag=None if segs else "未识别到语音内容。")

    def _realtime_spawn_text(self, audio_path, language, context) -> str:
        """一次性子进程的实时转录（常驻不可用时的后备，行为同旧实现）。"""
        with tempfile.TemporaryDirectory() as td:
            out_base = Path(td) / "seg_out"
            cmd = self._build_cmd(Path(audio_path), out_base, language,
                                  word_level=False, context=context)
            with self._lock:
                try:
                    subprocess.run(
                        cmd, capture_output=True, stdin=subprocess.DEVNULL,
                        creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
                        timeout=_RT_PROC_TIMEOUT_SECS,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"crispasr.exe 实时转录超时（>{_RT_PROC_TIMEOUT_SECS}s），"
                        "已终止子进程。请检查显卡驱动或改用 CPU 版本。")
            txt = out_base.with_suffix(".txt")
            return (txt.read_text(encoding="utf-8", errors="replace").strip()
                    if txt.exists() else "")

    # ══ 子程序命令建构（★ 由你定夺：speed/quality 取舍）═════════════════

    def _build_cmd(self, audio_path: Path, out_base: Path,
                   language: str | None, word_level: bool = True,
                   context: str | None = None) -> list[str]:
        """组合 crispasr.exe 命令行参数。

        ───────────────────────────────────────────────────────────────
        ★ 学习贡献点：这里是整个引擎唯一有「设计取舍」的地方。
          我们在 2026-06-17 的实测（Breeze-ASR-26 q5_0 / Vulkan / 126s 台语
          新闻）数据如下，请据此决定默认参数：

            设置                 fallback  时间    RTF     质量
            默认(-bo5+fallback)   129      142s   0.84x   人名最准
            -bo 2(保留fallback)    50       47.5s  2.65x   质量平平
            -nf -bo 1(关fallback)   0        3.6s  ~35x    几乎不损(推荐)

          可用的相关标志：
            -nf            关闭温度 fallback（避免难段反复重解码，最大提速）
            -bo N          best-of 候选数（1=贪婪最快；默认 5）
            -bs N          beam size（默认 greedy）
            -fa            flash attention（Vulkan 上安全，建议开）

        固定部分（已帮你写好）：模型、语言、Vulkan 设备、输出格式、静音。
        word_level=True 时加 `-ml 1` → 字符级时间轴（喂共享分行器，与 Qwen
        路径统一）；word_level=False（实时模式）→ 纯文字。
        你只需把速度/质量相关标志填入 `tuning`（约 1–4 个元素）。
        ───────────────────────────────────────────────────────────────
        """
        cmd = [
            str(self._exe),
            "-m", str(self._model_path),
        ]
        # 非 whisper 架构（Qwen3-ASR 等）需显式指定后端，crispasr 才用对解码路径。
        # whisper 平时**不传**：省略 --backend 时 crispasr 走 legacy whisper 主路径，
        # 功能最全且 -pp 有真实进度行。但该路径没接线 CTC 对齐器——-am/-falign 被
        # 静默忽略（0.8.44/0.8.48/0.8.49 实测输出逐字节相同）；因此仅在挂 FA 时才
        # 显式 --backend whisper 路由进统一 dispatch，对齐器才会生效（代价：统一
        # 路径不出 progress 行，进度改由 _run_streaming 走行活跃度模拟，与 qwen3
        # 同款；实测 0.8.44 起统一路径支持 --backend whisper）。
        unified_whisper = (self._backend == "whisper" and word_level
                           and self.use_aligner and bool(self._aligner_path))
        if self._backend:
            if self._backend != "whisper" or unified_whisper:
                cmd += ["--backend", self._backend]
            if self._backend != "whisper":
                # qwen3 后端默认要用 ggml-tiny.bin 做 LID（语言识别）；找不到时
                # crispasr 会自己联网下载，网络不通就干等 ~44 秒才继续（0.8.38/0.8.42
                # 实测）。这里显式指到本地的 whisper ggml（base 已随包），彻底免下载。
                lid = _lid_model_path(self._exe.parent) if self._exe else None
                if lid:
                    cmd += ["--lid-model", str(lid)]
        # 加速版本：Vulkan／CUDA 走 --gpu-backend + 设备 id；CPU 版挂 -ng
        # （CPU 版核心没有 ggml-vulkan/cuda DLL，硬给 --gpu-backend 会启动失败）
        if self._gpu_backend:
            cmd += ["--gpu-backend", self._gpu_backend,
                    "-dev", str(self._device_id)]
        else:
            cmd += ["-ng"]
        # CPU 线程数：不给 -t 时 crispasr 只用 min(4, 核心数)，见 _default_threads
        cmd += ["-t", str(self._threads)]
        cmd += [
            "-of", str(out_base),         # 输出文件名（不含副文件名）
            "-np",                        # 不印多余消息
        ]
        lang_code = _LANG_MAP.get(language, None) if language else None
        if word_level:
            # -ml 1 → 每段一字（字符级时间轴），等价于 FA 的字级输出
            # -pp → 打印「progress = NN% (x/N slices)」，供流式解析驱动进度条
            cmd += ["-osrt", "-ml", "1", "-pp"]
            # -sow：whisper 按「词」而非 BPE token 切段。-ml 1 原本按 token 切，长词
            # 会被拆成 "incom" "preh" "ensible"，且段文字被裁掉词首空白，分行器无从
            # 还原词界（拼成 "incom preh ensible"）。但 legacy 主路径对不以空白分词
            # 的语言（中文等）挂 -sow 会把整句并成一段、丢掉字级时间轴，故只在语言
            # 明确为空白分词语言时挂；FA（统一 dispatch）路径的切段由对齐器决定，
            # 中英文实测都安全，一律挂（顺带消除对齐器在中文 token 前补的空白）。
            if self._backend == "whisper" and (unified_whisper
                                               or lang_code in _SPACE_DELIMITED):
                cmd += ["-sow"]
        else:
            cmd += ["-otxt"]

        if lang_code:
            cmd += ["-l", lang_code]

        # 识别提示（hint）→ --prompt 初始提示（whisper.cpp 语义：偏置解码的
        # 用词与风格）。whisper 后端原生支持；qwen3 后端实测接受该标志、不
        # 回显提示内容（2026-10 本地 0.8.4x 验证），故两后端统一传递。
        # 截断到 500 字：提示最多占 n_text_ctx/2，过长会挤掉正文上下文。
        if context:
            cmd += ["--prompt", str(context).strip()[:500]]

        # ── 时间轴对齐（FA）：用 qwen3 ForcedAligner GGUF 覆盖 whisper native 时间戳 ──
        # 仅文件模式（word_level）需要精确字级时间轴；实时模式只取文字故不挂。
        #   -am <gguf>  加载 CTC 对齐器模型
        #   -falign     强制改用对齐器的字级时间轴（即使 whisper backend 自带）
        # 注意：此处 -fa 是 flash-attn、-falign 才是 force-aligner，两者并存不冲突。
        if word_level and self.use_aligner and self._aligner_path:
            cmd += ["-am", str(self._aligner_path), "-falign"]

        # 速度/质量取舍（实测决策，2026-06-17）：
        #   -fa  flash attention（Vulkan 安全）；-nf 关闭温度 fallback
        #   → 126s 台语新闻 142s→3.6s（0.84x→~35x），质量几乎不损。
        #   保留 fallback 虽人名略准但慢 40 倍、无法批量处理，故默认关闭。
        tuning: list[str] = ["-fa", "-nf"]

        cmd += tuning
        cmd.append(str(audio_path))
        return cmd

    # ══ 流式执行 crispasr.exe，解析 -pp 进度 → 驱动进度条 ═══════════════════
    def _run_streaming(self, cmd: list[str], progress_cb=None) -> int:
        """以 Popen 流式读 crispasr 输出，解析「progress = NN%」实时回报进度。

        crispasr（whisper 后端）一次处理整档（内部切 N 个 slice），`-pp` 会输出
        「crispasr: progress = NN% (x/N slices)」。**qwen3 后端不输出 progress
        行**（-pp 仅 whisper 后端实现，0.8.38/0.8.42 实测）——此前 qwen3 长档
        转录进度条恒为 0%，使用者以为卡死、关窗取消（反馈实录）。故这里加
        「行活跃度」回报：qwen3 路径下每收到一行输出就小幅推进模拟进度
        （对数逼近 90%，永不触顶），至少让使用者看到「在动」。

        取消支持：cancel_event（threading.Event）置位时终止子进程并抛
        RuntimeError("已取消")——读取循环阻塞在 proc.stdout 上时靠 cancel_watch
        线程兜底杀进程，避免「点了取消却要等下一行输出才生效」。

        停滞看门狗：超过 _PROC_STALL_TIMEOUT_SECS 没有任何新输出行 → 视为
        挂死（Vulkan 初始化卡死等），杀掉子进程并抛错，避免永久持有
        self._lock 把整个引擎卡死。
        """
        import time as _time
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
        )
        pat = re.compile(r"progress\s*=\s*(\d+)\s*%")
        tail: list[str] = []      # 末尾输出（崩溃原因诊断用）
        cancel_event = getattr(self, "cancel_event", None)
        is_qwen = (self._backend or "").startswith("qwen3")
        # 统一 dispatch 的 whisper（挂 FA 时才走，见 _build_cmd）同样不出 progress
        # 行 → 与 qwen3 共用行活跃度模拟进度。
        whisper_fa = self.whisper_fa
        # qwen3 活跃度进度：输出行数 → 模拟百分比（对数逼近 90）
        sim_base = 0

        def _cancel_watch():
            # proc.stdout 无输出的整段时间里，读取循环不醒来；由独立线程
            # 负责在取消或停滞时杀掉子进程，让读取循环立刻收到 EOF。
            while True:
                if cancel_event.wait(timeout=_PROC_STALL_TIMEOUT_SECS):
                    return                      # 置位 = 用户取消 → 杀进程
                # 超时未取消：若读取循环长时间无新输出 → 判挂死
                if _time.monotonic() - last_line_t.v > _PROC_STALL_TIMEOUT_SECS:
                    stall_flag[0] = True
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return

        class _V:
            __slots__ = ("v",)
            def __init__(self, v): self.v = v

        last_line_t = _V(_time.monotonic())
        stall_flag = [False]

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(target=_cancel_watch, daemon=True)
            watcher.start()
        cancelled = False
        try:
            for line in proc.stdout:
                last_line_t.v = _time.monotonic()
                t = line.rstrip()
                if t:
                    tail.append(t)
                    if len(tail) > 15:
                        tail.pop(0)
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                m = pat.search(line)
                if m and progress_cb:
                    p = int(m.group(1))
                    # 上限 99：留「完成」给写档后的最终回报，避免 100% 却还在写字幕
                    progress_cb(min(99, p), 100, f"CrispASR 转录中… {p}%")
                elif progress_cb and (is_qwen or whisper_fa):
                    # qwen3／统一 whisper（FA）无真实进度 → 行活跃度近似
                    # （每行 +ε，对数逼近 90%）
                    sim_base += 1
                    p = int(90 * (1 - 2.718281828 ** (-sim_base / 40)))
                    if p > 0:
                        progress_cb(min(90, p), 100, "CrispASR 转录中…（模型运行中，暂无精确进度）")
            # 循环正常结束（无更多输出）后再查一次：cancel_watch 杀掉进程时
            # EOF 与置位几乎同时发生，最后一行之后才置位的场景在此捕获。
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
        finally:
            try:
                proc.wait(timeout=_PROC_SHUTDOWN_GRACE_SECS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if watcher is not None:
                # 收尾：确保 watcher 不挂在未置位的 event 上（正常结束时）
                cancel_event.set()
                watcher.join(timeout=1)
                # 复位：切片转录逐片调 process_file，而调用端只在入口 clear
                # 一次；置位残留会让第 2 片起刚启动就被 watch 线程判「已取消」
                # 杀掉。每次运行结束时复位，保证下次启动是干净状态。
                cancel_event.clear()
        if cancelled:
            raise RuntimeError("已取消。")
        if stall_flag[0]:
            detail = "　|　".join(tail[-5:]) if tail else "（无输出）"
            raise RuntimeError(
                f"crispasr.exe 超过 {_PROC_STALL_TIMEOUT_SECS} 秒无任何输出，疑似挂死，已终止。"
                f"最后输出：{detail}\n"
                "请检查显存是否充足、加速版本是否与显卡匹配，或改用 CPU 推理版本重试。")
        if proc.returncode != 0:
            detail = "　|　".join(tail[-5:]) if tail else "（无输出）"
            raise RuntimeError(
                f"crispasr.exe 异常退出（返回码 {proc.returncode}）：{detail}\n"
                "请检查显存是否充足、加速版本是否与显卡匹配，"
                "或改用 CPU 推理版本重试。")
        return proc.returncode

    # ══ 文件转录 → SRT（字级时间轴 → 共享分行器，与 Qwen 路径统一）════════

    def process_file(self, audio_path: Path, progress_cb=None,
                     language: str | None = None, context: str | None = None,
                     diarize: bool = False, n_speakers: int | None = None,
                     original_path: Path | None = None,
                     out_format: str | None = None) -> Path | None:
        """音频 → SRT，返回 SRT 路径（None=无输出）。

        流程：crispasr `-ml 1` 取字符级时间轴 → 解析 → 共享的
        `_ts_chatllm_to_subtitle_lines` 分行（含孤儿合并）。

        说话者分离（diarize）：用与后端无关的外部 ONNX（diar_engine，CPU）取得
        说话者段落，再依时间把每行字幕指派给对应说话者。whisper/qwen 后端皆适用
        —— crispasr 单次出字级时间轴 × diar 段落，零额外子程序。需先由上层
        （webview_backend._ensure_diarization / app.py）把 diar_engine 挂上。
        context（hint）仅 whisper 后端支持（--prompt 初始提示）；qwen3 后端
        无对应机制，静默忽略。

        返回 TranscribeResult（output=字幕路径；None=未产出）。
        """
        audio_path = Path(audio_path)
        if progress_cb:
            progress_cb(0, 1, "CrispASR 转录中…")

        # 非 ASCII 路径防线：crispasr.exe（0.8.35 实测）对含非 ASCII 字符的
        # 音频路径（含目录）报「input file not found」——其内部文件打开不走
        # Windows UTF-16 API。完整路径无法纯 ASCII 编码时，复制进一个保证
        # ASCII 的临时目录（子进程输出用的同一个 TemporaryDirectory）再传。
        # diar_audio：说话者分离仍用原路径（soundfile 支持 Unicode；临时副本
        # 在转录结束即删，diarization 若再用它会静默失败）。
        import uuid as _uuid
        ascii_dir = None
        diar_audio = audio_path
        try:
            str(audio_path.resolve(True)).encode("ascii")
        except (UnicodeEncodeError, OSError):
            try:
                ascii_dir = tempfile.mkdtemp(prefix="stt_ascii_")
                ascii_copy = Path(ascii_dir) / f"audio_{_uuid.uuid4().hex[:8]}{audio_path.suffix or '.flac'}"
                shutil.copyfile(audio_path, ascii_copy)
                audio_path = ascii_copy
            except Exception:
                ascii_dir = None   # 复制失败 → 仍按原路径尝试

        try:
            with tempfile.TemporaryDirectory() as td:
                out_base = Path(td) / "crisp_out"
                cmd = self._build_cmd(audio_path, out_base, language,
                                      word_level=True, context=context)
                with self._lock:
                    self._run_streaming(cmd, progress_cb)
                srt_tmp = out_base.with_suffix(".srt")
                if not srt_tmp.exists():
                    return TranscribeResult()
                raw = srt_tmp.read_text(encoding="utf-8", errors="replace")
        finally:
            if ascii_dir is not None:
                shutil.rmtree(ascii_dir, ignore_errors=True)

        # ── 依后端选对「断句契约」（使用者要求：Qwen 要走 Qwen 逻辑，非 Whisper）──
        #   标点（各后端统一，与 OpenVINO 路径 _split_to_lines 相同、中英文一致）：
        #     标点只当切行点、不进字幕行。whisper 的标点是独立 token
        #     （"Americans" "," "ask"）、qwen3／对齐器则粘在词尾（"so," "間："），
        #     两种都剔出 items、只留 raw_text 当切点——否则独立标点会沦为下一行
        #     开头（", ask not"）、被空格拼成 "country ."；粘连标点则让分行器错过
        #     切点（「開放時間：早上…五」+「點。」）。
        #   Whisper(Breeze)：无标点 → 保留空白语句边界、break_on_space=True 逼近 Qwen。
        #     挂 FA 时例外：统一 dispatch 路径 token 前的空白只是分词残留（中文实测
        #     「开| 放| 时|间」），当切点会逐字乱切；该路径改以对齐器的真实停顿切行。
        #   _fa_blocked 的内插路径维持原契约（标点随字内插、行首标点移回前行）。
        is_qwen = (self._backend or "").startswith("qwen3")
        fa_blocked = getattr(self, "_fa_blocked", False)
        whisper_fa = self.whisper_fa
        ts_items, raw_text = _parse_srt_words(raw, drop_punct=(is_qwen or not fa_blocked),
                                              strip_attached=not fa_blocked)
        if not ts_items:
            return TranscribeResult()
        if fa_blocked:
            # 没有对齐器时 `-ml 1` 对 qwen3 后端无效，每个 item 会是一整段长文字，
            # 共享分行器便无从断句（实测整段可达 60 秒）。这里把长 item 依字符
            # **等时内插**摊成字符级，让分行器照常做标点切行／MAX_CHARS 保护。
            # 内插时间轴当然不如 CTC 对齐精准，但在「粗到不能看」与「约略正确」
            # 之间，后者对字幕明显有用；且段落起止仍是后端给的真实时间。
            ts_items = _interpolate_chars(ts_items)

        lines5 = _ts_chatllm_to_subtitle_lines(
            ts_items, raw_text, 0.0, None,
            break_on_space=(not is_qwen and not whisper_fa), with_words=True,
        )
        if not lines5:
            return TranscribeResult()
        if fa_blocked:
            lines5 = _fix_leading_punct(lines5)
        lines = [(s, e, t, sp) for (s, e, t, sp, _w) in lines5]

        # 说话者分离（外部 ONNX，与 whisper/qwen 后端无关）：依时间中点指派每行说话者。
        # diarize 只重新指派 speaker、不增减行，故与 lines5 一一对应 → 可 zip 并回字级。
        if diarize and self.diar_engine is not None and getattr(self.diar_engine, "ready", False):
            lines = self._apply_diarization(diar_audio, lines, n_speakers, progress_cb)

        # 字级结果：speaker 取 diarize 后的 lines，words 取 lines5
        rich = [
            {"start": a[0], "end": a[1], "text": a[2], "speaker": a[3], "words": b[4]}
            for a, b in zip(lines, lines5)
        ]

        # 共享写出层：依全域设置（或 out_format 覆盖）产出 .srt 或 .txt。
        ref = original_path if original_path is not None else diar_audio
        out = write_transcript(ref, lines, out_format)
        if progress_cb:
            progress_cb(1, 1, "完成")
        return TranscribeResult(output=out, segments_rich=rich)

    def _apply_diarization(self, audio_path, lines, n_speakers, progress_cb=None):
        """把每行字幕依时间中点指派给 diar_engine 检测到的说话者。

        lines:  [(start, end, text, None), ...]（_ts_chatllm_to_subtitle_lines 产出）
        返回：  [(start, end, text, "说话者N"), ...]；失败/无段落时原样返回。
        """
        try:
            from audio_io import load_audio_16k_mono
            if progress_cb:
                progress_cb(1, 1, "说话者分离中…")
            audio, _ = load_audio_16k_mono(Path(audio_path), SAMPLE_RATE)
            segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
        except Exception as _e:
            try:
                from applog import log_warn
                log_warn(f"说话者分离失败，字幕保持无说话者标签："
                         f"{type(_e).__name__}: {_e}")
            except Exception:
                pass
            return lines
        if not segs:
            return lines

        def _spk_at(s: float, e: float) -> str:
            # 重叠最长的说话者段（跨界行不再随机归给中点一侧）；无交集取最近段
            return _assign_speaker_by_overlap(s, e, segs)

        return [(s, e, txt, _spk_at(s, e)) for (s, e, txt, _spk) in lines]

    # ══ 实时/单段转录 → 纯文字 ═════════════════════════════════════════

    def transcribe(self, audio: np.ndarray, max_tokens: int = 300,
                   language: str | None = None, context: str | None = None) -> str:
        """16kHz float32 音频 → 文字（旧接口；保留给既有的 ndarray 调用端）。

        录音页现在走 `transcribe_realtime`（可选常驻 --server）。这里落地成
        临时 wav 后同样交给它，两条入口共用一条实现，不再各留一套 spawn。
        """
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "seg.wav"
            sf.write(str(wav), audio, SAMPLE_RATE)
            res = self.transcribe_realtime(wav, language=language, context=context)
        return "\n".join(s["text"] for s in (res.segments_rich or []))


# ── SRT 解析 / 写入（模块层工具）──────────────────────────────────────

def _srt_ts_to_sec(ts: str) -> float:
    """'00:00:01,560' → 1.56 秒。"""
    ts = ts.strip().replace(".", ",")
    hms, _, ms = ts.partition(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def _fix_leading_punct(lines5: list[tuple]) -> list[tuple]:
    """把「以标点开头」的字幕行，其开头标点移回前一行行尾。

    只发生在无对齐器的内插路径：crispasr 的每个 SRT 区块各自结束，区块边界刚好
    落在标点前时，该标点就成了下一行的开头（「，直询台上」）。标点属于前一句，
    移回去即可。首行的开头标点没有前一行可归，直接去掉。
    """
    out: list[tuple] = []
    for (s, e, t, spk, w) in lines5:
        lead = ""
        while t and t[0] in _PUNCT:
            lead, t = lead + t[0], t[1:]
        if lead and out:
            ps, pe, pt, pspk, pw = out[-1]
            out[-1] = (ps, pe, pt + lead, pspk, pw)
        if t:
            out.append((s, e, t, spk, w))
    return out


def _interpolate_chars(items: list[tuple[str, float, float]]
                       ) -> list[tuple[str, float, float]]:
    """把「一段长文字 + 起止时间」摊成字符级（等时内插）。

    只处理长度 > 1 的 item；已是单字符的原样保留，故对本来就有字级时间轴的
    路径（挂了 FA）调用也不会有副作用。
    """
    out: list[tuple[str, float, float]] = []
    for text, s, e in items:
        n = len(text)
        if n <= 1:
            out.append((text, s, e))
            continue
        step = (e - s) / n if e > s else 0.0
        for i, ch in enumerate(text):
            out.append((ch, s + step * i, s + step * (i + 1)))
    return out


# 数字内的连接符（3.5／1,250／9:30）不是句界。-ml 1 会把它们拆成独立 token
# （"3" "." "5"），剔除标点、切行之前须先并回同一个数字——否则中文行会把
# "增长了3" "." "5%" 拼成「增长了35%」，英文行则切成 "3" / "5"。
_NUM_JOINERS = frozenset(".,:")
# 依附在数字上的符号：货币在前（$1,250）、百分号在后（50%）。
_CURRENCY = frozenset("$€£¥")


def _merge_number_tokens(toks: list[list]) -> list[list]:
    """把被 `-ml 1` 拆开的数字 token 并回一个。

    toks：[[lead_ws, text, start, end], ...]（text 已去首尾空白；"" 为纯空白块）
    规则：
      • 「…3」「.」「5…」或「…3.」「5…」→ 3.5（小数点、时间冒号接任意位数）
      • 千分位逗号只接「恰好 3 位数」→ 1,250；「3」「,」「5」视为列举，不并
      • 「$」「1」→ $1；对齐器把货币符号粘在前词尾（"for$" "1"）→ 移到数字前
      • 「50」「%」→ 50%
    """
    def _digits_ok(joiner: str, t: str) -> bool:
        if not t[:1].isdigit():
            return False
        if joiner != ",":
            return True
        return len(t) >= 3 and t[:3].isdigit() and not t[3:4].isdigit()

    toks = [list(t) for t in toks]
    out: list[list] = []
    i = 0
    while i < len(toks):
        cur = toks[i]
        while i + 1 < len(toks):
            t, nxt = cur[1], toks[i + 1]
            nt = nxt[1]
            if (t[-1:].isdigit() and len(nt) == 1 and nt in _NUM_JOINERS
                    and i + 2 < len(toks) and _digits_ok(nt, toks[i + 2][1])):
                cur[1] = t + nt + toks[i + 2][1]          # 「3」「.」「5」
                cur[3] = toks[i + 2][3]
                i += 2
            elif (len(t) >= 2 and t[-1] in _NUM_JOINERS and t[-2].isdigit()
                    and _digits_ok(t[-1], nt)):
                cur[1] = t + nt                            # 「3.」「5」
                cur[3] = nxt[3]
                i += 1
            elif t and all(c in _CURRENCY for c in t) and nt[:1].isdigit():
                cur[1] = t + nt                            # 「$」「1」
                cur[3] = nxt[3]
                i += 1
            elif t[-1:].isdigit() and nt == "%":
                cur[1] = t + nt                            # 「50」「%」
                cur[3] = nxt[3]
                i += 1
            elif len(t) >= 2 and t[-1] in _CURRENCY and nt[:1].isdigit():
                toks[i + 1] = [nxt[0], t[-1] + nt, nxt[2], nxt[3]]
                cur[1] = t[:-1]                            # 「for$」「1」→「for」「$1」
                break
            else:
                break
        out.append(cur)
        i += 1
    return out


def _strip_edge_punct(t: str) -> str:
    """去掉粘在词首尾的标点（so, → so、間： → 間）；数字内部的 . , : 保留（3.5）。"""
    i, j = 0, len(t)
    while i < j and t[i] in _PUNCT:
        i += 1
    while j > i and t[j - 1] in _PUNCT:
        j -= 1
    return t[i:j]


def _parse_srt_words(srt_text: str, drop_punct: bool = False,
                     strip_attached: bool = False
                     ) -> tuple[list[tuple[str, float, float]], str]:
    """解析 `-ml 1` 字符级 SRT。

    返回 (ts_items, raw_text)：
        ts_items : [(char, start_s, end_s), ...]（不含空白项）
        raw_text : 字符对接，并在 whisper 的「空白语句边界」处保留一个空白，
                   供分行器 break_on_space 在语句边界切行。

    drop_punct=True：标点字符只保留在 raw_text 当「切点」，**不**进 ts_items ——
    这正是 OpenVINO 路径的契约（items=内容词、标点仅在 raw_text；英文／中文
    统一，见 app._split_to_lines）。crispasr `-ml 1` 会把标点也输出成独立字符段
    （whisper："Americans" "," "ask"），若放进 items 会让断句器的 raw_text 指标
    双重前进、标点沦为下一行开头（", ask not"），故需在此剔除。同时把被拆开的
    数字并回（3 . 5 → 3.5，见 _merge_number_tokens），免得小数点被当切点。

    strip_attached=True：标点粘在词尾的 token（qwen3／对齐器："so," "間："）也
    剥掉词首尾标点再进 items；raw_text 仍保留完整 token，切点照常生效。
    """
    tokens: list[list] = []                 # [lead_ws, text, start, end]
    # 正规化换行（crispasr SRT 为 CRLF），再以空行切块
    blocks = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n\n")
    for block in blocks:
        rows = block.split("\n")
        tl_idx = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if tl_idx is None:
            continue
        a, _, b = rows[tl_idx].partition("-->")
        try:
            s = _srt_ts_to_sec(a)
            e = _srt_ts_to_sec(b)
        except (ValueError, IndexError):
            continue
        raw = "".join(rows[tl_idx + 1:])    # 不 strip，保留前导空白
        tokens.append([raw[:1].isspace(), raw.strip(), s, e])
    if drop_punct:
        tokens = _merge_number_tokens(tokens)

    items: list[tuple[str, float, float]] = []
    raw_parts: list[str] = []
    for lead, token, s, e in tokens:
        if token == "":
            raw_parts.append(" ")           # 纯空白块 = 语句边界
            continue
        if lead:                            # 字符前带空白 = 边界 + 字符
            raw_parts.append(" ")
        raw_parts.append(token)             # 标点一律进 raw_text（供切点检测）
        if drop_punct and all(ch in _PUNCT for ch in token):
            continue                        # 纯标点：只当切点，不进 items
        if strip_attached:
            token = _strip_edge_punct(token)
        items.append((token, s, e))
    return items, "".join(raw_parts)



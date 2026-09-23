"""webview_backend.py — WebView 共享后端逻辑（与窗口/传输无关）

把「状态 / 设置 / 设备 / 端点 / 转录」等业务逻辑集中于此，重用既有
ASREngine。不依赖 pywebview，也不依赖 HTTP —— 可被 webview_server
（桌面）直接调用，亦可被测试直接 import 验证。

设计原则：
  • 不碰 GUI、不碰 socket；纯函数式后端。
  • transcribe() 以 progress_cb(pct:int, status:str) 回报进度，由上层
    （HTTP 层）转成 SSE 或其他通道推给前端。
  • 模型较重 → load() 设计成可在背景线程调用。
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path

import app as core          # 重用 ASREngine / 常数 / 诊断函数（__main__ guard 保护）

BASE_DIR = Path(__file__).resolve().parent

# ── 模型目录：引擎 → 模型 → (backend, settings patch)。──────────────────
#   引擎即推理核心（两组）：OpenVINO（纯 CPU 的 Qwen）与 CrispASR（GPU 加速，
#   跑 Whisper GGML 与 Qwen3-ASR GGUF）。settings 键与 app.py 兼容：
#     openvino → cpu_model_size("0.6B"/"1.7B")
#     crispasr → crisp_model("whisper"/"qwen3") + whisper_size/qwen 量化键
#   每个 entry：(引擎标签, 模型标签, backend, settings patch)
_ENGINE_OV  = "OpenVINO"
_ENGINE_CASR = "CrispASR"
# OpenAI Whisper 模型标签映射（供目录 / _current_selection / 下拉等使用）
_WHISPER_SIZES = ("base", "small", "medium", "large", "turbo")
_WHISPER_LABEL_BY_SIZE = {
    "base": "Whisper Base", "small": "Whisper Small", "medium": "Whisper Medium",
    "large": "Whisper Large", "turbo": "Whisper Large Turbo",
}
_MODEL_CATALOG = [
    (_ENGINE_OV, "Qwen3-ASR-0.6B",      "openvino", {"cpu_model_size": "0.6B"}),
    (_ENGINE_OV, "Qwen3-ASR-1.7B INT8", "openvino", {"cpu_model_size": "1.7B"}),
    (_ENGINE_CASR, "Qwen3-ASR-1.7B Q4 (CRISPASR)", "crispasr", {"crisp_model": "qwen3", "crisp_qwen_quant": "q4"}),
    (_ENGINE_CASR, "Qwen3-ASR-1.7B Q8 (CRISPASR)", "crispasr", {"crisp_model": "qwen3", "crisp_qwen_quant": "q8"}),
    # 日语动漫特化（cstr/qwen3-asr-1.7b-ja-anime）：同架构、针对日语微调，日文歌词/台词识别较佳。
    (_ENGINE_CASR, "Qwen3-ASR-1.7B 日语动漫 Q4 (CRISPASR)", "crispasr", {"crisp_model": "qwen3-ja", "crisp_qwen_quant": "q4"}),
    (_ENGINE_CASR, "Qwen3-ASR-1.7B 日语动漫 Q8 (CRISPASR)", "crispasr", {"crisp_model": "qwen3-ja", "crisp_qwen_quant": "q8"}),
    # OpenAI Whisper 官方模型（ggerganov/whisper.cpp GGML）：CrispASR 的 whisper
    # 后端直接兼容。5 档尺寸覆盖速度／精度需求。
] + [
    (_ENGINE_CASR, _WHISPER_LABEL_BY_SIZE[size], "crispasr",
     {"crisp_model": "whisper", "whisper_size": size})
    for size in _WHISPER_SIZES
]
_QWEN_CASR_QUANT_LABEL = {"q4": "Qwen3-ASR-1.7B Q4 (CRISPASR)",
                          "q8": "Qwen3-ASR-1.7B Q8 (CRISPASR)"}
_QWEN_JA_CASR_QUANT_LABEL = {"q4": "Qwen3-ASR-1.7B 日语动漫 Q4 (CRISPASR)",
                             "q8": "Qwen3-ASR-1.7B 日语动漫 Q8 (CRISPASR)"}
# Whisper 尺寸提示（模型标签 → 提醒文字）
_WHISPER_MODEL_NOTES = {
    "Whisper Large Turbo": ("Turbo（large-v3-turbo）：Large 级精度、解码速度快约 8 倍，"
                            "显存占用约为 Large 的一半，推荐优先。"),
    "Whisper Large":       "Large-v2：精度最高但速度最慢，适合追求极限准确度的离线场景。",
}
# 模型标签 → 提醒文字（Whisper 尺寸提示）
_MODEL_NOTES = _WHISPER_MODEL_NOTES


def _sanitize_tag(tag: str) -> str:
    """引擎/模型标签 → 文件名安全片段（供输出字幕文件名拼接）。

    去除 Windows 文件名非法字符与路径分隔符、压掉多余空白；空串回 "model"。
    """
    for ch in '<>:"/\\|?*':
        tag = tag.replace(ch, " ")
    tag = " ".join(tag.split())
    return tag or "model"

# crispasr-Qwen 系列的 crisp_model 值（皆走 --backend qwen3-1.7b、同置 ov_models/、
# 共享 crisp_qwen_quant，差别只在权重与下载来源）。"whisper" 不在此列。
_QWEN_CASR_MODELS = ("qwen3", "qwen3-ja")
# 各 crispasr-Qwen 模型在下载消息中的中文注记
_QWEN_CASR_NOTE = {"qwen3-ja": "（日语动漫）"}


def _qwen_casr_dl(crisp_model: str):
    """依 crisp_model 返回对应的 (文件名, 存在检查, 下载) 三个下载器函数。

    qwen3     → 标准 Qwen3-ASR-1.7B（cstr/qwen3-asr-1.7b-GGUF）
    qwen3-ja  → 日语动漫特化（cstr/qwen3-asr-1.7b-ja-anime-GGUF）
    让所有「crispasr-Qwen」路径共享一处分派，避免各模型分支散落各方法。
    """
    from downloader import (qwen3_asr_gguf_filename, quick_check_qwen3_asr_gguf,
                            download_qwen3_asr_gguf, qwen3_asr_ja_gguf_filename,
                            quick_check_qwen3_asr_ja_gguf, download_qwen3_asr_ja_gguf)
    if crisp_model == "qwen3-ja":
        return (qwen3_asr_ja_gguf_filename, quick_check_qwen3_asr_ja_gguf,
                download_qwen3_asr_ja_gguf)
    return (qwen3_asr_gguf_filename, quick_check_qwen3_asr_gguf,
            download_qwen3_asr_gguf)

# ── 长音频 FFmpeg 切片转录 ────────────────────────────────────────────
#   超长音频一次性塞给引擎会造成：内存峰值高、OpenVINO 的 KV-Cache 线性
#   膨胀、转录中途失败要整段重来。这里先用 ffmpeg 把音频按等长窗口切片（相邻
#   片段重叠 SLICE_OVERLAP 秒，避免切点把一个字/一句话劈成两半造成漏识），
#   逐片转录，再把各片 SRT 解析后按片起点平移、重叠区去重合并成整份结果。
#   说话者分离只在单片内做（跨片依时间中点指派，与整段行为一致）。
_SLICE_THRESHOLD_SECS = 2 * 3600     # 超过 2 小时启用切片
_SLICE_WINDOW_SECS = 30 * 60         # 每片 30 分钟（内存充足时的上限）
_SLICE_OVERLAP_SECS = 15             # 相邻片段重叠 15 秒

# ── 切片时长自适应（依可用内存）────────────────────────────
# 内存依据：切片解码进内存后是 16kHz float32 mono（64KB/s），VAD/引擎处理
# 还会产生数份拷贝（按 ~5 倍估算）。默认 30 分钟片 ≈ 110MB 原始 ≈ 550MB
# 实占，8GB 空闲内存的机器毫无压力；空闲内存只有几百 MB 的机器若仍切
# 30 分钟，转录中易触发系统交换甚至 OOM。故按可用物理内存收缩片长：
#   预算 = 可用内存 × 25%（给引擎/OS 留大头）→ 秒数 = 预算 ÷ 每秒实占
# 夹紧 [5 分钟, 30 分钟]：下限保证重叠区（15s）不退化，上限即原行为。
_SLICE_MEM_BUDGET_FRAC = 0.25
_SLICE_MEM_PER_SEC = 65536 * 5       # 64KB/s 原始 × ~5 份处理拷贝
_SLICE_WINDOW_MIN_SECS = 5 * 60


def _avail_phys_bytes() -> int | None:
    """可用物理内存（字节）；Windows GlobalMemoryStatusEx，失败回 None。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return int(st.ullAvailPhys)
    except Exception:
        return None


def _slice_window_secs() -> int:
    """依可用物理内存智能决定切片时长（秒）。

    内存充足 → 30 分钟（与原固定值一致，行为不变）；空闲内存紧张时按
    预算公式收缩，最低 5 分钟。查询失败（非 Windows 等）回默认上限。
    """
    avail = _avail_phys_bytes()
    if not avail:
        return _SLICE_WINDOW_SECS
    budget = avail * _SLICE_MEM_BUDGET_FRAC
    secs = int(budget / _SLICE_MEM_PER_SEC)
    return max(_SLICE_WINDOW_MIN_SECS, min(_SLICE_WINDOW_SECS, secs))


def _ffprobe_duration(ff: Path, audio: Path) -> float | None:
    """ffprobe 读取音频总时长（秒）；失败返回 None。"""
    ffprobe = ff.parent / ("ffprobe.exe" if ff.suffix.lower() == ".exe"
                           else "ffprobe")
    if not ffprobe.exists():
        ffprobe = ff.with_name("ffprobe" + ff.suffix)
    try:
        import subprocess
        r = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            capture_output=True, text=True, timeout=30,
            creationflags=0x08000000 if sys.platform == "win32" else 0)
        return float(r.stdout.strip())
    except Exception:
        return None


def _slice_plan(duration: float, window_secs: int | None = None) -> list[tuple[float, float]]:
    """总时长 → [(片起点, 片终点)]，相邻片重叠 _SLICE_OVERLAP_SECS。

    window_secs 省略时依可用内存自适应（_slice_window_secs）。
    """
    window = window_secs if window_secs is not None else _slice_window_secs()
    step = window - _SLICE_OVERLAP_SECS
    spans = []
    start = 0.0
    while start < duration - 0.5:
        end = min(start + window, duration)
        spans.append((start, end))
        if end >= duration:
            break
        start += step
    return spans or [(0.0, duration)]


def _merge_sliced_segments(all_parts: list[list[dict]]) -> list[dict]:
    """各片 segments（已含绝对时间）→ 去重合并。

    重叠区去重：以「后一片」为准，丢弃 start 落在前一片已收录行之后的重复行
    （重叠区两片都会识别出同样的文字，取后片的起点做截断线）。
    """
    if not all_parts:
        return []
    merged = list(all_parts[0])
    for part in all_parts[1:]:
        if not part:
            continue
        # 截断线：前一片最后一行的起点（重叠区识别内容重复，以此为界丢弃）
        cut = merged[-1]["start"] if merged else None
        for seg in part:
            if cut is not None and seg["start"] < cut:
                continue
            merged.append(seg)
    return merged


def _srt_ts_fmt(sec: float) -> str:
    ms = max(0, int(round(sec * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 说话者分离是「与后端无关的外部 ONNX」(diarize.py / DiarizationEngine)：
# OpenVINO 与 CRISPASR(whisper/qwen) 皆支持——前者 process_file 内置 use_diar 分支，
# 后者由 crisp_engine._apply_diarization 依时间指派。diar_engine 由 _ensure_diarization 挂上。

# 识别语言：crispasr 使用的常用语言清单（OpenVINO 改用 processor 的）
_COMMON_LANGS = [
    "Chinese", "English", "Japanese", "Korean", "Cantonese", "French", "German",
    "Spanish", "Portuguese", "Russian", "Arabic", "Thai", "Vietnamese",
    "Indonesian", "Malay",
]


def _srt_ts_to_sec(ts: str) -> float:
    """'00:00:01,560' → 1.56 秒。"""
    ts = ts.strip().replace(".", ",")
    hms, _, ms = ts.partition(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def _parse_srt(srt_text: str) -> list[dict]:
    """解析 SRT 文字为 [{"start","end","text"}]（原 api_server._parse_srt 的内联版，
    端点服务移除后仅供本地 SRT 解析使用）。"""
    blocks = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n\n")
    out = []
    for block in blocks:
        rows = [r for r in block.split("\n") if r.strip()]
        tl = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if tl is None:
            continue
        a, _, b = rows[tl].partition("-->")
        try:
            start = _srt_ts_to_sec(a)
            end = _srt_ts_to_sec(b)
        except (ValueError, IndexError):
            continue
        text = "\n".join(rows[tl + 1:]).strip()
        if text:
            out.append({"start": start, "end": end, "text": text})
    return out


def parse_srt_to_segments(srt_path) -> list[dict]:
    """把 process_file 产出的 SRT 解析成前端要的 segments。

    说话者分离时，SRT 文字以「说话者 N：…」开头 → 拆出 speaker 与 text。
    """
    if not srt_path or not Path(srt_path).exists():
        return []
    raw = _parse_srt(Path(srt_path).read_text(encoding="utf-8"))
    out = []
    for s in raw:
        text, speaker = s["text"], None
        if "：" in text[:8]:
            head, _, rest = text.partition("：")
            if head.startswith("说话者") or head.startswith("说话者"):
                digits = "".join(c for c in head if c.isdigit())
                speaker = int(digits) if digits else None
                text = rest
        out.append({"start": s["start"], "end": s["end"], "speaker": speaker, "text": text})
    return out


def _rich_to_segments(rich) -> list[dict]:
    """引擎侧通道 _last_segments_rich → 前端 segments（含字级 words，卡拉OK用）。

    引擎存的 speaker 是原始标签（"说话者N" 或 None）；此处正规化成整数，
    与 parse_srt_to_segments 的前端契约一致。words 原样带出（[]=无对齐）。
    """
    out = []
    for seg in (rich or []):
        spk = seg.get("speaker")
        speaker = None
        if isinstance(spk, int):
            speaker = spk
        elif isinstance(spk, str):
            digits = "".join(c for c in spk if c.isdigit())
            speaker = int(digits) if digits else None
        out.append({
            "start": seg["start"], "end": seg["end"],
            "speaker": speaker, "text": seg["text"],
            "words": seg.get("words") or [],
        })
    return out


class WebBackend:
    """桌面 WebView 的后端。HTTP 层（webview_server）持有一个实例。"""

    def __init__(self, on_event=None):
        """on_event(event:str, payload:dict)：状态/进度推送回调（可选）。"""
        self.engine = core.ASREngine()
        self._loaded = False
        self._load_err = None
        self._loading = False
        self._cancel = False
        self._transcribing = False       # 单文件识别进行中（关闭确认用）
        self._recording = False          # 前端录音进行中（由 /api/record-state 同步）
        self._on_event = on_event
        self._theme_cb = None            # 主题变更回调（app_webview 用来同步窗口标题栏深浅）
        self._lock = threading.Lock()          # 转录/加载互斥（整个引擎调用期间被持有）
        self._load_gate = threading.Lock()     # start_load 的 check-and-set（非重入，短持有）
        self._transcribe_gate = threading.Lock()  # transcribe 忙碌 check-and-set（非重入）
        self._active_label = None        # 实际加载的 (引擎标签, 模型标签)——字幕文件名/日志用
        self._seed_defaults()            # 首次启动（无 backend）→ 种子默认模型
        self._apply_runtime_prefs()      # 启动即套用持久化偏好（VAD/简繁/镜像/格式）

    # ── 首次启动默认：1.7B Qwen on CRISPASR Q4（多数机器跑得动）─────────────
    def _seed_defaults(self):
        """settings.json 尚无 backend 时，种下默认模型选择。

        多数状况下「Qwen3-ASR-1.7B Q4（CRISPASR/Vulkan）」可顺利运行（GPU Vulkan，
        繁体佳）。写入 settings 后，后续 _persisted_backend/_load_worker 等皆读到它。
        既有使用者（已有 backend）不受影响。
        """
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        if cur.get("backend"):
            return                       # 既有设置 → 不覆盖
        cur.setdefault("backend", "crispasr")
        cur.setdefault("crisp_model", "qwen3")
        cur.setdefault("crisp_qwen_quant", "q4")
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── 事件推送 ────────────────────────────────────────────
    def _emit(self, event: str, payload: dict):
        if self._on_event:
            try:
                self._on_event(event, payload)
            except Exception:
                pass

    # ── 模型加载（背景线程调用；支持热切换，可重复调用）──────────────
    def start_load(self):
        # 热切换：不再限制「整个进程只加载一次」。转录进行中或正在加载时
        # 忽略新的加载请求（前端同时会禁用按钮）。check-and-set 在锁内完成，
        # 防止并发请求双通过 guard、起两个加载线程。
        with self._load_gate:
            if self._loading or self._transcribing:
                return
            self._loading = True
        threading.Thread(target=self._load_worker, name="model-loader", daemon=True).start()

    @staticmethod
    def _release_engine(old):
        """显式释放旧引擎（OpenVINO 编译模型体量可观，不等 GC）。"""
        if old is None:
            return
        import gc
        try:
            old.ready = False
        except Exception:
            pass
        del old
        gc.collect()

    def _load_worker(self):
        # 依 settings.backend 全新加载对应引擎。支持热切换：
        # 持 self._lock 与转录互斥（转录也持此锁）。失败回退优先级：
        # ① 之前加载过的旧引擎（保留引用、ready 不动，加载成功后才释放）
        # → ② OpenVINO（若失败的不是它）。加载成功后清空 _load_err。
        from applog import log_model, log_error
        backend = self._persisted_backend()           # openvino / crispasr
        eng_label, model_label = self._current_selection()
        log_model(f"开始加载：{eng_label} · {model_label}（backend={backend}）")
        old_engine = getattr(self, "engine", None) if self._loaded else None
        try:
            with self._lock:
                try:
                    self.engine = None                    # 先摘下旧引擎（引用仍留在 old_engine）
                    if backend == "crispasr":
                        self._load_crispasr()
                    else:
                        backend = "openvino"
                        self._load_openvino()
                    self._loaded = True
                    self._active_backend = backend
                    self._remember_active(backend)
                    self._load_err = None                 # 成功加载后清掉上次失败的错误残留
                    self._release_engine(old_engine)      # 新引擎就绪，此时才释放旧的
                    old_engine = None
                    log_model(f"加载完成：{eng_label} · {model_label}")
                    self._emit("status", {"modelReady": True})
                except Exception as e:
                    traceback.print_exc()
                    log_error(f"模型加载失败（{eng_label} · {model_label}）", exc=e)
                    head = str(e).splitlines()[0][:110] if str(e) else type(e).__name__
                    # 加载失败回退优先级：① 之前加载过的旧引擎（还能用）→ ② OpenVINO（若失败的不是它）
                    if old_engine is not None:
                        self.engine = old_engine          # ready 未动，立即可继续转录
                        old_engine = None
                        self._loaded = True
                        self._load_err = f"{eng_label} · {model_label} 加载失败，已回退原引擎：{head}"
                        log_model(f"回退原引擎：{self._active_backend}")
                        self._emit("status", {"modelReady": True, "error": self._load_err})
                    elif backend != "openvino":
                        self._emit("progress", {"pct": 0, "status": f"{backend} 加载失败，改用 CPU 核心…"})
                        try:
                            self._load_openvino()
                            self._loaded = True
                            self._active_backend = "openvino"
                            self._remember_active("openvino")
                            self._load_err = f"{backend} 核心加载失败，已退回 CPU(OpenVINO)：{head}"
                            self._emit("status", {"modelReady": True, "error": self._load_err})
                        except Exception:
                            traceback.print_exc()
                    if self.engine is None:
                        self._load_err = str(e)
                        self._emit("status", {"modelReady": False, "error": str(e)})
        finally:
            self._loading = False

    # ── 各核心加载（重用既有引擎类别与 downloader）─────────────
    def _settings_raw(self) -> dict:
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            return json.loads(Path(f).read_text(encoding="utf-8")) if Path(f).exists() else {}
        except Exception:
            return {}

    def _vk_device_id(self, s: dict) -> int:
        import re
        m = re.search(r"GPU:(\d+)", s.get("device", "") or "")
        return int(m.group(1)) if m else 0

    # 每段最长秒数：使用者设置 → 套到当前引擎，但永远压到「模型天花板」。
    # 天花板＝引擎类别的 max_chunk_secs（0.6B 30、1.7B 10），由音频
    # 编码器导出长度写死，超过会被静默截断掉字，故只允许往短调，绝不超过。
    # crispasr（无 max_chunk_secs 类别属性）走自身窗口，不受此设置影响。
    _CHUNK_FLOOR = 5
    def _apply_chunk_secs(self) -> None:
        eng = getattr(self, "engine", None)
        ceiling = getattr(type(eng), "max_chunk_secs", None) if eng is not None else None
        if not isinstance(ceiling, int):
            return                                  # crispasr 等：不以 max_chunk_secs 分段
        try:
            want = int(self._settings_raw().get("chunk_secs", 0) or 0)
        except Exception:
            want = 0
        eff = ceiling if want <= 0 else max(self._CHUNK_FLOOR, min(want, ceiling))
        try:
            eng.max_chunk_secs = eff                # 实例属性遮盖类别默认，仅影响本实例
        except Exception:
            pass

    def _dl_progress(self, *a):
        """下载进度回调（容忍 progress_cb(frac) 或 progress_cb(frac, msg)）。"""
        frac = a[0] if a else 0
        msg = a[1] if len(a) > 1 else "下载中…"
        try:
            self._emit("progress", {"pct": int(float(frac) * 100), "status": str(msg)})
        except Exception:
            pass

    def _st(self, m):
        self._emit("progress", {"pct": 0, "status": m})

    def _load_openvino(self):
        s = self._settings_raw()
        model_dir = Path(s.get("model_dir", str(core._DEFAULT_MODEL_DIR)))
        use_17b = "1.7B" in s.get("cpu_model_size", "0.6B")
        cpu_threads = int(s.get("cpu_threads", 0) or 0)
        if use_17b:
            from downloader import quick_check_1p7b, download_1p7b
            if not quick_check_1p7b(model_dir):
                self._st("下载 1.7B 模型（约 4.3 GB）…")
                download_1p7b(model_dir, progress_cb=self._dl_progress)
        else:
            # 0.6B：文件缺失时按需补齐（含 VAD onnx）。此前只靠开发环境手工放置，
            # 干净机器上选 0.6B 会直接在 eng.load 崩溃（找不到 audio_encoder_model.xml）。
            from downloader import quick_check, download_all
            if not quick_check(model_dir):
                self._st("下载 0.6B 模型（约 1.2 GB）…")
                download_all(model_dir, progress_cb=self._dl_progress)
        eng = core.ASREngine1p7B() if use_17b else core.ASREngine()
        eng.load(device="CPU", model_dir=model_dir, cb=self._st, cpu_threads=cpu_threads)
        self.engine = eng

    def _load_crispasr(self):
        from crisp_engine import CrispWhisperEngine
        from downloader import (quick_check_crispasr, download_crispasr_core,
                                quick_check_whisper_ggml, download_whisper_ggml,
                                whisper_ggml_filename,
                                quick_check_aligner_gguf, download_aligner_gguf,
                                aligner_gguf_filename, crispasr_gpu_backend)
        s = self._settings_raw()
        crispasr_dir = self._crispasr_dir()
        crisp_model = s.get("crisp_model", "whisper")    # whisper / qwen3 / qwen3-ja
        quant = s.get("crisp_quant", "q5")
        fa_enabled = bool(s.get("crisp_fa", True))
        fa_quant = s.get("crisp_fa_quant", "q5")
        variant = self._crisp_variant()                  # 加速版本（vulkan/cuda…）

        # crispasr.exe 核心（即推理加速器，依 variant 对应 Vulkan/CUDA/CPU build）
        # 一律需要。带 variant 检查 → 版本升级或使用者改选加速版本时都会判定为
        # 「不就绪」而重新下载对应的 zip——即「下载并加载模型」时一并取得加速器。
        if not quick_check_crispasr(crispasr_dir, variant):
            self._st(f"下载 CrispASR 加速器（{variant}）…")
            download_crispasr_core(crispasr_dir, progress_cb=self._dl_progress,
                                   variant=variant)

        if crisp_model in _QWEN_CASR_MODELS:
            # Qwen3-ASR GGUF（与 OV 模型同置 ov_models/；crisp_engine 依文件名含 "1.7b"
            # 自动推断 --backend qwen3-1.7b）。标准版与日语动漫版共享此路径，仅来源/
            # 文件名不同（由 _qwen_casr_dl 分派）。Q4/Q8 两量化，缺文件自动下载（cstr repo）。
            fname_fn, check_fn, download_fn = _qwen_casr_dl(crisp_model)
            qquant = s.get("crisp_qwen_quant", "q8")
            model_dir = Path(s.get("model_dir", str(getattr(core, "_DEFAULT_MODEL_DIR",
                                                            BASE_DIR / "ov_models"))))
            model_path = model_dir / fname_fn(qquant)
            if not check_fn(model_dir, qquant):
                _note = _QWEN_CASR_NOTE.get(crisp_model, "")
                self._st(f"下载 Qwen3-ASR-1.7B{_note} {qquant.upper()} GGUF…")
                download_fn(model_dir, qquant, progress_cb=self._dl_progress)
        else:
            # OpenAI Whisper 官方 GGML（ggerganov/whisper.cpp）：whisper 后端原生支持。
            # 尺寸（base/small/medium/large/turbo）由 whisper_size 决定，缺文件自动下载。
            size = s.get("whisper_size", "base")
            model_path = crispasr_dir / whisper_ggml_filename(size)
            if not quick_check_whisper_ggml(crispasr_dir, size):
                self._st(f"下载 OpenAI Whisper（{size}）模型…")
                download_whisper_ggml(crispasr_dir, size, progress_cb=self._dl_progress)
        aligner_path = None
        if fa_enabled:
            if quick_check_aligner_gguf(crispasr_dir, fa_quant):
                aligner_path = crispasr_dir / aligner_gguf_filename(fa_quant)
            else:
                try:
                    self._st("下载时间轴对齐器…")
                    download_aligner_gguf(crispasr_dir, fa_quant, progress_cb=self._dl_progress)
                    aligner_path = crispasr_dir / aligner_gguf_filename(fa_quant)
                except Exception:
                    aligner_path = None       # 退回 Whisper 自带时间轴
        eng = CrispWhisperEngine()
        eng.load(model_path=model_path, crispasr_dir=crispasr_dir,
                 device_id=self._vk_device_id(s), cb=self._st, aligner_path=aligner_path,
                 gpu_backend=crispasr_gpu_backend(variant))
        self.engine = eng

    # ── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict:
        active = getattr(self, "_active_backend", "openvino")
        return {
            "modelReady": bool(getattr(self.engine, "ready", False)),
            "loading": self._loading,
            "transcribing": self._transcribing,          # 转录中 → 前端禁切换
            "error": self._load_err,
            "backend": self._backend_label(active, "CPU · OpenVINO INT8"),
            "backendKey": self._persisted_backend(),       # 已记住的核心选择
            "activeBackend": active,                        # 实际加载中的核心
            "device": "GPU" if active == "crispasr" else "CPU",
            "version": self._app_version(),
            "appName": "语音识别小工具",
            # 是否已有「任一组」模型下载完成（信息用）
            "hasAnyModel": self._has_any_model(),
            # 目前选择的模型是否已就绪 → 前端决定开始页（就绪=语音转文字／否=模型页等下载）
            "selectedReady": self.selected_model_present(),
        }

    def selected_model_present(self) -> bool:
        """目前持久化选择的『那一个』模型文件是否已下载。

        决定两件事：① 启动时是否自动加载（present→直接载；缺→不自动下载，等
        使用者在模型页选好、按「下载并加载」才触发，让老手能先改选 Whisper 等）；
        ② 前端开始页（present→语音转文字；缺→模型页）。
        """
        try:
            from downloader import (quick_check, quick_check_1p7b, quick_check_whisper_ggml,
                                    quick_check_crispasr)
            s = self._settings_raw()
            be = s.get("backend", "openvino")
            model_dir = self._model_dir()
            if be == "openvino":
                return (quick_check_1p7b(model_dir)
                        if "1.7B" in s.get("cpu_model_size", "0.6B") else quick_check(model_dir))
            if be == "crispasr":
                cd = self._crispasr_dir()
                if not quick_check_crispasr(cd):
                    return False                  # 核心 exe 缺（理论上随包附带）→ 当作未就绪
                cm = s.get("crisp_model", "whisper")
                if cm in _QWEN_CASR_MODELS:
                    _, check_fn, _ = _qwen_casr_dl(cm)
                    return check_fn(model_dir, s.get("crisp_qwen_quant", "q8"))
                return quick_check_whisper_ggml(cd, s.get("whisper_size", "base"))
        except Exception:
            traceback.print_exc()
        return False

    def selected_model_present_for(self, backend: str, patch: dict) -> bool:
        """给定目录项（backend + settings patch）的模型文件是否已下载。

        CLI `profiles` 用：与 selected_model_present() 同样的判断，但吃
        「任意一项」而非目前 settings 里选定的。
        """
        try:
            from downloader import (quick_check, quick_check_1p7b, quick_check_whisper_ggml,
                                    quick_check_crispasr)
            s = patch
            model_dir = self._model_dir()
            if backend == "openvino":
                return (quick_check_1p7b(model_dir)
                        if "1.7B" in s.get("cpu_model_size", "0.6B") else quick_check(model_dir))
            if backend == "crispasr":
                cd = self._crispasr_dir()
                if not quick_check_crispasr(cd):
                    return False
                cm = s.get("crisp_model", "whisper")
                if cm in _QWEN_CASR_MODELS:
                    _, check_fn, _ = _qwen_casr_dl(cm)
                    return check_fn(model_dir, s.get("crisp_qwen_quant", "q8"))
                return quick_check_whisper_ggml(cd, s.get("whisper_size", "base"))
        except Exception:
            traceback.print_exc()
        return False

    def _has_any_model(self) -> bool:
        """机器上是否已有任一可用模型（任一核心任一量化）。供首启页面决策。"""
        try:
            from downloader import (quick_check, quick_check_1p7b, quick_check_whisper_ggml,
                                    quick_check_qwen3_asr_gguf, quick_check_qwen3_asr_ja_gguf)
            model_dir = self._model_dir()
            crispasr_dir = self._crispasr_dir()
            if quick_check(model_dir) or quick_check_1p7b(model_dir):
                return True
            for q in ("base", "small", "medium", "large", "turbo"):
                if quick_check_whisper_ggml(crispasr_dir, q):
                    return True
            for q in ("q4", "q8"):
                if quick_check_qwen3_asr_gguf(model_dir, q) or quick_check_qwen3_asr_ja_gguf(model_dir, q):
                    return True
            # 旧版遗留的 chatllm ASR .bin（无对应核心，仅作「已有模型」判定）
            if (model_dir / "qwen3-asr-1.7b.bin").exists():
                return True
        except Exception:
            traceback.print_exc()
        return False

    def _persisted_backend(self) -> str:
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            if Path(f).exists():
                return json.loads(Path(f).read_text(encoding="utf-8")).get("backend", "openvino")
        except Exception:
            pass
        return "openvino"

    def _app_version(self) -> str:
        try:
            import version
            # WebView 版优先用独立版本字符串（webview 0.1）；缺则退回语义化版本。
            return getattr(version, "WEBVIEW_VERSION", None) or version.__version__
        except Exception:
            return ""

    # ── 转录 ────────────────────────────────────────────────
    # opts: {path, language, diarize, nSpeakers, align, hint}
    # progress_cb(pct:int, status:str)
    def transcribe(self, opts: dict, progress_cb=None) -> dict:
        import time as _time
        from applog import log_transcribe
        if not getattr(self.engine, "ready", False):
            raise RuntimeError("模型尚未加载完成，请稍候再试。")
        # 忙碌 check-and-set（独立小锁；self._lock 会在整个引擎调用期间被持有，
        # 不能用来做这道 gate——否则两道门互相等死锁）。并发第二请求直接拒绝，
        # 且只有置位者能在 finally 里清 flag，避免先结束者清掉后到者的忙碌状态。
        if not self._transcribe_gate.acquire(blocking=False):
            raise RuntimeError("已有识别任务进行中，请稍候再试。")
        path = opts.get("path")
        if not path or not Path(path).exists():
            raise RuntimeError("找不到音频文件。")
        self._cancel = False
        self._transcribing = True
        _t0 = _time.monotonic()
        eng_label, model_label = getattr(self, "_active_label", None) or self._current_selection()
        log_transcribe(f"开始转录：{Path(path).name}"
                       f"（{eng_label} · {model_label}，"
                       f"语言={opts.get('language') or '自动'}"
                       f"{'，分离' if opts.get('diarize') else ''}）")

        def _cb(i, total, msg):
            if progress_cb:
                pct = min(99, int((i / max(total, 1)) * 100))
                progress_cb(pct, msg)

        # 视频 → 先抽音轨
        audio_path = Path(path)
        tmp_extra = None
        try:
            ext = audio_path.suffix.lower()
            from ffmpeg_utils import VIDEO_EXTS
            if ext in VIDEO_EXTS:
                from ffmpeg_utils import extract_audio_to_wav
                ff = self._ensure_ffmpeg()
                if not ff:
                    raise RuntimeError("上传为视频但找不到 ffmpeg，且自动下载失败，无法抽音轨。")
                wav = audio_path.with_suffix(".extracted.wav")
                extract_audio_to_wav(audio_path, wav, ff)
                audio_path, tmp_extra = wav, wav

            want_align = bool(opts.get("align", True))
            want_diar = bool(opts.get("diarize"))

            # ── 时间轴对齐（FA）：缺模型则按需下载（移植自 app.py 的协调层）──
            if want_align:
                self._ensure_fa()
            if hasattr(self.engine, "use_aligner") and getattr(self.engine, "_fa_bin", None):
                self.engine.use_aligner = want_align

            # ── 说话者分离：外部 ONNX，与后端无关。缺模型则下载 + 挂上 diar_engine ──
            if want_diar:
                if not (getattr(self.engine, "diar_engine", None)
                        and getattr(self.engine.diar_engine, "ready", False)):
                    try:
                        self._ensure_diarization()
                    except Exception as e:
                        head = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
                        self._emit("progress", {"pct": 0, "status": f"说话者分离模型下载失败：{head}"})

            n_spk = opts.get("nSpeakers")
            n_speakers = int(n_spk) if str(n_spk).isdigit() else None

            # 语言：空字符串 / 自动检测 / auto → None（让引擎自动检测）
            _lang = (opts.get("language") or "").strip()
            if _lang in ("", "自动检测", "auto"):
                _lang = None

            # 输出落点：subtitles/ 下、用「原始上传文件名」命名（可在「开启输出文件夹」找到）。
            # 上传文件名为使用者可控（含 LAN/外网端点）→ 必须只取基名，杜绝路径穿越。
            srt_dir = Path(getattr(core, "SRT_DIR", BASE_DIR / "subtitles"))
            srt_dir.mkdir(parents=True, exist_ok=True)
            raw_name = opts.get("name") or Path(path).name
            out_name = Path(str(raw_name).replace("\\", "/")).name   # 去除任何目录成分
            if not out_name or out_name.startswith("."):
                out_name = "transcript"
            # 文件名拼接「实际加载」的引擎与模型（非 settings 里记住的选择——
            # 忙碌期间选择可被改而引擎未换，标签必须反映真正跑的引擎）：
            # 唯一.flac → 唯一 [CrispASR · Whisper Base].flac
            # （标签安全化：去路径非法字符、压空白；「.」换「·」——无扩展名文件的
            # out_ref 以「…0.6B]」结尾时，write_transcript 的 ref.stem 与切片路径的
            # with_suffix 会把它当扩展名剥掉，产生残缺文件名）
            tag = _sanitize_tag(f"{eng_label} · {model_label}").replace(".", "·")
            p = Path(out_name)
            out_name = f"{p.stem} [{tag}]{p.suffix}" if p.suffix else f"{p.stem} [{tag}]"
            out_ref = srt_dir / out_name
            # 防呆：确认解析后仍在 subtitles/ 内
            try:
                if srt_dir.resolve() not in out_ref.resolve().parents:
                    out_ref = srt_dir / "transcript"
            except Exception:
                out_ref = srt_dir / "transcript"

            with self._lock:
                self._apply_chunk_secs()      # 依设置确定本次 max_chunk_secs（压到模型上限）
                # 清除上一轮字级残留，避免本轮若提前 return 时读到旧数据
                try:
                    self.engine._last_segments_rich = None
                except Exception:
                    pass

                # ── 长音频切片：超过阈值先用 ffmpeg 切片再逐片转录 ──────────
                ff = self._resolve_ffmpeg()
                dur = _ffprobe_duration(ff, audio_path) if ff else None
                if dur and dur > _SLICE_THRESHOLD_SECS:
                    srt = self._transcribe_sliced(
                        audio_path, dur, out_ref, _cb, _lang, want_diar, n_speakers)
                else:
                    srt = self.engine.process_file(
                        audio_path,
                        progress_cb=_cb,
                        language=_lang,
                        context=(opts.get("hint") or "").strip() or None,
                        diarize=want_diar,
                        n_speakers=n_speakers,
                        original_path=out_ref,
                        out_format="srt",
                    )
        finally:
            self._transcribing = False
            try:
                self._transcribe_gate.release()   # 只有置位者会走到这里（acquire 失败早已抛出）
            except RuntimeError:
                pass
            if tmp_extra:
                try:
                    Path(tmp_extra).unlink(missing_ok=True)
                except Exception:
                    pass

        if not srt:
            diag = getattr(self.engine, "_last_vad_diag", None)
            log_transcribe(f"转录未产生字幕：{Path(path).name}（{diag or '未检测到人声'}）")
            raise RuntimeError(diag or "未产生字幕（未检测到人声）。")

        # UI 永远以内存中的 segments 渲染波形/字幕卡（需时间轴），故引擎固定产 SRT。
        # 但「输出格式」设置决定使用者实际取得的存档文件：选纯文字 → 另写 .txt、移除 .srt。
        # 优先用引擎的字级结构（含 words，驱动卡拉OK）；无则退回解析 SRT（行级）。
        rich = getattr(self.engine, "_last_segments_rich", None)
        segments = _rich_to_segments(rich) if rich else parse_srt_to_segments(srt)
        saved = str(srt)
        out_fmt = (self._settings_raw().get("output_format", "srt") or "srt").lower()
        if out_fmt == "txt":
            try:
                from subtitle_lines import write_transcript
                lines = [(seg["start"], seg["end"], seg["text"],
                          (f'说话者{seg["speaker"]}' if seg.get("speaker") else None))
                         for seg in segments]
                txt = write_transcript(Path(srt), lines, out_format="txt")
                try:
                    Path(srt).unlink(missing_ok=True)
                except Exception:
                    pass
                saved = str(txt)
            except Exception:
                traceback.print_exc()          # 退回 SRT，不让转录整体失败

        log_transcribe(
            f"转录完成：{Path(path).name} → {Path(saved).name}"
            f"（{len(segments)} 段，耗时 {_time.monotonic() - _t0:.1f}s，"
            f"{eng_label} · {model_label}）")
        if progress_cb:
            progress_cb(100, "完成")
        return {"segments": segments, "srtPath": saved}

    def _resolve_ffmpeg(self):
        """解析 ffmpeg：优先采用设置中手动指定的 ffmpeg_path，否则自动检测。

        与 setting.py 的检测顺序一致（使用者手动指定 → 系统 PATH / App 目录）。
        """
        from ffmpeg_utils import find_ffmpeg
        p = (self._settings_raw().get("ffmpeg_path", "") or "").strip()
        if p and Path(p).exists():
            return Path(p)
        return find_ffmpeg()

    # ── 长音频切片转录 ──────────────────────────────────────────────────
    def _transcribe_sliced(self, audio_path: Path, duration: float,
                           out_ref: Path, _cb, lang, want_diar, n_speakers):
        """超过阈值的长音频：ffmpeg 切片 → 逐片转录 → 合并 SRT。

        每片独立调 engine.process_file（重叠 15 秒防切点截字），产出各自 SRT
        后解析成 segments，合并去重，最后一次性写出整份 SRT（经 write_transcript
        以 out_ref 命名落盘）。任一片失败即中止（与整段转录的失败语义一致）。
        """
        import subprocess
        import tempfile
        from ffmpeg_utils import _NO_WINDOW
        from subtitle_lines import write_transcript

        ff = self._resolve_ffmpeg()
        if not ff:
            raise RuntimeError("长音频切片需要 ffmpeg，且自动下载失败。")

        plan = _slice_plan(duration)
        all_segments: list[list[dict]] = []

        with tempfile.TemporaryDirectory(prefix="asr_slice_") as td:
            tmp = Path(td)
            for i, (start, end) in enumerate(plan, 1):
                if self._cancel:
                    raise RuntimeError("已取消。")
                seg_wav = tmp / f"slice_{i:03d}.wav"
                cmd = [str(ff), "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                       "-i", str(audio_path), "-vn", "-ar", "16000", "-ac", "1",
                       "-f", "wav", str(seg_wav)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      creationflags=_NO_WINDOW)
                if proc.returncode != 0 or not seg_wav.exists():
                    err = proc.stderr.decode(errors="replace").splitlines()
                    raise RuntimeError(
                        f"ffmpeg 切片失败（片 {i}/{len(plan)}）："
                        f"{err[-1] if err else '未知错误'}")

                def _slice_cb(done, total, msg, _i=i, _n=len(plan)):
                    # engine 的回调是 (i, total, msg) 三参；这里先把片内进度换算成
                    # 「全局百分比」再以 (pct, 100, msg) 交给 _cb——直接转发 done/total
                    # 会让进度条在每片边界从 0 重爬（N 片来回跳 N 次）。
                    base = (_i - 1) / _n
                    span = 1.0 / _n
                    frac = (done / max(total, 1)) if total else 0
                    _cb(min(99, int((base + span * frac) * 100)), 100,
                        f"切片转录 {msg}（第 {_i}/{_n} 片）")

                part_srt = self.engine.process_file(
                    seg_wav,
                    progress_cb=_slice_cb,
                    language=lang,
                    context=None,
                    diarize=want_diar,
                    n_speakers=n_speakers,
                    original_path=None,
                    out_format="srt",
                )
                if not part_srt:
                    continue      # 该片无声 → 跳过（如纯音乐片段）
                # 片内时间已是绝对时间（-ss 放在 -i 前 = 输入侧定位，
                # 输出的 pts 从 0 起算，故把片起点平移回全局时间轴）
                part_segs = _rich_to_segments(
                    getattr(self.engine, "_last_segments_rich", None)) \
                    if getattr(self.engine, "_last_segments_rich", None) \
                    else parse_srt_to_segments(part_srt)
                for seg in part_segs:
                    seg["start"] += start
                    seg["end"] += start
                    for w in (seg.get("words") or []):
                        w["start"] = w.get("start", 0) + start
                        w["end"] = w.get("end", 0) + start
                all_segments.append(part_segs)

        merged = _merge_sliced_segments(all_segments)
        if not merged:
            return None
        lines = [(seg["start"], seg["end"], seg["text"],
                  (f'说话者{seg["speaker"]}' if seg.get("speaker") else None))
                 for seg in merged]
        out = out_ref.with_suffix(".srt")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for idx, (s0, e0, text, spk) in enumerate(merged, 1):
                prefix = f"{spk}：" if spk else ""
                f.write(f"{idx}\n{_srt_ts_fmt(s0)} --> {_srt_ts_fmt(e0)}\n"
                        f"{prefix}{text}\n\n")
        return out

    def _ensure_ffmpeg(self):
        """返回可用的 ffmpeg；找不到时自 HF 下载精简 zip 解压后再找。

        ffmpeg 不随安装包附带 → 首次需要（转录视频）时自动下载到 <app>/ffmpeg/。
        下载进度经 self._dl_progress → SSE 推给前端。失败回 None（由上层报错）。
        """
        ff = self._resolve_ffmpeg()
        if ff:
            return ff
        try:
            from ffmpeg_utils import get_default_ffmpeg_dest
            from downloader import quick_check_ffmpeg, download_ffmpeg
            dest = get_default_ffmpeg_dest()
            if not quick_check_ffmpeg(dest):
                self._st("下载 ffmpeg（视频音轨提取）…")
                download_ffmpeg(dest, progress_cb=self._dl_progress)
            return self._resolve_ffmpeg()
        except Exception:
            traceback.print_exc()
            return None

    def _model_dir(self) -> Path:
        s = self._settings_raw()
        return Path(s.get("model_dir", str(getattr(core, "_DEFAULT_MODEL_DIR",
                                                   BASE_DIR / "ov_models"))))

    # ── 按需下载：说话者分离 / FA（移植自 app.py 的 CTk 下载协调层）──────────
    def _ensure_diarization(self) -> bool:
        """确保说话者分离 ONNX 模型存在并把 diar_engine 挂到当前引擎。

        diar 是与后端无关的外部 ONNX，OpenVINO/CRISPASR 共享。缺模型则自动
        下载（约 32 MB），完成后建立 DiarizationEngine 挂上 self.engine.diar_engine。
        返回是否就绪。
        """
        from downloader import quick_check_diarization, download_diarization
        from diarize import DiarizationEngine
        model_dir = self._model_dir()
        diar_dir = model_dir / "diarization"
        if not quick_check_diarization(model_dir):
            self._st("下载说话者分离模型（约 32 MB）…")
            download_diarization(diar_dir, progress_cb=self._dl_progress)
        eng = DiarizationEngine(diar_dir)
        if getattr(eng, "ready", False):
            try:
                self.engine.diar_engine = eng
            except Exception:
                pass
            return True
        return False

    def _ensure_fa(self) -> bool:
        """确保「时间轴对齐」模型存在（移植自 app.py _check_aligner_model）。

        CRISPASR：FA aligner gguf 已于 _load_crispasr 处理，这里略过。
        OpenVINO：需 chatllm ForcedAligner .bin（约 939 MB），缺则下载并
        重新加载引擎内的对齐器（eng._load_aligner）。返回是否就绪。
        """
        backend = getattr(self, "_active_backend", "openvino")
        if backend == "crispasr":
            return bool(getattr(self.engine, "_fa_bin", None))
        try:
            from downloader import quick_check_aligner, download_aligner
            model_dir = self._model_dir()
            if not quick_check_aligner(model_dir):
                self._st("下载时间轴对齐模型（约 939 MB）…")
                download_aligner(model_dir, progress_cb=self._dl_progress)
            if hasattr(self.engine, "_load_aligner"):
                self.engine._load_aligner(cb=self._st)
            return bool(getattr(self.engine, "_fa_bin", None)
                        or getattr(self.engine, "use_aligner", False))
        except Exception:
            traceback.print_exc()
            return False

    def cancel(self):
        self._cancel = True
        return True

    def has_running_tasks(self) -> bool:
        """是否有不可中断丢失的工作进行中（转录／模型下载加载／录音）。"""
        return bool(self._transcribing or self._loading or self._recording)

    def running_task_label(self) -> str:
        """当前进行中任务的可读描述（弹窗正文用）；无任务回空串。"""
        if self._transcribing:
            return "正在识别音频"
        if self._loading:
            return "正在下载／加载模型"
        if self._recording:
            return "正在录音"
        return ""

    def set_recording(self, on: bool):
        """前端录音开始／结束 → 同步到后端（关窗确认依据之一）。"""
        self._recording = bool(on)

    def open_output_dir(self) -> bool:
        import os
        try:
            d = getattr(core, "SRT_DIR", BASE_DIR / "subtitles")
            Path(d).mkdir(parents=True, exist_ok=True)
            os.startfile(str(d))
            return True
        except Exception:
            return False

    def browse_folder(self, start: str = "", title: str = "选择文件夹") -> dict:
        """弹出原生「选择文件夹」对话框（tkinter，无主窗口也可靠）。

        返回 {"ok": bool, "path": str}；用户取消 → ok=False。
        在后台线程里被 HTTP handler 调用是安全的：tkinter 对话框自建
        隐藏根窗口、自带消息循环，调用返回即销毁。
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                start_dir = start if start and Path(start).exists() \
                    else str(self._model_dir())
                path = filedialog.askdirectory(
                    title=title, initialdir=start_dir, parent=root)
            finally:
                root.destroy()
            return {"ok": bool(path), "path": path or ""}
        except Exception as e:
            return {"ok": False, "path": "", "error": str(e)}

    def browse_file(self, start: str = "", title: str = "选择文件") -> dict:
        """弹出原生「选择文件」对话框（用于 FFmpeg 可执行文件路径）。

        返回 {"ok": bool, "path": str}；用户取消 → ok=False。
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                initial = Path(start) if start and Path(start).exists() else None
                path = filedialog.askopenfilename(
                    title=title, initialdir=str(initial.parent) if initial else "",
                    initialfile=initial.name if initial else "",
                    filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")],
                    parent=root)
            finally:
                root.destroy()
            return {"ok": bool(path), "path": path or ""}
        except Exception as e:
            return {"ok": False, "path": "", "error": str(e)}

    def open_releases(self) -> dict:
        """「检查更新」→ 以系统浏览器开启 GitHub Releases 页。"""
        import webbrowser
        try:
            import version
            url = getattr(version, "GITHUB_RELEASES_PAGE", "")
        except Exception:
            url = ""
        if not url:
            return {"ok": False, "url": "", "error": "未配置更新链接"}
        try:
            webbrowser.open(url)
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "url": url, "error": str(e)}

    # ── 模型下拉（核心 + 模型）＋ 识别语言 ─────────────────────────────
    #   持久化后走「切换=重启」(同 set_backend 的理由：避免就地切核心死机)。
    def _remember_active(self, backend: str):
        """记住「实际加载」的选择识别码与标签（字幕文件名/日志标注用）。"""
        self._active_backend = backend
        self._active_identity = self._identity_for(backend, self._settings_raw())
        self._active_label = self._current_selection()

    def _identity_for(self, backend: str, s: dict):
        """把 (backend + 相关 settings) 化为可比较的识别码（决定要不要重启）。"""
        if backend == "openvino":
            return ("openvino", s.get("cpu_model_size", "0.6B"))
        if backend == "crispasr":
            cm = s.get("crisp_model", "whisper")
            if cm == "whisper":
                extra = s.get("whisper_size", "base")
            else:
                extra = s.get("crisp_qwen_quant", "q8")   # qwen3 量化也纳入识别
            # 加速版本也纳入识别：换 Vulkan↔CUDA 等同换核心，需重启才生效
            return ("crispasr", cm, extra, self._crisp_variant())
        return ("openvino",)

    def _current_selection(self):
        """settings → (引擎标签, 模型标签)，供下拉预选。"""
        s = self._settings_raw()
        be = s.get("backend", "openvino")
        if be == "crispasr":
            cm = s.get("crisp_model", "whisper")
            if cm == "qwen3-ja":
                qq = s.get("crisp_qwen_quant", "q8")
                return (_ENGINE_CASR, _QWEN_JA_CASR_QUANT_LABEL.get(qq, _QWEN_JA_CASR_QUANT_LABEL["q8"]))
            if cm == "qwen3":
                qq = s.get("crisp_qwen_quant", "q8")
                return (_ENGINE_CASR, _QWEN_CASR_QUANT_LABEL.get(qq, _QWEN_CASR_QUANT_LABEL["q8"]))
            size = s.get("whisper_size", "base")
            if size not in _WHISPER_SIZES:
                size = "base"
            return (_ENGINE_CASR, _WHISPER_LABEL_BY_SIZE[size])
        sz = s.get("cpu_model_size", "0.6B")
        return (_ENGINE_OV, "Qwen3-ASR-1.7B INT8" if "1.7B" in sz else "Qwen3-ASR-0.6B")

    _BACKENDS = {0: "openvino", 2: "crispasr"}
    _BACKEND_LABELS = {
        "openvino": "CPU · OpenVINO INT8",
        # CRISPASR 的加速版本可切（Vulkan/CUDA/CPU）→ 实际标签由 _backend_label()
        # 依 crisp_variant 补上；这里只留字典的默认值供旧调用端使用。
        "crispasr": "GPU · CRISPASR",
    }

    @staticmethod
    def _note_for(model_label: str = "") -> str:
        """该模型要显示的提醒文字；没有则空字符串。"""
        return _MODEL_NOTES.get(model_label, "")

    def _backend_label(self, backend: str, fallback: str = "") -> str:
        """核心显示名。CRISPASR 会补上目前选定的加速版本（Vulkan／CUDA／CPU）。"""
        label = self._BACKEND_LABELS.get(backend)
        if label is None:
            return fallback or backend
        if backend == "crispasr":
            try:
                from downloader import crispasr_variants
                v = self._crisp_variant()
                meta = crispasr_variants().get(v)
                if meta:
                    kind = "CPU" if meta["gpu_backend"] is None else "GPU"
                    return f"{kind} · CRISPASR（{meta.get('short') or meta['label']}）"
            except Exception:
                pass
        return label

    def set_backend(self, idx) -> dict:
        # 热切换：记住选择即可，实际加载由前端「下载并加载模型」触发。
        backend = self._BACKENDS.get(int(idx) if str(idx).isdigit() else 0, "openvino")
        label = self._backend_label(backend)
        self._persist_backend(backend)
        active = getattr(self, "_active_backend", "openvino")
        if backend == active:
            return {"ok": True, "backend": backend, "restartRequired": False,
                    "message": f"「{label}」已是目前使用的核心。"}
        return {"ok": True, "backend": backend, "restartRequired": False,
                "message": (f"已选定「{label}」。点「下载并加载模型」即可热切换"
                            f"（首次启用会自动下载对应模型）。")}

    def _persist_backend(self, backend: str):
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        cur["backend"] = backend
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_model_options(self) -> dict:
        """核心/模型阶层 + 目前选择 + 每模型对应架构标签（前端渲染下拉用）。"""
        cur_core, cur_model = self._current_selection()
        order, by_core = [], {}
        for core_label, model_label, be, _patch in _MODEL_CATALOG:
            if core_label not in by_core:
                by_core[core_label] = []
                order.append(core_label)
            by_core[core_label].append({
                "label": model_label, "backend": be,
                "arch": self._backend_label(be, be),
                "note": self._note_for(model_label),
            })
        active = getattr(self, "_active_backend", None)
        return {
            "cores": [{"label": c, "models": by_core[c]} for c in order],
            "current": {"core": cur_core, "model": cur_model},
            "activeArch": self._backend_label(active, "") if active else "",
        }

    def set_model(self, core_label: str, model_label: str) -> dict:
        """选定 (核心,模型) → 写对应 settings 键、回是否需重启。"""
        entry = next((e for e in _MODEL_CATALOG
                      if e[0] == core_label and e[1] == model_label), None)
        if entry is None:   # 防呆：退回该核心首项 / 目录首项
            entry = next((e for e in _MODEL_CATALOG if e[0] == core_label),
                         _MODEL_CATALOG[0])
        core_label, model_label, backend, patch = entry

        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        cur["backend"] = backend
        cur.update(patch)                 # cpu_model_size / crisp_model / crisp_quant
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        arch = self._backend_label(backend, backend)
        # 热切换：任何时刻都支持就地切换。转录/加载进行中 → 前端禁用切换
        # （canLoadNow=False），此处置灰只影响「立即加载」，选择本身仍被记住。
        identity_changed = (self._identity_for(backend, cur)
                            != getattr(self, "_active_identity", None))
        busy = self._loading or self._transcribing
        can_load_now = not busy
        if identity_changed:
            if busy:
                msg = (f"已记住「{core_label} · {model_label}」（{arch}）。"
                       f"当前有任务进行中，完成后可点「下载并加载模型」热切换。")
            else:
                msg = (f"已选定「{core_label} · {model_label}」（{arch}）。"
                       f"点「下载并加载模型」即可热切换（不需重启）。")
        else:
            msg = f"「{core_label} · {model_label}」已是目前使用的模型。"
        note = self._note_for(model_label)
        if note:                          # 模型提示 → 一并提醒
            msg += "\n" + note
        return {
            "ok": True, "core": core_label, "model": model_label,
            "backend": backend, "arch": arch, "restartRequired": False,
            "canLoadNow": can_load_now,
            "identityChanged": identity_changed,
            "note": note,
            "message": msg,
        }

    def request_load(self) -> dict:
        """前端「下载并加载模型」→ 触发背景加载（读已持久化的 backend 选择）。

        仅在尚未加载时有效（start_load 内含 guard）。下载/加载进度经 SSE
        "progress" 事件推送，完成时 "status" 事件 modelReady=True。
        """
        already = self._loaded or self._loading
        self.start_load()
        return {"ok": True, "loading": True, "alreadyLoaded": bool(self._loaded),
                "wasBusy": bool(already)}

    def get_languages(self) -> dict:
        """识别语言清单：OpenVINO 用 processor.supported_languages，其余用常用清单。

        value="" 代表自动检测（transcribe 会转成 None）；其余 value 即引擎接受的
        语言字符串（如 "Chinese"）。
        """
        langs = None
        try:
            proc = getattr(self.engine, "processor", None)
            if proc is not None and getattr(proc, "supported_languages", None):
                langs = list(proc.supported_languages)
        except Exception:
            langs = None
        if not langs:
            langs = _COMMON_LANGS
        return {"languages": [{"label": "自动检测", "value": ""}]
                + [{"label": l, "value": l} for l in langs]}

    def _crispasr_dir(self) -> Path:
        # 用 core.BASE_DIR（frozen 时 = sys.executable 旁；非 webview_backend 的
        # __file__，后者在 onefile 冻结后指向 _MEIPASS 临时目录）→ 才找得到随包
        # 附带／执行时下载到 exe 旁的 crispasr/ 核心。
        s = self._settings_raw()
        app_dir = getattr(core, "BASE_DIR", BASE_DIR)
        return Path(s.get("crispasr_dir", str(app_dir / "crispasr")))

    def _crisp_variant(self) -> str:
        """目前选定的加速版本；settings 没有就依硬件推荐一个并记住。"""
        from downloader import (crispasr_variants, recommend_crispasr_variant)
        s = self._settings_raw()
        v = s.get("crisp_variant")
        if v in crispasr_variants():
            return v
        try:
            v = recommend_crispasr_variant()["recommended"]
        except Exception:
            v = "vulkan"
        self._persist_setting("crisp_variant", v)
        return v

    def get_accel(self) -> dict:
        """返回硬件检测结果 + 加速版本菜单（供前端「推理加速版本」区块渲染）。

        加速版本只对 CrispASR 引擎有意义：选 OpenVINO 时 applicable=False
        （前端据此隐藏该区块），CrispASR 时列出全部 Vulkan/CUDA/CPU build。
        """
        from downloader import (recommend_crispasr_variant, installed_crispasr_info,
                                _CRISPASR_VERSION)
        backend = self._persisted_backend()
        if backend != "crispasr":
            return {"ok": True, "applicable": False, "engine": backend,
                    "options": [], "selected": None, "recommended": None,
                    "reason": "", "hardware": None, "installed": None,
                    "latest": _CRISPASR_VERSION, "needsDownload": False}
        try:
            info = recommend_crispasr_variant()
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "applicable": True, "engine": backend,
                    "error": f"硬件检测失败：{e}", "options": []}
        installed = installed_crispasr_info(self._crispasr_dir())
        selected  = self._crisp_variant()
        return {
            "ok": True,
            "applicable": True,
            "engine": backend,
            "selected": selected,
            "recommended": info["recommended"],
            "reason": info["reason"],
            "options": info["options"],
            "hardware": info["hardware"],
            "installed": installed,          # {"version","variant"}；None＝尚未安装
            "latest": _CRISPASR_VERSION,
            # 已装但版本／版本别不符 → 前端提示「加载时会自动重新下载」
            "needsDownload": (installed.get("version") != _CRISPASR_VERSION
                              or installed.get("variant") != selected),
        }

    def set_accel(self, variant: str) -> dict:
        """记住使用者选的加速版本。实际下载发生在下次加载模型时。"""
        from downloader import crispasr_variants, installed_crispasr_info, _CRISPASR_VERSION
        variants = crispasr_variants()
        if variant not in variants:
            return {"ok": False, "message": f"未知的加速版本：{variant}"}
        self._persist_setting("crisp_variant", variant)
        meta = variants[variant]
        installed = installed_crispasr_info(self._crispasr_dir())
        fresh = (installed.get("version") == _CRISPASR_VERSION
                 and installed.get("variant") == variant)
        if fresh:
            return {"ok": True, "variant": variant, "restartRequired": False,
                    "message": f"已选用「{meta['label']}」（核心已就绪）。"}
        return {"ok": True, "variant": variant, "restartRequired": False,
                "message": (f"已记住「{meta['label']}」。点「下载并加载模型」时会一并下载该加速器"
                            f"（约 {meta['size_mb']} MB）并热切换。")}

    def list_devices(self) -> dict:
        import platform
        devices = [{"kind": "cpu", "name": platform.processor() or "CPU", "note": "使用中"}]
        diag = {"level": None, "text": ""}

        vk, source = self._probe_gpu_devices()
        try:
            for d in (vk.get("devices") if vk else []) or []:
                gb = d.get("vram_free", 0) / (1024 ** 3)
                devices.append({"kind": "gpu", "name": d.get("name", "GPU"),
                                "note": f"{gb:.1f} GB 可用" if gb else ""})
            if source is None:
                # 没有可用的检测器（crispasr 未就位）
                diag = {"level": "info",
                        "text": "GPU 检测需要 CrispASR 核心；启用 GPU 核心时会自动下载，"
                                "之后即可在此列出可用的独立显卡。目前仅 CPU 推理可用。"}
            elif vk and vk.get("error"):
                diag = {"level": "warn",
                        "text": f"GPU 检测未完成（{source}）：{vk['error']}　已自动改用 CPU 推理。"}
            elif not (vk and vk.get("devices")):
                diag = {"level": "info", "text": "未检测到可用的独立 GPU，仅 CPU 推理可用。"}
        except Exception as e:
            diag = {"level": "warn", "text": f"GPU 检测例外：{e}"}
        return {"devices": devices, "diag": diag}

    def _probe_gpu_devices(self):
        """返回 (探测结果 dict, 来源标签)；找不到检测器时回 (None, None)。

        走 CrispASR（crispasr.exe --diagnostics）。
        """
        try:
            from crisp_engine import probe_crispasr_devices, _find_exe
            cd = self._crispasr_dir()
            if _find_exe(cd):                       # 直接或子文件夹找到 crispasr.exe
                return probe_crispasr_devices(cd), "CrispASR"
        except Exception:
            traceback.print_exc()
        return None, None

    # ── 启动自检：每核心 × 每能力，实际探测文件/键路是否就绪 ─────────────
    #   status：green=已就绪 / yellow=缺但会自动下载 / red=缺且需处理 / na=不适用
    def health_check(self) -> dict:
        from downloader import (quick_check, quick_check_1p7b, quick_check_aligner,
                                quick_check_diarization, quick_check_crispasr,
                                quick_check_whisper_ggml, quick_check_qwen3_asr_gguf,
                                quick_check_qwen3_asr_ja_gguf, quick_check_aligner_gguf,
                                crispasr_variants, _CRISPASR_VERSION)
        s = self._settings_raw()
        model_dir = self._model_dir()
        crispasr_dir = self._crispasr_dir()

        def item(key, label, ok, ok_d, miss_d, downloadable=True):
            return {"key": key, "label": label,
                    "status": "green" if ok else ("yellow" if downloadable else "red"),
                    "detail": ok_d if ok else miss_d}

        def _vad_ok():
            try:
                if getattr(self.engine, "vad_sess", None) is not None:
                    return True
                base = Path(getattr(core, "_MEIPASS", BASE_DIR)) if getattr(core, "_MEIPASS", None) else BASE_DIR
                for p in (model_dir / "silero_vad_v4.onnx",
                          BASE_DIR / "ov_models" / "silero_vad_v4.onnx"):
                    if p.exists():
                        return True
            except Exception:
                pass
            return False

        # 共享：说话者分离（外部 ONNX，所有核心共享）＋ ffmpeg
        diar_ok = quick_check_diarization(model_dir)
        ff = self._resolve_ffmpeg()

        # OpenVINO（CPU）路径
        ov_size = s.get("cpu_model_size", "0.6B")
        ov_model_ok = quick_check_1p7b(model_dir) if "1.7B" in ov_size else quick_check(model_dir)
        ov = {
            "label": "OpenVINO（CPU：Qwen）", "backend": "openvino",
            "items": [
                item("model", f"ASR 模型（{ov_size}）", ov_model_ok,
                     "已下载", "未下载（启用时自动下载）"),
                item("vad", "语音分段 VAD（silero）", _vad_ok(), "已内置", "缺 VAD onnx", downloadable=False),
                item("fa", "时间轴对齐 FA（ForcedAligner .bin）", quick_check_aligner(model_dir),
                     "已下载", "未下载（约 939MB，启用对齐时下载）"),
            ],
        }

        # CRISPASR 路径：核心 exe + Whisper/Qwen 模型 + FA aligner gguf + 共享 diar
        #   核心的「就绪」条件含版本与加速版本 → 旧版或选了别的加速版本都算未就绪，
        #   会在加载时自动重新下载（消息里标明目前选的是哪一个 build）。
        variant = self._crisp_variant()
        variants_meta = crispasr_variants()
        vmeta = variants_meta.get(variant, variants_meta["vulkan"])
        core_ok = quick_check_crispasr(crispasr_dir, variant)
        whisper_size = s.get("whisper_size", "base")
        qwen_quant = s.get("crisp_qwen_quant", "q8")
        fa_quant = s.get("crisp_fa_quant", "q5")
        crisp = {
            "label": (f"CRISPASR（{vmeta.get('short') or vmeta['label']}："
                      f"OpenAI Whisper + Qwen）"),
            "backend": "crispasr",
            "items": [
                item("core", (f"CrispASR {_CRISPASR_VERSION} 核心"
                              f"（{vmeta.get('short') or vmeta['label']}）"),
                     core_ok, "已下载",
                     f"未下载或版本不符（约 {vmeta['size_mb']}MB，启用时下载）"),
                item("whisper", f"OpenAI Whisper 模型（{whisper_size}）",
                     quick_check_whisper_ggml(crispasr_dir, whisper_size),
                     "已下载", "未下载（启用时下载）"),
                item("qwen", f"Qwen3-ASR-1.7B 模型（{qwen_quant.upper()}）",
                     quick_check_qwen3_asr_gguf(model_dir, qwen_quant),
                     "已下载", "未下载（启用时下载）"),
                item("qwen_ja", f"Qwen3-ASR-1.7B 日语动漫模型（{qwen_quant.upper()}）",
                     quick_check_qwen3_asr_ja_gguf(model_dir, qwen_quant),
                     "已下载", "未下载（日语增强，启用时下载）"),
                item("fa", f"时间轴对齐 FA（aligner gguf {fa_quant.upper()}）",
                     quick_check_aligner_gguf(crispasr_dir, fa_quant),
                     "已下载", "未下载（约 643MB，启用时下载）"),
            ],
        }

        shared = [
            item("ffmpeg", "FFmpeg（视频抽音轨用）", bool(ff),
                 f"已检测：{ff}" if ff else "", "未下载（转录视频时自动下载解压）", downloadable=True),
            item("diar_shared", "说话者分离模型（OpenVINO/CRISPASR 共享）", diar_ok,
                 "已下载", "未下载（启用分离时自动下载）"),
        ]

        cores = [ov, crisp]

        # 红灯：缺且不可自动补（目前仅 VAD/ffmpeg 属此类）
        reds = sum(1 for c in cores for it in c["items"] if it["status"] == "red")
        yellows = sum(1 for c in cores for it in c["items"] if it["status"] == "yellow")
        return {
            "cores": cores, "shared": shared,
            "summary": {"red": reds, "yellow": yellows,
                        "ok": reds == 0},
            "activeBackend": getattr(self, "_active_backend", None),
        }

    # ── 设置（读写既有 settings.json）───────────────────────
    def get_settings(self) -> dict:
        s = {}
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            if Path(f).exists():
                s = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            s = {}
        # 简繁词汇转换已从设置页移除：引擎固定输出模型原文（简体）。
        return {
            "scale": int(s.get("ui_scale", 100)),
            "modelDir": s.get("model_dir", ""),
            "format": s.get("output_format", "srt"),
            "mirror": s.get("hf_mirror", ""),
            "ffmpeg": s.get("ffmpeg_path", ""),
            "theme": s.get("appearance", "light"),
            "uiLang": s.get("ui_lang", "简体中文"),
            "vad": float(s.get("vad_threshold", 0.5)),
            "chunkSecs": int(s.get("chunk_secs", 0) or 0),   # 0=自动（依模型上限）
            # 模型页模式：basic（用途导向，默认）／advanced（核心+模型全手动）
            "modelMode": (s.get("model_mode") if s.get("model_mode") in ("basic", "advanced")
                          else "basic"),
        }

    def set_settings(self, patch: dict) -> dict:
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        cur = {}
        try:
            if f.exists():
                cur = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        patch = patch or {}
        key_map = {"scale": "ui_scale", "format": "output_format",
                   "mirror": "hf_mirror", "ffmpeg": "ffmpeg_path", "theme": "appearance",
                   "uiLang": "ui_lang", "vad": "vad_threshold", "chunkSecs": "chunk_secs",
                   "modelMode": "model_mode"}
        for k, v in patch.items():
            if k in key_map:
                cur[key_map[k]] = v
        # 模型下载位置：非空 → 创建目录并记绝对路径；空 → 回退默认（exe 旁 ov_models/）。
        # 变更后需重启才对新位置生效（已加载的引擎仍指向旧目录）。
        self._model_dir_changed = False
        if "modelDir" in patch:
            raw = str(patch.get("modelDir") or "").strip().strip('"')
            if raw:
                try:
                    d = Path(raw).expanduser()
                    d.mkdir(parents=True, exist_ok=True)
                    new_dir = str(d)
                    self._model_dir_changed = (new_dir != self._model_dir().__str__())
                    cur["model_dir"] = new_dir
                except Exception:
                    pass          # 无效路径 → 不写入
            else:
                self._model_dir_changed = "model_dir" in cur
                cur.pop("model_dir", None)      # 空 → 回退默认位置
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        # ── 实时套用副作用（与 app.py 的 _on_*_change 对齐）──────────────
        if "vad" in patch:
            self._apply_vad(patch["vad"])      # _detect_speech_groups 读全域，立即生效
        if "mirror" in patch:
            self._apply_mirror(patch["mirror"])
        if "format" in patch:
            self._apply_output_format(patch["format"])
        if "theme" in patch and self._theme_cb:    # 通知窗口层同步标题栏深浅
            try:
                self._theme_cb(patch["theme"])
            except Exception:
                pass
        result = self.get_settings()
        if getattr(self, "_model_dir_changed", False):
            result["modelDirChanged"] = True
            result["message"] = "模型下载位置已更新。已下载的模型不会自动迁移；下次下载将存入新位置（重启后生效）。"
        return result

    def set_theme_callback(self, cb):
        """由 app_webview 注册：theme 变更时调用 cb(theme_str) 同步窗口标题栏深浅。"""
        self._theme_cb = cb

    def persisted_theme(self) -> str:
        """目前持久化的外观设置（light/dark/system），供窗口层决定初始标题栏。"""
        return self._settings_raw().get("appearance", "light") or "light"

    def _persist_setting(self, key: str, value):
        """写单个设置键到 settings.json（保留其他键）。"""
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        cur[key] = value
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _apply_mirror(self, base):
        try:
            import downloader as _dl
            _dl.set_mirror((base or "").strip())
        except Exception:
            pass

    def _apply_output_format(self, fmt):
        try:
            import subtitle_lines as _subs
            _subs.OUTPUT_FORMAT = "txt" if str(fmt).lower() == "txt" else "srt"
        except Exception:
            pass

    def _apply_vad(self, value):
        """把 VAD 阈值实时套到 app 模块全域 —— _detect_speech_groups 于调用时读取。"""
        try:
            core.VAD_THRESHOLD = float(value)
        except Exception:
            pass

    def _apply_runtime_prefs(self):
        """启动时把持久化偏好一次套用到各引擎模块全域。

        在 start_load() 之前调用 → 模块标志先就位，VAD/镜像/输出格式皆于首次转录前生效。
        """
        s = self._settings_raw()
        try:
            self._apply_vad(s.get("vad_threshold", 0.5))
        except Exception:
            pass
        self._apply_mirror(s.get("hf_mirror", ""))
        self._apply_output_format(s.get("output_format", "srt"))

    # ── 批次（已移除批次识别功能，保留界面兼容空实现）────────
    def get_batch(self) -> dict:
        return {"summary": {"done": 0, "total": 0}, "items": []}

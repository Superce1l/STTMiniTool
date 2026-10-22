"""
app.py — ASR 引擎层（OpenVINO CPU 核心）

本模块只保留引擎与共享常量，供 webview_backend / cli_mode 复用：
  • ASREngine（0.6B）／ASREngine1p7B（1.7B INT8 KV-Cache）—— OpenVINO 推理
  • Silero VAD 分段（_detect_speech_groups）与 VAD 模型查找
  • 路径 / 设置常量（BASE_DIR、SETTINGS_FILE、SRT_DIR、_DEFAULT_MODEL_DIR…）

旧 CustomTkinter 桌面界面已移除；桌面 / WebView 唯一入口是 app_webview.py。
"""
from __future__ import annotations

# ── UTF-8 模式：在所有其他 import 之前设置 ────────────────────────────
# 解决 Traditional Chinese Windows（cp950）上第三方包用系统默认编码
# 读取 UTF-8 文件时出现 "utf-8 codec can't decode byte 0xa6" 的问题。
# PYTHONUTF8=1 等效于 `python -X utf8`，让所有 open() 默认使用 UTF-8。
import os as _os, sys as _sys, io as _io
_os.environ.setdefault("PYTHONUTF8", "1")
# 同步修正 stdout/stderr（避免 print 中文在 cp950 console 出错）
for _stream_name in ("stdout", "stderr"):
    _s = getattr(_sys, _stream_name)
    if hasattr(_s, "buffer") and _s.encoding.lower() not in ("utf-8", "utf8"):
        setattr(_sys, _stream_name,
                _io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace"))
del _os, _sys, _io, _stream_name, _s

import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

# ── 诊断辅助（原属 chatllm_engine，供 ASREngine 的 VAD 诊断路径使用）────
from chatllm_engine import (
    format_vad_diag, detect_degenerate_asr,
)

# ── CrispASR 后端（可选，OpenAI Whisper / Qwen3 走 Vulkan）─────────────
try:
    from crisp_engine import CrispWhisperEngine
    _CRISPASR_AVAILABLE = True
except Exception:
    _CRISPASR_AVAILABLE = False
    CrispWhisperEngine  = None

# ── 字幕分行（全引擎共享，与 CrispASR/Whisper 统一）────────────────────
from subtitle_lines import MAX_CHARS, _ts_chatllm_to_subtitle_lines, write_transcript
import subtitle_lines as _subs
from engine_types import TranscribeResult

def _resolve_backend(core: str, model_label: str):
    """(核心, 模型标签) → (backend, 量化/尺寸)。三层选择的中央映射。

    返回：
      ("crispasr", "base"|"small"|"medium"|"large"|"turbo") — OpenAI Whisper
      ("openvino", "1.7B"|"0.6B")   — Qwen OpenVINO
    """
    if "Whisper" in core:
        label = model_label.lower()
        size = next((s for s in ("turbo", "large", "medium", "small", "base")
                     if s in label), "base")
        return "crispasr", size
    if "1.7B INT8" in model_label:
        return "openvino", "1.7B"
    return "openvino", "0.6B"


def _core_for_backend(backend: str) -> str:
    """backend → 核心标签（加载完成后同步 UI 用）。"""
    return "Whisper" if backend == "crispasr" else "Qwen"


def _ui_core_model(settings: dict):
    """settings → (核心标签, 模型标签)，供加载后同步 UI 下拉。"""
    backend = settings.get("backend", "openvino")
    if backend == "crispasr":
        size = settings.get("whisper_size", "base")
        label = {"base": "Whisper Base", "small": "Whisper Small",
                 "medium": "Whisper Medium", "large": "Whisper Large",
                 "turbo": "Whisper Large Turbo"}.get(size, "Whisper Base")
        return "Whisper", label
    sz = settings.get("cpu_model_size", "0.6B")
    return "Qwen", ("Qwen3-ASR-1.7B INT8" if "1.7B" in sz else "Qwen3-ASR-0.6B")

# ── 路径 ──────────────────────────────────────────────
# PyInstaller 冻结时，模型应放在 EXE 旁边（非 _internal/）
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
# PyInstaller onefile 解压的临时资源根（_MEIPASS）：--add-data 进来的文件在这里，
# 不在 EXE 旁。随包附带的小档（如 silero_vad_v4.onnx）要从这里找得到，否则冻结后
# 在 BASE_DIR（EXE 旁）找不到 → VAD 加载失败、模型永远停在「加载 VAD…」。
_MEIPASS_DIR = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None
_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"
SETTINGS_FILE      = BASE_DIR / "settings.json"
SRT_DIR            = BASE_DIR / "subtitles"
SRT_DIR.mkdir(exist_ok=True)

# ── 常数 ────────# 常数
SAMPLE_RATE   = 16000
VAD_CHUNK     = 512
VAD_CONTEXT_SAMPLES = 64   # silero-vad v5（master onnx）：每窗需附带上窗末 64 样本 context
# VAD 阈值不再开放设置：基准 0.50 起步，「说话轻/远场录音」未检出时自动
# 降阈值重试（见 _vad_retry_thresholds）。真静音/纯音乐（全段概率极低）不重试。
VAD_THRESHOLD = 0.5
_VAD_RETRY_STEPS     = (0.35, 0.22)   # 自动重试采用的更低阈值（由高到低）
_VAD_RETRY_MIN_PROB  = 0.18           # 全段最高人声概率低于此 → 判真静音，不重试
MAX_GROUP_SEC = 20
MIN_SUB_SEC   = 0.6
GAP_SEC       = 0.08

# VAD 批量推理（v4 路径、长音频）：silero 的 RNN 状态沿 batch 维并行推进，
# 把时间轴连续切 VAD_BATCH 段同时喂——段内逐窗状态与顺序推理一致，仅块边界
# 状态从零热身（onnxruntime 实测 60s 音频 0.37s→0.11s，0.5 阈值判定完全一致；
# 边界热身差由下游 PAD/MERGE 平滑吸收）。窗数低于 VAD_BATCH_MIN_WINDOWS
# （约 3.2s）不批量：短音频顺序推理本就极快，且行为与旧版逐窗完全一致。
VAD_BATCH = 8
VAD_BATCH_MIN_WINDOWS = 100

RT_SILENCE_CHUNKS    = 25   # ~0.8s 静音后触发转录
RT_MAX_BUFFER_CHUNKS = 600  # ~19s 上限强制转录

# ── ForcedAligner 相关常数 ─────────────────────────────────────────
GPU_MODEL_DIR      = BASE_DIR / "GPUModel"
ALIGNER_MODEL_NAME = "Qwen3-ForcedAligner-0.6B"

# 断句标点集合 _ZH_CLAUSE_END / _EN_SENT_END 已移至 subtitle_lines（共享）


# ══════════════════════════════════════════════════════
# 共享工具函数
# ══════════════════════════════════════════════════════

def _vad_is_v5(vad_sess) -> bool:
    """判断 VAD 会话是 v5 接口（state/stateN）还是 v4 接口（h/c）。

    上游仓库不随包附带 silero onnx（.gitignore 排除），用户环境里既可能是
    v4 老文件也可能是 v5 新文件 —— 两种输入名不同，按会话实际输入自动适配。
    """
    names = {i.name for i in vad_sess.get_inputs()}
    return "state" in names and "h" not in names


def _vad_retry_thresholds(stats: dict | None) -> tuple[float, ...]:
    """首跑未检出人声时，应依次重试的更低阈值；无可信人声迹象回空。

    只回**低于本段最高概率**的阈值——高于它的重试必然空手而归，纯属浪费。
    """
    max_p = float((stats or {}).get("max_prob", 0.0))
    if max_p < _VAD_RETRY_MIN_PROB:
        return ()
    return tuple(t for t in _VAD_RETRY_STEPS if t < max_p)


def _detect_speech_groups(audio: np.ndarray, vad_sess, max_group_sec: int = MAX_GROUP_SEC,
                          stats: dict | None = None,
                          threshold: float | None = None) -> list[tuple[float, float, np.ndarray]]:
    """Silero VAD 分段，返回 [(start_s, end_s, chunk), ...]

    可传入 stats dict，函数会填入 n_chunks / max_prob / mean_prob / threshold /
    n_segments，供上层在「未检测到人声」时产生明确诊断（见 format_vad_diag）。
    兼容 v4（输入 h/c/sr，状态 [2,1,64]）与 v5（输入 state/sr，状态 [2,N,128]）。
    长音频（v4）走批量推理提速（见 VAD_BATCH 注释）；其余逐窗顺序推理。
    threshold 省略时用基准 VAD_THRESHOLD（自动重试由 _vad_segment_auto 编排）。
    """
    thr = VAD_THRESHOLD if threshold is None else float(threshold)
    v5 = _vad_is_v5(vad_sess)
    n  = len(audio) // VAD_CHUNK
    if n == 0:
        return [(0.0, len(audio) / SAMPLE_RATE, audio)] if len(audio) else []
    h  = np.zeros((2, 1, 64), dtype=np.float32)
    c  = np.zeros((2, 1, 64), dtype=np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(VAD_CONTEXT_SAMPLES, dtype=np.float32)
    sr = SAMPLE_RATE          # v5 的 sr 需列表可迭代；v4 用 0-d int64 张量
    probs = np.empty(n, dtype=np.float32)

    if (not v5) and n >= VAD_BATCH_MIN_WINDOWS:
        # ── 批量路径（v4，长音频）：连续分块并行 ─────────────────────
        nb = min(VAD_BATCH, n)
        per = -(-n // nb)                       # ceil：各块等长，尾块可短
        ranges = [(j * per, min(n, (j + 1) * per)) for j in range(nb)]
        ranges = [(lo, hi) for lo, hi in ranges if hi > lo]
        nb = len(ranges)
        bh = np.zeros((2, nb, 64), dtype=np.float32)
        bc = np.zeros((2, nb, 64), dtype=np.float32)
        for i in range(per):
            act = [j for j in range(nb) if ranges[j][0] + i < ranges[j][1]]
            if not act:
                break
            idx = np.array([ranges[j][0] + i for j in act])
            win = np.stack([audio[g * VAD_CHUNK:(g + 1) * VAD_CHUNK]
                            for g in idx]).astype(np.float32)
            out, bhn, bcn = vad_sess.run(
                None, {"input": win, "h": bh[:, act], "c": bc[:, act],
                       "sr": np.array(sr, dtype=np.int64)})
            bh[:, act] = bhn
            bc[:, act] = bcn
            probs[idx] = out[:, 0]
    else:
        # ── 顺序路径（v5 一律走这里；v4 短音频同旧版逐窗行为）─────────
        for i in range(n):
            chunk = audio[i*VAD_CHUNK:(i+1)*VAD_CHUNK].astype(np.float32)[np.newaxis, :]
            if v5:
                # silero-vad master（v5）是「外部 context」版：每窗输入 =
                # [上一窗末 64 样本] + [512 新样本]，共 576 样本。
                x = np.concatenate([context, chunk[0]])[np.newaxis, :]
                out, state = vad_sess.run(
                    None, {"input": x, "state": state, "sr": [sr]})
                context = chunk[0][-VAD_CONTEXT_SAMPLES:]
            else:
                out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c,
                                                "sr": np.array(sr, dtype=np.int64)})
            probs[i] = float(out[0, 0])
    probs = probs.tolist()
    if stats is not None:
        stats["n_chunks"]  = len(probs)
        stats["max_prob"]  = max(probs) if probs else 0.0
        stats["mean_prob"] = (sum(probs) / len(probs)) if probs else 0.0
        stats["threshold"] = thr
    if not probs:
        return [(0.0, len(audio) / SAMPLE_RATE, audio)]

    MIN_CH = 16; PAD = 5; MERGE = 16
    raw: list[tuple[int, int]] = []
    in_sp = False; s0 = 0
    for i, p in enumerate(probs):
        if p >= thr and not in_sp:
            s0 = i; in_sp = True
        elif p < thr and in_sp:
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
        # 取完整语音段并多含一点尾巴；mel 会在 processor 补零到固定长度，整秒对齐
        # 非必要。旧版 floor 会丢尾段近 1 秒；句尾轻声字也常被 Silero 的 ge 排除。
        ch = audio[gs: min(len(audio), ge + _tail)].astype(np.float32)
        if len(ch) < SAMPLE_RATE // 2:      # 最小 0.5 秒
            continue
        result.append((gs / SAMPLE_RATE, ge / SAMPLE_RATE, ch))
    return result


def _split_to_lines(text: str) -> list[str]:
    """以标点符号切分短句，移除标点，每句独立成行。

    断句规则（英文/中文统一）：
    1. 所有标点（,.!?;: 及中文，。？！；：…—）→ 立即切行，标点不输出
    2. 英文整字为最小单位，词前补空格（词界）
    3. MAX_CHARS 保护：超限才强制换行
    """
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = text.replace("</asr_text>", "").strip()
    if not text:
        return []

    # 中文、英文标点统一触发切行（含英文逗号）
    PUNCT = frozenset('，。？！；：…—、.,!?;:')
    lines: list[str] = []
    buf   = ""

    i = 0
    while i < len(text):
        ch = text[i]

        # ── 标点符号：切行，标点不加入输出（隐藏）────────────────────
        if ch in PUNCT:
            if buf.strip():
                lines.append(buf.strip())
            buf = ""
            i += 1
            continue

        # ── 英文单字：整字收集，词前补空格（词界）────────────────────
        if ch.isalpha() and ord(ch) < 128:
            j = i
            while j < len(text) and text[j].isalpha() and ord(text[j]) < 128:
                j += 1
            word = text[i:j]
            prefix = " " if buf and not buf.endswith(" ") else ""
            if len(buf) + len(prefix) + len(word) > MAX_CHARS and buf.strip():
                lines.append(buf.strip())
                buf = word
            else:
                buf += prefix + word
            i = j
            continue

        # ── 空格：保留分词间距 ────────────────────────────────────────
        if ch == " ":
            if buf and not buf.endswith(" "):
                buf += " "
            i += 1
            if len(buf.rstrip()) >= MAX_CHARS:
                lines.append(buf.strip())
                buf = ""
            continue

        # ── 中文/日文/数字等：逐字累积 ────────────────────────────────
        buf += ch
        i += 1
        if len(buf) >= MAX_CHARS:
            lines.append(buf.strip())
            buf = ""

    if buf.strip():
        lines.append(buf.strip())
    return [l for l in lines if l.strip()]



# _srt_ts 已移至 subtitle_lines（共享）


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


def _find_vad_model(model_dir: Path | None = None) -> Path | None:
    """查找 Silero VAD ONNX：指定 model_dir → ov_models/ → GPUModel/ → _MEIPASS（打包）。

    冻结后随包附带的 silero_vad_v4.onnx 位于 _MEIPASS/ov_models（--add-data），
    故必须把它纳入搜索，否则在 EXE 旁的 ov_models/ 找不到 → VAD 加载卡死。
    """
    candidates: list[Path] = []
    if model_dir is not None:
        candidates += [Path(model_dir) / "silero_vad_v4.onnx",
                       Path(model_dir) / "silero_vad.onnx"]
    candidates += [
        _DEFAULT_MODEL_DIR / "silero_vad_v4.onnx",
        GPU_MODEL_DIR / "silero_vad_v4.onnx",
        _DEFAULT_MODEL_DIR / "silero_vad.onnx",
        GPU_MODEL_DIR / "silero_vad.onnx",
    ]
    if _MEIPASS_DIR is not None:
        candidates += [_MEIPASS_DIR / "ov_models" / "silero_vad_v4.onnx",
                       _MEIPASS_DIR / "silero_vad_v4.onnx"]
    for p in candidates:
        if p.exists():
            return p
    return None


# _merge_orphan_lines / _ts_to_subtitle_lines / _ts_chatllm_to_subtitle_lines
# 已移至 subtitle_lines（共享，全引擎统一）；旧版实现随重构移除。


# ══════════════════════════════════════════════════════
# ASR 引擎
# ══════════════════════════════════════════════════════

class ASREngine:
    """封装所有模型。transcribe() 加互斥锁，多线程安全。"""

    max_chunk_secs: int = 30   # 每段最长音频（秒），子类别可覆盖
    _OV_SUBDIR: str = "qwen3_asr_int8"   # model_dir 下的 OV 模型子目录，子类别可覆盖
    # 时间轴对齐用 chatllm 原生 FA（无需 torch）；OpenVINO 卡拉OK逐字模式依赖。
    # 同一文件名与下载流程，让「时间轴对齐」UI 在 CPU/GPU 后端行为一致。
    FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

    def __init__(self):
        self.ready       = False
        self._lock       = threading.Lock()
        # 取消事件：webview_backend.cancel() 置位 → process_file 在段间提前
        # 返回（与 CrispWhisperEngine.cancel_event 同一契约，调用端统一处理）。
        self.cancel_event = threading.Event()
        self.vad_sess    = None
        self.audio_enc   = None
        self.embedder    = None
        self.dec_req     = None
        self.processor   = None   # LightProcessor（不含 torch）
        self.pad_id      = None
        self.diar_engine = None   # DiarizationEngine（可选）
        self.aligner     = None   # 兼容标志（chatllm FA 就绪时为 True）
        self.use_aligner = False  # 是否启用时间轴对齐
        self._fa         = None   # ChatLLMAligner（chatllm 原生 FA）
        self._fa_bin     = None   # FA .bin 路径（就绪时非 None）
        self._model_dir  = None   # 模型文件夹（FA .bin 与下载位置）

    def load(self, device: str = "CPU", model_dir: Path = None, cb=None, cpu_threads: int = 0):
        """从背景线程调用。cb(msg) 用于更新 UI 状态。
        cpu_threads: 0=OpenVINO 自动，>0=指定逻辑核心数（LATENCY hint）
        """
        import onnxruntime as ort
        import openvino as ov
        from processor_numpy import LightProcessor

        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        self._model_dir = model_dir   # FA .bin 与按需下载位置
        ov_dir   = model_dir / self._OV_SUBDIR   # 0.6B / 1.7B 由类属性决定

        # ── CPU 线程设置 ─────────────────────────────────────────────
        # LATENCY hint：单一请求最低延迟（不同于 THROUGHPUT 批处理模式）
        # ENABLE_HYPER_THREADING YES：确保 P-core HT 与 E-core 均被使用
        cpu_cfg: dict = {}
        if device == "CPU":
            cpu_cfg["PERFORMANCE_HINT"] = "LATENCY"
            cpu_cfg["ENABLE_HYPER_THREADING"] = "YES"
            if cpu_threads > 0:
                cpu_cfg["INFERENCE_NUM_THREADS"] = str(cpu_threads)
        # VAD 路径：含 _MEIPASS（打包）等多处搜索，冻结后才找得到随包附带的 onnx。
        vad_path = _find_vad_model(model_dir)

        def _s(msg):
            if cb: cb(msg)

        if vad_path is None:
            raise FileNotFoundError(
                "找不到 silero_vad_v4.onnx（VAD 模型）。请确认已随安装包附带或放入 ov_models/。")
        _s("加载 VAD 模型…")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

        _s("加载说话者分离模型…")
        try:
            from diarize import DiarizationEngine
            diar_dir = model_dir / "diarization"
            eng = DiarizationEngine(diar_dir)
            self.diar_engine = eng if eng.ready else None
        except Exception:
            self.diar_engine = None

        _s(f"编译 ASR 模型（{device}）…")
        core = ov.Core()
        self.audio_enc = core.compile_model(str(ov_dir / "audio_encoder_model.xml"),      device, cpu_cfg)
        self.embedder  = core.compile_model(str(ov_dir / "thinker_embeddings_model.xml"), device, cpu_cfg)
        self._compile_decoder(core, ov_dir, device, cpu_cfg)   # 0.6B stateful / 1.7B KV-cache

        _s("加载 Processor（纯 numpy）…")
        self.processor = LightProcessor(ov_dir)
        self.pad_id    = self.processor.pad_id

        # ── ForcedAligner（chatllm main.exe 子程序，CPU -ngl 0，无需 torch）──
        #    缺 .bin 时静默退回比例估算；UI 会引导使用者按需下载。
        self._load_aligner(cb=_s)

        # 抑制 "Setting pad_token_id to eos_token_id" 重复警告
        try:
            import transformers.utils.logging as _tf_logging
            import logging as _logging
            _tf_logging.get_logger("transformers.generation.utils").setLevel(_logging.ERROR)
        except Exception:
            pass

        self.ready     = True
        aligner_info = "  + ForcedAligner" if self.use_aligner else ""
        _s(f"编译完成（{device}{aligner_info}）")

    # ── 时间轴对齐（chatllm 原生 FA，CPU -ngl 0）──────────────────────────
    def _load_aligner(self, cb=None):
        """检测 chatllm 版 ForcedAligner .bin（与 ASR .bin 同层 model_dir）。

        OpenVINO 后端沿用 chatllm main.exe 子程序做字级对齐（CPU -ngl 0），
        完全不需 PyTorch。缺 .bin 或验证失败时静默退回比例估算，UI 会引导
        使用者按需下载。
        """
        from fa_aligner import ChatLLMAligner
        self.aligner     = None
        self.use_aligner = False
        self._fa_bin     = None
        # FA 一律以 CPU 子程序执行（n_gpu_layers=0），与 OpenVINO 设备选择无关，
        # 避免 Vulkan/Intel 设备 id 不一致；CPU 对齐已足够快。
        self._fa = ChatLLMAligner(fa_dir=self._model_dir, n_gpu_layers=0)
        if self._fa.load(cb=cb):
            self._fa_bin     = self._fa._fa_bin
            self.aligner     = True   # 兼容标志（沿用 use_aligner 判断流程）
            self.use_aligner = True

    def _align_chunk(
        self,
        wav_path: str,
        ref_text: str,
        language: str = "Chinese",
    ) -> list[tuple[str, float, float]]:
        """委派 chatllm FA 对齐「参考文字 + 音频」，返回字级时间轴。"""
        if self._fa is None or not self._fa_bin:
            return []
        return self._fa.align_chunk(wav_path, ref_text, language)

    # ── Decoder 挂勾（0.6B stateful；ASREngine1p7B 覆盖为 KV-cache）───────
    #    2026-08 修正：旧版 ASREngine1p7B 的 load/推理覆盖在 ce1f8b0 重构时被
    #    误删，导致「1.7B INT8」实际上加载 0.6B 目录（只装 1.7B 的机器直接
    #    报 Could not open qwen3_asr_int8/...xml）。改以挂勾点分离差异。

    def _compile_decoder(self, core, ov_dir: Path, device: str, cpu_cfg: dict):
        """编译 decoder。0.6B：单一 stateful decoder_model.xml。"""
        dec_comp     = core.compile_model(str(ov_dir / "decoder_model.xml"), device, cpu_cfg)
        self.dec_req = dec_comp.create_infer_request()

    def _encode_audio(self, mel) -> np.ndarray:
        """mel → audio embeds。1.7B 导出的输入不含 batch 维度，由子类覆盖。"""
        return list(self.audio_enc({"mel": mel}).values())[0]

    def _generate_tokens(self, combined: np.ndarray, max_tokens: int) -> list[int]:
        """自回归贪婪解码（0.6B stateful decoder）。返回生成的 token id 列表。

        transcribe() 与 process_file() 共享此循环（原本两处内联重复）。
        """
        L   = combined.shape[1]
        pos = np.arange(L, dtype=np.int64)[np.newaxis, :]
        self.dec_req.reset_state()
        out    = self.dec_req.infer({0: combined, "position_ids": pos})
        logits = list(out.values())[0]
        eos = self.processor.eos_id
        eot = self.processor.eot_id
        gen: list[int] = []
        nxt = int(np.argmax(logits[0, -1, :])); cur = L
        while nxt not in (eos, eot) and len(gen) < max_tokens:
            gen.append(nxt)
            emb = list(self.embedder(
                {"input_ids": np.array([[nxt]], dtype=np.int64)}
            ).values())[0]
            out    = self.dec_req.infer(
                {0: emb, "position_ids": np.array([[cur]], dtype=np.int64)}
            )
            logits = list(out.values())[0]
            nxt = int(np.argmax(logits[0, -1, :])); cur += 1
        return gen

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """将 16kHz float32 音频转录为繁体中文。
        language : 强制语言（如 "Chinese"），None 表示自动检测
        context  : 识别提示（歌词/关键字），放入 system message
        """
        with self._lock:
            # ── 前处理（纯 numpy，不需 torch）────────────────────────
            mel, ids = self.processor.prepare(audio, language=language, context=context)

            # ── 音频编码 + 文字 Embedding ────────────────────────────
            ae = self._encode_audio(mel)
            te = list(self.embedder({"input_ids": ids}).values())[0]

            # ── 音频特征填入音频 pad 位置 ─────────────────────────────
            combined = te.copy()
            mask = ids[0] == self.pad_id
            np_ = int(mask.sum()); na = ae.shape[1]
            if np_ != na:
                mn = min(np_, na)
                combined[0, np.where(mask)[0][:mn]] = ae[0, :mn]
            else:
                combined[0, mask] = ae[0]

            # ── Decoder 自回归生成（0.6B stateful / 1.7B KV-cache 挂勾）──
            gen = self._generate_tokens(combined, max_tokens)

            # ── 解码（纯 Python BPE decode）──────────────────────────
            raw = self.processor.decode(gen)
            if "<asr_text>" in raw:
                raw = raw.split("<asr_text>", 1)[1]
            text = raw.strip()
            return text

    @staticmethod
    def _quiet_split_pos(chunk: np.ndarray, target: int, lo: int, hi: int) -> int:
        """在 [lo, hi]（样本索引）内找短窗 RMS 最低的位置作为切点。

        人声在停顿处能量显著走低，以此把切分点吸附到词句间隙而非字词中间；
        范围无效时回退 target。
        """
        win, hop = 400, 200   # 25ms 窗 / 12.5ms 步进 @16kHz
        lo = max(lo, win)
        hi = min(hi, len(chunk))
        if lo >= hi:
            return target
        best, best_e = target, None
        for p in range(lo, hi + 1, hop):
            seg = chunk[p - win:p]
            e = float(np.mean(seg * seg))
            if best_e is None or e < best_e:
                best, best_e = p, e
        return best

    def _enforce_chunk_limit(
        self,
        groups: list[tuple[float, float, np.ndarray, "str | None"]],
    ) -> list[tuple[float, float, np.ndarray, "str | None"]]:
        """将超过 max_chunk_secs 的音频段落切分为多个子片段。

        不论是说话者分离路径或 VAD 单段路径，都可能产生比模型
        输入长度（max_chunk_secs）更长的 chunk。若不切分，
        _extract_mel() 会静默截断尾段，造成掉字。

        切分策略：n 片（n = ceil(L/max)），每片长度硬性约束在
        [1 秒, max_samples]；实际切点在「可行窗」内吸附到能量最低的停顿处。
        可行窗同时受三重约束（左 ≥ max(prev+1s, L-(n-k)*max) 保证剩余可切、
        右 ≤ min(prev+max, L-(n-k)*1s) 保证本片不超限），数学上可证非空
        —— 停顿吸附的漂移不会把任何一片挤出上限（旧实现按固定步进硬切，
        既会劈开字词又可能因末段无约束产生超长片被 mel 静默截断）。
        """
        max_samples = self.max_chunk_secs * SAMPLE_RATE
        search = int(1.5 * SAMPLE_RATE)      # 切点距理想位置的偏好半径
        result = []
        for t0, t1, chunk, spk in groups:
            L = len(chunk)
            if L <= max_samples:
                result.append((t0, t1, chunk, spk))
                continue
            n = -(-L // max_samples)                 # ceil(L / max_samples)
            ideal = L / n
            bounds = [0]
            prev = 0
            for k in range(1, n):
                # 可行窗（非空可证）：保证本片 ≤ max 且剩余 (n-k) 片仍各 ≥ 1s
                lo = max(prev + SAMPLE_RATE, L - (n - k) * max_samples)
                hi = min(prev + max_samples, L - (n - k) * SAMPLE_RATE)
                # 偏好窗：理想切点 ±1.5s；与可行窗无交集时退到可行窗全域
                target = int(round(ideal * k))
                plo, phi = target - search, target + search
                if max(lo, plo) <= min(hi, phi):
                    lo, hi = max(lo, plo), min(hi, phi)
                cut = self._quiet_split_pos(chunk, target, lo, hi)
                cut = max(lo, min(cut, hi))
                bounds.append(cut)
                prev = cut
            bounds.append(L)
            for a, b in zip(bounds, bounds[1:]):
                piece = chunk[a:b]
                if len(piece) < SAMPLE_RATE:      # 防御：异常短残余不产出
                    continue
                piece_t0 = t0 + a / SAMPLE_RATE
                result.append((piece_t0, min(t1, piece_t0 + len(piece) / SAMPLE_RATE),
                               piece, spk))
        return result

    def _vad_segment_auto(self, audio: np.ndarray, progress_cb=None):
        """VAD 分段 + 灵敏度自适应（取代旧的「VAD 灵敏度」手动设置）。

        基准阈值（0.50）未检出人声、但全段最高人声概率达到可信下限时，自动
        逐级降低阈值重试——覆盖「说话轻／远场录音被门槛挡掉」这一用户手动
        调低设置的主因；真静音／纯音乐（概率极低）不浪费重试，直接给诊断。
        返回 (groups, diag)：成功时 diag=None；失败时 groups=None、diag 为
        含自动重试说明的明确原因。
        """
        stats: dict = {}
        groups = _detect_speech_groups(audio, self.vad_sess, self.max_chunk_secs,
                                       stats=stats)
        if groups:
            return groups, None
        tried = False
        for thr in _vad_retry_thresholds(stats):
            tried = True
            if progress_cb:
                progress_cb(0, 1, f"未检测到人声，自动降低灵敏度重试（门槛 {thr:.2f}）…")
            retry_stats: dict = {}
            groups = _detect_speech_groups(audio, self.vad_sess, self.max_chunk_secs,
                                           stats=retry_stats, threshold=thr)
            if groups:
                return groups, None
            stats = retry_stats
        diag = format_vad_diag(stats)
        if tried:
            diag += "（已自动尝试降低灵敏度，仍无有效语音段落。）"
        return None, diag

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language: str | None = None,
        context: str | None = None,
        diarize: bool = False,
        n_speakers: int | None = None,
        original_path: Path | None = None,
        out_format: str | None = None,
    ) -> Path | None:
        """音频 → 字幕档，返回 TranscribeResult（output=字幕路径，None=未产出）。
        language   : 强制语言（如 "Chinese"），None 表示自动检测
        context    : 识别提示（歌词/关键字），放入 system message
        diarize    : True 时用说话者分离取代 VAD，SRT 加说话者前缀
        n_speakers : 指定说话者人数（None=自动检测）
        out_format : "srt" | "txt"；None 采全域设置（端点固定传 "srt"）
        """
        from audio_io import load_audio_16k_mono
        audio, _ = load_audio_16k_mono(audio_path, SAMPLE_RATE)
        deg_count = 0                # 本次退化（异常）输出段数
        deg_reason = None

        # ── 分段策略：说话者分离 vs 传统 VAD ─────────────────────────
        # groups_spk: [(g0_sec, g1_sec, audio_chunk, speaker_label | None), ...]
        use_diar = diarize and self.diar_engine is not None and self.diar_engine.ready
        if use_diar:
            diar_segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
            if not diar_segs:
                return TranscribeResult()
            groups_spk = [
                (t0, t1,
                 audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)],
                 spk)
                for t0, t1, spk in diar_segs
            ]
        else:
            if self.vad_sess is None:
                diag = "⚠ VAD 模型（silero_vad）未加载，无法分段"
                if progress_cb:
                    progress_cb(0, 1, diag)
                return TranscribeResult(diag=diag)
            vad_groups, diag = self._vad_segment_auto(audio, progress_cb)
            if not vad_groups:
                if progress_cb:
                    progress_cb(0, 1, diag)
                return TranscribeResult(diag=diag)
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        # 强制切分超过 max_chunk_secs 的片段（两条路径都需要）
        groups_spk = self._enforce_chunk_limit(groups_spk)

        # ── ASR 逐段转录 ─────────────────────────────────────────────
        all_subs: list[tuple[float, float, str, str | None]] = []
        all_rich: list[dict] = []    # 与 all_subs 对应的字级结构（卡拉OK用）
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            # 取消检查：段间立即返回（长音频不必等整段跑完）。
            # crispasr 引擎的同名事件由子进程终止实现，语义一致。
            if self.cancel_event.is_set():
                return TranscribeResult(cancelled=True,
                                        segments_rich=all_rich or None,
                                        degenerate=deg_count)
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total,
                            f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")

            # ── ASR 转录（取简体原始输出，对齐后再繁化）─────────────────
            max_tok = 400 if language == "Japanese" else 300
            with self._lock:
                mel, ids = self.processor.prepare(
                    chunk, language=language, context=context)
                ae = self._encode_audio(mel)
                te = list(self.embedder({"input_ids": ids}).values())[0]
                combined = te.copy()
                mask = ids[0] == self.pad_id
                np_ = int(mask.sum()); na = ae.shape[1]
                if np_ != na:
                    mn = min(np_, na)
                    combined[0, np.where(mask)[0][:mn]] = ae[0, :mn]
                else:
                    combined[0, mask] = ae[0]
                # 生成（0.6B stateful / 1.7B KV-cache 挂勾）
                gen = self._generate_tokens(combined, max_tok)
                raw_decoded = self.processor.decode(gen)
                if "<asr_text>" in raw_decoded:
                    raw_decoded = raw_decoded.split("<asr_text>", 1)[1]
                raw_text = raw_decoded.strip()

            if not raw_text:
                continue

            # 退化输出（复诵提示范本 / 高度重复）→ 不写入垃圾字幕，跳过并记录
            deg = detect_degenerate_asr(raw_text)
            if deg:
                deg_count += 1
                deg_reason = deg
                if progress_cb:
                    progress_cb(i, total, f"[{i+1}/{total}] 略过异常输出：{deg}")
                continue

            # ── ForcedAligner 精确时间轴对齐（chatllm .bin，CPU -ngl 0）──
            #    把 chunk 写成临时 wav，交给 chatllm main.exe 做字级对齐，
            #    再用 _ts_chatllm_to_subtitle_lines 依字级时间轴＋标点切行。
            aligned = False
            if self.use_aligner and self._fa_bin is not None:
                import tempfile as _tempfile
                import soundfile as _sf
                with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
                    _tmp_wav = _tf.name
                try:
                    _sf.write(_tmp_wav, chunk, SAMPLE_RATE, subtype="PCM_16")
                    align_lang = language if (language and language != "自动检测") else "Chinese"
                    ts_items = self._align_chunk(_tmp_wav, raw_text, align_lang)
                    if ts_items:
                        subs = _ts_chatllm_to_subtitle_lines(
                            ts_items, raw_text, g0, spk, with_words=True,
                        )
                        if subs:
                            for (s, e, t, sp, words) in subs:
                                all_subs.append((s, e, t, sp))
                                all_rich.append({"start": s, "end": e, "text": t,
                                                 "speaker": sp, "words": words})
                            aligned = True
                except Exception:
                    aligned = False  # 静默 fallback 到比例估算
                finally:
                    try:
                        os.remove(_tmp_wav)
                    except OSError:
                        pass

            if not aligned:
                # ── 比例估算 Fallback ──────────────────────────────────────
                text = raw_text
                lines = _split_to_lines(text)
                for s, e, line in _assign_ts(lines, g0, g1):
                    all_subs.append((s, e, line, spk))
                    all_rich.append({"start": s, "end": e, "text": line,
                                     "speaker": spk, "words": []})   # 无对齐 → 前端内插

        if not all_subs:
            # 全部段落都退化（垃圾输出）→ 给明确原因，而非「未检测到人声」
            diag = None
            if deg_count > 0:
                diag = (
                    f"⚠ 模型输出异常（{deg_reason or '推理退化'}），"
                    f"共 {deg_count} 段被略过，未产生有效字幕。"
                    f"此核心／设备可能不兼容，建议改用 CPU(OpenVINO) 核心。"
                )
            return TranscribeResult(diag=diag, degenerate=deg_count)

        if progress_cb:
            progress_cb(total, total, "写入字幕…")

        # 以原始文件的目录与文件名输出（视频抽音轨时 audio_path 是临时路径）
        # 共享写出层：依全域设置（或 out_format 覆盖）产出 .srt 或 .txt。
        ref = original_path if original_path is not None else audio_path
        out = write_transcript(ref, all_subs, out_format)
        return TranscribeResult(output=out, segments_rich=all_rich,
                                degenerate=deg_count)


# ══════════════════════════════════════════════════════
# ASR 引擎 — 1.7B INT8 KV-cache 版本
# ══════════════════════════════════════════════════════

class ASREngine1p7B(ASREngine):
    """
    Qwen3-ASR-1.7B OpenVINO KV-cache 引擎（INT8 版本）。

    模型目录：ov_models/qwen3_asr_1p7b_kv_int8/
      audio_encoder_model.xml       — mel(128,1000)  → audio_embeds(1,130,2048)
      thinker_embeddings_model.xml  — input_ids      → token_embeds
      decoder_prefill_kv_model.xml  — prefill pass   → logit + past_keys + past_vals
      decoder_kv_model.xml          — decode step    → logit + new_keys  + new_vals
    """

    _OV_SUBDIR     = "qwen3_asr_1p7b_kv_int8"
    max_chunk_secs = 10   # audio_encoder 导出固定 T=1000（10s）

    def __init__(self):
        super().__init__()
        self.pf_model = None   # compiled prefill model
        self.dc_model = None   # compiled decode-step model

    # ── 挂勾覆盖（load/transcribe/process_file 全部沿用基底类）────────────
    #    2026-08 修正：KV-cache 推理实现曾于 ce1f8b0 重构时被误删，导致本类
    #    退化成空壳、继承 0.6B 的 load 而读错目录。自 2343122 版本复原并改写
    #    为挂勾覆盖，让 1.7B 同步取得 cpu_threads/FA 对齐等后续新功能。

    def _compile_decoder(self, core, ov_dir: Path, device: str, cpu_cfg: dict):
        """1.7B：prefill + decode-step 两个 KV-cache 模型，不用 stateful decoder。"""
        self.pf_model = core.compile_model(
            str(ov_dir / "decoder_prefill_kv_model.xml"), device, cpu_cfg)
        self.dc_model = core.compile_model(
            str(ov_dir / "decoder_kv_model.xml"),         device, cpu_cfg)
        self.dec_req  = None

    def _encode_audio(self, mel) -> np.ndarray:
        """1.7B audio_encoder 导出的输入不含 batch 维度：mel (128, 1000)。"""
        return list(self.audio_enc({"mel": mel[0]}).values())[0]

    def _generate_tokens(self, combined: np.ndarray, max_tokens: int) -> list[int]:
        """KV-cache 贪婪解码：O(L²) prefill 一次 + O(n) 逐 token decode。"""
        seq_len = combined.shape[1]
        pos_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        pf_out  = self.pf_model({"input_embeds": combined, "position_ids": pos_ids})
        pf_vals = list(pf_out.values())
        logits  = pf_vals[0]   # (1, 1, vocab)
        past_k  = pf_vals[1]   # (28, 1, 8, L, 128)
        past_v  = pf_vals[2]

        eos = self.processor.eos_id
        eot = self.processor.eot_id
        nxt = int(np.argmax(logits[0, -1, :]))
        if nxt in (eos, eot):
            return []

        gen = [nxt]
        cur = seq_len
        for _ in range(max_tokens - 1):
            new_emb = list(self.embedder(
                {"input_ids": np.array([[nxt]], dtype=np.int64)}
            ).values())[0]
            dc_out = self.dc_model({
                "new_embed":   new_emb,
                "new_pos":     np.array([[cur]], dtype=np.int64),
                "past_keys":   past_k,
                "past_values": past_v,
            })
            dc_vals = list(dc_out.values())
            logits  = dc_vals[0]
            past_k  = dc_vals[1]
            past_v  = dc_vals[2]
            nxt = int(np.argmax(logits[0, -1, :]))
            if nxt in (eos, eot):
                break
            gen.append(nxt)
            cur += 1
        return gen




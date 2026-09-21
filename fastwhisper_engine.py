"""fastwhisper_engine.py — Faster-Whisper-XXL (CTranslate2) 推理后端

定位：第四个推理引擎（ASREngine[OpenVINO] / ChatLLMASREngine[chatllm] /
CrispWhisperEngine[crispasr] 之后），承接「Whisper（Faster-Whisper-XXL）」
模型选项。Purfview 的 standalone 打包：自带全套依赖与 ffmpeg，CTranslate2
推理 + Silero VAD，OpenAI Whisper 全系模型（base→large-v3）开箱即用，
CPU / CUDA 皆可（--device auto 自动探测）。

设计：与 crisp_engine 同一契约——调用外部 exe → 收 SRT → 共享分行器
（_ts_chatllm_to_subtitle_lines）→ 说话者分离挂接。区别：
  • 模型不是本地 ggml 文件而是「尺寸名」（base/.../large-v3），引擎自己
    按 --model_dir 找/下载（HuggingFace Systran fared-Whisper 系）。
  • 进度经 --print_progress 的「NN%」行解析（与 crispasr -pp 同法）。
  • 字级时间轴：--word_timestamps true 产 word SRT，喂分行器保卡拉OK。

对外界面（webview_backend._load_fastwhisper）：
    eng = FastWhisperEngine()
    eng.load(engine_dir=..., model_size=..., cb=set_status)
    srt = eng.process_file(audio_path, progress_cb=..., language=..., ...)
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# 与 Qwen/Whisper 路径统一的字幕分行（全引擎共享）
from subtitle_lines import _ts_chatllm_to_subtitle_lines

SAMPLE_RATE = 16000     # 说话者分离解码用（load_audio_16k_mono），保持引擎契约

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = None
if sys.platform == "win32":
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUP_INFO.wShowWindow = 0  # SW_HIDE

_PROC_SHUTDOWN_GRACE_SECS = 30

# 上游「语言名 → ISO 码」由 exe 自行接受全名/码，这里只需把 UI 的
# 语言值（Chinese/English/...）直接透传即可，无需映射表。
# （本引擎不做 OpenCC 转换，无日语特判；与 crisp_engine 的差异见 process_file。）


class FastWhisperEngine:
    """Faster-Whisper-XXL 引擎封装。契约与 CrispWhisperEngine 对齐。"""

    def __init__(self):
        self.ready = False
        self._exe: Path | None = None
        self._engine_dir: Path | None = None
        self._model_size = "small"
        self._device = "auto"
        self._compute_type = "auto"
        self._cb = None
        self.diar_engine = None          # 与其他引擎同契约（由上层挂 diar）
        self.use_aligner = False         # XXL 自带 --word_timestamps，无需外挂 FA
        self._last_segments_rich = None
        import threading
        self._lock = threading.Lock()

    # ── 加载 ────────────────────────────────────────────────
    def load(self, engine_dir: Path, model_size: str = "small",
             cb=None, device: str = "auto"):
        """确认引擎 exe 存在并预检（模型文件由 exe 按需取/下载）。

        engine_dir：含 faster-whisper-xxl.exe 的目录（_xxl_data 同层）。
        model_size：base/small/medium/large-v2/large-v3/large-v3-turbo 等
                    （传给 --model；XXL 支持全部 OpenAI 尺寸与 distil 系）。
        device：auto/cpu/cuda。XXL 自检测；CUDA 无驱动时 exe 自回退 CPU。
        """
        self._cb = cb
        engine_dir = Path(engine_dir)
        exe = engine_dir / "faster-whisper-xxl.exe"
        if not exe.is_file():
            found = list(engine_dir.glob("**/faster-whisper-xxl.exe"))
            if not found:
                raise RuntimeError(f"找不到 faster-whisper-xxl.exe：{engine_dir}")
            exe = found[0]
        self._exe = exe
        self._engine_dir = exe.parent
        self._model_size = model_size or "small"
        self._device = device or "auto"
        self.ready = True
        if cb:
            cb(f"Faster-Whisper-XXL 就绪（模型 {self._model_size}）")

    # ── 命令行 ──────────────────────────────────────────────
    def _build_cmd(self, audio_path: Path, out_dir: Path,
                   language: str | None, hint: str | None) -> list[str]:
        """组合 faster-whisper-xxl.exe 参数。

        关键选择（与 crispasr 路径的取舍对齐）：
          --print_progress   解析「NN%」行驱动进度条（同 crispasr -pp）
          --word_timestamps  字级时间轴 → 共享分行器（卡拉OK/精准断句）
          --vad_filter       Silero VAD 断段，长音频免自带切片
          --beep_off --hallucinations_list_off  静默/防幻听提示音
        initial_prompt：转录提示（歌词/术语）经 --initial_prompt 传入。
        """
        cmd = [
            str(self._exe),
            "-m", self._model_size,
            "--device", self._device,
            "--compute_type", self._compute_type,
            "--output_format", "srt",
            "--output_dir", str(out_dir),
            "--print_progress",
            "--word_timestamps", "true",
            "--vad_filter", "true",
            "--beep_off",
            "--hallucinations_list_off",
        ]
        if language:
            cmd += ["-l", language]
        if hint:
            cmd += ["--initial_prompt", hint]
        # 模型缓存目录：钉在引擎目录旁 _models，不落 C 盘用户缓存
        cmd += ["--model_dir", str(self._engine_dir / "_models")]
        cmd.append(str(audio_path))
        return cmd

    # ── 流式执行 + 进度解析 ─────────────────────────────────
    def _run_streaming(self, cmd: list[str], progress_cb=None) -> int:
        """Popen 流式读输出，解析「NN%」进度行（同 crisp_engine._run_streaming）。"""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
        )
        pat = re.compile(r"(\d{1,3})\s*%")
        try:
            for line in proc.stdout:
                m = pat.search(line)
                if m and progress_cb:
                    p = int(m.group(1))
                    if 0 <= p <= 100:
                        progress_cb(min(99, p), 100, f"Faster-Whisper 转录中… {p}%")
        finally:
            try:
                proc.wait(timeout=_PROC_SHUTDOWN_GRACE_SECS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"faster-whisper-xxl.exe 异常退出（返回码 {proc.returncode}）。\n"
                "请检查显存是否充足，或在模型页改用其他核心。")
        return proc.returncode

    # ── 文件转录 → SRT（与 crisp_engine 同一后处理管线）═════
    def process_file(self, audio_path: Path, progress_cb=None,
                     language: str | None = None, context: str | None = None,
                     diarize: bool = False, n_speakers: int | None = None,
                     original_path: Path | None = None,
                     out_format: str | None = None) -> Path | None:
        """音频 → SRT。后处理与 crisp_engine 共用分行器，输出契约一致。"""
        # _parse_srt_words / _fix_leading_punct 定义在 crisp_engine（Whisper 系
        # SRT 解析与断句修正的既有归属），直接复用避免复制逻辑。
        from crisp_engine import _parse_srt_words, _fix_leading_punct

        audio_path = Path(audio_path)
        if progress_cb:
            progress_cb(0, 1, "Faster-Whisper 转录中…")

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            cmd = self._build_cmd(audio_path, out_dir, language, context)
            with self._lock:
                self._run_streaming(cmd, progress_cb)
            srt_tmp = out_dir / (audio_path.stem + ".srt")
            if not srt_tmp.exists():
                # XXL 对同名输出有加后缀逻辑，兜底 glob
                found = list(out_dir.glob("*.srt"))
                if not found:
                    return None
                srt_tmp = found[0]
            raw = srt_tmp.read_text(encoding="utf-8", errors="replace")

        # ── 与 crisp_engine 同一断句契约：whisper 无标点 → break_on_space=True ──
        ts_items, raw_text = _parse_srt_words(raw, drop_punct=False)
        if not ts_items:
            return None
        cc_use = None  # XXL 路径：模型原文（简/原语言），不做 OpenCC 繁化
        self._last_segments_rich = None
        lines5 = _ts_chatllm_to_subtitle_lines(
            ts_items, raw_text, 0.0, None, cc_use, True,
            break_on_space=True, with_words=True,
        )
        if not lines5:
            return None
        lines5 = _fix_leading_punct(lines5)
        lines = [(s, e, t, sp) for (s, e, t, sp, _w) in lines5]

        # 说话者分离：与其他引擎相同的外部 ONNX diar（由上层挂 diar_engine）
        if diarize and self.diar_engine is not None and getattr(self.diar_engine, "ready", False):
            lines = self._apply_diarization(audio_path, lines, n_speakers, progress_cb)

        # 字级 rich（卡拉OK）：与 crisp_engine 相同的 words 结构。
        # speaker 取 diarize 后的 lines、words 取 lines5（两者行数一致，zip 对齐），
        # 否则开启分离时侧通道丢失说话者标签，与落盘 SRT 不一致。
        try:
            self._last_segments_rich = [
                {"start": a[0], "end": a[1], "text": a[2], "speaker": a[3], "words": b[4]}
                for a, b in zip(lines, lines5)]
        except Exception:
            self._last_segments_rich = None

        # 写出 SRT：write_transcript 的 ref 决定落盘名（与 crispasr 的
        # original_path 语义一致——上层传 subtitles/<原始文件名>）。
        # out_format 转发给 write_transcript（与 crisp_engine 契约一致）；
        # 走 transcribe() 兼容入口时已显式固定 "srt"。
        from subtitle_lines import write_transcript as _wt
        out_ref = Path(original_path) if original_path else audio_path
        return _wt(out_ref, lines, out_format=out_format or "srt")

    # ── 说话者分离（与 crisp_engine 相同的指派逻辑）─────────
    def _apply_diarization(self, audio_path, lines, n_speakers, progress_cb):
        """按 diar 段落把每行指派给说话者（时间中点归属）。"""
        if progress_cb:
            progress_cb(1, 1, "说话者分离中…")
        try:
            # diar_engine.diarize 需要 16kHz float32 ndarray（与 crisp_engine 一致，
            # 先解码再传入；直接传 Path 会在引擎内抛错被 except 吞掉 → 静默无标签）。
            from audio_io import load_audio_16k_mono
            audio, _ = load_audio_16k_mono(Path(audio_path), SAMPLE_RATE)
            diar_segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
        except Exception:
            return lines
        if not diar_segs:
            return lines

        def _spk_of(mid: float) -> str | None:
            for t0, t1, spk in diar_segs:
                if t0 <= mid <= t1:
                    return spk
            # 未落在任一段（静音间隙）→ 取最近段
            return min(diar_segs, key=lambda d: min(abs(d[0] - mid), abs(d[1] - mid)))[2]

        out = []
        for (s, e, t, _old) in lines:
            spk = _spk_of((s + e) / 2)
            label = None
            if spk:
                digits = "".join(c for c in str(spk) if c.isdigit())
                label = f"说话者{int(digits)}" if digits else str(spk)
            out.append((s, e, t, label))
        return out

    # ── 实时/单段（录制视图逐段上传走 process_file，无需单独实现）───
    def transcribe(self, audio_path: Path, language=None, hint=None):
        """兼容旧接口：返回文本列表（录制路径目前只走 process_file）。"""
        srt = self.process_file(audio_path, language=language, context=hint,
                                out_format="srt")   # 显式 srt：不受全域 txt 设置影响
        if not srt:
            return []
        from webview_backend import parse_srt_to_segments   # 实际定义处；延迟导入避免循环
        return parse_srt_to_segments(srt)

    def rebuild_cc(self):
        pass  # XXL 路径不做 OpenCC，保持契约即可

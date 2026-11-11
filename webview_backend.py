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

import re
import sys
import threading
import traceback
from pathlib import Path

import app as core          # 重用 ASREngine / 常数 / 诊断函数（__main__ guard 保护）
from engine_types import TranscribeResult
from settings_store import SettingsStore
from transcription_service import TranscribeService

BASE_DIR = Path(__file__).resolve().parent

# ── 模型目录（拆分为 model_catalog.py；此处兼容再导出全部符号）────────
# tests/ 与 cli_mode 仍从 webview_backend 导入 _MODEL_CATALOG 等。
from model_catalog import (_ENGINE_OV, _ENGINE_CASR, _WHISPER_SIZES,
                           _WHISPER_LABEL_BY_SIZE, _MODEL_CATALOG,
                           _QWEN_CASR_QUANT_LABEL, _QWEN_JA_CASR_QUANT_LABEL,
                           _WHISPER_MODEL_NOTES, _MODEL_NOTES, _sanitize_tag,
                           _QWEN_CASR_MODELS, _QWEN_CASR_NOTE, _qwen_casr_dl,
                           _COMMON_LANGS)

# ── 长音频 FFmpeg 切片转录（纯函数层拆分为 slicing.py）────────────────
# 切片计划、重叠区合并去重、SRT 解析、跨片说话者匹配等构件见 slicing.py；
# 本模块保留调度层（_transcribe_sliced）并兼容再导出，tests/ 不受影响。
#
# 说话者分离是「与后端无关的外部 ONNX」(diarize.py / DiarizationEngine)：
# OpenVINO 与 CRISPASR(whisper/qwen) 皆支持——前者 process_file 内置 use_diar 分支，
# 后者由 crisp_engine._apply_diarization 依时间指派。diar_engine 由 _ensure_diarization 挂上。
from slicing import (_SLICE_THRESHOLD_SECS, _SLICE_WINDOW_SECS,
                     _SLICE_OVERLAP_SECS, _SLICE_MEM_BUDGET_FRAC,
                     _SLICE_MEM_PER_SEC, _SLICE_WINDOW_MIN_SECS,
                     _avail_phys_bytes, _slice_window_secs, _ffprobe_duration,
                     _slice_plan, _merge_sliced_segments,
                     _SPK_MATCH_SIM, _match_speaker, _srt_ts_to_sec,
                     _parse_srt, parse_srt_to_segments, _rich_to_segments)


# 说话者分离是「与后端无关的外部 ONNX」(diarize.py / DiarizationEngine)：
# OpenVINO 与 CRISPASR(whisper/qwen) 皆支持——前者 process_file 内置 use_diar 分支，
# 后者由 crisp_engine._apply_diarization 依时间指派。diar_engine 由 _ensure_diarization 挂上。

# 跨片说话者匹配阈值 _SPK_MATCH_SIM / _match_speaker：见 slicing.py（兼容再导出）。


class WebBackend(TranscribeService):
    """桌面 WebView 的后端。HTTP 层（webview_server）持有一个实例。

    转录编排（transcribe / _transcribe_sliced / ffmpeg 就绪）继承自
    transcription_service.TranscribeService；本类持有全部共享状态并负责
    模型加载、设置、历史与队列管理。
    """

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
        # settings.json 单点读写（锁 + 原子写 + mtime 缓存）；cli_mode 的临时
        # 副本机制在构造本类前改 core.SETTINGS_FILE，故这里取到的已是生效路径。
        self.settings = SettingsStore(
            getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        # ── 批量转录队列：并发第二请求不再直接报错而是排队（串行执行）────
        # 事件经 _queue_event → SSE（event=queue）推送；/api/transcribe 的
        # 同步语义不变（仍走 transcribe()），队列仅供批量入口使用。
        from transcription_queue import TaskQueue
        self._tx_queue = TaskQueue(on_event=self._queue_event)
        self._seed_defaults()            # 首次启动（无 backend）→ 种子默认模型
        self._apply_runtime_prefs()      # 启动即套用持久化偏好（VAD/简繁/镜像/格式）

    # ── 首次启动默认：1.7B Qwen on CRISPASR Q4（多数机器跑得动）─────────────
    def _seed_defaults(self):
        """settings.json 尚无 backend 时，种下默认模型选择。

        多数状况下「Qwen3-ASR-1.7B Q4（CRISPASR/Vulkan）」可顺利运行（GPU Vulkan，
        繁体佳）。写入 settings 后，后续 _persisted_backend/_load_worker 等皆读到它。
        既有使用者（已有 backend）不受影响。
        """
        if self.settings.read().get("backend"):
            return                       # 既有设置 → 不覆盖
        self.settings.update({"backend": "crispasr", "crisp_model": "qwen3",
                              "crisp_qwen_quant": "q4"})

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
        # 快照本次加载的选择：加载（含下载）期间 set_model 仍可写入 settings
        # （「选择本身仍被记住」），完成时若重读会把「已选」误当「已加载」。
        load_identity = self._identity_for(backend, self._settings_raw())
        log_model(f"开始加载：{eng_label} · {model_label}（backend={backend}）")
        # 解析后的实际目录留痕：「明明有模型却说未下载」类反馈的第一排查点
        # （settings 残留 model_dir／crispasr_dir 指向别处时，此处一眼可见）。
        log_model(f"模型目录：{self._model_dir()}")
        if backend == "crispasr":
            log_model(f"CrispASR 目录：{self._crispasr_dir()}")
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
                    self._remember_active(backend, load_identity,
                                          (eng_label, model_label))
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
        return self.settings.read()

    def _vk_device_id(self, s: dict) -> int:
        import re
        m = re.search(r"GPU:(\d+)", s.get("device", "") or "")
        return int(m.group(1)) if m else 0

    # 每段最长秒数已改为任务自适应（不再有设置项）：
    # 开「字级对齐」→ 段上限压到 _ALIGN_CHUNK_SECS（抑制长段歌词/语音的
    # 「段尾字级跑掉」漂移，与 1.7B 模型出厂上限一致）；关对齐 → 模型上限
    # （0.6B 30s，段更少、吞吐更高）。永远不超过模型天花板（音频编码器导出
    # 长度写死，超了会被静默截断掉字）。crispasr（无 max_chunk_secs 类别属性）
    # 走自身窗口，不受影响。
    _ALIGN_CHUNK_SECS = 10

    def _apply_chunk_secs(self, align: bool) -> None:
        eng = getattr(self, "engine", None)
        ceiling = getattr(type(eng), "max_chunk_secs", None) if eng is not None else None
        if not isinstance(ceiling, int):
            return                                  # crispasr 等：不以 max_chunk_secs 分段
        eff = min(self._ALIGN_CHUNK_SECS, ceiling) if align else ceiling
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

    @staticmethod
    def _log_missing(label: str, root, missing: list):
        """触发补下载时的缺失清单留痕（logs/ 供远程排查；截前 4 个防刷屏）。"""
        from applog import log_model
        head = ", ".join(str(m) for m in missing[:4])
        more = f" 等 {len(missing)} 个" if len(missing) > 4 else ""
        log_model(f"{label}缺 {len(missing)} 个文件（{root}）：{head}{more}")

    def _load_openvino(self):
        s = self._settings_raw()
        model_dir = Path(s.get("model_dir", str(core._DEFAULT_MODEL_DIR)))
        use_17b = "1.7B" in s.get("cpu_model_size", "0.6B")
        cpu_threads = int(s.get("cpu_threads", 0) or 0)
        if use_17b:
            from downloader import quick_check_1p7b, download_1p7b, missing_files_1p7b
            if not quick_check_1p7b(model_dir):
                self._log_missing("1.7B 模型", model_dir, missing_files_1p7b(model_dir))
                self._st("下载 1.7B 模型（约 4.3 GB）…")
                download_1p7b(model_dir, progress_cb=self._dl_progress)
        else:
            # 0.6B：文件缺失时按需补齐（含 VAD onnx）。此前只靠开发环境手工放置，
            # 干净机器上选 0.6B 会直接在 eng.load 崩溃（找不到 audio_encoder_model.xml）。
            from downloader import quick_check, download_all, missing_files
            if not quick_check(model_dir):
                self._log_missing("0.6B 模型", model_dir, missing_files(model_dir))
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
                                aligner_gguf_filename, crispasr_gpu_backend,
                                crispasr_core_issue)
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
            from applog import log_model
            issue = crispasr_core_issue(crispasr_dir, variant) or "缺 crispasr.exe"
            log_model(f"CrispASR 加速器不就绪（{crispasr_dir}）：{issue} → 重新下载 zip")
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
                self._log_missing(f"Qwen3-ASR-1.7B{_note} {qquant.upper()} GGUF",
                                  model_dir, [fname_fn(qquant)])
                self._st(f"下载 Qwen3-ASR-1.7B{_note} {qquant.upper()} GGUF…")
                download_fn(model_dir, qquant, progress_cb=self._dl_progress)
        else:
            # OpenAI Whisper 官方 GGML（ggerganov/whisper.cpp）：whisper 后端原生支持。
            # 尺寸（base/small/medium/large/turbo）由 whisper_size 决定，缺文件自动下载。
            size = s.get("whisper_size", "base")
            model_path = crispasr_dir / whisper_ggml_filename(size)
            if not quick_check_whisper_ggml(crispasr_dir, size):
                self._log_missing(f"Whisper（{size}）GGML", crispasr_dir,
                                  [whisper_ggml_filename(size)])
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
                 gpu_backend=crispasr_gpu_backend(variant),
                 cpu_threads=int(s.get("cpu_threads", 0) or 0))
        self.engine = eng

    # ── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict:
        active = getattr(self, "_active_backend", "openvino")
        loaded = getattr(self, "_active_label", None)
        # 热切换语义：实际加载的引擎身份 vs 当前持久化选择的身份。
        # 不一致 → 左下角状态点显示「待切换」（此刻转录跑的仍是旧模型）。
        active_identity = getattr(self, "_active_identity", None)
        selection_differs = bool(
            active_identity is not None
            and active_identity != self._identity_for(self._persisted_backend(),
                                                      self._settings_raw()))
        selected_ready = self.selected_model_present()
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
            "selectedReady": selected_ready,
            # 未就绪时的判定依据（检查了哪个目录、缺哪些文件）；就绪时为 None
            "selectedDetail": None if selected_ready else self._selected_model_detail(),
            # 左下角状态点用：实际加载的模型 vs 当前选择的模型
            "activeModel": f"{loaded[0]} · {loaded[1]}" if loaded else None,
            "selectedModel": " · ".join(self._current_selection()),
            "selectionDiffers": selection_differs,
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

    def _selected_model_detail(self) -> dict:
        """目前所选模型的检查明细：判定的目录 + 缺了哪些文件。

        「模型未下载」提示的判定依据展示：quick_check 家族只回真假，使用者
        看到「未下载」却无法核对程序到底查了哪个目录、缺什么（用户反馈实录：
        ov_models 明明有模型，重启后仍提示重新下载——多为 settings 残留
        model_dir 指向别处、或所选模型与目录里的文件不是同一套）。只在
        selected_model_present() 为 False 时由 get_status() 调用，正常就绪
        零开销。missing 相对 model_dir，missingCrispasr 相对 crispasr_dir。
        """
        from downloader import (quick_check, quick_check_1p7b, quick_check_crispasr,
                                quick_check_whisper_ggml, missing_files,
                                missing_files_1p7b, missing_files_whisper_ggml,
                                missing_files_qwen3_asr_gguf,
                                missing_files_qwen3_asr_ja_gguf, crispasr_core_issue)
        s = self._settings_raw()
        be = s.get("backend", "openvino")
        model_dir = self._model_dir()
        detail = {"backend": be, "modelDir": str(model_dir),
                  "crispasrDir": None, "missing": [],
                  "missingCrispasr": [], "coreIssue": ""}
        if be == "openvino":
            if "1.7B" in s.get("cpu_model_size", "0.6B"):
                if not quick_check_1p7b(model_dir):
                    detail["missing"] = missing_files_1p7b(model_dir)
            elif not quick_check(model_dir):
                detail["missing"] = missing_files(model_dir)
            return detail
        cd = self._crispasr_dir()
        detail["crispasrDir"] = str(cd)
        if not quick_check_crispasr(cd):        # 与 selected_model_present 同口径：不带 variant
            detail["coreIssue"] = crispasr_core_issue(cd)
        cm = s.get("crisp_model", "whisper")
        if cm in _QWEN_CASR_MODELS:
            quant = s.get("crisp_qwen_quant", "q8")
            miss = (missing_files_qwen3_asr_ja_gguf(model_dir, quant)
                    if cm == "qwen3-ja"
                    else missing_files_qwen3_asr_gguf(model_dir, quant))
            detail["missing"] = miss
        else:
            detail["missingCrispasr"] = missing_files_whisper_ggml(
                cd, s.get("whisper_size", "base"))
        return detail

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
        return self.settings.read().get("backend", "openvino")

    def _app_version(self) -> str:
        try:
            import version
            # WebView 版优先用独立版本字符串（webview 0.1）；缺则退回语义化版本。
            return getattr(version, "WEBVIEW_VERSION", None) or version.__version__
        except Exception:
            return ""

    # ── 转录编排已拆分：transcription_service.TranscribeService（本类继承之）──
    # transcribe / _transcribe_sliced / _resolve_ffmpeg / _ensure_ffmpeg 见该模块；
    # self.* 状态（engine/_cancel/_emit/_dl_progress/settings…）仍归本类所有。

    # （_transcribe_sliced / _ensure_ffmpeg 亦在 transcription_service 内。）

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
        """请求取消当前转录。

        _cancel 标志由切片循环/收尾检查；引擎侧的 cancel_event（CrispASR）
        让单片转录中的子进程立即被终止，而非等到下一片边界。
        OpenVINO 引擎逐段（max_chunk_secs）转录，本标志在段间生效。
        """
        self._cancel = True
        cancel_ev = getattr(self.engine, "cancel_event", None)
        if cancel_ev is not None:
            cancel_ev.set()
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

    def check_update(self) -> dict:
        """真实更新检查：GitHub API 查最新 release，与当前版本比对。

        网络失败（离线/限流）回 ok=False，由前端提示而非误报「已是最新」。
        """
        try:
            import version
            cur = str(getattr(version, "WEBVIEW_VERSION", None)
                      or getattr(version, "__version__", "") or "")
            page = str(getattr(version, "GITHUB_RELEASES_PAGE", ""))
            repo = str(getattr(version, "GITHUB_REPO", ""))
            if not repo:
                return {"ok": False, "error": "未配置更新来源", "url": page}
            import requests
            r = requests.get(
                f"https://api.github.com/repos/{repo}/releases/latest",
                timeout=8, headers={"Accept": "application/vnd.github+json"})
            r.raise_for_status()
            tag = str(r.json().get("tag_name") or "").strip()
            latest = tag.lstrip("vV")
            return {"ok": True, "current": cur, "latest": latest or tag,
                    "hasUpdate": self._ver_newer(latest, cur), "url": page}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "url": ""}

    @staticmethod
    def _ver_newer(a: str, b: str) -> bool:
        """语义化版本比较 a > b（逐段数值比较，短的补 0）。

        预发布号（-rc.1 / -beta2…）按 semver 语义视为**低于**同数值正式版：
        "2.3.0-rc.1" < "2.3.0"（旧实现把 "-rc.1" 拆成数值段记 0，会误判
        预发布比正式版新）。构建元数据（+build.1）不参与比较。
        """
        def parse(v: str) -> tuple[list[int], bool]:
            v = v.strip().lstrip("vV").split("+", 1)[0]
            pre = "-" in v
            main = re.split(r"[-.]", v.split("-", 1)[0])
            nums = [int(x) if x.isdigit() else 0 for x in main if x != ""]
            return nums, pre

        na, a_pre = parse(a)
        nb, b_pre = parse(b)
        n = max(len(na), len(nb))
        na += [0] * (n - len(na))
        nb += [0] * (n - len(nb))
        if na != nb:
            return na > nb
        # 数值相同：正式版 > 预发布（a 已发布、b 仍是预发布才算更新）
        return (not a_pre) and b_pre

    # ── 转录历史（subtitles/ 目录的产出回填）───────────────────────────
    def list_history(self) -> dict:
        """subtitles/ 下最近的字幕文件（新→旧，最多 50 条）。"""
        d = Path(getattr(core, "SRT_DIR", BASE_DIR / "subtitles"))
        items = []
        try:
            d.mkdir(parents=True, exist_ok=True)
            for p in d.iterdir():
                if p.suffix.lower() not in (".srt", ".txt") or not p.is_file():
                    continue
                st = p.stat()
                items.append({"name": p.name, "mtime": int(st.st_mtime),
                              "size": st.st_size})
        except Exception:
            traceback.print_exc()
        items.sort(key=lambda x: -x["mtime"])
        return {"items": items[:50]}

    def load_history(self, name: str) -> dict:
        """按文件名回填历史字幕 → segments（只接受基名，杜绝路径穿越）。"""
        safe = Path(str(name or "").replace("\\", "/")).name
        d = Path(getattr(core, "SRT_DIR", BASE_DIR / "subtitles"))
        p = d / safe
        try:
            if d.resolve() not in p.resolve().parents:
                return {"ok": False, "error": "路径不合法"}
        except Exception:
            return {"ok": False, "error": "路径不合法"}
        if not p.is_file():
            return {"ok": False, "error": "找不到文件"}
        try:
            if p.suffix.lower() == ".srt":
                segments = parse_srt_to_segments(p)
            else:
                text = p.read_text(encoding="utf-8", errors="replace")
                segments = ([{"start": 0.0, "end": 0.0, "speaker": None,
                              "text": text}] if text.strip() else [])
            return {"ok": True, "name": p.name, "segments": segments}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── 模型下拉（核心 + 模型）＋ 识别语言 ─────────────────────────────
    #   持久化后走「切换=重启」(同 set_backend 的理由：避免就地切核心死机)。
    def _remember_active(self, backend: str, identity=None, labels=None):
        """记住「实际加载」的选择识别码与标签（字幕文件名/日志标注用）。

        identity/labels 缺省时读当下 settings——仅适用于加载中不存在选择
        漂移的路径（如加载失败退回 OpenVINO：退回动作本身就用的当前设置）。
        正常加载请传 _load_worker 开始时拍下的快照。
        """
        self._active_backend = backend
        self._active_identity = (identity if identity is not None
                                 else self._identity_for(backend, self._settings_raw()))
        self._active_label = labels if labels is not None else self._current_selection()

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
        self.settings.update({"backend": backend})

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

        cur = self.settings.update({"backend": backend, **patch})

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

    def delete_history(self, name: str) -> dict:
        """删除 subtitles/ 里的一个历史字幕文件到回收站（Windows）。

        与 list_history/load_history 同一套 basename 防穿越约束。回收站
        删除走 PowerShell VisualBasic API；非 Windows／调用失败回退为普通
        删除（调用端如实收到 recycled=False）。文件不存在回 ok=False。
        """
        safe = Path(str(name or "").replace("\\", "/")).name
        d = Path(getattr(core, "SRT_DIR", BASE_DIR / "subtitles"))
        p = d / safe
        if not safe or safe.startswith("."):
            return {"ok": False, "error": "路径不合法"}
        try:
            if d.resolve() not in p.resolve().parents:
                return {"ok": False, "error": "路径不合法"}
        except Exception:
            return {"ok": False, "error": "路径不合法"}
        if not p.is_file():
            return {"ok": False, "error": "找不到文件"}

        recycled = False
        if sys.platform == "win32":
            try:
                import subprocess
                # 单引号转义：文件名含 ' 时以 '' 闭合再重启字符串
                ps = ("Add-Type -AssemblyName Microsoft.VisualBasic; "
                      "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
                      f"'{str(p).replace(chr(39), chr(39)*2)}', "
                      "'OnlyErrorDialogs', 'SendToRecycleBin')")
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps],
                    capture_output=True, timeout=15,
                    creationflags=0x08000000)
                recycled = (r.returncode == 0 and not p.exists())
            except Exception:
                traceback.print_exc()
        if not recycled:
            try:
                p.unlink()
            except OSError as e:
                return {"ok": False, "error": f"删除失败：{e}"}
        return {"ok": True, "recycled": recycled}

    # ── 设置（读写既有 settings.json）───────────────────────
    def get_settings(self) -> dict:
        s = self.settings.read()
        # 简繁词汇转换已从设置页移除：引擎固定输出模型原文（简体）。
        return {
            "scale": int(s.get("ui_scale", 100)),
            "modelDir": s.get("model_dir", ""),
            "format": s.get("output_format", "srt"),
            "mirror": s.get("hf_mirror", ""),
            "ffmpeg": s.get("ffmpeg_path", ""),
            "theme": s.get("appearance", "light"),
            "uiLang": s.get("ui_lang", "简体中文"),
            # VAD 灵敏度与每段最长秒数已改任务自适应（无设置项，见 _apply_chunk_secs
            # 与 app._vad_segment_auto）；settings.json 里的旧键忽略不读。
            "modelMode": (s.get("model_mode") if s.get("model_mode") in ("basic", "advanced")
                          else "basic"),
        }

    def set_settings(self, patch: dict) -> dict:
        old_model_dir = self._model_dir()
        patch = patch or {}
        mapped = {}
        key_map = {"scale": "ui_scale", "format": "output_format",
                   "mirror": "hf_mirror", "ffmpeg": "ffmpeg_path", "theme": "appearance",
                   "uiLang": "ui_lang", "modelMode": "model_mode"}
        for k, v in patch.items():
            if k in key_map:
                mapped[key_map[k]] = v
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
                    self._model_dir_changed = (new_dir != str(old_model_dir))
                    mapped["model_dir"] = new_dir
                except Exception:
                    pass          # 无效路径 → 不写入
            else:
                self._model_dir_changed = "model_dir" in self.settings.read()
                self.settings.remove("model_dir")      # 空 → 回退默认位置
        if mapped:
            cur = self.settings.update(mapped)
        else:
            cur = self.settings.read()
        # ── 实时套用副作用（与 app.py 的 _on_*_change 对齐）──────────────
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
        self.settings.update({key: value})

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

    def _apply_runtime_prefs(self):
        """启动时把持久化偏好一次套用到各引擎模块全域。

        在 start_load() 之前调用 → 模块标志先就位，镜像/输出格式皆于首次转录前生效。
        （VAD 灵敏度已改自动重试、无设置项，见 app._vad_segment_auto。）
        """
        s = self._settings_raw()
        self._apply_mirror(s.get("hf_mirror", ""))
        self._apply_output_format(s.get("output_format", "srt"))

    # ── 批量转录队列（transcription_queue.TaskQueue 的包装）───────────────
    def _queue_event(self, kind: str, job: dict):
        """队列事件 → SSE（event=queue）。job 已脱敏（无 fn），可直接序列化。"""
        self._emit("queue", {"kind": kind, **job})

    def enqueue_transcribe(self, opts: dict, progress_cb=None) -> str:
        """把一个转录任务排入队列，立即返回 job_id。

        执行体复用 transcribe()（含切片/取消/输出命名全部语义）；排队中取消
        → 直接跳过，运行中取消 → cancel_job 会接通现有取消链（self._cancel +
        引擎 cancel_event）。并发上限由 TaskQueue(maxsize) 控制，超出抛错。
        """
        label = Path(str(opts.get("path", ""))).name or "transcribe"

        def _job(_cancel_event):
            # 取消走 WebBackend.cancel()（cancel_job 接通），不走队内事件：
            # 引擎子进程终止/切片段间检查两条既有链路都已挂在 self._cancel 上。
            return self.transcribe(opts, progress_cb=progress_cb)

        return self._tx_queue.submit(_job, label=label)

    def queue_status(self) -> dict:
        """队列快照（全部任务 + 当前运行 id）。"""
        return {"jobs": self._tx_queue.status(),
                "current": self._tx_queue.current_id()}

    def cancel_job(self, jid: str) -> bool:
        """取消指定任务（排队中→跳过；运行中→接通现有取消链）。"""
        ok = self._tx_queue.cancel(jid)
        if ok and self._tx_queue.current_id() == jid:
            self.cancel()                # 运行中：复用 /api/cancel 的取消链
        return ok

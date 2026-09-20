"""
crisp_engine.py — CrispASR (ggml / Vulkan) Whisper 推理后端

定位：把 CrispASR 的 whisper backend 当作本项目的第三个推理引擎（与
ASREngine[OpenVINO] / ChatLLMASREngine[chatllm] 并列），用来加载 Whisper
类模型（默认 OpenAI Whisper 官方 GGML）。

设计重点：
  • CrispASR 是 whisper.cpp fork，crispasr.exe 自己会输出 SRT（含分段时间轴），
    因此本引擎不需重造 VAD/chunk/FA，本质是「调用 crispasr.exe → 读 SRT →
    OpenCC 繁化 → 写回」。
  • GPU 后端由 load(gpu_backend=) 决定，对应「使用者选的 CrispASR 加速版本」
    （downloader.crispasr_gpu_backend）：
      - vulkan（默认）：Windows 通吃 Intel/AMD 核显与 NVIDIA 独显，体积最小。
      - cuda：NVIDIA 专用，需下载 CUDA build（0.8.32 起在 Blackwell 实测正常；
        早期 0.7.1 的 fattn crash 为 nemotron 后端专属问题）。
      - None：CPU build，改挂 -ng（该 build 没有 ggml-vulkan/cuda DLL）。
  • 界面对齐 app.py 既有引擎契约：load / ready / transcribe / process_file /
    processor / diar_engine / use_aligner / aligner / _lock / rebuild_cc。

对外界面（app.py 的 _load_models[crispasr 分支] 会这样调用）：
    eng = CrispWhisperEngine()
    eng.load(model_path=..., crispasr_dir=..., device_id=0, cb=set_status)
    srt = eng.process_file(audio_path, progress_cb=..., language=..., ...)
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

# 与 Qwen 路径统一的字幕分行（字级时间轴 → 字幕行，全引擎共享）
from subtitle_lines import (_ts_chatllm_to_subtitle_lines, _srt_ts, write_transcript,
                            _ZH_CLAUSE_END, _EN_SENT_END)

# 中文/英文标点集合（与断句器一致）：Qwen 模式下标点只当切点、不进 items
_PUNCT = _ZH_CLAUSE_END | _EN_SENT_END

# ── 输出语言标志（由 app.py 切换时同步设置，与 chatllm_engine 行为一致）──
_output_simplified: bool = True    # True=输出模型原始（简体）；False=OpenCC 转换
_vocab_convert:     bool = True    # True=s2twp(含台湾词)；False=s2t(仅字形)


def _opencc_config() -> str:
    return "s2twp" if _vocab_convert else "s2t"


# ── 共享常数 ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16000

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Windows：隐藏子程序控制台窗口（避免识别时画面闪烁）
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = None

# 实时/单段转录子进程超时：单段音频通常 <30s，crispasr 处理应在几分钟内完成。
# 超时即认为挂死（Vulkan 初始化卡死等），杀进程并抛错，避免永久持有 self._lock。
_RT_PROC_TIMEOUT_SECS = 600
# 流式模式收尾宽限：正常退出等待上限（秒）；超时强杀。
_PROC_SHUTDOWN_GRACE_SECS = 30
if sys.platform == "win32":
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUP_INFO.wShowWindow = 0  # SW_HIDE

# 默认 crispasr 目录（仿 chatllm/ 惯例）与模型文件名
_DEFAULT_CRISPASR_DIR = BASE_DIR / "crispasr"
_DEFAULT_MODEL_NAME   = "ggml-base.bin"   # OpenAI Whisper Base（ggerganov/whisper.cpp）
# qwen3 ForcedAligner GGUF（FA 对齐器）文件名样式，置于 crispasr_dir 同层
_ALIGNER_GLOB = "qwen3-forced-aligner-0.6b-*.gguf"


def _find_aligner_gguf(crispasr_dir: Path) -> Path | None:
    """在 crispasr 目录查找 qwen3 ForcedAligner GGUF（取第一个符合者）。"""
    for g in sorted(crispasr_dir.glob(_ALIGNER_GLOB)):
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

    这是 chatllm `probe_vulkan_devices()` 的对应物，但走 CrispASR 自带的
    `--diagnostics`（会列举 ggml 后端与设备后立即退出，不需模型/音频）。
    让设备检测**不再依赖 chatllm**（main.exe 未随安装包提供时也能检测 GPU）。

    解析目标是 diagnostics 输出中的「registered devices」区块，例如：
        registered devices : 2
          [0] gpu    name=Vulkan0 desc=NVIDIA GeForce RTX 5090 mem=31418/32186 MiB id=0000:01:00.0
          [1] cpu    name=CPU desc=13th Gen Intel(R) Core(TM) i5-13500 mem=79978/98045 MiB id=?
    每行 → {'id','name','vram_free'(bytes),'backend','is_cpu'}；mem 为 free/total MiB。

    返回 dict 与 probe_vulkan_devices() 同结构：
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
            pending.append({
                "id":        int(m.group(1)),
                "name":      m.group(4).strip(),                 # 人类可读名称
                "vram_free": int(m.group(5)) * 1024 * 1024,      # MiB → bytes
                "backend":   "CPU" if is_cpu else "VULKAN",
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
#   一直没事，是因为既有模型都吐简体、OpenCC 繁化发生在对齐**之后** —— 对齐器
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


class CrispWhisperEngine:
    """CrispASR(whisper backend) 推理引擎。子程序模式调用 crispasr.exe。"""

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
        self.cc          = None    # OpenCC 转换器

    # ══ 加载 ══════════════════════════════════════════════════════════

    def load(self, model_path: Path = None, crispasr_dir: Path = None,
             device_id: int = 0, cb=None, aligner_path: Path = None,
             backend: str | None = None, gpu_backend: str | None = "vulkan"):
        """验证 crispasr.exe 与模型存在即可（子程序模式无常驻加载）。

        aligner_path：qwen3 ForcedAligner GGUF；None 时自动在 crispasr_dir 内查找。
        找到 → 启用 FA（process_file 加 `-am <gguf> -falign`）；找不到 → 退回
        whisper native 时间轴。

        backend：crispasr 后端名（whisper / qwen3 / qwen3-1.7b …）。None 时依
        模型文件名自动推断（whisper GGML→whisper、qwen3*.gguf→qwen3-1.7b）。

        gpu_backend：crispasr.exe 的 `--gpu-backend`（vulkan / cuda）。传 None
        代表这份核心是 CPU 版或使用者选了纯 CPU 推理 → 改挂 `-ng`。调用端应
        依「实际安装的加速版本」给值（downloader.crispasr_gpu_backend）。
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
        # 后端：显式指定优先，否则依模型文件名自动推断
        self._backend    = backend or _infer_backend(model_path)

        # ── 检测 FA 对齐器 GGUF（有则默认启用）──────────────────────────
        #   先判断这颗模型能不能用对齐器（见 _FA_INCOMPATIBLE 的说明）；不能用
        #   就整条 FA 关掉，避免对齐器吐出私用区乱码并把时间轴对歪。
        self._fa_blocked = _fa_incompatible(model_path)
        if self._fa_blocked:
            _s("此模型原生输出繁体，与 CTC 对齐器不兼容 → 禁用字级对齐，"
               "改用后端自带时间轴")
        ap = Path(aligner_path) if aligner_path else _find_aligner_gguf(crispasr_dir)
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

        import opencc
        self.cc = opencc.OpenCC(_opencc_config())
        self.ready = True
        _fa_tag = "  + FA" if self._aligner_path else ""
        _s(f"就绪（CrispASR / Vulkan / {self._backend}  {model_path.name}{_fa_tag}）")

    def rebuild_cc(self):
        """依目前词汇转换标志重建 OpenCC（免重新加载）。"""
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            pass

    # ══ 子程序命令建构（★ 由你定夺：speed/quality 取舍）═════════════════

    def _build_cmd(self, audio_path: Path, out_base: Path,
                   language: str | None, word_level: bool = True) -> list[str]:
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
        # 非 whisper 架构（Qwen3-ASR 等）需显式指定后端，crispasr 才用对解码路径
        if self._backend and self._backend != "whisper":
            cmd += ["--backend", self._backend]
        # 加速版本：Vulkan／CUDA 走 --gpu-backend + 设备 id；CPU 版挂 -ng
        # （CPU 版核心没有 ggml-vulkan/cuda DLL，硬给 --gpu-backend 会启动失败）
        if self._gpu_backend:
            cmd += ["--gpu-backend", self._gpu_backend,
                    "-dev", str(self._device_id)]
        else:
            cmd += ["-ng"]
        cmd += [
            "-of", str(out_base),         # 输出文件名（不含副文件名）
            "-np",                        # 不印多余消息
        ]
        if word_level:
            # -ml 1 → 每段一字（字符级时间轴），等价于 FA 的字级输出
            # -pp → 打印「progress = NN% (x/N slices)」，供流式解析驱动进度条
            cmd += ["-osrt", "-ml", "1", "-pp"]
        else:
            cmd += ["-otxt"]

        lang_code = _LANG_MAP.get(language, None) if language else None
        if lang_code:
            cmd += ["-l", lang_code]

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

        crispasr 一次处理整档（内部切 N 个 slice），`-pp` 会输出
        「crispasr: progress = NN% (x/N slices)」。据此把单一阻塞子程序变成
        会动的进度条（granularity = slice 数，长档较细、短档较粗）。
        stderr 并入 stdout 一起读；无 `-pp` 或解析不到时，行为等同旧版。

        超时保护：Vulkan 初始化已知可能卡死（probe 路径为此用 10-15s 超时）。
        转录本身可长达数十分钟，故超时按「无输出行间隔」计（_PROC_TIMEOUT_SECS
        内没有任何新输出视为挂死）：超时杀掉子进程并抛 RuntimeError，
        避免永久持有 self._lock 把整个引擎卡死。
        """
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
        )
        pat = re.compile(r"progress\s*=\s*(\d+)\s*%")
        try:
            for line in proc.stdout:
                m = pat.search(line)
                if m and progress_cb:
                    p = int(m.group(1))
                    # 上限 99：留「完成」给写档后的最终回报，避免 100% 却还在写字幕
                    progress_cb(min(99, p), 100, f"CrispASR 转录中… {p}%")
        finally:
            try:
                proc.wait(timeout=_PROC_SHUTDOWN_GRACE_SECS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"crispasr.exe 异常退出（返回码 {proc.returncode}）。\n"
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

        流程：crispasr `-ml 1` 取字符级时间轴 → 解析 → 与 OpenVINO/chatllm
        共享的 `_ts_chatllm_to_subtitle_lines` 分行（含 OpenCC 繁化、孤儿合并）。

        说话者分离（diarize）：用与后端无关的外部 ONNX（diar_engine，CPU）取得
        说话者段落，再依时间把每行字幕指派给对应说话者。whisper/qwen 后端皆适用
        —— crispasr 单次出字级时间轴 × diar 段落，零额外子程序。需先由上层
        （webview_backend._ensure_diarization / app.py）把 diar_engine 挂上。
        context（hint）目前仍未支持，静默忽略。
        """
        audio_path = Path(audio_path)
        if progress_cb:
            progress_cb(0, 1, "CrispASR 转录中…")

        with tempfile.TemporaryDirectory() as td:
            out_base = Path(td) / "crisp_out"
            cmd = self._build_cmd(audio_path, out_base, language, word_level=True)
            with self._lock:
                self._run_streaming(cmd, progress_cb)
            srt_tmp = out_base.with_suffix(".srt")
            if not srt_tmp.exists():
                return None
            raw = srt_tmp.read_text(encoding="utf-8", errors="replace")

        # ── 依后端选对「断句契约」（使用者要求：Qwen 要走 Qwen 逻辑，非 Whisper）──
        #   Qwen3-ASR：模型自带中文标点 → 标点剔出 items、只留 raw_text 当切点，
        #              break_on_space=False → 与 chatllm/OpenVINO **完全相同**的标点切行。
        #   Whisper(Breeze)：无标点 → 保留空白语句边界、break_on_space=True 逼近 Qwen。
        is_qwen = (self._backend or "").startswith("qwen3")
        ts_items, raw_text = _parse_srt_words(raw, drop_punct=is_qwen)
        if not ts_items:
            return None
        if getattr(self, "_fa_blocked", False):
            # 没有对齐器时 `-ml 1` 对 qwen3 后端无效，每个 item 会是一整段长文字，
            # 共享分行器便无从断句（实测整段可达 60 秒）。这里把长 item 依字符
            # **等时内插**摊成字符级，让分行器照常做标点切行／MAX_CHARS 保护。
            # 内插时间轴当然不如 CTC 对齐精准，但在「粗到不能看」与「约略正确」
            # 之间，后者对字幕明显有用；且段落起止仍是后端给的真实时间。
            ts_items = _interpolate_chars(ts_items)

        # 日语：跳过 OpenCC 繁化。s2twp/s2t 是「简体中文→繁体中文」转换，套在日文上
        # 会把日文汉字（会/静/図/学）误转成繁体字形（会/静/图/学），对日语读者是错的
        # （ja-anime 模型本就针对日语，尤其该保留原生字形）。中文/台语维持繁化不变；
        # 自动检测无法预知语言，保守仍走繁化。
        cc_use = None if _LANG_MAP.get(language) == "ja" else self.cc

        self._last_segments_rich = None   # 本次字级结果（卡拉OK侧通道）
        lines5 = _ts_chatllm_to_subtitle_lines(
            ts_items, raw_text, 0.0, None, cc_use, _output_simplified,
            break_on_space=(not is_qwen), with_words=True,
        )
        if not lines5:
            return None
        if getattr(self, "_fa_blocked", False):
            lines5 = _fix_leading_punct(lines5)
        lines = [(s, e, t, sp) for (s, e, t, sp, _w) in lines5]

        # 说话者分离（外部 ONNX，与 whisper/qwen 后端无关）：依时间中点指派每行说话者。
        # diarize 只重新指派 speaker、不增减行，故与 lines5 一一对应 → 可 zip 并回字级。
        if diarize and self.diar_engine is not None and getattr(self.diar_engine, "ready", False):
            lines = self._apply_diarization(audio_path, lines, n_speakers, progress_cb)

        # 字级结果挂上实例：speaker 取 diarize 后的 lines，words 取 lines5
        self._last_segments_rich = [
            {"start": a[0], "end": a[1], "text": a[2], "speaker": a[3], "words": b[4]}
            for a, b in zip(lines, lines5)
        ]

        # 共享写出层：依全域设置（或 out_format 覆盖）产出 .srt 或 .txt。
        ref = original_path if original_path is not None else audio_path
        out = write_transcript(ref, lines, out_format)
        if progress_cb:
            progress_cb(1, 1, "完成")
        return out

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
        except Exception:
            return lines
        if not segs:
            return lines

        def _spk_at(t: float) -> str:
            for (t0, t1, lab) in segs:           # 中点落在哪个说话者段落
                if t0 <= t < t1:
                    return lab
            return min(segs, key=lambda g: abs((g[0] + g[1]) / 2.0 - t))[2]  # 否则取最近段

        return [(s, e, txt, _spk_at((s + e) / 2.0)) for (s, e, txt, _spk) in lines]

    # ══ 实时/单段转录 → 纯文字 ═════════════════════════════════════════

    def transcribe(self, audio: np.ndarray, max_tokens: int = 300,
                   language: str | None = None, context: str | None = None) -> str:
        """16kHz float32 音频 → 文字（实时模式用，不需字级时间轴）。"""
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "seg.wav"
            sf.write(str(wav), audio, SAMPLE_RATE)
            out_base = Path(td) / "seg_out"
            cmd = self._build_cmd(wav, out_base, language, word_level=False)
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
            text = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
        # 日语跳过繁化（同 process_file 理由）；中文/台语维持 s2twp/s2t。
        if _output_simplified or _LANG_MAP.get(language) == "ja" or self.cc is None:
            return text
        return self.cc.convert(text)


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


def _parse_srt_words(srt_text: str, drop_punct: bool = False
                     ) -> tuple[list[tuple[str, float, float]], str]:
    """解析 `-ml 1` 字符级 SRT。

    返回 (ts_items, raw_text)：
        ts_items : [(char, start_s, end_s), ...]（不含空白项）
        raw_text : 字符对接，并在 whisper 的「空白语句边界」处保留一个空白，
                   供分行器 break_on_space 在语句边界切行。

    drop_punct=True（Qwen 模式）：标点字符只保留在 raw_text 当「切点」，**不**进
    ts_items —— 这正是 chatllm/OpenVINO Qwen 路径的契约（items=内容词、标点仅在
    raw_text）。crispasr `-ml 1` 会把标点也输出成独立字符段，若放进 items 会让
    断句器的 raw_text 指标双重前进、标点沦为下一行开头，故需在此剔除。
    """
    items: list[tuple[str, float, float]] = []
    raw_parts: list[str] = []
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
        if raw.strip() == "":
            raw_parts.append(" ")           # 纯空白块 = 语句边界
            continue
        if raw[:1].isspace():               # 字符前带空白 = 边界 + 字符
            raw_parts.append(" ")
        token = raw.strip()
        raw_parts.append(token)             # 标点一律进 raw_text（供切点检测）
        if drop_punct and token and all(ch in _PUNCT for ch in token):
            continue                        # Qwen：标点不进 items
        items.append((token, s, e))
    return items, "".join(raw_parts)



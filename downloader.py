"""
模型完整性检查与自动下载工具

使用纯标准库（urllib）直接下载，支持断点续传。
不依赖 huggingface_hub / torch / transformers。

用法（命令行）：
    python downloader.py            ← 检查后自动下载缺少的模型
    python downloader.py --check    ← 只检查，不下载
"""
from __future__ import annotations

import hashlib
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _ssl_ctx() -> ssl.SSLContext:
    """建立 SSL Context，优先序：certifi bundle → 系统默认 → 不验证（fallback）。

    PyInstaller EXE 中 Python 的 CA bundle 路径常失效，
    certifi 包自带 Mozilla cacert.pem，是最可靠的修法。
    若两者都不可用，才退回「不验证」模式（只用于可信任的 HuggingFace URL）。
    """
    # 优先：certifi 包的 CA bundle
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    # 次选：系统默认（开发环境通常正常）
    try:
        return ssl.create_default_context()
    except Exception:
        pass
    # 最后降级：不验证（frozen EXE CA bundle 完全缺失时）
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    return ctx

# ── 路径（PyInstaller 冻结时指向 EXE 旁边）────────────────────────────
import sys as _sys
if getattr(_sys, "frozen", False):
    BASE_DIR = Path(_sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"

# ── HuggingFace 仓库 ───────────────────────────────────────────────────
# 主要来源（dseditor 备份仓库）；失败时自动切换至备用来源
_HF_REPO_PRIMARY  = "dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
_HF_REPO_FALLBACK = "Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
_HF_BASE_PRIMARY  = f"https://huggingface.co/{_HF_REPO_PRIMARY}/resolve/main"
_HF_BASE_FALLBACK = f"https://huggingface.co/{_HF_REPO_FALLBACK}/resolve/main"
_HF_REPO  = _HF_REPO_PRIMARY   # 兼容旧版引用
_HF_BASE  = _HF_BASE_PRIMARY   # 兼容旧版引用
_VAD_URL  = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
_UA       = "Mozilla/5.0 (compatible; STTMiniTool-downloader)"

# ── HuggingFace 镜像站 ─────────────────────────────────────────────────
# 中国大陆等地直连 huggingface.co 常逾时；可改走镜像（如 hf-mirror.com）。
# set_mirror() 设置后，_download_file() 会把所有 huggingface.co 的网址
# 改写到镜像网域；空字符串代表使用官方来源（默认）。
HF_OFFICIAL = "https://huggingface.co"
_MIRROR_BASE: str = ""   # 例："https://hf-mirror.com"


def set_mirror(base: str | None):
    """设置 HuggingFace 镜像站台基底网址（如 'https://hf-mirror.com'）。

    传入空字符串或 None 表示恢复使用官方 huggingface.co。
    仅改写 huggingface.co 来源；GitHub 等其他网域不受影响。
    """
    global _MIRROR_BASE
    base = (base or "").strip().rstrip("/")
    _MIRROR_BASE = base


def _apply_mirror(url: str) -> str:
    """若已设置镜像且 url 指向 huggingface.co，改写为镜像网域。"""
    if _MIRROR_BASE and url.startswith(HF_OFFICIAL):
        return _MIRROR_BASE + url[len(HF_OFFICIAL):]
    return url

# ── 1.7B INT8 KV-cache 模型仓库 ───────────────────────────────────────
_HF_1P7B_REPO = "dseditor/Qwen3-ASR-1.7B-INT8_OpenVINO"
_HF_1P7B_BASE = f"https://huggingface.co/{_HF_1P7B_REPO}/resolve/main"

_1P7B_REQUIRED_BIN: list[str] = [
    "audio_encoder_model.bin",
    "thinker_embeddings_model.bin",
    "decoder_prefill_kv_model.bin",
    "decoder_kv_model.bin",
]
_1P7B_REQUIRED_OTHER: list[str] = [
    "audio_encoder_model.xml",
    "thinker_embeddings_model.xml",
    "decoder_prefill_kv_model.xml",
    "decoder_kv_model.xml",
    "prompt_template.json",
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "chat_template.json",
]

# ── 说话者分离模型（直接 URL，非 HF API）──────────────────────────────
_DIAR_BASE = "https://huggingface.co/altunenes/speaker-diarization-community-1-onnx/resolve/main"
DIAR_FILES: dict[str, str] = {
    "segmentation-community-1.onnx": f"{_DIAR_BASE}/segmentation-community-1.onnx",
    "embedding_model.onnx":          f"{_DIAR_BASE}/embedding_model.onnx",
}

# ── 必要文件清单 ───────────────────────────────────────────────────────
# 大型 .bin 附 SHA256；小型配置文件只检查存在即可。
REQUIRED_BIN: dict[str, str] = {
    "audio_encoder_model.bin":      "d892464d9b6986719dd6e5c3962b880a2708d874c2c9bdead8958581be2dacb9",
    "decoder_model.bin":            "cc4363c401f5faf41e2bfcb4aea80c72144b8ea66d13ca5ca62cf49421a25778",
    "thinker_embeddings_model.bin": "a7818fcbd77240fb8705bc47c2a15da98498056cdd419742b7685719b5dc2a44",
}
REQUIRED_OTHER: list[str] = [
    "audio_encoder_model.xml",
    "thinker_embeddings_model.xml",
    "decoder_model.xml",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def _get_paths(model_dir: Path) -> tuple[Path, Path]:
    """返回 (ov_dir, vad_path)。"""
    return model_dir / "qwen3_asr_int8", model_dir / "silero_vad_v4.onnx"


# ── Git LFS 指针文件检测 ────────────────────────────────────────────────
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

def _file_is_real(path: Path) -> bool:
    """返回 True 表示文件存在且不是 Git LFS pointer。

    当使用者以「git clone」取得 HuggingFace 模型仓库但未安装
    git-lfs 时，所有 LFS 追踪的文件（*.bin、*.onnx、*.npy 等）
    在磁碟上会是约 130 bytes 的 pointer 文本文件：
        version https://git-lfs.github.com/spec/v1
        oid sha256:<hash>
        size <bytes>
    Path.exists() 对 pointer 返回 True，导致下载器误以为
    文件已完整下载而跳过，最终模型无法加载。
    本函数以读取前 43 bytes 来识别并拒绝 LFS pointer。
    """
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(len(_LFS_MAGIC))
        return header != _LFS_MAGIC
    except OSError:
        return False


# ── ForcedAligner（chatllm .bin，单文件，无需 torch）────────────────────
_FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"
_FA_BIN_URL  = (
    "https://huggingface.co/dseditor/Collection/resolve/main/"
    "qwen3-focedaligner-0.6b.bin"
)


def quick_check_aligner(model_dir: Path) -> bool:
    """快速检查 chatllm ForcedAligner .bin 是否存在（非 LFS pointer）。"""
    return _file_is_real(Path(model_dir) / _FA_BIN_NAME)


def download_aligner(model_dir: Path, progress_cb=None):
    """下载 chatllm ForcedAligner .bin 至 model_dir（约 939 MB）。

    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下载失败时抛出例外。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / _FA_BIN_NAME
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, "时间轴对齐模型已存在")
        return

    if progress_cb:
        progress_cb(0.0, "下载时间轴对齐模型…")

    def _cb(done: int, total: int):
        if progress_cb and total > 0:
            progress_cb(
                done / total,
                f"下载时间轴对齐模型… {done/1_048_576:.0f} / {total/1_048_576:.0f} MB",
            )

    _download_file(_FA_BIN_URL, dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, "时间轴对齐模型下载完成！")


def quick_check_diarization(model_dir: Path) -> bool:
    """快速检查说话者分离模型是否存在且非 LFS pointer。"""
    diar_dir = model_dir / "diarization"
    return all(_file_is_real(diar_dir / fname) for fname in DIAR_FILES)


def download_diarization(diar_dir: Path, progress_cb=None):
    """
    下载说话者分离 ONNX 模型至 diar_dir。
    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下载失败时抛出例外。
    """
    diar_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = len(DIAR_FILES)

    for idx, (fname, url) in enumerate(DIAR_FILES.items()):
        dest = diar_dir / fname
        if _file_is_real(dest):
            if progress_cb:
                progress_cb((idx + 1) / total_tasks, f"✅ {fname}（已存在）")
            continue

        base_pct = idx / total_tasks
        span_pct = 1.0 / total_tasks
        if progress_cb:
            progress_cb(base_pct, f"下载 {fname}…")

        def _file_cb(done: int, total: int,
                     _b=base_pct, _s=span_pct, _f=fname):
            if progress_cb and total > 0:
                progress_cb(
                    _b + _s * done / total,
                    f"下载 {_f}…  {done/1_048_576:.1f} / {total/1_048_576:.1f} MB",
                )

        _download_file(url, dest, progress_cb=_file_cb)
        if progress_cb:
            progress_cb(base_pct + span_pct, f"✅ {fname}")

    if progress_cb:
        progress_cb(1.0, "说话者分离模型下载完成！")


def quick_check_1p7b(model_dir: Path) -> bool:
    """快速检查 1.7B KV-cache INT8 模型是否完整（非 LFS pointer）。"""
    kv_dir = model_dir / "qwen3_asr_1p7b_kv_int8"
    for fname in _1P7B_REQUIRED_BIN + _1P7B_REQUIRED_OTHER:
        if not _file_is_real(kv_dir / fname):
            return False
    return True


def download_1p7b(model_dir: Path, progress_cb=None):
    """
    从 HuggingFace 下载 1.7B KV-cache INT8 模型至 model_dir/qwen3_asr_1p7b_kv_int8/。
    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下载失败时抛出例外。
    """
    kv_dir = model_dir / "qwen3_asr_1p7b_kv_int8"
    kv_dir.mkdir(parents=True, exist_ok=True)

    all_files = _1P7B_REQUIRED_BIN + _1P7B_REQUIRED_OTHER
    tasks = [f for f in all_files if not _file_is_real(kv_dir / f)]

    if not tasks:
        if progress_cb:
            progress_cb(1.0, "所有 1.7B 文件已存在")
        return

    total = len(tasks)
    for idx, fname in enumerate(tasks):
        dest     = kv_dir / fname
        base_pct = idx / total
        span_pct = 1.0 / total

        if progress_cb:
            progress_cb(base_pct, f"下载 {fname}…")

        def _file_cb(done: int, total_b: int,
                     _b=base_pct, _s=span_pct, _f=fname):
            if progress_cb and total_b > 0:
                progress_cb(
                    _b + _s * done / total_b,
                    f"下载 {_f}…  {done/1_048_576:.1f} / {total_b/1_048_576:.1f} MB",
                )

        url = f"{_HF_1P7B_BASE}/{fname}"
        _download_file(url, dest, progress_cb=_file_cb)

        if progress_cb:
            progress_cb(base_pct + span_pct, f"✅ {fname}")

    if progress_cb:
        progress_cb(1.0, "1.7B 模型下载完成！")


# ══════════════════════════════════════════════════════════════════════
# CrispASR(Whisper) 核心 + Breeze-ASR-26 GGML 模型（按需下载，不进 EXE）
# ══════════════════════════════════════════════════════════════════════

# ── CrispASR 核心：多加速版本（Vulkan / CUDA / CPU）─────────────────────
# 不再固定版本：下载时向 GitHub API 查询最新 release（结果缓存于内存，
# 进程生命周期内只查一次），失败或离线时回退到 _CRISPASR_FALLBACK_VERSION。
# 资产命名跨版本一致（crispasr-windows-x86_64-<variant>.zip），故只需替换
# tag 段即可。GitHub 来源不套用 HF 镜像。
#
# Windows 官方只出 Vulkan / CUDA / CPU 三条路——**没有** HIP(ROCm) 也没有 SYCL，
# 所以 AMD 与 Intel 显卡的最佳解就是 Vulkan；只有 NVIDIA 才多一条 CUDA 可选。
# CUDA 包含 runtime DLL（cudart/cublas/cublasLt）故体积极大，是否值得换那点速度
# 由使用者自己在「推理加速版本」决定，我们只负责推荐。
_CRISPASR_FALLBACK_VERSION = "0.8.33"      # API 不可达时的保底版本
_CRISPASR_REPO_API = ("https://api.github.com/repos/CrispStrobe/CrispASR/"
                      "releases/latest")
_crispasr_ver_cache: str | None = None     # 进程级缓存；None＝尚未解析
_CRISPASR_API_TIMEOUT = 10                 # 秒；启动页查询不能久等


def crispasr_version() -> str:
    """当前应下载的 CrispASR 版本号（不带 v 前缀）。

    优先取 GitHub 最新 release 的 tag（v0.8.34 → 0.8.34）；网络失败／超时／
    响应异常时回退 _CRISPASR_FALLBACK_VERSION。结果缓存，进程内只查一次
    （GPU 页／自检页／下载会重复调用，不能每次都打 API）。
    """
    global _crispasr_ver_cache
    if _crispasr_ver_cache is not None:
        return _crispasr_ver_cache
    ver = _CRISPASR_FALLBACK_VERSION
    try:
        import json as _json
        req = urllib.request.Request(
            _CRISPASR_REPO_API, headers={"User-Agent": _UA, "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=_CRISPASR_API_TIMEOUT,
                                    context=_ssl_ctx()) as resp:
            tag = (_json.loads(resp.read().decode("utf-8")).get("tag_name") or "").strip()
        if tag.lower().startswith("v") and tag[1:].replace(".", "").isdigit():
            ver = tag[1:]
    except Exception:
        pass                                # 离线／限流 → 保底版本
    _crispasr_ver_cache = ver
    return ver


def _crispasr_rel_base() -> str:
    """当前版本的 release 资产基 URL（…/releases/download/vX.Y.Z）。"""
    return ("https://github.com/CrispStrobe/CrispASR/releases/"
            f"download/v{crispasr_version()}")


def __getattr__(name: str):
    """模块级 __getattr__：旧调用端 `from downloader import _CRISPASR_VERSION`
    改走动态版本解析（PEP 562）。保留旧名字以免逐处改动调用端。"""
    if name == "_CRISPASR_VERSION":
        return crispasr_version()
    raise AttributeError(name)

# variant → 中继数据。gpu_backend 直接喂给 crispasr.exe 的 `--gpu-backend`
# （None＝不给 GPU 后端、改挂 `-ng` 纯 CPU 推理）。
_CRISPASR_VARIANTS: dict[str, dict] = {
    "vulkan": {
        "zip":         "crispasr-windows-x86_64-vulkan.zip",
        "label":       "Vulkan（通用 GPU）",
        "short":       "Vulkan",
        "size_mb":     35,
        "gpu_backend": "vulkan",
        "desc":        "NVIDIA / AMD / Intel 通吃，体积最小。已长期实测稳定，"
                       "无独立显卡时亦可走核显。",
    },
    "cuda12": {
        "zip":         "crispasr-windows-x86_64-cuda.zip",
        "label":       "CUDA 12（NVIDIA 专用）",
        "short":       "CUDA 12",
        "size_mb":     692,
        "gpu_backend": "cuda",
        "desc":        "同 CUDA 13 但用较旧的 runtime，给 580 以下驱动的"
                       " NVIDIA 显卡；自带 CUDA 12 runtime。",
    },
    "cuda13": {
        "zip":         "crispasr-windows-x86_64-cuda13.zip",
        "label":       "CUDA 13（NVIDIA 新驱动）",
        "short":       "CUDA 13",
        "size_mb":     486,
        "gpu_backend": "cuda",
        "desc":        "NVIDIA 用户首选：原生 CUDA 加速，自带 CUDA 13 runtime"
                       "（无需另装 Toolkit）；需要 580 以上的 NVIDIA 驱动。",
    },
    "cpu": {
        "zip":         "crispasr-windows-x86_64-cpu.zip",
        "label":       "CPU（纯处理器）",
        "short":       "CPU",
        "size_mb":     8,
        "gpu_backend": None,
        "desc":        "完全不碰显卡。速度最慢，但在显卡驱动有问题时最不会出事。",
    },
    "cpu-legacy": {
        "zip":         "crispasr-windows-x86_64-cpu-legacy.zip",
        "label":       "CPU 兼容版（无 AVX2 的老 CPU）",
        "short":       "CPU 兼容版",
        "size_mb":     8,
        "gpu_backend": None,
        "desc":        "给 2013 年前、不支持 AVX2 的老处理器；一般 CPU 版开不起来"
                       "（闪退／非法指令）时才选这个。",
    },
}
_CRISPASR_DEFAULT_VARIANT = "vulkan"

# 安装纪录：记下这份 crispasr/ 目录「实际装了哪个版本／哪个加速版本」，
# 用来判断是否需要重新下载（升版或使用者换加速版本）。
_CRISPASR_MARKER = "crispasr_build.json"


def crispasr_variants() -> dict[str, dict]:
    """返回可选的加速版本表（唯读副本，供 UI 列菜单）。"""
    return {k: dict(v) for k, v in _CRISPASR_VARIANTS.items()}


def crispasr_gpu_backend(variant: str | None) -> str | None:
    """variant → crispasr.exe 的 `--gpu-backend` 值（None＝纯 CPU）。"""
    v = _CRISPASR_VARIANTS.get(variant or _CRISPASR_DEFAULT_VARIANT)
    return (v or _CRISPASR_VARIANTS[_CRISPASR_DEFAULT_VARIANT])["gpu_backend"]


# ── 硬件检测（零依赖，且必须在下载 crispasr 之前就能跑）─────────────────
#   注意：既有的 probe_crispasr_devices() 是跑 `crispasr.exe --diagnostics`
#   得到的，但我们得在「还没下载核心」时就决定要抓哪个 zip → 先有鸡先有蛋。
#   故这里改走 OS 自带的管道：Win32_VideoController(CIM) 认厂商、nvidia-smi
#   认驱动版本。两者都不需要预先安装任何东西。
_CREATE_NO_WINDOW = 0x08000000 if _sys.platform == "win32" else 0


def _run_hidden(cmd: list[str], timeout: int = 12) -> str:
    """执行外部指令并取回 stdout（隐藏控制台窗口）；失败回空字符串。"""
    import subprocess
    si = None
    if _sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=_CREATE_NO_WINDOW,
                           startupinfo=si)
        return r.stdout or ""
    except Exception:
        return ""


def _vendor_of(name: str, compat: str) -> str:
    """由显卡名称／厂商字符串判断 vendor 代号。"""
    s = f"{compat} {name}".lower()
    if "nvidia" in s:
        return "nvidia"
    if "amd" in s or "advanced micro" in s or "ati " in s or "radeon" in s:
        return "amd"
    if "intel" in s:
        return "intel"
    if "microsoft basic" in s or "remote" in s:
        return "virtual"          # 远程桌面／基本显卡，不算真 GPU
    return "other"


def detect_hardware() -> dict:
    """检测显卡与 NVIDIA 驱动版本（Windows）。

    返回：
      {"gpus": [{"vendor","name","vram_mb","driver"}...],   # 已排除虚拟显卡
       "vendors": ["nvidia","intel"],                        # 去重后的厂商清单
       "nvidia_driver": 581.15 | None,                       # 主版号.次版号
       "has_discrete": bool,
       "error": str | None}
    """
    info = {"gpus": [], "vendors": [], "nvidia_driver": None,
            "has_discrete": False, "error": None}
    if _sys.platform != "win32":
        info["error"] = "目前仅支持 Windows 的硬件检测"
        return info

    # ① 显卡清单（CIM/WMI）。AdapterRAM 对 4GB 以上的卡会溢位，仅供参考。
    ps = ("Get-CimInstance Win32_VideoController | "
          "Select-Object Name,AdapterCompatibility,DriverVersion,AdapterRAM | "
          "ConvertTo-Json -Compress")
    out = _run_hidden(["powershell", "-NoProfile", "-NonInteractive",
                       "-Command", ps])
    try:
        import json as _json
        data = _json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
        for d in data:
            name   = (d.get("Name") or "").strip()
            compat = (d.get("AdapterCompatibility") or "").strip()
            vendor = _vendor_of(name, compat)
            if vendor == "virtual" or not name:
                continue
            ram = d.get("AdapterRAM") or 0
            info["gpus"].append({
                "vendor": vendor, "name": name,
                "vram_mb": int(ram) // (1024 * 1024) if ram and ram > 0 else 0,
                "driver": (d.get("DriverVersion") or "").strip(),
            })
    except Exception as e:
        info["error"] = f"显卡列举失败：{e}"

    info["vendors"] = sorted({g["vendor"] for g in info["gpus"]})
    # Intel/AMD 的核显也会出现在清单里；「有独立显卡」以 NVIDIA 或 AMD 独显字样判断
    info["has_discrete"] = any(
        g["vendor"] == "nvidia" or
        (g["vendor"] == "amd" and "radeon" in g["name"].lower()
         and "graphics" not in g["name"].lower())
        for g in info["gpus"])

    # ② NVIDIA 驱动版本（决定 CUDA 12 还是 13）
    if "nvidia" in info["vendors"]:
        smi = _run_hidden(["nvidia-smi",
                           "--query-gpu=driver_version",
                           "--format=csv,noheader"])
        for line in smi.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(".")
                info["nvidia_driver"] = float(f"{parts[0]}.{parts[1]}"
                                              if len(parts) > 1 else parts[0])
            except Exception:
                pass
            break
    return info


# CUDA 13 在 Windows 需要 580 以上的 NVIDIA 驱动；再旧就只能走 CUDA 12。
_CUDA13_MIN_DRIVER = 580.0


def recommend_crispasr_variant(hw: dict | None = None) -> dict:
    """依硬件推荐加速版本，并返回整份可选清单（供 UI 直接渲染）。

    返回 {"recommended": "cuda13", "reason": "…", "hardware": {...},
          "options": [{"key","label","size_mb","desc","recommended":bool,
                       "suitable":bool,"note":str}, …]}

    推荐政策：
      • 有 NVIDIA   → 默认 CUDA（依驱动版本选 CUDA 13 / CUDA 12）：NVIDIA 用户的
                      原生加速路线，Flash Attention／GEMM 走原生核心，大模型
                      （Whisper Large／Turbo、Qwen3-1.7B）推理速度普遍优于 Vulkan。
      • 没有 NVIDIA → 一律 Vulkan（Windows 没有 ROCm/SYCL 版，Vulkan 就是最佳解）
    """
    hw = hw or detect_hardware()
    vendors = hw.get("vendors", [])
    drv     = hw.get("nvidia_driver")
    has_nv  = "nvidia" in vendors

    # NVIDIA 使用者可选的 CUDA 版本：驱动够新才给 13，否则 12。
    cuda_key = None
    if has_nv:
        cuda_key = "cuda13" if (drv is None or drv >= _CUDA13_MIN_DRIVER) else "cuda12"

    # 推荐：NVIDIA → CUDA；其余 → Vulkan
    recommended = cuda_key if has_nv and cuda_key else _CRISPASR_DEFAULT_VARIANT
    if not hw.get("gpus"):
        reason = ("未检测到显卡信息，先以 Vulkan 版为默认——它在没有独立显卡时"
                  "会自动退回 CPU 推理，最不会出事。")
    elif has_nv:
        names = "、".join(g["name"] for g in hw["gpus"] if g["vendor"] == "nvidia")
        reason = (f"检测到 NVIDIA 显卡（{names}）。已为你推荐原生 CUDA 加速"
                  f"（{_CRISPASR_VARIANTS[cuda_key]['label']}，自带 CUDA runtime，"
                  f"无需另装 CUDA Toolkit）；若下载量大或驱动兼容有顾虑，可改选 "
                  f"Vulkan（仅 35 MB）。")
    elif "amd" in vendors or "intel" in vendors:
        names = "、".join(g["name"] for g in hw["gpus"]
                          if g["vendor"] in ("amd", "intel"))
        reason = (f"检测到 {'AMD' if 'amd' in vendors else 'Intel'} 显卡"
                  f"（{names}）。CrispASR 的 Windows 版没有 ROCm／SYCL，"
                  f"Vulkan 就是这张卡的最佳解。")
    else:
        reason = "未检测到可加速的显卡，使用 Vulkan 版（会自动退回 CPU 推理）。"

    options = []
    for key, meta in _CRISPASR_VARIANTS.items():
        suitable, note = True, ""
        if key in ("cuda12", "cuda13"):
            if not has_nv:
                suitable, note = False, "需要 NVIDIA 显卡"
            elif key == "cuda13" and drv is not None and drv < _CUDA13_MIN_DRIVER:
                suitable, note = False, (f"需要 {_CUDA13_MIN_DRIVER:.0f} 以上的驱动"
                                         f"（目前 {drv:.2f}）")
            elif key != cuda_key:
                note = "驱动版本建议改用另一个 CUDA 版本"
        elif key == "cpu-legacy":
            note = "只有在一般 CPU 版闪退时才需要"
        elif key == "vulkan" and has_nv:
            note = "体积最小的通用备选；NVIDIA 用户建议优先 CUDA"
        options.append({"key": key, "label": meta["label"],
                        "size_mb": meta["size_mb"], "desc": meta["desc"],
                        "recommended": key == recommended,
                        "suitable": suitable, "note": note})

    return {"recommended": recommended, "reason": reason,
            "hardware": hw, "options": options}

# OpenAI Whisper 官方模型 GGML（ggerganov/whisper.cpp，HuggingFace 官方镜像）：
# CrispASR 的 whisper 后端（whisper.cpp 分支）原生兼容这批单文件 GGML。
# 尺寸 → (repo, 文件名, 约大小 MB)。turbo = large-v3-turbo。
_WHISPER_SIZES: dict[str, tuple[str, str, int]] = {
    "base":   ("ggerganov/whisper.cpp", "ggml-base.bin",        148),
    "small":  ("ggerganov/whisper.cpp", "ggml-small.bin",       488),
    "medium": ("ggerganov/whisper.cpp", "ggml-medium.bin",      1530),
    "large":  ("ggerganov/whisper.cpp", "ggml-large-v2.bin",    3110),
    "turbo":  ("ggerganov/whisper.cpp", "ggml-large-v3-turbo.bin", 1620),
}
_WHISPER_DEFAULT_SIZE = "base"


def whisper_ggml_filename(size: str) -> str:
    """尺寸代码（base/small/medium/large/turbo）→ Whisper GGML 文件名。"""
    return _WHISPER_SIZES.get(size, _WHISPER_SIZES[_WHISPER_DEFAULT_SIZE])[1]


def quick_check_whisper_ggml(crispasr_dir: Path,
                             size: str = _WHISPER_DEFAULT_SIZE) -> bool:
    """指定尺寸的 OpenAI Whisper GGML 是否存在（排除 LFS pointer）。"""
    fname = whisper_ggml_filename(size)
    d = Path(crispasr_dir)
    return _file_is_real(d / fname) or any(_file_is_real(p) for p in d.glob(f"**/{fname}"))


def download_whisper_ggml(crispasr_dir: Path,
                          size: str = _WHISPER_DEFAULT_SIZE, progress_cb=None):
    """下载 OpenAI Whisper GGML 至 crispasr_dir（与 crispasr.exe 同层）。

    来源为 ggerganov/whisper.cpp（HuggingFace）。GitHub 来源不套用 HF 镜像，
    但 whisper.cpp 的模型托管在 HF 上 → 支持镜像回退（_download_file_with_fallback）。
    """
    repo, fname, _mb = _WHISPER_SIZES.get(size, _WHISPER_SIZES[_WHISPER_DEFAULT_SIZE])
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    dest = Path(crispasr_dir) / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"Whisper {size} 已存在")
        return
    Path(crispasr_dir).mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.0, f"下载 OpenAI Whisper（{size}）…")

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(done / total_b,
                        f"下载 Whisper {size}…  {done/1_048_576:.1f} / {total_b/1_048_576:.1f} MB")

    _download_file(url, dest, progress_cb=_cb)
    if not _file_is_real(dest):
        raise RuntimeError(f"Whisper {size} 下载后校验失败：{dest}")
    if progress_cb:
        progress_cb(1.0, f"OpenAI Whisper（{size}）就绪")


def installed_crispasr_info(crispasr_dir: Path) -> dict:
    """读取 crispasr/ 的安装纪录；没有纪录档时返回 version=None。

    0.8.8 以前的旧安装没有纪录档 → 回 {"version": None, "variant": None}，
    调用端据此判定「需要升级」。
    """
    import json as _json
    f = Path(crispasr_dir) / _CRISPASR_MARKER
    try:
        d = _json.loads(f.read_text(encoding="utf-8"))
        return {"version": d.get("version"), "variant": d.get("variant")}
    except Exception:
        return {"version": None, "variant": None}


def _write_crispasr_marker(crispasr_dir: Path, variant: str):
    import json as _json
    try:
        (Path(crispasr_dir) / _CRISPASR_MARKER).write_text(
            _json.dumps({"version": crispasr_version(), "variant": variant},
                        ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def quick_check_crispasr(crispasr_dir: Path, variant: str | None = None) -> bool:
    """CrispASR 核心是否就绪。

    variant 为 None（旧调用端）：只看 crispasr.exe 在不在，维持原行为。
    variant 有值：另外要求安装纪录的版本与加速版本都吻合——版本升级或使用者
    改选加速版本时就会回 False，触发重新下载。
    """
    crispasr_dir = Path(crispasr_dir)
    exists = ((crispasr_dir / "crispasr.exe").exists()
              or any(crispasr_dir.glob("**/crispasr.exe")))
    if not exists or variant is None:
        return exists
    info = installed_crispasr_info(crispasr_dir)
    return info["version"] == crispasr_version() and info["variant"] == variant


def download_crispasr_core(crispasr_dir: Path, progress_cb=None,
                           variant: str | None = None):
    """下载指定加速版本的 CrispASR zip 并解压（扁平化）至 crispasr_dir。

    progress_cb(pct: float, msg: str)。GitHub 来源不套用 HF 镜像。
    variant 省略时用默认的 Vulkan 版（维持旧调用端行为）。

    换版本／升版时会**先清掉旧的 .exe/.dll**：不同加速版本（例如 Vulkan 版的
    ggml-vulkan.dll）与不同 CrispASR 版本的 DLL 混在同一个文件夹会载到不兼容
    的 ggml，症状是启动即闪退。模型文件（.bin/.gguf）不动，换版本不必重新下载模型。
    """
    import shutil
    import tempfile
    import zipfile

    variant = variant if variant in _CRISPASR_VARIANTS else _CRISPASR_DEFAULT_VARIANT
    meta    = _CRISPASR_VARIANTS[variant]
    version = crispasr_version()           # 下载时才解析最新版本（含缓存）
    url     = f"{_crispasr_rel_base()}/{meta['zip']}"

    crispasr_dir = Path(crispasr_dir)
    crispasr_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.0, f"下载 CrispASR 核心（{meta['label']}，"
                         f"约 {meta['size_mb']} MB）…")

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                0.9 * done / total_b,
                f"下载核心…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / meta["zip"]
        _download_file(url, zpath, progress_cb=_cb)
        if progress_cb:
            progress_cb(0.92, "解压 CrispASR 核心…")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)

        # 旧的可执行文件／DLL 全清（模型文件保留）
        try:
            (crispasr_dir / _CRISPASR_MARKER).unlink(missing_ok=True)
            for old in crispasr_dir.iterdir():
                if old.is_file() and old.suffix.lower() in (".exe", ".dll"):
                    old.unlink(missing_ok=True)
        except Exception:
            pass                      # 文件被占用时就让 copy 覆盖，不中断流程

        # 扁平化：所有 .exe / .dll 移到 crispasr_dir 根目录
        for f in Path(td).glob("**/*"):
            if f.is_file() and f.suffix.lower() in (".exe", ".dll"):
                shutil.copy(f, crispasr_dir / f.name)

    if not quick_check_crispasr(crispasr_dir):
        raise RuntimeError("CrispASR 核心解压后仍找不到 crispasr.exe")
    _write_crispasr_marker(crispasr_dir, variant)
    if progress_cb:
        progress_cb(1.0, f"CrispASR {version}（{meta['label']}）就绪")


# ── ffmpeg（按需下载，不随安装包附带）──────────────────────────────────
#   桌面 CTk 版走 BtbN essentials（ffmpeg_utils.FFmpegDownloadDialog）；webview /
#   EXE 版改抓我们自备的精简 zip（只含 ffmpeg.exe[/ffprobe.exe]），体积小、来源稳定。
_FFMPEG_ZIP_URL = "https://huggingface.co/dseditor/Collection/resolve/main/ffmpeg.zip"


def quick_check_ffmpeg(dest_dir: Path) -> bool:
    """ffmpeg.exe 是否已就绪（dest_dir 根目录或子文件夹）。"""
    dest_dir = Path(dest_dir)
    if (dest_dir / "ffmpeg.exe").exists():
        return True
    return any(dest_dir.glob("**/ffmpeg.exe"))


def download_ffmpeg(dest_dir: Path, progress_cb=None):
    """下载 ffmpeg.zip（HF）并解压 ffmpeg.exe/ffprobe.exe 至 dest_dir（扁平化）。

    progress_cb(pct: float, msg: str)。仿 download_crispasr_core 的流程。
    """
    import shutil
    import tempfile
    import zipfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.0, "下载 ffmpeg（视频音轨提取）…")

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                0.9 * done / total_b,
                f"下载 ffmpeg…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "ffmpeg.zip"
        _download_file(_FFMPEG_ZIP_URL, zpath, progress_cb=_cb)
        if progress_cb:
            progress_cb(0.92, "解压 ffmpeg…")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)
        # 扁平化：把 ffmpeg.exe / ffprobe.exe 移到 dest_dir 根目录（忽略 zip 内层级）
        for f in Path(td).glob("**/*"):
            if f.is_file() and f.name.lower() in ("ffmpeg.exe", "ffprobe.exe"):
                shutil.copy(f, dest_dir / f.name)

    if not quick_check_ffmpeg(dest_dir):
        raise RuntimeError("ffmpeg 解压后仍找不到 ffmpeg.exe")
    if progress_cb:
        progress_cb(1.0, "ffmpeg 就绪")



# ── Faster-Whisper-XXL 引擎（Purfview，独立下载的第三条 Whisper 路径）────
# faster-whisper-xxl.exe：CTranslate2 (faster-whisper) 的 standalone 打包，
# 自带全部依赖与 ffmpeg，比 whisper.cpp 系更快且自带 VAD/对齐。不进安装包
# —— 1.3GB 7z 按需下载，解压后放 <model_dir>/Faster-Whisper-XXL/。
# 压缩格式是 BCJ2 滤镜的 .7z：py7zr 不支持，须用官方独立解压器 7zr.exe
# （约 0.6MB，仅处理 .7z），缺则自动下载到 tools/。
_FWXXL_URL = ("https://github.com/Purfview/whisper-standalone-win/releases/"
              "download/Faster-Whisper-XXL/Faster-Whisper-XXL_r245.4_windows.7z")
_FWXXL_VERSION = "r245.4"
_FWXXL_DIRNAME = "Faster-Whisper-XXL"          # 解压后的目录名（含 exe）
_FWXXL_7ZR_URL = "https://www.7-zip.org/a/7zr.exe"
_FWXXL_SIZE_MB = 1358


def fwxxl_dir(model_dir: Path) -> Path:
    """Faster-Whisper-XXL 引擎目录：<model_dir>/Faster-Whisper-XXL/。"""
    return Path(model_dir) / _FWXXL_DIRNAME


def quick_check_fwxxl(model_dir: Path) -> bool:
    """faster-whisper-xxl.exe 是否已就绪（目录根或子层，排除空壳）。"""
    d = fwxxl_dir(model_dir)
    return (d / "faster-whisper-xxl.exe").is_file() \
        or any(d.glob("**/faster-whisper-xxl.exe"))


# UI 尺寸代码 → XXL 模型缓存目录名（_models/ 下）。XXL 从 HuggingFace 取
# Systran/faster-whisper-<size>；turbo 用 Purview 官方仓库 large-v3-turbo。
# 与 webview_backend._FWWHISPER_MODEL_ARG 的 --model 参数保持同源语义。
_FWWHISPER_CACHE_NAMES = {
    "base":   "faster-whisper-base",
    "small":  "faster-whisper-small",
    "medium": "faster-whisper-medium",
    "large":  "faster-whisper-large-v2",
    "turbo":  "faster-whisper-large-v3-turbo",
}


def fwxxl_model_present(model_dir: Path, size: str) -> bool:
    """指定尺寸的 Whisper 模型是否已缓存（_models/<名字>/model.bin 存在）。"""
    name = _FWWHISPER_CACHE_NAMES.get(size)
    if not name:
        return False
    d = fwxxl_dir(model_dir) / "_models" / name
    return (d / "model.bin").is_file()


def fwxxl_models_status(model_dir: Path) -> dict[str, bool]:
    """五个尺寸的模型缓存状态（自检面板逐项显示用）。"""
    return {size: fwxxl_model_present(model_dir, size)
            for size in _FWWHISPER_CACHE_NAMES}


def fwxxl_model_present_dir(engine_dir: Path, size: str) -> bool:
    """同 fwxxl_model_present，但直接给定引擎目录（嵌套布局下与 load() 的
    exe 解析结果保持一致，供自检面板对齐实际引擎位置）。"""
    name = _FWWHISPER_CACHE_NAMES.get(size)
    if not name:
        return False
    return ((Path(engine_dir) / "_models" / name / "model.bin").is_file())


def _ensure_7zr(tools_dir: Path, progress_cb=None) -> Path:
    """确保 7zr.exe 存在（缺则自 7-zip.org 下载）；返回其路径。"""
    exe = Path(tools_dir) / "7zr.exe"
    if exe.is_file() and exe.stat().st_size > 100_000:
        return exe
    exe.parent.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.0, "下载 7z 解压工具…")
    _download_file(_FWXXL_7ZR_URL, exe, progress_cb=None)
    return exe


def download_fwxxl(model_dir: Path, progress_cb=None):
    """下载 Faster-Whisper-XXL r245.4（1.3GB 7z）并解压至 <model_dir>/。

    progress_cb(pct: float, msg: str)。GitHub 来源不套用 HF 镜像。
    解压用 7zr.exe（BCJ2 滤镜 py7zr 不支持）；解压目标为 <model_dir> 根、
    归档内自带 Faster-Whisper-XXL/ 顶层目录。下载的 .7z 落在 model_dir
    （约 1.3GB，解压完删除以省空间；断点续传友好——中断后重跑继续）。
    """
    import subprocess as _sp
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    zpath = model_dir / f"Faster-Whisper-XXL_{_FWXXL_VERSION}_windows.7z"

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                0.9 * done / total_b,
                f"下载 Faster-Whisper-XXL…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    if not (zpath.is_file() and zpath.stat().st_size >= _FWXXL_SIZE_MB * 1024 * 1024 * 0.98):
        if progress_cb:
            progress_cb(0.0, f"下载 Faster-Whisper-XXL {_FWXXL_VERSION}（约 {_FWXXL_SIZE_MB} MB）…")
        _download_file(_FWXXL_URL, zpath, progress_cb=_cb)

    if progress_cb:
        progress_cb(0.92, "解压 Faster-Whisper-XXL（约 3.5 GB，需数分钟）…")
    sevenzr = _ensure_7zr(BASE_DIR_TOOLS(), progress_cb=progress_cb)
    dest = model_dir
    dest.mkdir(parents=True, exist_ok=True)
    proc = _sp.run([str(sevenzr), "x", "-y", f"-o{dest}", str(zpath)],
                   capture_output=True, creationflags=_CREATE_NO_WINDOW)
    if proc.returncode != 0 or not quick_check_fwxxl(dest):
        # 坏包/坏解压器不保留：尺寸门槛拦不住「截断到 98.5%」的残档，留着只会
        # 让每次重试都跳过下载、在同一处失败。删除后下次自动重新下载。
        for bad in (zpath, sevenzr):
            try:
                bad.unlink(missing_ok=True)
            except OSError:
                pass
        err = (proc.stderr or b"").decode(errors="replace").strip().splitlines()
        raise RuntimeError("Faster-Whisper-XXL 解压失败（已清理损坏文件，重试将重新下载）："
                           + (err[-1] if err else f"7zr 返回码 {proc.returncode}"))
    try:
        zpath.unlink(missing_ok=True)       # 解压成功即删 7z 省空间
    except Exception:
        pass
    if progress_cb:
        progress_cb(1.0, f"Faster-Whisper-XXL {_FWXXL_VERSION} 就绪")


def BASE_DIR_TOOLS() -> Path:
    """7zr.exe 的存放目录：frozen 时 EXE 旁 tools/，开发时项目 tools/。"""
    if getattr(_sys, "frozen", False):
        return Path(_sys.executable).parent / "tools"
    return Path(__file__).parent / "tools"


# ── qwen3 ForcedAligner GGUF（CrispASR -am 对齐器，Whisper 核心专用 FA）──
# crispasr.exe -am <gguf> -falign：用 CTC 对齐器的字级时间轴覆盖 whisper 自带
# 的（较粗）时间戳。同作者(cstr)上传，与 crispasr 的 -am 界面兼容。
_ALIGNER_GGUF_REPO  = "cstr/qwen3-forced-aligner-0.6b-GGUF"
_ALIGNER_GGUF_BASE  = f"https://huggingface.co/{_ALIGNER_GGUF_REPO}/resolve/main"
_ALIGNER_GGUF_FILES = {
    "q4": "qwen3-forced-aligner-0.6b-q4_k.gguf",   # 轻量 ~529 MB
    "q5": "qwen3-forced-aligner-0.6b-q5_0.gguf",   # 标准 ~643 MB
    "q8": "qwen3-forced-aligner-0.6b-q8_0.gguf",   # 精确 ~986 MB
}
# 对齐（CTC）对量化不敏感，标准 q5 即足够精确，默认用之。
_ALIGNER_GGUF_DEFAULT = "q5"


def aligner_gguf_filename(quant: str = _ALIGNER_GGUF_DEFAULT) -> str:
    """量化代码（q4/q5/q8）→ qwen3 ForcedAligner GGUF 文件名（未知时返回 q5）。"""
    return _ALIGNER_GGUF_FILES.get(quant, _ALIGNER_GGUF_FILES[_ALIGNER_GGUF_DEFAULT])


def quick_check_aligner_gguf(crispasr_dir: Path,
                             quant: str = _ALIGNER_GGUF_DEFAULT) -> bool:
    """指定量化的 ForcedAligner GGUF 是否存在（排除 LFS pointer）。"""
    return _file_is_real(crispasr_dir / aligner_gguf_filename(quant))


def download_aligner_gguf(crispasr_dir: Path,
                          quant: str = _ALIGNER_GGUF_DEFAULT, progress_cb=None):
    """下载 qwen3 ForcedAligner GGUF 至 crispasr_dir（与 crispasr.exe 同层）。

    progress_cb(pct: float, msg: str)。支持断点续传与 HF 镜像。
    """
    crispasr_dir.mkdir(parents=True, exist_ok=True)
    fname = aligner_gguf_filename(quant)
    dest  = crispasr_dir / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"{fname} 已存在")
        return

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                done / total_b,
                f"下载 {fname}…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    _download_file(f"{_ALIGNER_GGUF_BASE}/{fname}", dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, f"✅ {fname}")


# ── Qwen3-ASR-1.7B GGUF（CrispASR -casr 后端，Qwen 核心跑在 crispasr.exe）──
# crispasr.exe --backend qwen3-1.7b：把 Qwen3-ASR 1.7B 当作 CrispASR 的后端，
# 走 Vulkan（Intel/AMD/NVIDIA 通吃），繁体断句质量佳（见 crispasr-nemotron-eval）。
# 同作者(cstr)上传的 GGUF；crisp_engine._infer_backend 依文件名含 "1.7b" 自动推断。
_QWEN3_ASR_GGUF_REPO = "cstr/qwen3-asr-1.7b-GGUF"
_QWEN3_ASR_GGUF_BASE = f"https://huggingface.co/{_QWEN3_ASR_GGUF_REPO}/resolve/main"
_QWEN3_ASR_GGUF_FILES = {
    "q4": "qwen3-asr-1.7b-q4_k.gguf",   # 轻量
    "q8": "qwen3-asr-1.7b-q8_0.gguf",   # 精确
}
_QWEN3_ASR_GGUF_DEFAULT = "q8"


def qwen3_asr_gguf_filename(quant: str = _QWEN3_ASR_GGUF_DEFAULT) -> str:
    """量化代码（q4/q8）→ Qwen3-ASR-1.7B GGUF 文件名（未知时返回 q8 精确）。"""
    return _QWEN3_ASR_GGUF_FILES.get(quant, _QWEN3_ASR_GGUF_FILES[_QWEN3_ASR_GGUF_DEFAULT])


def quick_check_qwen3_asr_gguf(model_dir: Path,
                               quant: str = _QWEN3_ASR_GGUF_DEFAULT) -> bool:
    """指定量化的 Qwen3-ASR-1.7B GGUF 是否存在（排除 LFS pointer）。"""
    return _file_is_real(Path(model_dir) / qwen3_asr_gguf_filename(quant))


def download_qwen3_asr_gguf(model_dir: Path,
                            quant: str = _QWEN3_ASR_GGUF_DEFAULT, progress_cb=None):
    """下载指定量化的 Qwen3-ASR-1.7B GGUF 至 model_dir（通常为 ov_models/）。

    progress_cb(pct: float, msg: str)。支持断点续传与 HF 镜像。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fname = qwen3_asr_gguf_filename(quant)
    dest  = model_dir / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"{fname} 已存在")
        return

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                done / total_b,
                f"下载 {fname}…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    _download_file(f"{_QWEN3_ASR_GGUF_BASE}/{fname}", dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, f"✅ {fname}")


# ── Qwen3-ASR-1.7B 日本动漫特化 GGUF（cstr，日语识别增强）──────────────────
# 与标准 qwen3-asr-1.7b 同架构（crisp_engine._infer_backend 依文件名含 "1.7b" 一样
# 推断 --backend qwen3-1.7b），差别只在权重针对日语／动漫语音微调 → 日文歌词、
# 台词识别明显较佳。同置 ov_models/；缺文件按需下载。只有 q4/q8 两量化。
_QWEN3_ASR_JA_GGUF_REPO = "cstr/qwen3-asr-1.7b-ja-anime-GGUF"
_QWEN3_ASR_JA_GGUF_BASE = f"https://huggingface.co/{_QWEN3_ASR_JA_GGUF_REPO}/resolve/main"
_QWEN3_ASR_JA_GGUF_FILES = {
    "q4": "qwen3-asr-1.7b-ja-anime-q4_k.gguf",   # 轻量 ~1.33 GB
    "q8": "qwen3-asr-1.7b-ja-anime-q8_0.gguf",   # 精确 ~2.51 GB
}
_QWEN3_ASR_JA_GGUF_DEFAULT = "q8"


def qwen3_asr_ja_gguf_filename(quant: str = _QWEN3_ASR_JA_GGUF_DEFAULT) -> str:
    """量化代码（q4/q8）→ Qwen3-ASR-1.7B ja-anime GGUF 文件名（未知时返回 q8）。"""
    return _QWEN3_ASR_JA_GGUF_FILES.get(quant, _QWEN3_ASR_JA_GGUF_FILES[_QWEN3_ASR_JA_GGUF_DEFAULT])


def quick_check_qwen3_asr_ja_gguf(model_dir: Path,
                                  quant: str = _QWEN3_ASR_JA_GGUF_DEFAULT) -> bool:
    """指定量化的 ja-anime GGUF 是否存在（排除 LFS pointer）。"""
    return _file_is_real(Path(model_dir) / qwen3_asr_ja_gguf_filename(quant))


def download_qwen3_asr_ja_gguf(model_dir: Path,
                               quant: str = _QWEN3_ASR_JA_GGUF_DEFAULT, progress_cb=None):
    """下载指定量化的 Qwen3-ASR-1.7B ja-anime GGUF 至 model_dir（通常为 ov_models/）。

    progress_cb(pct: float, msg: str)。支持断点续传与 HF 镜像。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fname = qwen3_asr_ja_gguf_filename(quant)
    dest  = model_dir / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"{fname} 已存在")
        return

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                done / total_b,
                f"下载 {fname}…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    _download_file(f"{_QWEN3_ASR_JA_GGUF_BASE}/{fname}", dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, f"✅ {fname}")


# ══════════════════════════════════════════════════════════════════════
# 完整性检查
# ══════════════════════════════════════════════════════════════════════

def _sha256(path: Path, progress_cb=None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done  = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
            done += len(buf)
            if progress_cb:
                progress_cb(done, total)
    return h.hexdigest()


def quick_check(model_dir: Path) -> bool:
    """快速存在性检查（排除 Git LFS pointer，不计算哈希）。"""
    ov_dir, vad_path = _get_paths(model_dir)
    if not _file_is_real(vad_path):
        return False
    for fname in list(REQUIRED_BIN) + REQUIRED_OTHER:
        if not _file_is_real(ov_dir / fname):
            return False
    return True


def full_verify(model_dir: Path, progress_cb=None) -> tuple[bool, str]:
    """存在 + SHA256 完整验证。"""
    ov_dir, vad_path = _get_paths(model_dir)
    if not _file_is_real(vad_path):
        return False, f"遗失：{vad_path.name}"
    for fname in list(REQUIRED_BIN) + REQUIRED_OTHER:
        if not _file_is_real(ov_dir / fname):
            return False, f"遗失：{fname}"

    total_files = len(REQUIRED_BIN)
    for i, (fname, expected) in enumerate(REQUIRED_BIN.items()):
        fpath = ov_dir / fname
        if progress_cb:
            progress_cb(i / total_files * 0.9, f"验证 {fname}…")

        def _inner(done, total, _i=i, _f=fname):
            if progress_cb:
                progress_cb((_i + done / total) / total_files * 0.9, f"验证 {_f}…")

        actual = _sha256(fpath, _inner)
        if actual != expected:
            return False, f"{fname} 哈希不符（文件可能损坏）"

    if progress_cb:
        progress_cb(1.0, "✅ 所有模型完整")
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════
# 直接 HTTP 下载（断点续传）
# ══════════════════════════════════════════════════════════════════════

def _download_file(url: str, dest: Path, progress_cb=None):
    """
    下载单一文件至 dest，支持断点续传（Resume）。
    progress_cb(done_bytes: int, total_bytes: int)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    url = _apply_mirror(url)   # 套用镜像站改写（若有设置）
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")

    try:
        resp = urllib.request.urlopen(req, timeout=30, context=_ssl_ctx())
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # 416 Range Not Satisfiable = 文件已完整，直接视为成功
            return
        raise

    content_length = int(resp.headers.get("Content-Length", 0))
    total = existing + content_length if content_length else 0

    # 追加写入（resume）或全新写入
    mode = "ab" if existing > 0 and resp.status == 206 else "wb"
    if mode == "wb":
        existing = 0

    done = existing
    try:
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1 << 16)   # 64 KB
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(done, total)
    finally:
        resp.close()


def _download_file_with_fallback(
    fname: str,
    dest: Path,
    progress_cb=None,
):
    """
    先尝试主要 HF 来源，若连接失败则自动切换至备用来源。
    VAD 等非 HF 文件（url 已为完整 URL）直接下载，不套用备援。
    """
    primary_url  = f"{_HF_BASE_PRIMARY}/{fname}"
    fallback_url = f"{_HF_BASE_FALLBACK}/{fname}"

    try:
        _download_file(primary_url, dest, progress_cb)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as primary_err:
        # 主要来源失败，切换备用
        print(f"\n⚠ 主要来源失败（{primary_err}），切换至备用来源…")
        _download_file(fallback_url, dest, progress_cb)


# ══════════════════════════════════════════════════════════════════════
# 批量下载所有模型
# ══════════════════════════════════════════════════════════════════════

def download_all(model_dir: Path, progress_cb=None):
    """
    下载所有缺少的模型至 model_dir。
    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下载失败时抛出例外。
    """
    ov_dir, vad_path = _get_paths(model_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)

    # 建立下载任务清单 (dest, hf_fname_or_direct_url, is_direct_url)
    # _file_is_real() 同时排除「不存在」与「Git LFS pointer」两种情况
    tasks: list[tuple[Path, str, bool]] = []
    for fname in list(REQUIRED_BIN.keys()) + REQUIRED_OTHER:
        dest = ov_dir / fname
        # 小型配置文件若已存在则跳过；大型 .bin 若存在也先跳过（full_verify 再补）
        # 使用 _file_is_real() 避免将 Git LFS 指标档误判为有效文件
        if not _file_is_real(dest):
            tasks.append((dest, fname, False))   # HF 相对路径，使用备援机制
    if not _file_is_real(vad_path):
        tasks.append((vad_path, _VAD_URL, True))  # 直接 URL，不需备援

    if not tasks:
        if progress_cb:
            progress_cb(1.0, "所有文件已存在")
        return

    total_tasks = len(tasks)
    for idx, (dest, fname_or_url, is_direct) in enumerate(tasks):
        fname = dest.name
        base_pct = idx / total_tasks
        span_pct = 1.0 / total_tasks

        if progress_cb:
            progress_cb(base_pct, f"下载 {fname}…")

        def _file_cb(done: int, total: int,
                     _b=base_pct, _s=span_pct, _f=fname):
            if progress_cb and total > 0:
                progress_cb(
                    _b + _s * done / total,
                    f"下载 {_f}…  {done/1_048_576:.1f} / {total/1_048_576:.1f} MB",
                )

        if is_direct:
            _download_file(fname_or_url, dest, progress_cb=_file_cb)
        else:
            _download_file_with_fallback(fname_or_url, dest, progress_cb=_file_cb)

        if progress_cb:
            progress_cb(base_pct + span_pct, f"✅ {fname}")

    if progress_cb:
        progress_cb(1.0, "下载完成！")


# ══════════════════════════════════════════════════════════════════════
# 命令行界面
# ══════════════════════════════════════════════════════════════════════

def _cli_bar(pct: float, msg: str):
    filled = int(pct * 40)
    bar    = "█" * filled + "░" * (40 - filled)
    print(f"\r[{bar}] {pct*100:5.1f}%  {msg:<45}", end="", flush=True)


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    model_dir  = _DEFAULT_MODEL_DIR

    print("=== Qwen3-ASR 模型完整性检查 ===\n")
    print(f"模型路径：{model_dir}\n")

    if quick_check(model_dir):
        print("所有文件存在，正在验证哈希…")
        ok, msg = full_verify(model_dir, progress_cb=_cli_bar)
        print()
        if ok:
            print("✅ 模型完整，无需下载")
        else:
            print(f"❌ {msg}")
            if not check_only:
                print("正在重新下载损坏的文件…")
                download_all(model_dir, _cli_bar)
                print("\n✅ 完成")
    else:
        print("模型不完整或尚未下载")
        if check_only:
            sys.exit(1)
        print(f"从 HuggingFace 下载（约 1.2 GB）：{_HF_REPO_PRIMARY}（备用：{_HF_REPO_FALLBACK}）")
        print("首次下载视网络速度可能需要 5–30 分钟\n")
        download_all(model_dir, _cli_bar)
        print("\n✅ 下载完成")

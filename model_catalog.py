"""model_catalog.py — 模型目录元数据（webview_backend 拆分，2026-11）

引擎 → 模型 → (backend, settings patch) 的中央目录，以及模型标签、
下载分派等纯元数据。与 GUI/CLI 共享（cli_mode --profile 同源）。

从 webview_backend 原样搬移；webview_backend 仍兼容再导出全部符号。
"""
from __future__ import annotations

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

# 识别语言：crispasr 使用的常用语言清单（OpenVINO 改用 processor 的）
_COMMON_LANGS = [
    "Chinese", "English", "Japanese", "Korean", "Cantonese", "French", "German",
    "Spanish", "Portuguese", "Russian", "Arabic", "Thai", "Vietnamese",
    "Indonesian", "Malay",
]


def _sanitize_tag(tag: str) -> str:
    """引擎/模型标签 → 文件名安全片段（供输出字幕文件名拼接）。

    去除 Windows 文件名非法字符与路径分隔符、压掉多余空白；空串回 "model"。
    """
    for ch in '<>:"/\\\\|?*':
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

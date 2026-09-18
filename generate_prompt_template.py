"""
一次性工具：生成 prompt_template.json
需要在 venv（含 torch + transformers + qwen_asr）中执行一次。
输出的 JSON 让 processor_numpy.py 不需任何 torch/transformers 即可运作。

用法：
    python generate_prompt_template.py
"""
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
OV_DIR   = BASE_DIR / "ov_models" / "qwen3_asr_int8"
OUT_PATH = BASE_DIR / "prompt_template.json"

print("加载 qwen_asr processor…")
import qwen_asr  # noqa
from qwen_asr.inference.qwen3_asr import (
    Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor,
    SUPPORTED_LANGUAGES,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr",   Qwen3ASRConfig,                   exist_ok=True)
AutoModel.register( Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor,            exist_ok=True)

processor = AutoProcessor.from_pretrained(str(OV_DIR), fix_mistral_regex=True)

# ── 建立 prompt（与 app.py transcribe() 完全一致）──────────────────
msgs = [
    {"role": "system", "content": ""},
    {"role": "user",   "content": [{"type": "audio", "audio": ""}]},
]
prompt_text = processor.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=False
)
print(f"Prompt text: {repr(prompt_text)}")

# ── 用静音音频跑一次，取得完整 input_ids ──────────────────────────
dummy_audio = np.zeros(480000, dtype=np.float32)
inp = processor(text=[prompt_text], audio=[dummy_audio], return_tensors="np", padding=True)
ids = inp["input_ids"][0].tolist()

AUDIO_PAD_ID = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
print(f"audio_pad_id  = {AUDIO_PAD_ID}")
print(f"Total tokens  = {len(ids)}")

# 找出音频 pad 区段的位置
pad_positions = [i for i, x in enumerate(ids) if x == AUDIO_PAD_ID]
assert pad_positions, "找不到 <|audio_pad|>，请确认模型正确"
print(f"Audio pad 位置：{pad_positions[0]}..{pad_positions[-1]}，共 {len(pad_positions)} 个")

prefix_ids = ids[: pad_positions[0]]
suffix_ids = ids[pad_positions[-1] + 1 :]
print(f"Prefix IDs ({len(prefix_ids)}): {prefix_ids}")
print(f"Suffix IDs ({len(suffix_ids)}): {suffix_ids}")

# ── 确认解码 id 设置 ──────────────────────────────────────────────
eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
eot_id = processor.tokenizer.eos_token_id or 151643

# 所有 special token id（decode 时跳过）
special_ids = set()
for tok_id_str, info in processor.tokenizer.added_tokens_decoder.items():
    if info.special:
        special_ids.add(int(tok_id_str))

print(f"EOS id        = {eos_id}")
print(f"EOT id        = {eot_id}")
print(f"Special token count: {len(special_ids)}")

# ── 预计算所有语言的强制语言 suffix IDs ─────────────────────────────
# 格式：语言名称 → [language_id..., lang_name_id..., asr_text_id]
# 推理时附加到 suffix_ids 之后，让 decoder 直接生成文字内容（不含语言前缀）
ASR_TEXT_ID = processor.tokenizer.convert_tokens_to_ids("<asr_text>")
print(f"asr_text_id   = {ASR_TEXT_ID}")

language_suffix_ids: dict[str, list[int]] = {}
for lang in SUPPORTED_LANGUAGES:
    ids = processor.tokenizer.encode(f"language {lang}", add_special_tokens=False)
    language_suffix_ids[lang] = ids + [ASR_TEXT_ID]
print(f"语言数量：{len(language_suffix_ids)}")

# ── 存储 ──────────────────────────────────────────────────────────
template = {
    "prefix_ids":          prefix_ids,
    "suffix_ids":          suffix_ids,
    "n_audio_tokens":      len(pad_positions),
    "audio_pad_id":        AUDIO_PAD_ID,
    "eos_id":              eos_id,
    "eot_id":              eot_id,
    "special_ids":         sorted(special_ids),
    "prompt_text":         prompt_text,
    "asr_text_id":         ASR_TEXT_ID,
    "language_suffix_ids": language_suffix_ids,
    "supported_languages": list(SUPPORTED_LANGUAGES),
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(template, f, indent=2, ensure_ascii=False)

# ── 存储 mel filters（直接从 WhisperFeatureExtractor 取出）─────────
import numpy as np
fe = processor.feature_extractor
mel_filters = fe.mel_filters   # shape: (n_freqs, n_mels) = (201, 128)
mel_filters_path = BASE_DIR / "ov_models" / "mel_filters.npy"
np.save(str(mel_filters_path), mel_filters)
print(f"mel_filters shape: {mel_filters.shape} → {mel_filters_path}")

print(f"\n✅  已存储至 {OUT_PATH}")
print(f"✅  mel_filters 存储至 {mel_filters_path}")
print(f"    prefix={len(prefix_ids)} tokens, audio_pad={len(pad_positions)}, suffix={len(suffix_ids)} tokens")
print(f"    语言 suffix IDs 已预计算（{len(language_suffix_ids)} 种语言）")

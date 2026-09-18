"""
processor_numpy.py
─────────────────────────────────────────────────────────────────────
纯 numpy 实现 Qwen3-ASR Processor，完整取代 torch / transformers / qwen_asr。

功能：
  • Mel 特征提取  ─ 与 WhisperFeatureExtractor 完全对齐
  • BPE 解码      ─ byte-level GPT-2 风格，从 vocab.json 读取
  • Prompt 组装   ─ 从 prompt_template.json 读取预计算 IDs

依赖：
  numpy（已有）、pathlib（标准库）
  不需要 torch、transformers、qwen_asr

使用：
  from processor_numpy import LightProcessor
  proc = LightProcessor(ov_dir)
  mel, ids = proc.prepare(audio_float32_16khz)
  text     = proc.decode(generated_token_ids)
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Mel 特征提取（对齐 WhisperFeatureExtractor）
# ══════════════════════════════════════════════════════════════════════

# 参数来源：preprocessor_config.json
_N_FFT         = 400
_HOP           = 160
_N_MELS        = 128
_N_SAMPLES     = 480_000              # 30s × 16000
_NB_FRAMES     = 3000                 # nb_max_frames
_PAD_LEN       = (_NB_FRAMES - 1) * _HOP + _N_FFT   # = 480240（使 center=False 刚好 3000 frames）
_SR            = 16_000
_FMIN          = 0.0
_FMAX          = 8_000.0


_MEL_FILTERS: np.ndarray | None = None


def _load_mel_filters(model_dir: Path | None = None) -> np.ndarray:
    """
    加载从 WhisperFeatureExtractor 导出的 mel filterbank。
    形状为 (n_freqs, n_mels) = (201, 128)，由 generate_prompt_template.py 产生。

    不重新计算，避免 mel_scale/norm 参数不一致导致的精确度损失。
    """
    global _MEL_FILTERS
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS

    # 搜索 mel_filters.npy
    candidates: list[Path] = []
    if model_dir is not None:
        candidates.append(model_dir.parent / "mel_filters.npy")  # ov_models/mel_filters.npy
        candidates.append(model_dir / "mel_filters.npy")
    candidates.append(Path(__file__).parent / "ov_models" / "mel_filters.npy")

    for p in candidates:
        if p.exists():
            raw = np.load(str(p))       # (201, 128) or (128, 201)
            # 确保形状是 (n_mels, n_freqs) = (128, 201)
            if raw.shape == (_N_MELS, _N_FFT // 2 + 1):
                _MEL_FILTERS = raw.astype(np.float32)
            elif raw.shape == (_N_FFT // 2 + 1, _N_MELS):
                _MEL_FILTERS = raw.T.astype(np.float32)
            else:
                raise ValueError(f"mel_filters.npy shape {raw.shape} 不符预期")
            return _MEL_FILTERS

    raise FileNotFoundError(
        "找不到 ov_models/mel_filters.npy。\n"
        "请先执行：python generate_prompt_template.py"
    )


# 周期性汉宁窗（与 transformers window_function(periodic=True) 一致）
_HANN_WINDOW: np.ndarray = np.hanning(_N_FFT + 1)[:-1].astype(np.float32)


def extract_mel(audio: np.ndarray) -> np.ndarray:
    """
    输入：float32 音频，16kHz，任意长度
    输出：[1, 128, 3000] float32 mel 矩阵

    与 transformers WhisperFeatureExtractor 行为完全对齐：
      • 先截断/补零至 n_samples = 480000（30 秒）
      • center=True：两端各加 n_fft//2 = 200 个反射样本
      • 滑窗 STFT（周期汉宁窗）→ 取前 3000 frames
    """
    # 1. 截断至 n_samples；若更短则补零（center padding 前需补足）
    audio = audio.astype(np.float32)
    if len(audio) > _N_SAMPLES:
        audio = audio[:_N_SAMPLES]
    if len(audio) < _N_SAMPLES:
        audio = np.pad(audio, (0, _N_SAMPLES - len(audio)))  # 补零至 480000

    # 2. center=True：两端各加 n_fft//2 个反射样本 → 480400 个样本
    half = _N_FFT // 2  # 200
    audio_c = np.pad(audio, half, mode="reflect")             # (480400,)

    # 3. sliding_window_view → 3001 frames（取前 3000）
    frames = np.lib.stride_tricks.sliding_window_view(audio_c, _N_FFT)[::_HOP]
    frames = frames[:_NB_FRAMES].astype(np.float32)           # (3000, 400)
    windowed = frames * _HANN_WINDOW                           # (3000, 400)

    # 4. FFT → power spectrum
    stft  = np.fft.rfft(windowed, axis=1)                     # (3000, 201)
    power = np.abs(stft).astype(np.float32) ** 2              # (3000, 201)

    # 5. Mel filterbank
    mel = (_load_mel_filters() @ power.T)                     # (128, 3000)

    # 6. Log scale + Whisper 正规化
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    return log_mel[np.newaxis, :, :].astype(np.float32)       # (1, 128, 3000)


# ══════════════════════════════════════════════════════════════════════
# BPE 解码（byte-level GPT-2 风格）
# ══════════════════════════════════════════════════════════════════════

def _build_byte_decoder() -> dict[str, int]:
    """
    GPT-2 byte-to-unicode mapping 的反向版本（unicode char → byte value）。
    vocab.json 中的 token 字符串使用此编码。
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 结果：unicode char → byte value
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER: dict[str, int] = _build_byte_decoder()


def _bpe_decode(token_strings: list[str]) -> str:
    """
    将 BPE token 字符串列表解码回 UTF-8 文字。
    先拼接 byte-level unicode 字符串，再逐字符转回 bytes，最后 UTF-8 decode。
    """
    merged = "".join(token_strings)
    byte_vals = []
    for ch in merged:
        bval = _BYTE_DECODER.get(ch)
        if bval is not None:
            byte_vals.append(bval)
        # 未知字符跳过（不应出现）
    try:
        return bytes(byte_vals).decode("utf-8", errors="replace")
    except Exception:
        return merged


# ══════════════════════════════════════════════════════════════════════
# LightProcessor：组合上面两个组件
# ══════════════════════════════════════════════════════════════════════

class LightProcessor:
    """
    对应 ASREngine 中原本的 processor + pad_id。

    属性（供 app.py 使用）：
        pad_id  : int   ← <|audio_pad|> 的 token id
        eos_id  : int   ← <|im_end|>
        eot_id  : int   ← <|endoftext|>
        supported_languages : list[str]  ← 支持的语言名称清单
    """

    def __init__(self, model_dir: Path):
        """
        model_dir : OV_DIR，含 vocab.json、prompt_template.json
        prompt_template.json 从 generate_prompt_template.py 产生。
        """
        # ── 预先加载 mel filters（避免 extract_mel 每次重找路径）─────
        _load_mel_filters(model_dir)
        self._model_dir = model_dir

        # ── 读取 prompt template ──────────────────────────────────────
        # 搜索顺序：OV_DIR（模型特定设置优先）→ BASE_DIR → 本文件同层
        tpl_path = model_dir / "prompt_template.json"
        if not tpl_path.exists():
            tpl_path = model_dir.parent.parent / "prompt_template.json"
        if not tpl_path.exists():
            tpl_path = Path(__file__).parent / "prompt_template.json"
        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)

        self._prefix_ids: list[int]  = tpl["prefix_ids"]
        self._suffix_ids: list[int]  = tpl["suffix_ids"]
        self._n_audio:    int        = tpl["n_audio_tokens"]
        self.pad_id:      int        = tpl["audio_pad_id"]
        self.eos_id:      int        = tpl["eos_id"]
        self.eot_id:      int        = tpl["eot_id"]
        self._special_ids: set[int]  = set(tpl["special_ids"])
        # Mel 长度：从 template 读取，允许各模型不同（0.6B: 480000/3000，1.7B: 160000/1000）
        self._n_samples: int = tpl.get("n_samples", _N_SAMPLES)
        self._nb_frames: int = tpl.get("nb_frames", _NB_FRAMES)

        # ── 语言相关（供 UI 显示与强制语言功能）──────────────────────
        self._language_suffix_ids: dict[str, list[int]] = tpl.get("language_suffix_ids", {})
        self.supported_languages: list[str] = tpl.get("supported_languages", list(self._language_suffix_ids.keys()))

        # prefix 结构：[im_start, system, \n] | [im_end, \n, im_start, user, \n, audio_start]
        # context 插入位置：prefix[:3] + encode(context) + prefix[3:]
        self._prefix_sys_head: list[int] = self._prefix_ids[:3]   # 3 tokens
        self._prefix_sys_tail: list[int] = self._prefix_ids[3:]   # 6 tokens

        # ── 建立 id → token string 的对映（BPE decode 用）────────────
        vocab_path = model_dir / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)   # str → id
        self._id2str: dict[int, str] = {v: k for k, v in vocab.items()}

        # 补上 added_tokens（tokenizer_config.json 里的特殊 token）
        tc_path = model_dir / "tokenizer_config.json"
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        for tok_id_str, info in tc.get("added_tokens_decoder", {}).items():
            self._id2str[int(tok_id_str)] = info["content"]

        # BPE 编码器：延迟初始化，仅在需要 context hint 时才加载
        self._bpe_tokenizer = None

    # ── BPE 编码器（用于 context/hint 动态 tokenize）────────────────

    def _get_bpe_tokenizer(self):
        """
        延迟加载 BPE tokenizer（使用 tokenizers 包，纯 Rust 实现）。
        从 vocab.json + merges.txt 建立，不需 transformers。
        """
        if self._bpe_tokenizer is not None:
            return self._bpe_tokenizer
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel

            bpe = BPE.from_file(
                str(self._model_dir / "vocab.json"),
                str(self._model_dir / "merges.txt"),
                unk_token="<|endoftext|>",
            )
            tok = Tokenizer(bpe)
            tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self._bpe_tokenizer = tok
        except ImportError:
            raise ImportError(
                "需要 tokenizers 包才能使用 hint 功能：pip install tokenizers"
            )
        return self._bpe_tokenizer

    def encode_text(self, text: str) -> list[int]:
        """将任意文字 BPE encode 为 token IDs（用于 hint/context）。"""
        return self._get_bpe_tokenizer().encode(text).ids

    # ── Mel 特征提取（per-instance，使用自身 n_samples / nb_frames）────

    def _extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """输出 [1, 128, nb_frames] float32 mel，长度由 prompt_template 决定。"""
        audio = audio.astype(np.float32)
        if len(audio) > self._n_samples:
            audio = audio[:self._n_samples]
        if len(audio) < self._n_samples:
            audio = np.pad(audio, (0, self._n_samples - len(audio)))

        half = _N_FFT // 2
        audio_c = np.pad(audio, half, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(audio_c, _N_FFT)[::_HOP]
        frames = frames[:self._nb_frames].astype(np.float32)
        windowed = frames * _HANN_WINDOW

        stft  = np.fft.rfft(windowed, axis=1)
        power = np.abs(stft).astype(np.float32) ** 2
        mel   = (_load_mel_filters(self._model_dir) @ power.T)

        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel[np.newaxis, :, :].astype(np.float32)   # (1, 128, nb_frames)

    # ── 外部 API ──────────────────────────────────────────────────────

    def prepare(
        self,
        audio: np.ndarray,
        language: str | None = None,
        context: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        输入：16kHz float32 音频
        参数：
            language : 强制语言名称（如 "Chinese"、"English"），None 表示自动检测
            context  : 识别提示（歌词、关键字等），放入 system message
        输出：(mel, input_ids)
            mel       : [1, 128, 3000] float32
            input_ids : [1, L]         int64
        """
        mel = self._extract_mel(audio)

        # ── 组装 prefix（含 context/hint）────────────────────────────
        if context and context.strip():
            ctx_ids = self.encode_text(context.strip())
            prefix_ids = self._prefix_sys_head + ctx_ids + self._prefix_sys_tail
        else:
            prefix_ids = self._prefix_ids

        # ── 组装 suffix（含强制语言）─────────────────────────────────
        if language and language in self._language_suffix_ids:
            suffix_ids = self._suffix_ids + self._language_suffix_ids[language]
        else:
            suffix_ids = self._suffix_ids

        ids = np.array(
            prefix_ids + [self.pad_id] * self._n_audio + suffix_ids,
            dtype=np.int64,
        )[np.newaxis, :]

        return mel, ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """
        将生成的 token id 列表解码为 UTF-8 字符串。
        skip_special=True 时跳过 special tokens（含 <asr_text>）。
        """
        parts: list[str] = []
        for tid in token_ids:
            if skip_special and tid in self._special_ids:
                continue
            s = self._id2str.get(tid, "")
            if s:
                parts.append(s)
        return _bpe_decode(parts)

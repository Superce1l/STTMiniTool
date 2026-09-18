"""audio_io.py — 不依赖 librosa / numba 的音频加载

背景
----
`librosa` 会在 import 阶段加载 `numba`。numba 0.60 不兼容 numpy ≥ 2.1
（抛「numba needs numpy 2.0 or less」），导致 `import librosa` 直接失败，
整条转录管线（文件上传、录音上传、批量）全中。降 numpy 会牵动 openvino，
因此改以零 numba 的组合取代 librosa.load：

  • soundfile（libsndfile）— 读 wav / flac / ogg / mp3，回原生取样率
  • soxr（C 绑定）        — 高质量重采样；缺则退 scipy.resample_poly；
                            再缺则退 numpy 线性插值
  • ffmpeg 后援           — soundfile 读不了的格式（m4a / aac / wma…）
                            先用 ffmpeg 转 16k 单声道 wav 再读

对外：
  load_audio_16k_mono(path) -> (np.float32 mono @16k, 16000)
  audio_duration(path)      -> 秒数（float）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SR = 16000
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """重采样（soxr → scipy → numpy 线性插值），皆不依赖 numba。"""
    if sr_in == sr_out:
        return data.astype(np.float32, copy=False)
    try:
        import soxr
        return soxr.resample(data, sr_in, sr_out).astype(np.float32)
    except Exception:
        pass
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(data, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = max(1, int(round(len(data) * sr_out / sr_in)))
        xp = np.linspace(0, len(data) - 1, n)
        return np.interp(xp, np.arange(len(data)), data).astype(np.float32)


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32)


def _ffmpeg_to_wav(path: Path) -> Path | None:
    """soundfile 读不了时，用 ffmpeg 转 16k 单声道 wav，回临时路径。"""
    try:
        import subprocess
        import tempfile
        from ffmpeg_utils import find_ffmpeg
        ff = find_ffmpeg()
        if not ff:
            return None
        out = Path(tempfile.mkdtemp(prefix="aio_")) / "audio.wav"
        subprocess.run(
            [str(ff), "-y", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SR), "-f", "wav", str(out)],
            check=True, capture_output=True,
            creationflags=_CREATE_NO_WINDOW,
        )
        return out if out.exists() else None
    except Exception:
        return None


def load_audio_16k_mono(path, target_sr: int = SR):
    """加载音频为 (np.float32 mono @ target_sr, target_sr)。不使用 librosa/numba。

    soundfile 直接可读 → 读后（必要时）重采样；读不了的格式退回 ffmpeg
    转 16k wav 再读。
    """
    import soundfile as sf
    p = Path(path)
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
        data = _to_mono(data)
    except Exception:
        wav = _ffmpeg_to_wav(p)
        if wav is None:
            raise
        data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        data = _to_mono(data)
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
    return data, target_sr


def audio_duration(path) -> float:
    """音频长度（秒），不使用 librosa。soundfile 标头优先，失败才实际解码。"""
    import soundfile as sf
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        data, sr = load_audio_16k_mono(path)
        return len(data) / float(sr)
    except Exception:
        return 0.0

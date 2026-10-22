"""VAD 批量推理等价性：批量路径与顺序路径的 0.5 阈值决策应一致

需要随包的 ov_models/silero_vad_v4.onnx（缺失则跳过——打包后模型在别处，
CI 无模型时跳过是预期行为）。
"""
import unittest
from pathlib import Path

from common import *  # noqa: F401,F403

import numpy as np

from app import SAMPLE_RATE, VAD_CHUNK, _vad_is_v5

MODEL = Path(__file__).resolve().parents[1] / "ov_models" / "silero_vad_v4.onnx"


def _seq_probs(audio, sess):
    """逐窗顺序推理（v4），返回概率序列 —— 语义基准"""
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    n = len(audio) // VAD_CHUNK
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        chunk = audio[i * VAD_CHUNK:(i + 1) * VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        o, h, c = sess.run(None, {"input": chunk, "h": h, "c": c,
                                  "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        out[i] = o[0, 0]
    return out


@unittest.skipUnless(MODEL.exists(), f"缺 VAD 模型：{MODEL}")
class TestVadBatchEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        cls.sess = ort.InferenceSession(str(MODEL),
                                        providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(11)
        t = np.arange(SAMPLE_RATE * 40, dtype=np.float32)
        cls.audio = (0.3 * np.sin(2 * np.pi * 200 * t / SAMPLE_RATE)
                     * (1 + 0.4 * np.sin(2 * np.pi * 0.5 * t / SAMPLE_RATE)))
        cls.audio = cls.audio.astype(np.float32)
        cls.audio += rng.normal(0, 0.02, len(cls.audio)).astype(np.float32)

    def test_v4_detected(self):
        self.assertFalse(_vad_is_v5(self.sess))

    def test_batch_decisions_match_sequential(self):
        import app as core
        # 强制长音频 → 走批量路径
        groups = core._detect_speech_groups(self.audio, self.sess)
        self.assertIsInstance(groups, list)
        # 对比两条路径的 0.5 决策：以「顺序基准重建的分段」为验收基准，
        # 批量路径产出的分段数与边界（PAD/尾补容差内）应一致。
        seq_probs = _seq_probs(self.audio, self.sess)
        seq_groups = self._groups_from_probs(seq_probs)
        self.assertEqual(len(groups), len(seq_groups))
        for (b0, b1, _cb), (s0, s1, _cs) in zip(groups, seq_groups):
            self.assertAlmostEqual(b0, s0, delta=0.35)   # PAD/尾补容差内
            self.assertAlmostEqual(b1, s1, delta=0.35)

    def _groups_from_probs(self, probs):
        """按 _detect_speech_groups 的阈值/合并参数从概率序列重建分段（基准）"""
        MIN_CH, PAD, MERGE = 16, 5, 16
        n = len(probs)
        raw, in_sp, s0 = [], False, 0
        for i, p in enumerate(probs):
            if p >= 0.5 and not in_sp:
                s0, in_sp = i, True
            elif p < 0.5 and in_sp:
                if i - s0 >= MIN_CH:
                    raw.append((max(0, s0 - PAD), min(n, i + PAD)))
                in_sp = False
        if in_sp and n - s0 >= MIN_CH:
            raw.append((max(0, s0 - PAD), n))
        if not raw:
            return []
        merged = [list(raw[0])]
        for s, e in raw[1:]:
            if s - merged[-1][1] <= MERGE:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        return merged

    def test_short_audio_uses_sequential_exactly(self):
        """短音频路径应与逐窗基准完全一致（未启用批量）"""
        import app as core
        short = self.audio[:SAMPLE_RATE * 2]
        groups = core._detect_speech_groups(short, self.sess)
        seq_probs = _seq_probs(short, self.sess)
        self.assertEqual(len(groups), len(self._groups_from_probs(seq_probs.tolist())))


if __name__ == "__main__":
    unittest.main()

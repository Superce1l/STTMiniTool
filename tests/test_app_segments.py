"""app.py 分段/断句工具与引擎契约"""
import unittest

from common import *  # noqa: F401,F403

import numpy as np
from engine_types import TranscribeResult
from app import GAP_SEC, SAMPLE_RATE, ASREngine, _assign_ts, _resolve_backend, _split_to_lines


class TestSplitToLines(unittest.TestCase):
    def test_punct_split(self):
        self.assertEqual(_split_to_lines("你好，世界。"), ["你好", "世界"])

    def test_max_chars(self):
        text = "字" * 45
        lines = _split_to_lines(text)
        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines), text)

    def test_latin_words(self):
        lines = _split_to_lines("hello world this is a long sentence ok")
        for l in lines:
            self.assertLessEqual(len(l), 20)
        self.assertIn("hello", lines[0])

    def test_asr_text_tag(self):
        self.assertEqual(_split_to_lines("<asr_text>内容</asr_text>"),
                         ["内容"])


class TestAssignTs(unittest.TestCase):
    def test_proportional(self):
        out = _assign_ts(["短", "比上一次更长的一句话"], 0.0, 10.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], 0.0)
        self.assertGreaterEqual(out[-1][1], 10.0)   # 末行结束不早于段尾
        for (s0, e0, _t0), (s1, _e1, _t1) in zip(out, out[1:]):
            self.assertAlmostEqual(s1, e0 + GAP_SEC, places=6)

    def test_empty(self):
        self.assertEqual(_assign_ts([], 0, 1), [])


class TestQuietSplitPos(unittest.TestCase):
    def test_prefers_silence(self):
        sr = SAMPLE_RATE
        chunk = np.ones(sr * 10, dtype=np.float32) * 0.5
        chunk[sr * 5: sr * 5 + 2000] = 0.0        # 中间有 0.125s 静音
        pos = ASREngine._quiet_split_pos(chunk, sr * 5 + 3000, sr * 4, sr * 6)
        # 切点吸附到静音窗内（RMS 最低处）
        self.assertTrue(sr * 5 - sr * 0.05 <= pos <= sr * 5 + 2000 + sr * 0.05)

    def test_invalid_range_fallback(self):
        chunk = np.ones(1000, dtype=np.float32)
        self.assertEqual(ASREngine._quiet_split_pos(chunk, 500, 900, 800), 500)


class TestEnforceChunkLimit(unittest.TestCase):
    def test_split_overlong(self):
        eng = ASREngine()
        eng.max_chunk_secs = 10
        sr = SAMPLE_RATE
        chunk = np.random.default_rng(1).normal(0, 0.1, sr * 35).astype(np.float32)
        groups = [(0.0, 35.0, chunk, None)]
        out = eng._enforce_chunk_limit(groups)
        self.assertGreaterEqual(len(out), 3)
        for t0, t1, piece, spk in out:
            self.assertLessEqual(len(piece), 10 * sr)
            self.assertGreaterEqual(len(piece), sr)          # ≥1s
            self.assertIsNone(spk)

    def test_passthrough(self):
        eng = ASREngine()
        sr = SAMPLE_RATE
        chunk = np.zeros(sr, dtype=np.float32)
        out = eng._enforce_chunk_limit([(5.0, 6.0, chunk, "说话者1")])
        self.assertEqual(out, [(5.0, 6.0, chunk, "说话者1")])


class TestResolveBackend(unittest.TestCase):
    def test_whisper(self):
        self.assertEqual(_resolve_backend("Whisper (CRISPASR)", "Whisper Large Turbo"),
                         ("crispasr", "turbo"))
        self.assertEqual(_resolve_backend("Whisper", "Whisper Base"), ("crispasr", "base"))

    def test_openvino(self):
        self.assertEqual(_resolve_backend("Qwen", "Qwen3-ASR-1.7B INT8"), ("openvino", "1.7B"))
        self.assertEqual(_resolve_backend("Qwen", "Qwen3-ASR-0.6B"), ("openvino", "0.6B"))


class TestEngineContract(unittest.TestCase):
    """两引擎的取消事件契约与结构化返回值（webview_backend 统一依赖）"""

    def test_cancel_event_present(self):
        from crisp_engine import CrispWhisperEngine
        for eng in (ASREngine(), CrispWhisperEngine()):
            self.assertTrue(hasattr(eng, "cancel_event"))
            self.assertFalse(eng.cancel_event.is_set())
            eng.cancel_event.set()
            self.assertTrue(eng.cancel_event.is_set())
            eng.cancel_event.clear()

    def test_transcribe_result_defaults(self):
        r = TranscribeResult()
        self.assertIsNone(r.output)
        self.assertIsNone(r.segments_rich)
        self.assertIsNone(r.diag)
        self.assertFalse(r.cancelled)
        self.assertEqual(r.degenerate, 0)


if __name__ == "__main__":
    unittest.main()

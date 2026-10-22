"""SRT 解析与 CrispASR 字符级 SRT 的重建工具"""
import unittest

from common import *  # noqa: F401,F403

import crisp_engine
from crisp_engine import (_fix_leading_punct, _infer_backend, _interpolate_chars,
                          _parse_srt_words)
from webview_backend import _parse_srt, parse_srt_to_segments

SRT = """1
00:00:01,000 --> 00:00:02,000
你好

2
00:00:02,500 --> 00:00:03,200
说话者1：世界

"""


class TestParseSrt(unittest.TestCase):
    def test_basic(self):
        segs = _parse_srt(SRT)
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0]["start"], 1.0)
        self.assertAlmostEqual(segs[1]["end"], 3.2)
        self.assertEqual(segs[1]["text"], "说话者1：世界")

    def test_missing_file_returns_empty(self):
        import tempfile
        from pathlib import Path
        segs = parse_srt_to_segments(
            Path(tempfile.gettempdir()) / "stt_nonexist_file.srt")
        self.assertEqual(segs, [])

    def test_speaker_split(self):
        import tempfile, os
        from pathlib import Path
        with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False,
                                         encoding="utf-8") as f:
            f.write(SRT)
            p = f.name
        try:
            segs = parse_srt_to_segments(p)
            self.assertIsNone(segs[0]["speaker"])
            self.assertEqual(segs[1]["speaker"], 1)
            self.assertEqual(segs[1]["text"], "世界")
        finally:
            os.unlink(p)


class TestSrtWords(unittest.TestCase):
    def test_char_level_roundtrip(self):
        srt = """1
00:00:00,000 --> 00:00:00,200
你

2
00:00:00,200 --> 00:00:00,400
好

3
00:00:00,400 --> 00:00:00,500
，

4
00:00:00,500 --> 00:00:00,700
世

"""
        items, raw = _parse_srt_words(srt, drop_punct=True)
        self.assertEqual([t for t, _s, _e in items], ["你", "好", "世"])
        self.assertEqual(raw, "你好，世")

    def test_whitespace_boundary(self):
        srt = """1
00:00:00,000 --> 00:00:00,200
 Hi

2
00:00:00,200 --> 00:00:00,400
 there

"""
        items, raw = _parse_srt_words(srt)
        self.assertEqual(raw, " Hi there")
        self.assertEqual([t for t, _s, _e in items], ["Hi", "there"])

    def test_interpolate_chars(self):
        out = _interpolate_chars([("你好", 1.0, 2.0)])
        self.assertEqual([(t, round(s, 4), round(e, 4)) for t, s, e in out],
                         [("你", 1.0, 1.5), ("好", 1.5, 2.0)])
        # 单字符原样保留
        self.assertEqual(_interpolate_chars([("x", 0.0, 1.0)]), [("x", 0.0, 1.0)])

    def test_fix_leading_punct(self):
        # 契约：行首标点移回前一行行尾，其余文字仍独立成行
        lines5 = [(0.0, 1.0, "你好", None, []),
                  (2.0, 3.0, "，世界", None, [])]
        out = _fix_leading_punct(lines5)
        self.assertEqual(out[0][2], "你好，")
        self.assertEqual(out[1][2], "世界")
        self.assertEqual(out[1][0], 2.0)   # 时间轴归属不动

    def test_fix_leading_punct_first_line(self):
        out = _fix_leading_punct([(0.0, 1.0, "。开头", None, [])])
        self.assertEqual(out[0][2], "开头")


class TestInferBackend(unittest.TestCase):
    def test_whisper(self):
        self.assertEqual(_infer_backend("ggml-base.bin"), "whisper")
        self.assertEqual(_infer_backend(r"C:\x\whisper large-v3.bin"), "whisper")

    def test_qwen3(self):
        self.assertEqual(_infer_backend("qwen3-asr-1.7b-q4_k.gguf"), "qwen3-1.7b")
        self.assertEqual(_infer_backend("qwen3-asr-0.6b.gguf"), "qwen3")


if __name__ == "__main__":
    unittest.main()

"""subtitle_lines — 字级时间轴 → 字幕行（全引擎共享的核心断句逻辑）"""
import tempfile
import unittest
from pathlib import Path

from common import *  # noqa: F401,F403  (sys.path 注入)

from subtitle_lines import (_srt_ts, _strip_pua, _ts_chatllm_to_subtitle_lines,
                            lines_to_txt, write_transcript, MAX_CHARS)


class TestStripPua(unittest.TestCase):
    def test_removes_pua(self):
        s = "正常字\uE000\ueF8B混入私用区"
        self.assertEqual(_strip_pua(s), "正常字混入私用区")

    def test_keeps_normal(self):
        self.assertEqual(_strip_pua("中文 English 123。"), "中文 English 123。")
        self.assertEqual(_strip_pua(""), "")
        self.assertEqual(_strip_pua(None), None)


class TestSrtTs(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_srt_ts(0), "00:00:00,000")
        self.assertEqual(_srt_ts(1.56), "00:00:01,560")
        self.assertEqual(_srt_ts(3661.5), "01:01:01,500")

    def test_rounding_overflow(self):
        # 0.9999 → 进位到下一秒
        self.assertEqual(_srt_ts(0.9999), "00:00:01,000")


class TestTsToSubtitleLines(unittest.TestCase):
    def test_punct_split(self):
        # Qwen 契约：标点只在 raw_text，items 仅内容字
        items = [("你", 0.0, 0.2), ("好", 0.2, 0.4),
                 ("世", 0.6, 0.8), ("界", 0.8, 1.0)]
        out = _ts_chatllm_to_subtitle_lines(items, "你好，世界。", 0.0, None)
        self.assertEqual([(t, s, e) for s, e, t, _ in out],
                         [("你好", 0.0, 0.4), ("世界", 0.6, 1.0)])

    def test_chunk_offset(self):
        items = [("字", 0.0, 0.3)]
        out = _ts_chatllm_to_subtitle_lines(items, "字", 10.0, None)
        self.assertEqual(out[0][0], 10.0)
        self.assertAlmostEqual(out[0][1], 10.3)

    def test_max_chars_overflow(self):
        items = [(ch, i * 0.1, i * 0.1 + 0.1) for i, ch in enumerate("abcdefghij" * 4)]
        raw = "".join(ch for ch, _s, _e in items)
        out = _ts_chatllm_to_subtitle_lines(items, raw, 0.0, None)
        self.assertGreater(len(out), 1)
        for _s, _e, t, _spk in out:
            self.assertLessEqual(len(t.replace(" ", "")), MAX_CHARS)

    def test_pause_break(self):
        # 字间空隙 ≥0.35s 且行已累积 ≥5 字 → 停顿处切行（无标点）
        chars = "一二三四五六七八九十"
        items, raw = [], ""
        t = 0.0
        for i, ch in enumerate(chars):
            gap = 0.5 if i == 5 else 0.0     # 「六」前停顿 0.5s（此时行已 5 字）
            t += gap
            items.append((ch, t, t + 0.1))
            raw += ch
            t += 0.1
        out = _ts_chatllm_to_subtitle_lines(items, raw, 0.0, None)
        self.assertEqual([t2 for _s, _e, t2, _ in out],
                         ["一二三四五", "六七八九十"])

    def test_orphan_merge(self):
        # 句尾「吧」单独成行 → 并回前一行
        items = [("你", 0.0, 0.2), ("好", 0.2, 0.4), ("吧", 0.4, 0.6)]
        out = _ts_chatllm_to_subtitle_lines(items, "你好吧", 0.0, None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][2], "你好吧")

    def test_words_output(self):
        items = [("你", 0.0, 0.2), ("好", 0.2, 0.4)]
        out = _ts_chatllm_to_subtitle_lines(items, "你好", 0.0, None, with_words=True)
        s, e, t, spk, words = out[0]
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0]["text"], "你")
        self.assertEqual(words[1]["start"], 0.2)

    def test_latin_words_join_with_space(self):
        # whisper 契约（break_on_space=True）：空白＝语句边界 → 逐词切行
        items = [("Hello", 0.0, 0.3), ("world", 0.3, 0.6)]
        out = _ts_chatllm_to_subtitle_lines(items, "Hello world", 0.0, None,
                                            break_on_space=True)
        self.assertEqual([(t, s, e) for s, e, t, _ in out],
                         [("Hello", 0.0, 0.3), ("world", 0.3, 0.6)])
        # Qwen 契约（默认 False）：空白仅分隔拉丁词，不切行
        out2 = _ts_chatllm_to_subtitle_lines(items, "Hello world", 0.0, None)
        self.assertEqual(out2[0][2], "Hello world")


class TestWriteTranscript(unittest.TestCase):
    def _lines(self):
        return [(0.0, 1.0, "第一句", None), (1.2, 2.0, "第二句", "说话者1")]

    def test_srt(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "audio"
            out = write_transcript(ref, self._lines(), out_format="srt")
            self.assertEqual(out.name, "audio.srt")
            text = out.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,000", text)
            self.assertIn("说话者1：第二句", text)

    def test_txt_with_speaker(self):
        with tempfile.TemporaryDirectory() as td:
            out = write_transcript(Path(td) / "a", self._lines(), out_format="txt")
            self.assertEqual(out.read_text(encoding="utf-8"),
                             "第一句\n说话者1：第二句")

    def test_txt_latin_spacing(self):
        lines = [(0, 1, "how are", None), (1, 2, "you", None)]
        with tempfile.TemporaryDirectory() as td:
            out = write_transcript(Path(td) / "a", lines, out_format="txt")
            self.assertEqual(out.read_text(encoding="utf-8"), "how are you")


if __name__ == "__main__":
    unittest.main()

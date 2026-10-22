"""长音频切片：切片计划与重叠区合并去重"""
import unittest

from common import *  # noqa: F401,F403

from webview_backend import (_SLICE_OVERLAP_SECS, _merge_sliced_segments,
                             _slice_plan)


class TestSlicePlan(unittest.TestCase):
    def test_short_audio_single_span(self):
        self.assertEqual(_slice_plan(60.0, window_secs=1800), [(0.0, 60.0)])

    def test_overlap(self):
        spans = _slice_plan(3600.0, window_secs=600)
        self.assertGreater(len(spans), 1)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertAlmostEqual(a1 - b0, _SLICE_OVERLAP_SECS, places=3)
        self.assertAlmostEqual(spans[-1][1], 3600.0, places=3)

    def test_exact_multiple(self):
        # 时长恰好等于一个窗口 → 只有一片
        self.assertEqual(len(_slice_plan(1800.0, window_secs=1800)), 1)


def _seg(s, e, t, spk=None):
    return {"start": s, "end": e, "text": t, "speaker": spk}


class TestMergeSliced(unittest.TestCase):
    def test_no_overlap_passthrough(self):
        parts = [[_seg(0, 2, "甲")], [_seg(5, 7, "乙")]]
        merged = _merge_sliced_segments(parts)
        self.assertEqual([m["text"] for m in merged], ["甲", "乙"])

    def test_dup_prefers_longer_text(self):
        # 重叠区同一句：前片截断版「今天天」 vs 后片完整版「今天天气很好」
        parts = [
            [_seg(0, 10, "前半段内容"), _seg(10, 13, "今天天")],
            [_seg(10.5, 15.0, "今天天气很好"), _seg(15, 20, "后半段内容")],
        ]
        merged = _merge_sliced_segments(parts)
        texts = [m["text"] for m in merged]
        self.assertIn("今天天气很好", texts)
        self.assertNotIn("今天天", texts)
        # 时间轴有序
        starts = [m["start"] for m in merged]
        self.assertEqual(starts, sorted(starts))

    def test_dup_identical_text(self):
        parts = [
            [_seg(0, 5, "同一句")],
            [_seg(4, 9, "同一句"), _seg(10, 12, "新的话")],
        ]
        merged = _merge_sliced_segments(parts)
        self.assertEqual(len([m for m in merged if m["text"] == "同一句"]), 1)

    def test_empty_parts(self):
        self.assertEqual(_merge_sliced_segments([]), [])
        self.assertEqual(_merge_sliced_segments([[], [_seg(0, 1, "x")]]),
                         [_seg(0, 1, "x")])


if __name__ == "__main__":
    unittest.main()

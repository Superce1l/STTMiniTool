"""cli apply --base 合并语义（apply_merge_segments 纯函数，不落盘）

锁定的契约（与 GUI/Agent 工作流约定）：
  • 时间轴与说话者永远取基准（--base）——校正端常把时间字段改坏；
  • text 永远采用校正稿；
  • 按 1-based 序号 i 对齐，校正稿缺段时该段保留基准原文；
  • base 为空时原样返回校正稿。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

from cli_mode import apply_merge_segments


def _seg(i, start, end, text, speaker=None):
    d = {"i": i, "start": start, "end": end, "text": text}
    if speaker is not None:
        d["speaker"] = speaker
    return d


class TestApplyMerge(unittest.TestCase):
    def test_base_none_returns_edited_as_is(self):
        segs = [_seg(1, 0.0, 1.0, "校正")]
        self.assertIs(apply_merge_segments(segs, None), segs)
        self.assertIs(apply_merge_segments(segs, []), segs)

    def test_text_from_edited_timeline_from_base(self):
        base = [_seg(1, 0.0, 1.0, "原始", speaker=1)]
        edited = [_seg(1, 99.0, 999.0, "校正稿")]     # 时间字段被校正端改坏
        out = apply_merge_segments(edited, base)
        self.assertEqual(out, [{"i": 1, "start": 0.0, "end": 1.0,
                                "text": "校正稿", "speaker": 1}])

    def test_missing_edited_segment_keeps_base_text(self):
        base = [_seg(1, 0.0, 1.0, "第一句"), _seg(2, 1.0, 2.0, "第二句")]
        edited = [_seg(1, 0, 0, "第一句校正")]
        out = apply_merge_segments(edited, base)
        self.assertEqual([s["text"] for s in out], ["第一句校正", "第二句"])
        self.assertEqual([s["start"] for s in out], [0.0, 1.0])

    def test_segments_beyond_base_are_dropped(self):
        # 校正稿比基准多出的段无处安放时间轴 → 丢弃（现行为锁定）
        base = [_seg(1, 0.0, 1.0, "only")]
        edited = [_seg(1, 0, 0, "only"), _seg(2, 0, 0, "多出来的")]
        out = apply_merge_segments(edited, base)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "only")

    def test_noncontiguous_i_alignment(self):
        base = [_seg(3, 5.0, 6.0, "三"), _seg(7, 8.0, 9.0, "七")]
        edited = [_seg(3, 0, 0, "三改"), _seg(7, 0, 0, "七改")]
        out = apply_merge_segments(edited, base)
        self.assertEqual([(s["i"], s["text"]) for s in out],
                         [(3, "三改"), (7, "七改")])

    def test_speaker_taken_from_base_not_edited(self):
        base = [_seg(1, 0.0, 1.0, "原始", speaker=2)]
        edited = [_seg(1, 0, 0, "改", speaker=9)]   # 校正稿的 speaker 不采信
        out = apply_merge_segments(edited, base)
        self.assertEqual(out[0]["speaker"], 2)


if __name__ == "__main__":
    unittest.main()

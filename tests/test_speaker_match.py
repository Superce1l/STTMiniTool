"""跨片说话者声纹映射（webview_backend._match_speaker）"""
import unittest

from common import *  # noqa: F401,F403

import numpy as np
from webview_backend import _SPK_MATCH_SIM, _match_speaker
from slicing import assign_speaker_by_overlap


def _emb(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


class TestMatchSpeaker(unittest.TestCase):
    def test_first_assignment_and_growth(self):
        g = []
        self.assertEqual(_match_speaker(_emb([1, 0, 0]), g), 1)
        self.assertEqual(_match_speaker(_emb([0, 1, 0]), g), 2)   # 正交 → 新人
        self.assertEqual(len(g), 2)

    def test_same_voice_same_label(self):
        g = []
        e = _emb([1, 0.1, 0])
        l1 = _match_speaker(e, g)
        l2 = _match_speaker(_emb([1, 0.12, 0.02]), g)   # 轻微抖动 → 同一人
        self.assertEqual(l1, l2)

    def test_distinct_voices_new_label(self):
        g = []
        _match_speaker(_emb([1, 0, 0]), g)
        l = _match_speaker(_emb([-1, 0, 0]), g)         # 反向 → 新人
        self.assertEqual(l, 2)

    def test_running_centroid_updates(self):
        g = []
        _match_speaker(_emb([1, 0, 0]), g)
        before = g[0]["emb"].copy()
        _match_speaker(_emb([0.9, 0.1, 0]), g)
        self.assertEqual(g[0]["n"], 2)
        self.assertFalse(np.allclose(before, g[0]["emb"]))
        self.assertAlmostEqual(float(np.linalg.norm(g[0]["emb"])), 1.0, places=5)

    def test_threshold_semantics(self):
        # 相似度恰在阈值以下 → 新开一人
        g = []
        _match_speaker(_emb([1, 0, 0]), g)
        cos = 0.5   # < _SPK_MATCH_SIM
        b = np.sqrt(1 - cos * cos)
        l = _match_speaker(_emb([cos, b, 0]), g)
        self.assertEqual(l, 2)
        self.assertGreater(_SPK_MATCH_SIM, 0.5)


class TestAssignSpeakerByOverlap(unittest.TestCase):
    """行 → 说话者指派：重叠最长段优先（取代旧的「中点落点」规则）。"""

    SPANS = [(0.0, 10.0, "说话者1"), (10.0, 20.0, "说话者2")]

    def test_majority_overlap_wins(self):
        # 行 8–12s：与 1 号重叠 2s、与 2 号重叠 2s → 平手取先出现（1 号）
        self.assertEqual(assign_speaker_by_overlap(8.0, 12.0, self.SPANS), "说话者1")
        # 行 6–14s：与 1 号重叠 4s、与 2 号重叠 4s → 平手取先
        self.assertEqual(assign_speaker_by_overlap(6.0, 14.0, self.SPANS), "说话者1")
        # 行 9–13s：与 1 号重叠 1s、与 2 号重叠 3s → 2 号（旧中点规则会给 1 号）
        self.assertEqual(assign_speaker_by_overlap(9.0, 13.0, self.SPANS), "说话者2")

    def test_adjacent_spans_same_as_midpoint_rule(self):
        # 连续相邻两段：重叠最长规则与「中点落点」规则数学等价（锁定不回归）
        self.assertEqual(assign_speaker_by_overlap(9.0, 12.0, self.SPANS), "说话者2")
        self.assertEqual(assign_speaker_by_overlap(6.0, 14.0, self.SPANS), "说话者1")

    def test_overlap_differs_from_midpoint_rule(self):
        """真正分叉场景：两段时间轴本身交叠时，中点规则与重叠规则会给出不同结果。"""
        spans = [(0.0, 8.0, "说话者1"), (6.0, 20.0, "说话者2")]
        # 行 6.5–8.5：中点 7.5 落在 1 号（旧规则），与 2 号重叠 2.5s > 1 号 1.5s
        self.assertEqual(assign_speaker_by_overlap(6.5, 8.5, spans), "说话者2")

    def test_no_overlap_falls_back_to_nearest(self):
        self.assertEqual(assign_speaker_by_overlap(25.0, 26.0, self.SPANS), "说话者2")

    def test_empty_spans_returns_none(self):
        self.assertIsNone(assign_speaker_by_overlap(1.0, 2.0, []))


if __name__ == "__main__":
    unittest.main()

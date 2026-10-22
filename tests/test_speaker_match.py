"""跨片说话者声纹映射（webview_backend._match_speaker）"""
import unittest

from common import *  # noqa: F401,F403

import numpy as np
from webview_backend import _SPK_MATCH_SIM, _match_speaker


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


if __name__ == "__main__":
    unittest.main()

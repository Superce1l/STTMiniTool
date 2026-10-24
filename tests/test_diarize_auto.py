"""说话者自动计数：谱特征值间隙（_eigengap_k）与重叠子窗（_subwindows）"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

import numpy as np

from diarize import EMB_WIN_SEC, _eigengap_k, _subwindows


def _cluster(center, n, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=len(center))
    v = v / np.linalg.norm(v)
    pts = center + rng.normal(0, scale, (n, len(center)))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


class TestEigengapK(unittest.TestCase):
    def test_single_cluster(self):
        e = _cluster(np.zeros(32), 20, seed=1)
        self.assertEqual(_eigengap_k(e), 1)

    def test_two_clusters(self):
        a = _cluster(np.eye(32)[0] * 0.9, 12, seed=2)
        b = _cluster(np.eye(32)[1] * 0.9, 12, seed=3)
        self.assertEqual(_eigengap_k(np.vstack([a, b])), 2)

    def test_three_clusters(self):
        pts = np.vstack([
            _cluster(np.eye(32)[i] * 0.9, 10, seed=4 + i) for i in range(3)
        ])
        self.assertEqual(_eigengap_k(pts), 3)

    def test_shuffled_order(self):
        a = _cluster(np.eye(32)[0] * 0.9, 10, seed=7)
        b = _cluster(np.eye(32)[1] * 0.9, 10, seed=8)
        m = np.vstack([a, b])
        np.random.default_rng(9).shuffle(m)
        self.assertEqual(_eigengap_k(m), 2)

    def test_few_samples(self):
        e = np.eye(3, 32)                       # 3 个近正交样本
        self.assertIn(_eigengap_k(e), (1, 2, 3))  # 不崩溃即可


class TestSubwindows(unittest.TestCase):
    def test_short_segment_equal_split(self):
        wins = _subwindows(0.0, 3.0)
        self.assertEqual(len(wins), 1)
        self.assertAlmostEqual(wins[0][1] - wins[0][0], 3.0)

    def test_overlap_and_coverage(self):
        t0, t1 = 0.0, 12.0
        wins = _subwindows(t0, t1)
        self.assertGreater(len(wins), 2)                     # 重叠加密
        for a, b in wins:
            self.assertLessEqual(b - a, EMB_WIN_SEC + 1e-6)  # 不超窗长
        self.assertAlmostEqual(min(a for a, _ in wins), t0, places=5)
        self.assertAlmostEqual(max(b for _, b in wins), t1, places=5)  # 覆盖到尾
        for (a1, _), (a2, _) in zip(wins, wins[1:]):         # 步进 ≈ 半窗
            self.assertAlmostEqual(a2 - a1, EMB_WIN_SEC / 2, delta=0.8)

    def test_tail_window_added(self):
        wins = _subwindows(0.0, 7.3)                          # 非整窗尾
        self.assertAlmostEqual(max(b for _, b in wins), 7.3, places=5)
        for a, b in wins:
            self.assertGreaterEqual(b, EMB_WIN_SEC - 0.01) if b < 7.3 else None

    def test_no_tiny_windows(self):
        wins = _subwindows(0.0, 9.0)
        for a, b in wins:
            self.assertGreaterEqual(b - a, EMB_WIN_SEC * 0.9)


if __name__ == "__main__":
    unittest.main()

"""说话者自动计数：群心距离搜索（_cluster_auto）与重叠子窗（_subwindows）

合成数据刻意模拟真实声纹的相似度量级（说话者中心两两 cosine ~0.75-0.8、
窗内噪声使同人窗相似 ~0.7）——旧测试用近正交中心（跨群相似 ≈0）太干净，
先后放过了 argmax-silhouette 与特征值间隙两版缺陷（详见 _cluster_auto 文档）。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

import numpy as np

from diarize import (EMB_WIN_SEC, TINY_CLUSTER_WINDOWS, SPLIT_SURE_DIST,
                     DiarizationEngine, _absorb_tiny, _min_centroid_dist,
                     _subwindows)

_DIM = 64


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _cluster_auto(embeddings) -> list[int]:
    """不加载 ONNX 直接调聚类（模型缺失时 DiarizationEngine.ready=False
    但纯 scipy 的 _cluster_auto 仍可用）。"""
    eng = DiarizationEngine(Path("nonexistent-diar-dir"))
    return eng._cluster_auto(embeddings)


class _Synth:
    """真实量级的合成声纹：中心两两 cosine 可控，成员 = 中心 + σ·随机单位向量。

    校准自真实对唱歌曲的实测值：不同人声纹中心 cosine ≈ 0.58（群心距离
    0.417）、同人窗相似 ≈ 0.7（σ=0.55 → 1/(1+σ²) ≈ 0.77）。
    """
    SIGMA = 0.55
    SPK_COS = 0.56      # 不同说话者中心的 cosine（群心距 ≈ 0.44）
    SUB_COS = 0.68      # 同一人子群中心的 cosine（群心距 ≈ 0.32 < 门槛）

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def center_from(self, base, cos_ang: float) -> np.ndarray:
        r = self.rng.standard_normal(len(base))
        r -= (r @ base) * base
        return _unit(cos_ang * base + np.sqrt(max(0.0, 1 - cos_ang ** 2)) * _unit(r))

    def members(self, center, n: int) -> np.ndarray:
        out = []
        for _ in range(n):
            r = self.rng.standard_normal(len(center))
            out.append(_unit(center + self.SIGMA * _unit(r)))
        return np.stack(out)


def _speaker_set(labels) -> set:
    return set(np.asarray(labels).tolist())


class TestClusterAuto(unittest.TestCase):
    def test_single_speaker(self):
        s = _Synth(seed=1)
        labels = _cluster_auto(s.members(_unit(np.ones(_DIM)), 40).tolist())
        self.assertEqual(_speaker_set(labels), {1})

    def test_two_speakers_similar_timbre(self):
        s = _Synth(seed=2)
        c1 = _unit(s.rng.standard_normal(_DIM))
        c2 = s.center_from(c1, _Synth.SPK_COS)
        pts = np.vstack([s.members(c1, 20), s.members(c2, 20)])
        labels = _cluster_auto(pts.tolist())
        self.assertEqual(len(_speaker_set(labels)), 2)
        self.assertEqual(_min_centroid_dist(pts, np.asarray(labels)),
                         _min_centroid_dist(pts, np.asarray(
                             [1] * 20 + [2] * 20)), )   # 切分与真值一致

    def test_three_speakers(self):
        s = _Synth(seed=3)
        c1 = _unit(s.rng.standard_normal(_DIM))
        c2 = s.center_from(c1, _Synth.SPK_COS)
        c3 = s.center_from(c2, _Synth.SPK_COS)
        pts = np.vstack([s.members(c1, 12), s.members(c2, 12),
                         s.members(c3, 12)])
        labels = _cluster_auto(pts.tolist())
        self.assertEqual(len(_speaker_set(labels)), 3)

    def test_shuffled_order(self):
        s = _Synth(seed=4)
        c1 = _unit(s.rng.standard_normal(_DIM))
        pts = np.vstack([s.members(c1, 15),
                         s.members(s.center_from(c1, _Synth.SPK_COS), 15)])
        np.random.default_rng(5).shuffle(pts)
        self.assertEqual(len(_speaker_set(_cluster_auto(pts.tolist()))), 2)

    def test_same_person_subclusters_not_split(self):
        """回归：同一人的两个子群（如主歌/副歌音区）心距 ~0.35 < 门槛，
        不得与「不同人 ≥0.39」混淆而切成 3 人（旧 argmax-silhouette 病态）。"""
        s = _Synth(seed=6)
        c_a = _unit(s.rng.standard_normal(_DIM))
        sub_a = s.center_from(c_a, _Synth.SUB_COS)   # 同人子群心距 ≈ 0.32 < 0.38 门槛
        c_b = s.center_from(c_a, _Synth.SPK_COS)     # 真正的第二人
        pts = np.vstack([s.members(sub_a, 15), s.members(c_a, 15),
                         s.members(c_b, 18)])
        labels = _cluster_auto(pts.tolist())
        self.assertEqual(len(_speaker_set(labels)), 2)

    def test_outlier_window_absorbed(self):
        """间奏/杂讯单窗自成一群 → 被吸收进最近大群，不产生假说话者。"""
        s = _Synth(seed=7)
        c1 = _unit(s.rng.standard_normal(_DIM))
        pts = np.vstack([s.members(c1, 20),
                         s.members(s.center_from(c1, _Synth.SPK_COS), 20),
                         _unit(np.ones(_DIM) * 0.02
                               + s.rng.standard_normal(_DIM))[np.newaxis]])
        labels = _cluster_auto(pts.tolist())
        self.assertEqual(len(_speaker_set(labels)), 2)

    def test_few_samples_no_crash(self):
        pts = np.eye(3, _DIM)                 # 3 个近正交样本
        self.assertIn(len(_speaker_set(_cluster_auto(pts.tolist()))), (1, 2, 3))


class TestAbsorbTiny(unittest.TestCase):
    def test_tiny_groups_merged_to_nearest(self):
        s = _Synth(seed=8)
        c1 = _unit(s.rng.standard_normal(_DIM))
        c2 = s.center_from(c1, _Synth.SPK_COS)
        a, b = s.members(c1, 10), s.members(c2, 10)
        tiny = np.stack([_unit(c1 + 0.02 * s.rng.standard_normal(_DIM))
                         for _ in range(2)])        # 贴近 a 的两个碎窗
        emb = np.vstack([a, b, tiny])
        labels = np.array([1] * 10 + [2] * 10 + [3, 3])
        out = _absorb_tiny(labels, emb)
        self.assertEqual(set(out[20:]), {1})          # 并入最近的 a 群
        self.assertEqual(_speaker_set(out), {1, 2})

    def test_no_big_group_unchanged(self):
        labels = np.array([1, 1, 2, 3])               # 没有够大群
        out = _absorb_tiny(labels, np.eye(4, _DIM))
        self.assertTrue((out == labels).all())

    def test_all_big_unchanged(self):
        labels = np.array([1] * 5 + [2] * 5)
        out = _absorb_tiny(labels, np.eye(10, _DIM))
        self.assertTrue((out == labels).all())


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

    def test_threshold_sanity(self):
        # 门槛必须落在「同人子群 ≤0.368 / 不同人 ≥0.39」实测空隙内
        self.assertLess(TINY_CLUSTER_WINDOWS, 5)
        self.assertGreater(SPLIT_SURE_DIST, 0.368)
        self.assertLess(SPLIT_SURE_DIST, 0.39)


if __name__ == "__main__":
    unittest.main()

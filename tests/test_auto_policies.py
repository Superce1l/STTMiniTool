"""设置项自动化后的策略测试：每段秒数任务自适应 + VAD 自动降阈值重试"""
import json
import tempfile
import unittest
from pathlib import Path

from common import *  # noqa: F401,F403

import app as core_app
from app import (_VAD_RETRY_MIN_PROB, _VAD_RETRY_STEPS, _vad_retry_thresholds)
from webview_backend import WebBackend

PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "ov_models" / "silero_vad_v4.onnx"


class _BackendCase(unittest.TestCase):
    """构造固定 settings 的 WebBackend（写自己的夹具，不依赖开发者本机
    的实时 settings.json——它会被 GUI 使用者的真实选择改写）"""

    def setUp(self):
        self._prev = core_app.SETTINGS_FILE
        d = Path(tempfile.mkdtemp())
        core_app.SETTINGS_FILE = d / "settings.json"
        core_app.SETTINGS_FILE.write_text(
            json.dumps({"backend": "openvino", "cpu_model_size": "0.6B"}),
            encoding="utf-8")
        self.be = WebBackend()

    def tearDown(self):
        core_app.SETTINGS_FILE = self._prev


class TestAutoChunkSecs(_BackendCase):
    """每段最长秒数：任务自适应（开对齐→短段抗漂移；关→模型上限）"""

    def test_align_shortens_0p6b(self):
        eng = core_app.ASREngine()                 # 天花板 30s
        self.be.engine = eng
        self.be._apply_chunk_secs(True)
        self.assertEqual(eng.max_chunk_secs, self.be._ALIGN_CHUNK_SECS)
        self.be._apply_chunk_secs(False)
        self.assertEqual(eng.max_chunk_secs, 30)

    def test_ceiling_respected_1p7b(self):
        eng = core_app.ASREngine1p7B()             # 天花板 10s
        self.be.engine = eng
        for align in (True, False):
            self.be._apply_chunk_secs(align)
            self.assertEqual(eng.max_chunk_secs, 10)

    def test_crispasr_noop(self):
        from crisp_engine import CrispWhisperEngine
        eng = CrispWhisperEngine()
        self.be.engine = eng
        self.be._apply_chunk_secs(True)            # 不应抛错、不应注入属性
        self.assertFalse(hasattr(eng, "max_chunk_secs"))

    def test_settings_no_longer_expose_vad_chunk(self):
        s = self.be.get_settings()
        self.assertNotIn("vad", s)
        self.assertNotIn("chunkSecs", s)
        # 旧的设置键被忽略：写入无效字段不会影响转录策略
        self.be.set_settings({"vad": 0.9, "chunkSecs": 5})
        eng = core_app.ASREngine()
        self.be.engine = eng
        self.be._apply_chunk_secs(True)
        self.assertEqual(eng.max_chunk_secs, 10)


class TestVadRetryThresholds(unittest.TestCase):
    """VAD 自动降阈值：真静音不重试；重试阈值必须低于本段最高概率"""

    def test_silence_no_retry(self):
        self.assertEqual(_vad_retry_thresholds({"max_prob": 0.02}), ())
        self.assertEqual(_vad_retry_thresholds({}), ())
        self.assertEqual(_vad_retry_thresholds(None), ())

    def test_retry_steps_below_max_prob(self):
        self.assertEqual(_vad_retry_thresholds({"max_prob": 0.30}), (0.22,))
        self.assertEqual(_vad_retry_thresholds({"max_prob": 0.40}), (0.35, 0.22))
        self.assertEqual(_vad_retry_thresholds({"max_prob": 0.49}), _VAD_RETRY_STEPS)

    def test_min_prob_gate(self):
        self.assertEqual(_vad_retry_thresholds({"max_prob": _VAD_RETRY_MIN_PROB - 0.01}), ())
        self.assertEqual(_vad_retry_thresholds({"max_prob": 0.20}), ())   # 无更低的档位


@unittest.skipUnless(MODEL.exists(), f"缺 VAD 模型：{MODEL}")
class TestVadAutoSegment(unittest.TestCase):
    """_vad_segment_auto：显式阈值参数生效；真静音走「不重试」路径"""

    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        cls.sess = ort.InferenceSession(str(MODEL),
                                        providers=["CPUExecutionProvider"])

    def test_explicit_threshold_changes_stats(self):
        import numpy as np
        rng = np.random.default_rng(5)
        audio = rng.normal(0, 0.02, core_app.SAMPLE_RATE * 4).astype(np.float32)
        stats_hi, stats_lo = {}, {}
        core_app._detect_speech_groups(audio, self.sess, stats=stats_hi,
                                       threshold=0.9)
        core_app._detect_speech_groups(audio, self.sess, stats=stats_lo,
                                       threshold=0.1)
        self.assertEqual(stats_hi["threshold"], 0.9)
        self.assertEqual(stats_lo["threshold"], 0.1)
        self.assertEqual(stats_hi["n_segments"], 0)   # 噪声过不了 0.9

    def test_silence_gives_diag_without_retry(self):
        eng = core_app.ASREngine()
        eng.vad_sess = self.sess
        groups, diag = eng._vad_segment_auto(
            __import__("numpy").zeros(core_app.SAMPLE_RATE * 4, dtype=__import__("numpy").float32))
        self.assertFalse(groups)                      # 失败契约：None（falsy）
        self.assertIn("未检测到人声", diag)


class TestStatusModelFields(_BackendCase):
    """左下角状态点契约：实际加载 vs 当前选择（热切换「待切换」判定）"""

    def test_fresh_backend_no_active(self):
        st = self.be.get_status()
        self.assertIsNone(st["activeModel"])
        self.assertFalse(st["selectionDiffers"])
        self.assertEqual(st["selectedModel"], "OpenVINO · Qwen3-ASR-0.6B")

    def test_loaded_then_switch_selection_differs(self):
        import app as core
        self.be.engine = core.ASREngine()            # ready 标志之外仅需 _remember_active
        self.be.engine.ready = True
        self.be._remember_active("openvino")
        st = self.be.get_status()
        self.assertEqual(st["activeModel"], "OpenVINO · Qwen3-ASR-0.6B")
        self.assertFalse(st["selectionDiffers"])     # 加载的就是所选

        # 换选择（不加载）→ 待切换；activeModel 不变（跑的仍是旧模型）
        self.be.set_model("CrispASR", "Whisper Small")
        st = self.be.get_status()
        self.assertEqual(st["selectedModel"], "CrispASR · Whisper Small")
        self.assertEqual(st["activeModel"], "OpenVINO · Qwen3-ASR-0.6B")
        self.assertTrue(st["selectionDiffers"])

        # 加载完成（身份同步）→ 差异消除
        self.be._remember_active("crispasr")
        self.assertFalse(self.be.get_status()["selectionDiffers"])

    def test_accel_variant_counts_as_differs(self):
        # 加速版本纳入身份（vulkan→cuda 等同换核心）
        import app as core
        self.be.engine = core.ASREngine()
        self.be.engine.ready = True
        self.be._remember_active("crispasr")
        self.be.settings.update({"crisp_variant": "cpu"})
        st = self.be.get_status()
        self.assertTrue(st["selectionDiffers"])


if __name__ == "__main__":
    unittest.main()

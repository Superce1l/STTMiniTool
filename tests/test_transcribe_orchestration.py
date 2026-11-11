"""WebBackend.transcribe 编排（假引擎注入，不加载真模型 / 不碰网络）

覆盖编排层契约：
  • 引擎调用参数（语言归一 / hint→context / out_format 固定 srt）
  • 成功路径（segments + srtPath；字级 rich 优先于 SRT 解析）
  • 非致命降级 warnings 透传
  • 取消 / 未产出（diag）/ 未加载 / 并发忙碌 四条失败路径
  • output_format=txt：另写 .txt 并移除 .srt
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

import app as core
from engine_types import TranscribeResult
from subtitle_lines import write_transcript
from webview_backend import WebBackend

RICH = [
    {"start": 0.0, "end": 1.0, "text": "第一句", "speaker": "说话者1",
     "words": [{"w": "第", "start": 0.0, "end": 0.2}]},
    {"start": 1.2, "end": 2.0, "text": "第二句", "speaker": None, "words": []},
]


class FakeEngine:
    """process_file 契约的最小实现：经共享写出层落盘 SRT，记录调用参数。"""
    max_chunk_secs = 30
    ready = True

    def __init__(self, result: TranscribeResult):
        self.cancel_event = threading.Event()
        self.diar_engine = None
        self._fa_bin = None
        self.use_aligner = False
        self._result = result
        self.calls = []

    def process_file(self, audio_path, **kw):
        self.calls.append({"audio_path": audio_path, **kw})
        r = self._result
        # 取消 / 诊断 / 已产 SRT：原样透传（包装成正常输出会丢失这些标志，
        # 编排层就永远不会走「已取消 / 未产出」分支——第一次真机跑测试时踩过）
        if r.cancelled or r.diag or r.output is not None:
            return r
        lines = [(s["start"], s["end"], s["text"],
                  (f"说话者{s['speaker']}" if isinstance(s.get("speaker"), int) else None))
                 for s in (r.segments_rich or [])]
        out = write_transcript(kw["original_path"], lines, out_format="srt")
        return TranscribeResult(output=out, segments_rich=r.segments_rich,
                                degenerate=r.degenerate, warnings=r.warnings)


class _OrchestrationCase(unittest.TestCase):
    def setUp(self):
        self._prev_srt = core.SRT_DIR
        self._prev_settings = core.SETTINGS_FILE
        self.d = Path(tempfile.mkdtemp()) / "subtitles"
        self.d.mkdir(parents=True)
        core.SRT_DIR = self.d
        core.SETTINGS_FILE = Path(tempfile.mkdtemp()) / "settings.json"
        core.SETTINGS_FILE.write_text("{}", encoding="utf-8")
        self.be = WebBackend()
        self.be._active_label = ("OpenVINO", "Qwen3-ASR-0.6B")
        self.audio = self.d.parent / "唯一.flac"
        self.audio.write_bytes(b"RIFF....WAVE")   # 假引擎不读内容，仅需存在且非视频扩展名

    def tearDown(self):
        core.SRT_DIR = self._prev_srt
        core.SETTINGS_FILE = self._prev_settings

    def _transcribe(self, **overrides):
        opts = {"path": str(self.audio), "language": "", "hint": "",
                "diarize": False, "align": False}
        opts.update(overrides)
        return self.be.transcribe(opts)

    def _ok_result(self, warnings=None):
        return TranscribeResult(segments_rich=[dict(s) for s in RICH],
                                warnings=warnings)


class TestTranscribeOrchestration(_OrchestrationCase):
    def test_success_segments_and_srt_path(self):
        self.be.engine = FakeEngine(self._ok_result())
        res = self._transcribe()
        self.assertEqual(len(res["segments"]), 2)
        self.assertEqual(res["segments"][0]["text"], "第一句")
        self.assertEqual(res["segments"][0]["speaker"], 1)     # 原始标签正规化为整数
        self.assertTrue(Path(res["srtPath"]).exists())
        self.assertNotIn("warnings", res)

    def test_engine_call_contract(self):
        self.be.engine = FakeEngine(self._ok_result())
        self._transcribe(language="auto", hint="人名、术语", align=True)
        kw = self.be.engine.calls[0]
        self.assertIsNone(kw["language"])                      # auto → None
        self.assertEqual(kw["context"], "人名、术语")
        self.assertEqual(kw["out_format"], "srt")              # 端点固定产 SRT
        self.assertEqual(kw["original_path"].parent, core.SRT_DIR)

    def test_warnings_passthrough(self):
        self.be.engine = FakeEngine(
            self._ok_result(warnings=["2 段字级对齐失败，已用比例估算时间轴"]))
        res = self._transcribe()
        self.assertEqual(res["warnings"],
                         ["2 段字级对齐失败，已用比例估算时间轴"])

    def test_output_format_txt_replaces_srt(self):
        self.be.engine = FakeEngine(self._ok_result())
        self.be.settings.update({"output_format": "txt"})
        res = self._transcribe()
        self.assertTrue(res["srtPath"].endswith(".txt"))
        self.assertTrue(Path(res["srtPath"]).exists())
        srt_counterpart = Path(res["srtPath"]).with_suffix(".srt")
        self.assertFalse(srt_counterpart.exists())             # SRT 已被移除

    def test_cancelled_raises(self):
        self.be.engine = FakeEngine(TranscribeResult(cancelled=True))
        with self.assertRaisesRegex(RuntimeError, "已取消"):
            self._transcribe()

    def test_no_output_raises_diag(self):
        self.be.engine = FakeEngine(TranscribeResult(diag="⚠ 全程静音"))
        with self.assertRaisesRegex(RuntimeError, "全程静音"):
            self._transcribe()

    def test_engine_not_ready_rejected(self):
        eng = FakeEngine(self._ok_result())
        eng.ready = False
        self.be.engine = eng
        with self.assertRaisesRegex(RuntimeError, "尚未加载"):
            self._transcribe()

    def test_concurrent_second_request_rejected(self):
        self.be.engine = FakeEngine(self._ok_result())
        self.be._transcribe_gate.acquire()      # 模拟已有任务占住忙碌门
        try:
            with self.assertRaisesRegex(RuntimeError, "已有识别任务"):
                self._transcribe()
        finally:
            self.be._transcribe_gate.release()

    def test_gate_released_after_failure(self):
        self.be.engine = FakeEngine(TranscribeResult(diag="无输出"))
        with self.assertRaises(RuntimeError):
            self._transcribe()
        # 失败后忙碌门必须回到可 acquisition 状态（置位者清理）
        self.assertTrue(self.be._transcribe_gate.acquire(blocking=False))
        self.be._transcribe_gate.release()

    def test_missing_audio_rejected(self):
        self.be.engine = FakeEngine(self._ok_result())
        with self.assertRaisesRegex(RuntimeError, "找不到音频"):
            self._transcribe(path=str(self.d.parent / "不存在.flac"))


if __name__ == "__main__":
    unittest.main()

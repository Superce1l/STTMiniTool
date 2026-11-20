"""实时分段（常驻 --server）契约测试

覆盖：
  • _split_realtime_lines / _realtime_segments_from_api 的切句与时间轴退化
  • transcribe_realtime 走常驻服务时的字段构造（语言/提示/格式）
  • 常驻服务出错 → 本段退回一次性 spawn（不丢字），且坏进程被停掉
  • 常驻服务起不来 → 只试一次，之后永久退回 spawn（不反复等超时）
  • 编排层 realtime 分支：只回 segments、不写字幕文件、批量转录前释放显存
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

import app as core
import crisp_engine
from engine_types import TranscribeResult
from webview_backend import WebBackend

SRT = "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n"


# ── 常驻服务替身 ─────────────────────────────────────────────────────
class _FakeServer:
    """记录构造参数与请求字段；默认返回两段 JSON。"""

    def __init__(self, argv, cwd):
        self.argv = list(argv)
        self.cwd = cwd
        self.calls = []
        self.started = 0
        self.stopped = 0

    def start(self, timeout=None):
        self.started += 1

    def alive(self):
        return True

    def tail(self, n=6):
        return ""

    def stop(self):
        self.stopped += 1

    def transcribe_file(self, path, fields=None, timeout=None):
        self.calls.append({"path": str(path), **(fields or {})})
        return {"segments": [{"start": 0.0, "end": 1.2, "text": "你好。"},
                             {"start": 1.2, "end": 2.0, "text": "世界"}],
                "text": "你好。世界"}


class _BrokenServer(_FakeServer):
    def transcribe_file(self, path, fields=None, timeout=None):
        raise RuntimeError("server boom")


class _StartFailServer(_FakeServer):
    starts = 0

    def start(self, timeout=None):
        _StartFailServer.starts += 1
        raise RuntimeError("这份核心不带 --server")


def _engine() -> crisp_engine.CrispWhisperEngine:
    eng = crisp_engine.CrispWhisperEngine()
    eng._exe = Path("C:/nonexistent/crispasr.exe")
    eng._model_path = Path("C:/nonexistent/ggml-base.bin")
    eng._backend = "whisper"
    eng._gpu_backend = "vulkan"
    eng._device_id = 0
    eng._threads = 4
    return eng


class TestRealtimeParsing(unittest.TestCase):
    def test_split_sentences_keeps_chinese_and_english(self):
        lines = crisp_engine._split_realtime_lines("你好。今天天气不错！Hello there. 再见")
        self.assertEqual(lines, ["你好。", "今天天气不错！", "Hello there.", "再见"])

    def test_split_empty(self):
        self.assertEqual(crisp_engine._split_realtime_lines("   "), [])

    def test_segments_keep_api_timeline(self):
        segs = crisp_engine._realtime_segments_from_api(
            {"segments": [{"start": 1.0, "end": 2.5, "text": "第一句。第二句"}]})
        self.assertEqual([s["text"] for s in segs], ["第一句。", "第二句"])
        self.assertEqual([s["start"] for s in segs], [1.0, 1.0])
        self.assertEqual([s["end"] for s in segs], [2.5, 2.5])

    def test_segments_fall_back_to_text(self):
        segs = crisp_engine._realtime_segments_from_api({"text": "只有文字。没有分段"})
        self.assertEqual([s["text"] for s in segs], ["只有文字。", "没有分段"])
        self.assertTrue(all(s["start"] == 0.0 for s in segs))


class TestTranscribeRealtime(unittest.TestCase):
    def test_uses_server_with_language_and_prompt(self):
        with patch.object(crisp_engine, "_CrispServer", _FakeServer):
            eng = _engine()
            res = eng.transcribe_realtime(Path("seg.webm"), language="Chinese",
                                          context="人名、术语")
            call = eng._srv.calls[0]
        self.assertEqual([s["text"] for s in res.segments_rich], ["你好。", "世界"])
        self.assertEqual(call["language"], "zh")            # 语言名 → whisper 代码
        self.assertEqual(call["prompt"], "人名、术语")
        self.assertEqual(call["response_format"], "verbose_json")

    def test_server_failure_falls_back_to_spawn(self):
        with patch.object(crisp_engine, "_CrispServer", _BrokenServer):
            eng = _engine()
            eng._realtime_spawn_text = lambda *a, **k: "退回成功。第二句"
            res = eng.transcribe_realtime(Path("seg.webm"), language="Chinese")
            stopped = eng._srv is None
        self.assertEqual([s["text"] for s in res.segments_rich], ["退回成功。", "第二句"])
        self.assertTrue(stopped)                            # 坏进程已丢弃

    def test_start_failure_only_tried_once(self):
        _StartFailServer.starts = 0
        with patch.object(crisp_engine, "_CrispServer", _StartFailServer):
            eng = _engine()
            eng._realtime_spawn_text = lambda *a, **k: "后备。"
            eng.transcribe_realtime(Path("a.webm"), language="Chinese")
            eng.transcribe_realtime(Path("b.webm"), language="Chinese")
            failed = eng._srv_failed
        self.assertTrue(failed)
        self.assertEqual(_StartFailServer.starts, 1)

    def test_server_argv_matches_model_and_device(self):
        eng = _engine()
        argv = eng._server_argv()
        self.assertEqual(argv[0], str(eng._exe))
        self.assertIn("-m", argv)
        self.assertIn(str(eng._model_path), argv)
        self.assertIn("--gpu-backend", argv)
        self.assertNotIn("--server", argv)      # 由 _CrispServer.start() 追加

    def test_empty_result_reports_diag(self):
        with patch.object(crisp_engine, "_CrispServer", _FakeServer):
            eng = _engine()
            eng._srv = None
            eng.server_mode = True
            srv = _FakeServer([], ".")
            srv.transcribe_file = lambda *a, **k: {"text": "", "segments": []}
            with patch.object(crisp_engine, "_CrispServer", lambda *a, **k: srv):
                res = eng.transcribe_realtime(Path("silence.webm"))
        self.assertFalse(res.segments_rich)
        self.assertIn("未识别", res.diag)


class _FakeRTEngine:
    ready = True
    max_chunk_secs = 30

    def __init__(self):
        self.cancel_event = threading.Event()
        self.diar_engine = None
        self._fa_bin = None
        self.use_aligner = False
        self.calls = []
        self.released = 0

    def process_file(self, *a, **kw):
        raise AssertionError("realtime 请求不应走 process_file")

    def release_realtime(self):
        self.released += 1

    def transcribe_realtime(self, path, language=None, context=None):
        self.calls.append({"path": str(path), "language": language,
                           "context": context})
        return TranscribeResult(segments_rich=[
            {"start": 0.0, "end": 1.0, "text": "第一句", "speaker": None, "words": []},
            {"start": 1.0, "end": 1.0, "text": "   ", "speaker": None, "words": []},
        ])


class _FakeFileEngine:
    """非实时路径：只验证批量转录前会释放常驻服务。"""

    ready = True
    max_chunk_secs = 30

    def __init__(self):
        self.cancel_event = threading.Event()
        self.diar_engine = None
        self._fa_bin = None
        self.use_aligner = False
        self.released = 0

    def release_realtime(self):
        self.released += 1

    def process_file(self, audio_path, **kw):
        out = kw["original_path"]
        out.write_text(SRT, encoding="utf-8")
        return TranscribeResult(output=out, segments_rich=[])


class TestRealtimeOrchestration(unittest.TestCase):
    def setUp(self):
        self._prev_srt = core.SRT_DIR
        self._prev_settings = core.SETTINGS_FILE
        self.d = Path(tempfile.mkdtemp()) / "subtitles"
        self.d.mkdir(parents=True)
        core.SRT_DIR = self.d
        core.SETTINGS_FILE = Path(tempfile.mkdtemp()) / "settings.json"
        core.SETTINGS_FILE.write_text("{}", encoding="utf-8")
        self.be = WebBackend()
        self.be._active_label = ("CrispASR", "Whisper Base")
        self.audio = self.d.parent / "recording.webm"
        self.audio.write_bytes(b"RIFF....WAVE")
        # 非实时路径（本文件只有一个用例）用 wav：webm 属视频扩展名，会先走
        # ffmpeg 抽音轨，而测试环境不保证有 ffmpeg。
        self.audio_wav = self.d.parent / "clip.wav"
        self.audio_wav.write_bytes(b"RIFF....WAVE")

    def tearDown(self):
        core.SRT_DIR = self._prev_srt
        core.SETTINGS_FILE = self._prev_settings

    def _transcribe(self, **overrides):
        opts = {"path": str(self.audio), "language": "", "hint": "",
                "diarize": False, "align": False}
        opts.update(overrides)
        return self.be.transcribe(opts)

    def test_realtime_returns_segments_without_artifacts(self):
        eng = _FakeRTEngine()
        self.be.engine = eng
        res = self._transcribe(realtime=True, language="auto", hint="术语")
        self.assertTrue(res["realtime"])
        self.assertIsNone(res["srtPath"])
        self.assertEqual([s["text"] for s in res["segments"]], ["第一句"])
        self.assertEqual(eng.calls[0]["language"], None)       # auto → None
        self.assertEqual(eng.calls[0]["context"], "术语")
        self.assertEqual(list(self.d.iterdir()), [])           # 不落字幕文件
        self.assertTrue(self.be._transcribe_gate.acquire(blocking=False))
        self.be._transcribe_gate.release()

    def test_realtime_empty_result_raises(self):
        eng = _FakeRTEngine()
        eng.transcribe_realtime = lambda *a, **k: TranscribeResult(diag="全程静音")
        self.be.engine = eng
        with self.assertRaisesRegex(RuntimeError, "全程静音"):
            self._transcribe(realtime=True)

    def test_file_job_releases_resident_service(self):
        eng = _FakeFileEngine()
        self.be.engine = eng
        res = self._transcribe(path=str(self.audio_wav))
        self.assertEqual(eng.released, 1)      # 批量转录前停常驻服务，让出显存
        self.assertTrue(Path(res["srtPath"]).exists())


class TestServerLifecycle(unittest.TestCase):
    """常驻服务自身的启动失败路径与录音结束后的宽限释放。"""

    def test_start_failure_does_not_deadlock(self):
        """进程秒退 → start() 必须报错返回（曾因 Lock 非重入自锁死）。"""
        import sys as _sys
        srv = crisp_engine._CrispServer(
            [_sys.executable, "-c", "import sys; sys.exit(1)"], ".")
        box = {}

        def _go():
            try:
                srv.start(timeout=5)
                box["result"] = "started"
            except Exception as e:
                box["result"] = type(e).__name__
                box["msg"] = str(e)

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        t.join(15)
        self.assertFalse(t.is_alive(), "start() 在失败路径上卡住（锁不可重入）")
        self.assertEqual(box.get("result"), "RuntimeError")
        self.assertIn("启动即退出", box.get("msg", ""))


class _RecordingEngine:
    ready = True

    def __init__(self):
        self.prepared = 0
        self.released = 0
        self.cancel_event = threading.Event()

    def prepare_realtime(self):
        self.prepared += 1

    def release_realtime(self):
        self.released += 1


class TestRecordingHooks(unittest.TestCase):
    def setUp(self):
        self._prev_settings = core.SETTINGS_FILE
        core.SETTINGS_FILE = Path(tempfile.mkdtemp()) / "settings.json"
        core.SETTINGS_FILE.write_text("{}", encoding="utf-8")
        self.be = WebBackend()
        self.eng = _RecordingEngine()
        self.be.engine = self.eng

    def tearDown(self):
        self.be._cancel_rt_release()
        core.SETTINGS_FILE = self._prev_settings

    def test_record_start_warms_server(self):
        self.be.set_recording(True)
        self.assertEqual(self.eng.prepared, 1)
        self.be._cancel_rt_release()

    def test_record_stop_defers_release_for_last_segment(self):
        self.be.set_recording(True)
        self.be.set_recording(False)
        self.assertEqual(self.eng.released, 0)      # 不立刻停（收尾段还要上传）
        self.be._release_rt_if_idle()
        self.assertEqual(self.eng.released, 1)

    def test_deferred_release_skipped_while_still_recording(self):
        self.be.set_recording(True)
        self.be._release_rt_if_idle()
        self.assertEqual(self.eng.released, 0)      # 还在录音 → 保留常驻
        self.be._cancel_rt_release()


if __name__ == "__main__":
    unittest.main()

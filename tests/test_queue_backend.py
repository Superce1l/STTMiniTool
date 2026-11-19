"""WebBackend 队列层：入队 / spool / 结果缓存 / 暂停恢复 / 清空。

不起 HTTP、不加载模型：用 __new__ 装配一个只含队列字段的后端，transcribe
替换为普通函数，即可覆盖 /api/enqueue 背后的全部语义。

（沙箱环境的系统临时目录可能不可写：设 STT_TEST_TMP 指向可写目录即可。）
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

from transcription_queue import TaskQueue
from webview_backend import WebBackend


def wait_until(cond, timeout=10.0, msg="condition not met"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError(msg)


def _bare_backend(root: Path, sink: list):
    """只装配队列相关字段的 WebBackend（不碰引擎/模型/设置）。"""
    be = WebBackend.__new__(WebBackend)
    be.engine = object()                 # cancel() 会 getattr(engine, "cancel_event")
    be._cancel = False
    be._on_event = lambda ev, payload: sink.append((ev, payload))
    be._queue_spool = Path(root) / "queue-spool"
    be._queue_archive = Path(root) / "queue-archive"
    be._job_spool = {}
    be._job_results = {}
    be._archive = {}
    be._tx_queue = TaskQueue(on_event=be._queue_event)
    return be


class QueueBackendCase(unittest.TestCase):
    def setUp(self):
        base = os.environ.get("STT_TEST_TMP") or None
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)   # 沙箱专用可写根
        self.root = Path(tempfile.mkdtemp(prefix="stt_queue_", dir=base))
        self.events = []
        self.be = _bare_backend(self.root, self.events)

    def tearDown(self):
        try:
            self.be.close_queue()
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)

    # ── helpers ──────────────────────────────────────────────
    def _upload(self, name="clip.wav", data=b"RIFFdata"):
        src = self.root / ("up_" + name)
        src.write_bytes(data)
        return self.be.enqueue_from_upload(src, name, {"align": "1"})

    def _state(self, jid):
        for j in self.be.queue_status()["jobs"]:
            if j["id"] == jid:
                return j["state"]
        return None

    def _spool_dirs(self):
        root = self.be.queue_spool_root()
        return [p for p in root.glob("*")] if root.exists() else []

    # ── 用例 ────────────────────────────────────────────────
    def test_upload_moves_to_spool_runs_and_cleans(self):
        seen = []

        def fake(opts, progress_cb=None):
            seen.append(opts["name"])
            Path(opts["path"]).read_bytes()       # 任务能读到换进来的文件
            if progress_cb:
                progress_cb(50, "识别中")
            return {"segments": [{"start": 0, "end": 1, "text": "hi"}],
                    "srtPath": "subtitles/x.srt"}

        self.be.transcribe = fake
        jid = self._upload("a.wav")
        wait_until(lambda: self._state(jid) == "finished", msg="job did not finish")
        self.assertEqual(seen, ["a.wav"])
        self.assertEqual(self.be.queue_result(jid)["segments"][0]["text"], "hi")
        wait_until(lambda: self._spool_dirs() == [], msg="spool not cleaned")
        self.assertIsNone(self.be.queue_result("nope"))

    def test_progress_is_forwarded_with_job_id(self):
        def fake(opts, progress_cb=None):
            if progress_cb:
                progress_cb(42, "识别中")
            return {"segments": [], "srtPath": ""}

        self.be.transcribe = fake
        jid = self._upload()
        wait_until(lambda: self._state(jid) == "finished")
        prog = [p for ev, p in self.events if ev == "progress"]
        self.assertTrue(prog and all(p.get("jobId") == jid for p in prog), prog)

    def test_failure_marks_failed_and_cleans_spool(self):
        def boom(opts, progress_cb=None):
            raise RuntimeError("引擎炸了")

        self.be.transcribe = boom
        jid = self._upload()
        wait_until(lambda: self._state(jid) == "failed")
        wait_until(lambda: self._spool_dirs() == [], msg="failed spool not cleaned")

    def test_pause_keeps_spool_then_resume_finishes(self):
        gate = threading.Event()

        def slow(opts, progress_cb=None):
            gate.wait(timeout=10)
            return {"segments": [], "srtPath": ""}

        self.be.transcribe = slow
        jid = self._upload("slow.wav")
        wait_until(lambda: self.be.queue_status()["current"] == jid, msg="not running")

        self.be.queue_pause()
        gate.set()
        wait_until(lambda: self._state(jid) == "paused", msg="not paused")
        self.assertTrue(self._spool_dirs())          # 暂停 → 输入保留（可恢复）

        self.assertEqual(self.be.queue_resume()["resumed"], 1)
        wait_until(lambda: self._state(jid) == "finished", msg="not resumed")
        wait_until(lambda: self._spool_dirs() == [], msg="spool not cleaned")

    def test_cancel_and_clear_finished(self):
        gate = threading.Event()

        def slow(opts, progress_cb=None):
            gate.wait(timeout=10)
            return {"segments": [], "srtPath": ""}

        self.be.transcribe = slow
        jid = self._upload("cancel.wav")
        wait_until(lambda: self.be.queue_status()["current"] == jid)
        self.assertTrue(self.be.cancel_job(jid))
        gate.set()
        wait_until(lambda: self._state(jid) in ("cancelled", "finished"))
        wait_until(lambda: self.be.queue_status()["current"] is None)

        cleared = self.be.queue_clear_finished()
        self.assertGreaterEqual(cleared["removed"], 1)
        self.assertEqual(cleared["queue"]["jobs"], [])

    def test_client_gone_pauses_only_when_running(self):
        self.be.queue_on_client_gone()               # 空闲 → 不动作
        self.assertFalse(self.be.queue_status()["paused"])

        gate = threading.Event()

        def slow(opts, progress_cb=None):
            gate.wait(timeout=10)
            return {"segments": [], "srtPath": ""}

        self.be.transcribe = slow
        jid = self._upload("bg.wav")
        wait_until(lambda: self.be.queue_status()["current"] == jid)
        self.be.queue_on_client_gone()
        gate.set()
        wait_until(lambda: self._state(jid) == "paused")
        self.assertTrue(self.be.queue_status()["paused"])

    # ── 持久化 / 重试 / 排序 / 同步接口 ─────────────────────
    def _restart(self):
        """在同一磁盘目录上再建一个后端（模拟重启）。"""
        be2 = _bare_backend(self.root, [])
        be2._restore_queue()
        return be2

    def test_finished_job_survives_restart(self):
        self.be.transcribe = lambda opts, progress_cb=None: {
            "segments": [{"start": 0, "end": 1, "text": "hi"}], "srtPath": "x.srt"}
        jid = self._upload("keep.wav")
        wait_until(lambda: self._state(jid) == "finished")
        wait_until(lambda: self._spool_dirs() == [])

        be2 = self._restart()
        jobs = {j["id"]: j for j in be2.queue_status()["jobs"]}
        self.assertIn(jid, jobs)
        self.assertEqual(jobs[jid]["state"], "finished")
        self.assertTrue(jobs[jid]["hasResult"])
        self.assertTrue(jobs[jid]["hasAudio"])
        self.assertEqual(be2.queue_result(jid)["segments"][0]["text"], "hi")
        self.assertIsNotNone(be2.queue_audio_path(jid))
        be2.close_queue()

    def test_pending_job_restored_paused(self):
        d = self.be.new_spool_dir()
        (d / "input.wav").write_bytes(b"RIFFdata")
        self.be._write_spool_meta(d, {"name": "p.wav", "language": None},
                                  "p.wav", {"name": "p.wav", "size": 8}, "input.wav")
        be2 = self._restart()
        st = be2.queue_status()
        self.assertTrue(st["paused"])
        self.assertEqual(len(st["jobs"]), 1)
        self.assertEqual(st["jobs"][0]["meta"]["name"], "p.wav")
        self.assertEqual(st["jobs"][0]["state"], "queued")
        be2.close_queue()

    def test_retry_failed_job_reuses_input(self):
        def boom(opts, progress_cb=None):
            raise RuntimeError("boom")

        self.be.transcribe = boom
        jid = self._upload("bad.wav")
        wait_until(lambda: self._state(jid) == "failed")
        # 归档由队列事件异步完成，等输入文件进档再重试
        wait_until(lambda: jid in self.be._archive, msg="not archived")
        self.be.transcribe = lambda opts, progress_cb=None: {
            "segments": [{"start": 0, "end": 1, "text": "retried"}], "srtPath": "y.srt"}
        r = self.be.queue_retry(jid)
        self.assertTrue(r["ok"], r)
        wait_until(lambda: self._state(r["jobId"]) == "finished")
        self.assertEqual(self.be.queue_result(r["jobId"])["segments"][0]["text"], "retried")

    def test_reorder_queued_jobs(self):
        gate = threading.Event()

        def slow(opts, progress_cb=None):
            gate.wait(timeout=5)
            return {"segments": [], "srtPath": ""}

        self.be.transcribe = slow
        blocker = self._upload("blocker.wav")
        wait_until(lambda: self.be.queue_status()["current"] == blocker)
        j1 = self._upload("one.wav")
        j2 = self._upload("two.wav")
        r = self.be.queue_reorder(j2, j1)          # two 挪到 one 之前
        self.assertTrue(r["ok"], r)
        order = [j["id"] for j in r["queue"]["jobs"] if j["state"] == "queued"]
        self.assertEqual(order, [j2, j1])
        gate.set()

    def test_transcribe_via_queue_returns_result_and_stays_hidden(self):
        self.be.transcribe = lambda opts, progress_cb=None: {
            "segments": [{"start": 0, "end": 1, "text": "sync"}], "srtPath": "z.srt"}
        seen = []
        res = self.be.transcribe_via_queue({"name": "live.wav", "path": "x"},
                                           on_job=seen.append)
        self.assertEqual(res["segments"][0]["text"], "sync")
        self.assertEqual(len(seen), 1)
        jid = seen[0]
        wait_until(lambda: jid not in [j["id"] for j in self.be.queue_status()["jobs"]],
                   msg="hidden job not dropped")
        self.assertNotIn(jid, self.be._archive)


if __name__ == "__main__":
    unittest.main()

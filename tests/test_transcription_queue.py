"""transcription_queue.TaskQueue：顺序 / 事件 / 失败 / 取消 / 上限 / 关闭

全部用普通函数作任务体（不碰磁盘与引擎），可在任何环境运行。
等待一律用条件轮询（上限 5s），不依赖脆弱的时序假设。
"""
import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

from transcription_queue import TaskQueue


def wait_until(cond, timeout=5.0, msg="condition not met"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    raise AssertionError(msg)


class _QCase(unittest.TestCase):
    def setUp(self):
        self.events = []
        self._ev_lock = threading.Lock()
        self.q = TaskQueue(on_event=lambda kind, job: self._rec(kind, job))

    def _rec(self, kind, job):
        with self._ev_lock:
            self.events.append((kind, job["id"], job["state"]))

    def _events_for(self, jid):
        with self._ev_lock:
            return [k for k, jid2, _s in self.events if jid2 == jid]

    def tearDown(self):
        self.q.close()


class TestTaskQueue(_QCase):
    def test_fifo_order_and_event_sequence(self):
        done, lk = [], threading.Lock()

        def mk(tag):
            return lambda ce: (with_lock(lk, done.append, tag), tag)[1]

        ja = self.q.submit(mk("a"), label="a")
        jb = self.q.submit(mk("b"), label="b")
        jc = self.q.submit(mk("c"), label="c")
        wait_until(lambda: len(done) == 3, msg="not all jobs finished")
        self.assertEqual(done, ["a", "b", "c"])           # 严格 FIFO
        for jid in (ja, jb, jc):
            self.assertEqual(self._events_for(jid),
                             ["queued", "started", "finished"])

    def test_failed_job_records_error_and_queue_continues(self):
        done, lk = [], threading.Lock()

        def boom(ce):
            raise ValueError("引擎炸了")

        def ok(ce):
            with_lock(lk, done.append, "ok")
            return 42

        ja = self.q.submit(boom, label="bad")
        jb = self.q.submit(ok, label="good")
        wait_until(lambda: len(done) == 1, msg="second job never ran")
        st = {j["id"]: j for j in self.q.status()}
        self.assertEqual(st[ja]["state"], "failed")
        self.assertIn("ValueError", st[ja]["error"])
        self.assertEqual(st[jb]["state"], "finished")
        self.assertEqual(self._events_for(ja), ["queued", "started", "failed"])

    def test_cancel_queued_job_skips_execution(self):
        gate = threading.Event()
        ran, lk = [], threading.Lock()

        def blocker(ce):
            gate.wait(timeout=5)
            return "first"

        def never(ce):
            with_lock(lk, ran.append, "never")
            return "second"

        self.q.submit(blocker, label="first")     # 占住 worker
        jb = self.q.submit(never, label="second")
        wait_until(lambda: self.q.current_id() is not None, msg="worker idle")
        self.assertTrue(self.q.cancel(jb))
        gate.set()
        wait_until(lambda: self.q.current_id() is None, msg="first not done")
        wait_until(lambda: {j["id"]: j["state"] for j in self.q.status()}[jb]
                   == "cancelled", msg="b state")
        self.assertEqual(ran, [])                 # 从未执行
        self.assertEqual(self._events_for(jb), ["queued", "cancelled"])

    def test_cancel_running_job_signals_cancel_event(self):
        seen = []

        def slow(ce):
            ok = ce.wait(timeout=5)
            seen.append(ok)
            return "interrupted" if ok else "done"

        ja = self.q.submit(slow, label="slow")
        wait_until(lambda: self.q.current_id() == ja, msg="job not started")
        self.assertTrue(self.q.cancel(ja))        # 运行中 → 置位 cancel_event
        wait_until(lambda: bool(seen), msg="job not finished")
        self.assertEqual(seen, [True])            # 协作式取消信号送达

    def test_queue_full_rejects(self):
        gate = threading.Event()
        q2 = TaskQueue(maxsize=1)
        self.addCleanup(q2.close)
        q2.submit(lambda ce: gate.wait(timeout=5), label="running")   # 占住 worker
        wait_until(lambda: q2.current_id() is not None)
        q2.submit(lambda ce: "queued", label="fill")                  # 队列 1/1
        with self.assertRaisesRegex(RuntimeError, "已满"):
            q2.submit(lambda ce: "over", label="over")
        gate.set()

    def test_status_snapshot_is_json_serializable(self):
        self.q.submit(lambda ce: "x", label="one")
        wait_until(lambda: not self.q.current_id(), msg="not finished")
        snap = self.q.status()
        json.dumps(snap, ensure_ascii=False)      # 不抛 = 可序列化
        for j in snap:
            self.assertNotIn("fn", j)
            self.assertNotIn("cancel_event", j)

    def test_close_stops_worker_thread(self):
        self.q.submit(lambda ce: "x", label="only")
        wait_until(lambda: not self.q.current_id())
        self.q.close(wait=True)
        wait_until(lambda: self.q._thread is None or not self.q._thread.is_alive(),
                   msg="worker thread did not exit")
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            self.q.submit(lambda ce: "x")          # 关闭后拒绝新任务


def with_lock(lk, fn, *a):
    with lk:
        return fn(*a)


if __name__ == "__main__":
    unittest.main()

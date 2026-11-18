"""instance_lock：跨进程互斥、产物锁、单实例守卫、原子写临时名唯一

覆盖反馈 #6（同时开两个实例互相打架）对应的三个原语：
  • FileLock.named / for_artifact —— 跨进程互斥（含持有者被杀后自动释放）
  • SingleInstance —— 第二个实例拿不到锁，且能读到对方的 pid/端口/URL
  • atomic_write_text —— 并发写同一目标不再撞固定名 .tmp
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from common import *  # noqa: F401,F403

import instance_lock as IL
from subtitle_lines import atomic_write_text

ROOT = str(Path(__file__).resolve().parents[1])


class _StateDirMixin(unittest.TestCase):
    """每个测试一个独立 state 目录（STTMINITOOL_STATE_DIR 覆盖）。"""

    def setUp(self):
        self._old = os.environ.get(IL._ENV_STATE_DIR)
        self.dir = Path(tempfile.mkdtemp())
        os.environ[IL._ENV_STATE_DIR] = str(self.dir)

    def tearDown(self):
        if self._old is None:
            os.environ.pop(IL._ENV_STATE_DIR, None)
        else:
            os.environ[IL._ENV_STATE_DIR] = self._old
        shutil.rmtree(self.dir, ignore_errors=True)

    # 子进程用同一份环境（含 state 目录覆盖）
    def _spawn(self, code: str) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-c", code, ROOT],
                                stdout=subprocess.PIPE, text=True,
                                env=dict(os.environ))


class TestFileLock(_StateDirMixin):
    def test_state_dir_override(self):
        self.assertTrue(str(IL.state_dir()).startswith(str(self.dir)))

    def test_second_lock_on_same_name_blocked(self):
        a = IL.FileLock.named("res")
        self.assertTrue(a.acquire())
        try:
            b = IL.FileLock.named("res", timeout=0.1)
            self.assertFalse(b.acquire())
        finally:
            a.release()
        c = IL.FileLock.named("res", timeout=0.5)
        self.assertTrue(c.acquire())
        c.release()

    def test_artifact_locks_are_per_path(self):
        p1, p2 = self.dir / "a.srt", self.dir / "b.srt"
        l1 = IL.FileLock.for_artifact(p1)
        l2 = IL.FileLock.for_artifact(p2)
        self.assertTrue(l1.acquire())
        self.assertTrue(l2.acquire())          # 不同产物互不影响
        dup = IL.FileLock.for_artifact(p1, timeout=0.1)
        self.assertFalse(dup.acquire())        # 同一产物只能一个持有
        l1.release()
        self.assertTrue(dup.acquire())
        dup.release()
        l2.release()

    def test_cross_process_lock(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]);"
                "import instance_lock as IL;"
                "l = IL.FileLock.named('cp');"
                "assert l.acquire();"
                "print('READY', flush=True);"
                "import time; time.sleep(30)")
        p = self._spawn(code)
        try:
            self.assertEqual(p.stdout.readline().strip(), "READY")
            self.assertFalse(IL.FileLock.named("cp", timeout=0.2).acquire())
        finally:
            p.kill()
            p.wait(timeout=10)

    def test_unwritable_lock_is_not_reported_as_running(self):
        # 锁路径是个目录 → open 失败：必须区分「打不开」与「已被占用」，
        # 否则 state 目录不可写时程序会误报「已有实例在运行」。
        bad = IL.state_dir() / "locks" / "name-instance-bad.lock"
        bad.mkdir(parents=True)
        si = IL.SingleInstance("bad")
        self.assertFalse(si.acquire())
        self.assertTrue(si.lock.unavailable)   # 打不开 ≠ 已被占用
        self.assertFalse(si.already_running)

    def test_lock_released_when_holder_dies(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]);"
                "import instance_lock as IL;"
                "l = IL.FileLock.named('dead');"
                "assert l.acquire();"
                "print('READY', flush=True);"
                "import time; time.sleep(30)")
        p = self._spawn(code)
        self.assertEqual(p.stdout.readline().strip(), "READY")
        p.kill()                                # 模拟第二个实例崩溃/被结束
        p.wait(timeout=10)
        lk = IL.FileLock.named("dead", timeout=3.0)
        self.assertTrue(lk.acquire())           # 内核已释放，不会留死锁
        lk.release()


class TestSingleInstance(_StateDirMixin):
    def test_second_instance_reads_first_state(self):
        a = IL.SingleInstance("t")
        self.assertTrue(a.acquire())
        try:
            a.write_state(pid=os.getpid(), port=1234, url="http://127.0.0.1:1234/")
            b = IL.SingleInstance("t")
            self.assertFalse(b.acquire())
            self.assertTrue(b.already_running)
            st = b.read_state()
            self.assertEqual(st.get("port"), 1234)
            self.assertEqual(st.get("url"), "http://127.0.0.1:1234/")
        finally:
            a.release()
        self.assertEqual(b.read_state(), {})     # release 清掉 state 文件
        c = IL.SingleInstance("t")
        self.assertTrue(c.acquire())
        c.release()


class TestAtomicWriteConcurrent(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.target = self.dir / "out.srt"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_concurrent_same_target_no_crash_and_no_tmp_left(self):
        texts = [f"{i}\n" + ("x" * 5000) + f"\n{i}" for i in range(8)]
        errs = []

        def w(t):
            try:
                for _ in range(20):
                    atomic_write_text(self.target, t)
            except Exception as e:              # 旧实现会在这里 FileNotFoundError
                errs.append(repr(e))

        ts = [threading.Thread(target=w, args=(t,)) for t in texts]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        self.assertEqual(errs, [])
        self.assertIn(self.target.read_text(encoding="utf-8"), texts)
        leftovers = [p.name for p in self.dir.iterdir() if p.name != self.target.name]
        self.assertEqual(leftovers, [])


class TestUniqueArtifact(_StateDirMixin):
    """同一份字幕两个实例只许一个写，第二个自动退到「(2)」。"""

    def test_second_caller_gets_suffix_then_returns_after_release(self):
        base = self.dir / "会议 [CrispASR · Qwen3].srt"
        p1, l1 = IL.unique_artifact(base)
        self.assertEqual(p1, base)
        self.assertIsNotNone(l1)
        p2, l2 = IL.unique_artifact(base)
        self.assertEqual(p2.name, "会议 [CrispASR · Qwen3] (2).srt")
        self.assertIsNotNone(l2)
        l1.release()
        p3, l3 = IL.unique_artifact(base)
        self.assertEqual(p3, base)          # 对方结束后回到原名
        for lk in (l2, l3):
            if lk is not None:
                lk.release()

    def test_all_taken_falls_back_without_lock(self):
        base = self.dir / "y.srt"
        p1, l1 = IL.unique_artifact(base, max_tries=1)
        self.assertIsNotNone(l1)
        p2, l2 = IL.unique_artifact(base, max_tries=1)
        self.assertEqual(p2, base)          # 退不出去 → 原名（绝不挡住转录）
        self.assertIsNone(l2)
        l1.release()


class TestLogInstanceSuffix(unittest.TestCase):
    """并行实例日志按 PID 分文件（反馈 #6：作者要 log 时两实例混成一条）。"""

    def test_suffix_changes_log_filename(self):
        import applog
        old = applog._SUFFIX
        try:
            applog.set_instance_suffix("")
            p1 = applog.log_path()
            applog.set_instance_suffix("-pid999")
            p2 = applog.log_path()
            self.assertIsNotNone(p1)
            self.assertIsNotNone(p2)
            self.assertEqual(p2.name, p1.name.replace(".log", "-pid999.log"))
        finally:
            applog.set_instance_suffix(old)


if __name__ == "__main__":
    unittest.main()

"""instance_lock.py — 跨进程互斥与单实例守卫（零第三方依赖）

背景
----
STTMiniTool 的安装目录是「单实例假设」：subtitles/ 输出、settings.json、
logs/、ov_models/ 与 crispasr/（解压前会先删 .exe/.dll 再 copy）都是共享
的。同时开两个实例（反馈 #6）会互相覆盖输出、并发解压正在运行的
crispasr.exe、把日志混成一条，还可能让「转完最后一步写盘」失败。

原语
----
• FileLock        —— 跨进程互斥。Windows 用 msvcrt 的 1 字节区域锁，POSIX
                     用 fcntl.flock。持有者崩溃／被结束时由内核自动释放，
                     不会留下死锁（不像 O_EXCL 建锁文件那样需要手工清理）。
• FileLock.named(name)         —— 全局资源（模型下载、核心解压）。
• FileLock.for_artifact(path)  —— 产物（同一份字幕两个实例只许一个写）。
• SingleInstance  —— 单实例守卫：持锁期间把 pid／端口／URL 写进 state 文件，
                     第二个实例据此提示「已有实例在运行」并聚焦已有窗口。

state 目录：%LOCALAPPDATA%\\STTMiniTool（POSIX：$XDG_STATE_HOME/STTMiniTool）。
环境变量 STTMINITOOL_STATE_DIR 可整体覆盖（测试用）。
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_ENV_STATE_DIR = "STTMINITOOL_STATE_DIR"
_LOCK_DIR_NAME = "locks"


def state_dir() -> Path:
    """进程间共享的小文件目录（锁与 instance.json）。"""
    override = os.environ.get(_ENV_STATE_DIR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "STTMiniTool"
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return Path(base) / "STTMiniTool"


def _locks_dir() -> Path:
    d = state_dir() / _LOCK_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    """锁文件名安全化：只留字母数字与 -_，其余折成下划线。"""
    out = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    return out[:80] or "lock"


def atomic_replace(src, dst, attempts: int = 40, delay: float = 0.01) -> None:
    """os.replace 的短重试版本（跨进程安全替换）。

    Windows 上两个进程/线程同时替换**同一目标**时，目标会在对方的
    rename 期间被占用 → PermissionError（拒绝访问）。原子写的语义不变
    （要么旧文件、要么新文件），这里只把这种瞬时冲突重试掉。
    """
    last = None
    for _ in range(max(1, attempts)):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            last = e
            if getattr(e, "errno", None) not in (errno.EACCES, errno.EPERM, errno.EBUSY):
                raise
            time.sleep(delay)
    raise last


def _try_lock(f) -> bool:
    """在已打开文件的第 0 字节处尝试非阻塞独占锁。"""
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(f) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class FileLock:
    """跨进程独占锁（不可重入：同一对象重复 acquire 直接返回 True）。

    timeout 秒内反复尝试，拿不到返回 False；进程退出／崩溃由内核释放锁。
    用法：
        lk = FileLock.named("downloads")
        if not lk.acquire():
            raise RuntimeError("另一个实例正在使用")
        try: ...
        finally: lk.release()
    """

    def __init__(self, path, timeout: float = 0.0, poll: float = 0.1):
        self.path = Path(path)
        self.timeout = max(0.0, float(timeout))
        self.poll = max(0.01, float(poll))
        self._f = None
        self.unavailable = False   # True = 锁文件本身打不开（≠ 已被占用）

    # ── 构造器 ──────────────────────────────────────────────────────
    @classmethod
    def named(cls, name: str, timeout: float = 0.0) -> "FileLock":
        """按逻辑名取锁（锁文件集中放在 state 目录，不污染产物目录）。"""
        return cls(_locks_dir() / f"name-{_slug(name)}.lock", timeout=timeout)

    @classmethod
    def for_artifact(cls, path, timeout: float = 0.0) -> "FileLock":
        """按产物绝对路径取锁：不同路径互不相干，同一路径全局唯一。"""
        try:
            key = str(Path(path).resolve())
        except OSError:
            key = str(path)
        digest = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:20]
        return cls(_locks_dir() / f"artifact-{digest}.lock", timeout=timeout)

    # ── 语义 ────────────────────────────────────────────────────────
    @property
    def locked(self) -> bool:
        return self._f is not None

    def acquire(self) -> bool:
        if self._f is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            f = open(self.path, "a+b")
        except OSError:
            self.unavailable = True        # 目录不可写等 → 不是「已被占用」
            return False
        deadline = time.monotonic() + self.timeout
        while True:
            if _try_lock(f):
                self._f = f
                return True
            if time.monotonic() >= deadline:
                try:
                    f.close()
                except OSError:
                    pass
                return False
            time.sleep(self.poll)

    def release(self) -> None:
        f, self._f = self._f, None
        if f is None:
            return
        _unlock(f)
        try:
            f.close()
        except OSError:
            pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def unique_artifact(path, max_tries: int = 20):
    """取产物锁并给出唯一路径：已被别的实例占用 → "<stem> (2)<后缀>" …

    两个实例转同一个源文件会算出同一个字幕名（互相覆盖；txt 分支还会删掉
    对方的 .srt —— 反馈 #6）。返回 (最终路径, 已持有的锁)——锁拿不到时返回
    (原路径, None)，绝不挡住转录；调用方负责在结束时 release()。
    """
    base = Path(path)
    try:
        for i in range(1, max(1, int(max_tries)) + 1):
            if i == 1:
                cand = base
            else:
                cand = base.with_name(f"{base.stem} ({i}){base.suffix}")
            lock = FileLock.for_artifact(cand)
            if lock.acquire():
                return cand, lock
    except Exception:
        pass
    return base, None


class SingleInstance:
    """单实例守卫：拿到锁 = 本进程是唯一实例。

    拿不到锁说明已有实例在运行（锁由内核维持，进程没了就自动释放），
    此时 read_state() 可读到对方写的 pid／端口／URL。
    """

    def __init__(self, name: str = "STTMiniTool", timeout: float = 0.0):
        self.name = name
        self.lock = FileLock.named(f"instance-{name}", timeout=timeout)
        self.already_running = False
        self.acquired = False

    def acquire(self) -> bool:
        self.acquired = self.lock.acquire()
        # 锁文件打不开（state 目录不可写）≠ 已有实例：不能据此弹「已在运行」
        self.already_running = (not self.acquired) and not self.lock.unavailable
        return self.acquired

    @property
    def state_path(self) -> Path:
        return state_dir() / f"{_slug(self.name)}.json"

    def write_state(self, **info) -> None:
        """原子写 state 文件（只有持锁者调用；失败静默——守卫不挡启动）。"""
        info.setdefault("pid", os.getpid())
        info.setdefault("startedAt", time.time())
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(
                f"{self.state_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def read_state(self) -> dict:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def clear_state(self) -> None:
        try:
            self.state_path.unlink(missing_ok=True)
        except OSError:
            pass

    def release(self) -> None:
        if self.acquired:
            self.clear_state()      # 只有持有者才删 state，别误删对方的
        self.lock.release()

"""transcription_queue.py — 转录任务队列（单 worker FIFO，零第三方依赖）

此前 WebBackend.transcribe 的并发语义是「第二请求直接报错」——批量转录
（CLI 多文件 / 未来前端批量）无从谈起。本模块提供通用任务队列：

  • 串行执行：同一时刻至多一个任务在跑（引擎本来就非并发安全）
  • submit(fn, label) → job_id：fn(cancel_event) 为执行体，结果/异常由
    队列记录并经 on_event 推送
  • cancel(job_id)：排队中 → 标记跳过；运行中 → 置位 cancel_event
    （执行体自行配合检查，与引擎的取消契约一致）
  • on_event(kind, job)：queued / started / finished / failed / cancelled
    （job 为脱敏快照，不含 fn——可直接 JSON 序列化给 SSE/前端）

纯逻辑、无磁盘与网络依赖：tests/test_transcription_queue.py 用普通函数
即可覆盖全部语义（顺序、事件序列、失败续跑、跳过、运行中取消、队列上限、
关闭）。
"""
from __future__ import annotations

import queue
import threading
import uuid

# 排队上限：超出直接拒绝（防无界堆积吃内存/磁盘；转录任务是 GB 级输入）
DEFAULT_MAXSIZE = 20


class TaskQueue:
    """单 worker FIFO 任务队列。"""

    def __init__(self, on_event=None, maxsize: int = DEFAULT_MAXSIZE):
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self._on_event = on_event or (lambda kind, job: None)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._current: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 队列空闲后 worker 退出，下次 submit 再拉起（长期挂机不留常驻线程）
        self._idle_poll = 0.2   # 秒

    # ── 公开 API ─────────────────────────────────────────────────────
    def submit(self, fn, label: str = "") -> str:
        """入队一个任务：fn(cancel_event) 在 worker 线程执行。返回 job_id。"""
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "label": label, "state": "queued", "fn": fn,
               "result": None, "error": None, "cancel_event": threading.Event()}
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("队列已关闭，不再接受新任务。")
            self._jobs[jid] = job
        try:
            self._q.put_nowait(jid)
        except queue.Full:
            with self._lock:
                self._jobs.pop(jid, None)
            raise RuntimeError(f"任务队列已满（上限 {self._q.maxsize}），请稍后再试。")
        self._emit("queued", job)
        self._ensure_worker()
        return jid

    def cancel(self, jid: str) -> bool:
        """取消任务：排队中 → 直接跳过；运行中 → 置位取消事件（协作式）。"""
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return False
            if self._current == jid:
                job["cancel_event"].set()
                return True           # 执行体自行决定如何收尾（state 由其结果定）
            if job["state"] == "queued":
                job["state"] = "cancelled"
        self._emit("cancelled", job)
        return True

    def status(self) -> list[dict]:
        """全部任务的脱敏快照（不含 fn/cancel_event，可 JSON 序列化）。"""
        with self._lock:
            jobs = [self._snapshot(j) for j in self._jobs.values()]
        # queued 在前、其余按插入序；快照顺序稳定
        return jobs

    def current_id(self) -> str | None:
        with self._lock:
            return self._current

    def close(self, wait: bool = False):
        """停止接受新任务并结束 worker（wait=True 时等线程退出）。"""
        self._stop.set()
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    # ── 内部 ─────────────────────────────────────────────────────────
    @staticmethod
    def _snapshot(job: dict) -> dict:
        return {"id": job["id"], "label": job["label"], "state": job["state"],
                "error": job["error"]}

    def _emit(self, kind: str, job: dict):
        try:
            self._on_event(kind, self._snapshot(job))
        except Exception:
            pass                     # 事件回调异常不得影响任务执行

    def _ensure_worker(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="transcription-queue")
            self._thread.start()

    def _run(self):
        while True:
            try:
                jid = self._q.get(timeout=self._idle_poll)
            except queue.Empty:
                with self._lock:
                    if self._stop.is_set() or not self._jobs or all(
                            j["state"] in ("finished", "failed", "cancelled")
                            for j in self._jobs.values()):
                        return          # 空闲且无待办 → 线程退出（下次 submit 重拉）
                continue
            with self._lock:
                job = self._jobs.get(jid)
                if job is None:
                    continue
                if job["state"] == "cancelled":
                    continue                     # 排队期间被取消 → 跳过（cancel 时已发事件）
                job["state"] = "running"
                self._current = jid
            self._emit("started", job)
            try:
                job["result"] = job["fn"](job["cancel_event"])
                job["state"] = "finished"
            except Exception as e:
                job["state"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
            finally:
                with self._lock:
                    self._current = None
            self._emit(job["state"], job)

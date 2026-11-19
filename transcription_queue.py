"""transcription_queue.py — 转录任务队列（单 worker FIFO，零第三方依赖）

此前 WebBackend.transcribe 的并发语义是「第二请求直接报错」——批量转录
（CLI 多文件 / 前端批量）无从谈起。本模块提供通用任务队列：

  • 串行执行：同一时刻至多一个任务在跑（引擎本来就非并发安全）
  • submit(fn, label, meta, front=False) → job_id：fn(cancel_event) 为执行体，
    结果/异常由队列记录并经 on_event 推送；front=True 插到队首（录音等
    交互式任务优先）
  • move(job_id, before=None)：调整「排队中」任务的顺序（前端拖动排序）
  • cancel(job_id)：排队中 → 标记跳过；运行中 → 置位 cancel_event 并立刻
    标记 cancelling（前端即时反馈），终态由执行体收尾后决定
  • pause() / resume()：暂停派发。运行中的任务被置位 pause_requested，执行体
    停下后该任务状态记为 paused 并留在「恢复队列」，resume() 后仍按原顺序先跑
    （刷新/断连时暂停进行中的任务、保留排队）
  • on_event(kind, job)：queued / started / cancelling / paused / finished /
    failed / cancelled（job 为脱敏快照，不含 fn——可直接 JSON 序列化给 SSE）
  • update_progress(job_id, pct, status)：把当前进度写进快照
  • job_result(job_id)：执行体返回值（后端转存为「最近完成结果」）

status() 按「执行顺序」返回快照（运行中 → 恢复队列 → 排队 → 其余），前端
直接照此顺序渲染；拖动排序后无需自己重排。

纯逻辑、无磁盘与网络依赖：tests/test_transcription_queue.py 覆盖顺序、事件
序列、失败续跑、跳过、运行中取消、暂停/恢复、重排、队列上限、关闭。
"""
from __future__ import annotations

import threading
import uuid
from collections import deque

# 排队上限：超出直接拒绝（防无界堆积吃内存/磁盘；转录任务是 GB 级输入）
DEFAULT_MAXSIZE = 20

_TERMINAL = ("finished", "failed", "cancelled")


class TaskQueue:
    """单 worker FIFO 任务队列（支持暂停/恢复、重排、优先入队）。"""

    def __init__(self, on_event=None, maxsize: int = DEFAULT_MAXSIZE):
        self._pending: deque[str] = deque()      # 排队中的 job id（执行顺序）
        self._resume: deque[str] = deque()       # 被暂停打断的任务（恢复后优先跑）
        self._maxsize = max(1, int(maxsize))
        self._cv = threading.Condition(threading.RLock())
        self._jobs: dict[str, dict] = {}
        self._current: str | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_event = on_event or (lambda kind, job: None)
        # 队列空闲后 worker 退出，下次 submit 再拉起（长期挂机不留常驻线程）
        self._idle_poll = 0.2

    # ── 公开 API ─────────────────────────────────────────────────────
    def submit(self, fn, label: str = "", meta: dict | None = None,
               front: bool = False, job_id: str | None = None) -> str:
        """入队一个任务：fn(cancel_event) 在 worker 线程执行。返回 job_id。

        meta：可 JSON 序列化的附注（文件名/大小/来源等），随快照给前端。
        front：插到队首（交互式任务优先，例如录音分段）。
        job_id：调用方预分配 id（便于先把输入文件/暂存目录登记好，避免
                任务瞬间完成时登记还没做——实测过这个竞态）。
        """
        jid = job_id or uuid.uuid4().hex[:12]
        job = {"id": jid, "label": label, "state": "queued", "fn": fn,
               "meta": dict(meta or {}), "pct": 0, "status": "",
               "result": None, "error": None, "cancelling": False,
               "pause_requested": False,
               "cancel_event": threading.Event()}
        with self._cv:
            if self._stop.is_set():
                raise RuntimeError("队列已关闭，不再接受新任务。")
            if len(self._pending) >= self._maxsize:
                raise RuntimeError(f"任务队列已满（上限 {self._maxsize}），请稍后再试。")
            self._jobs[jid] = job
            (self._pending.appendleft if front else self._pending.append)(jid)
            self._cv.notify_all()
        self._emit("queued", job)
        self._ensure_worker()
        return jid

    def cancel(self, jid: str) -> bool:
        """取消任务：排队中/已暂停 → 直接跳过；运行中 → 置位取消事件（协作式）。

        运行中的任务立刻转入 cancelling（前端据此**立即**显示「正在取消…」，
        不必等引擎收尾的几秒）；终态仍由执行体收尾后决定。
        """
        job = None
        kind = None
        with self._cv:
            job = self._jobs.get(jid)
            if job is None:
                return False
            if self._current == jid:
                job["cancel_event"].set()
                if job.get("state") == "running":
                    job["cancelling"] = True
                    job["status"] = "正在取消…"
                kind = "cancelling"
            elif job["state"] in ("queued", "paused"):
                job["state"] = "cancelled"
                for d in (self._pending, self._resume):
                    try:
                        d.remove(jid)
                    except ValueError:
                        pass
                kind = "cancelled"
                self._cv.notify_all()
        if kind:
            self._emit(kind, job)
        return True

    def move(self, jid: str, before: str | None = None) -> bool:
        """把「排队中」的任务移到 before 之前（before=None 移到队尾）。

        只允许移动 queued 状态的任务；运行中/已暂停/已终态一律拒绝。
        """
        with self._cv:
            job = self._jobs.get(jid)
            if job is None or job["state"] != "queued":
                return False
            try:
                self._pending.remove(jid)
            except ValueError:
                return False
            if before:
                try:
                    self._pending.insert(self._pending.index(before), jid)
                except ValueError:
                    self._pending.append(jid)      # 目标不在队里 → 落到队尾
            else:
                self._pending.append(jid)
            self._cv.notify_all()
        self._emit("reordered", job)
        return True

    def pause(self) -> str | None:
        """暂停派发：停止启动新任务，并把运行中的任务标记为「请尽快停下」。

        返回正在运行的 job_id（调用方负责接通引擎取消链让它真的停下），
        没有运行中的任务时返回 None。
        """
        with self._cv:
            self._paused.set()
            self._cv.notify_all()
            jid = self._current
            if jid:
                job = self._jobs.get(jid)
                if job is not None:
                    job["pause_requested"] = True
            return jid

    def resume(self) -> int:
        """恢复派发：被暂停的任务重新排队（在原队列之前先跑）。返回恢复数。"""
        with self._cv:
            if not self._paused.is_set():
                return 0
            self._paused.clear()
            job_list = [self._jobs[j] for j in self._resume
                        if j in self._jobs and self._jobs[j]["state"] == "paused"]
            for job in job_list:
                job["state"] = "queued"
                job["pct"] = 0
                job["status"] = ""
            self._cv.notify_all()
        for job in job_list:
            self._emit("queued", job)
        self._ensure_worker()
        return len(job_list)

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def adopt(self, jid: str, label: str = "", meta: dict | None = None,
              state: str = "finished", error: str | None = None) -> bool:
        """把一个「已终态」的历史任务放进列表（重启恢复归档用，不执行）。

        只影响展示/清理；结果由调用方另外提供（后端从归档 meta 读）。
        """
        with self._cv:
            if jid in self._jobs:
                return False
            self._jobs[jid] = {**{"id": jid, "label": label, "state": state, "fn": None,
                                  "meta": dict(meta or {}),
                                  "pct": 100 if state == "finished" else 0,
                                  "status": "", "result": None, "error": error,
                                  "cancelling": False, "pause_requested": False,
                                  "cancel_event": threading.Event()}}
        return True

    def status(self) -> list[dict]:
        """按执行顺序返回全部任务的脱敏快照（可 JSON 序列化）。"""
        with self._cv:
            order: list[str] = []
            if self._current:
                order.append(self._current)
            order.extend(self._resume)
            order.extend(self._pending)
            seen = set(order)
            order.extend(jid for jid in self._jobs if jid not in seen)
            return [self._snapshot(self._jobs[j]) for j in order if j in self._jobs]

    def current_id(self) -> str | None:
        with self._cv:
            return self._current

    def pending_ids(self) -> list[str]:
        """排队中的任务 id（执行顺序）。"""
        with self._cv:
            return list(self._pending)

    def update_progress(self, jid: str, pct, status: str = ""):
        """把进度写进快照（worker 线程调用；前端按事件取）。"""
        with self._cv:
            job = self._jobs.get(jid)
            if job is None:
                return
            try:
                job["pct"] = max(0, min(100, int(pct)))
            except (TypeError, ValueError):
                pass
            if status is not None:
                job["status"] = str(status)

    def job_result(self, jid: str):
        """取执行体返回值（完成后由调用方转存；不存在回 None）。"""
        with self._cv:
            job = self._jobs.get(jid)
            return None if job is None else job.get("result")

    def drop(self, jid: str) -> bool:
        """移除一个已终态的特定任务（录音等 hidden 临时任务用，避免无限累积）。"""
        with self._cv:
            job = self._jobs.get(jid)
            if job is None or job["state"] not in _TERMINAL:
                return False
            self._jobs.pop(jid, None)
            return True

    def clear_finished(self) -> list[str]:
        """移除已终态的任务，返回被移除的 id（后端据此释放结果/文件）。"""
        with self._cv:
            ids = [jid for jid, j in self._jobs.items() if j["state"] in _TERMINAL]
            for jid in ids:
                self._jobs.pop(jid, None)
        return ids

    def pending_count(self) -> int:
        """尚未进入终态的任务数（排队 + 暂停 + 运行中）。"""
        with self._cv:
            return sum(1 for j in self._jobs.values() if j["state"] not in _TERMINAL)

    def close(self, wait: bool = False):
        """停止接受新任务并结束 worker（wait=True 时等线程退出）。"""
        self._stop.set()
        self._paused.set()
        with self._cv:
            self._cv.notify_all()
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    # ── 内部 ─────────────────────────────────────────────────────────
    @staticmethod
    def _snapshot(job: dict) -> dict:
        return {"id": job["id"], "label": job["label"], "state": job["state"],
                "error": job["error"], "meta": dict(job.get("meta") or {}),
                "pct": job.get("pct", 0), "status": job.get("status", ""),
                "cancelling": bool(job.get("cancelling"))}

    def _emit(self, kind: str, job: dict):
        try:
            self._on_event(kind, self._snapshot(job))
        except Exception:
            pass                     # 事件回调异常不得影响任务执行

    def _all_terminal(self) -> bool:
        return all(j["state"] in _TERMINAL for j in self._jobs.values())

    def _ensure_worker(self):
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="transcription-queue")
            self._thread.start()

    def _take(self, timeout: float) -> str | None:
        """取下一个可执行的任务 id（暂停时回 None，不阻塞太久）。"""
        with self._cv:
            if self._stop.is_set() or self._paused.is_set():
                return None
            if self._resume:
                return self._resume.popleft()
            if self._pending:
                return self._pending.popleft()
            self._cv.wait(timeout)
            if self._stop.is_set() or self._paused.is_set():
                return None
            if self._resume:
                return self._resume.popleft()
            if self._pending:
                return self._pending.popleft()
            return None

    def _run(self):
        while True:
            if self._stop.is_set():
                return
            jid = self._take(self._idle_poll)
            if jid is None:
                with self._cv:
                    if self._stop.is_set():
                        return
                    if self._all_terminal():
                        return    # 无待办 → 退出（submit/resume 会重新拉起）
                continue
            with self._cv:
                job = self._jobs.get(jid)
                if job is None:
                    continue
                if job["state"] == "cancelled":
                    continue                     # 排队期间被取消 → 跳过
                if self._paused.is_set():
                    # 暂停抢在启动之前：不启动引擎，直接交回恢复队列
                    job["state"] = "paused"
                    job["pct"] = 0
                    job["status"] = ""
                    self._resume.appendleft(jid)
                else:
                    job["state"] = "running"
                    job["pct"] = 0
                    job["status"] = ""
                    job["cancelling"] = False
                    self._current = jid
            if job["state"] == "paused":
                self._emit("paused", job)
                continue
            self._emit("started", job)
            paused = False
            try:
                job["result"] = job["fn"](job["cancel_event"])
                job["state"] = "finished"
            except Exception as e:
                job["state"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
            finally:
                with self._cv:
                    self._current = None
                    job["cancelling"] = False
                    paused = bool(job.get("pause_requested"))
                    if paused:
                        job["pause_requested"] = False
                        job["state"] = "paused"
                        job["error"] = None
                        job["result"] = None
                        job["pct"] = 0
                        job["status"] = ""
                        self._resume.appendleft(job["id"])
                    elif job["state"] == "failed" and job["cancel_event"].is_set():
                        job["state"] = "cancelled"
                        job["error"] = None
                    if job["state"] != "finished":
                        job["status"] = ""      # 终态不保留过程文案
            if paused:
                self._emit("paused", job)
            else:
                self._emit(job["state"], job)


if __name__ == "__main__":
    import time

    def demo(ce):
        time.sleep(0.2)
        return "ok"

    q = TaskQueue(on_event=lambda k, j: print(k, j))
    q.submit(demo, label="demo")
    time.sleep(1.0)
    q.close(wait=True)

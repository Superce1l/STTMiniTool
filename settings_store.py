"""settings_store.py — settings.json 的统一读写层

旧实现里 settings.json 的「读 → 改 → 写」散落在 webview_backend 各处
（_persist_backend / set_model / set_settings / _persist_setting 各写一遍），
无锁也无原子性：
  • 两处并发写会互相覆盖（读改写序列丢更新）；
  • 进程中途被杀可能留下截断的 JSON，且无备份可恢复。
SettingsStore 收敛为单点：线程锁 + mtime 失效的内存缓存 + 临时文件
os.replace 原子替换（替换本身原子，进程任何时刻被杀都不会留半截文件）。

cli_mode 的临时设置副本机制不受影响：它先改 core.SETTINGS_FILE 再构造
WebBackend，Store 在构造时取路径，读写自然落在副本上。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class SettingsStore:
    """单文件 JSON 设置的线程安全读写。"""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._cache: dict | None = None
        self._sig: tuple | None = None     # (mtime_ns, size)：外部修改检测信号

    def read(self) -> dict:
        """读完整设置（内存缓存；文件被外部修改时按 mtime+size 重新加载）。"""
        with self._lock:
            try:
                st = self.path.stat()
                sig = (st.st_mtime_ns, st.st_size)
            except OSError:
                self._cache, self._sig = {}, None
                return {}
            # 仅 mtime 不够：Windows 同一计时 tick 内的两次写 mtime 相同，
            # 会漏检「外部改坏 + 大小恰好不变」之外的全部变更；补上 size。
            if self._cache is not None and sig == self._sig:
                return self._cache
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                # 损坏/半截文件：返回空但不缓存 —— 外部修复后下次能读到
                return {}
            self._cache = data
            self._sig = sig
            return data

    def update(self, patch: dict) -> dict:
        """合并写入（浅合并）并返回更新后的完整设置。"""
        with self._lock:
            cur = dict(self.read())
            cur.update(patch or {})
            self._write(cur)
            return cur

    def remove(self, *keys) -> dict:
        """删除键并返回更新后的完整设置（不存在的键忽略）。"""
        with self._lock:
            cur = dict(self.read())
            changed = False
            for k in keys:
                if k in cur:
                    cur.pop(k)
                    changed = True
            if changed:
                self._write(cur)
            return cur

    def _write(self, cur: dict):
        """原子落盘 + 刷新缓存（调用方须已持锁）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            st = self.path.stat()
            self._sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._sig = None
        self._cache = cur

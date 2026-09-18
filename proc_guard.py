"""proc_guard.py — 进程清理：父进程结束 = 连带终止所有子程序（Windows）

问题背景
--------
本项目的 GPU 推理核心会以 subprocess 衍生子程序：
  • crispasr.exe（CrispASR / Whisper Vulkan）
  • chatllm main.exe（ForcedAligner 时间轴对齐）
Windows 不像 POSIX 会在父进程死亡时连带回收子程序——若使用者在「识别
到一半」关窗、或主进程崩溃／被任务管理器强制结束，这些子程序就成为孤儿
残留（仍占 GPU / 内存）。

解法
----
把主进程绑进一个 **KILL_ON_JOB_CLOSE 的 Windows Job Object**：之后派生的
子程序会自动继承同一个 Job；当主进程的最后一个 Job handle 关闭（即主进程
结束，含被强制结束／崩溃）时，OS 会自动终止 Job 内所有成员 → 子程序无一
幸免。`app.py`（CTk）与 `app_webview.py`（WebView）两个进入点共享本模块。

注意：chatllm 在 webview 走的是 **in-process libchatllm.dll**（非子程序），
Job Object 管不到它——那一条由各进入点以 `os._exit(0)` 硬退出、交给 OS
回收（见各进入点关闭流程）。本模块只负责「真正的子程序」。
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

# 保留 job handle 的模块级引用：若被 GC 回收，handle 提早关闭会「提早触发
# kill」把自己与子程序一起杀掉。存在这里确保其生命周期 == 进程生命周期。
_JOB_HANDLE = None


def setup_kill_on_close_job():
    """把目前进程加入一个 KILL_ON_JOB_CLOSE 的 Job Object（幂等）。

    返回 job handle（成功）或 None（非 Windows／失败，静默降级）。重复调用
    只会建立一次；handle 由模块级变量持有，调用端无须自行保管。
    """
    global _JOB_HANDLE
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    if sys.platform != "win32":
        return None

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800   # 容许浏览器 helper 自请脱离
    JobObjectExtendedLimitInformation = 9

    class _BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC),
            ("IoInfo", _IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXTENDED()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        if not k32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            return None
        # Win8+ 允许嵌套 Job；即使本进程已在别的 Job 内也多半成功。失败则降级。
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return None
        _JOB_HANDLE = job
        return job
    except Exception:
        return None

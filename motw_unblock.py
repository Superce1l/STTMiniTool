"""motw_unblock.py — 启动时自清程序目录的 Mark-of-the-Web（onedir 发行版必需）

背景
----
发行 ZIP 从浏览器下载后自带 MOTW（Zone.Identifier 备用数据流）；Explorer／
7-Zip 解压时把 ADS 传播到每个解压出的文件。.NET 运行时（pythonnet/
clr_loader → Microsoft.Web.WebView2 / WinForms / Python.Runtime）拒绝加载
带 MOTW 的托管程序集，症状是原生窗口静默回退 Edge --app 或只开白窗。

onefile 布局在运行时把 DLL 解压到 %TEMP%（无 ADS）天然免疫；onedir 的
DLL 直接躺在程序目录里 → 改为启动时自清。删除 ADS 不需要管理员：
Zone.Identifier 是普通备用数据流，用户对自己的文件有写权限即可删除
（与资源管理器「解除锁定」/ Unblock-File 同义）。顺带清掉 exe 本体的
MOTW —— 第二次启动连 SmartScreen「仍要运行」都不会再弹。

做法
----
遍历 EXE 旁整棵目录（含 _internal、crispasr 等），对每个文件
DeleteFileW("<path>:Zone.Identifier")。文件没有 ADS 时内核回
ERROR_FILE_NOT_FOUND，无害；FAT32 卷没有 ADS 概念，同样无害。
开销：每个文件一次内核调用，_internal 数千文件合计约几十毫秒，且清过
之后次次启动都是空扫。

约束：必须在任何 clr / .NET 程序集载入之前完成（app_webview.main 第一行）。
全程吞异常——清理失败绝不挡启动；失败时 pywebview 会走既有 Edge fallback。

只在 frozen（发行包）下执行：源码运行目录里不会有 MOTW，扫了也白扫。
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path


def unblock_app_files() -> int:
    """删除发行目录内所有文件的 Zone.Identifier ADS，返回清掉的个数。

    非 Windows／非 frozen／异常一律回 0 且不抛——调用端无须处理任何失败。
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return 0
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.DeleteFileW.argtypes = [wintypes.LPCWSTR]
        k32.DeleteFileW.restype = wintypes.BOOL
    except Exception:
        return 0

    root = Path(sys.executable).parent
    removed = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                # "<path>:Zone.Identifier"——ADS 语法；无 ADS 时内核报
                # ERROR_FILE_NOT_FOUND（返回 0），不视为错误也不计数。
                if k32.DeleteFileW(os.path.join(dirpath, name) + ":Zone.Identifier"):
                    removed += 1
    except OSError:
        pass
    return removed

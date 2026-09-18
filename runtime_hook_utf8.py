# runtime_hook_utf8.py
# PyInstaller runtime hook — runs before ANY user code or imports.
#
# Problem: Traditional Chinese Windows uses cp950 (Big5) as the system
# default encoding. Third-party libraries that call open() without an
# explicit encoding= parameter will use cp950, causing:
#   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa6 in position 0
# when reading UTF-8 files (e.g. opencc config files).
#
# Fix: Set PYTHONUTF8=1 which is equivalent to `python -X utf8`, making
# all text-mode file I/O default to UTF-8 regardless of system locale.
#
# Note: PYTHONUTF8 must be set before Python interpreter initialization
# to affect the C-level getpreferredencoding() result. PyInstaller runtime
# hooks run early enough for this to take effect.
import os
import sys

os.environ["PYTHONUTF8"] = "1"

# ── SSL CA 凭证修复（PyInstaller EXE 特有）─────────────────────────────
# 问题：冻结 EXE 中 Python ssl 模块找不到 CA bundle，导致 HTTPS 下载时
#       抛出 SSLCertVerificationError，但 HTTP 完全正常。
# 修法：尝试用 certifi 的 cacert.pem，若无则指向 Windows 系统凭证库
if getattr(sys, "frozen", False):
    try:
        import certifi
        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE",    ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
    except ImportError:
        pass   # certifi 未安装时靠 _download_file 的 fallback 处理

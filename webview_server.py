"""webview_server.py — 桌面 WebView 的本机 HTTP 服务器（纯标准库）

把 webview/ 静态前端与 /api/* 业务端点以同一个 stdlib HTTP 服务器提供，
并以 SSE（/api/events）把进度/状态主动推给前端。设计取向与 Eel /
flaskwebgui 相同（本机 server + 浏览器渲染），但零第三方相依，方便
PyInstaller 打包（与既有 api_server.py 一致的做法）。

安全：只绑 127.0.0.1（loopback），外部连不到、且不触发防火墙提示，
故桌面用途免密钥。

路由：
  GET  /                      → webview/index.html
  GET  /css/* /js/* …         → webview/ 静态档
  GET  /health                → {"status","model_ready"}
  GET  /api/status            → 后端状态
  GET  /api/settings          → 设置；POST 改设置（json patch）
  GET  /api/devices           → 设备 + 诊断
  POST /api/backend           → {index} 切换推理核心
  POST /api/transcribe        → multipart 上传转录，回 {segments,srtPath}
  GET  /api/events            → SSE：data: {"event","payload"} 进度/状态推送
"""
from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from webview_backend import WebBackend


def resolve_web_dir() -> Path:
    """前端资产目录：源码用项目 webview/；PyInstaller 冻结时用 _MEIPASS。

    冻结时刻意放在 `webview_assets/`，避免与 pywebview 包本身的
    `webview/` 目录在 _internal/ 撞名混在一起（两者皆名为 webview）。
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        for name in ("webview_assets", "webview"):     # 优先干净的 webview_assets
            d = base / name
            if (d / "index.html").is_file():
                return d
        return base / "webview_assets"
    return Path(__file__).resolve().parent / "webview"


WEB_DIR = resolve_web_dir()


def _parse_multipart(body: bytes, boundary: bytes):
    """解析 multipart/form-data（原 api_server._parse_multipart 的内联精简版）。

    返回 (fields: dict[str,str], files: dict[str,(filename,bytes)])。
    """
    fields: dict[str, str] = {}
    files: dict[str, tuple] = {}
    delim = b"--" + boundary
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not _:
            continue
        name = disp = filename = None
        for line in header_blob.decode("utf-8", errors="replace").split("\r\n"):
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "content-disposition":
                disp = val
                for tok in val.split(";"):
                    tok = tok.strip()
                    if tok.startswith("name="):
                        name = tok[5:].strip('"')
                    elif tok.startswith("filename="):
                        filename = tok[9:].strip('"')
        if name is None:
            continue
        if filename is not None:
            files[name] = (filename, content)
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


class _EventHub:
    """SSE 广播：每个连接一个 Queue，publish() 推给全部订阅者。"""

    def __init__(self):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: str, payload: dict):
        msg = {"event": event, "payload": payload}
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except Exception:
                pass


class WebViewServer:
    """本机前端服务器。持有 WebBackend + EventHub。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.hub = _EventHub()
        self.backend = WebBackend(on_event=self.hub.publish)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._want_port = port
        self.port = port

    def start(self):
        if self._httpd:
            return
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):    # 静音默认存取日志
                pass

            # ── 回应辅助 ──────────────────────────────────────
            def _json(self, obj, code=200):
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _err(self, code, msg):
                self._json({"error": {"message": msg}}, code)

            def _read_json_body(self) -> dict:
                n = int(self.headers.get("Content-Length", 0))
                if not n:
                    return {}
                try:
                    return json.loads(self.rfile.read(n).decode("utf-8"))
                except Exception:
                    return {}

            # ── DNS rebinding / CSRF 防护 ─────────────────────
            #   恶意网站可能用 DNS rebinding 把网域指向 127.0.0.1，或跨站
            #   POST 到本机 /api（端点有副作用）。要求 Host/Origin 必须是
            #   本机回环位址 —— Edge --app 加载的就是 127.0.0.1:port，会通过。
            def _check_host(self) -> bool:
                allowed = {f"127.0.0.1:{server.port}", f"localhost:{server.port}"}
                if self.headers.get("Host", "") not in allowed:
                    self._err(403, "bad host")
                    return False
                origin = self.headers.get("Origin")
                if origin and origin not in {f"http://{a}" for a in allowed}:
                    self._err(403, "bad origin")
                    return False
                return True

            # ── GET ───────────────────────────────────────────
            def do_GET(self):
                path = urlparse(self.path).path
                if (path.startswith("/api/") or path == "/health") and not self._check_host():
                    return
                if path == "/health":
                    b = server.backend
                    return self._json({"status": "ok",
                                       "model_ready": bool(getattr(b.engine, "ready", False))})
                if path == "/api/status":
                    return self._json(server.backend.get_status())
                if path == "/api/settings":
                    return self._json(server.backend.get_settings())
                if path == "/api/devices":
                    return self._json(server.backend.list_devices())
                if path == "/api/accel":          # 硬件检测 + CrispASR 加速版本清单
                    return self._json(server.backend.get_accel())
                if path == "/api/model-options":
                    return self._json(server.backend.get_model_options())
                if path == "/api/basic-profiles":   # 基础模式：用途 → 建议模型
                    return self._json(server.backend.get_basic_profiles())
                if path == "/api/languages":
                    return self._json(server.backend.get_languages())
                if path == "/api/health-check":
                    return self._json(server.backend.health_check())
                if path == "/api/events":
                    return self._sse()
                if path.startswith("/api/"):
                    return self._err(404, "unknown api")
                return self._static(path)

            # ── POST ──────────────────────────────────────────
            def do_POST(self):
                path = urlparse(self.path).path
                if not self._check_host():        # 所有 POST 皆为 /api，一律验证
                    return
                try:
                    if path == "/api/settings":
                        return self._json(server.backend.set_settings(self._read_json_body()))
                    if path == "/api/backend":
                        return self._json(server.backend.set_backend(
                            self._read_json_body().get("index")))
                    if path == "/api/model":
                        body = self._read_json_body()
                        return self._json(server.backend.set_model(
                            body.get("core"), body.get("model")))
                    if path == "/api/accel":         # 选定加速版本（下次加载时下载）
                        return self._json(server.backend.set_accel(
                            self._read_json_body().get("variant")))
                    if path == "/api/load":           # 首次选定模型 → 就地下载并加载
                        return self._json(server.backend.request_load())
                    if path == "/api/browse-folder":  # 原生「选择文件夹」对话框
                        body = self._read_json_body()
                        return self._json(server.backend.browse_folder(
                            body.get("start", ""), body.get("title", "选择文件夹")))
                    if path == "/api/browse-file":    # 原生「选择文件」对话框
                        body = self._read_json_body()
                        return self._json(server.backend.browse_file(
                            body.get("start", ""), body.get("title", "选择文件")))
                    if path == "/api/transcribe":
                        return self._transcribe()
                    if path == "/api/cancel":
                        return self._json({"ok": server.backend.cancel()})
                    if path == "/api/open-output":
                        return self._json({"ok": server.backend.open_output_dir()})
                    if path == "/api/check-update":
                        return self._json(server.backend.open_releases())
                    return self._err(404, "unknown api")
                except Exception as e:
                    return self._err(500, str(e))

            # ── 静态档 ────────────────────────────────────────
            def _static(self, path):
                if path in ("", "/"):
                    path = "/index.html"
                target = (WEB_DIR / path.lstrip("/")).resolve()
                # 防目录穿越：必须仍在 WEB_DIR 内
                if WEB_DIR.resolve() not in target.parents and target != WEB_DIR.resolve():
                    return self._err(403, "forbidden")
                if not target.is_file():
                    return self._err(404, "not found")
                data = target.read_bytes()
                ctype = _CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)

            # ── SSE 进度/状态流式 ─────────────────────────────
            def _sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = server.hub.subscribe()
                # 连上时先补一次目前状态，避免错过加载完成事件
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    init = {"event": "status",
                            "payload": {"modelReady": bool(getattr(server.backend.engine, "ready", False))}}
                    self.wfile.write(f"data: {json.dumps(init, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")     # 心跳，维持连接
                            self.wfile.flush()
                            continue
                        self.wfile.write(f"data: {json.dumps(msg, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    server.hub.unsubscribe(q)

            # ── 转录（multipart 上传）─────────────────────────
            def _transcribe(self):
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype or "boundary=" not in ctype:
                    return self._err(400, "需以 multipart/form-data 上传 file")
                boundary = ctype.split("boundary=", 1)[1].strip().strip('"').encode()
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                fields, files = _parse_multipart(body, boundary)
                if "file" not in files or not files["file"][1]:
                    return self._err(400, "缺少 file 或内容为空")
                filename, data = files["file"]

                tmp_dir = Path(tempfile.mkdtemp(prefix="asr_web_"))
                ext = Path(filename).suffix or ".wav"
                in_path = tmp_dir / ("upload" + ext)
                in_path.write_bytes(data)

                opts = {
                    "path": str(in_path),
                    "name": filename,        # 原始上传文件名 → 输出字幕用它命名、落 subtitles/
                    "language": (fields.get("language") or "").strip() or None,
                    "diarize": (fields.get("diarize", "") or "").lower() in ("1", "true", "on", "yes"),
                    "nSpeakers": fields.get("n_speakers", ""),
                    "align": (fields.get("align", "1") or "").lower() in ("1", "true", "on", "yes"),
                    "hint": fields.get("hint", ""),
                }
                try:
                    result = server.backend.transcribe(
                        opts, progress_cb=lambda pct, status:
                        server.hub.publish("progress", {"pct": pct, "status": status}))
                    return self._json(result)
                finally:
                    try:
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

        self._httpd = ThreadingHTTPServer((self.host, self._want_port), _Handler)
        self.port = self._httpd.server_address[1]      # 取得实际绑定的端口（port=0 时）
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


if __name__ == "__main__":
    # 独立启动（除错用）：起 server，打印网址。只有「选择的模型已下载」才自动
    # 加载（与 app_webview 一致：首次/未下载 → 等使用者于模型页按下载，避免硬抓）。
    srv = WebViewServer(port=8765)
    srv.start()
    if srv.backend.selected_model_present():
        srv.backend.start_load()
    print(f"WebView server: {srv.url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.stop()

"""downloader 完整性防线：SHA256 清单、TLS 降级防线、TOFU pin、HTTP 下载流程

分层：
  • 纯函数（清单覆盖 / 主机名单 / pin 读写）——无网络无磁盘依赖（pin 用临时目录）
  • HTTP 流程（本机 ThreadingHTTPServer 回环）——成功 / 截断 / TOFU 篡改 / 断点续传
"""
import json
import re
import sys
import tempfile
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

import downloader as D


class TestSha256Manifest(unittest.TestCase):
    def test_hashes_wellformed(self):
        for key, digest in D._SHA256_MANIFEST.items():
            self.assertRegex(key, r"^[^/]+/[^/]+/.+$", key)
            self.assertRegex(digest, r"^[0-9a-f]{64}$", key)

    def test_all_model_download_urls_covered(self):
        """项目实际使用的每个 HF 下载 URL 都必须能推导出清单内的键。"""
        urls = [
            D._VAD_URL, D._FA_BIN_URL, D._FFMPEG_ZIP_URL,
            f"{D._HF_1P7B_BASE}/audio_encoder_model.bin",
            f"{D._HF_1P7B_BASE}/decoder_kv_model.bin",
            f"{D._DIAR_BASE}/segmentation-community-1.onnx",
            f"{D._DIAR_BASE}/embedding_model.onnx",
            "https://huggingface.co/dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO/resolve/main/decoder_model.bin",
            "https://huggingface.co/Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO/resolve/main/decoder_model.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            f"{D._ALIGNER_GGUF_BASE}/qwen3-forced-aligner-0.6b-q5_0.gguf",
            f"{D._QWEN3_ASR_GGUF_BASE}/qwen3-asr-1.7b-q8_0.gguf",
            f"{D._QWEN3_ASR_JA_GGUF_BASE}/qwen3-asr-1.7b-ja-anime-q4_k.gguf",
        ]
        for u in urls:
            self.assertIn(D._sha_key_for_url(u), D._SHA256_MANIFEST, u)

    def test_github_release_zip_uncovered_goes_tofu(self):
        self.assertIsNone(D._sha_key_for_url(
            "https://github.com/CrispStrobe/CrispASR/releases/download/"
            "v0.8.44/crispasr-windows-x86_64-vulkan.zip"))

    def test_mirror_rewrite_keeps_key(self):
        """镜像只换域名：官方 URL 与镜像 URL 推导出同一把键。"""
        official = "https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/main/x.gguf"
        self.assertEqual(D._sha_key_for_url(official),
                         D._sha_key_for_url(official))


class TestTlsFallbackGuard(unittest.TestCase):
    def test_used_hosts_allowlisted(self):
        for host in ("huggingface.co", "hf-mirror.com", "github.com",
                     "raw.githubusercontent.com"):
            self.assertIn(host, D._TLS_FALLBACK_HOSTS)

    def test_url_host_extraction(self):
        self.assertEqual(D._url_host("https://hf-mirror.com/a/b"), "hf-mirror.com")
        self.assertEqual(D._url_host("https://github.com/x"), "github.com")
        self.assertEqual(D._url_host("garbage"), "")

    def test_cert_failure_on_unknown_host_reraises(self):
        """非名单内主机证书失败必须原样抛错（绝不静默降级为不验证）。"""
        err = urllib.error.URLError(reason=ssl_cert_error())
        req = object()
        with mock.patch.object(D.urllib.request, "urlopen", side_effect=err) as m:
            with self.assertRaises(urllib.error.URLError):
                D._download_file("https://evil.example.com/x.bin",
                                 Path(tempfile.gettempdir()) / "unused")
        m.assert_called_once()   # 未发生第二次（降级）请求


class TestPins(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            scope = Path(td)
            self.assertIsNone(D._load_pin(scope, "http://x/y.zip"))
            D._save_pin(scope, "http://x/y.zip", "a" * 64)
            self.assertEqual(D._load_pin(scope, "http://x/y.zip"), "a" * 64)

    def test_corrupt_or_malformed_pin_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            f = D._pins_path(Path(td))
            f.write_text("not json", encoding="utf-8")
            self.assertIsNone(D._load_pin(Path(td), "http://x"))
            f.write_text(json.dumps({"http://x": "short"}), encoding="utf-8")
            self.assertIsNone(D._load_pin(Path(td), "http://x"))


def ssl_cert_error():
    import ssl
    try:
        return ssl.SSLCertVerificationError(
            "CERTIFICATE_VERIFY_FAILED", "self signed certificate")
    except AttributeError:   # 极老 Python 兜底（项目要求 3.10+，不会走到）
        return OSError("CERTIFICATE_VERIFY_FAILED")


class _Server(ThreadingHTTPServer):
    """可编程回环服务器：按 payload/mode 应答。daemon_threads = True。"""
    daemon_threads = True
    payload = b""
    declare_extra = 0        # Content-Length 多声明的字节数（模拟静默截断）
    support_range = False    # 是否支持 Range（断点续传 206）


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):      # 静默测试输出
        pass

    def do_GET(self):
        payload = type(self.server).payload
        rng = self.headers.get("Range")
        if rng and type(self.server).support_range and rng.startswith("bytes="):
            start = int(rng.removeprefix("bytes=").split("-")[0])
            body = payload[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        total = len(payload) + type(self.server).declare_extra
        self.send_response(200)
        self.send_header("Content-Length", str(total))
        self.end_headers()
        self.wfile.write(payload)


class TestDownloadFileHttp(unittest.TestCase):
    """经真实回环 HTTP 验证 _download_file 的完整性与断点续传。"""

    def setUp(self):
        self.srv = _Server(("127.0.0.1", 0), _Handler)
        self.port = self.srv.server_address[1]
        self._thread = None
        self.url = f"http://127.0.0.1:{self.port}/core.zip"   # 非 HF 形态 → TOFU
        self._td = tempfile.TemporaryDirectory()
        self.scope = Path(self._td.name)
        self.dest = self.scope / "core.zip"

    def tearDown(self):
        if self._thread:
            self.srv.shutdown()
            self._thread.join(timeout=5)
        self._td.cleanup()

    def _serve(self, payload, declare_extra=0, support_range=False):
        type(self.srv).payload = payload
        type(self.srv).declare_extra = declare_extra
        type(self.srv).support_range = support_range
        import threading
        self._thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self._thread.start()

    def test_success_and_tofu_pin_recorded(self):
        payload = b"crispasr-core" * 1000
        self._serve(payload)
        D._download_file(self.url, self.dest, pin_scope=self.scope)
        self.assertEqual(self.dest.read_bytes(), payload)
        pin = D._load_pin(self.scope, self.url)
        self.assertIsInstance(pin, str)
        self.assertEqual(pin, D._sha256(self.dest))

    def test_tofu_tamper_detected_on_redownload(self):
        """同 URL 内容被替换（同长度）→ 哈希与首下载 pin 不符，删档报错。"""
        import hashlib
        self._serve(b"A" * 4096)
        D._download_file(self.url, self.dest, pin_scope=self.scope)
        first = self.dest.read_bytes()
        first_hash = hashlib.sha256(first).hexdigest()
        self.dest.unlink()
        self._serve(b"B" * 4096)     # 同长度、不同内容
        with self.assertRaises(RuntimeError) as ctx:
            D._download_file(self.url, self.dest, pin_scope=self.scope)
        self.assertFalse(self.dest.exists())          # 残档已删除
        self.assertIn("首次下载", str(ctx.exception))
        # pin 本身未被篡改值覆盖
        self.assertEqual(D._load_pin(self.scope, self.url), first_hash)

    def test_truncated_download_deleted(self):
        """声明长度 > 实发字节（连接中断静默截断）→ 删档抛错。"""
        payload = b"Z" * 8192
        self._serve(payload, declare_extra=1024)
        with self.assertRaises(RuntimeError) as ctx:
            D._download_file(self.url, self.dest)
        self.assertFalse(self.dest.exists())
        self.assertIn("只收到", str(ctx.exception))

    def test_resume_completes_partial_file(self):
        """断点续传：本地已有前半段 + 服务器支持 Range → 补全后内容完整。"""
        payload = bytes(range(256)) * 64        # 16KB，可切片
        self._serve(payload, support_range=True)
        partial = self.scope / "core.zip"
        partial.write_bytes(payload[:8000])     # 预置残档（模拟上次中断）
        D._download_file(self.url, self.dest)
        self.assertEqual(self.dest.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()

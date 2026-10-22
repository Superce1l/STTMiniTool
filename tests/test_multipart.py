"""webview_server 流式 multipart 解析：边界情况与内容精确性"""
import tempfile
import unittest
from pathlib import Path

from common import *  # noqa: F401,F403

from webview_server import _parse_multipart_stream

BND = b"----WebKitFormBoundaryABC123"


def build(file_bytes, fname="audio.wav", with_fields=True):
    parts = [b"--" + BND + b"\r\n"]
    if with_fields:
        parts += [b'Content-Disposition: form-data; name="language"\r\n\r\n',
                  b"Chinese\r\n--" + BND + b"\r\n",
                  b'Content-Disposition: form-data; name="diarize"\r\n\r\n1\r\n--',
                  BND, b"\r\n"]
    parts += [b'Content-Disposition: form-data; name="file"; filename="',
              fname.encode(), b'"\r\nContent-Type: audio/wav\r\n\r\n',
              file_bytes, b"\r\n--" + BND, b"--\r\n"]
    return b"".join(parts)


class _ChunkedReader:
    """按固定块大小吐出 body，模拟任意分包的 socket 读取"""

    def __init__(self, body, chunk):
        self.body, self.chunk = body, chunk

    def read(self, n):
        take = min(n, self.chunk)
        r, self.body = self.body[:take], self.body[take:]
        return r


class TestMultipartStream(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _parse(self, body, chunk=1 << 20):
        return _parse_multipart_stream(_ChunkedReader(body, chunk), BND,
                                       self.dir, len(body))

    def test_binary_exact_across_chunk_sizes(self):
        data = bytes(range(256)) * 41 + b"\r\n\xff\x00tail\r\n"
        body = build(data)
        for chunk in (1, 3, 7, 64, 1 << 20):
            fields, files = self._parse(body, chunk)
            self.assertEqual(fields, {"language": "Chinese", "diarize": "1"})
            self.assertEqual(files["file"][0], "audio.wav")
            self.assertEqual(files["file"][1].read_bytes(), data, f"chunk={chunk}")

    def test_false_boundary_inside_content(self):
        # 内容里出现「边界前缀但后面不是 \r\n/--」→ 按内容处理
        tricky = b"hello\r\n--" + BND[:10] + b"\r\nworld\r\n--" + BND + b"X"
        _f, files = self._parse(build(tricky), chunk=5)
        self.assertEqual(files["file"][1].read_bytes(), tricky)

    def test_empty_file(self):
        _f, files = self._parse(build(b""))
        self.assertEqual(files["file"][1].stat().st_size, 0)

    def test_file_before_field(self):
        parts = [b"--" + BND + b"\r\n",
                 b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n',
                 b"BIN\r\nDATA\r\n", b"--" + BND + b"\r\n",
                 b'Content-Disposition: form-data; name="hint"\r\n\r\n',
                 "歌词，测试".encode("utf-8") + b"\r\n", b"--" + BND + b"--\r\n"]
        body = b"".join(parts)
        fields, files = self._parse(body, chunk=2)
        self.assertEqual(fields["hint"], "歌词，测试")
        self.assertEqual(files["file"][1].read_bytes(), b"BIN\r\nDATA")

    def test_no_boundary_garbage(self):
        fields, files = self._parse(b"not multipart at all")
        self.assertEqual((fields, files), ({}, {}))

    def test_filename_suffix_sanitized(self):
        _f, files = self._parse(build(b"x", fname='weird.<>?".wav'))
        self.assertTrue(str(files["file"][1]).endswith(".wav"))


if __name__ == "__main__":
    unittest.main()

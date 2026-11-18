"""webview_server._client_disconnected — 断连看门狗的探测语义

_transcribe 在转录期间每秒探测一次连接：对端关闭（浏览器 abort／关标签页／
关窗）→ 立刻取消转录并记日志（反馈 #4/#5：「界面已报错、后端还在烧 GPU」）。
用 socketpair 覆盖三种情形：连接在线、有数据在途（keep-alive 上的后续请求，
不得误判为断连）、对端已关闭。
"""
import socket
import time
import unittest

from common import *  # noqa: F401,F403

from webview_server import _client_disconnected


class TestClientDisconnected(unittest.TestCase):
    def setUp(self):
        self.a, self.b = socket.socketpair()

    def tearDown(self):
        for s in (self.a, self.b):
            try:
                s.close()
            except OSError:
                pass

    def test_open_connection_is_alive(self):
        self.assertFalse(_client_disconnected(self.a))

    def test_pending_data_is_not_disconnect(self):
        # keep-alive 上恰好有下一请求的数据 → 视为仍在线（否则会误杀正在收结果的请求）
        self.b.sendall(b"x")
        self.assertFalse(_client_disconnected(self.a))
        self.assertEqual(self.a.recv(1), b"x")

    def test_peer_close_is_disconnect(self):
        self.b.close()
        time.sleep(0.1)          # 让 FIN 到达对端
        self.assertTrue(_client_disconnected(self.a))


if __name__ == "__main__":
    unittest.main()

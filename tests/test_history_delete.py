"""最近转录：删除功能（回收站 + 路径防穿越）"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

import app as core
from webview_backend import WebBackend

PROJECT = Path(__file__).resolve().parents[1]


class _HistCase(unittest.TestCase):
    def setUp(self):
        self._prev_srt = core.SRT_DIR
        self.d = Path(tempfile.mkdtemp()) / "subtitles"
        self.d.mkdir(parents=True)
        core.SRT_DIR = self.d
        self._prev_settings = core.SETTINGS_FILE
        core.SETTINGS_FILE = Path(tempfile.mkdtemp()) / "settings.json"
        core.SETTINGS_FILE.write_text("{}", encoding="utf-8")   # 自备夹具，不读实时设置
        self.be = WebBackend()

    def tearDown(self):
        core.SRT_DIR = self._prev_srt
        core.SETTINGS_FILE = self._prev_settings


class TestDeleteHistory(_HistCase):
    def test_delete_removes_file_and_refreshes_list(self):
        f = self.d / "a [X · Y].srt"
        f.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        self.assertTrue(f.exists())
        r = self.be.delete_history(f.name)
        self.assertTrue(r["ok"], r)
        self.assertFalse(f.exists())
        self.assertEqual(self.be.list_history()["items"], [])

    def test_path_traversal_rejected(self):
        secret = self.d.parent / "secret.srt"
        secret.write_text("x", encoding="utf-8")
        for bad in ("../secret.srt", "..\\secret.srt", "/abs/path.srt",
                    "sub/../../secret.srt", "", ".", ".."):
            r = self.be.delete_history(bad)
            self.assertFalse(r["ok"], bad)
        self.assertTrue(secret.exists())        # 防穿越：目标文件未被动过

    def test_missing_file(self):
        r = self.be.delete_history("不存在.srt")
        self.assertFalse(r["ok"])

    def test_special_characters_in_name(self):
        # 单引号/空格/中文/方括号——PowerShell 命令的转义必须扛住（" 在
        # Windows 文件名中非法，不测）
        name = "歌'词 测试 [CrispASR · Base].srt"
        f = self.d / name
        f.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        r = self.be.delete_history(name)
        self.assertTrue(r["ok"], r)
        self.assertFalse(f.exists())

    def test_recycled_flag_type(self):
        f = self.d / "flag.txt.srt"
        f.write_text("x", encoding="utf-8")
        r = self.be.delete_history(f.name)
        self.assertTrue(r["ok"])
        self.assertIsInstance(r["recycled"], bool)


if __name__ == "__main__":
    unittest.main()

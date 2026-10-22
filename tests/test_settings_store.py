"""SettingsStore：锁、原子写、缓存失效、损坏恢复"""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from common import *  # noqa: F401,F403

from settings_store import SettingsStore


class TestSettingsStore(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "settings.json"
        self.store = SettingsStore(self.path)

    def test_empty_read(self):
        self.assertEqual(self.store.read(), {})

    def test_update_merge_and_remove(self):
        self.store.update({"backend": "crispasr", "n": 1})
        self.store.update({"n": 2})
        self.assertEqual(self.store.read(), {"backend": "crispasr", "n": 2})
        self.store.remove("n")
        self.assertEqual(self.store.read(), {"backend": "crispasr"})
        self.store.remove("不存在的键")            # 忽略、不报错

    def test_atomic_no_tmp_left(self):
        self.store.update({"a": 1})
        self.assertFalse(Path(str(self.path) + ".tmp").exists())
        # 落盘内容合法 JSON
        json.loads(self.path.read_text(encoding="utf-8"))

    def test_concurrent_writes_no_loss(self):
        def w(i):
            for j in range(30):
                self.store.update({f"k{i}": j})
        ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        r = self.store.read()
        for i in range(4):
            self.assertIn(f"k{i}", r)

    def test_corrupt_file_recovers(self):
        self.store.update({"a": 1})
        self.path.write_text("{corrupt", encoding="utf-8")
        self.assertEqual(self.store.read(), {})
        self.store.update({"b": 2})               # 损坏后可继续写
        self.assertEqual(self.store.read(), {"b": 2})

    def test_external_edit_picked_up(self):
        self.store.update({"a": 1})
        self.path.write_text(json.dumps({"a": 2}), encoding="utf-8")
        self.assertEqual(self.store.read(), {"a": 2})

    def test_non_dict_content(self):
        self.path.write_text('[1, 2]', encoding="utf-8")
        self.assertEqual(self.store.read(), {})


if __name__ == "__main__":
    unittest.main()

"""check_update 的版本比较（_ver_newer）：semver 语义 + 预发布号处理

锁定的契约：
  • 逐段数值比较（"2.10" > "2.9"，不是字典序）；
  • 预发布号低于同数值正式版（"2.3.0-rc.1" < "2.3.0"，semver 规则——
    旧实现把 "-rc.1" 拆成数值段记 0，误判预发布更新）；
  • 构建元数据（+build.1）不参与比较；
  • 非法/空串回 False（不误报有更新）。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # common（单跑时）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # 项目根

from webview_backend import WebBackend

newer = WebBackend._ver_newer


class TestVerNewer(unittest.TestCase):
    def test_numeric_segment_compare(self):
        self.assertTrue(newer("2.10", "2.9"))          # 旧实现的回归点（字典序会错）
        self.assertTrue(newer("2.3.1", "2.3"))
        self.assertFalse(newer("2.3", "2.3.0"))        # 补零相等

    def test_prerelease_older_than_release(self):
        self.assertFalse(newer("2.3.0-rc.1", "2.3.0"))
        self.assertFalse(newer("2.3.0-beta2", "2.3.0"))
        self.assertTrue(newer("2.3.1", "2.3.0-rc.1"))  # 下一正式版高于预发布

    def test_build_metadata_ignored(self):
        self.assertFalse(newer("2.3.0+build.1", "2.3.0"))
        self.assertFalse(newer("2.3.0", "2.3.0+build.1"))

    def test_equal_versions_not_newer(self):
        self.assertFalse(newer("2.3.0", "2.3.0"))
        self.assertFalse(newer("v2.3.0", "2.3.0"))     # v 前缀归一
        self.assertFalse(newer("2.3.0-rc.1", "2.3.0-rc.1"))

    def test_garbage_is_not_newer(self):
        self.assertFalse(newer("", "2.3.0"))
        self.assertFalse(newer("abc", "2.3.0"))

    def test_shorter_sides_pad_with_zero(self):
        self.assertTrue(newer("3", "2.9.9"))
        self.assertFalse(newer("2.3.0-rc.1", "2.3.0-rc"))


if __name__ == "__main__":
    unittest.main()

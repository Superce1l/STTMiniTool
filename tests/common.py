"""tests/ — 单元测试（stdlib unittest，无新增依赖）

运行：.venv/Scripts/python -m unittest discover -s tests -v
覆盖：字幕分行、切片计划与合并、SRT 解析、分段工具、SettingsStore、
流式 multipart 解析、引擎契约、VAD 批量等价性（有随包模型时）。
"""
import sys
from pathlib import Path

# 让测试可以从任意 CWD 运行：把项目根加进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

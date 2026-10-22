"""engine_types.py — 引擎层共享的结果类型（app.py 与 crisp_engine 共用）

放在独立小模块而非 app.py：app.py 在 import 时会尝试加载 crisp_engine，
crisp_engine 若回过头 import app 会形成循环导入。subtitle_lines 只管字幕
分行，结果对象的归属放这里最干净。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscribeResult:
    """process_file 的结构化返回值（取代旧的实例属性侧通道）。

    旧实现把字级 segments、诊断信息挂在引擎实例上（`_last_segments_rich` /
    `_last_vad_diag`），由调用方事后 getattr 回读——残留清理时序出过两轮
    bug。现在一次调用一次结果：
      output        : 字幕输出路径；None = 未产出（无声／退化／取消）
      segments_rich : 字级 segments [{"start","end","text","speaker","words"}]
                      （卡拉OK逐字用）；None = 引擎未产出字级结构
      diag          : 未产出时的明确原因（VAD 诊断／退化输出说明），可空
      cancelled     : 因 cancel_event 置位而中途取消
      degenerate    : 被判为异常输出而略过的段数
    """
    output: Path | None = None
    segments_rich: list | None = None
    diag: str | None = None
    cancelled: bool = False
    degenerate: int = 0

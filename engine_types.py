"""engine_types.py — 引擎层共享的结果类型与常数（app.py 与 crisp_engine 共用）

放在独立小模块而非 app.py：app.py 在 import 时会尝试加载 crisp_engine，
crisp_engine 若回过头 import app 会形成循环导入。subtitle_lines 只管字幕
分行，结果对象与跨引擎常数放这里最干净。

2026-11 起兼收原本散落各处的重复常数/子进程样板（零依赖，任何模块可
安全导入；不得 import 项目内其他模块，避免循环导入）：
  • SAMPLE_RATE / VAD_THRESHOLD / FA_BIN_NAME
  • base_dir()                 — 冻结时 EXE 旁、否则源码目录
  • CREATE_NO_WINDOW / startup_info() — Windows 隐藏子进程控制台窗口样板

（不叫 common.py：tests/ 已有同名 fixture 模块，unittest discovery 时会
按 sys.path 顺序遮蔽本文件，导致 `from common import X` 导入失败。）
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 音频 / ASR 常数 ───────────────────────────────────────────────────
SAMPLE_RATE = 16000          # 全链路统一 16kHz mono（加载 / VAD / 引擎）

# VAD 判定阈值。app 侧不再开放设置：未检出时自动降灵敏度重试
# （见 app._vad_retry_thresholds）；chatllm_engine 的诊断文案引用同一基准。
VAD_THRESHOLD = 0.5

# chatllm 来源文件名沿用上游拼字（foced 非 forced），与下载来源一致。
FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

# ── Windows 子进程样板 ────────────────────────────────────────────────
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def base_dir() -> Path:
    """App 根目录（PyInstaller 冻结时为 EXE 旁，否则为源码目录）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def startup_info():
    """subprocess.STARTUPINFO（SW_HIDE，避免识别时画面闪烁）；非 Windows 回 None。"""
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        return si
    return None


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
      warnings      : 非致命降级说明（如字级对齐失败回退比例估算）；
                      有产出时也可非空，UI 可据此提示质量降级。可空。
    """
    output: Path | None = None
    segments_rich: list | None = None
    diag: str | None = None
    cancelled: bool = False
    degenerate: int = 0
    warnings: list[str] | None = None

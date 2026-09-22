"""
chatllm_engine.py — ASR 输出诊断辅助（原 chatllm 推理后端的共享函数）

chatllm（Vulkan）推理后端已移除；这里保留两个与推理无关的纯函数，
供 app.py 的 OpenVINO ASREngine 使用：
  • format_vad_diag — 「未检测到人声」的明确诊断
  • detect_degenerate_asr — 识别退化输出（复诵范本／高度重复）检测
"""
from __future__ import annotations

import re
from collections import Counter

VAD_THRESHOLD = 0.5


def format_vad_diag(stats: dict) -> str:
    """把 _detect_speech_groups 的 stats 转成「为什么没检测到人声」的明确说明。

    区分三种情况：音频过短 / 全段机率过低（纯背景音或门槛太高）/ 有讯号但段落太短。
    """
    n    = stats.get("n_chunks", 0)
    mp   = stats.get("max_prob", 0.0)
    th   = stats.get("threshold", VAD_THRESHOLD)
    nseg = stats.get("n_segments", 0)
    if n == 0:
        return "⚠ 未检测到人声：音频过短或为空档。"
    if mp < th:
        return (f"⚠ 未检测到人声：全段人声机率偏低（最高 {mp:.2f} < 门槛 {th:.2f}）。"
                f"可能为纯背景音／音乐，或门槛过高 —— 可至『设置 → VAD 灵敏度』调低后重试。")
    if nseg == 0:
        return (f"⚠ 未检测到人声：检测到语音频号（最高 {mp:.2f}）但每段都过短（< 0.5 秒）未成段。")
    return "⚠ 未检测到人声：无有效语音分段。"


def detect_degenerate_asr(text: str) -> str | None:
    """检测 ASR 退化输出（模型复诵提示范本 / 高度重复），返回原因字符串；正常回 None。

    部分不兼容 GPU 上模型会吐出复诵 system prompt 的垃圾，
    例如 'language en<asr_text>[transcription]' 或同字无限重复。这类内容不该被当成
    字幕写出，需明确判为推理失败（issue #30）。
    """
    t = (text or "").strip()
    if not t:
        return None   # 空字符串交由上层「未检测到人声／跳过」逻辑处理
    low = t.lower()
    # 1) 残留 prompt 范本 token（最明确的退化讯号）
    for tok in ("<asr_text", "asr_text>", "[transcription]", "[asr_text]"):
        if tok in low:
            return "模型输出残留提示范本（疑似未正常识别）"
    # 2) 整段就是语言标记 'language xx'（exact，避免误判 'Language is power'）
    if re.fullmatch(r"language\s+[a-z]{2}", low):
        return "模型仅复诵语言标记，未产生转录内容"
    # 2b) 反复复诵语言标记：要求短语占输出绝对主导（去空白后 <60 字符），
    #     避免误杀「一段提到 language 三次以上的正常英文讲座」。
    compact = re.sub(r"\s+", "", t)
    if low.count("language ") >= 3 and len(compact) < 60:
        return "模型反复复诵语言标记（疑似推理退化）"
    # 3) 单一字符高度重复
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 12 and max(Counter(compact).values()) / len(compact) >= 0.85:
        return "模型输出高度重复（疑似推理退化）"
    return None

"""subtitle_lines.py — 字级时间轴 → 字幕行（后端无关，全引擎共享）

把「字级 (word, start_sec, end_sec) + ASR 原文」转成字幕行的逻辑集中于此，
让 OpenVINO / CrispASR(Whisper) 等引擎产出**一致**的字幕断句与
时间轴（标点切行 + MAX_CHARS/MAX_WORDS 保护 + 孤儿行合并）。

公开符号：
    MAX_CHARS, _ZH_CLAUSE_END, _EN_SENT_END   断句常数
    _srt_ts(s)                                秒 → SRT 时间戳
    _merge_orphan_lines(lines)                合并过短孤儿行
    _ts_chatllm_to_subtitle_lines(...)        字级 list → [(start,end,text,spk)]
"""
from __future__ import annotations

import os
from pathlib import Path

# ── 断句常数（与旧 app.py 行为一致）──────────────────────────────────
MAX_CHARS      = 20
_ZH_CLAUSE_END = frozenset('，。？！；：…—、·')
_EN_SENT_END   = frozenset('.,!?;')

# ── 声学停顿切行（无标点时的补充句界信号）────────────────────────────
# ASR 偶尔在两个句子之间漏打标点，此后标点切行失效，唯一的换行依据只剩
# MAX_CHARS 强制保护 —— 切点与语句无关，产出「上一句尾巴＋下一句开头」
# 混在一行的字幕。字级时间轴上有现成的补充信号：字与字之间的空隙（停顿）。
# 句间停顿普遍 ≥0.3s、短语内几乎不留空隙，故「当前行已有一定长度＋停顿
# 够长」时主动切行，效果等同一个看不见的逗号，与「标点即切行」的设计一致。
PAUSE_BREAK_SEC = 0.35   # 字间空隙 ≥ 此值（秒）视为句界
PAUSE_MIN_ZH    = 5      # 停顿前行至少已累积的中文字数（防把短语切碎）
PAUSE_MIN_WORDS = 3      # 含拉丁词的行最少词数
MIN_SPLIT_GAP   = 0.15   # 超限回退切行时，词间空隙的最小可信值（秒）；
                         # 须明显大于连续语流的字间隙（≲0.1s）、小于句间停顿（≥0.3s）


def _strip_pua(s: str) -> str:
    """滤除 Unicode 私用区字符（BMP U+E000–U+F8FF 与两个补充平面）。

    合法的中文／日文／英文字幕不会用到私用区；出现在这里的一律是下游组件把
    「无法映射的内部 ID」渲染成字符的产物，属于乱码。滤掉比留著好——留著会
    一路污染 SRT、JSON 与外部 Agent。
    """
    if not s:
        return s
    return "".join(c for c in s
                   if not (0xE000 <= ord(c) <= 0xF8FF
                           or 0xF0000 <= ord(c) <= 0xFFFFD
                           or 0x100000 <= ord(c) <= 0x10FFFD))


def _srt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def srt_text(lines: list[tuple[float, float, str, str | None]]) -> str:
    """字幕行 → SRT 全文（含说话者前缀与空行分隔）。

    write_transcript 与长音频切片合并写出共用此单一构造点，
    避免「切片路径手写循环、常规路径走 write_transcript」两套格式漂移。
    """
    out = []
    for idx, (s, e, line, spk) in enumerate(lines, 1):
        prefix = f"{spk}：" if spk else ""
        out.append(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{prefix}{line}\n")
    return "\n".join(out) + ("\n" if out else "")


def atomic_write_text(path, text: str) -> None:
    """原子落盘文本：临时文件写完 → os.replace 原子替换。

    转录中途进程被杀（关窗 / /api/shutdown）不会留下半截 SRT——
    与 SettingsStore 同一模式；替换目标所在目录需已存在。
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ── ASS / WebVTT 导出 ──────────────────────────────────────────────────
# 与前端「另存」菜单同一套视觉规格（ASS 头沿用 app.buildAss 的 Style 行）；
# 后端出口供 CLI -f ass|vtt 与未来 UI 另存使用。

def _ass_ts(s: float) -> str:
    """秒 → ASS 时间戳 H:MM:SS.CS（小时不补零、厘秒两位，与前端 assTs 一致）。"""
    ms = max(0, int(round(s * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h}:{m:02d}:{sec:02d}.{ms // 10:02d}"


_ASS_HEADER = [
    "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
    "[V4+ Styles]",
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    "Style: Default,Noto Sans SC,72,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,2,1,2,80,80,80,1",
    "", "[Events]",
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
]


def ass_text(lines: list[tuple[float, float, str, str | None]]) -> str:
    """字幕行 → ASS（无卡拉OK；说话者前缀与 SRT 一致）。"""
    out = list(_ASS_HEADER)
    for (s, e, line, spk) in lines:
        text = f"{spk}：{line}" if spk else line
        out.append(f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Default,,0,0,0,,{text}")
    return "\n".join(out) + "\n"


def ass_karaoke_text(rich: list[dict]) -> str:
    """字级 segments（TranscribeResult.segments_rich 形态）→ ASS 卡拉OK。

    与前端 buildAss 同一算法：逐字/词 ``\\k`` 拍点（厘秒），下一字的起点钳止
    本字时长（防字级时间轴交叠出负值），最小 1 厘秒；无 words 的段退回整行
    纯文本。
    """
    out = list(_ASS_HEADER)
    for seg in (rich or []):
        spk = seg.get("speaker")
        prefix = f"{spk}：" if spk else ""
        words = seg.get("words") or []
        seg_end = float(seg.get("end", 0.0))
        if words:
            parts = []
            for i, w in enumerate(words):
                end_t = (float(words[i + 1]["start"])
                         if i < len(words) - 1 else seg_end)
                end_t = max(end_t, float(w.get("end", end_t)))
                k = max(1, round((end_t - float(w.get("start", 0.0))) * 100))
                parts.append("{\\k" + str(k) + "}" + str(w.get("text", "")))
            text = prefix + "".join(parts)
        else:
            text = prefix + str(seg.get("text", ""))
        out.append(f"Dialogue: 0,{_ass_ts(float(seg.get('start', 0.0)))},"
                   f"{_ass_ts(seg_end)},Default,,0,0,0,,{text}")
    return "\n".join(out) + "\n"


def vtt_text(lines: list[tuple[float, float, str, str | None]]) -> str:
    """字幕行 → WebVTT（句点毫秒分隔；说话者前缀与 SRT 一致）。"""
    def _vtt_ts(s: float) -> str:
        ms = max(0, int(round(s * 1000)))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        sec, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

    out = ["WEBVTT", ""]
    for idx, (s, e, line, spk) in enumerate(lines, 1):
        prefix = f"{spk}：" if spk else ""
        out.append(f"{idx}\n{_vtt_ts(s)} --> {_vtt_ts(e)}\n{prefix}{line}\n")
    return "\n".join(out)


# ── 全域输出格式（"srt" | "txt"）────────────────────────────────────────
# 由 app 启动 / 设置变更时同步（app._on_output_format_change → 改写此值）。
# write_transcript() 在 out_format=None 时读此值，使「批次 / 单档 / 录制」
# 三条路径全域一致；端点需内部解析 SRT，固定以 out_format="srt" 覆盖。
OUTPUT_FORMAT = "srt"


def lines_to_txt(lines: list[tuple[float, float, str, str | None]]) -> str:
    """字幕行 → 纯文字。

    • 无说话者：整段文字相连成一行。中文直接拼接（重建连续逐字稿）；
      相邻行首尾为拉丁字母时插入空格（英文行内以空格分词，
      行间直接拼接会把 "how are you"+"doing today" 黏成 "youdoing"）。
    • 有说话者：每段一行，保留「说话者N：」前缀，便于分辨发言者。
    """
    has_spk = any(spk for (_s, _e, _t, spk) in lines)
    if has_spk:
        return "\n".join(
            (f"{spk}：{t}" if spk else t) for (_s, _e, t, spk) in lines
        )

    def _latin(ch: str) -> bool:
        return bool(ch) and ch.isascii() and ch.isalpha()

    texts = [t for (_s, _e, t, _spk) in lines]
    if not texts:
        return ""
    out = texts[0]
    for t in texts[1:]:
        if _latin(out[-1:]) or _latin(t[:1]):
            out += " "
        out += t
    return out


def write_transcript(
    ref,
    lines: list[tuple[float, float, str, str | None]],
    out_format: str | None = None,
):
    """把字幕行依格式写成 .srt 或 .txt，返回实际输出路径（Path）。

    所有引擎（OpenVINO / CrispASR）与录制转换共享此单一写出点，
    确保全域输出格式一致。

    参数：
        ref        : 决定输出目录与主文件名的参考路径（通常为原始音频）。
        lines      : [(start_sec, end_sec, text, speaker|None), ...]
        out_format : "srt" | "txt" | "ass" | "vtt"；None 时采用全域 OUTPUT_FORMAT。
                     端点固定传 "srt"（内部需解析时间轴），不受全域影响。
                     ASS 卡拉OK（字级拍点）请改用 ass_karaoke_text(rich)。
    """
    from pathlib import Path
    fmt = (out_format or OUTPUT_FORMAT or "srt").lower()
    ref = Path(ref)
    if fmt == "txt":
        out = ref.parent / (ref.stem + ".txt")
        atomic_write_text(out, lines_to_txt(lines))
        return out
    if fmt == "vtt":
        out = ref.parent / (ref.stem + ".vtt")
        atomic_write_text(out, vtt_text(lines))
        return out
    if fmt == "ass":
        out = ref.parent / (ref.stem + ".ass")
        atomic_write_text(out, ass_text(lines))
        return out
    out = ref.parent / (ref.stem + ".srt")
    atomic_write_text(out, srt_text(lines))
    return out


def _merge_orphan_lines(
    lines: list[tuple[float, float, str, str | None]],
    min_chars: int = 1,
    max_gap: float = 0.8,
) -> list[tuple[float, float, str, str | None]]:
    """合并过短的孤立字幕行（如句尾「吧」单独成行）到相邻行。

    FA 断句时 MAX_WORDS 与标点切行偶尔会叠加，在子句中间切一刀，把
    句尾语助词（吧/啊/呢/了…）留成独立一行。此处在输出前把这类「孤儿行」
    并回相邻行：优先并入前一行（时间连续、同说话者），首行孤儿则并入下一行。
    含拉丁词时以空格 join，纯中文直接相接。

    默认仅并「单字」孤儿：单字几乎都是句尾语助词，向后并入前一行最安全；
    两字以上可能是句首短词，向后并易误接，故不处理。
    """
    if not lines:
        return lines

    def _has_latin(t: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in t)

    def _vlen(t: str) -> int:
        return len(t.replace(" ", ""))

    def _is_orphan(t: str) -> bool:
        # 纯中文且可见字数极少才视为孤儿；含拉丁词（英文/数字词）不并
        return (not _has_latin(t)) and 0 < _vlen(t) <= min_chars

    def _join(a: str, b: str) -> str:
        sep = " " if (_has_latin(a) or _has_latin(b)) else ""
        return f"{a}{sep}{b}"

    merged: list[tuple[float, float, str, str | None]] = []
    for (s, e, t, spk) in lines:
        if (_is_orphan(t) and merged
                and merged[-1][3] == spk
                and s - merged[-1][1] <= max_gap):
            ps, _pe, pt, pspk = merged[-1]
            merged[-1] = (ps, e, _join(pt, t), pspk)
        else:
            merged.append((s, e, t, spk))

    # 首行仍是孤儿（无前一行可并）→ 并入下一行
    if len(merged) >= 2 and _is_orphan(merged[0][2]):
        s0, _e0, t0, spk0 = merged[0]
        s1, e1, t1, spk1 = merged[1]
        if spk0 == spk1 and s1 - merged[0][1] <= max_gap:
            merged[1] = (s0, e1, _join(t0, t1), spk1)
            merged.pop(0)

    return merged


def _merge_orphan_lines_rich(
    lines: list[tuple],
    min_chars: int = 1,
    max_gap: float = 0.8,
) -> list[tuple]:
    """`_merge_orphan_lines` 的 5-tuple 版（卡拉OK字级用）。

    行为与 4-tuple 版完全相同（同样的孤儿判定、间隔/说话者条件），差别只在
    并行时连同第 5 元素 ``words`` 字级清单一起对接，使字幕卡与卡拉OK逐字
    高亮的断句保持一致。
    """
    if not lines:
        return lines

    def _has_latin(t: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in t)

    def _vlen(t: str) -> int:
        return len(t.replace(" ", ""))

    def _is_orphan(t: str) -> bool:
        return (not _has_latin(t)) and 0 < _vlen(t) <= min_chars

    def _join(a: str, b: str) -> str:
        sep = " " if (_has_latin(a) or _has_latin(b)) else ""
        return f"{a}{sep}{b}"

    merged: list[tuple] = []
    for (s, e, t, spk, w) in lines:
        if (_is_orphan(t) and merged
                and merged[-1][3] == spk
                and s - merged[-1][1] <= max_gap):
            ps, _pe, pt, pspk, pw = merged[-1]
            merged[-1] = (ps, e, _join(pt, t), pspk, pw + w)
        else:
            merged.append((s, e, t, spk, w))

    if len(merged) >= 2 and _is_orphan(merged[0][2]):
        s0, _e0, t0, spk0, w0 = merged[0]
        s1, e1, t1, spk1, w1 = merged[1]
        if spk0 == spk1 and s1 - merged[0][1] <= max_gap:
            merged[1] = (s0, e1, _join(t0, t1), spk1, w0 + w1)
            merged.pop(0)

    return merged


def _ts_chatllm_to_subtitle_lines(
    ts_items,
    raw_text: str,
    chunk_offset: float,
    spk: str | None,
    break_on_space: bool = False,
    with_words: bool = False,
):
    """字级 (word, start_sec, end_sec) + ASR 原文 → 字幕行。

    标点切行 + MAX_CHARS/MAX_WORDS 保护；word_list 直接取自字级时间轴，
    与时间 1:1 对应，后端无关（chatllm FA / Whisper 字级皆适用）。

    参数：
        ts_items: list[tuple[str, float, float]]  → (word, start_sec, end_sec)
        break_on_space: True 时把 raw_text 的「空白」也当切点。
            用于 Whisper（无标点，但以空白标记语句边界）→ 等同 Qwen 在标点切，
            逼近 Qwen 断句质量。Qwen/chatllm 路径维持 False（空白仅分隔拉丁词）。
        with_words: True 时每行多带一个 ``words`` 字级清单（卡拉OK逐字高亮用），
            返回 5-tuple ``(start, end, text, spk, words)``；
            ``words = [{"start": 秒, "end": 秒, "text": 显示字}, ...]``。
            默认 False → 维持既有 4-tuple 行为（SRT/批次/端点完全不受影响）。

    返回：
        with_words=False → list[(start, end, text, spk)]
        with_words=True  → list[(start, end, text, spk, words)]
    """
    _all_punct = _ZH_CLAUSE_END | _EN_SENT_END
    MAX_WORDS    = 8
    MAX_ZH_CHARS = MAX_CHARS
    # 内部一律以 5-tuple (start, end, text, spk, words) 累积，返回前依 with_words 决定保留与否
    result: list[tuple] = []

    if not ts_items or not raw_text.strip():
        return result

    word_list = [w for (w, _s, _e) in ts_items]
    n = len(ts_items)

    # 私用区（PUA）字符防线：这些不是任何模型的合法输出，而是下游组件把「无法
    # 映射的类别 ID」直接当成字符渲染的结果（实例：crispasr 的 qwen3 CTC 对齐器
    # 遇到其类别表外的繁体字，会输出 U+E000+class_id）。它们对使用者是乱码，
    # 也会污染下游（SRT／JSON／Agent）。在唯一的文字出口统一滤掉，任何引擎都受惠。
    def _conv(s: str) -> str:
        return _strip_pua(s)

    seg_idx:   list[int] = []   # 当前行的 ts_items 索引
    seg_words: list[str] = []   # 当前行的原始 word
    ri = 0                      # raw_text 扫描位置

    def _is_latin_word(w: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in w)

    def _emit():
        nonlocal seg_idx, seg_words
        if not seg_idx:
            seg_idx = []; seg_words = []
            return
        start = chunk_offset + ts_items[seg_idx[0]][1]
        end   = chunk_offset + ts_items[seg_idx[-1]][2]
        if any(_is_latin_word(w) for w in seg_words):
            text = " ".join(seg_words)
        else:
            text = "".join(seg_words)
        text = _conv(text)
        # 字级清单：每个 ts_item → 一个高亮单位（中文＝单字、拉丁＝整词）
        words = [
            {"start": chunk_offset + ts_items[k][1],
             "end":   chunk_offset + ts_items[k][2],
             "text":  _conv(word_list[k])}
            for k in seg_idx
        ]
        if end > start and text.strip():
            result.append((start, end, text.strip(), spk, words))
        seg_idx = []; seg_words = []

    def _over_limit() -> bool:
        if any(_is_latin_word(w) for w in seg_words):
            return len(seg_words) > MAX_WORDS
        return sum(len(w) for w in seg_words) > MAX_ZH_CHARS

    def _pause_break(gap: float) -> bool:
        """行已具一定长度 + 字间空隙够长 → 停顿处主动切行。"""
        if gap < PAUSE_BREAK_SEC or not seg_idx:
            return False
        if any(_is_latin_word(w) for w in seg_words):
            return len(seg_words) >= PAUSE_MIN_WORDS
        return sum(len(w) for w in seg_words) >= PAUSE_MIN_ZH

    def _emit_overflow():
        """超限切行：优先回退到行内「最长且像句界」的停顿处切开，让断点尽量
        贴近语义边界；行内全无停顿时才在溢出点直接切。候选停顿须位于行的前
        1/3 之后，避免把行切成两字孤儿＋一长串。"""
        nonlocal seg_idx, seg_words
        best_k, best_gap = -1, MIN_SPLIT_GAP
        lo = max(1, len(seg_idx) // 3)
        for k in range(lo, len(seg_idx)):
            gap = float(ts_items[seg_idx[k]][1]) - float(ts_items[seg_idx[k - 1]][2])
            if gap >= best_gap:            # ≥，取最靠后的最大停顿
                best_k, best_gap = k, gap
        if best_k > 0:
            tail_idx, tail_words = seg_idx[best_k:], seg_words[best_k:]
            seg_idx, seg_words = seg_idx[:best_k], seg_words[:best_k]
            _emit()
            seg_idx, seg_words = tail_idx, tail_words
        else:
            _emit()

    for wi in range(n):
        word = word_list[wi]

        # 在 raw_text 中前进到 word 位置；遇到标点（或 whisper 空白）→ 先切行
        hit_punct = False
        while ri < len(raw_text):
            c = raw_text[ri]
            if c in _all_punct:
                hit_punct = True; ri += 1; continue
            if c == " ":
                if break_on_space:
                    hit_punct = True   # whisper：空白＝语句边界，视同切点
                ri += 1; continue
            break

        if hit_punct:
            _emit()
        elif wi > 0:
            # 标点缺席时以声学停顿补位：本 word 距上一 word 的空隙够长、
            # 且当前行已有一定长度 → 在停顿处切行，避免句间无标点时靠
            # MAX_CHARS 硬切产出「两句混一行」。
            gap = float(ts_items[wi][1]) - float(ts_items[wi - 1][2])
            if _pause_break(gap):
                _emit()

        seg_idx.append(wi)
        seg_words.append(word)

        # 跳过 word 在 raw_text 中占用的字符（只计内容字符，忽略标点/空格）。
        # word 本身可能含标点（数字 3.5、或粘连标点的 token）：若以 len(word)
        # 计数，会多吞下一个词的字符，让其后的切点整体错位一格。
        consumed = 0
        word_len = sum(1 for ch in word if ch not in _all_punct and ch != " ")
        while ri < len(raw_text) and consumed < word_len:
            c = raw_text[ri]
            if c in _all_punct or c == " ":
                ri += 1; continue
            ri += 1; consumed += 1

        if _over_limit():
            _emit_overflow()

    _emit()
    merged = _merge_orphan_lines_rich(result)
    if with_words:
        return merged
    return [(s, e, t, spk) for (s, e, t, spk, _w) in merged]

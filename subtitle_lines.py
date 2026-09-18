"""subtitle_lines.py — 字级时间轴 → 字幕行（后端无关，全引擎共享）

把「字级 (word, start_sec, end_sec) + ASR 原文」转成字幕行的逻辑集中于此，
让 OpenVINO / chatllm / CrispASR(Whisper) 三种引擎产出**一致**的字幕断句与
时间轴（标点切行 + MAX_CHARS/MAX_WORDS 保护 + 孤儿行合并）。

公开符号：
    MAX_CHARS, _ZH_CLAUSE_END, _EN_SENT_END   断句常数
    _srt_ts(s)                                秒 → SRT 时间戳
    _merge_orphan_lines(lines)                合并过短孤儿行
    _ts_chatllm_to_subtitle_lines(...)        字级 list → [(start,end,text,spk)]
"""
from __future__ import annotations

# ── 断句常数（与旧 app.py 行为一致）──────────────────────────────────
MAX_CHARS      = 20
_ZH_CLAUSE_END = frozenset('，。？！；：…—、·')
_EN_SENT_END   = frozenset('.,!?;')


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

    所有引擎（OpenVINO / chatllm / CrispASR）与录制转换共享此单一写出点，
    确保全域输出格式一致。

    参数：
        ref        : 决定输出目录与主文件名的参考路径（通常为原始音频）。
        lines      : [(start_sec, end_sec, text, speaker|None), ...]
        out_format : "srt" | "txt"；None 时采用全域 OUTPUT_FORMAT。
                     端点固定传 "srt"（内部需解析时间轴），不受全域影响。
    """
    from pathlib import Path
    fmt = (out_format or OUTPUT_FORMAT or "srt").lower()
    ref = Path(ref)
    if fmt == "txt":
        out = ref.parent / (ref.stem + ".txt")
        out.write_text(lines_to_txt(lines), encoding="utf-8")
        return out
    out = ref.parent / (ref.stem + ".srt")
    with open(out, "w", encoding="utf-8") as f:
        for idx, (s, e, line, spk) in enumerate(lines, 1):
            prefix = f"{spk}：" if spk else ""
            f.write(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{prefix}{line}\n\n")
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
    cc,
    simplified: bool,
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

    # 繁化：text（整行）沿用既有「整行转换」确保 SRT 输出零变化；
    # words[].text 走「逐字转换」（卡拉OK高亮单位）——两者在极少数 s2twp
    # 词组转换情境可能有细微差异，但卡拉OK检视只读 words，不影响字幕卡/SRT。
    def _conv(s: str) -> str:
        s = _strip_pua(s)
        return cc.convert(s) if (not simplified and cc is not None) else s

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

        seg_idx.append(wi)
        seg_words.append(word)

        # 跳过 word 在 raw_text 中占用的字符（依长度计数，忽略标点/空格）
        consumed = 0
        word_len = len(word)
        while ri < len(raw_text) and consumed < word_len:
            c = raw_text[ri]
            if c in _all_punct or c == " ":
                ri += 1; continue
            ri += 1; consumed += 1

        if _over_limit():
            _emit()

    _emit()
    merged = _merge_orphan_lines_rich(result)
    if with_words:
        return merged
    return [(s, e, t, spk) for (s, e, t, spk, _w) in merged]

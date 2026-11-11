"""slicing.py — 长音频切片转录的纯函数层（webview_backend 拆分，2026-11）

超长音频一次性塞给引擎会造成：内存峰值高、OpenVINO 的 KV-Cache 线性膨胀、
转录中途失败要整段重来。先用 ffmpeg 把音频按等长窗口切片（相邻片段重叠
SLICE_OVERLAP 秒，避免切点把一个字/一句话劈成两半造成漏识），逐片转录，
再把各片 SRT 解析后按片起点平移、重叠区去重合并成整份结果。

本模块只含纯函数与常数（切片计划、合并去重、SRT 解析、跨片说话者匹配），
不持有任何引擎/服务状态——WebBackend 的调度层（_transcribe_sliced）留在
webview_backend，从本模块导入这些构件。单元测试经 webview_backend 的
兼容再导出导入，也可直接 import 本模块。

从 webview_backend 原样搬移；行为不变。
"""
from __future__ import annotations

import sys
from pathlib import Path

_SLICE_THRESHOLD_SECS = 2 * 3600     # 超过 2 小时启用切片
_SLICE_WINDOW_SECS = 30 * 60         # 每片 30 分钟（内存充足时的上限）
_SLICE_OVERLAP_SECS = 15             # 相邻片段重叠 15 秒

# ── 切片时长自适应（依可用内存）────────────────────────────
# 内存依据：切片解码进内存后是 16kHz float32 mono（64KB/s），VAD/引擎处理
# 还会产生数份拷贝（按 ~5 倍估算）。默认 30 分钟片 ≈ 110MB 原始 ≈ 550MB
# 实占，8GB 空闲内存的机器毫无压力；空闲内存只有几百 MB 的机器若仍切
# 30 分钟，转录中易触发系统交换甚至 OOM。故按可用物理内存收缩片长：
#   预算 = 可用内存 × 25%（给引擎/OS 留大头）→ 秒数 = 预算 ÷ 每秒实占
# 夹紧 [5 分钟, 30 分钟]：下限保证重叠区（15s）不退化，上限即原行为。
_SLICE_MEM_BUDGET_FRAC = 0.25
_SLICE_MEM_PER_SEC = 65536 * 5       # 64KB/s 原始 × ~5 份处理拷贝
_SLICE_WINDOW_MIN_SECS = 5 * 60


def _avail_phys_bytes() -> int | None:
    """可用物理内存（字节）；Windows GlobalMemoryStatusEx，失败回 None。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return int(st.ullAvailPhys)
    except Exception:
        return None


def _slice_window_secs() -> int:
    """依可用物理内存智能决定切片时长（秒）。

    内存充足 → 30 分钟（与原固定值一致，行为不变）；空闲内存紧张时按
    预算公式收缩，最低 5 分钟。查询失败（非 Windows 等）回默认上限。
    """
    avail = _avail_phys_bytes()
    if not avail:
        return _SLICE_WINDOW_SECS
    budget = avail * _SLICE_MEM_BUDGET_FRAC
    secs = int(budget / _SLICE_MEM_PER_SEC)
    return max(_SLICE_WINDOW_MIN_SECS, min(_SLICE_WINDOW_SECS, secs))


def _ffprobe_duration(ff: Path, audio: Path) -> float | None:
    """ffprobe 读取音频总时长（秒）；失败返回 None。"""
    ffprobe = ff.parent / ("ffprobe.exe" if ff.suffix.lower() == ".exe"
                           else "ffprobe")
    if not ffprobe.exists():
        ffprobe = ff.with_name("ffprobe" + ff.suffix)
    try:
        import subprocess
        r = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            capture_output=True, text=True, timeout=30,
            creationflags=0x08000000 if sys.platform == "win32" else 0)
        return float(r.stdout.strip())
    except Exception:
        return None


def _slice_plan(duration: float, window_secs: int | None = None) -> list[tuple[float, float]]:
    """总时长 → [(片起点, 片终点)]，相邻片重叠 _SLICE_OVERLAP_SECS。

    window_secs 省略时依可用内存自适应（_slice_window_secs）。
    """
    window = window_secs if window_secs is not None else _slice_window_secs()
    step = window - _SLICE_OVERLAP_SECS
    spans = []
    start = 0.0
    while start < duration - 0.5:
        end = min(start + window, duration)
        spans.append((start, end))
        if end >= duration:
            break
        start += step
    return spans or [(0.0, duration)]


def _merge_sliced_segments(all_parts: list[list[dict]]) -> list[dict]:
    """各片 segments（已含绝对时间）→ 去重合并。

    重叠区（相邻片 15 秒）里两片会识别出同一句；跨越片边界的句子必在其中
    一侧被截断：跨片尾的句子在前一片是前半、后一片完整；跨片首的句子在
    前一片完整、后一片只剩尾段。同一句只保留**更完整的版本**（文字更长者，
    其次时长更长者；完全等同取后一片）——无条件取某一片都会丢字：取后一片
    会丢掉「后一片自其起点截断的尾段」，取前一片则会丢掉「前一片自其终点
    截断的前半」。
    """
    if not all_parts:
        return []

    def _dur(seg: dict) -> float:
        return max(0.0, seg["end"] - seg["start"])

    def _norm(t: str) -> str:
        return "".join(str(t).split())

    def _is_dup(a: dict, b: dict) -> bool:
        ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
        if ov <= 0:
            return False
        ta, tb = _norm(a.get("text", "")), _norm(b.get("text", ""))
        if ta and ta == tb:
            return True
        shorter = min(_dur(a), _dur(b))
        return shorter > 0 and ov > 0.4 * shorter   # 重叠过半＝同一句（容时间抖动）

    def _prefer(a: dict, b: dict) -> dict:
        """同一句的两个版本 → 更完整者；平手取 b（后一片，时间戳距片缘较远）。"""
        ta, tb = _norm(a.get("text", "")), _norm(b.get("text", ""))
        if len(ta) != len(tb):
            return a if len(ta) > len(tb) else b
        da, db = _dur(a), _dur(b)
        if abs(da - db) > 0.05:
            return a if da > db else b
        return b

    merged = list(all_parts[0])
    for part in all_parts[1:]:
        if not part:
            continue
        out: list[dict] = []
        i = 0                       # merged 中尚未确定归属的行指针
        for seg in part:
            while i < len(merged) and merged[i]["end"] <= seg["start"]:
                out.append(merged[i])              # 早于 seg 结束 → 不可能再重复
                i += 1
            hit = False
            j = i
            while j < len(merged) and merged[j]["start"] < seg["end"]:
                if _is_dup(merged[j], seg):
                    merged[j] = _prefer(merged[j], seg)
                    hit = True
                    break
                j += 1
            if not hit:
                out.append(seg)
        out.extend(merged[i:])
        out.sort(key=lambda s: (s["start"], s["end"]))
        merged = out
    return merged


# 跨片说话者匹配阈值：声纹中心 cosine 相似度 ≥ 此值视为同一人。
# 跨片匹配同一人要比 diarize.SPLIT_SURE_DIST（0.38 → sim 0.62）宽松——
# 同一人跨片的中心漂移大于片内子群，取 sim ≥ 0.70 折衷误并/误开两类错误。
_SPK_MATCH_SIM = 0.70


def _match_speaker(emb, global_spk: list) -> int:
    """把一个片内声纹中心匹配到全局说话者；与谁都够不像则新开一人。

    global_spk: [{"label": int, "emb": ndarray, "n": 样本数}]（中心就地更新）
    返回全局说话者编号（1 起）。跨片映射的关键：每片独立聚类产生的「说话者1」
    编号互不相干，唯有声纹可比——匹配上同一全局中心才能保证整份字幕里
    同一人始终是同一个编号。
    """
    import numpy as _np
    best, best_sim = None, _SPK_MATCH_SIM
    for g in global_spk:
        sim = float(emb @ g["emb"])
        if sim > best_sim:
            best, best_sim = g, sim
    if best is None:
        label = (global_spk[-1]["label"] + 1) if global_spk else 1
        global_spk.append({"label": label, "emb": emb, "n": 1})
        return label
    # 运行中心更新：同一个人跨片的声纹微差取滑动平均，后续片匹配更稳
    n = best["n"] + 1
    c = best["emb"] * best["n"] + emb
    nrm = float(_np.linalg.norm(c))
    best["emb"] = c / nrm if nrm > 1e-9 else c
    best["n"] = n
    return best["label"]


def assign_speaker_by_overlap(start: float, end: float,
                              spans: list[tuple]) -> str | None:
    """字幕行 [start, end) → 说话者标签：取**重叠最长**的说话者段。

    旧实现按「行时间中点落点」指派：跨界的行（前半属 A、后半属 B）整行归
    给中点一侧，听感随机。改为重叠最长段更稳；行与任何段都无交集时按旧
    规则取时间最近的段兜底；spans 为空返回 None。
    spans: [(start, end, label), ...]
    """
    best, best_ov = None, 0.0
    for (t0, t1, lab) in spans:
        ov = min(end, t1) - max(start, t0)
        if ov > best_ov:
            best, best_ov = lab, ov
    if best is not None:
        return best
    if not spans:
        return None
    mid = (start + end) / 2.0
    return min(spans, key=lambda g: abs((g[0] + g[1]) / 2.0 - mid))[2]


def _srt_ts_to_sec(ts: str) -> float:
    """'00:00:01,560' → 1.56 秒。"""
    ts = ts.strip().replace(".", ",")
    hms, _, ms = ts.partition(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def _parse_srt(srt_text: str) -> list[dict]:
    """解析 SRT 文字为 [{"start","end","text"}]（原 api_server._parse_srt 的内联版，
    端点服务移除后仅供本地 SRT 解析使用）。"""
    blocks = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n\n")
    out = []
    for block in blocks:
        rows = [r for r in block.split("\n") if r.strip()]
        tl = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if tl is None:
            continue
        a, _, b = rows[tl].partition("-->")
        try:
            start = _srt_ts_to_sec(a)
            end = _srt_ts_to_sec(b)
        except (ValueError, IndexError):
            continue
        text = "\n".join(rows[tl + 1:]).strip()
        if text:
            out.append({"start": start, "end": end, "text": text})
    return out


def parse_srt_to_segments(srt_path) -> list[dict]:
    """把 process_file 产出的 SRT 解析成前端要的 segments。

    说话者分离时，SRT 文字以「说话者 N：…」开头 → 拆出 speaker 与 text。
    """
    if not srt_path or not Path(srt_path).exists():
        return []
    raw = _parse_srt(Path(srt_path).read_text(encoding="utf-8"))
    out = []
    for s in raw:
        text, speaker = s["text"], None
        if "：" in text[:8]:
            head, _, rest = text.partition("：")
            # 简体为主（引擎统一输出简体）；保留繁体判断以防上游标签未转换
            if head.startswith("说话者") or head.startswith("說話者"):
                digits = "".join(c for c in head if c.isdigit())
                speaker = int(digits) if digits else None
                text = rest
        out.append({"start": s["start"], "end": s["end"], "speaker": speaker, "text": text})
    return out


def _rich_to_segments(rich) -> list[dict]:
    """引擎返回的字级 segments（TranscribeResult.segments_rich）→ 前端格式。

    引擎给的 speaker 是原始标签（"说话者N" 或 None/int）；此处正规化成整数，
    与 parse_srt_to_segments 的前端契约一致。words 原样带出（[]=无对齐）。
    """
    out = []
    for seg in (rich or []):
        spk = seg.get("speaker")
        speaker = None
        if isinstance(spk, int):
            speaker = spk
        elif isinstance(spk, str):
            digits = "".join(c for c in spk if c.isdigit())
            speaker = int(digits) if digits else None
        out.append({
            "start": seg["start"], "end": seg["end"],
            "speaker": speaker, "text": seg["text"],
            "words": seg.get("words") or [],
        })
    return out

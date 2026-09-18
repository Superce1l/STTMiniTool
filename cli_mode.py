"""cli_mode.py — 无头命令行模式（给外部 Agent／脚本使用）

为什么能这么薄
--------------
`webview_backend.WebBackend` 本来就是为「无 UI」设计的那一层：
  • `_load_worker()` 是**完全同步**的，且缺模型会自动下载
  • `transcribe(opts)` 直接回 `{"segments": [...], "srtPath": ...}`
所以 CLI 不需要 [[cli-mode-plan]] 当初评估的引擎抽离重构 —— WebView EXE 早就
把 app.py 整包进去了（`webview_backend` 第一行就 `import app as core`），
再为了「避开 tkinter」而重构没有意义。

输出契约（Agent 友善）
----------------------
  • **stdout 只放结果**：`--json` 时是一份可直接 `json.loads` 的干净 JSON。
    加载消息、进度、警告一律走 **stderr**（连引擎内部的 print 也被导去 stderr，
    见 `_stdout_to_stderr`），避免污染 stdout。
  • **退出码**：0 成功／1 失败／2 参数错误／130 使用者中断。
  • 失败时 stderr 印人话错误，`--json` 模式另在 stdout 给 `{"ok": false, "error": …}`。

子命令
------
  transcribe <音频…>   转录，输出 SRT／TXT／JSON
  apply <edited.json>  把（Agent 校正过的）text 按 index 写回，**保留原时间轴**
  status               核心／模型／加速版本就绪状况
  profiles             可用的用途 profile 清单

`apply` 存在的理由：让 Agent 修字幕时「只准改 text」。若直接把整份 SRT 交给
模型重写，时间戳很容易被顺手改坏且难以察觉；改成回填 index → 时间轴在结构上
不可能被破坏。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

# 这些子命令名称同时是 app_webview 判断「要不要走无头模式」的依据
SUBCOMMANDS = ("transcribe", "apply", "status", "profiles")


# ── stdout 保护 ────────────────────────────────────────────────────────
#   引擎与下载器内部有不少 print()，若直接落到 stdout 会让 Agent 的
#   json.loads 炸掉。加载／转录期间把 sys.stdout 换成 stderr，最后再用
#   保存下来的「真 stdout」写结果。
@contextlib.contextmanager
def _stdout_to_stderr():
    real = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield real
    finally:
        sys.stdout = real


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ── 暂时性设置覆盖 ─────────────────────────────────────────────────────
#   --profile / --core-variant 这类「只想影响这一次执行」的选项，不应该偷改
#   使用者 GUI 的持久化设置。作法：把 settings.json 复制到临时文件，改指
#   app.SETTINGS_FILE 到那份副本 —— WebBackend 底下所有读写设置的代码都走
#   同一个常数，故一次改点全体生效，且真正的 settings.json 完全不被碰到。
_ORIG_SETTINGS: Path | None = None      # 被临时副本取代前的真实配置文件路径


def _use_scratch_settings(core) -> Path:
    global _ORIG_SETTINGS
    real = Path(core.SETTINGS_FILE)
    _ORIG_SETTINGS = real
    tmp = Path(tempfile.mkdtemp(prefix="qwenasr-cli-")) / "settings.json"
    try:
        if real.exists():
            shutil.copyfile(real, tmp)
    except Exception:
        pass
    core.SETTINGS_FILE = tmp
    return tmp


def _make_backend(args):
    """建好 WebBackend 并（必要时）套用暂时设置，返回 (backend, core)。"""
    import app as core
    from webview_backend import WebBackend

    if getattr(args, "profile", None) or getattr(args, "persist", False) is False:
        # 没有 --persist 就一律用临时设置副本，确保 CLI 不会改到 GUI 的设置。
        _use_scratch_settings(core)

    be = WebBackend()                      # on_event=None → 不推事件、全静音
    if getattr(args, "profile", None):
        prof = next((p for p in be.get_basic_profiles()["profiles"]
                     if p["key"] == args.profile), None)
        if prof is None:
            raise SystemExit(f"未知的 profile：{args.profile}（用 `profiles` 子命令查看）")
        be.set_model(prof["core"], prof["model"])
        _log(f"[profile] {args.profile} → {prof['core']} · {prof['model']}")
    return be, core


def _load(be):
    """同步加载模型（缺文件会自动下载，进度走 stderr）。"""
    _log("[load] 加载模型中…（首次使用会自动下载）")
    be._load_worker()
    if not getattr(be.engine, "ready", False):
        raise RuntimeError(be._load_err or "模型加载失败")
    _log(f"[load] 就绪：{be.get_status()['backend']}")


# ══════════════════════════════════════════════════════════════════════
# transcribe
# ══════════════════════════════════════════════════════════════════════
def _segments_payload(res: dict) -> list[dict]:
    """WebBackend 的 segments → CLI 对外契约（稳定字段名，含 1-based index）。"""
    out = []
    for i, s in enumerate(res.get("segments") or [], 1):
        item = {"i": i,
                "start": round(float(s.get("start", 0.0)), 3),
                "end": round(float(s.get("end", 0.0)), 3),
                "text": s.get("text", "")}
        if s.get("speaker"):
            item["speaker"] = s["speaker"]
        out.append(item)
    return out


def _srt_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:                      # 进位溢出
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(segs: list[dict], dest: Path) -> Path:
    lines = []
    for n, s in enumerate(segs, 1):
        lines += [str(n),
                  f"{_srt_ts(s['start'])} --> {_srt_ts(s['end'])}",
                  s.get("text", ""), ""]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def cmd_transcribe(args) -> int:
    be, _core = _make_backend(args)
    with _stdout_to_stderr() as out:
        _load(be)
        results, failed = [], 0
        for raw in args.audio:
            p = Path(raw)
            if not p.exists():
                _log(f"[错误] 找不到文件：{p}")
                failed += 1
                results.append({"input": str(p), "ok": False, "error": "找不到文件"})
                continue
            _log(f"[转录] {p.name}")
            try:
                res = be.transcribe({
                    "path": str(p),
                    "language": args.language or "",
                    "hint": args.hint or "",
                    "diarize": bool(args.diarize),
                    "nSpeakers": args.speakers,
                    "align": not args.no_align,
                }, progress_cb=lambda pct, msg: _log(f"  {pct:3d}%  {msg}"))
            except Exception as e:
                _log(f"[错误] {p.name}：{e}")
                failed += 1
                results.append({"input": str(p), "ok": False, "error": str(e)})
                continue
            segs = _segments_payload(res)
            item = {"input": str(p), "ok": True,
                    "srtPath": res.get("srtPath"), "segments": segs,
                    "text": "".join(s["text"] for s in segs)}
            # -o 只在单文件时有意义；多文件一律用引擎的默认输出位置
            if args.output and len(args.audio) == 1:
                dest = Path(args.output)
                if args.format == "txt":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text("\n".join(s["text"] for s in segs), encoding="utf-8")
                elif args.format == "json":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(segs, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                else:
                    _write_srt(segs, dest)
                item["outPath"] = str(dest)
            results.append(item)

    # ── 结果写回真正的 stdout ──
    if args.json:
        payload = results[0] if len(results) == 1 else {"results": results}
        print(json.dumps(payload, ensure_ascii=False), file=out)
    else:
        for r in results:
            print(r.get("outPath") or r.get("srtPath") or f"FAILED\t{r['input']}", file=out)
    return 1 if failed else 0


# ══════════════════════════════════════════════════════════════════════
# apply —— 把校正过的 text 写回，时间轴原封不动
# ══════════════════════════════════════════════════════════════════════
def cmd_apply(args) -> int:
    src = Path(args.edited)
    if not src.exists():
        _log(f"[错误] 找不到文件：{src}")
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    segs = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(segs, list) or not segs:
        _log("[错误] JSON 需为 segments 数组，或含 segments 字段的对象。")
        return 2

    base = None
    if args.base:                      # 有基准文件 → 只采用它的 text，时间轴以基准为准
        bdata = json.loads(Path(args.base).read_text(encoding="utf-8"))
        base = bdata.get("segments") if isinstance(bdata, dict) else bdata
        by_i = {int(s.get("i", n)): s for n, s in enumerate(segs, 1)}
        merged = []
        for n, b in enumerate(base, 1):
            i = int(b.get("i", n))
            merged.append({"i": i, "start": b["start"], "end": b["end"],
                           "text": (by_i.get(i) or b).get("text", ""),
                           **({"speaker": b["speaker"]} if b.get("speaker") else {})})
        segs = merged
    else:
        bad = [n for n, s in enumerate(segs, 1)
               if "start" not in s or "end" not in s]
        if bad:
            _log(f"[错误] 第 {bad[:5]} 段缺 start/end；"
                 f"请保留原本的时间字段，或用 --base 指定原始 JSON。")
            return 2
        # 没有基准文件时做一次健检：时间轴退化（全零、全同、倒退）几乎必然是
        # 校正端把时间字段改坏了。这里不挡（调用端可能真的自备时间轴），但要
        # 出声——静默产出一份时间全错的字幕比报错更难查。
        degenerate = all(float(s["end"]) <= float(s["start"]) for s in segs)
        backwards = any(float(segs[i]["start"]) < float(segs[i - 1]["start"])
                        for i in range(1, len(segs)))
        if degenerate or backwards:
            _log("[警告] 时间轴看起来不正常（"
                 + ("每段长度皆为 0；" if degenerate else "")
                 + ("段落时间倒退；" if backwards else "")
                 + "）。校正字幕时请加 --base <原始JSON>，"
                   "让时间轴以原始文件为准。")

    dest = Path(args.output) if args.output else src.with_suffix(".srt")
    if args.format == "txt":
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(s.get("text", "") for s in segs), encoding="utf-8")
    else:
        _write_srt(segs, dest)
    if args.json:
        print(json.dumps({"ok": True, "outPath": str(dest), "count": len(segs)},
                         ensure_ascii=False))
    else:
        print(dest)
    return 0


# ══════════════════════════════════════════════════════════════════════
# status / profiles
# ══════════════════════════════════════════════════════════════════════
def cmd_status(args) -> int:
    be, core = _make_backend(args)
    with _stdout_to_stderr() as out:
        st = be.get_status()
        accel = be.get_accel()
        health = be.health_check()
        # status 是在「还没加载任何模型」时查的，故 get_status()["backend"]
        # （= 实际加载中的核心）会是默认值而非使用者选的。这里改报**已选定**
        # 的核心，才不会误导 Agent。
        payload = {
            "appName": st.get("appName"), "version": st.get("version"),
            "backend": be._backend_label(st.get("backendKey"), st.get("backend")),
            "backendKey": st.get("backendKey"),
            "selectedReady": st.get("selectedReady"),
            "hasAnyModel": st.get("hasAnyModel"),
            "accel": {"selected": accel.get("selected"),
                      "recommended": accel.get("recommended"),
                      "installed": accel.get("installed"),
                      "needsDownload": accel.get("needsDownload")},
            "hardware": (accel.get("hardware") or {}).get("gpus"),
            "cores": [{"label": c["label"],
                       "items": [{"label": i["label"], "status": i["status"]}
                                 for i in c["items"]]}
                      for c in health.get("cores", [])],
            # 路径诊断：模型下载到哪、设置从哪读。冻结成 EXE 后这几条路径最容易
            # 与预期不符（BASE_DIR vs _MEIPASS），没有它就只能猜。
            "paths": {
                "settings":     str(_ORIG_SETTINGS or getattr(core, "SETTINGS_FILE", "")),
                "settingsInUse": str(getattr(core, "SETTINGS_FILE", "")),
                "settingsExists": Path(_ORIG_SETTINGS or
                                       getattr(core, "SETTINGS_FILE", ".")).exists(),
                "modelDir":    str(be._model_dir()),
                "crispasrDir": str(be._crispasr_dir()),
                "baseDir":     str(getattr(core, "BASE_DIR", "")),
            },
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"{payload['appName']} {payload['version']}")
        print(f"已选核心：{payload['backend']}　模型就绪：{payload['selectedReady']}")
        print(f"加速：{payload['accel']['selected']}"
              f"（建议 {payload['accel']['recommended']}）")
    return 0


def cmd_profiles(args) -> int:
    be, _core = _make_backend(args)
    with _stdout_to_stderr() as out:
        d = be.get_basic_profiles()
        payload = {"tier": d["tier"], "hardware": d["hardware"],
                   "profiles": [{"key": p["key"], "title": p["title"],
                                 "desc": p["desc"], "note": p["note"],
                                 "core": p["core"], "model": p["model"],
                                 "present": p["present"]}
                                for p in d["profiles"]]}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload["hardware"])
        for p in payload["profiles"]:
            mark = "✓" if p["present"] else " "
            print(f"  [{mark}] {p['key']:6s} {p['title']}  →  {p['model']}")
    return 0


# ══════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="QwenASR", add_help=True,
        description="语音识别小工具 — 无头命令行模式（给 Agent／脚本使用）。"
                    "不带子命令执行则开启图形界面。")
    ap.add_argument("--json", action="store_true",
                    help="stdout 输出 JSON（进度一律走 stderr）")
    ap.add_argument("--persist", action="store_true",
                    help="允许本次执行改写 settings.json（默认用临时副本，"
                         "不影响图形界面的设置）")

    # 让 --json / --persist 放在子命令**前后皆可**（Agent 两种写法都很自然）。
    # 子命令这份用 SUPPRESS 当默认：没给就不覆盖 namespace，于是顶层那份的值
    # 得以保留 —— 否则 argparse 会用子解析器的默认值盖掉顶层已解析的结果。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="stdout 输出 JSON")
    common.add_argument("--persist", action="store_true", default=argparse.SUPPRESS,
                        help="允许改写 settings.json")

    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", parents=[common], help="转录音频／视频为字幕")
    t.add_argument("audio", nargs="+", help="音频或视频文件路径（可多个）")
    t.add_argument("-o", "--output", help="输出文件路径（仅单一输入时有效）")
    t.add_argument("-f", "--format", choices=("srt", "txt", "json"), default="srt",
                   help="-o 的输出格式（默认 srt）")
    t.add_argument("-l", "--language", default="",
                   help="语言，如 Chinese／English／Japanese；留空＝自动检测")
    t.add_argument("--hint", default="", help="识别提示（人名、术语、背景说明）")
    t.add_argument("--profile", choices=("zh", "ja", "whisper"),
                   help="按用途选模型（见 profiles 子命令）")
    t.add_argument("--diarize", action="store_true", help="启用说话者分离")
    t.add_argument("--speakers", type=int, help="说话者人数（留空＝自动）")
    t.add_argument("--no-align", action="store_true", help="关闭字级时间轴对齐")
    t.set_defaults(func=cmd_transcribe)

    a = sub.add_parser("apply", parents=[common], help="把校正过的 segments JSON 写回字幕（保留时间轴）")
    a.add_argument("edited", help="校正后的 JSON（segments 数组或含 segments 的对象）")
    a.add_argument("-o", "--output", help="输出路径（默认与输入同名的 .srt）")
    a.add_argument("-f", "--format", choices=("srt", "txt"), default="srt")
    a.add_argument("--base", help="原始 JSON；给了就以它的时间轴为准，"
                                  "只采用校正档的 text（最安全）")
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("status", parents=[common], help="核心／模型／加速版本就绪状况")
    s.set_defaults(func=cmd_status)

    p = sub.add_parser("profiles", parents=[common], help="列出用途 profile 与各自的建议模型")
    p.set_defaults(func=cmd_profiles)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _log("已中断。")
        return 130
    except SystemExit as e:
        _log(str(e))
        return 2
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())

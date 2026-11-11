"""transcription_service.py — 转录编排层（webview_backend 拆分，2026-11）

WebBackend 的「转录」内聚块：
  • transcribe        — 单片/整段转录编排（视频抽音轨、FA/说话者保障、
                        输出命名与格式、进度与取消契约）
  • _transcribe_sliced — >2h 长音频的 ffmpeg 切片调度与合并
  • _resolve_ffmpeg / _ensure_ffmpeg — ffmpeg 就绪（含按需下载）

作为 mixin 被 WebBackend 继承：self.* 状态（engine/_cancel/_emit/
_dl_progress/settings/_transcribe_gate…）仍归 WebBackend 所有，方法经
继承解析，行为不变。模块只依赖切片纯函数层（slicing）、模型目录
（model_catalog）与引擎层（app）；不得 import webview_backend（循环）。
"""
from __future__ import annotations

import traceback
from pathlib import Path

import app as core
from engine_types import TranscribeResult, base_dir
from model_catalog import _sanitize_tag
from slicing import (_SLICE_THRESHOLD_SECS, _ffprobe_duration, _slice_plan,
                     _merge_sliced_segments, _match_speaker,
                     assign_speaker_by_overlap as _assign_speaker_by_overlap,
                     parse_srt_to_segments, _rich_to_segments)

BASE_DIR = base_dir()


class TranscribeService:
    """转录编排 mixin（设计见模块 docstring）。"""

    # ── 转录 ────────────────────────────────────────────────
    # opts: {path, language, diarize, nSpeakers, align, hint}
    # progress_cb(pct:int, status:str)
    def transcribe(self, opts: dict, progress_cb=None) -> dict:
        import time as _time
        from applog import log_transcribe
        if not getattr(self.engine, "ready", False):
            raise RuntimeError("模型尚未加载完成，请稍候再试。")
        # 忙碌 check-and-set（独立小锁；self._lock 会在整个引擎调用期间被持有，
        # 不能用来做这道 gate——否则两道门互相等死锁）。并发第二请求直接拒绝，
        # 且只有置位者能在 finally 里清 flag，避免先结束者清掉后到者的忙碌状态。
        if not self._transcribe_gate.acquire(blocking=False):
            raise RuntimeError("已有识别任务进行中，请稍候再试。")
        path = opts.get("path")
        if not path or not Path(path).exists():
            raise RuntimeError("找不到音频文件。")
        self._cancel = False
        # 引擎侧取消事件（两引擎同契约）：crispasr 据此杀子进程；OpenVINO 在
        # process_file 段间检查。「单片/单段转录中」也能立即取消。
        cancel_ev = getattr(self.engine, "cancel_event", None)
        if cancel_ev is not None:
            cancel_ev.clear()
        self._transcribing = True
        _t0 = _time.monotonic()
        eng_label, model_label = getattr(self, "_active_label", None) or self._current_selection()
        log_transcribe(f"开始转录：{Path(path).name}"
                       f"（{eng_label} · {model_label}，"
                       f"语言={opts.get('language') or '自动'}"
                       f"{'，分离' if opts.get('diarize') else ''}）")

        def _cb(i, total, msg):
            if progress_cb:
                pct = min(99, int((i / max(total, 1)) * 100))
                progress_cb(pct, msg)

        # 视频 → 先抽音轨
        audio_path = Path(path)
        tmp_extra = None
        try:
            ext = audio_path.suffix.lower()
            from ffmpeg_utils import VIDEO_EXTS
            if ext in VIDEO_EXTS:
                from ffmpeg_utils import extract_audio_to_wav
                ff = self._ensure_ffmpeg()
                if not ff:
                    raise RuntimeError("上传为视频但找不到 ffmpeg，且自动下载失败，无法抽音轨。")
                wav = audio_path.with_suffix(".extracted.wav")
                extract_audio_to_wav(audio_path, wav, ff)
                audio_path, tmp_extra = wav, wav

            want_align = bool(opts.get("align", True))
            want_diar = bool(opts.get("diarize"))

            # ── 时间轴对齐（FA）：缺模型则按需下载（移植自 app.py 的协调层）──
            if want_align:
                self._ensure_fa()
            if hasattr(self.engine, "use_aligner") and getattr(self.engine, "_fa_bin", None):
                self.engine.use_aligner = want_align

            # ── 说话者分离：外部 ONNX，与后端无关。缺模型则下载 + 挂上 diar_engine ──
            if want_diar:
                if not (getattr(self.engine, "diar_engine", None)
                        and getattr(self.engine.diar_engine, "ready", False)):
                    try:
                        self._ensure_diarization()
                    except Exception as e:
                        head = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
                        self._emit("progress", {"pct": 0, "status": f"说话者分离模型下载失败：{head}"})

            n_spk = opts.get("nSpeakers")
            n_speakers = int(n_spk) if str(n_spk).isdigit() else None

            # 语言：空字符串 / 自动检测 / auto → None（让引擎自动检测）
            _lang = (opts.get("language") or "").strip()
            if _lang in ("", "自动检测", "auto"):
                _lang = None

            # 输出落点：subtitles/ 下、用「原始上传文件名」命名（可在「开启输出文件夹」找到）。
            # 上传文件名为使用者可控（含 LAN/外网端点）→ 必须只取基名，杜绝路径穿越。
            srt_dir = Path(getattr(core, "SRT_DIR", BASE_DIR / "subtitles"))
            srt_dir.mkdir(parents=True, exist_ok=True)
            raw_name = opts.get("name") or Path(path).name
            out_name = Path(str(raw_name).replace("\\\\", "/")).name   # 去除任何目录成分
            if not out_name or out_name.startswith("."):
                out_name = "transcript"
            # 文件名拼接「实际加载」的引擎与模型（非 settings 里记住的选择——
            # 忙碌期间选择可被改而引擎未换，标签必须反映真正跑的引擎）：
            # 唯一.flac → 唯一 [CrispASR · Whisper Base].flac
            # （标签安全化：去路径非法字符、压空白；「.」换「·」——无扩展名文件的
            # out_ref 以「…0.6B]」结尾时，write_transcript 的 ref.stem 与切片路径的
            # with_suffix 会把它当扩展名剥掉，产生残缺文件名）
            tag = _sanitize_tag(f"{eng_label} · {model_label}").replace(".", "·")
            p = Path(out_name)
            out_name = f"{p.stem} [{tag}]{p.suffix}" if p.suffix else f"{p.stem} [{tag}]"
            out_ref = srt_dir / out_name
            # 防呆：确认解析后仍在 subtitles/ 内
            try:
                if srt_dir.resolve() not in out_ref.resolve().parents:
                    out_ref = srt_dir / "transcript"
            except Exception:
                out_ref = srt_dir / "transcript"

            with self._lock:
                self._apply_chunk_secs(want_align)   # 任务自适应的每段上限（对齐→短段抗漂移）

                # ── 长音频切片：超过阈值先用 ffmpeg 切片再逐片转录 ──────────
                ff = self._resolve_ffmpeg()
                dur = _ffprobe_duration(ff, audio_path) if ff else None
                if dur and dur > _SLICE_THRESHOLD_SECS:
                    res = self._transcribe_sliced(
                        audio_path, dur, out_ref, _cb, _lang, want_diar, n_speakers)
                else:
                    res = self.engine.process_file(
                        audio_path,
                        progress_cb=_cb,
                        language=_lang,
                        context=(opts.get("hint") or "").strip() or None,
                        diarize=want_diar,
                        n_speakers=n_speakers,
                        original_path=out_ref,
                        out_format="srt",
                    )
        finally:
            self._transcribing = False
            try:
                self._transcribe_gate.release()   # 只有置位者会走到这里（acquire 失败早已抛出）
            except RuntimeError:
                pass
            if tmp_extra:
                try:
                    Path(tmp_extra).unlink(missing_ok=True)
                except Exception:
                    pass

        if self._cancel or (res is not None and res.cancelled):
            log_transcribe(f"转录已取消：{Path(path).name}")
            raise RuntimeError("已取消。")
        if not res.output:
            diag = res.diag
            log_transcribe(f"转录未产生字幕：{Path(path).name}（{diag or '未检测到人声'}）")
            raise RuntimeError(diag or "未产生字幕（未检测到人声）。")

        # UI 永远以内存中的 segments 渲染波形/字幕卡（需时间轴），故引擎固定产 SRT。
        # 但「输出格式」设置决定使用者实际取得的存档文件：选纯文字 → 另写 .txt、移除 .srt。
        # 优先用引擎的字级结构（含 words，驱动卡拉OK）；无则退回解析 SRT（行级）。
        rich = res.segments_rich
        srt_path = res.output
        segments = _rich_to_segments(rich) if rich else parse_srt_to_segments(srt_path)
        saved = str(srt_path)
        out_fmt = (self._settings_raw().get("output_format", "srt") or "srt").lower()
        if out_fmt == "txt":
            try:
                from subtitle_lines import write_transcript
                lines = [(seg["start"], seg["end"], seg["text"],
                          (f'说话者{seg["speaker"]}' if seg.get("speaker") else None))
                         for seg in segments]
                txt = write_transcript(Path(srt_path), lines, out_format="txt")
                try:
                    Path(srt_path).unlink(missing_ok=True)
                except Exception:
                    pass
                saved = str(txt)
            except Exception:
                traceback.print_exc()          # 退回 SRT，不让转录整体失败

        log_transcribe(
            f"转录完成：{Path(path).name} → {Path(saved).name}"
            f"（{len(segments)} 段，耗时 {_time.monotonic() - _t0:.1f}s，"
            f"{eng_label} · {model_label}）")
        if progress_cb:
            progress_cb(100, "完成")
        resp = {"segments": segments, "srtPath": saved}
        # 非致命降级（如字级对齐失败回退比例估算）透传给前端可提示
        if res.warnings:
            resp["warnings"] = res.warnings
        return resp

    def _resolve_ffmpeg(self):
        """解析 ffmpeg：优先采用设置中手动指定的 ffmpeg_path，否则自动检测。

        与 setting.py 的检测顺序一致（使用者手动指定 → 系统 PATH / App 目录）。
        """
        from ffmpeg_utils import find_ffmpeg
        p = (self._settings_raw().get("ffmpeg_path", "") or "").strip()
        if p and Path(p).exists():
            return Path(p)
        return find_ffmpeg()

    # ── 长音频切片转录 ──────────────────────────────────────────────────
    def _transcribe_sliced(self, audio_path: Path, duration: float,
                           out_ref: Path, _cb, lang, want_diar, n_speakers):
        """超过阈值的长音频：ffmpeg 切片 → 逐片转录 → 合并 SRT。

        每片独立调 engine.process_file（重叠 15 秒防切点截字），产出各自 SRT
        后解析成 segments，合并去重，最后一次性写出整份 SRT（经 write_transcript
        以 out_ref 命名落盘）。任一片失败即中止（与整段转录的失败语义一致）。

        说话者分离：不再交引擎逐片自做（每片独立聚类的标签编号互不相干，
        同一人在片 1 是「说话者1」、片 2 可能变「说话者2」），改由本层统一
        做——逐片聚类取声纹中心 → 跨片映射到全局编号（_match_speaker）→
        合并去重后按时间中点统一指派，整份字幕的说话者编号保持一致。
        返回 TranscribeResult（output=整份 SRT 路径；None=全片无声）。
        """
        import subprocess
        import tempfile
        from ffmpeg_utils import _NO_WINDOW

        ff = self._resolve_ffmpeg()
        if not ff:
            raise RuntimeError("长音频切片需要 ffmpeg，且自动下载失败。")

        # 说话者分离引擎（want_diar 但未就绪 → 尝试补挂一次，与 transcribe 同语义）
        diar = getattr(self.engine, "diar_engine", None) if want_diar else None
        if want_diar and not (diar and getattr(diar, "ready", False)):
            try:
                self._ensure_diarization()
            except Exception:
                traceback.print_exc()
            diar = getattr(self.engine, "diar_engine", None)
        if not (diar and getattr(diar, "ready", False)):
            diar = None

        plan = _slice_plan(duration)
        all_segments: list[list[dict]] = []
        _silent_parts: list[bool] = []   # 逐片记录是否无声（全静音时给诊断用）
        global_spk: list[dict] = []      # 全局说话者声纹中心（跨片映射目标）
        slice_diar: list[list[tuple[float, float, int]]] = []   # 各片 (绝对起, 绝对止, 全局标签)

        with tempfile.TemporaryDirectory(prefix="asr_slice_") as td:
            tmp = Path(td)
            for i, (start, end) in enumerate(plan, 1):
                if self._cancel:
                    return TranscribeResult(cancelled=True)
                seg_wav = tmp / f"slice_{i:03d}.wav"
                # -ss 放 -i 前（输入侧快速定位，大偏移时不解码整段前缀）。
                # 此时输入时间戳被重置为 0，-to <全局终点> 会被误读成
                # 「重置后时间轴的终点」→ 第 2 片起每片严重超长。故用
                # -t <片长> 显式指定时长，语义不受 -ss 位置影响。
                cmd = [str(ff), "-y", "-ss", f"{start:.3f}",
                       "-t", f"{end - start:.3f}",
                       "-i", str(audio_path), "-vn", "-ar", "16000", "-ac", "1",
                       "-f", "wav", str(seg_wav)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      creationflags=_NO_WINDOW)
                if proc.returncode != 0 or not seg_wav.exists():
                    err = proc.stderr.decode(errors="replace").splitlines()
                    raise RuntimeError(
                        f"ffmpeg 切片失败（片 {i}/{len(plan)}）："
                        f"{err[-1] if err else '未知错误'}")

                # 说话者分离：转录前对本片独立聚类 → 标签映射到全局编号
                if diar is not None:
                    try:
                        from audio_io import load_audio_16k_mono
                        seg_audio, _ = load_audio_16k_mono(seg_wav, core.SAMPLE_RATE)
                        segs, cents = diar.diarize_detailed(
                            seg_audio, n_speakers=n_speakers)
                        mapped = []
                        for (t0, t1, lbl) in segs:
                            g = (_match_speaker(cents[lbl], global_spk)
                                 if lbl in cents else 1)
                            mapped.append((t0 + start, t1 + start, g))
                        slice_diar.append(mapped)
                    except Exception:
                        traceback.print_exc()   # 分离失败 → 本片无标签，不挡转录

                def _slice_cb(done, total, msg, _i=i, _n=len(plan)):
                    # engine 的回调是 (i, total, msg) 三参；这里先把片内进度换算成
                    # 「全局百分比」再以 (pct, 100, msg) 交给 _cb——直接转发 done/total
                    # 会让进度条在每片边界从 0 重爬（N 片来回跳 N 次）。
                    base = (_i - 1) / _n
                    span = 1.0 / _n
                    frac = (done / max(total, 1)) if total else 0
                    _cb(min(99, int((base + span * frac) * 100)), 100,
                        f"切片转录 {msg}（第 {_i}/{_n} 片）")

                # diarize=False：分离已由本层统一处理（引擎逐片自做标签会跨片漂移）
                part = self.engine.process_file(
                    seg_wav,
                    progress_cb=_slice_cb,
                    language=lang,
                    context=None,
                    diarize=False,
                    n_speakers=n_speakers,
                    original_path=None,
                    out_format="srt",
                )
                if part.cancelled:
                    return TranscribeResult(cancelled=True)
                if not part.output:
                    _silent_parts.append(True)
                    continue      # 该片无声 → 跳过（如纯音乐片段）
                _silent_parts.append(False)
                # 片内时间已是绝对时间（-ss 放在 -i 前 = 输入侧定位，
                # 输出的 pts 从 0 起算，故把片起点平移回全局时间轴）
                part_segs = _rich_to_segments(part.segments_rich) \
                    if part.segments_rich else parse_srt_to_segments(part.output)
                for seg in part_segs:
                    seg["start"] += start
                    seg["end"] += start
                    seg.pop("speaker", None)   # 说话者稍后由全局分离结果统一指派
                    for w in (seg.get("words") or []):
                        w["start"] = w.get("start", 0) + start
                        w["end"] = w.get("end", 0) + start
                all_segments.append(part_segs)

        merged = _merge_sliced_segments(all_segments)
        if not merged:
            # 所有片都无声（每片 process_file 回 None）→ 给出与整段路径
            # 对等的诊断，而非笼统的「未检测到人声」。
            return TranscribeResult(diag=(
                f"⚠ 切片转录未产生字幕：{sum(_silent_parts)}/{len(plan)} 片"
                f"未识别到内容（音频可能为纯音乐/静音，或音轨编码异常）。"))

        # 说话者指派：取与行**重叠最长**的说话者段（跨界行不再随机归给中点
        # 一侧）；行与任何段都无交集时取时间最近的段兜底。
        flat = [seg for segs in slice_diar for seg in segs]
        if flat:
            for seg in merged:
                seg["speaker"] = _assign_speaker_by_overlap(
                    seg["start"], seg["end"], flat)

        lines = [(seg["start"], seg["end"], seg["text"],
                  (f'说话者{seg["speaker"]}' if seg.get("speaker") else None))
                 for seg in merged]
        out = out_ref.with_suffix(".srt")
        out.parent.mkdir(parents=True, exist_ok=True)
        # 共享构造 + 原子写出（与 write_transcript 同一格式来源）：
        # 转录中途进程退出不再留下半截 SRT。
        from subtitle_lines import srt_text, atomic_write_text
        atomic_write_text(out, srt_text(lines))
        rich = [{"start": seg["start"], "end": seg["end"], "text": seg["text"],
                 "speaker": seg.get("speaker"), "words": seg.get("words") or []}
                for seg in merged]
        return TranscribeResult(output=out, segments_rich=rich)

    def _ensure_ffmpeg(self):
        """返回可用的 ffmpeg；找不到时自 HF 下载精简 zip 解压后再找。

        ffmpeg 不随安装包附带 → 首次需要（转录视频）时自动下载到 <app>/ffmpeg/。
        下载进度经 self._dl_progress → SSE 推给前端。失败回 None（由上层报错）。
        """
        ff = self._resolve_ffmpeg()
        if ff:
            return ff
        try:
            from ffmpeg_utils import get_default_ffmpeg_dest
            from downloader import quick_check_ffmpeg, download_ffmpeg
            dest = get_default_ffmpeg_dest()
            if not quick_check_ffmpeg(dest):
                self._st("下载 ffmpeg（视频音轨提取）…")
                download_ffmpeg(dest, progress_cb=self._dl_progress)
            return self._resolve_ffmpeg()
        except Exception:
            traceback.print_exc()
            return None

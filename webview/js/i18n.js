/* ============================================================
   i18n.js — 界面语言（简体中文 / English）
   用法：在 HTML 元素挂 data-i18n="key"（textContent）、
        data-i18n-html（innerHTML）、data-i18n-ph（placeholder）、
        data-i18n-title（title）。调用 window.I18N.setLang(uiLang) 切换。
   字典仅涵盖静态界面字符串；后端动态消息（进度/错误）维持原语言。
   ============================================================ */
(function () {
  "use strict";

  // uiLang 设置值（与 #set-lang 选项一致）→ locale code
  const LANG_MAP = { "简体中文": "hans", "English": "en" };
  let LOCALE = "hans";

  // key: [简体中文, English]
  const D = {
    "nav.file": ["音频", "Audio"],
    "nav.record": ["录制", "Record"],
    "nav.model": ["模型", "Model"],
    "nav.settings": ["设置", "Settings"],

    "view.file": ["音频转字幕", "Audio to Subtitles"],
    "view.record": ["录制转换", "Record & Transcribe"],
    "view.model": ["模型与设备", "Model & Devices"],
    "view.settings": ["设置", "Settings"],

    "file.dropBig": ["拖拽音频到此，或<b>点击选择文件</b>", "Drop audio here, or <b>click to choose a file</b>"],
    "file.dropSub": ["支持 mp3 / wav / m4a / 视频音轨", "Supports mp3 / wav / m4a / video audio"],
    "file.run": ["开始转换", "Start"],
    "file.running": ["识别中…", "Transcribing…"],
    "file.preparing": ["准备中…", "Preparing…"],
    "file.logStart": ["开始转换：", "Start: "],
    "file.logDone": ["完成，共 {n} 段", "Done — {n} segments"],
    "file.logSaved": ["，已输出 ", ", saved to "],
    "file.saved": ["已存档字幕：", "Subtitle saved: "],
    "file.openDir": ["打开输出文件夹", "Open output folder"],
    "file.saveSub": ["保存字幕", "Save subtitles"],
    "save.srt": ["SRT 字幕", "SRT subtitles"],
    "save.ass": ["ASS 卡拉OK（逐字）", "ASS karaoke (word timing)"],
    "save.lrc": ["LRC 歌词", "LRC lyrics"],
    "save.txt": ["纯文本", "Plain text"],
    "file.history": ["最近转录", "Recent transcripts"],
    "file.historyLoaded": ["已载入历史字幕：", "Loaded transcript: "],
    "file.historyDel": ["删除（到回收站）", "Delete (to Recycle Bin)"],
    "file.historyDelConfirm": ["确定删除字幕文件「{name}」吗？\n（移到回收站，可恢复）", "Delete subtitle file \"{name}\"?\n(Moved to Recycle Bin — recoverable)"],
    "file.historyDeleted": ["已删除：", "Deleted: "],
    "file.historyDeletedPerm": ["（已直接删除）", " (deleted permanently)"],
    "file.historyDelFail": ["删除失败：", "Delete failed: "],
    "file.verify": ["字幕校验", "Verify"],
    "file.diar": ["说话者分离", "Speaker diarization"],
    "file.align": ["时间轴对齐", "Timestamp align"],
    "file.hintTitle": ["识别提示（可选）", "Recognition hint (optional)"],
    "file.hintDesc": ["粘贴歌词、关键字或背景说明，可提升识别准确度", "Paste lyrics, keywords or context to improve accuracy"],
    "file.hintPh": ["例如：本段为产品营销 Podcast，会出现“转化率”“漏斗”等营销术语…", "e.g. This is a marketing podcast with terms like 'conversion rate', 'funnel'…"],
    "file.loadTxt": ["读入 TXT…", "Load TXT…"],
    "file.resultTitle": ["识别结果", "Result"],
    "file.speakers": ["说话者", "Speakers"],
    "file.spkRenameHint": ["点击标签可重命名", "Click a label to rename"],
    "file.spkSwitchHint": ["点击切换此行的说话者", "Click to switch this line's speaker"],
    "file.spkSwitched": ["第 {i} 行说话者：{old} → {new}（导出与合并判定同步）", "Line {i} speaker: {old} → {new} (exports & merge rules follow)"],
    "file.spkCur": ["当前", "current"],
    "file.spkNew": ["＋ 新说话者", "＋ New speaker"],
    "file.spkRenameMenu": ["重命名说话者…", "Rename speaker…"],
    "file.waveHint": ["点字幕选中片段，播放键循环播放；点波形移动播放头", "Click a line to select its span (▶ loops it); click the wave to move the playhead"],
    "file.spkRenamed": ["说话者已重命名：{old} → {new}（各段落与导出同步）", "Speaker renamed: {old} → {new} (applied to all segments & exports)"],

    "edit.line": ["编辑此行", "Edit this line"],
    "edit.save": ["保存（Enter）", "Save (Enter)"],
    "edit.split": ["拆分此行：进入后点击放置光标，点亮的 √ 切开", "Split this line: place the caret, then click the lit ✓"],
    "edit.splitHow": ["点击文本中要拆分的位置，再点亮的 √ 或按 Ctrl+Enter", "Click where to split in the text, then the lit ✓ (or Ctrl+Enter)"],
    "edit.splitNeedCaret": ["请先点击文本放置光标，再拆分", "Click in the text to place the caret first"],
    "edit.splitEmpty": ["拆分点两侧都需要有文字", "Both sides of the split need text"],
    "edit.splitTooShort": ["此段太短，无法拆分", "This line is too short to split"],
    "edit.splitDone": ["已拆分为 2 段", "Split into 2 segments"],
    "merge.mode": ["合并段落", "Merge segments"],
    "merge.selCount": ["已选 {n} 段 · 需相邻且同一说话者", "{n} selected · must be adjacent, same speaker"],
    "merge.ok": ["合并", "Merge"],
    "merge.cancel": ["取消", "Cancel"],
    "merge.needTwo": ["请至少选择 2 段", "Select at least 2 segments"],
    "merge.notAdjacent": ["所选段落必须相邻", "Selected segments must be adjacent"],
    "merge.diffSpeaker": ["所选段落说话者不一致", "Selected segments have different speakers"],
    "merge.done": ["已合并 {n} 段", "Merged {n} segments"],
    "common.langAuto": ["语言 自动", "Lang Auto"],
    "common.spkAuto": ["人数 自动", "Count Auto"],

    "record.mic": ["麦克风", "Microphone"],
    "record.refresh": ["刷新", "Refresh"],
    "record.liveTitle": ["实时字幕", "Live captions"],
    "record.liveDesc": ["说完停顿自动识别；可清除或按输出格式保存", "Auto-transcribes on pause; clear or save by output format"],
    "record.autosave": ["实时保存", "Auto-save"],
    "record.clear": ["清除", "Clear"],
    "record.save": ["保存字幕", "Save subtitles"],
    "record.note": ["录制转换：检测到说话停顿时才识别，句中短暂停顿不会中断。按麦克风开始，再按一次结束并完成最后一段。", "Transcribes on speech pauses; brief mid-sentence pauses won't cut. Tap the mic to start, tap again to finish the last segment."],
    "record.placeholder": ["开始录音后，识别结果会逐段出现在这里。", "After recording starts, results appear here segment by segment."],
    "record.permGranted": ["麦克风已授权", "Microphone allowed"],
    "record.permPrompt": ["按麦克风时将请求授权", "Will ask permission on record"],
    "record.permDenied": ["麦克风权限被拒。请在窗口地址栏的权限图标中允许麦克风，或在系统设置开放本程序的麦克风访问后再试。", "Microphone blocked. Allow it via the permission icon in the address bar, or enable mic access for this app in system settings, then retry."],
    "record.permDeniedShort": ["麦克风被拒", "Mic blocked"],
    "record.unsupported": ["不支持录音", "Recording unsupported"],
    "record.defaultMic": ["默认麦克风", "Default microphone"],
    "record.micN": ["麦克风 {n}", "Microphone {n}"],
    "record.noApi": ["此环境不支持录音 API。", "Recording API unavailable in this environment."],
    "record.denied": ["麦克风权限被拒。", "Microphone permission denied. "],
    "record.micFail": ["无法取得麦克风：", "Could not get microphone: "],
    "record.listening": ["聆听中…说完停顿约 2 秒会自动识别；再按一次结束并完成最后一段。", "Listening… auto-transcribes ~2s after each pause; tap again to finish the last segment."],
    "record.stopped": ["已停止。", "Stopped."],
    "record.recFailed": ["识别失败：", "Transcription failed: "],
    "record.autosaveOn": ["实时存档已启用：每段识别完成后会自动写入所选文件。", "Auto-save on: each finished segment is written to the chosen file."],
    "record.autosaveFail": ["实时存档写入失败：", "Auto-save write failed: "],

    "model.health": ["系统自检", "System check"],
    "model.recheck": ["重新检查", "Re-check"],
    "model.healthRed": ["{n} 项需处理", "{n} need attention"],
    "model.healthYellow": ["核心就绪 · {n} 项将于启用时自动下载", "Cores ready · {n} auto-download on use"],
    "model.healthOk": ["全部就绪", "All ready"],
    "model.shared": ["共享组件", "Shared components"],
    "model.inUse": ["使用中", "Active"],
    "model.accelNeeds": ["加速器尚未下载或版本不符（最新 {v}），点「下载并加载模型」时会一并取得。", "Accelerator missing or outdated (latest {v}) — it will be fetched with “Download & load model”."],
    "model.pick": ["选择模型", "Choose a model"],
    "model.core": ["推理引擎", "Inference engine"],
    "model.load": ["下载并加载模型", "Download & load model"],
    "model.loading": ["下载／加载中…", "Downloading / loading…"],
    "model.goto": ["前往语音转文字", "Go to transcription"],
    "model.busy": ["转录进行中，无法切换", "Transcribing — switching unavailable"],
    "model.nModels": ["{n} 种模型可选", "{n} models"],
    "model.checkDir": ["程序检查的模型目录：", "Model directory being checked: "],
    "model.checkCoreDir": ["CrispASR 核心目录：", "CrispASR core directory: "],
    "model.checkCoreIssue": ["加速器：{issue}", "Accelerator: {issue}"],
    "model.checkMissing": ["缺少 {n} 个文件：", "{n} file(s) missing: "],
    "model.checkMissingCore": ["核心目录缺少 {n} 个文件：", "{n} file(s) missing in core directory: "],
    "model.devices": ["检测到的设备", "Detected devices"],
    "status.ready": ["模型已就绪", "Model ready"],
    "status.loading": ["加载模型中…", "Loading model…"],
    "status.needModel": ["尚未加载模型", "No model loaded"],
    "status.notDownloaded": ["模型未下载", "Model not downloaded"],
    "status.switchPending": ["待切换", "Switch pending"],
    "status.tipLoading": ["正在下载／加载模型…", "Downloading / loading model…"],
    "status.tipNotDownloaded": ["所选模型尚未下载——到「模型」页点「下载并加载模型」", "Selected model isn't downloaded — open the Model page and click “Download & load model”"],
    "status.tipNotDownloadedDir": ["所选模型尚未下载——程序检查的目录：\n{dir}", "Selected model isn't downloaded — directory being checked:\n{dir}"],
    "status.tipCoreIssue": ["加速器：{issue}", "Accelerator: {issue}"],
    "status.tipMissing": ["缺少：{files}", "Missing: {files}"],
    "status.tipNotLoaded": ["模型文件已备——到「模型」页点「下载并加载模型」", "Model files present — open the Model page and click “Download & load model”"],
    "status.tipSwitch": ["已加载：{loaded}\n已选择：{sel}\n到「模型」页点「下载并加载模型」完成切换", "Loaded: {loaded}\nSelected: {sel}\nOpen the Model page and click “Download & load model” to switch"],
    "status.tipFallback": ["本次加载失败，已回退旧模型继续使用：{e}", "Loading failed — fell back to the previous model: {e}"],
    "status.tipReady": ["当前模型：{m}", "Active model: {m}"],
    "model.accel": ["推理加速版本", "Acceleration build"],
    "model.diag": ["诊断", "Diagnostics"],

    "set.scale": ["界面缩放", "UI scale"],
    "set.scaleDesc": ["调整文字与控件大小", "Adjust text and control size"],
    "set.format": ["输出格式", "Output format"],
    "set.formatDesc": ["全局默认，影响单文件输出", "Global default for transcription output"],
    "set.fmtSrt": ["SRT 字幕", "SRT subtitles"],
    "set.fmtTxt": ["纯文本", "Plain text"],
    "set.modelDir": ["模型下载位置", "Model download location"],
    "set.modelDirDesc": ["Qwen / Whisper 模型的存储目录；留空＝程序目录下的 ov_models（重启后生效）", "Where Qwen / Whisper models are stored; blank = ov_models next to the app (takes effect after restart)"],
    "set.mirror": ["HuggingFace 镜像站", "HuggingFace mirror"],
    "set.mirrorDesc": ["下载模型较慢时可改用镜像", "Use a mirror if downloads are slow"],
    "set.ffmpeg": ["FFmpeg 路径", "FFmpeg path"],
    "set.ffmpegDesc": ["处理视频音轨所需（留空则需要时自动下载）", "Needed for video audio (auto-downloads if blank)"],
    "set.theme": ["外观主题", "Appearance"],
    "set.themeLight": ["浅色", "Light"],
    "set.themeDark": ["深色", "Dark"],
    "set.themeSystem": ["跟随系统", "System"],
    "set.lang": ["界面语言", "Language"],
    "set.appDesc": ["本地推理 · 数据不离开你的电脑", "Local inference · data stays on your PC"],
    "set.checkUpdate": ["检查更新", "Check update"],
    "set.checking": ["检查中…", "Checking…"],
    "set.newVersion": ["新版本 v{v}！打开下载页…", "New v{v}! Opening download page…"],
    "set.upToDate": ["已是最新 v{v}", "Up to date (v{v})"],
    "set.checkFailed": ["检查失败（网络？）", "Check failed (network?)"],
  };

  function idx() { return LOCALE === "en" ? 1 : 0; }
  function t(key) { const e = D[key]; return e ? (e[idx()] || e[0]) : null; }

  function apply(root) {
    root = root || document;
    root.querySelectorAll("[data-i18n]").forEach(el => { const v = t(el.dataset.i18n); if (v != null) el.textContent = v; });
    root.querySelectorAll("[data-i18n-html]").forEach(el => { const v = t(el.dataset.i18nHtml); if (v != null) el.innerHTML = v; });
    root.querySelectorAll("[data-i18n-ph]").forEach(el => { const v = t(el.dataset.i18nPh); if (v != null) el.setAttribute("placeholder", v); });
    root.querySelectorAll("[data-i18n-title]").forEach(el => { const v = t(el.dataset.i18nTitle); if (v != null) el.title = v; });
  }

  function setLang(uiLang) {
    LOCALE = LANG_MAP[uiLang] || "hans";
    document.documentElement.lang = LOCALE === "en" ? "en" : "zh-Hans";
    apply(document);
  }

  window.I18N = { setLang, t, apply, get locale() { return LOCALE; } };
})();

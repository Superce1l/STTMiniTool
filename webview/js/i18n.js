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
    "file.openDir": ["打开输出文件夹", "Open output folder"],
    "file.saveSub": ["保存字幕", "Save subtitles"],
    "file.verify": ["字幕校验", "Verify"],
    "file.diar": ["说话者分离", "Speaker diarization"],
    "file.align": ["时间轴对齐", "Timestamp align"],
    "file.hintTitle": ["识别提示（可选）", "Recognition hint (optional)"],
    "file.hintDesc": ["粘贴歌词、关键字或背景说明，可提升识别准确度", "Paste lyrics, keywords or context to improve accuracy"],
    "file.hintPh": ["例如：本段为产品营销 Podcast，会出现“转化率”“漏斗”等营销术语…", "e.g. This is a marketing podcast with terms like 'conversion rate', 'funnel'…"],
    "file.loadTxt": ["读入 TXT…", "Load TXT…"],
    "file.resultTitle": ["识别结果", "Result"],
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

    "model.health": ["系统自检", "System check"],
    "model.recheck": ["重新检查", "Re-check"],
    "model.pick": ["选择模型", "Choose a model"],
    "model.core": ["推理引擎", "Inference engine"],
    "model.load": ["下载并加载模型", "Download & load model"],
    "model.loading": ["下载／加载中…", "Downloading / loading…"],
    "model.goto": ["前往语音转文字", "Go to transcription"],
    "model.busy": ["转录进行中，无法切换", "Transcribing — switching unavailable"],
    "model.nModels": ["{n} 种模型可选", "{n} models"],
    "model.devices": ["检测到的设备", "Detected devices"],
    "status.ready": ["模型已就绪", "Model ready"],
    "status.loading": ["加载模型中…", "Loading model…"],
    "status.needModel": ["尚未加载模型", "No model loaded"],
    "model.accel": ["推理加速版本", "Acceleration build"],
    "model.diag": ["诊断", "Diagnostics"],

    "set.scale": ["界面缩放", "UI scale"],
    "set.scaleDesc": ["调整文字与控件大小", "Adjust text and control size"],
    "set.format": ["输出格式", "Output format"],
    "set.formatDesc": ["全局默认，影响单文件输出", "Global default for transcription output"],
    "set.fmtSrt": ["SRT 字幕", "SRT subtitles"],
    "set.fmtTxt": ["纯文本", "Plain text"],
    "set.vad": ["语音检测灵敏度（VAD）", "Voice detection (VAD)"],
    "set.vadDesc": ["降低阈值可减少漏识（被当成空白的片段可能有声音）；提高则减少假阳性。默认 0.50", "Lower threshold reduces misses; higher reduces false positives. Default 0.50"],
    "set.chunk": ["每段最长秒数（字级对齐）", "Max chunk seconds (alignment)"],
    "set.chunkDesc": ["调短可减少长段歌词／语音“段尾字级逐渐跑掉、换段才归位”的漂移。实际上限依模型自动压低（0.6B 30s、1.7B 10s）；CrispASR 走自身窗口，不受此设置影响。", "Shorter chunks reduce word-timing drift toward the end of long sung/spoken segments. The cap is auto-limited per model (0.6B 30s, 1.7B 10s); CrispASR uses its own window and is unaffected."],
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

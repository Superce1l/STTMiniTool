/* ============================================================
   bridge.js — 统一前端桥接层 window.QwenAPI
   三版共享同一前端的关键：UI 只调用 QwenAPI.*，由本层检测并路由：
     • web  : 本机/局域网 HTTP（桌面 = 本机 server + Edge；端点 = LAN）
              方法打 /api/*，进度经 SSE /api/events 推回
     • mock : 纯展示（file:// 或无后端，回范例数据供设计预览）
   进度等长时间事件通过内建 event bus；web 模式由服务器以 SSE 推送，
   mock 模式自行模拟 —— UI 两者皆以 QwenAPI.on('progress', …) 接收。
   ============================================================ */
(function () {
  "use strict";

  // ── event bus ───────────────────────────────────────────
  const listeners = {};
  function on(ev, fn) { (listeners[ev] ||= new Set()).add(fn); return () => off(ev, fn); }
  function off(ev, fn) { listeners[ev]?.delete(fn); }
  function emit(ev, payload) { listeners[ev]?.forEach(fn => { try { fn(payload); } catch (e) { console.error(e); } }); }

  let MODE = "mock";
  const KEY = new URLSearchParams(location.search).get("k") || "";
  const isHttp = () => location.protocol === "http:" || location.protocol === "https:";

  // HTTP 时探测 /health：有后端 → web；纯静态/无 API → 降级 mock。
  async function probeHttp() {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 1500);
      const r = await fetch("/health", { signal: ctrl.signal, headers: authHeaders() });
      clearTimeout(t);
      return r.ok;
    } catch { return false; }
  }

  const ready = new Promise(resolve => {
    async function settle() {
      MODE = (isHttp() && await probeHttp()) ? "web" : "mock";
      console.info("[QwenAPI] transport =", MODE);
      if (MODE === "web") connectSSE();
      resolve(MODE);
    }
    if (document.readyState === "complete") setTimeout(settle, 0);
    else window.addEventListener("load", () => setTimeout(settle, 0), { once: true });
  });

  // ── web：SSE 进度/状态流式 ──────────────────────────────
  function connectSSE() {
    try {
      const es = new EventSource("/api/events" + (KEY ? "?k=" + encodeURIComponent(KEY) : ""));
      es.onmessage = e => {
        try { const m = JSON.parse(e.data); if (m && m.event) emit(m.event, m.payload); }
        catch { /* 心跳/注释行，略过 */ }
      };
      es.onerror = () => { /* EventSource 会自动重连 */ };
    } catch (e) { console.warn("[QwenAPI] SSE 无法建立", e); }
  }

  // ── web：fetch 辅助 ─────────────────────────────────────
  function authHeaders(extra) {
    const h = Object.assign({}, extra || {});
    if (KEY) h["Authorization"] = "Bearer " + KEY;
    return h;
  }
  function withKey(path) { return KEY ? path + (path.includes("?") ? "&" : "?") + "k=" + encodeURIComponent(KEY) : path; }
  async function apiGet(path) {
    const r = await fetch(withKey(path), { headers: authHeaders() });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }
  async function apiPost(path, body) {
    const r = await fetch(withKey(path), {
      method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).error?.message || msg; } catch {}
      throw new Error(msg);
    }
    return r.json();
  }

  // ════════════════════════════════════════════════════════
  // 公开 API（web 打 /api/*；mock 回范例数据）
  // 桌面独有能力（pickFile）在 web/mock 回 null → UI 改用 <input> fallback
  // ════════════════════════════════════════════════════════
  const api = {
    on, off, _emit: emit,
    get mode() { return MODE; },
    ready,

    async getStatus() {
      if (MODE === "web") return apiGet("/api/status");
      return MOCK.status;
    },

    async pickFile() { return null; },            // 浏览器环境一律用 <input type=file>
    async loadHintTxt() { return null; },

    // 原生「选择文件夹」对话框（后端 tkinter）；取消/不支持 → {ok:false}
    async browseFolder(start) {
      if (MODE === "web") { try { return await apiPost("/api/browse-folder", { start }); } catch {} }
      return { ok: false, path: "" };
    },
    // 原生「选择文件」对话框（FFmpeg 可执行文件等）
    async browseFile(start) {
      if (MODE === "web") { try { return await apiPost("/api/browse-file", { start }); } catch {} }
      return { ok: false, path: "" };
    },

    // opts: {file, language, diarize, nSpeakers, align, hint}
    async transcribe(opts) {
      if (MODE === "web") return webTranscribe(opts);
      return mockTranscribe(opts);
    },
    async cancel() {
      if (MODE === "web") { try { await apiPost("/api/cancel", {}); } catch {} return true; }
      MOCK.cancelled = true; return true;
    },

    async openOutputDir() {
      if (MODE === "web") { try { await apiPost("/api/open-output", {}); } catch {} return true; }
      return false;
    },
    async checkUpdate() {
      // 更新链接由后端配置（RELEASES_URL）；未配置时后端返回 ok:false 且不跳转
      if (MODE === "web") { try { return await apiPost("/api/check-update", {}); } catch {} }
      return { ok: false, url: "", error: "未配置更新链接" };
    },

    async listDevices() {
      if (MODE === "web") return apiGet("/api/devices");
      return MOCK.devices;
    },
    // CrispASR 推理加速版本（Vulkan / CUDA / CPU）：检测硬件 → 给推荐 + 菜单
    async getAccel() {
      if (MODE === "web") return apiGet("/api/accel");
      return MOCK.accel;
    },
    async setAccel(variant) {
      if (MODE === "web") return apiPost("/api/accel", { variant });
      MOCK.accel.selected = variant;
      return { ok: true, variant, restartRequired: true,
               message: "（展示模式）已记住加速版本，重新启动后套用" };
    },
    async setBackend(id) {
      if (MODE === "web") return apiPost("/api/backend", { index: id });
      return true;
    },
    async getModelOptions() {
      if (MODE === "web") return apiGet("/api/model-options");
      return MOCK.modelOptions;
    },
    // 基础模式：用途清单（后端已套用硬件建议）
    async getBasicProfiles() {
      if (MODE === "web") return apiGet("/api/basic-profiles");
      return MOCK.basicProfiles;
    },
    async setModel(core, model) {
      if (MODE === "web") return apiPost("/api/model", { core, model });
      return { ok: true, core, model, arch: "（mock）", restartRequired: false, canLoadNow: true,
               message: `（展示模式）已选「${core} · ${model}」，点「下载并加载模型」开始` };
    },
    // 首次选定模型 → 就地下载并加载（进度走 SSE "progress"，完成走 "status"）
    async startLoad() {
      if (MODE === "web") return apiPost("/api/load", {});
      // mock：模拟加载完成
      setTimeout(() => { MOCK.status.modelReady = true; emit("status", { modelReady: true }); }, 1200);
      return { ok: true, loading: true };
    },
    async getLanguages() {
      if (MODE === "web") return apiGet("/api/languages");
      return MOCK.languages;
    },
    async getHealthCheck() {
      if (MODE === "web") return apiGet("/api/health-check");
      return MOCK.health;
    },

    async getSettings() {
      if (MODE === "web") return apiGet("/api/settings");
      return MOCK.settings;
    },
    async setSettings(patch) {
      if (MODE === "web") return apiPost("/api/settings", patch);
      Object.assign(MOCK.settings, patch); return MOCK.settings;
    },
  };

  // ── web 转录：POST /api/transcribe（进度走 SSE）──────────
  async function webTranscribe(opts) {
    if (!opts.file) throw new Error("请先选择文件");
    const fd = new FormData();
    fd.append("file", opts.file, opts.file.name);
    if (opts.language) fd.append("language", opts.language);
    fd.append("align", opts.align ? "1" : "0");
    fd.append("diarize", opts.diarize ? "1" : "0");
    if (opts.nSpeakers && opts.nSpeakers !== "auto") fd.append("n_speakers", String(opts.nSpeakers));
    if (opts.hint) fd.append("hint", opts.hint);
    emit("progress", { pct: 3, status: "上传中…" });
    const r = await fetch(withKey("/api/transcribe"), { method: "POST", body: fd, headers: authHeaders() });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).error?.message || msg; } catch {}
      throw new Error(msg);
    }
    return r.json();   // {segments, srtPath}
  }

  // ════════════════════════════════════════════════════════
  // mock：纯展示数据（让设计稿在浏览器中完全可互动）
  // ════════════════════════════════════════════════════════
  const MOCK = {
    cancelled: false,
    status: { modelReady: true, backend: "GPU · CRISPASR（Vulkan）", device: "NVIDIA GeForce RTX", version: "webview 0.1", appName: "语音识别小工具", hasAnyModel: false, selectedReady: false },
    settings: { scale: 100, format: "srt", vocab: "off", mirror: "", ffmpeg: "", theme: "light", uiLang: "简体中文", vad: 0.5, chunkSecs: 0 },
    basicProfiles: {
      tier: "gpu", accel: "Vulkan",
      hardware: "（展示模式）检测到独立显卡：NVIDIA GeForce RTX 4070 —— 可用最准的模型。",
      current: { core: "Qwen", model: "Qwen3-ASR-1.7B Q8 (CRISPASR)" },
      profiles: [
        { key: "zh", title: "中文（标准）", desc: "一般用途首选：中文为主、断句与标点最完整。",
          note: "", core: "Qwen", model: "Qwen3-ASR-1.7B Q8 (CRISPASR)",
          arch: "GPU · CRISPASR（Vulkan）", present: true, selected: true },
        { key: "ja", title: "日本语", desc: "日语／动漫特化，保留日文原生汉字。", note: "",
          core: "Qwen", model: "Qwen3-ASR-1.7B 日语动漫 Q8 (CRISPASR)",
          arch: "GPU · CRISPASR（Vulkan）", present: false, selected: false },
        { key: "whisper", title: "OpenAI Whisper（通用）", desc: "多语言通用模型，99 种语言，速度快。",
          note: "", core: "Whisper", model: "Whisper Base",
          arch: "GPU · CRISPASR（Vulkan）", present: false, selected: false },
      ],
    },
    accel: {
      ok: true, selected: "vulkan", recommended: "vulkan",
      reason: "（展示模式）检测到 NVIDIA 显卡。默认仍用 Vulkan（仅 34 MB、实测速度接近）。",
      latest: "0.8.32", installed: { version: "0.8.32", variant: "vulkan" }, needsDownload: false,
      hardware: { gpus: [{ vendor: "nvidia", name: "NVIDIA GeForce RTX 4070", vram_mb: 8192 }] },
      options: [
        { key: "vulkan", label: "Vulkan（通用 GPU）", size_mb: 34, recommended: true, suitable: true, note: "",
          desc: "NVIDIA / AMD / Intel 通吃，体积最小。" },
        { key: "cuda13", label: "CUDA 13（NVIDIA 新驱动）", size_mb: 484, recommended: false, suitable: true, note: "",
          desc: "NVIDIA 专用，自带 CUDA 13 runtime。" },
        { key: "cuda12", label: "CUDA 12（NVIDIA 专用）", size_mb: 690, recommended: false, suitable: true,
          note: "驱动版本建议改用另一个 CUDA 版本", desc: "NVIDIA 专用，自带 CUDA 12 runtime。" },
        { key: "cpu", label: "CPU（纯处理器）", size_mb: 8, recommended: false, suitable: true, note: "",
          desc: "完全不碰显卡。" },
        { key: "cpu-legacy", label: "CPU 兼容版（无 AVX2 的老 CPU）", size_mb: 8, recommended: false,
          suitable: true, note: "只有在一般 CPU 版闪退时才需要", desc: "给不支持 AVX2 的老处理器。" },
      ],
    },
    devices: {
      devices: [
        { kind: "cpu", name: "AMD Ryzen 5 9600X 6-Core", note: "使用中" },
        { kind: "gpu", name: "NVIDIA GeForce RTX 4070", note: "8.0 GB 可用" },
      ],
      diag: { level: "info", text: "（展示数据）GPU 由 CrispASR 检测；实机会列出可用的独立显卡与可用内存。" },
    },
    modelOptions: {
      cores: [
        { label: "Qwen", models: [
          { label: "Qwen3-ASR-0.6B", backend: "openvino", arch: "CPU · OpenVINO INT8", note: "" },
          { label: "Qwen3-ASR-1.7B INT8", backend: "openvino", arch: "CPU · OpenVINO INT8", note: "" },
          { label: "Qwen3-ASR-1.7B Q4 (CRISPASR/Vulkan)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）", note: "" },
          { label: "Qwen3-ASR-1.7B Q8 (CRISPASR/Vulkan)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）", note: "" },
          { label: "Qwen3-ASR-1.7B Q8（Vulkan · 兼容）", backend: "chatllm", arch: "GPU · chatllm Vulkan", note: "⚠️ chatllm 核心在部分 AMD 核显／APU 有已知兼容问题，且核心二进位未随安装包提供（保留供既有使用者向下兼容）；若遇死机或无输出，建议改用「Qwen · CRISPASR/Vulkan」核心。" },
        ]},
        { label: "Whisper", models: [
          { label: "Whisper Base", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
          { label: "Whisper Small", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
          { label: "Whisper Medium", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
          { label: "Whisper Large", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
          { label: "Whisper Large Turbo", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
        ]},
      ],
      current: { core: "Qwen", model: "Qwen3-ASR-0.6B" },
      activeArch: "CPU · OpenVINO INT8",
    },
    languages: {
      languages: [
        { label: "自动检测", value: "" }, { label: "Chinese", value: "Chinese" },
        { label: "English", value: "English" }, { label: "Japanese", value: "Japanese" },
        { label: "Korean", value: "Korean" }, { label: "Cantonese", value: "Cantonese" },
      ],
    },
    health: {
      summary: { red: 0, yellow: 3, ok: true }, activeBackend: "openvino",
      cores: [
        { label: "Qwen · OpenVINO（CPU）", backend: "openvino", items: [
          { key: "model", label: "ASR 模型（0.6B）", status: "green", detail: "已下载" },
          { key: "vad", label: "语音分段 VAD（silero）", status: "green", detail: "已内建" },
          { key: "fa", label: "时间轴对齐 FA", status: "yellow", detail: "未下载（约 939MB，启用对齐时下载）" },
          { key: "diar", label: "说话者分离（外部 ONNX）", status: "yellow", detail: "未下载（约 32MB，启用分离时下载）" },
        ]},
        { label: "CRISPASR（Vulkan：Whisper + Qwen）", backend: "crispasr", items: [
          { key: "core", label: "CrispASR 核心（crispasr.exe）", status: "green", detail: "已下载" },
          { key: "whisper", label: "OpenAI Whisper 模型（Base）", status: "green", detail: "已下载" },
          { key: "qwen", label: "Qwen3-ASR-1.7B 模型（Q8）", status: "yellow", detail: "未下载（启用时下载）" },
          { key: "fa", label: "时间轴对齐 FA（aligner gguf Q5）", status: "green", detail: "已下载" },
          { key: "diar", label: "说话者分离（外部 ONNX，与 OpenVINO 共享）", status: "yellow", detail: "未下载" },
        ]},
      ],
      shared: [
        { key: "ffmpeg", label: "FFmpeg（视频抽音轨用）", status: "green", detail: "已检测：C:/ffmpeg/ffmpeg.exe" },
        { key: "diar_shared", label: "说话者分离模型（共享）", status: "yellow", detail: "未下载（启用分离时自动下载）" },
      ],
    },
    segments: [
      { start: 3.0, end: 7.2, speaker: 1, text: "大家好，今天我们来聊聊本地语音识别这个主题。" },
      { start: 7.4, end: 11.8, speaker: 2, text: "对啊，最吸引我的是数据完全不用上传云端，隐私有保障。" },
      { start: 12.0, end: 16.5, speaker: 1, text: "没错，而且这套支持说话者分离，会议记录整理起来特别方便。" },
      { start: 16.8, end: 21.0, speaker: 2, text: "时间轴对齐也很实用，字幕可以直接导出成 SRT 或纯文本。" },
      { start: 21.3, end: 26.0, speaker: 1, text: "如果显卡兼容，用 GPU 核心速度会再快上好几倍。" },
      { start: 26.2, end: 31.0, speaker: 2, text: "那我们下半段就来实际示范一次完整的转录流程吧。" },
    ],
  };

  // mock 字级：把每行文字（去标点）平均洒进该行时间，近似 FA 字级时间轴，
  // 让卡拉OK模式在纯展示（file:// / 无后端）下也能完整预览。真实后端回的
  // words 为 FA 不等距时间，但前端渲染只读 words → mock 与实机同一条路径。
  MOCK.segments.forEach(s => {
    const chars = [...s.text].filter(c => !"，。？！；：、…—·".includes(c));
    const dur = chars.length ? (s.end - s.start) / chars.length : 0;
    s.words = chars.map((ch, i) => ({
      start: s.start + i * dur, end: s.start + (i + 1) * dur, text: ch,
    }));
  });

  async function mockTranscribe() {
    MOCK.cancelled = false;
    const steps = [
      [12, "加载音讯…"], [25, "语音分段（VAD）…"], [44, "识别中 · 第 3/8 段"],
      [68, "识别中 · 第 5/8 段"], [86, "时间轴对齐…"], [100, "完成"],
    ];
    for (const [pct, status] of steps) {
      if (MOCK.cancelled) throw new Error("已取消");
      await sleep(420);
      emit("progress", { pct, status });
    }
    return { segments: MOCK.segments, srtPath: "subtitles/B.srt" };
  }
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  window.QwenAPI = api;
})();

/* ============================================================
   app.js — UI 逻辑（与后端解耦，全程只通过 window.QwenAPI）
   ============================================================ */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const API = window.QwenAPI;
  // i18n 取字（含 {n} 等简单插值）；无字典或无此键时回退默认值 def。
  function T(key, def, vars) {
    let s = (window.I18N && I18N.t(key)) || def || key;
    if (vars) for (const k in vars) s = s.replace("{" + k + "}", vars[k]);
    return s;
  }

  const VIEW_TITLES = {
    file: "音频转字幕", record: "录制转换",
    model: "模型与设备", settings: "设置",
  };

  // ── 导览切换 ────────────────────────────────────────────
  let _curView = "file";
  function viewTitle(name) { return (window.I18N && I18N.t("view." + name)) || VIEW_TITLES[name] || ""; }
  function refreshViewTitle() { $("#view-title").textContent = viewTitle(_curView); }
  function switchView(name) {
    _curView = name;
    $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
    $$(".view").forEach(v => v.classList.toggle("active", v.dataset.view === name));
    $("#view-title").textContent = viewTitle(name);
    $("#view-ctx").textContent = "";
    if (name === "model") renderModel();
    if (name === "record") { enumerateMics(); refreshMicPerm(); }   // 列举设备 + 麦克风权限状态
  }
  $("#nav").addEventListener("click", e => {
    const btn = e.target.closest(".nav-item");
    if (btn) switchView(btn.dataset.view);
  });

  // ── 状态栏 ──────────────────────────────────────────────
  async function refreshStatus() {
    const s = await API.getStatus();
    const el = $("#model-status");
    el.classList.toggle("loading", !s.modelReady);
    $(".t", el).textContent = s.modelReady ? T("status.ready", "模型已就绪")
      : (s.loading ? T("status.loading", "加载模型中…") : T("status.needModel", "尚未加载模型"));
    // 版本徽章：纯数字版本前缀 v（如 v1.0.9）；已含文字者（如 webview 0.1）原样显示
    if (s.version) $("#app-version").textContent = /^\d/.test(s.version) ? "v" + s.version : s.version;
  }

  // ════════════════════════════════════════════════════════
  // 音频转字幕
  // ════════════════════════════════════════════════════════
  let picked = null;             // {path?, name, file?} 或 null
  const drop = $("#drop"), hiddenFile = $("#hidden-file");

  function setFile(f) {
    picked = f;
    if (!f) { drop.classList.remove("has-file"); renderDropEmpty(); $("#btn-run").disabled = true; return; }
    drop.classList.add("has-file");
    const meta = f.sizeSec ? `${f.sizeSec} 秒` : (f.size ? (f.size / 1048576).toFixed(1) + " MB" : "");
    drop.innerHTML = `<span class="file-chip">${escapeHtml(f.name)}
      ${meta ? `<span class="meta">${meta}</span>` : ""}
      <button class="x" title="移除">✕</button></span>`;
    drop.querySelector(".x").addEventListener("click", ev => { ev.stopPropagation(); setFile(null); });
    $("#btn-run").disabled = false;
  }
  function renderDropEmpty() {
    drop.innerHTML = `<div class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/></svg></div>
      <div class="big">拖曳音讯到此，或<b>点击选择文件</b></div>
      <div class="sub">支持 mp3 / wav / m4a / 视频音轨</div>`;
  }

  // 桌面：原生对话框；web/mock：fallback 到 <input type=file>
  drop.addEventListener("click", async () => {
    if (drop.classList.contains("has-file")) return;
    const native = await API.pickFile();
    if (native) setFile(native);
    else hiddenFile.click();
  });
  hiddenFile.addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });
  ["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("hot"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("hot"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });

  // 读入 TXT：浏览器沙箱用隐藏 <input type=file> + FileReader 读文字进提示框
  const _txtInput = document.createElement("input");
  _txtInput.type = "file"; _txtInput.accept = ".txt,text/plain"; _txtInput.style.display = "none";
  document.body.appendChild(_txtInput);
  _txtInput.addEventListener("change", () => {
    const f = _txtInput.files && _txtInput.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = () => { $("#hint-box").value = String(r.result || ""); };
    r.readAsText(f, "utf-8");
    _txtInput.value = "";
  });
  $("#btn-load-txt").addEventListener("click", () => _txtInput.click());

  // 转录
  let running = false;
  $("#btn-run").addEventListener("click", async () => {
    if (!picked || running) return;
    running = true;
    const btn = $("#btn-run");
    btn.disabled = true; btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>识别中…`;
    $("#result").hidden = true;
    showProgress(0, "准备中…");
    logLine("▶ 开始转换：" + picked.name);
    try {
      const res = await API.transcribe({
        path: picked.path, file: picked.file || picked,
        language: $("#sel-lang").value || null,
        diarize: $("#sw-diar").checked,
        nSpeakers: $("#sel-spk").value,
        align: $("#sw-align").checked,
        hint: $("#hint-box").value.trim(),
      });
      renderResult(res.segments);
      logLine("✓ 完成，共 " + res.segments.length + " 段" + (res.srtPath ? "，已输出 " + res.srtPath : ""));
      $("#btn-open-dir").disabled = false;
      $("#btn-save-sub").disabled = false;
    } catch (err) {
      showProgress(0, "");
      $("#progress").hidden = true;
      logLine("✕ " + (err.message || err));
    } finally {
      running = false;
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>开始转换`;
    }
  });

  // 进度分派：模型就地加载中 → 模型页进度；批量执行中 → 当前任务列；否则 → 单文件进度条
  API.on("progress", ({ pct, status }) => {
    if (_modelLoading) {
      setModelProg(pct, status);
    } else {
      showProgress(pct, status);
    }
  });
  function showProgress(pct, status) {
    $("#progress").hidden = false;
    $("#prog-bar").style.width = pct + "%";
    $("#prog-pct").textContent = Math.round(pct) + "%";
    if (status != null) $("#prog-status").textContent = status;
  }

  // ════ Signature：真实波形 + 播放 + 分段标注（Suno 式）════
  //   用 Web Audio decodeAudioData 解出真实峰值画波形；<audio> 播放，
  //   播放头跟走、已播段标色；波形下方对齐时间轴的字幕区块 + 字幕卡
  //   随播放高亮，皆可点击 seek —— 肉眼即可看字幕与音讯的对齐准确度。
  const WAVE_N = 200;
  let audioEl = null, audioUrl = null, waveDur = 0, curSegs = [];

  async function renderResult(segments) {
    $("#result").hidden = false;
    curSegs = segments;
    _karaCache = null;                      // 新结果 → 重建卡拉OK字级（段落索引已变）
    cleanupAudio();

    const file = picked && (picked.file || (picked instanceof Blob ? picked : null));
    let peaks = null;
    waveDur = segments.length ? segments[segments.length - 1].end : 0;

    if (file instanceof Blob) {
      audioUrl = URL.createObjectURL(file);
      audioEl = new Audio(audioUrl);
      try {                                   // 解码取真实峰值（mp3/wav/m4a/webm…）
        const ab = await file.arrayBuffer();
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const buf = await ctx.decodeAudioData(ab.slice(0));
        waveDur = buf.duration || waveDur;
        peaks = computePeaks(buf, WAVE_N);
        ctx.close();
      } catch (e) { peaks = null; }           // 不支持的格式 → 平波形，但仍可播
    }
    waveDur = waveDur || 60;

    $("#wave-dur").textContent = fmtClock(waveDur);
    drawWave($("#wave"), peaks, segments, waveDur);
    renderSegStrip(segments, waveDur);
    renderSubs(segments);
    wireAudio();
    syncStickyOffset();
  }

  // 波形面板为 sticky 钉在顶端 → 滚动容器需保留等高的顶部空间，
  // 否则 scrollIntoView 自动卷到的字幕卡会被面板遮住。
  function syncStickyOffset() {
    const panel = $(".wave-panel"), host = $(".view-host");
    if (!panel || !host) return;
    host.style.scrollPaddingTop = (panel.offsetHeight + 12) + "px";
  }
  window.addEventListener("resize", syncStickyOffset);

  // 真实峰值：每桶取绝对值最大，整体正规化
  function computePeaks(buf, n) {
    const ch = buf.getChannelData(0);
    const block = Math.max(1, Math.floor(ch.length / n));
    const peaks = [];
    for (let i = 0; i < n; i++) {
      let mx = 0;
      const base = i * block;
      for (let j = 0; j < block; j++) { const v = Math.abs(ch[base + j] || 0); if (v > mx) mx = v; }
      peaks.push(mx);
    }
    const norm = Math.max(...peaks, 1e-4);
    return peaks.map(p => p / norm);
  }

  function drawWave(container, peaks, segments, dur) {
    const n = peaks ? peaks.length : WAVE_N;
    let bars = "";
    for (let i = 0; i < n; i++) {
      const amp = peaks ? peaks[i] : 0.12;
      bars += `<div class="b" style="height:${Math.max(3, amp * 64).toFixed(1)}px"></div>`;
    }
    let marks = "";                            // 分段边界标记
    segments.forEach(s => {
      marks += `<div class="seg-mark" style="left:${((s.start / dur) * 100).toFixed(2)}%"></div>`;
    });
    container.innerHTML = `<div class="playhead" id="playhead" style="left:0%"></div>${marks}${bars}`;
  }

  // 波形下方：依时间对齐的字幕区块（Suno 式，可点击 seek）
  function renderSegStrip(segments, dur) {
    let host = $("#seg-strip");
    if (!host) { host = document.createElement("div"); host.id = "seg-strip"; host.className = "seg-strip"; $(".wave-panel").appendChild(host); }
    host.innerHTML = "";
    segments.forEach((s, i) => {
      const left = (s.start / dur) * 100, w = Math.max(0.6, ((s.end - s.start) / dur) * 100);
      const blk = document.createElement("div");
      blk.className = "seg-block" + (s.speaker ? " spk-" + (((s.speaker - 1) % 3) + 1) : "");
      blk.style.left = left.toFixed(2) + "%"; blk.style.width = w.toFixed(2) + "%";
      blk.dataset.seg = i; blk.title = s.text;
      blk.textContent = s.text;
      blk.addEventListener("click", () => seekTo(s.start));
      host.appendChild(blk);
    });
  }

  function renderSubs(segments) {
    const host = $("#subs"); host.innerHTML = "";
    segments.forEach((s, i) => host.appendChild(subCard(s, i)));
  }
  function subCard(s, idx) {
    const el = document.createElement("div");
    // spkc-N：依说话者标示左侧外框色（与 chip 同色系，不污染卡片背景）
    el.className = "sub-card" + (s.speaker ? " spkc-" + (((s.speaker - 1) % 3) + 1) : "");
    el.dataset.seg = idx;
    const spk = s.speaker ? `<span class="chip spk-${((s.speaker - 1) % 3) + 1}">说话者 ${s.speaker}</span>` : "";
    el.innerHTML = `<span class="tc">${fmtClock(s.start)} → ${fmtClock(s.end)}</span>
      <div class="body"><div class="spk">${spk}</div><div class="txt">${escapeHtml(s.text)}</div></div>
      <button class="sub-edit" title="编辑此行" aria-label="编辑此行">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
      </button>`;
    el.addEventListener("click", e => {
      if (e.target.closest(".sub-edit") || el.classList.contains("editing")) return;
      seekTo(s.start);
    });
    el.querySelector(".sub-edit").addEventListener("click", e => { e.stopPropagation(); beginEdit(el, idx); });
    return el;
  }

  // 行内校对：笔 → 该行可编辑，Enter 存储 / Esc 取消；同步回 curSegs 与波形对齐区块。
  // curSegs 为下载字幕的数据来源，故编辑后存档即反映校对结果。
  function beginEdit(card, idx) {
    if (card.classList.contains("editing")) return;
    card.classList.add("editing");
    const txt = card.querySelector(".txt");
    const orig = curSegs[idx].text;
    txt.contentEditable = "true"; txt.spellcheck = false;
    txt.focus();
    const range = document.createRange(); range.selectNodeContents(txt);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    function finish(save) {
      txt.removeEventListener("keydown", onKey);
      txt.removeEventListener("blur", onBlur);
      txt.contentEditable = "false";
      card.classList.remove("editing");
      const val = (txt.textContent || "").trim();
      if (save && val && val !== orig) {
        curSegs[idx].text = val; txt.textContent = val;
        curSegs[idx].words = reflowWords(curSegs[idx], val);  // 卡拉OK字级跟著修正
        _karaCache = null;                                    // 失效缓存 → 下次重绘该行
        const blk = $(`.seg-block[data-seg="${idx}"]`);
        if (blk) { blk.textContent = val; blk.title = val; }
      } else {
        txt.textContent = orig;               // 取消或清空 → 还原
      }
      txt.blur();
    }
    function onKey(e) {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    }
    function onBlur() { finish(true); }
    txt.addEventListener("keydown", onKey);
    txt.addEventListener("blur", onBlur);
  }

  // ── 播放接线 ────────────────────────────────────────────
  function wireAudio() {
    const wave = $("#wave"), play = $("#wave-play");
    const bars = $$(".b", wave);
    if (audioEl) {
      audioEl.addEventListener("timeupdate", onTime);
      audioEl.addEventListener("play", () => setPlayIcon(true));
      audioEl.addEventListener("pause", () => setPlayIcon(false));
      audioEl.addEventListener("ended", () => { setPlayIcon(false); });
    }
    // 点波形 seek
    wave.onclick = e => {
      const r = wave.getBoundingClientRect();
      seekTo(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * waveDur);
    };
    if (play) play.onclick = () => { if (!audioEl) return; audioEl.paused ? audioEl.play() : audioEl.pause(); };
    // 停止：暂停并回到开头（像播放器的 stop）
    const stop = $("#wave-stop");
    if (stop) stop.onclick = () => { if (!audioEl) return; audioEl.pause(); audioEl.currentTime = 0; onTime(); };

    function onTime() {
      const t = audioEl.currentTime, ratio = waveDur ? t / waveDur : 0;
      const ph = $("#playhead"); if (ph) ph.style.left = (ratio * 100).toFixed(2) + "%";
      const playedIdx = Math.floor(ratio * bars.length);
      bars.forEach((b, i) => b.classList.toggle("played", i <= playedIdx));
      const cur = curSegs.findIndex(s => t >= s.start && t < s.end);
      $$(".sub-card").forEach(c => c.classList.toggle("playing", +c.dataset.seg === cur));
      $$(".seg-block").forEach(c => c.classList.toggle("playing", +c.dataset.seg === cur));
      const tm = $("#wave-cur"); if (tm) tm.textContent = fmtClock(t);
      // 自动滚动到播放中的字幕卡
      if (cur >= 0) { const el = $(`.sub-card[data-seg="${cur}"]`); if (el && playFollow) el.scrollIntoView({ block: "nearest" }); }
    }
  }
  let playFollow = true;
  function setPlayIcon(playing) {
    const b = $("#wave-play"); if (!b) return;
    b.innerHTML = playing
      ? `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
  }
  function seekTo(sec) { if (audioEl) { audioEl.currentTime = sec; audioEl.play().catch(() => {}); } }
  function cleanupAudio() {
    try { if (audioEl) { audioEl.pause(); audioEl.src = ""; } } catch (e) {}
    if (audioUrl) { try { URL.revokeObjectURL(audioUrl); } catch (e) {} audioUrl = null; }
    audioEl = null;
  }

  // ════ 卡拉OK模式（逐字高亮放大跳动，由字级 words 驱动）════
  //   另开 requestAnimationFrame 循环（60fps）读 audioEl.currentTime，逐字更新，
  //   比 <audio> timeupdate（~4fps）顺得多。数据源：curSegs[].words（后端 FA 字级；
  //   无对齐时 wordsForSeg 以行时间平均内插）。
  let karaokeOn = false, karaokeRAF = 0, _karaCache = null;   // _karaCache: {idx, words, spans}

  /**
   * 卡拉OK逐字状态：给「该行字级 words」与「目前播放秒数 t」，返回与 words 等长的
   * 状态数组，每元素 { sung, active, scale }：
   *   sung   已唱过（t 已过该字 end）→ CSS .sung 填主题色
   *   active 正在唱（t 落在该字 [start,end)）→ CSS .active 高亮
   *   scale  1=原始大小；>1=放大（updateKaraoke 会映射成 transform：放大 + 微上抬）
   *
   * 目前 sung/active 已实现（KTV 式逐字填色已可运作）。放大跳动曲线留给你 ——
   * 这是「逐字放大跳动」效果的核心，也是最有设计选择的一段。
   */
  function karaokeCharStates(words, t) {
    return words.map(w => {
      const sung = t >= w.end;
      const active = t >= w.start && t < w.end;
      let scale = 1;
      // ── TODO（你来写）：放大跳动曲线（5~8 行）──────────────────────────
      //   目标：字被唱到的瞬间「弹」一下再回落（Apple Music 动态歌词风）。
      //   手上的量：w.start / w.end（该字时间窗）、t（目前秒数）、active。
      //   设计空间（任选或自创）：
      //     • 进度 p = (t - w.start) / (w.end - w.start)，clamp 0..1
      //     • 起跳即最大、唱完归位： if (active) scale = 1 + 0.25 * (1 - p)
      //     • 预备动作：在 w.start 前 ~0.1s 先微幅放大（anticipation）
      //     • 建议上限 scale ≤ 1.3，避免整行版面抖动过猛
      //   （把你的曲线写进 scale 即可，其余渲染/映射都已接好。）
      // ──────────────────────────────────────────────────────────────────
      return { sung, active, scale };
    });
  }

  // 卡拉OK高亮单位：去空白与标点（与后端 FA words 惯例一致，标点不单独成一拍）。
  const _KARA_PUNCT = "，。？！；：、…—·,.!?;:";
  function karaokeUnits(text) {
    return [...(text || "")].filter(c => c.trim() && !_KARA_PUNCT.includes(c));
  }
  // 把某段 chars 依行时间平均内插成 words（无 FA、或编辑后字数改变时用）。
  function interpWords(seg, units) {
    const dur = units.length ? (seg.end - seg.start) / units.length : 0;
    return units.map((ch, i) => ({ start: seg.start + i * dur, end: seg.start + (i + 1) * dur, text: ch }));
  }
  // 取某段字级 words；后端未对齐（words 空）→ 行时间平均内插（时间为估计值）。
  function wordsForSeg(seg) {
    if (seg.words && seg.words.length) return seg.words;
    return interpWords(seg, karaokeUnits(seg.text));
  }
  // 行内校对后同步卡拉OK字级：字数不变（多为单字修正）→ 保留原 FA 时间只换字；
  // 字数改变 → 该行 FA 已不对应，改以行时间平均重洒。返回新 words。
  function reflowWords(seg, newText) {
    const units = karaokeUnits(newText);
    const old = seg.words || [];
    if (old.length && old.length === units.length) {
      return old.map((w, i) => ({ start: w.start, end: w.end, text: units[i] }));
    }
    return interpWords(seg, units);
  }

  // 把某段渲染成一排 .kara-char span，返回 {words, spans}（spans 与 words 等长）。
  function renderKaraokeLine(idx) {
    const seg = curSegs[idx], line = $("#kara-line");
    line.innerHTML = "";
    const words = wordsForSeg(seg), spans = [];
    words.forEach(w => {
      const span = document.createElement("span");
      span.className = "kara-char"; span.textContent = w.text;
      line.appendChild(span); spans.push(span);
      if (/[a-z]/i.test(w.text)) {                 // 拉丁词后补空白格
        const sp = document.createElement("span"); sp.className = "kara-char space"; line.appendChild(sp);
      }
    });
    const next = curSegs[idx + 1];
    $("#kara-next").textContent = next ? next.text : "";
    return { words, spans };
  }

  // 每帧更新：定位目前段 → （换段才重绘）→ 算每字状态 → 套用 transform。
  function updateKaraoke(t) {
    if (!curSegs.length) return;
    let idx = curSegs.findIndex(s => t >= s.start && t < s.end);
    if (idx < 0) idx = _karaCache ? _karaCache.idx : 0;   // 空隙：停在上一段（唱完状态）
    if (!_karaCache || _karaCache.idx !== idx) {
      const built = renderKaraokeLine(idx);
      _karaCache = { idx, words: built.words, spans: built.spans };
    }
    const { words, spans } = _karaCache;
    const states = karaokeCharStates(words, t);          // ← 使用者实现的高亮曲线
    for (let i = 0; i < spans.length; i++) {
      const st = states[i] || {};
      spans[i].classList.toggle("sung", !!st.sung);
      spans[i].classList.toggle("active", !!st.active);
      const sc = +st.scale || 1;
      // scale→transform：放大同时微微上抬（跳动感）。视觉映射固定于此，曲线由 states 决定。
      spans[i].style.transform = sc === 1 ? "" : `translateY(${((sc - 1) * -12).toFixed(1)}px) scale(${sc.toFixed(3)})`;
    }
  }

  function karaokeTick() {
    if (!karaokeOn) return;
    updateKaraoke(audioEl ? audioEl.currentTime : 0);
    karaokeRAF = requestAnimationFrame(karaokeTick);
  }
  function setKaraoke(on) {
    karaokeOn = on;
    $("#karaoke").hidden = !on;
    $("#subs").hidden = on;
    $("#btn-karaoke").classList.toggle("kara-on", on);
    cancelAnimationFrame(karaokeRAF);
    if (on) { _karaCache = null; karaokeTick(); }
  }
  $("#btn-karaoke") && $("#btn-karaoke").addEventListener("click", () => setKaraoke(!karaokeOn));

  $("#btn-open-dir").addEventListener("click", () => API.openOutputDir());

  // ── 字幕存档：由内存中的 curSegs 在前端组档，下载到使用者指定位置 ──
  //   （浏览器沙箱不暴露来源档真实路径，无法写回来源文件夹，故走下载。
  //    SRT/TXT 格式比照 subtitle_lines，含「说话者N：」前缀。）
  function srtTs(s) {
    const ms = Math.round(s * 1000);
    const h = String(Math.floor(ms / 3600000)).padStart(2, "0");
    const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, "0");
    const sec = String(Math.floor((ms % 60000) / 1000)).padStart(2, "0");
    return `${h}:${m}:${sec},${String(ms % 1000).padStart(3, "0")}`;
  }
  function buildSrt(segs) {
    return segs.map((s, i) => {
      const spk = s.speaker ? `说话者${s.speaker}：` : "";
      return `${i + 1}\n${srtTs(s.start)} --> ${srtTs(s.end)}\n${spk}${s.text}\n`;
    }).join("\n");
  }
  function buildTxt(segs) {
    return segs.some(s => s.speaker)
      ? segs.map(s => (s.speaker ? `说话者${s.speaker}：` : "") + s.text).join("\n")
      : segs.map(s => s.text).join("");
  }
  async function saveSubtitle() {
    if (!curSegs.length) return;
    let fmt = "srt";
    try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    const text = fmt === "txt" ? buildTxt(curSegs) : buildSrt(curSegs);
    const base = (picked && picked.name) ? picked.name.replace(/\.[^.]+$/, "") : "字幕";
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = base + (fmt === "txt" ? ".txt" : ".srt");
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    logLine("✓ 已存档字幕：" + a.download);
  }
  $("#btn-save-sub").addEventListener("click", saveSubtitle);

  // ── 记录 ────────────────────────────────────────────────
  function logLine(msg) {
    const el = $("#file-log"); el.hidden = false;
    const d = document.createElement("div"); d.className = "ln"; d.textContent = msg;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  }

  // ════════════════════════════════════════════════════════
  // 模型与设备（核心 + 模型下拉 + 架构标签）
  // ════════════════════════════════════════════════════════
  let _modelOpt = null;     // 最近一次 getModelOptions 结果
  let _curCore = null;      // 目前选中的核心标签

  // ── 系统自检：每核心 × 每能力，显示 Green/Yellow/Red ──────────────
  async function renderHealth() {
    let h;
    try { h = await API.getHealthCheck(); } catch (e) { return; }
    const sum = $("#health-sum");
    if (h.summary) {
      if (h.summary.red > 0) { sum.className = "chip chip-live"; sum.textContent = `${h.summary.red} 项需处理`; }
      else if (h.summary.yellow > 0) { sum.className = "chip chip-accent"; sum.textContent = `核心就绪 · ${h.summary.yellow} 项将于启用时自动下载`; }
      else { sum.className = "chip chip-ok"; sum.textContent = "全部就绪"; }
    }
    const host = $("#health-panel"); host.innerHTML = "";
    const dot = s => `<span class="hdot hdot-${s}"></span>`;
    const groups = [...(h.cores || [])];
    if (h.shared && h.shared.length) groups.push({ label: "共享组件", items: h.shared });
    groups.forEach(g => {
      const card = document.createElement("div"); card.className = "health-core";
      const active = g.backend && g.backend === h.activeBackend;
      card.innerHTML = `<div class="hc-title">${escapeHtml(g.label)}` +
        (active ? ` <span class="chip chip-ok" style="font-size:10px">使用中</span>` : ``) + `</div>`;
      (g.items || []).forEach(it => {
        const row = document.createElement("div"); row.className = "hc-item";
        row.innerHTML = `${dot(it.status)}<span class="hi-label">${escapeHtml(it.label)}</span>` +
          `<span class="hi-detail">${escapeHtml(it.detail || "")}</span>`;
        card.appendChild(row);
      });
      host.appendChild(card);
    });
  }
  $("#btn-health-recheck") && $("#btn-health-recheck").addEventListener("click", renderHealth);

  async function renderModel() {
    renderHealth();
    // 核心卡 + 模型下拉
    try {
      _modelOpt = await API.getModelOptions();
      _curCore = _modelOpt.current.core;
      renderCoreCards();
      populateModels(_curCore, _modelOpt.current.model);
      // 已记住的选择 vs 实际加载的架构不同 → 提示重启
      const curArch = selectedArch();
      if (_modelOpt.activeArch && curArch && _modelOpt.activeArch !== curArch) {
        modelMsg(`已记住「${_curCore} · ${$("#model-select").value}」（${curArch}），`
          + `将于重新启动后套用（目前以 ${_modelOpt.activeArch} 运行）。`, "info");
      } else modelMsg("", null);
      updateLoadBtn();              // 设置「下载并加载／前往转录」按钮初始状态
    } catch (e) {}
    // 检测设备 + 诊断
    const d = await API.listDevices();
    const host = $("#dev-list"); host.innerHTML = "";
    d.devices.forEach(dev => {
      const cpu = dev.kind === "cpu";
      const el = document.createElement("div"); el.className = "dev";
      el.innerHTML = `<span class="ic">${cpu ? ICON_CPU : ICON_GPU}</span>
        <span class="nm">${dev.kind === "cpu" ? "CPU · " : "GPU · "}${escapeHtml(dev.name)}</span>
        <span class="vram">${escapeHtml(dev.note || "")}</span>`;
      host.appendChild(el);
    });
    await renderAccel();
    await renderBasic();
    const diag = $("#gpu-diag");
    if (d.diag && d.diag.text) {
      diag.hidden = false;
      diag.className = "banner " + (d.diag.level === "warn" ? "banner-warn" : "banner-info");
      const txt = diag.querySelector("div"); if (txt) txt.innerHTML = escapeHtml(d.diag.text);
    } else diag.hidden = true;
  }

  function coreObj(label) {
    const cs = _modelOpt && _modelOpt.cores || [];
    return cs.find(c => c.label === label) || cs[0];
  }
  function renderCoreCards() {
    const host = $("#core-cards"); host.innerHTML = "";
    (_modelOpt.cores || []).forEach(core => {
      const card = document.createElement("label");
      card.className = "radio-card" + (core.label === _curCore ? " sel" : "");
      card.dataset.core = core.label;
      card.innerHTML = `<span class="rd"></span><div class="info">
        <div class="t">${escapeHtml(core.label)}</div>
        <div class="d">${escapeHtml(T("model.nModels", `${core.models.length} 种模型可选`, { n: core.models.length }))}</div></div>`;
      host.appendChild(card);
    });
  }
  function populateModels(coreLabel, modelLabel) {
    const core = coreObj(coreLabel); if (!core) return;
    const sel = $("#model-select"); sel.innerHTML = "";
    core.models.forEach(m => {
      const o = document.createElement("option");
      o.value = m.label; o.textContent = m.label; o.dataset.arch = m.arch;
      o.dataset.note = m.note || "";
      if (m.label === modelLabel) o.selected = true;
      sel.appendChild(o);
    });
    if (!core.models.some(m => m.label === modelLabel) && core.models[0]) sel.value = core.models[0].label;
    updateArch();
  }
  function selectedArch() { const o = $("#model-select").selectedOptions[0]; return o ? o.dataset.arch : ""; }
  function updateArch() {
    const o = $("#model-select").selectedOptions[0];
    $("#model-arch").textContent = (o && o.dataset.arch) || "";
    const note = o && o.dataset.note, el = $("#model-note");   // chatllm AMD 等提醒
    if (note) { el.hidden = false; el.textContent = note; } else { el.hidden = true; el.textContent = ""; }
  }

  // ── 模型页模式：基础（用途导向）／进阶（核心＋模型全手动）──────────
  //   两模式共享同一颗加载钮与 set_model／request_load 管道，故不会状态不一致；
  //   基础模式只是「帮使用者依硬件挑好那一颗」而已。
  let _modelMode = "basic";
  function applyModelMode(mode, persist) {
    _modelMode = (mode === "advanced") ? "advanced" : "basic";
    const adv = _modelMode === "advanced";
    const bp = $("#basic-pane"), ap = $("#advanced-pane"), ax = $("#advanced-extra");
    if (bp) bp.hidden = adv;
    if (ap) ap.hidden = !adv;
    if (ax) ax.hidden = !adv;          // 设备／加速版本／诊断属进阶
    $$("#model-mode button").forEach(b => b.classList.toggle("on", b.dataset.v === _modelMode));
    if (persist) { try { API.setSettings({ modelMode: _modelMode }); } catch (e) {} }
  }
  $("#model-mode") && $("#model-mode").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    applyModelMode(b.dataset.v, true);
  });

  async function renderBasic() {
    let d;
    try { d = await API.getBasicProfiles(); } catch { return; }
    const host = $("#basic-profiles");
    if (!d || !host) return;
    const hw = $("#basic-hw");
    if (hw) {
      const t = hw.querySelector("div");
      if (t) t.textContent = `${d.hardware}　加速方式：${d.accel}`;
    }
    host.innerHTML = "";
    (d.profiles || []).forEach(p => {
      const card = document.createElement("label");
      card.className = "radio-card" + (p.selected ? " sel" : "");
      card.dataset.key = p.key;
      card.dataset.core = p.core; card.dataset.model = p.model;
      card.dataset.note = p.note || "";
      const state = p.present
        ? `<span class="chip chip-muted" style="font-weight:400">已下载</span>`
        : `<span class="chip chip-muted" style="font-weight:400">需下载</span>`;
      card.innerHTML = `<span class="rd"></span><div class="info">
        <div class="t">${escapeHtml(p.title)}　${state}</div>
        <div class="d">${escapeHtml(p.desc || "")}</div>
        <div class="d" style="margin-top:4px;font-family:var(--font-mono);font-size:11.5px">
          建议模型：${escapeHtml(p.model)}</div></div>`;
      host.appendChild(card);
    });
    basicNote((d.profiles || []).find(p => p.selected));
  }
  function basicNote(p) {
    const el = $("#basic-note"); if (!el) return;
    const txt = p && p.note;
    if (!txt) { el.hidden = true; el.textContent = ""; return; }
    el.hidden = false; el.textContent = txt;
  }
  $("#basic-profiles") && $("#basic-profiles").addEventListener("click", async e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    $$(".radio-card", $("#basic-profiles")).forEach(c => c.classList.toggle("sel", c === card));
    basicNote({ note: card.dataset.note });
    // 走与进阶模式完全相同的套用路径（会处理「需重启」等状态）
    _curCore = card.dataset.core;
    if (_modelOpt) populateModels(_curCore, card.dataset.model);
    await applyModel(card.dataset.core, card.dataset.model);
    await renderBasic();
  });

  // ── 推理加速版本（CrispASR 的 Vulkan / CUDA / CPU build）───────────
  //   后端依 WMI + nvidia-smi 检测硬件后给推荐；这里只负责呈现与回写选择。
  //   实际下载发生在下次加载模型时（换版本＝换核心，需重启）。
  async function renderAccel() {
    let a;
    try { a = await API.getAccel(); } catch { return; }
    const host = $("#accel-cards");
    if (!a || !a.ok || !host) { if (host) host.innerHTML = ""; return; }
    host.innerHTML = "";
    (a.options || []).forEach(op => {
      const card = document.createElement("label");
      const sel = op.key === a.selected;
      card.className = "radio-card" + (sel ? " sel" : "");
      card.dataset.variant = op.key;
      if (!op.suitable) { card.style.opacity = ".55"; }
      const tags = [];
      if (op.recommended) tags.push("建议");
      if (a.installed && a.installed.variant === op.key
          && a.installed.version === a.latest) tags.push("已安装");
      const tagHtml = tags.length
        ? ` <span class="chip chip-muted" style="font-weight:400">${tags.join(" · ")}</span>` : "";
      const noteHtml = op.note ? `　<span style="color:var(--warn,#b26a00)">${escapeHtml(op.note)}</span>` : "";
      card.innerHTML = `<span class="rd"></span><div class="info">
        <div class="t">${escapeHtml(op.label)}　<span style="font-weight:400;color:var(--muted)">约 ${op.size_mb} MB</span>${tagHtml}</div>
        <div class="d">${escapeHtml(op.desc || "")}${noteHtml}</div></div>`;
      host.appendChild(card);
    });
    const rb = $("#accel-reason");
    if (rb) {
      if (a.reason) {
        rb.hidden = false;
        const t = rb.querySelector("div"); if (t) t.textContent = a.reason;
      } else rb.hidden = true;
    }
    accelMsg(a.needsDownload
      ? `目前选定的核心尚未下载或版本不符（最新 ${a.latest}），会在下次加载模型时自动取得。`
      : "", "info");
  }
  function accelMsg(text, level) {
    const el = $("#accel-msg"); if (!el) return;
    if (!text) { el.hidden = true; el.innerHTML = ""; return; }
    el.hidden = false; el.style.marginTop = "10px";
    el.className = "banner " + (level === "warn" ? "banner-warn" : "banner-info");
    el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg><div>${escapeHtml(text)}</div>`;
  }
  const _accelHost = $("#accel-cards");
  if (_accelHost) _accelHost.addEventListener("click", async e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    $$(".radio-card", _accelHost).forEach(c => c.classList.toggle("sel", c === card));
    try {
      const res = await API.setAccel(card.dataset.variant);
      if (res && res.message) accelMsg(res.message, res.restartRequired ? "warn" : "info");
      await renderAccel();
    } catch (err) { accelMsg(String(err), "warn"); }
  });

  // 点核心卡 → 切核心、模型回该核心首项并套用
  $("#core-cards").addEventListener("click", async e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    _curCore = card.dataset.core;
    $$(".radio-card", $("#core-cards")).forEach(c => c.classList.toggle("sel", c === card));
    const core = coreObj(_curCore);
    const first = core && core.models[0] ? core.models[0].label : "";
    populateModels(_curCore, first);
    await applyModel(_curCore, first);
  });
  // 换模型 → 套用
  $("#model-select").addEventListener("change", async e => {
    updateArch();
    await applyModel(_curCore, e.target.value);
  });
  let _modelLoading = false;       // 模型「就地下载并加载」进行中（进度导到模型页）
  async function applyModel(core, model) {
    try {
      const res = await API.setModel(core, model);
      if (res && res.message) modelMsg(res.message, res.restartRequired ? "warn" : "info");
      else modelMsg("", null);
      updateLoadBtn(res);
    } catch (err) { modelMsg("套用失败：" + (err.message || err), "warn"); }
  }
  // 依 setModel 结果 + 目前加载状态，决定「下载并加载」按钮的文字/可用性
  async function updateLoadBtn(res) {
    const btn = $("#btn-model-load"); if (!btn) return;
    let st = {};
    try { st = await API.getStatus(); } catch (e) {}
    btn.classList.remove("btn-ghost");
    if (_modelLoading || st.loading) {
      btn.disabled = true; btn.textContent = T("model.loading", "下载／加载中…"); return;
    }
    if (st.modelReady && !(res && res.restartRequired)) {
      // 已就绪且未要求重启 → 引导前往转录
      btn.disabled = false; btn.textContent = T("model.goto", "前往语音转文字"); btn.dataset.act = "goto"; return;
    }
    if (res && res.restartRequired) {
      // 已加载其他核心、切换需重启 → 按钮反色（不可就地加载）
      btn.disabled = true; btn.textContent = T("model.needRestart", "切换核心需重新启动"); btn.dataset.act = ""; return;
    }
    // 尚未加载 → 可就地下载并加载
    btn.disabled = false; btn.textContent = T("model.load", "下载并加载模型"); btn.dataset.act = "load";
  }
  $("#btn-model-load") && $("#btn-model-load").addEventListener("click", async () => {
    const btn = $("#btn-model-load");
    if (btn.dataset.act === "goto") { switchView("file"); return; }
    _modelLoading = true; btn.disabled = true; btn.textContent = "下载／加载中…";
    $("#model-progress").hidden = false;
    setModelProg(0, "准备加载…");
    modelMsg("正在下载并加载模型，请稍候——可留在本页观看进度。", "info");
    try { await API.startLoad(); }
    catch (err) { _modelLoading = false; modelMsg("加载失败：" + (err.message || err), "warn"); updateLoadBtn(); }
  });
  function setModelProg(pct, status) {
    $("#model-prog-bar").style.width = pct + "%";
    $("#model-prog-pct").textContent = Math.round(pct) + "%";
    if (status != null) $("#model-prog-status").textContent = status;
  }
  function modelMsg(text, level) {
    const el = $("#model-msg");
    if (!text) { el.hidden = true; el.innerHTML = ""; return; }
    el.hidden = false; el.style.marginTop = "10px";
    el.className = "banner " + (level === "warn" ? "banner-warn" : "banner-info");
    el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg><div>${escapeHtml(text)}</div>`;
  }
  const ICON_CPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>`;
  const ICON_GPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 18v2M18 18v2"/></svg>`;

  // ── 识别语言下拉（依已加载引擎；OpenVINO 用 processor 清单）────────
  async function loadLanguages() {
    try {
      const { languages } = await API.getLanguages();
      // 音频页与录制页各有自己的语言下拉，皆依目前引擎填同一份清单。
      ["#sel-lang", "#rec-lang"].forEach(id => {
        const sel = $(id); if (!sel) return;
        const cur = sel.value;
        sel.innerHTML = "";
        languages.forEach(l => {
          const o = document.createElement("option");
          o.value = l.value; o.textContent = l.value ? l.label : "语言 自动";
          sel.appendChild(o);
        });
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
      });
    } catch (e) {}
  }

  // ════════════════════════════════════════════════════════
  // 设置
  // ════════════════════════════════════════════════════════
  async function loadSettings() {
    const s = await API.getSettings();
    $("#set-scale").value = String(s.scale);
    applyScale(s.scale);
    segSet("#set-format", s.format); segSet("#set-theme", s.theme || "light");
    applyTheme(s.theme || "light");                 // 启动即套用深/浅色
    $("#set-mirror").value = s.mirror || "";
    $("#set-ffmpeg").value = s.ffmpeg || "";
    $("#set-model-dir").value = s.modelDir || "";
    if (s.vad != null) { $("#set-vad").value = s.vad; $("#set-vad-val").textContent = (+s.vad).toFixed(2); }
    // 每段最长秒数：0/未设 → 视为上限（slider 显示 30，后端依模型自动压低）
    const cs = (+s.chunkSecs > 0) ? +s.chunkSecs : 30;
    if ($("#set-chunk")) { $("#set-chunk").value = cs; $("#set-chunk-val").textContent = cs + "s"; }
    // 界面语言：菜单回填 + 套用 i18n（含目前视图标题）
    const uiLang = s.uiLang || "简体中文";
    if ([...$("#set-lang").options].some(o => o.value === uiLang)) $("#set-lang").value = uiLang;
    if (window.I18N) { I18N.setLang(uiLang); refreshViewTitle(); }
    applyModelMode(s.modelMode || "basic", false);   // 模型页默认「基础」
  }
  function segSet(sel, v) { $$(sel + " button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
  // 界面缩放：用 CSS zoom（Chromium/WebView2 支持）整体缩放，px 版面也能等比生效。
  function applyScale(pct) { document.body.style.zoom = (Math.max(50, Math.min(200, +pct || 100)) / 100); }

  $("#set-scale").addEventListener("change", e => {
    const v = +e.target.value; applyScale(v); API.setSettings({ scale: v });
  });
  $("#set-vad").addEventListener("input", e => {
    const v = +e.target.value; $("#set-vad-val").textContent = v.toFixed(2); API.setSettings({ vad: v });
  });
  $("#set-chunk") && $("#set-chunk").addEventListener("input", e => {
    const v = +e.target.value; $("#set-chunk-val").textContent = v + "s"; API.setSettings({ chunkSecs: v });
  });
  // segmented 控制：点击切换 + 回写设置
  [["#set-format", "format"], ["#set-theme", "theme"]].forEach(([sel, key]) => {
    $(sel).addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      $$(sel + " button").forEach(x => x.classList.toggle("on", x === b));
      API.setSettings({ [key]: b.dataset.v });
      if (key === "theme") applyTheme(b.dataset.v);
    });
  });
  $("#set-mirror").addEventListener("change", e => API.setSettings({ mirror: e.target.value.trim() }));
  // 外观主题：浅/深/跟随系统。system 解析成实际 light/dark 挂到 <html>，并监听 OS 变化。
  // 窗口标题栏深浅由后端（app_webview）依同一设置同步，故只需把偏好写回 setSettings。
  let _mqHandler = null;
  function resolveTheme(t) {
    if (t === "dark" || t === "light") return t;
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
  }
  function applyTheme(t) {
    document.documentElement.dataset.theme = resolveTheme(t);   // 实际 light/dark
    document.documentElement.dataset.themePref = t;             // 原始偏好
    if (window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (_mqHandler) { try { mq.removeEventListener("change", _mqHandler); } catch (e) {} _mqHandler = null; }
      if (t === "system") {
        _mqHandler = () => { document.documentElement.dataset.theme = resolveTheme("system"); };
        try { mq.addEventListener("change", _mqHandler); } catch (e) {}
      }
    }
  }
  $("#set-ffmpeg").addEventListener("change", e => API.setSettings({ ffmpeg: e.target.value.trim() }));
  // FFmpeg 路径：「浏览…」原生选择文件对话框（与模型下载位置同款交互）
  $("#btn-ffmpeg-browse") && $("#btn-ffmpeg-browse").addEventListener("click", async () => {
    const res = await API.browseFile($("#set-ffmpeg").value.trim());
    if (!res || !res.ok) return;                       // 用户取消 / 不支持
    $("#set-ffmpeg").value = res.path;
    API.setSettings({ ffmpeg: res.path.trim() });
  });
  // 模型下载位置：手输或「浏览…」原生对话框；保存后若目录变更 → 提示重启
  $("#set-model-dir").addEventListener("change", async e => {
    const res = await API.setSettings({ modelDir: e.target.value.trim() });
    if (res && res.modelDirChanged) modelMsg(res.message, "info");
    $("#set-model-dir").value = res.modelDir || "";
  });
  $("#btn-model-dir-browse") && $("#btn-model-dir-browse").addEventListener("click", async () => {
    const res = await API.browseFolder($("#set-model-dir").value.trim());
    if (!res || !res.ok) return;                       // 用户取消 / 不支持
    const saved = await API.setSettings({ modelDir: res.path });
    $("#set-model-dir").value = saved.modelDir || "";
    if (saved.modelDirChanged) modelMsg(saved.message, "info");
  });
  // 界面语言切换：写回设置 + 实时套用 i18n（含目前视图标题、模型页动态文字）
  $("#set-lang").addEventListener("change", e => {
    const v = e.target.value;
    API.setSettings({ uiLang: v });
    if (window.I18N) { I18N.setLang(v); refreshViewTitle(); }
    if (_curView === "model") renderModel();   // 重绘动态文字（核心卡描述等）
  });
  // 检查更新：开启 GitHub Releases 页（系统浏览器）
  $("#btn-check-update") && $("#btn-check-update").addEventListener("click", async () => {
    const btn = $("#btn-check-update"); const o = btn.textContent;
    btn.disabled = true; btn.textContent = "开启中…";
    try {
      const res = await API.checkUpdate();
      if (res && res.ok === false) flash(btn, "未配置更新链接");
    } catch (e) {}
    setTimeout(() => { btn.disabled = false; btn.textContent = o; }, 1500);
  });

  // ── 共享工具 ────────────────────────────────────────────
  function fmtClock(sec) {
    sec = Math.max(0, Math.round(sec));
    const m = String(Math.floor(sec / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${m}:${s}`;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  // ── 录制转换：MediaRecorder + 停顿检测(VAD)，分段上传识别 ──────
  //   127.0.0.1/localhost 属安全情境，getUserMedia 可用（不需 HTTPS）。
  //   说完停顿 → 切段 → 上传 /api/transcribe → 逐段附加实时字幕。
  const REC = {
    on: false, sec: 0, timer: null, raf: 0, firstResult: true,
    stream: null, ctx: null, analyser: null, recorder: null, chunks: [],
    speech: false, silentSince: 0, segStart: 0,
    segs: [],            // 累积识别结果（含合成时间轴），供清除／存档／实时存档
    fileHandle: null,    // 实时存档的 File System Access 文件 handle（若支持）
  };
  const SILENCE_MS = 2200, MIN_SEG_MS = 500, MAX_SEG_MS = 20000, VAD_THRESH = 0.014;
  const REC_LINE_SECS = 5;     // 录制无精确时间戳 → 每段合成固定 5s（比照 app.py）

  $("#rec-btn").addEventListener("click", () => REC.on ? stopRec() : startRec());

  function recSupported() {
    return navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder;
  }

  // ── 麦克风权限状态（整合进页面，而非仅依赖浏览器弹窗）──────────────
  //   浏览器原生「允许麦克风」提示由 WebView2/Edge 控制、无法被页面取代；
  //   但我们用 Permissions API 把「已授权／将询问／被拒」状态与指引显示在页内，
  //   被拒时给明确的页内排除说明，不再只是一行错误。
  async function refreshMicPerm() {
    const chip = $("#rec-perm"), help = $("#rec-perm-help");
    if (!chip) return;
    if (!recSupported()) {
      chip.hidden = false; chip.className = "chip chip-muted"; chip.textContent = "不支持录音";
      return;
    }
    let state = "prompt";
    try {
      if (navigator.permissions && navigator.permissions.query) {
        const st = await navigator.permissions.query({ name: "microphone" });
        state = st.state;                       // granted / denied / prompt
        st.onchange = () => refreshMicPerm();    // 状态变更实时更新
      }
    } catch (e) { state = "prompt"; }
    chip.hidden = false;
    if (state === "granted") {
      chip.className = "chip chip-ok"; chip.textContent = T("record.permGranted", "麦克风已授权");
      if (help) help.hidden = true;
    } else if (state === "denied") {
      chip.className = "chip chip-live"; chip.textContent = T("record.permDeniedShort", "麦克风被拒");
      if (help) help.hidden = false;
    } else {
      chip.className = "chip chip-muted"; chip.textContent = T("record.permPrompt", "按麦克风时将请求授权");
      if (help) help.hidden = true;
    }
  }

  // ── 麦克风设备列举（浏览器原生；对应 app.py 的 sounddevice 设备选择）──────
  //   设备标签在未授权前为空字符串 → 首次 getUserMedia 后再列举一次才有名称。
  async function enumerateMics() {
    const sel = $("#rec-mic");
    if (!sel || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const cur = sel.value;
      const devs = await navigator.mediaDevices.enumerateDevices();
      const mics = devs.filter(d => d.kind === "audioinput");
      sel.innerHTML = "";
      const def = document.createElement("option");
      def.value = ""; def.textContent = "默认麦克风";
      sel.appendChild(def);
      mics.forEach((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId;
        o.textContent = d.label || `麦克风 ${i + 1}`;
        sel.appendChild(o);
      });
      if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
    } catch (e) { /* 列举失败 → 保留默认选项 */ }
  }
  $("#rec-mic-refresh") && $("#rec-mic-refresh").addEventListener("click", enumerateMics);
  if (navigator.mediaDevices) {
    try { navigator.mediaDevices.addEventListener("devicechange", enumerateMics); } catch (e) {}
  }

  async function startRec() {
    if (!recSupported()) { recNote("此环境不支持录音 API。", true); return; }
    // 依选定的麦克风建立音讯约束（空值 → 系统默认设备）。
    const micId = $("#rec-mic") ? $("#rec-mic").value : "";
    const audioConstraint = { echoCancellation: true, noiseSuppression: true };
    if (micId) audioConstraint.deviceId = { exact: micId };
    try {
      REC.stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
    } catch (err) {
      // 被拒 / 无设备 → 页核显示权限指引，不只丢一行错误
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      recNote((denied ? "麦克风权限被拒。" : "无法取得麦克风：") + (err.message || err), true);
      refreshMicPerm();
      if (denied && $("#rec-perm-help")) $("#rec-perm-help").hidden = false;
      return;
    }
    enumerateMics();    // 授权后重新列举 → 取得真实设备名称
    refreshMicPerm();
    REC.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = REC.ctx.createMediaStreamSource(REC.stream);
    REC.analyser = REC.ctx.createAnalyser(); REC.analyser.fftSize = 1024;
    src.connect(REC.analyser);
    REC.on = true; REC.sec = 0;
    $("#rec-btn").classList.add("recording");
    const lw = $("#rec-live-wave"); lw.hidden = false;
    lw.innerHTML = Array.from({ length: 28 }, () => `<div class="b" style="height:4px"></div>`).join("");
    $("#rec-timer").textContent = "00:00";
    REC.timer = setInterval(() => { REC.sec++; $("#rec-timer").textContent = fmtClock(REC.sec); }, 1000);
    recNote("聆听中…说完停顿约 2 秒会自动识别；再按一次结束并完成最后一段。");
    startSeg(); monitor();
  }
  function startSeg() {
    REC.chunks = []; REC.speech = false; REC.silentSince = 0; REC.segStart = Date.now();
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
      .find(m => MediaRecorder.isTypeSupported(m)) || "";
    REC.recorder = mime ? new MediaRecorder(REC.stream, { mimeType: mime })
                        : new MediaRecorder(REC.stream);
    REC.recorder.ondataavailable = e => { if (e.data && e.data.size) REC.chunks.push(e.data); };
    REC.recorder.onstop = onSegStop;
    REC.recorder.start();
  }
  function cutSeg() { if (REC.recorder && REC.recorder.state === "recording") REC.recorder.stop(); }
  async function onSegStop() {
    const dur = Date.now() - REC.segStart;
    const blob = new Blob(REC.chunks, { type: REC.chunks[0] ? REC.chunks[0].type : "audio/webm" });
    const ok = REC.speech && dur >= MIN_SEG_MS && blob.size > 1200;
    if (REC.on) startSeg(); else teardownRec();    // 先续录，避免上传期间漏话
    if (ok) {
      try {
        const file = new File([blob], "recording.webm", { type: blob.type });
        const recLang = $("#rec-lang") ? $("#rec-lang").value : "";
        const res = await API.transcribe({ file, language: recLang || null, diarize: false, align: false });
        (res.segments || []).forEach(s => { if (s.text && s.text.trim()) appendRecLine(s.text.trim()); });
      } catch (err) { recNote("识别失败：" + (err.message || err), true); }
    }
  }
  function monitor() {
    const buf = new Uint8Array(REC.analyser.fftSize);
    const bars = $$(".b", $("#rec-live-wave"));
    const tick = () => {
      if (!REC.on) return;
      REC.analyser.getByteTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / buf.length);
      const base = Math.min(34, rms * 260);
      bars.forEach(b => b.style.height = Math.max(4, base * (0.5 + Math.random() * 0.5)).toFixed(0) + "px");
      const now = Date.now();
      if (rms >= VAD_THRESH) { REC.speech = true; REC.silentSince = 0; }
      else if (REC.speech) {
        if (!REC.silentSince) REC.silentSince = now;
        else if (now - REC.silentSince >= SILENCE_MS) cutSeg();   // 停顿 → 切段
      }
      if (Date.now() - REC.segStart >= MAX_SEG_MS && REC.speech) cutSeg();
      REC.raf = requestAnimationFrame(tick);
    };
    REC.raf = requestAnimationFrame(tick);
  }
  function stopRec() {
    REC.on = false; $("#rec-btn").classList.remove("recording");
    if (REC.raf) cancelAnimationFrame(REC.raf);
    clearInterval(REC.timer);
    $("#rec-live-wave").hidden = true;
    recNote("已停止。");
    cutSeg();    // 收尾段（onSegStop 会上传并 teardown）
  }
  function teardownRec() {
    try { REC.stream && REC.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { REC.ctx && REC.ctx.close(); } catch (e) {}
    REC.stream = REC.ctx = REC.analyser = REC.recorder = null;
  }
  function recNote(msg, warn) {
    const el = $(".rec-note");
    if (el) { el.textContent = msg; el.style.color = warn ? "var(--live)" : "var(--muted)"; }
  }
  function appendRecLine(text) {
    const host = $("#rec-subs");
    if (REC.firstResult) { host.innerHTML = ""; REC.firstResult = false; }
    // 合成时间轴：录制无精确时间戳，每段固定 5s、段间 0.1s（比照 app.py _on_rt_save）。
    const start = REC.segs.length ? REC.segs[REC.segs.length - 1].end + 0.1 : 0;
    const seg = { start, end: start + REC_LINE_SECS, speaker: null, text };
    REC.segs.push(seg);
    const el = document.createElement("div");
    el.className = "sub-card";
    el.innerHTML = `<span class="tc">${fmtClock(REC.sec)}</span>
      <div class="body"><div class="txt">${escapeHtml(text)}</div></div>`;
    host.appendChild(el);
    host.scrollTop = host.scrollHeight;
    $("#rec-save").disabled = false;
    if (REC.fileHandle) recAutosaveWrite();    // 实时存档：每段附加后即落盘
  }

  // ── 清除／存储／实时存档（对应 app.py 录制页的清除、存储 SRT、实时追加保存）──
  function recClear() {
    REC.segs = []; REC.firstResult = true;
    $("#rec-subs").innerHTML = `<div class="sub-card"><span class="tc">— · —</span>`
      + `<div class="body"><div class="txt" style="color:var(--muted)">开始录音后，识别结果会逐段出现在这里。</div></div></div>`;
    $("#rec-save").disabled = true;
  }
  async function recSave() {
    if (!REC.segs.length) return;
    let fmt = "srt";
    try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    const text = fmt === "txt" ? buildTxt(REC.segs) : buildSrt(REC.segs);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "").slice(0, 14);
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `realtime_${stamp}` + (fmt === "txt" ? ".txt" : ".srt");
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  $("#rec-clear") && $("#rec-clear").addEventListener("click", recClear);
  $("#rec-save") && $("#rec-save").addEventListener("click", recSave);

  // 实时存档：用 File System Access API（WebView2/Edge 支持）持续把目前累积的
  // 字幕写入使用者选定的文件 —— 重现 app.py「实时追加保存，可随时中断不遗失」。
  // 不支持的环境隐藏此开关（仍可用「存储字幕」一次性导出）。
  const FS_SAVE_OK = typeof window.showSaveFilePicker === "function";
  if (FS_SAVE_OK) $("#rec-autosave-wrap").hidden = false;
  $("#rec-autosave") && $("#rec-autosave").addEventListener("change", async e => {
    if (!e.target.checked) { REC.fileHandle = null; return; }
    try {
      let fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (err) {}
      const ext = fmt === "txt" ? "txt" : "srt";
      REC.fileHandle = await window.showSaveFilePicker({
        suggestedName: `realtime.${ext}`,
        types: [{ description: "字幕档", accept: { "text/plain": ["." + ext] } }],
      });
      await recAutosaveWrite();           // 立即写一次（含既有片段）
      recNote("实时存档已启用：每段识别完成后会自动写入所选文件。");
    } catch (err) {                         // 使用者取消选档 → 还原开关
      REC.fileHandle = null; e.target.checked = false;
    }
  });
  async function recAutosaveWrite() {
    if (!REC.fileHandle) return;
    try {
      let fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
      const text = fmt === "txt" ? buildTxt(REC.segs) : buildSrt(REC.segs);
      const w = await REC.fileHandle.createWritable();   // 截断重写（内容小，安全可靠）
      await w.write(text); await w.close();
    } catch (e) { recNote("实时存档写入失败：" + (e.message || e), true); REC.fileHandle = null; }
  }

  // ── 启动 ────────────────────────────────────────────────
  // 加载完成 → 更新就绪灯 + 语言清单（依引擎）；若为「就地下载并加载」流程 → 收尾
  API.on("status", async (s) => {
    refreshStatus(); loadLanguages();
    let ready = s && s.modelReady;
    if (ready == null) { try { ready = (await API.getStatus()).modelReady; } catch (e) {} }
    if (_modelLoading && ready) {
      _modelLoading = false;
      setModelProg(100, "完成");
      setTimeout(() => { $("#model-progress").hidden = true; }, 800);
      modelMsg("模型已就绪。已为你切换到「语音转文字」。", "info");
      renderHealth();                 // 自检色点刷新
      switchView("file");             // 完成 → 进入转录页
    } else if (s && s.error && _modelLoading) {
      _modelLoading = false;
      $("#model-progress").hidden = true;
      modelMsg("加载失败：" + s.error, "warn");
      updateLoadBtn();
    }
  });
  (async function init() {
    await API.ready;
    await refreshStatus();
    await loadLanguages();
    await loadSettings();
    // 开始页决策：目前选择的模型已就绪 → 直接进「语音转文字」；否则停在「模型」页，
    // 让使用者先确认硬件＋选择模型（可改 Whisper 等），按「下载并加载」才下载。
    let ready = false;
    try { ready = !!(await API.getStatus()).selectedReady; } catch (e) {}
    switchView(ready ? "file" : "model");
    console.info("[app] ready, mode =", API.mode, "selectedReady =", ready);
  })();
})();

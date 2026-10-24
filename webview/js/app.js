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
    if (name === "file") renderHistory();    // 最近转录（subtitles/ 产出回填）
  }
  $("#nav").addEventListener("click", e => {
    const btn = e.target.closest(".nav-item");
    if (btn) switchView(btn.dataset.view);
  });
  // 左下角状态点 → 直达「模型」页（状态与模型下载/热切换直接相关）
  $("#model-status").addEventListener("click", () => switchView("model"));

  // ── 状态栏 ──────────────────────────────────────────────
  // 状态点状态机（与「模型页按需下载 / 热切换」语义对齐）：
  //   loading           加载／下载中（琥珀）
  //   未就绪 + 未下载   模型未下载（灰）—— 引导到模型页
  //   未就绪 + 已下载   尚未加载模型（灰）
  //   就绪 + 选择不同   待切换（琥珀）—— 此刻转录跑的仍是旧模型
  //   就绪 + 有 error   已回退旧引擎（琥珀）—— tooltip 带原因
  //   就绪              模型已就绪（绿）—— tooltip 显示实际加载的模型
  // 点击状态点 → 直达「模型」页。
  async function refreshStatus() {
    const s = await API.getStatus();
    const el = $("#model-status");
    let key = "status.ready", def = "模型已就绪", cls = "";
    if (s.loading) {
      key = "status.loading"; def = "加载模型中…"; cls = "loading";
    } else if (!s.modelReady) {
      cls = "off";
      if (s.selectedReady === false) { key = "status.notDownloaded"; def = "模型未下载"; }
      else { key = "status.needModel"; def = "尚未加载模型"; }
    } else if (s.selectionDiffers) {
      key = "status.switchPending"; def = "待切换"; cls = "loading";
    } else if (s.error) {
      cls = "loading";    // 加载失败已回退旧引擎：仍可用，但以琥珀色提醒
    }
    el.classList.remove("loading", "off");
    if (cls) el.classList.add(cls);
    $(".t", el).textContent = T(key, def);
    el.title = statusTip(s);
    // 版本徽章：纯数字版本前缀 v（如 v1.0.9）；已含文字者（如 webview 0.1）原样显示
    if (s.version) $("#app-version").textContent = /^\d/.test(s.version) ? "v" + s.version : s.version;
  }
  function statusTip(s) {
    if (s.loading) return T("status.tipLoading", "正在下载／加载模型…");
    if (!s.modelReady) {
      return s.selectedReady === false
        ? T("status.tipNotDownloaded", "所选模型尚未下载——到「模型」页点「下载并加载模型」")
        : T("status.tipNotLoaded", "模型文件已备——到「模型」页点「下载并加载模型」");
    }
    if (s.selectionDiffers) {
      return T("status.tipSwitch", "已加载：{loaded}\n已选择：{sel}\n到「模型」页点「下载并加载模型」完成切换",
               { loaded: s.activeModel || "—", sel: s.selectedModel || "—" });
    }
    if (s.error) return T("status.tipFallback", "本次加载失败，已回退旧模型继续使用：{e}", { e: s.error });
    return T("status.tipReady", "当前模型：{m}", { m: s.activeModel || "—" });
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
      <div class="big">${T("file.dropBig", "拖曳音讯到此，或<b>点击选择文件</b>")}</div>
      <div class="sub">${T("file.dropSub", "支持 mp3 / wav / m4a / 视频音轨")}</div>`;
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
    running = true; _switchBusy = true;    // 转录中 → 模型页切换禁用
    const btn = $("#btn-run");
    btn.disabled = true; btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>${T("file.running", "识别中…")}`;
    $("#result").hidden = true;
    showProgress(0, T("file.preparing", "准备中…"));
    logLine("▶ " + T("file.logStart", "开始转换：") + picked.name);
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
      logLine("✓ " + T("file.logDone", "完成，共 {n} 段", { n: res.segments.length })
        + (res.srtPath ? T("file.logSaved", "，已输出 ") + res.srtPath : ""));
      $("#btn-open-dir").disabled = false;
      $("#btn-save-sub").disabled = false;
      renderHistory();
    } catch (err) {
      showProgress(0, "");
      $("#progress").hidden = true;
      logLine("✕ " + (err.message || err));
    } finally {
      running = false; _switchBusy = false;
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>${T("file.run", "开始转换")}`;
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
  // 说话者自定义名称：{编号: 名字}。新一轮转录/载入历史时重置（不同音频不同人）；
  // 未命中的编号回落默认「说话者N」。界面与导出（SRT/TXT/ASS/LRC）同步使用。
  let speakerNames = {};

  // 说话者显示名（chip 用，默认带空格）与导出名（与后端写出格式一致，无空格）
  function spkDisplay(n) { return speakerNames[n] || ("说话者 " + n); }
  function spkLabel(n) { return speakerNames[n] || ("说话者" + n); }

  // ── 段落编辑：拆分（编辑态光标处）与合并（相邻同说话者选择模式）────────
  let _mergeMode = false;          // 合并选择模式中（点卡片＝选择而非跳播）
  let _mergeSel = new Set();       // 已选段索引
  // 播放高亮用的元素缓存与位置提示（长音频数千段时避免每帧 O(n) DOM 查询）
  let _subEls = [], _blockEls = [], _playingIdx = -1, _playedBars = -1, _segHint = 0;
  let selRange = null;   // 当前选中片段 {start, end, idx}：点字幕行/区块设定，播放键循环

  // 定位 t 所在段：先看上次位置附近（播放连续），未命中则二分。
  // 返回段索引；间隙（无段覆盖）回 -1。
  function segIndexAt(t) {
    const n = curSegs.length;
    if (!n) return -1;
    for (const i of [_segHint, _segHint - 1, _segHint + 1]) {
      if (i >= 0 && i < n && t >= curSegs[i].start && t < curSegs[i].end) {
        _segHint = i; return i;
      }
    }
    let lo = 0, hi = n - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1, s = curSegs[mid];
      if (t < s.start) hi = mid - 1;
      else if (t >= s.end) lo = mid + 1;
      else { _segHint = mid; return mid; }
    }
    return -1;
  }

  async function renderResult(segments) {
    $("#result").hidden = false;
    curSegs = segments;
    speakerNames = {};                 // 新结果 → 说话者名称重置
    setMergeMode(false);               // 新结果 → 退出合并选择模式
    _karaCache = null;                      // 新结果 → 重建卡拉OK字级（段落索引已变）
    _subEls = []; _blockEls = []; _playingIdx = -1; _playedBars = -1; _segHint = 0;
    selRange = null;                     // 新结果 → 片段选中态重置
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
    renderSpeakerBar();
    wireAudio();
    syncStickyOffset();
  }

  // ── 说话者标签：结果区标签栏 + 点击改名（标签栏 chip 与字幕卡 chip 皆可）──
  function renderSpeakerBar() {
    const bar = $("#speaker-bar");
    if (!bar) return;
    const nums = [];
    curSegs.forEach(s => { if (s.speaker && !nums.includes(s.speaker)) nums.push(s.speaker); });
    if (!nums.length) { bar.hidden = true; bar.innerHTML = ""; return; }
    bar.hidden = false;
    const hint = T("file.spkRenameHint", "点击标签可重命名");
    bar.innerHTML = `<span class="lbl">${T("file.speakers", "说话者")}</span>`
      + `<span class="lbl-hint">${hint}</span>`;
    nums.sort((a, b) => a - b).forEach(n => {
      const chip = document.createElement("span");
      chip.className = "chip spk-" + (((n - 1) % 3) + 1);
      chip.textContent = spkDisplay(n);
      chip.title = hint;
      chip.addEventListener("click", e => { e.stopPropagation(); beginSpkRename(n, chip); });
      bar.appendChild(chip);
    });
  }

  // 就地改名（contentEditable，与字幕行编辑同一交互语言：Enter 存 / Esc 取消）。
  // 存入 speakerNames 后重渲染全部说话者视图——字幕卡、时间轴区块、标签栏。
  function beginSpkRename(n, chip) {
    if (chip.classList.contains("editing")) return;
    chip.classList.add("editing");
    const orig = spkDisplay(n);
    chip.contentEditable = "true"; chip.spellcheck = false;
    chip.focus();
    const range = document.createRange(); range.selectNodeContents(chip);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    function finish(save) {
      chip.removeEventListener("keydown", onKey);
      chip.removeEventListener("blur", onBlur);
      chip.contentEditable = "false";
      chip.classList.remove("editing");
      const val = (chip.textContent || "").trim();
      if (save && val && val !== orig) {
        speakerNames[n] = val;
        _playingIdx = -1;                     // 重渲染后高亮状态重建
        renderSpeakerBar();
        renderSegStrip(curSegs, waveDur);
        renderSubs(curSegs);
        logLine("✓ " + T("file.spkRenamed", "说话者已重命名：{old} → {new}（各段落与导出同步）",
                         { old: orig, new: val }));
      } else {
        chip.textContent = orig;               // 取消／清空 → 还原
      }
      chip.blur();
    }
    function onKey(e) {
      if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    }
    function onBlur() { finish(true); }
    chip.addEventListener("keydown", onKey);
    chip.addEventListener("blur", onBlur);
  }

  // ── 单行说话者切换：点字幕卡 chip 弹菜单 ────────────────────────────
  // 列出现有说话者（当前项打勾）＋「新说话者」（分离并错一人时的手工补救）
  // ＋「重命名…」（保留旧 chip 点击的改名入口）。切换只改这一行的
  // s.speaker；导出前缀、合并判定（同说话者相邻才可并）、波形区块
  // 颜色、标签栏都随 curSegs 自动同步。
  function spkSwitchMenu(idx, chipEl) {
    closeSpkMenu();                                  // 同时最多一个
    const cur = curSegs[idx].speaker;
    const nums = [];
    curSegs.forEach(s => { if (s.speaker && !nums.includes(s.speaker)) nums.push(s.speaker); });
    nums.sort((a, b) => a - b);

    const menu = document.createElement("div");
    menu.className = "spk-menu";
    nums.forEach(n => {
      const row = document.createElement("button");
      row.type = "button";
      if (n === cur) row.classList.add("cur");
      row.innerHTML = `<span class="dot spk-${((n - 1) % 3) + 1}"></span>`
        + `<span class="nm">${escapeHtml(spkDisplay(n))}</span>`
        + (n === cur ? `<span class="tick" title="${T("file.spkCur", "当前")}">✓</span>` : "");
      row.addEventListener("click", e => {
        e.stopPropagation();
        closeSpkMenu();
        if (n !== cur) switchSegSpeaker(idx, n);
      });
      menu.appendChild(row);
    });
    const sep = document.createElement("div"); sep.className = "sep"; menu.appendChild(sep);

    const newRow = document.createElement("button");
    newRow.type = "button"; newRow.className = "add";
    newRow.textContent = T("file.spkNew", "＋ 新说话者");
    newRow.addEventListener("click", e => {
      e.stopPropagation();
      closeSpkMenu();
      switchSegSpeaker(idx, (nums[nums.length - 1] || 0) + 1);
    });
    menu.appendChild(newRow);

    const renRow = document.createElement("button");
    renRow.type = "button"; renRow.className = "ren";
    renRow.textContent = T("file.spkRenameMenu", "重命名说话者…");
    renRow.addEventListener("click", e => {
      e.stopPropagation();
      closeSpkMenu();
      beginSpkRename(cur, chipEl);                   // 旧入口保留：改名仍一步直达
    });
    menu.appendChild(renRow);

    // 定位：chip 屏幕坐标（fixed），右侧越界时向左收
    document.body.appendChild(menu);
    const r = chipEl.getBoundingClientRect();
    const mw = menu.offsetWidth, mh = menu.offsetHeight;
    let x = Math.min(r.left, window.innerWidth - mw - 8);
    let y = r.bottom + 6;
    if (y + mh > window.innerHeight - 8) y = Math.max(8, r.top - mh - 6);
    menu.style.left = x + "px"; menu.style.top = y + "px";

    document.addEventListener("pointerdown", onSpkMenuAway, true);
    document.addEventListener("keydown", onSpkMenuKey, true);
    window.addEventListener("scroll", closeSpkMenu, true);
  }

  function onSpkMenuAway(e) { if (!e.target.closest(".spk-menu")) closeSpkMenu(); }
  function onSpkMenuKey(e) { if (e.key === "Escape") closeSpkMenu(); }

  function closeSpkMenu() {
    document.querySelectorAll(".spk-menu").forEach(m => m.remove());
    document.removeEventListener("pointerdown", onSpkMenuAway, true);
    document.removeEventListener("keydown", onSpkMenuKey, true);
    window.removeEventListener("scroll", closeSpkMenu, true);
  }

  function switchSegSpeaker(idx, n) {
    const old = spkDisplay(curSegs[idx].speaker);
    curSegs[idx].speaker = n;
    _playingIdx = -1;                                 // 重渲染后高亮状态重建
    renderSpeakerBar();
    renderSegStrip(curSegs, waveDur);
    renderSubs(curSegs);
    logLine("✓ " + T("file.spkSwitched", "第 {i} 行说话者：{old} → {new}（导出与合并判定同步）",
                     { i: idx + 1, old, new: spkDisplay(n) }));
  }

  // ── 拆分：编辑态光标处一分为二 ──────────────────────────────────────
  // contentEditable 内的光标字符偏移（以元素纯文本为坐标系）。
  // anchor="start" 且选中了一段文字时取选区起点（「高亮到哪就切到哪」）。
  function caretOffsetIn(el, anchor) {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    if (!el.contains(range.endContainer)) return null;
    const node = anchor === "start" && !range.collapsed ? range.startContainer : range.endContainer;
    const off  = anchor === "start" && !range.collapsed ? range.startOffset : range.endOffset;
    const pre = range.cloneRange();
    pre.selectNodeContents(el);
    pre.setEnd(node, off);
    return pre.toString().length;
  }

  // 拆分一段；成功返回 true，拆分点无效（null/首/尾）提示后返回 false
  function splitSegment(idx, full, pos) {
    if (pos == null) { logLine("⚠ " + T("edit.splitNeedCaret", "请先点击文本放置光标，再拆分")); return false; }
    const seg = curSegs[idx];
    if (!seg) return false;
    const L = full.slice(0, pos).trim(), R = full.slice(pos).trim();
    if (!L || !R) { logLine("⚠ " + T("edit.splitEmpty", "拆分点两侧都需要有文字")); return false; }
    // 切点时间：无字级 → 按字符比例；有字级 → 比例值吸附到最近的字边界
    let t = seg.start + (seg.end - seg.start) * (pos / Math.max(1, full.length));
    const words = seg.words || [];
    if (words.length) {
      let best = t, bestD = Infinity;
      for (const w of words) for (const b of [w.start, w.end]) {
        const d = Math.abs(b - t);
        if (d < bestD) { bestD = d; best = b; }
      }
      t = Math.max(seg.start + 0.05, Math.min(seg.end - 0.05, best));
    }
    curSegs.splice(idx, 1,
      { start: seg.start, end: t, speaker: seg.speaker, text: L,
        words: words.filter(w => (w.start + w.end) / 2 < t) },
      { start: t, end: seg.end, speaker: seg.speaker, text: R,
        words: words.filter(w => (w.start + w.end) / 2 >= t) });
    rerenderAfterEdit();
    logLine("✓ " + T("edit.splitDone", "已拆分为 2 段"));
    return true;
  }

  // ── 合并：相邻同说话者的多段 ────────────────────────────────────────
  function toggleMergeSel(idx, cardEl) {
    if (_mergeSel.has(idx)) _mergeSel.delete(idx);
    else _mergeSel.add(idx);
    cardEl.classList.toggle("msel", _mergeSel.has(idx));
    updateMergeBar();
  }

  function updateMergeBar() {
    const bar = $("#merge-bar"), n = _mergeSel.size;
    if (!bar) return;
    $("#merge-count").textContent = T("merge.selCount", "已选 {n} 段 · 需相邻且同一说话者", { n });
    $("#merge-ok").disabled = n < 2;
  }

  function setMergeMode(on) {
    _mergeMode = on;
    _mergeSel = new Set();
    const bar = $("#merge-bar"), btn = $("#btn-merge-mode");
    if (btn) { btn.classList.toggle("btn-primary", on); btn.classList.toggle("btn-ghost", !on); }
    if (bar) bar.hidden = !on;
    if (on) updateMergeBar();
    // 卡片重绘以清除/应用选择态，并让点卡片行为切换
    renderSubs(curSegs);
    _playingIdx = -1;
    syncStickyOffset();    // 操作栏显隐改变面板高度 → 滚动吸附偏移同步更新
  }

  function mergeSelectedSegments() {
    const idxs = [..._mergeSel].sort((a, b) => a - b);
    if (idxs.length < 2) { logLine("⚠ " + T("merge.needTwo", "请至少选择 2 段")); return; }
    // 仅允许「相邻 + 同一说话者」的多段合并：跨说话者（对白归一人属误导）
    // 与跨句子（时间轴与中间未选段重叠）均拒绝。
    for (let i = 1; i < idxs.length; i++) {
      if (idxs[i] !== idxs[i - 1] + 1) { logLine("⚠ " + T("merge.notAdjacent", "所选段落必须相邻")); return; }
    }
    const segs = idxs.map(i => curSegs[i]);
    const spk0 = segs[0].speaker;
    if (segs.some(s => s.speaker !== spk0)) { logLine("⚠ " + T("merge.diffSpeaker", "所选段落说话者不一致")); return; }
    const join = (a, b) => {
      const lat = s => /[A-Za-z]/.test(s);
      return a + (lat(a.slice(-1)) || lat(b[0]) ? " " : "") + b;
    };
    const merged = {
      start: segs[0].start,
      end: segs[segs.length - 1].end,
      speaker: spk0,
      text: segs.map(s => s.text).reduce(join),
      words: segs.flatMap(s => s.words || []),
    };
    curSegs.splice(idxs[0], idxs.length, merged);
    setMergeMode(false);
    rerenderAfterEdit();
    logLine("✓ " + T("merge.done", "已合并 {n} 段", { n: idxs.length }));
  }

  // 拆分/合并/改名后的统一重绘（索引变化 → 全量重建卡片与时间轴区块）
  function rerenderAfterEdit() {
    _karaCache = null;
    _playingIdx = -1;
    clearSel();                           // 时间轴已变 → 选中区间失效
    renderSpeakerBar();
    renderSegStrip(curSegs, waveDur);
    renderSubs(curSegs);
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
    _blockEls = [];
    segments.forEach((s, i) => {
      const left = (s.start / dur) * 100, w = Math.max(0.6, ((s.end - s.start) / dur) * 100);
      const blk = document.createElement("div");
      blk.className = "seg-block" + (s.speaker ? " spk-" + (((s.speaker - 1) % 3) + 1) : "");
      blk.style.left = left.toFixed(2) + "%"; blk.style.width = w.toFixed(2) + "%";
      blk.dataset.seg = i;
      blk.title = (s.speaker ? spkDisplay(s.speaker) + "：" : "") + s.text;
      blk.textContent = s.text;
      blk.addEventListener("click", () => selectSeg(i));
      host.appendChild(blk);
      _blockEls.push(blk);
    });
    if (selRange && _blockEls[selRange.idx]) _blockEls[selRange.idx].classList.add("sel");
  }

  function renderSubs(segments) {
    const host = $("#subs"); host.innerHTML = "";
    _subEls = [];
    segments.forEach((s, i) => { const el = subCard(s, i); host.appendChild(el); _subEls.push(el); });
    if (selRange && _subEls[selRange.idx]) _subEls[selRange.idx].classList.add("sel");
  }
  function subCard(s, idx) {
    const el = document.createElement("div");
    // spkc-N：依说话者标示左侧外框色（与 chip 同色系，不污染卡片背景）
    el.className = "sub-card" + (s.speaker ? " spkc-" + (((s.speaker - 1) % 3) + 1) : "")
      + (_mergeSel.has(idx) ? " msel" : "");
    el.dataset.seg = idx;
    const spk = s.speaker ? `<span class="chip spk-${((s.speaker - 1) % 3) + 1}" title="${T("file.spkSwitchHint", "点击切换此行的说话者")}">${escapeHtml(spkDisplay(s.speaker))}</span>` : "";
    el.innerHTML = `<span class="tc">${fmtClock(s.start)} → ${fmtClock(s.end)}</span>
      <div class="body"><div class="spk">${spk}</div><div class="txt">${escapeHtml(s.text)}</div></div>
      <button class="sub-edit" title="${T("edit.line", "编辑此行")}" aria-label="编辑此行">
        <svg class="ico-pencil" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
        <svg class="ico-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
      </button>
      <button class="sub-split" title="${T("edit.split", "拆分此行：进入编辑后，在光标处点亮的 √ 切开（Ctrl+Enter）")}">
        <svg class="ico-scissors" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.12 15.88"/><path d="M14.47 14.48 20 20"/><path d="M8.12 8.12 12 12"/></svg>
        <svg class="ico-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
      </button>`;
    el.addEventListener("click", e => {
      if (_mergeMode) { toggleMergeSel(idx, el); return; }   // 合并模式：点卡片＝选择
      if (el.classList.contains("editing")) return;          // 任一编辑模式中：卡片不响应跳播
      const chipEl = e.target.closest(".chip");
      if (chipEl && s.speaker) {          // 点说话者 chip → 弹切换菜单（不触发跳播）
        spkSwitchMenu(idx, chipEl);
        return;
      }
      if (e.target.closest(".sub-edit") || el.classList.contains("editing")) return;
      selectSeg(idx);                      // 点行 → 选中该句区间（不播放；播放键循环）
    });
    // 笔＝文字编辑模式（进入后按钮变 √ 保存）；剪刀＝拆分模式（进入后按钮变
    // √ 按光标切开，文字锁定）。两按钮 mousedown preventDefault 保住焦点，
    // 不让 blur 提前收起模式。
    const editBtn = el.querySelector(".sub-edit");
    editBtn.addEventListener("mousedown", e => e.preventDefault());
    editBtn.addEventListener("click", e => { e.stopPropagation(); beginEdit(el, idx); });
    const splitBtn = el.querySelector(".sub-split");
    splitBtn.addEventListener("mousedown", e => e.preventDefault());
    splitBtn.addEventListener("click", e => {
      e.stopPropagation();
      if (el.classList.contains("editing")) return;   // 拆分模式中的点击由该模式自理
      beginSplitEdit(el, idx);
      logLine("ℹ " + T("edit.splitHow", "点击文本中要拆分的位置，再点亮的 √ 或按 Ctrl+Enter"));
    });
    return el;
  }

  // 行内编辑两种互斥模式（同一时刻一张卡只能处其一）：
  //   文字编辑（笔进入）：笔按钮变 √（保存，等同 Enter）；只能改字，无拆分入口。
  //   拆分模式（剪刀进入）：剪刀按钮变 √（光标就绪点亮，点击切开）；文字锁定
  //   不可改（contentEditable 仅为放光标，beforeinput/paste/cut/drop 全拦截）。
  // 两模式的 Esc 退出、blur 收尾、Enter（编辑＝保存）行为一致。
  function beginEdit(card, idx) {
    if (card.classList.contains("editing")) return;
    card.classList.add("editing", "mode-edit");
    const txt = card.querySelector(".txt");
    const editBtn = card.querySelector(".sub-edit");
    const orig = curSegs[idx].text;
    txt.contentEditable = "true"; txt.spellcheck = false;
    txt.focus();
    const range = document.createRange(); range.selectNodeContents(txt);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    editBtn.title = T("edit.save", "保存（Enter）");
    let done = false;
    function finish(save) {
      if (done) return;
      done = true;
      txt.removeEventListener("keydown", onKey);
      txt.removeEventListener("blur", onBlur);
      editBtn.removeEventListener("click", onSaveClick);
      txt.contentEditable = "false";
      card.classList.remove("editing", "mode-edit");
      editBtn.title = T("edit.line", "编辑此行");
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
    function onSaveClick(e) { e.stopPropagation(); e.preventDefault(); finish(true); }
    txt.addEventListener("keydown", onKey);
    txt.addEventListener("blur", onBlur);
    editBtn.addEventListener("click", onSaveClick);
  }

  // 拆分模式：只能放光标/选区，文字不可编辑；√ 随光标就绪点亮，点击切开。
  function beginSplitEdit(card, idx) {
    if (card.classList.contains("editing")) return;
    card.classList.add("editing", "mode-split");
    const txt = card.querySelector(".txt");
    const splitBtn = card.querySelector(".sub-split");
    // 光标需要 contentEditable，但拆分模式禁止改字：先拦输入事件（键盘/
    // 粘贴/剪切/拖放），再以 MutationObserver 兜底——IME 组合输入等路径
    // 可能绕过 beforeinput，任何改动一律强制还原为原文。
    const origText = txt.textContent;
    const blockInput = e => { e.preventDefault(); };
    const enforceLock = () => {
      if (txt.textContent !== origText) {
        txt.textContent = origText;
        updateReady();
      }
    };
    const mo = new MutationObserver(enforceLock);
    txt.contentEditable = "true"; txt.spellcheck = false;
    txt.addEventListener("beforeinput", blockInput);
    txt.addEventListener("paste", blockInput);
    txt.addEventListener("cut", blockInput);
    txt.addEventListener("drop", blockInput);
    mo.observe(txt, { characterData: true, childList: true, subtree: true });
    txt.focus();
    const range = document.createRange(); range.selectNodeContents(txt);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    const updateReady = () => {
      const pos = caretOffsetIn(txt, "start");
      const full = txt.textContent || "";
      splitBtn.classList.toggle("ready",
        pos != null && !!full.slice(0, pos).trim() && !!full.slice(pos).trim());
    };
    document.addEventListener("selectionchange", updateReady);
    updateReady();               // 初始全选态 → 切点在首 → 置灰
    let done = false;
    function finish() {
      if (done) return;
      done = true;
      mo.disconnect();
      document.removeEventListener("selectionchange", updateReady);
      txt.removeEventListener("beforeinput", blockInput);
      txt.removeEventListener("paste", blockInput);
      txt.removeEventListener("cut", blockInput);
      txt.removeEventListener("drop", blockInput);
      txt.removeEventListener("keydown", onKey);
      txt.removeEventListener("blur", onBlur);
      splitBtn.removeEventListener("click", onSplitClick);
      splitBtn.classList.remove("ready");
      txt.contentEditable = "false";
      card.classList.remove("editing", "mode-split");
      txt.blur();
    }
    function doSplit() {
      if (splitSegment(idx, txt.textContent || "", caretOffsetIn(txt, "start"))) return;
      txt.classList.remove("split-hint");   // 失败 → 闪红提示，留在拆分模式
      void txt.offsetWidth;
      txt.classList.add("split-hint");
    }
    function onKey(e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doSplit(); }
      else if (e.key === "Escape") { e.preventDefault(); finish(); }
    }
    function onBlur() { finish(); }
    function onSplitClick(e) { e.stopPropagation(); doSplit(); }
    txt.addEventListener("keydown", onKey);
    txt.addEventListener("blur", onBlur);
    splitBtn.addEventListener("click", onSplitClick);
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
    // 点波形：只移动播放头（不自动播）；手动定位视为放弃片段选中
    wave.onclick = e => {
      const r = wave.getBoundingClientRect();
      const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * waveDur;
      clearSel();
      if (audioEl) { audioEl.currentTime = t; onTime(); }
    };
    // 播放键：有选中片段 → 从片段头起循环播放；无 → 正常连播
    if (play) play.onclick = () => {
      if (!audioEl) return;
      if (audioEl.paused) {
        if (selRange && (audioEl.currentTime < selRange.start
                         || audioEl.currentTime >= selRange.end - 0.02)) {
          audioEl.currentTime = selRange.start;
        }
        audioEl.play();
      } else audioEl.pause();
    };
    // 停止：暂停并回到片段头（无选中则回开头）
    const stop = $("#wave-stop");
    if (stop) stop.onclick = () => {
      if (!audioEl) return;
      audioEl.pause();
      audioEl.currentTime = selRange ? selRange.start : 0;
      onTime();
    };

    function onTime() {
      // 片段循环：播过区间尾 → 回区间头（timeupdate ~4fps，短片段也能回卷）
      if (selRange && !audioEl.paused && audioEl.currentTime >= selRange.end - 0.02) {
        audioEl.currentTime = selRange.start;
      }
      const t = audioEl.currentTime, ratio = waveDur ? t / waveDur : 0;
      const ph = $("#playhead"); if (ph) ph.style.left = (ratio * 100).toFixed(2) + "%";
      // 波形已播色：只在前进/后退跨桶时重刷（200 根，避免每次 timeupdate 全量 toggle）
      const playedIdx = Math.max(0, Math.min(bars.length - 1, Math.floor(ratio * bars.length)));
      if (playedIdx !== _playedBars) {
        if (playedIdx > _playedBars) {
          for (let i = _playedBars + 1; i <= playedIdx; i++) bars[i].classList.add("played");
        } else {
          bars.forEach((b, i) => b.classList.toggle("played", i <= playedIdx));
        }
        _playedBars = playedIdx;
      }
      // 当前段高亮：只动换段时的两行（缓存的元素数组，无 DOM 查询）
      const cur = segIndexAt(t);
      if (cur !== _playingIdx) {
        if (_playingIdx >= 0) {
          if (_subEls[_playingIdx]) _subEls[_playingIdx].classList.remove("playing");
          if (_blockEls[_playingIdx]) _blockEls[_playingIdx].classList.remove("playing");
        }
        if (cur >= 0) {
          if (_subEls[cur]) _subEls[cur].classList.add("playing");
          if (_blockEls[cur]) _blockEls[cur].classList.add("playing");
          if (playFollow) _subEls[cur].scrollIntoView({ block: "nearest" });
        }
        _playingIdx = cur;
      }
      const tm = $("#wave-cur"); if (tm) tm.textContent = fmtClock(t);
    }
  }
  let playFollow = true;
  function setPlayIcon(playing) {
    const b = $("#wave-play"); if (!b) return;
    b.innerHTML = playing
      ? `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
  }
  // ── 片段选中：点字幕行/区块 → 波形上亮出区间，播放键循环该片段 ──────
  function selectSeg(idx) {
    const s = curSegs[idx];
    if (!s || !waveDur) return;
    selRange = { start: s.start, end: Math.max(s.end, s.start + 0.05), idx };
    let ov = $("#wave-sel");
    if (!ov) { ov = document.createElement("div"); ov.id = "wave-sel"; ov.className = "wave-sel"; $("#wave").appendChild(ov); }
    ov.style.left = ((selRange.start / waveDur) * 100).toFixed(2) + "%";
    ov.style.width = (Math.max(0.4, ((selRange.end - selRange.start) / waveDur) * 100)).toFixed(2) + "%";
    _subEls.forEach((el, i) => el.classList.toggle("sel", i === idx));
    _blockEls.forEach((el, i) => el.classList.toggle("sel", i === idx));
  }

  function clearSel() {
    selRange = null;
    const ov = $("#wave-sel"); if (ov) ov.style.width = "0%";
    _subEls.forEach(el => el.classList.remove("sel"));
    _blockEls.forEach(el => el.classList.remove("sel"));
  }
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
      if (active) {
        // 起跳即最大、随进度二次缓出回落：唱到的瞬间「弹」一下再归位
        const dur = Math.max(0.05, w.end - w.start);
        const p = Math.min(1, (t - w.start) / dur);
        scale = 1 + 0.24 * (1 - p) * (1 - p);
      } else if (!sung) {
        // 预备动作：开唱前 ~0.12s 先微幅抬起蓄力（Apple Music 动态歌词风）
        const lead = w.start - t;
        if (lead <= 0.12) scale = 1 + 0.06 * (1 - lead / 0.12);
      }
      return { sung, active, scale };   // scale 上限 1.24 ≤ 1.3，整行不至抖动过猛
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

  // 每帧更新：定位目前段（提示位置 + 二分，见 segIndexAt）→（换段才重绘）→ 套用状态。
  function updateKaraoke(t) {
    if (!curSegs.length) return;
    let idx = segIndexAt(t);
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
  // 合并段落：进入/退出选择模式；确认/取消在 #merge-bar
  $("#btn-merge-mode") && $("#btn-merge-mode").addEventListener("click", () => setMergeMode(!_mergeMode));
  $("#merge-ok") && $("#merge-ok").addEventListener("click", mergeSelectedSegments);
  $("#merge-cancel") && $("#merge-cancel").addEventListener("click", () => setMergeMode(false));

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
      const spk = s.speaker ? `${spkLabel(s.speaker)}：` : "";
      return `${i + 1}\n${srtTs(s.start)} --> ${srtTs(s.end)}\n${spk}${s.text}\n`;
    }).join("\n");
  }
  function buildTxt(segs) {
    return segs.some(s => s.speaker)
      ? segs.map(s => (s.speaker ? `${spkLabel(s.speaker)}：` : "") + s.text).join("\n")
      : segs.map(s => s.text).join("");
  }
  // LRC 歌词：行首时间戳 [mm:ss.xx]，带说话者前缀
  function lrcTs(sec) {
    const s = Math.max(0, sec);
    const m = Math.floor(s / 60), ss = s - m * 60;
    return `${String(m).padStart(2, "0")}:${ss.toFixed(2).padStart(5, "0")}`;
  }
  function buildLrc(segs) {
    return segs.map(s =>
      `[${lrcTs(s.start)}]${s.speaker ? `${spkLabel(s.speaker)}：` : ""}${s.text}`
    ).join("\n") + "\n";
  }
  // ASS 卡拉OK：有字级 words 时输出 \k 逐字标签（厘秒拍点，播放器即卡拉OK高亮），
  // 无 words 退回整行显示。字间空隙并入前一字拍子，保持拍点连续。
  function buildAss(segs) {
    const ts = sec => {
      const cs = Math.max(0, Math.round(sec * 100));
      const h = Math.floor(cs / 360000), m = Math.floor(cs / 6000) % 60;
      const s = Math.floor(cs / 100) % 60, c = cs % 100;
      return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(c).padStart(2, "0")}`;
    };
    const header = [
      "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
      "[V4+ Styles]",
      "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
      "Style: Default,Noto Sans SC,72,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,2,1,2,80,80,80,1",
      "", "[Events]",
      "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ];
    const lines = segs.map(s => {
      const spk = s.speaker ? `${spkLabel(s.speaker)}：` : "";
      const words = (s.words && s.words.length) ? s.words : null;
      let text;
      if (words) {
        text = spk + words.map((w, i) => {
          const endT = i < words.length - 1 ? Math.max(w.end, words[i + 1].start) : s.end;
          const k = Math.max(1, Math.round((endT - w.start) * 100));
          return `{\\k${k}}${w.text}`;
        }).join("");
      } else {
        text = spk + s.text;
      }
      return `Dialogue: 0,${ts(s.start)},${ts(s.end)},Default,,0,0,0,,${text}`;
    });
    return header.concat(lines).join("\n") + "\n";
  }
  const _SAVE_EXTS = { srt: ".srt", txt: ".txt", lrc: ".lrc", ass: ".ass" };
  async function saveSubtitle(fmt) {
    if (!curSegs.length) return;
    if (!fmt || !_SAVE_EXTS[fmt]) {
      fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    }
    const text = fmt === "txt" ? buildTxt(curSegs)
      : fmt === "lrc" ? buildLrc(curSegs)
      : fmt === "ass" ? buildAss(curSegs)
      : buildSrt(curSegs);
    const base = (picked && picked.name) ? picked.name.replace(/\.[^.]+$/, "") : "字幕";
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = base + (_SAVE_EXTS[fmt] || ".srt");
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    logLine("✓ " + T("file.saved", "已存档字幕：") + a.download);
  }
  // 保存菜单：SRT（默认，按全局输出格式）/ ASS / LRC / TXT
  const saveMenu = $("#save-menu");
  $("#btn-save-sub").addEventListener("click", () => {
    if (!curSegs.length) return;
    saveMenu.hidden = !saveMenu.hidden;
  });
  saveMenu.addEventListener("click", e => {
    const b = e.target.closest("button[data-fmt]");
    if (!b) return;
    saveMenu.hidden = true;
    saveSubtitle(b.dataset.fmt);
  });
  document.addEventListener("click", e => {
    if (!e.target.closest(".save-wrap") && !saveMenu.hidden) saveMenu.hidden = true;
  });

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
      if (h.summary.red > 0) { sum.className = "chip chip-live"; sum.textContent = T("model.healthRed", "{n} 项需处理", { n: h.summary.red }); }
      else if (h.summary.yellow > 0) { sum.className = "chip chip-accent"; sum.textContent = T("model.healthYellow", "核心就绪 · {n} 项将于启用时自动下载", { n: h.summary.yellow }); }
      else { sum.className = "chip chip-ok"; sum.textContent = T("model.healthOk", "全部就绪"); }
    }
    const host = $("#health-panel"); host.innerHTML = "";
    const dot = s => `<span class="hdot hdot-${s}"></span>`;
    const groups = [...(h.cores || [])];
    if (h.shared && h.shared.length) groups.push({ label: T("model.shared", "共享组件"), items: h.shared });
    groups.forEach(g => {
      const card = document.createElement("div"); card.className = "health-core";
      const active = g.backend && g.backend === h.activeBackend;
      card.innerHTML = `<div class="hc-title">${escapeHtml(g.label)}` +
        (active ? ` <span class="chip chip-ok" style="font-size:10px">${T("model.inUse", "使用中")}</span>` : ``) + `</div>`;
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
      modelMsg("", null);
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
  function updateArch() {
    const o = $("#model-select").selectedOptions[0];
    $("#model-arch").textContent = (o && o.dataset.arch) || "";
    const note = o && o.dataset.note, el = $("#model-note");
    if (note) { el.hidden = false; el.textContent = note; } else { el.hidden = true; el.textContent = ""; }
  }

  // ── 模型页：引擎卡 + 模型下拉（先选引擎，再选模型）───────────────

  // ── （基础模式已移除；保留空实现以免调用端报错）─────────────────

  // ── 推理加速版本（CrispASR 的 Vulkan / CUDA / CPU build）───────────
  //   后端依 WMI + nvidia-smi 检测硬件后给推荐；这里只负责呈现与回写选择。
  //   加速版本只对 CrispASR 引擎有意义：选 OpenVINO 时整个区块隐藏。
  //   实际下载发生在「下载并加载模型」时（换版本＝换引擎组件，热切换）。
  async function renderAccel() {
    let a;
    const section = $("#accel-section");
    try { a = await API.getAccel(); } catch { if (section) section.hidden = true; return; }
    if (!a || !section) { if (section) section.hidden = true; return; }
    if (a.applicable === false) { section.hidden = true; return; }   // OpenVINO：无需加速器
    const host = $("#accel-cards");
    if (!a.ok || !host) { section.hidden = true; if (host) host.innerHTML = ""; return; }
    section.hidden = false;
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
      ? T("model.accelNeeds", "加速器尚未下载或版本不符（最新 {v}），点「下载并加载模型」时会一并取得。", { v: a.latest })
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
      if (res && res.message) accelMsg(res.message, "info");
      await renderAccel();
    } catch (err) { accelMsg(String(err), "warn"); }
  });

  // 点核心卡 → 切核心、模型回该核心首项并套用；加速版本区块随引擎刷新
  $("#core-cards").addEventListener("click", async e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    _curCore = card.dataset.core;
    $$(".radio-card", $("#core-cards")).forEach(c => c.classList.toggle("sel", c === card));
    const core = coreObj(_curCore);
    const first = core && core.models[0] ? core.models[0].label : "";
    populateModels(_curCore, first);
    await applyModel(_curCore, first);
    await renderAccel();          // 引擎切换 → 加速版本区块显隐/选项随之更新
  });
  // 换模型 → 套用
  $("#model-select").addEventListener("change", async e => {
    updateArch();
    await applyModel(_curCore, e.target.value);
  });
  let _modelLoading = false;       // 模型「就地下载并加载」进行中（进度导到模型页）
  let _switchBusy = false;         // 转录进行中 → 模型切换整体禁用
  // 转录/加载忙碌 → 引擎卡与模型下拉禁操作（热切换的并发保护）
  function _switchBlocked() { return _switchBusy; }
  async function applyModel(core, model) {
    if (_switchBlocked()) return;   // 转录中忽略选择变更（下拉会回弹到当前值）
    try {
      const res = await API.setModel(core, model);
      if (res && res.message) modelMsg(res.message, "info");
      else modelMsg("", null);
      updateLoadBtn(res);
    } catch (err) { modelMsg("套用失败：" + (err.message || err), "warn"); }
  }
  // 依 setModel 结果 + 目前加载状态，决定「下载并加载」按钮的文字/可用性（热切换版）
  async function updateLoadBtn(res) {
    const btn = $("#btn-model-load"); if (!btn) return;
    let st = {};
    try { st = await API.getStatus(); } catch (e) {}
    btn.classList.remove("btn-ghost");
    if (_modelLoading || st.loading) {
      btn.disabled = true; btn.textContent = T("model.loading", "下载／加载中…"); return;
    }
    if (st.transcribing) {
      btn.disabled = true; btn.textContent = T("model.busy", "转录进行中，无法切换"); btn.dataset.act = ""; return;
    }
    if (st.modelReady && !(res && res.identityChanged)) {
      // 已就绪且选择未变 → 引导前往转录
      btn.disabled = false; btn.textContent = T("model.goto", "前往语音转文字"); btn.dataset.act = "goto"; return;
    }
    // 选择有变化（或尚未加载）→ 热切换：就地下载并加载
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
    // 界面语言：菜单回填 + 套用 i18n（含目前视图标题）
    const uiLang = s.uiLang || "简体中文";
    if ([...$("#set-lang").options].some(o => o.value === uiLang)) $("#set-lang").value = uiLang;
    if (window.I18N) { I18N.setLang(uiLang); refreshViewTitle(); }
  }
  function segSet(sel, v) { $$(sel + " button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
  // 界面缩放：用 CSS zoom（Chromium/WebView2 支持）整体缩放，px 版面也能等比生效。
  function applyScale(pct) { document.body.style.zoom = (Math.max(50, Math.min(200, +pct || 100)) / 100); }

  $("#set-scale").addEventListener("change", e => {
    const v = +e.target.value; applyScale(v); API.setSettings({ scale: v });
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
  // 检查更新：后端比对 GitHub 最新 release；有新版自动打开下载页
  $("#btn-check-update") && $("#btn-check-update").addEventListener("click", async () => {
    const btn = $("#btn-check-update");
    btn.disabled = true; btn.textContent = T("set.checking", "检查中…");
    try {
      const res = await API.checkUpdate();
      if (res && res.ok) {
        if (res.hasUpdate) {
          flash(btn, T("set.newVersion", "新版本 v{v}！打开下载页…", { v: res.latest }));
          if (res.url) window.open(res.url, "_blank");
        } else {
          flash(btn, T("set.upToDate", "已是最新 v{v}", { v: res.current || res.latest }));
        }
      } else {
        flash(btn, T("set.checkFailed", "检查失败（网络？）"));
      }
    } catch (e) { flash(btn, T("set.checkFailed", "检查失败（网络？）")); }
  });

  // ── 转录历史：subtitles/ 最近产出，点击回填查看／再存档 ──────────────
  async function renderHistory() {
    const head = $("#history-head"), host = $("#history-list");
    if (!head || !host) return;
    let h;
    try { h = await API.listHistory(); } catch (e) { return; }
    const items = (h && h.items) || [];
    if (!items.length) { head.hidden = true; host.hidden = true; return; }
    head.hidden = false; host.hidden = false;
    host.innerHTML = "";
    items.forEach(it => {
      const el = document.createElement("button");
      el.className = "history-item";
      const when = new Date(it.mtime * 1000).toLocaleString();
      el.innerHTML = `<span class="hn">${escapeHtml(it.name)}</span>
        <span class="hm">${when} · ${Math.max(1, Math.round(it.size / 1024))} KB</span>
        <span class="hx" role="button" tabindex="0" title="${T("file.historyDel", "删除（到回收站）")}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/></svg>
        </span>`;
      el.addEventListener("click", async e => {
        if (e.target.closest(".hx")) { deleteHistoryItem(it, el); return; }
        try {
          const r = await API.loadHistory(it.name);
          if (r && r.ok && (r.segments || []).length) {
            renderResult(r.segments);
            $("#btn-open-dir").disabled = false;
            $("#btn-save-sub").disabled = false;
            logLine("✓ " + T("file.historyLoaded", "已载入历史字幕：") + it.name);
          }
        } catch (e) {}
      });
      host.appendChild(el);
    });
  }

  // 删除历史条目：确认 → 后端删（Windows 到回收站）→ 刷新列表
  async function deleteHistoryItem(it, el) {
    if (!window.confirm(T("file.historyDelConfirm", "确定删除字幕文件「{name}」吗？\n（移到回收站，可恢复）", { name: it.name }))) return;
    el.disabled = true;
    try {
      const r = await API.deleteHistory(it.name);
      if (r && r.ok) {
        logLine("✓ " + T("file.historyDeleted", "已删除：") + it.name
          + (r.recycled === false ? T("file.historyDeletedPerm", "（已直接删除）") : ""));
        renderHistory();
      } else {
        el.disabled = false;
        logLine("✕ " + T("file.historyDelFail", "删除失败：") + ((r && r.error) || "未知错误"));
      }
    } catch (err) {
      el.disabled = false;
      logLine("✕ " + T("file.historyDelFail", "删除失败：") + (err.message || err));
    }
  }

  // ── 共享工具 ────────────────────────────────────────────
  function fmtClock(sec) {
    sec = Math.max(0, Math.round(sec));
    const m = String(Math.floor(sec / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${m}:${s}`;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  // 按钮短暂提示文字后还原（检查更新结果等一次性反馈）
  function flash(btn, msg, ms = 2600) {
    const o = btn.textContent;
    btn.textContent = msg;
    btn.disabled = true;
    setTimeout(() => { btn.textContent = o; btn.disabled = false; }, ms);
  }

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
      chip.hidden = false; chip.className = "chip chip-muted"; chip.textContent = T("record.unsupported", "不支持录音");
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
      def.value = ""; def.textContent = T("record.defaultMic", "默认麦克风");
      sel.appendChild(def);
      mics.forEach((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId;
        o.textContent = d.label || T("record.micN", "麦克风 {n}", { n: i + 1 });
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
    if (!recSupported()) { recNote(T("record.noApi", "此环境不支持录音 API。"), true); return; }
    // 依选定的麦克风建立音讯约束（空值 → 系统默认设备）。
    const micId = $("#rec-mic") ? $("#rec-mic").value : "";
    const audioConstraint = { echoCancellation: true, noiseSuppression: true };
    if (micId) audioConstraint.deviceId = { exact: micId };
    try {
      REC.stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
    } catch (err) {
      // 被拒 / 无设备 → 页核显示权限指引，不只丢一行错误
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      recNote((denied ? T("record.denied", "麦克风权限被拒。") : T("record.micFail", "无法取得麦克风：")) + (err.message || err), true);
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
    REC.recStart = Date.now();      // 真实时间轴基准：段起止=录音内的真实秒数
    API.recordState(true);          // 同步后端：录音中 → 关窗需确认
    $("#rec-btn").classList.add("recording");
    const lw = $("#rec-live-wave"); lw.hidden = false;
    lw.innerHTML = Array.from({ length: 28 }, () => `<div class="b" style="height:4px"></div>`).join("");
    $("#rec-timer").textContent = "00:00";
    REC.timer = setInterval(() => { REC.sec++; $("#rec-timer").textContent = fmtClock(REC.sec); }, 1000);
    recNote(T("record.listening", "聆听中…说完停顿约 2 秒会自动识别；再按一次结束并完成最后一段。"));
    startSeg(); monitor();
  }
  function recElapsed() { return REC.recStart ? (Date.now() - REC.recStart) / 1000 : 0; }
  function startSeg() {
    REC.chunks = []; REC.speech = false; REC.silentSince = 0; REC.segStart = Date.now();
    REC.segStartElapsed = recElapsed();   // 本段在录音里的真实起点
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
      .find(m => MediaRecorder.isTypeSupported(m)) || "";
    REC.recorder = mime ? new MediaRecorder(REC.stream, { mimeType: mime })
                        : new MediaRecorder(REC.stream);
    REC.recorder.ondataavailable = e => { if (e.data && e.data.size) REC.chunks.push(e.data); };
    REC.recorder.onstop = onSegStop;
    REC.recorder.start();
  }
  function cutSeg() {
    if (REC.recorder && REC.recorder.state === "recording") {
      REC.segEndElapsed = recElapsed();   // 录制媒体止于 stop() 调用时刻
      REC.recorder.stop();
    }
  }
  async function onSegStop() {
    const dur = Date.now() - REC.segStart;
    const blob = new Blob(REC.chunks, { type: REC.chunks[0] ? REC.chunks[0].type : "audio/webm" });
    const ok = REC.speech && dur >= MIN_SEG_MS && blob.size > 1200;
    // 真实时间轴：本段起点=开录时刻、终点=切段时刻（静音间隔如实留白），
    // 导出的 SRT 可直接用于同步（旧版固定 5s 顺排，时间轴是假的）。
    const t0 = REC.segStartElapsed || 0;
    const t1 = Math.max(t0 + 0.5, REC.segEndElapsed || recElapsed());
    if (REC.on) startSeg(); else teardownRec();    // 先续录，避免上传期间漏话
    if (ok) {
      try {
        const file = new File([blob], "recording.webm", { type: blob.type });
        const recLang = $("#rec-lang") ? $("#rec-lang").value : "";
        const res = await API.transcribe({ file, language: recLang || null, diarize: false, align: false });
        (res.segments || []).forEach(s => {
          if (s.text && s.text.trim()) appendRecLine(s.text.trim(), t0, t1);
        });
      } catch (err) { recNote(T("record.recFailed", "识别失败：") + (err.message || err), true); }
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
    API.recordState(false);         // 同步后端：录音已结束
    if (REC.raf) cancelAnimationFrame(REC.raf);
    clearInterval(REC.timer);
    $("#rec-live-wave").hidden = true;
    recNote(T("record.stopped", "已停止。"));
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
  function appendRecLine(text, t0, t1) {
    const host = $("#rec-subs");
    if (REC.firstResult) { host.innerHTML = ""; REC.firstResult = false; }
    // 真实时间轴：t0/t1 为该段在录音里的真实起止秒（onSegStop 传入）
    const start = Number.isFinite(t0) ? t0
      : (REC.segs.length ? REC.segs[REC.segs.length - 1].end + 0.1 : 0);
    const end = Number.isFinite(t1) ? Math.max(t1, start + 0.5) : start + 5;
    const seg = { start, end, speaker: null, text };
    REC.segs.push(seg);
    const el = document.createElement("div");
    el.className = "sub-card";
    el.innerHTML = `<span class="tc">${fmtClock(start)}</span>
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
      recNote(T("record.autosaveOn", "实时存档已启用：每段识别完成后会自动写入所选文件。"));
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
    } catch (e) { recNote(T("record.autosaveFail", "实时存档写入失败：") + (e.message || e), true); REC.fileHandle = null; }
  }

  // ── 启动 ────────────────────────────────────────────────
  // 关闭兜底（Edge --app / 浏览器访问本地端点时无原生确认框）：
  // 有任务进行中 → 触发浏览器原生「确定要离开吗」提示。
  // 桌面原生窗口（URL 带 ?native=1）的关闭确认由 app_webview 的 closing
  // 事件处理（原生 MessageBox），这里不注册，避免确认后二次弹窗。
  if (!new URLSearchParams(location.search).has("native")) {
    window.addEventListener("beforeunload", (e) => {
      const busy = running || REC.on || _modelLoading;
      if (busy) { e.preventDefault(); e.returnValue = ""; }
    });
  }
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

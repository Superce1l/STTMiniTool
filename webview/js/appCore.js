/* ============================================================
   appCore.js — UI 逻辑核心（原 app.js IIFE → ES module 渐进迁移，行为零变化）

   迁移映射（原 app.js 行号 → 去向）：
   - IIFE 包裹 :6-7 与 :2554   → 模块顶层（模块作用域即原 IIFE 私有作用域）
   - $ / $$ / T :8-16          → util.js 导出（本文件经 import 解构，实现逐字一致）
   - _curView :24              → state.js state.curView
   - curSegs / speakerNames :271/:274 → state.js state.curSegs / state.speakerNames
   - _search :285              → state.js state.search
   - _mergeMode / _mergeSel :338-339  → state.js state.mergeMode / state.mergeSel
   - selRange :342             → state.js state.selRange
   - _modelLoading / _switchBusy :2008-2009 → state.js（与本模块/模型视图共享）
   - REC :2259-2265            → state.js state.REC（对象字面量原样）
   - refreshStatus :53-75      → 留本模块；新增写入 state.lastStatus（状态快照，
                                 供视图模块读取，替代对 appCore 的反向 import）
   - 模型页块 :1795-2064       → views/modelView.js（含其 listener 绑定，
                                 于该模块求值期绑定，时序等同原 IIFE 顶层）
   - loadLanguages :2065-2082  → 留本模块（音频页 + 录制页两处语言下拉共用）
   - 顶层 listener 绑定 / API.on / wire* / init 调用 → boot()
     （main.js 在 DOM ready 后调用；function/let/const 声明留在模块顶层，
      boot 内语句按原 app.js 相对顺序移入，执行语义不变）

   未迁出的纯局部状态（模块作用域私有，语义同原 IIFE 内 let/const）：
   picked / running / audioEl / audioUrl / waveDur / _editHistory / _subEls /
   _blockEls / _playingIdx / _playedBars / _segHint / _stickyPad / _virtualMode /
   _vwin / _spacerTop / _spacerBot / _scrollHost / _scrollRaf / _measureRaf /
   _editCtx / playFollow / karaokeOn / karaokeRAF / _karaCache /
   _recLatencyClosed / _mqHandler / VIEW_TITLES / WAVE_N / RENDER_THRESHOLD /
   SUB_GAP / SUB_DEFAULT_H / _KARA_PUNCT / _SAVE_EXTS / SILENCE_MS /
   MIN_SEG_MS / MAX_SEG_MS / VAD_THRESH / drop / hiddenFile

   红线：四视图切换 / 转录（同步+队列）/ 字幕编辑（委托、虚拟滚动窗口、脏区、
   撤销、搜索替换）/ 卡拉OK / 录制 / 模型加载热切换 / 设置 / 主题语言 ——
   行为零变化；性能包四项优化（虚拟滚动 / 事件委托 / 脏区更新 / 卡拉OK增量）
   实现代码原样保留，仅改写 import 与共享状态引用。
   ============================================================ */
import { state } from "./state.js";            // 跨视图共享可变状态（含 lastStatus/gotoView）
import * as STTUtil from "./util.js";          // 纯工具（escapeHtml/fmtClock/历史栈/虚拟窗口/卡拉OK游标…）
import { renderModel, renderHealth, setModelProg, modelMsg, updateLoadBtn }
  from "./views/modelView.js";                 // 模型页视图模块（单向依赖：appCore → modelView）
const { $, $$, T } = STTUtil;                  // 原顶层助手（实现逐字一致）
const API = window.QwenAPI;                    // 经典脚本 bridge.js 的全局（不 import 不存在的模块）

  const VIEW_TITLES = {
    file: "音频转字幕", record: "录制转换",
    model: "模型与设备", settings: "设置",
  };

  // ── 导览切换 ────────────────────────────────────────────
  function viewTitle(name) { return (window.I18N && window.I18N.t("view." + name)) || VIEW_TITLES[name] || ""; }
  function refreshViewTitle() { $("#view-title").textContent = viewTitle(state.curView); }
  function switchView(name) {
    state.curView = name;
    $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
    $$(".view").forEach(v => v.classList.toggle("active", v.dataset.view === name));
    $("#view-title").textContent = viewTitle(name);
    $("#view-ctx").textContent = "";
    if (name === "model") renderModel();
    if (name === "record") { enumerateMics(); refreshMicPerm(); updateRecLatencyHint(); }   // 列举设备 + 麦克风权限状态 + 延迟提示
    if (name === "file") renderHistory();    // 最近转录（subtitles/ 产出回填）
  }

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
    state.lastStatus = s;         // 最近状态快照：视图模块经 state.js 读取（不反向 import appCore）
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
    updateRecLatencyHint(s);      // 录制页提示条随状态刷新（热切换后模型可能变化）
  }
  function statusTip(s) {
    if (s.loading) return T("status.tipLoading", "正在下载／加载模型…");
    if (!s.modelReady) {
      if (s.selectedReady === false) {
        // 带判定依据：程序检查的目录 + 缺哪些文件（「明明有模型却说未下载」自查）
        const d = s.selectedDetail;
        if (d) {
          let tip = T("status.tipNotDownloadedDir",
                      "所选模型尚未下载——程序检查的目录：\n{dir}", { dir: d.modelDir });
          if (d.coreIssue)
            tip += "\n" + T("status.tipCoreIssue", "加速器：{issue}", { issue: d.coreIssue });
          const miss = (d.missing || []).concat(d.missingCrispasr || []);
          if (miss.length)
            tip += "\n" + T("status.tipMissing", "缺少：{files}", { files:
              miss.slice(0, 3).join("、") + (miss.length > 3 ? "…" : "") });
          return tip;
        }
        return T("status.tipNotDownloaded", "所选模型尚未下载——到「模型」页点「下载并加载模型」");
      }
      return T("status.tipNotLoaded", "模型文件已备——到「模型」页点「下载并加载模型」");
    }
    if (s.selectionDiffers) {
      return T("status.tipSwitch", "已加载：{loaded}\n已选择：{sel}\n到「模型」页点「下载并加载模型」完成切换",
               { loaded: s.activeModel || "—", sel: s.selectedModel || "—" });
    }
    if (s.error) return T("status.tipFallback", "本次加载失败，已回退旧模型继续使用：{e}", { e: s.error });
    return T("status.tipReady", "当前模型：{m}", { m: s.activeModel || "—" });
  }

  // ── 录制页实时识别延迟提示 ──────────────────────────────
  // 依据实测（GTX 1660 SUPER / 0.8.53）：CrispASR 每段转录的进程固定开销
  // （Vulkan init + 模型上显存）约 2-3s，占实时出字延迟 ~90%；whisper base
  // 固定开销约 0.5s；OpenVINO 为进程内推理、无冷启动。详见
  // crisp_engine.transcribe docstring 与 docs/upstream-feature-request-crispasr.md。
  let _recLatencyClosed = false;
  async function updateRecLatencyHint(status) {
    if (state.curView !== "record" || _recLatencyClosed) return;
    const el = $("#rec-latency");
    if (!el) return;
    const s = status || await API.getStatus().catch(() => null);
    if (!s) return;
    const backend = String(s.backend || "");
    const model = String(s.activeModel || "");
    let key = null;
    if (backend.includes("CRISPASR")) {
      key = /Base/i.test(model) ? "rec.latencyBase" : "rec.latencyLarge";
    } else if (backend.includes("OpenVINO")) {
      key = "rec.latencyCpu";
    }
    if (!key) { el.hidden = true; return; }
    const txt = el.querySelector("div");
    if (txt) txt.textContent = T(key);
    el.hidden = false;
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
    const meta = f.sizeSec ? T("file.sizeSec", "{n} 秒", { n: f.sizeSec })
                           : (f.size ? (f.size / 1048576).toFixed(1) + " MB" : "");
    drop.innerHTML = `<span class="file-chip">${STTUtil.escapeHtml(f.name)}
      ${meta ? `<span class="meta">${meta}</span>` : ""}
      <button class="x" title="${T("common.remove", "移除")}">✕</button></span>`;
    drop.querySelector(".x").addEventListener("click", ev => { ev.stopPropagation(); setFile(null); });
    $("#btn-run").disabled = false;
  }
  function renderDropEmpty() {
    drop.innerHTML = `<div class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/></svg></div>
      <div class="big">${T("file.dropBig", "拖曳音讯到此，或<b>点击选择文件</b>")}</div>
      <div class="sub">${T("file.dropSub", "支持 mp3 / wav / m4a / 视频音轨")}</div>`;
  }

  // 转录
  let running = false;
  // pct<0：无有效进度，只显示 status 文案（空 = 连接已恢复，撤掉提示文字，
  // 进度条保留原值等待下一条真实进度）。仅任务进行中才显示，空闲断线不打扰
  // （EventSource 自己会重连，无需用户操作）。
  function connHint(status) {
    if (!(running || state.modelLoading)) return;
    if (state.modelLoading) {
      $("#model-progress").hidden = false;
      $("#model-prog-status").textContent = status || "";
    } else {
      $("#progress").hidden = false;
      $("#prog-status").textContent = status || "";
    }
  }
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
  let audioEl = null, audioUrl = null, waveDur = 0;   // curSegs（原 :271 同行）→ state.js state.curSegs

  // 说话者显示名（chip 用，默认带空格；随界面语言）与导出名（与后端写出格式
  // 一致，无空格；固定简中以保证导出档案与后端产出一致，不随界面语言变化）
  function spkDisplay(n) { return state.speakerNames[n] || T("spk.default", "说话者 {n}", { n }); }
  function spkLabel(n) { return state.speakerNames[n] || ("说话者" + n); }

  // ── 编辑历史（撤销/重做，快照栈）与字幕搜索替换状态 ─────────────────
  //   快照 = { segs, names }（JSON 深拷贝）；push 存「编辑前」状态，栈深 50。
  //   各编辑入口（拆分/合并/行文本/说话者切换与改名/搜索替换）改动前先 pushHistory()。
  const _editHistory = STTUtil.createHistory(50);

  function snapshotState() {
    return {
      segs: JSON.parse(JSON.stringify(state.curSegs)),
      names: JSON.parse(JSON.stringify(state.speakerNames)),
    };
  }
  function pushHistory() {
    if (!state.curSegs.length) return;
    _editHistory.push(snapshotState());
    syncUndoRedoButtons();
  }
  // 恢复快照：深拷贝回 state.curSegs/state.speakerNames（栈内快照与现行数据不共享引用），
  // 退出合并模式（选择索引已失效）、关弹出菜单后全量重绘（清 _karaCache/选中态）。
  function applyHistoryState(st) {
    if (state.mergeMode) setMergeMode(false, false);
    closeSpkMenu();
    state.curSegs = JSON.parse(JSON.stringify(st.segs));
    state.speakerNames = JSON.parse(JSON.stringify(st.names || {}));
    rerenderAfterEdit();
    syncUndoRedoButtons();
  }
  function doUndo() {
    if (!state.curSegs.length) return;
    const r = _editHistory.undo(snapshotState());
    if (r) applyHistoryState(r.state);
  }
  function doRedo() {
    if (!state.curSegs.length) return;
    const r = _editHistory.redo(snapshotState());
    if (r) applyHistoryState(r.state);
  }
  function syncUndoRedoButtons() {
    const u = $("#btn-undo"), r = $("#btn-redo");
    if (!u || !r) return;
    u.disabled = !_editHistory.canUndo();
    r.disabled = !_editHistory.canRedo();
  }

  // ── 段落编辑：拆分（编辑态光标处）与合并（相邻同说话者选择模式）────────
  // 播放高亮用的元素缓存与位置提示（长音频数千段时避免每帧 O(n) DOM 查询）
  let _subEls = [], _blockEls = [], _playingIdx = -1, _playedBars = -1, _segHint = 0;

  // 定位 t 所在段：先看上次位置附近（播放连续），未命中则二分。
  // 返回段索引；间隙（无段覆盖）回 -1。
  function segIndexAt(t) {
    const n = state.curSegs.length;
    if (!n) return -1;
    for (const i of [_segHint, _segHint - 1, _segHint + 1]) {
      if (i >= 0 && i < n && t >= state.curSegs[i].start && t < state.curSegs[i].end) {
        _segHint = i; return i;
      }
    }
    let lo = 0, hi = n - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1, s = state.curSegs[mid];
      if (t < s.start) hi = mid - 1;
      else if (t >= s.end) lo = mid + 1;
      else { _segHint = mid; return mid; }
    }
    return -1;
  }

  async function renderResult(segments) {
    $("#result").hidden = false;
    state.curSegs = segments;
    state.speakerNames = {};                 // 新结果 → 说话者名称重置
    setMergeMode(false);               // 新结果 → 退出合并选择模式
    _editHistory.reset();              // 新结果 → 编辑历史清空
    syncUndoRedoButtons();
    resetSearch();                     // 新结果 → 搜索状态与输入框清空
    _karaCache = null;                      // 新结果 → 重建卡拉OK字级（段落索引已变）
    _vwin = null;                           // 新结果 → 丢弃虚拟窗口模型（高度全部重估）
    _subEls = []; _blockEls = []; _playingIdx = -1; _playedBars = -1; _segHint = 0;
    state.selRange = null;                     // 新结果 → 片段选中态重置
    cleanupAudio();

    const file = picked && (picked.file || (picked instanceof Blob ? picked : null));
    let peaks = null;
    waveDur = segments.length ? segments[segments.length - 1].end : 0;

    if (file instanceof Blob) {
      audioUrl = URL.createObjectURL(file);
      audioEl = new Audio(audioUrl);
      let ctx = null;                         // decode 失败也须 close（AudioContext 有数量上限）
      try {                                   // 解码取真实峰值（mp3/wav/m4a/webm…）
        const ab = await file.arrayBuffer();
        ctx = new (window.AudioContext || window.webkitAudioContext)();
        const buf = await ctx.decodeAudioData(ab.slice(0));
        waveDur = buf.duration || waveDur;
        peaks = computePeaks(buf, WAVE_N);
      } catch (e) { peaks = null; }           // 不支持的格式 → 平波形，但仍可播
      finally { try { if (ctx) ctx.close(); } catch (e) {} }   // 成功/失败都关：泄漏修复
    }
    waveDur = waveDur || 60;

    $("#wave-dur").textContent = STTUtil.fmtClock(waveDur);
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
    state.curSegs.forEach(s => { if (s.speaker && !nums.includes(s.speaker)) nums.push(s.speaker); });
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
  // 存入 state.speakerNames 后重渲染全部说话者视图——字幕卡、时间轴区块、标签栏。
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
        pushHistory();                        // 改名写 state.speakerNames 前记录快照
        state.speakerNames[n] = val;
        _playingIdx = -1;                     // 重渲染后高亮状态重建
        rerenderForRename(n);                 // 全部同名 chip 卡局部更新（≤阈值仍全量）
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
  // 颜色、标签栏都随 state.curSegs 自动同步。
  function spkSwitchMenu(idx, chipEl) {
    closeSpkMenu();                                  // 同时最多一个
    const cur = state.curSegs[idx].speaker;
    const nums = [];
    state.curSegs.forEach(s => { if (s.speaker && !nums.includes(s.speaker)) nums.push(s.speaker); });
    nums.sort((a, b) => a - b);

    const menu = document.createElement("div");
    menu.className = "spk-menu";
    nums.forEach(n => {
      const row = document.createElement("button");
      row.type = "button";
      if (n === cur) row.classList.add("cur");
      row.innerHTML = `<span class="dot spk-${((n - 1) % 3) + 1}"></span>`
        + `<span class="nm">${STTUtil.escapeHtml(spkDisplay(n))}</span>`
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
    const old = spkDisplay(state.curSegs[idx].speaker);
    pushHistory();                                    // 改 speaker 前记录快照
    state.curSegs[idx].speaker = n;
    _playingIdx = -1;                                 // 重渲染后高亮状态重建
    rerenderForSpeakerSwitch(idx);                    // 单卡 chip 局部更新（≤阈值仍全量）
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

  // 拆分一段；成功返回 true，拆分点无效（null/首/尾/太短）提示后返回 false。
  // opts.deferRender：只改数据不渲染（拆分模式用——先解绑编辑会话监听，
  // 再由调用方统一重绘，避免闭包钉住已被替换的卡片 DOM）。
  function splitSegment(idx, full, pos, opts) {
    if (pos == null) { logLine("⚠ " + T("edit.splitNeedCaret", "请先点击文本放置光标，再拆分")); return false; }
    const seg = state.curSegs[idx];
    if (!seg) return false;
    // 切点时间：无字级 → 按字符比例；有字级 → 比例值吸附到最近的字边界
    let t = seg.start + (seg.end - seg.start) * (pos / Math.max(1, full.length));
    let charCut = pos;                     // 文字切点（full 的字符偏移）
    let wordsL = null, wordsR = null;      // 有字级且对齐成功时按边界精确分派
    const words = seg.words || [];
    if (words.length) {
      let best = t, bestD = Infinity, bestB = null, bestW = null;
      for (const w of words) for (const b of [w.start, w.end]) {
        const d = Math.abs(b - t);
        if (d < bestD) { bestD = d; best = b; bestB = b; bestW = w; }
      }
      t = Math.max(seg.start + 0.05, Math.min(seg.end - 0.05, best));
      if (t <= seg.start || t >= seg.end) {   // 过短段落吸附后退化 → 拒绝拆分
        logLine("⚠ " + T("edit.splitTooShort", "此段太短，无法拆分"));
        return false;
      }
      // 文字切点跟到同一个字边界（words 文本在 full 里顺序对齐），两段的
      // text 与 words 完全一致 → 卡拉OK/ASS \k 不串行；对不齐则退回
      // 「比例切点 + 字中点分派」的旧逻辑。
      const offs = _alignWordOffsets(full, words);
      if (offs && offs.has(bestW)) {
        const o = offs.get(bestW);
        const wi = words.indexOf(bestW);
        const cut = (bestB === bestW.start) ? wi : wi + 1;   // 边界前的词归左
        charCut = (bestB === bestW.start) ? o.a : o.b;
        wordsL = words.slice(0, cut);
        wordsR = words.slice(cut);
      }
    }
    const L = full.slice(0, charCut).trim(), R = full.slice(charCut).trim();
    if (!L || !R) { logLine("⚠ " + T("edit.splitEmpty", "拆分点两侧都需要有文字")); return false; }
    pushHistory();                       // splice 重建数组前记录快照（校验通过才入栈）
    if (wordsL) {
      state.curSegs.splice(idx, 1,
        { start: seg.start, end: t, speaker: seg.speaker, text: L, words: wordsL },
        { start: t, end: seg.end, speaker: seg.speaker, text: R, words: wordsR });
    } else {
      state.curSegs.splice(idx, 1,
        { start: seg.start, end: t, speaker: seg.speaker, text: L,
          words: words.filter(w => (w.start + w.end) / 2 < t) },
        { start: t, end: seg.end, speaker: seg.speaker, text: R,
          words: words.filter(w => (w.start + w.end) / 2 >= t) });
    }
    if (opts && opts.deferRender) return true;   // 渲染延后：doSplitNow 统一收尾
    rerenderForSplit(idx);
    logLine("✓ " + T("edit.splitDone", "已拆分为 2 段"));
    return true;
  }

  // words 文本在 full 里的顺序对齐：逐词 indexOf（扫描位置只前进）。
  // full 常含 words 没有的标点/空白；任一词对不上 → 返回 null（放弃对齐）。
  function _alignWordOffsets(full, words) {
    const map = new Map();
    let scan = 0;
    for (const w of words) {
      const wt = String(w.text || "");
      if (!wt) continue;
      const i = full.indexOf(wt, scan);
      if (i < 0) return null;
      map.set(w, { a: i, b: i + wt.length });
      scan = i + wt.length;
    }
    return map;
  }

  // ── 合并：相邻同说话者的多段 ────────────────────────────────────────
  function toggleMergeSel(idx, cardEl) {
    if (state.mergeSel.has(idx)) state.mergeSel.delete(idx);
    else state.mergeSel.add(idx);
    cardEl.classList.toggle("msel", state.mergeSel.has(idx));
    updateMergeBar();
  }

  function updateMergeBar() {
    const bar = $("#merge-bar"), n = state.mergeSel.size;
    if (!bar) return;
    $("#merge-count").textContent = T("merge.selCount", "已选 {n} 段 · 需相邻且同一说话者", { n });
    $("#merge-ok").disabled = n < 2;
  }

  function setMergeMode(on, render = true) {
    if (state.mergeMode === on) return;   // 状态未变 → 免去重复的全量重绘
    state.mergeMode = on;
    state.mergeSel = new Set();
    const bar = $("#merge-bar"), btn = $("#btn-merge-mode"), sb = $("#search-bar");
    if (sb) sb.hidden = on;   // 合并选择模式 → 隐藏搜索区（避免点卡片选段与搜索跳转冲突）
    if (btn) { btn.classList.toggle("btn-primary", on); btn.classList.toggle("btn-ghost", !on); }
    if (bar) bar.hidden = !on;
    if (on) updateMergeBar();
    if (!render) { syncStickyOffset(); return; }   // 稍后由调用方统一重绘
    // 卡片重绘以清除/应用选择态，并让点卡片行为切换
    // （_playingIdx 先复位再渲染：虚拟模式挂载新卡时按当前值套用 playing）
    _playingIdx = -1;
    renderSubs(state.curSegs);
    syncStickyOffset();    // 操作栏显隐改变面板高度 → 滚动吸附偏移同步更新
  }

  function mergeSelectedSegments() {
    const idxs = [...state.mergeSel].sort((a, b) => a - b);
    if (idxs.length < 2) { logLine("⚠ " + T("merge.needTwo", "请至少选择 2 段")); return; }
    // 仅允许「相邻 + 同一说话者」的多段合并：跨说话者（对白归一人属误导）
    // 与跨句子（时间轴与中间未选段重叠）均拒绝。
    for (let i = 1; i < idxs.length; i++) {
      if (idxs[i] !== idxs[i - 1] + 1) { logLine("⚠ " + T("merge.notAdjacent", "所选段落必须相邻")); return; }
    }
    const segs = idxs.map(i => state.curSegs[i]);
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
    pushHistory();                       // 合并改写前记录快照
    state.curSegs.splice(idxs[0], idxs.length, merged);
    setMergeMode(false, false);        // 只复位状态与操作栏；重绘交给下面一次完成
    rerenderForMerge(idxs[0], idxs.length);
    logLine("✓ " + T("merge.done", "已合并 {n} 段", { n: idxs.length }));
  }

  // ── 重绘分发：全量 vs 编辑脏区 ──────────────────────────────────────
  //   rerenderAfterEdit 是全量路径：撤销/重做恢复、新转录、搜索重置，
  //   以及段数 ≤ RENDER_THRESHOLD（稠密模式）的一切编辑 —— 与历史行为一致。
  //   虚拟模式（段数 > 阈值）下，调用点已知的受影响 idx 走 rerenderForXxx：
  //   只替换受影响卡片的外层 HTML（renderCardHtml 单一模板来源），容器其余
  //   节点不动；结构变更（拆分/合并）后平移已挂载卡的 data-idx 并校正高度模型。
  function rerenderAfterEdit() {
    if (state.search.q) {                      // 文本可能已变 → 匹配与高亮先重算（渲染要用）
      computeSearchMatches();
      if (state.search.idx >= state.search.matches.length) state.search.idx = state.search.matches.length - 1;
    }
    _karaCache = null;
    _playingIdx = -1;
    clearSel();                           // 时间轴已变 → 选中区间失效
    renderSpeakerBar();
    renderSegStrip(state.curSegs, waveDur);
    renderSubs(state.curSegs);
  }

  // 拆分后：idx 处 1 段变 2 段，其后索引 +1
  function rerenderForSplit(idx) {
    if (!_virtualMode) { rerenderAfterEdit(); return; }
    if (state.search.q) {
      computeSearchMatches();
      if (state.search.idx >= state.search.matches.length) state.search.idx = state.search.matches.length - 1;
    }
    _karaCache = null;
    _playingIdx = -1;
    clearSel();
    renderSegStrip(state.curSegs, waveDur);     // 块数已变：区块全量重建（点击已委托，无逐块监听成本）
    _vwin.spliceAt(idx, 1, [null, null]); // 高度模型：1 → 2（先估算占位，挂载后实测校正）
    bumpMountedIdx(idx + 1, +1);          // 其后已挂载卡 data-idx +1（节点不动）
    patchCard(idx);                       // 被拆卡 → 左半
    patchCard(idx + 1);                   // 右半（未挂载则由 mountWindow 补挂）
    mountWindow();
    updateSpacers();
  }

  // 合并后：firstIdx 起 nMerged 段并为 1 段，其后索引 -(nMerged-1)
  function rerenderForMerge(firstIdx, nMerged) {
    if (!_virtualMode) { rerenderAfterEdit(); return; }
    if (state.search.q) {
      computeSearchMatches();
      if (state.search.idx >= state.search.matches.length) state.search.idx = state.search.matches.length - 1;
    }
    _karaCache = null;
    _playingIdx = -1;
    clearSel();
    renderSegStrip(state.curSegs, waveDur);
    for (let i = firstIdx + 1; i < firstIdx + nMerged; i++) unmountCard(i);  // 被吞卡卸载
    bumpMountedIdx(firstIdx + nMerged, -(nMerged - 1));
    $$("#subs .sub-card.msel").forEach(el => el.classList.remove("msel"));   // 选择态已清
    _vwin.spliceAt(firstIdx, nMerged, [null]);
    patchCard(firstIdx);                  // 保留卡 → 合并内容
    mountWindow();
    updateSpacers();
  }

  // 说话者改名：全部同名 chip 卡 + 同名区块 + 标签栏
  function rerenderForRename(n) {
    if (!_virtualMode) {
      renderSpeakerBar();
      renderSegStrip(state.curSegs, waveDur);
      renderSubs(state.curSegs);
      return;
    }
    renderSpeakerBar();
    for (let i = 0; i < state.curSegs.length; i++) {
      if (state.curSegs[i].speaker !== n) continue;
      patchSegBlock(i);
      if (_subEls[i]) patchCard(i);
    }
  }

  // 单行说话者切换：该行卡 chip + 对应区块 + 标签栏
  function rerenderForSpeakerSwitch(idx) {
    if (!_virtualMode) {
      renderSpeakerBar();
      renderSegStrip(state.curSegs, waveDur);
      renderSubs(state.curSegs);
      return;
    }
    renderSpeakerBar();
    patchSegBlock(idx);
    patchCard(idx);
  }

  // 搜索激活时的行文本编辑 / 替换：匹配已随新文本重算，卡片整窗口重挂
  //（窗口有界 → 有界成本），区块按受影响 idx 就地更新
  function rerenderForEditedText(idx) {
    if (!_virtualMode) { rerenderAfterEdit(); return; }
    computeSearchMatches();
    if (state.search.idx >= state.search.matches.length) state.search.idx = state.search.matches.length - 1;
    _karaCache = null;
    _playingIdx = -1;
    clearSel();
    patchSegBlock(idx);
    rerenderMounted();
  }
  function rerenderForReplace(affectedIdxs) {
    if (!_virtualMode) { rerenderAfterEdit(); return; }
    _karaCache = null;
    _playingIdx = -1;
    clearSel();
    patchSegBlocks(affectedIdxs);
    rerenderMounted();
  }

  // ── 编辑脏区 patch 原语 ─────────────────────────────────────────────
  // 替换一张已挂载卡的外层 HTML（renderCardHtml 为唯一模板来源）
  function patchCard(idx) {
    const old = _subEls[idx];
    if (!old || old.classList.contains("editing")) return;   // 编辑中的卡由编辑收尾负责
    const tmp = document.createElement("template");
    tmp.innerHTML = renderCardHtml(idx);
    const fresh = tmp.content.firstElementChild;
    if (state.selRange && state.selRange.idx === idx) fresh.classList.add("sel");
    if (_playingIdx === idx) fresh.classList.add("playing");
    old.replaceWith(fresh);
    _subEls[idx] = fresh;
    measureCard(idx, fresh);
    if (_virtualMode) updateSpacers();
  }
  // 结构变更后平移已挂载卡的 data-idx / 数组槽位（DOM 节点本身不动）
  function bumpMountedIdx(fromIdx, delta) {
    const keys = Object.keys(_subEls).map(Number)
      .filter(i => i >= fromIdx)
      .sort((a, b) => (delta > 0 ? b - a : a - b));   // 与平移反向遍历，避免槽位覆盖
    for (const i of keys) {
      const el = _subEls[i];
      delete _subEls[i];
      _subEls[i + delta] = el;
      el.dataset.idx = String(i + delta);
      el.dataset.seg = String(i + delta);
    }
  }
  // 重挂载全部已挂载卡（索引不变、内容重算：搜索高亮/替换后用；保留滚动位置）
  function rerenderMounted() {
    if (!_virtualMode) return;
    for (const key of Object.keys(_subEls)) unmountCard(+key);
    mountWindow();
    updateSpacers();
  }
  // 单个 seg-block 的文字/标题/说话者色就地更新
  function patchSegBlock(idx) {
    const blk = $(`.seg-block[data-seg="${idx}"]`);
    const s = state.curSegs[idx];
    if (!blk || !s) return;
    blk.textContent = s.text;
    blk.title = (s.speaker ? spkDisplay(s.speaker) + "：" : "") + s.text;
    blk.className = "seg-block" + (s.speaker ? " spk-" + (((s.speaker - 1) % 3) + 1) : "");
    if (state.selRange && state.selRange.idx === idx) blk.classList.add("sel");
    if (_playingIdx === idx) blk.classList.add("playing");
  }
  function patchSegBlocks(idxs) { for (const i of idxs || []) patchSegBlock(i); }

  // 波形面板为 sticky 钉在顶端 → 滚动容器需保留等高的顶部空间，
  // 否则 scrollIntoView 自动卷到的字幕卡会被面板遮住。
  // _stickyPad 供虚拟模式的估算定位（ensureCardVisible）复用同一偏移。
  let _stickyPad = 0;
  function syncStickyOffset() {
    const panel = $(".wave-panel"), host = $(".view-host");
    if (!panel || !host) return;
    _stickyPad = panel.offsetHeight + 12;
    host.style.scrollPaddingTop = _stickyPad + "px";
  }

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
  //   点击走 #seg-strip 容器委托（数千段时不再逐块挂 listener）。
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
      host.appendChild(blk);
      _blockEls.push(blk);
    });
    if (state.selRange && _blockEls[state.selRange.idx]) _blockEls[state.selRange.idx].classList.add("sel");
  }

  // ══ 字幕卡渲染：稠密（≤ RENDER_THRESHOLD）与虚拟窗口（> 阈值）双路径 ══
  //   稠密：一次挂载全部卡片（与历史行为完全一致）。
  //   虚拟：只挂载可视窗口 ± 缓冲（窗口高 ×2）的卡片，上下 spacer 撑高度；
  //   窗口计算与高度模型在 STTUtil.createVirtualWindow（Node 可单测），
  //   滚动经 rAF 节流差量挂载/卸载；模型高度含卡片间距（实测 offsetHeight+GAP）。
  const RENDER_THRESHOLD = 400;   // 段数分叉阈值：≤ 走全量路径，> 走虚拟窗口
  const SUB_GAP = 7;              // 卡片间距：与 css .subs gap / .subs.virtual margin 等值
  const SUB_DEFAULT_H = 64;       // 未实测段的估算卡高（含间距）
  let _virtualMode = false;       // 当前字幕列表是否虚拟窗口模式
  let _vwin = null;               // STTUtil.createVirtualWindow 实例（虚拟模式）
  let _spacerTop = null, _spacerBot = null;
  let _scrollHost = null, _scrollRaf = 0, _measureRaf = 0;

  function renderSubs(segments) {
    const host = $("#subs");
    const virtual = segments.length > RENDER_THRESHOLD;
    if (virtual !== _virtualMode) {           // 模式切换 → 重建窗口模型
      host.classList.toggle("virtual", virtual);
      if (!virtual) _vwin = null;
      _virtualMode = virtual;
    }
    host.innerHTML = "";
    _subEls = [];
    _spacerTop = _spacerBot = null;
    if (!virtual) {
      segments.forEach((s, i) => { const el = subCard(s, i); host.appendChild(el); _subEls.push(el); });
    } else {
      if (!_vwin || _vwin.count !== segments.length) {
        _vwin = STTUtil.createVirtualWindow({
          count: segments.length,
          viewport: subsViewport(),
          buffer: subsViewport() * 2,          // 窗口高 ×2 缓冲
          defaultHeight: SUB_DEFAULT_H,
        });
      } else {
        _vwin.setCount(segments.length);       // 段数不变的全量重绘 → 保留实测高度
      }
      _spacerTop = document.createElement("div"); _spacerTop.className = "v-spacer";
      _spacerBot = document.createElement("div"); _spacerBot.className = "v-spacer";
      host.appendChild(_spacerTop);
      host.appendChild(_spacerBot);
      ensureScrollWiring();
      mountWindow();
    }
    if (state.selRange && _subEls[state.selRange.idx]) _subEls[state.selRange.idx].classList.add("sel");
  }

  // 单卡外层 HTML（模板唯一来源：subCard 初挂与 patchCard 脏区替换都取这里）。
  // 卡片带 data-idx（段索引），动作按钮带 data-act（spk/edit/split）供委托分发。
  function renderCardHtml(idx) {
    const s = state.curSegs[idx];
    // spkc-N：依说话者标示左侧外框色（与 chip 同色系，不污染卡片背景）
    const cls = "sub-card" + (s.speaker ? " spkc-" + (((s.speaker - 1) % 3) + 1) : "")
      + (state.mergeSel.has(idx) ? " msel" : "");
    const spk = s.speaker ? `<span class="chip spk-${((s.speaker - 1) % 3) + 1}" data-act="spk" title="${T("file.spkSwitchHint", "点击切换此行的说话者")}">${STTUtil.escapeHtml(spkDisplay(s.speaker))}</span>` : "";
    return `<div class="${cls}" data-idx="${idx}" data-seg="${idx}"><span class="tc">${STTUtil.fmtClock(s.start)} → ${STTUtil.fmtClock(s.end)}</span>
      <div class="body"><div class="spk">${spk}</div><div class="txt">${segTextHtml(s, idx)}</div></div>
      <button class="sub-edit" data-act="edit" title="${T("edit.line", "编辑此行")}" aria-label="${T("edit.line", "编辑此行")}">
        <svg class="ico-pencil" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
        <svg class="ico-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
      </button>
      <button class="sub-split" data-act="split" title="${T("edit.split", "拆分此行：进入编辑后，在光标处点亮的 √ 切开（Ctrl+Enter）")}">
        <svg class="ico-scissors" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.12 15.88"/><path d="M14.47 14.48 20 20"/><path d="M8.12 8.12 12 12"/></svg>
        <svg class="ico-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
      </button></div>`;
  }
  // 由模板构建卡片元素。不挂任何 listener —— 交互全部走 #subs 上的委托分发。
  function subCard(s, idx) {
    const t = document.createElement("template");
    t.innerHTML = renderCardHtml(idx);
    return t.content.firstElementChild;
  }

  // ── 虚拟窗口 DOM 粘合 ────────────────────────────────────────────────
  function subsViewport() {
    const host = _scrollHost || document.querySelector(".view-host");
    return Math.max(120, host ? host.clientHeight : 600);
  }
  function ensureScrollWiring() {
    const host = document.querySelector(".view-host");
    if (!host) return;
    if (_scrollHost !== host) {
      if (_scrollHost) _scrollHost.removeEventListener("scroll", onHostScroll);
      _scrollHost = host;
      host.addEventListener("scroll", onHostScroll, { passive: true });
    }
  }
  function onHostScroll() {
    if (!_virtualMode || !_vwin) return;
    if (_scrollRaf) return;
    _scrollRaf = requestAnimationFrame(() => { _scrollRaf = 0; mountWindow(); });
  }
  // #subs 顶边相对滚动容器内容顶部的偏移（上方还有拖放区/控制区/波形面板）
  function subsTopOffset() {
    const subs = $("#subs"), host = _scrollHost;
    if (!subs || !host) return 0;
    const r1 = subs.getBoundingClientRect(), r2 = host.getBoundingClientRect();
    return Math.max(0, r1.top - r2.top + host.scrollTop);
  }
  function mountWindow() {
    if (!_virtualMode || !_vwin) return;
    const host = _scrollHost;
    if (!host) return;
    const st = Math.max(0, host.scrollTop - subsTopOffset());
    const range = _vwin.rangeForScroll(st);
    for (const key of Object.keys(_subEls)) {          // 卸载窗口外的卡
      const i = +key;
      if (i < range.first || i > range.last) unmountCard(i);
    }
    for (let i = range.first; i <= range.last; i++) {  // 补挂窗口内缺失的卡
      if (!_subEls[i]) mountCard(i);
    }
    updateSpacers();
  }
  function mountCard(idx) {
    if (!_vwin || !(idx >= 0 && idx < state.curSegs.length) || _subEls[idx]) return;
    const el = subCard(state.curSegs[idx], idx);
    if (state.selRange && state.selRange.idx === idx) el.classList.add("sel");
    if (_playingIdx === idx) el.classList.add("playing");
    let before = _spacerBot;                           // 插到第一个更大 idx 的已挂卡之前
    for (const key of Object.keys(_subEls)) {
      if (+key > idx) { before = _subEls[+key]; break; }
    }
    $("#subs").insertBefore(el, before);
    _subEls[idx] = el;
    measureCard(idx, el);
  }
  function unmountCard(idx) {
    const el = _subEls[idx];
    if (!el) return;
    el.remove();
    delete _subEls[idx];
  }
  function measureCard(idx, el) {
    if (!_virtualMode || !_vwin || !el) return;
    _vwin.setHeight(idx, el.offsetHeight + SUB_GAP);   // 模型高度含间距
  }
  function updateSpacers() {
    if (!_virtualMode || !_vwin || !_spacerTop || !_spacerBot) return;
    _spacerTop.style.height = Math.max(0, _vwin.offsetOf(_vwin.first)) + "px";
    _spacerBot.style.height = Math.max(0, _vwin.estimateTotal() - _vwin.offsetOf(_vwin.last + 1)) + "px";
  }
  function scheduleMeasure(idx) {
    if (!_virtualMode) return;
    if (_measureRaf) cancelAnimationFrame(_measureRaf);
    _measureRaf = requestAnimationFrame(() => {
      _measureRaf = 0;
      const el = _subEls[idx];
      if (el) { measureCard(idx, el); updateSpacers(); }
    });
  }

  // 滚动/跳转定位：未挂载段先按估算累计高度滚到附近（±50px 可接受），
  // 挂载后再用实测位置经 scrollIntoView 校正（scroll-padding-top 吸附已同步）。
  function ensureCardVisible(idx) {
    const el = _subEls[idx];
    if (!_virtualMode) { if (el) el.scrollIntoView({ block: "nearest" }); return; }
    if (!_vwin || !_scrollHost) return;
    if (!el) {
      const target = subsTopOffset() + _vwin.offsetOf(idx);
      _scrollHost.scrollTop = Math.max(0, target - _stickyPad - 8);
      mountWindow();
    }
    const el2 = _subEls[idx];
    if (el2) el2.scrollIntoView({ block: "nearest" });
  }

  // ── 事件委托：#subs 一组监听分派全部卡片交互（卡片零逐卡 listener）────
  //   动作表：data-act=spk → spkSwitchMenu；edit → beginEdit（编辑态 = 保存）；
  //   split → selectSeg + beginSplitEdit（编辑态 = 光标处切开）；
  //   无 act → selectSeg（选中区间）；合并模式下点击一律 toggleMergeSel。
  //   键盘（Enter/Esc/Ctrl+Enter）与 blur 收尾、拆分锁字（beforeinput/paste/
  //   cut/drop 拦截）同样经委托分发到 _editCtx 编辑会话。
  function wireSubsDelegation() {
    const host = $("#subs");
    if (!host || host.dataset.delegated) return;
    host.dataset.delegated = "1";
    host.addEventListener("click", onSubsClick);
    host.addEventListener("mousedown", onSubsMousedown);
    host.addEventListener("keydown", onSubsKeydown);
    host.addEventListener("focusout", onSubsFocusout);
    host.addEventListener("input", onSubsInput);
    host.addEventListener("beforeinput", onSubsBlockInput);
    host.addEventListener("paste", onSubsBlockInput);
    host.addEventListener("cut", onSubsBlockInput);
    host.addEventListener("drop", onSubsBlockInput);
  }
  function onSubsClick(e) {
    const cardEl = e.target.closest(".sub-card");
    if (!cardEl || !state.curSegs.length) return;
    const idx = +cardEl.dataset.idx;
    if (!(idx >= 0 && idx < state.curSegs.length)) return;
    const actEl = e.target.closest("[data-act]");
    const act = actEl && actEl.closest(".sub-card") === cardEl ? actEl.dataset.act : "";
    const ctx = _editCtx;
    // 编辑态的 √（笔/剪刀在编辑模式下变为确认钮）：原为 per-session click 监听
    if (ctx && ctx.card === cardEl && act === "edit" && ctx.mode === "edit") { finishEdit(true); return; }
    if (ctx && ctx.card === cardEl && act === "split" && ctx.mode === "split") { doSplitNow(); return; }
    if (state.mergeMode) { toggleMergeSel(idx, cardEl); return; }   // 合并模式：点卡＝选择
    if (cardEl.classList.contains("editing")) return;          // 编辑中：卡片不响应跳播
    const chipEl = e.target.closest(".chip");
    if (chipEl && state.curSegs[idx].speaker && !chipEl.classList.contains("editing")) {
      spkSwitchMenu(idx, chipEl);      // 点说话者 chip → 弹切换菜单；改名中的 chip 不响应
      return;
    }
    if (act === "edit") { beginEdit(cardEl, idx); return; }    // 笔 → 文字编辑
    if (act === "split") {                                     // 剪刀 → 拆分模式
      if (cardEl.classList.contains("editing")) return;
      selectSeg(idx);   // 原行为：剪刀与卡片同元素双监听，两处理都触发（选中 + 进拆分）
      beginSplitEdit(cardEl, idx);
      logLine("ℹ " + T("edit.splitHow", "点击文本中要拆分的位置，再点亮的 √ 或按 Ctrl+Enter"));
      return;
    }
    selectSeg(idx);                    // 点行 → 选中该句区间（不播放；播放键循环）
  }
  function onSubsMousedown(e) {
    // 笔/剪刀钮 mousedown preventDefault 保住焦点，不让 blur 提前收起模式
    if (e.target.closest(".sub-edit, .sub-split")) e.preventDefault();
  }
  function onSubsKeydown(e) {
    const ctx = _editCtx;
    if (!ctx || e.target !== ctx.txt) return;                  // 仅行内编辑的 .txt 消费键盘
    if (ctx.mode === "edit") {
      if (e.key === "Enter") { e.preventDefault(); finishEdit(true); }
      else if (e.key === "Escape") { e.preventDefault(); finishEdit(false); }
    } else {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doSplitNow(); }
      else if (e.key === "Escape") { e.preventDefault(); finishSplit(); }
    }
  }
  function onSubsFocusout(e) {
    const ctx = _editCtx;
    if (!ctx || e.target !== ctx.txt) return;
    if (ctx.mode === "edit") finishEdit(true); else finishSplit();
  }
  function onSubsInput(e) {
    const ctx = _editCtx;
    // 编辑中的行文本变化 → 该卡高度可能变化，虚拟模式下校正高度模型
    if (ctx && ctx.mode === "edit" && e.target === ctx.txt) scheduleMeasure(ctx.idx);
  }
  function onSubsBlockInput(e) {
    // 拆分模式锁字：键盘输入/粘贴/剪切/拖放一律拦截（IME 组字由
    // MutationObserver 兜底，见 beginSplitEdit）
    const ctx = _editCtx;
    if (ctx && ctx.mode === "split" && e.target === ctx.txt) e.preventDefault();
  }

  // 行内编辑两种互斥模式（同一时刻一张卡只能处其一）：
  //   文字编辑（笔进入）：笔按钮变 √（保存，等同 Enter）；只能改字，无拆分入口。
  //   拆分模式（剪刀进入）：剪刀按钮变 √（光标就绪点亮，点击切开）；文字锁定
  //   不可改（contentEditable 仅为放光标，beforeinput/paste/cut/drop 经 #subs
  //   委托拦截 + MutationObserver 兜底）。
  // 两模式的 Esc 退出、blur（focusout 委托）收尾、Enter（编辑＝保存）行为一致。
  // 编辑会话状态 _editCtx：键盘/焦点/按钮事件全部由 #subs 委托分发到这里，
  // 会话自身只保留必要的 per-session 资源（拆分模式的 Observer 与 selectionchange）。
  let _editCtx = null;   // {mode:"edit"|"split", card, idx, txt, btn, orig, done, mo, updateReady}

  function beginEdit(card, idx) {
    if (card.classList.contains("editing")) return;
    finishActiveEdit();                       // 已有另一卡在编辑 → 先收尾（防卡死编辑态）
    card.classList.add("editing", "mode-edit");
    const txt = card.querySelector(".txt");
    const editBtn = card.querySelector(".sub-edit");
    const orig = state.curSegs[idx].text;
    txt.contentEditable = "true"; txt.spellcheck = false;
    txt.focus();
    const range = document.createRange(); range.selectNodeContents(txt);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    editBtn.title = T("edit.save", "保存（Enter）");
    _editCtx = { mode: "edit", card, idx, txt, btn: editBtn, orig, done: false };
  }
  function finishEdit(save) {
    const ctx = _editCtx;
    if (!ctx || ctx.mode !== "edit" || ctx.done) return;
    ctx.done = true;
    _editCtx = null;
    const { card, idx, txt, btn, orig } = ctx;
    txt.contentEditable = "false";
    card.classList.remove("editing", "mode-edit");
    btn.title = T("edit.line", "编辑此行");
    const val = (txt.textContent || "").trim();
    if (save && val && val !== orig) {
      pushHistory();                                        // 改文本前记录快照
      state.curSegs[idx].text = val; txt.textContent = val;
      state.curSegs[idx].words = reflowWords(state.curSegs[idx], val);  // 卡拉OK字级跟著修正
      _karaCache = null;                                    // 失效缓存 → 下次重绘该行
      const blk = $(`.seg-block[data-seg="${idx}"]`);
      if (blk) { blk.textContent = val; blk.title = val; }
      if (state.search.q) rerenderForEditedText(idx);            // 搜索中 → 高亮随新文本重算
      else scheduleMeasure(idx);                            // 虚拟模式校正该卡实测高度
    } else {
      txt.textContent = orig;               // 取消或清空 → 还原
    }
    txt.blur();
  }

  // 拆分模式：只能放光标/选区，文字不可编辑；√ 随光标就绪点亮，点击切开。
  function beginSplitEdit(card, idx) {
    if (card.classList.contains("editing")) return;
    finishActiveEdit();
    card.classList.add("editing", "mode-split");
    const txt = card.querySelector(".txt");
    const splitBtn = card.querySelector(".sub-split");
    // 光标需要 contentEditable，但拆分模式禁止改字：输入事件（键盘/粘贴/
    // 剪切/拖放）经 #subs 委托统一拦截（onSubsBlockInput），再以
    // MutationObserver 兜底——IME 组合输入等路径可能绕过 beforeinput，
    // 任何改动一律强制还原为原文。
    const origText = txt.textContent;
    const mo = new MutationObserver(() => {
      if (txt.textContent !== origText) { txt.textContent = origText; updateReady(); }
    });
    txt.contentEditable = "true"; txt.spellcheck = false;
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
    _editCtx = { mode: "split", card, idx, txt, btn: splitBtn, origText, done: false, mo, updateReady };
  }
  function finishSplit() {
    const ctx = _editCtx;
    if (!ctx || ctx.mode !== "split" || ctx.done) return;
    ctx.done = true;
    _editCtx = null;
    ctx.mo.disconnect();
    document.removeEventListener("selectionchange", ctx.updateReady);
    ctx.btn.classList.remove("ready");
    ctx.txt.contentEditable = "false";
    ctx.card.classList.remove("editing", "mode-split");
    ctx.txt.blur();
  }
  function doSplitNow() {
    const ctx = _editCtx;
    if (!ctx || ctx.mode !== "split" || ctx.done) return;
    const { idx, txt } = ctx;
    const full = txt.textContent || "";
    const pos = caretOffsetIn(txt, "start");
    // 成功 → deferRender 只改数据，先 finishSplit() 解绑（拆分成功的重渲染会
    // 替换/移除聚焦元素，必须先解绑 selectionchange 监听与 MutationObserver，
    // 否则每次拆分泄漏一套闭包钉住已脱离的卡片 DOM），再统一补渲染。
    if (splitSegment(idx, full, pos, { deferRender: true })) {
      finishSplit();
      rerenderForSplit(idx);
      logLine("✓ " + T("edit.splitDone", "已拆分为 2 段"));
      return;
    }
    txt.classList.remove("split-hint");   // 失败 → 闪红提示，留在拆分模式
    void txt.offsetWidth;
    txt.classList.add("split-hint");
  }
  // 已有编辑会话在别卡 → 按模式收尾（保存文字 / 退出拆分）
  function finishActiveEdit() {
    if (!_editCtx) return;
    if (_editCtx.mode === "edit") finishEdit(true); else finishSplit();
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
        if (state.selRange && (audioEl.currentTime < state.selRange.start
                         || audioEl.currentTime >= state.selRange.end - 0.02)) {
          audioEl.currentTime = state.selRange.start;
        }
        audioEl.play();
      } else audioEl.pause();
    };
    // 停止：暂停并回到片段头（无选中则回开头）
    const stop = $("#wave-stop");
    if (stop) stop.onclick = () => {
      if (!audioEl) return;
      audioEl.pause();
      audioEl.currentTime = state.selRange ? state.selRange.start : 0;
      onTime();
    };

    function onTime() {
      // 片段循环：播过区间尾 → 回区间头（timeupdate ~4fps，短片段也能回卷）
      if (state.selRange && !audioEl.paused && audioEl.currentTime >= state.selRange.end - 0.02) {
        audioEl.currentTime = state.selRange.start;
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
          if (playFollow) ensureCardVisible(cur);   // 虚拟模式：估算定位 + 实测校正
        }
        _playingIdx = cur;
      }
      const tm = $("#wave-cur"); if (tm) tm.textContent = STTUtil.fmtClock(t);
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
    const s = state.curSegs[idx];
    if (!s || !waveDur) return;
    state.selRange = { start: s.start, end: Math.max(s.end, s.start + 0.05), idx };
    let ov = $("#wave-sel");
    if (!ov) { ov = document.createElement("div"); ov.id = "wave-sel"; ov.className = "wave-sel"; $("#wave").appendChild(ov); }
    ov.style.left = ((state.selRange.start / waveDur) * 100).toFixed(2) + "%";
    ov.style.width = (Math.max(0.4, ((state.selRange.end - state.selRange.start) / waveDur) * 100)).toFixed(2) + "%";
    _subEls.forEach((el, i) => el.classList.toggle("sel", i === idx));
    _blockEls.forEach((el, i) => el.classList.toggle("sel", i === idx));
  }

  function clearSel() {
    state.selRange = null;
    const ov = $("#wave-sel"); if (ov) ov.remove();   // 摘除而非缩为 0 宽：
    _subEls.forEach(el => el.classList.remove("sel")); // border-box 下 0 宽仍画 2px 边框残线
    _blockEls.forEach(el => el.classList.remove("sel"));
  }
  function cleanupAudio() {
    try { if (audioEl) { audioEl.pause(); audioEl.src = ""; } } catch (e) {}
    if (audioUrl) { try { URL.revokeObjectURL(audioUrl); } catch (e) {} audioUrl = null; }
    audioEl = null;
  }

  // ════ 卡拉OK模式（逐字高亮放大跳动，由字级 words 驱动）════
  //   另开 requestAnimationFrame 循环（60fps）读 audioEl.currentTime，逐字更新，
  //   比 <audio> timeupdate（~4fps）顺得多。数据源：state.curSegs[].words（后端 FA 字级；
  //   无对齐时 wordsForSeg 以行时间平均内插）。
  let karaokeOn = false, karaokeRAF = 0, _karaCache = null;   // _karaCache: {idx, words, spans}

  // 卡拉OK增量渲染（游标式）：rAF 每帧不再遍历整行 span 写 class/style ——
  //   - sung（已唱填色）：游标 STTUtil.karaokeCursor 维护每段 sungIdx，
  //     前进只给新区间 [from..to] 加 .sung；回退/seek 整段清一次再补 0..target；
  //     游标纯逻辑在 util.js（Node 可单测）。多数帧游标不动 → 零 DOM 写。
  //   - active/scale（放大跳动 + 开唱前蓄力）：每帧最多涉及活跃字与其预备字
  //     2-3 枚 span，与上一帧触达集合（cache.fx）做差量更新，其余 span 不触碰。
  //   视觉映射保持不变：scale → translateY((sc-1)*-12) scale(sc)；曲线：
  //   起跳即最大随进度二次缓出（active）、开唱前 ~0.12s 微幅抬起蓄力（lead）。
  function _karaScaleActive(w, t) {
    // 起跳即最大、随进度二次缓出回落：唱到的瞬间「弹」一下再归位
    const dur = Math.max(0.05, w.end - w.start);
    const p = Math.min(1, (t - w.start) / dur);
    return 1 + 0.24 * (1 - p) * (1 - p);   // scale 上限 1.24 ≤ 1.3，整行不至抖动过猛
  }
  function _karaScaleLead(w, t) {
    // 预备动作：开唱前 ~0.12s 先微幅抬起蓄力（Apple Music 动态歌词风）
    const lead = w.start - t;
    return 1 + 0.06 * (1 - lead / 0.12);
  }

  // 每帧更新：定位目前段（提示位置 + 二分，见 segIndexAt）→（换段才重绘）→ 游标推进。
  function updateKaraoke(t) {
    if (!state.curSegs.length) return;
    let idx = segIndexAt(t);
    if (idx < 0) idx = _karaCache ? _karaCache.idx : 0;   // 空隙：停在上一段（唱完状态）
    if (!_karaCache || _karaCache.idx !== idx) {
      const built = renderKaraokeLine(idx);
      _karaCache = { idx, words: built.words, spans: built.spans,
                     cursor: STTUtil.karaokeCursor(built.words.length), fx: [] };
    }
    const cache = _karaCache, words = cache.words, spans = cache.spans;
    // 1) sung 游标：只切换变化区间的状态类
    const sungIdx = STTUtil.karaokeSungIndex(words, t, cache.cursor.index);
    const mv = cache.cursor.advance(sungIdx);
    if (mv) {
      if (mv.backward) {                                  // 回退/seek → 整段重置一次
        for (let i = 0; i < spans.length; i++) spans[i].classList.remove("sung");
      }
      for (let i = mv.from; i <= mv.to; i++) {
        if (spans[i]) spans[i].classList.add("sung");
      }
    }
    // 2) active/scale：活跃/预备字差量更新（游标之后的连续小窗口）
    const fresh = [];
    for (let i = sungIdx + 1; i < words.length; i++) {
      const w = words[i];
      if (w.start - t > 0.12) break;                      // 蓄力窗口之外 → 停止扫描
      const active = t >= w.start && t < w.end;
      const scale = active ? _karaScaleActive(w, t)
                  : (t < w.start ? _karaScaleLead(w, t) : 1);
      fresh.push({ i, active, scale });
      if (fresh.length >= 4) break;                       // 异常重叠 words 限幅
    }
    for (const f of cache.fx) {                           // 上一帧触达、本帧不再触达 → 清样式
      if (fresh.some(x => x.i === f.i)) continue;
      const sp = spans[f.i];
      if (sp) { sp.classList.remove("active"); sp.style.transform = ""; }
    }
    for (const f of fresh) {
      const sp = spans[f.i];
      if (!sp) continue;
      sp.classList.toggle("active", f.active);
      // scale→transform：放大同时微微上抬（跳动感）。视觉映射固定于此。
      sp.style.transform = f.scale === 1 ? "" : `translateY(${((f.scale - 1) * -12).toFixed(1)}px) scale(${f.scale.toFixed(3)})`;
    }
    cache.fx = fresh;
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
    const seg = state.curSegs[idx], line = $("#kara-line");
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
    const next = state.curSegs[idx + 1];
    $("#kara-next").textContent = next ? next.text : "";
    return { words, spans };
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

  // ── 字幕搜索替换（纯文本、大小写不敏感；高亮在渲染时包裹进卡片）─────
  //   matches 全局有序（segIdx → start），bySeg 供各卡片 O(1) 取自己的命中列表；
  //   k = 全局匹配序号，用于标出「当前匹配」（.search-hl.cur）。
  function computeSearchMatches() {
    const needle = state.search.q.toLowerCase();
    const matches = [], bySeg = new Map();
    if (needle) {
      state.curSegs.forEach((s, segIdx) => {
        const hay = String(s.text || "").toLowerCase();
        let from = 0;
        for (;;) {
          const i = hay.indexOf(needle, from);
          if (i < 0) break;
          const rec = { segIdx, start: i, length: needle.length };
          rec.k = matches.length;
          matches.push(rec);
          if (!bySeg.has(segIdx)) bySeg.set(segIdx, []);
          bySeg.get(segIdx).push(rec);
          from = i + needle.length;
        }
      });
    }
    state.search.matches = matches;
    state.search.bySeg = bySeg;
  }

  // 卡片 .txt 的 HTML：无搜索/无命中 → 原样转义；有命中 → 命中子串包高亮 span
  function segTextHtml(s, idx) {
    const list = state.search.bySeg && state.search.bySeg.get(idx);
    if (!list || !list.length) return STTUtil.escapeHtml(s.text);
    const t = String(s.text || "");
    let html = "", pos = 0;
    for (const h of list) {
      if (h.start < pos) continue;         // 重叠保护（同一查询扫描本不重叠）
      html += STTUtil.escapeHtml(t.slice(pos, h.start));
      html += `<span class="search-hl${h.k === state.search.idx ? " cur" : ""}">`
        + STTUtil.escapeHtml(t.slice(h.start, h.start + h.length)) + `</span>`;
      pos = h.start + h.length;
    }
    return html + STTUtil.escapeHtml(t.slice(pos));
  }

  function updateSearchCount() {
    const el = $("#search-count");
    if (el) el.textContent = !state.search.q ? ""
      : (state.search.matches.length ? (state.search.idx + 1) + "/" + state.search.matches.length : "0/0");
    const has = !!(state.search.q && state.search.matches.length);
    const b1 = $("#search-replace"), b2 = $("#search-replace-all");
    if (b1) b1.disabled = !has;
    if (b2) b2.disabled = !has;
  }

  // 只重绘一张卡的文本（↑↓ 跳转时新旧两张卡局部更新，免全量重绘）
  function refreshCardText(idx) {
    const el = _subEls[idx];
    const txt = el && el.querySelector(".txt");
    if (txt) txt.innerHTML = segTextHtml(state.curSegs[idx], idx);
  }

  // 循环跳到第 k 个匹配：更新当前索引 → 局部重绘新旧卡 → 滚动到当前卡
  function gotoMatch(k) {
    const n = state.search.matches.length;
    if (!n) return;
    const prev = state.search.idx >= 0 && state.search.idx < n ? state.search.matches[state.search.idx].segIdx : -1;
    state.search.idx = ((k % n) + n) % n;
    updateSearchCount();
    const cur = state.search.matches[state.search.idx].segIdx;
    if (prev >= 0 && prev !== cur) refreshCardText(prev);
    refreshCardText(cur);   // 同段相邻匹配也要重绘（.cur 在卡内移动）
    ensureCardVisible(cur); // 虚拟模式：未挂载先按估算滚动定位，再实测校正
  }
  function nextMatch() { gotoMatch(state.search.idx + 1); }
  function prevMatch() { gotoMatch(state.search.idx < 0 ? state.search.matches.length - 1 : state.search.idx - 1); }

  // 新结果/清除：清空搜索状态与输入框（高亮随下次渲染自然消失）
  function resetSearch() {
    state.search.q = ""; state.search.matches = []; state.search.bySeg = null; state.search.idx = -1;
    const q = $("#search-q"), r = $("#search-r");
    if (q) q.value = "";
    if (r) r.value = "";
    updateSearchCount();
  }
  // Esc：清空退出（清高亮、失焦），不影响其他快捷键
  function clearSearch() {
    resetSearch();
    const q = $("#search-q");
    if (q) q.blur();
    _playingIdx = -1;                  // 先复位再渲染：虚拟挂载按当前值套用 playing
    renderSubs(state.curSegs);
  }

  // 替换后的当前匹配语义：指向改动位置之后的第一个匹配；无则回到第一个
  function keepSearchIdxNear(segIdx, start) {
    if (!state.search.matches.length) { state.search.idx = -1; updateSearchCount(); return; }
    let k = 0;
    if (segIdx >= 0) {
      const found = state.search.matches.findIndex(m =>
        m.segIdx > segIdx || (m.segIdx === segIdx && m.start >= start));
      if (found >= 0) k = found;
    }
    state.search.idx = k;
    updateSearchCount();
  }

  // 替换当前匹配处：push 历史 → 改文本（卡拉OK字级同步）→ 重算匹配并定位 → 重绘
  // （全量重绘不换滚动容器，滚动位置保持；当前匹配在改动处之后）
  function replaceCurrentMatch() {
    if (!state.search.q || !state.search.matches.length) return;
    if (state.search.idx < 0) { nextMatch(); return; }   // 尚未定位 → 先跳到第一个
    const h = state.search.matches[state.search.idx], seg = state.curSegs[h.segIdx];
    if (!seg) return;
    pushHistory();
    const rep = ($("#search-r") && $("#search-r").value) || "";
    const text = String(seg.text || "");
    seg.text = text.slice(0, h.start) + rep + text.slice(h.start + h.length);
    seg.words = reflowWords(seg, seg.text);
    computeSearchMatches();
    keepSearchIdxNear(h.segIdx, h.start + rep.length);
    rerenderForReplace([h.segIdx]);
    const cur = state.search.matches[state.search.idx];
    if (cur) ensureCardVisible(cur.segIdx);
  }

  // 全部替换：push 一次快照 → 遍历所有段逐个改写 → 重算匹配（从首个开始）→ 重绘
  function replaceAllMatches() {
    if (!state.search.q || !state.search.matches.length) return;
    pushHistory();
    const rep = ($("#search-r") && $("#search-r").value) || "";
    const needle = state.search.q.toLowerCase();
    const affected = [];                       // 受替换影响的段 idx（脏区更新用）
    state.curSegs.forEach((seg, i) => {
      const text = String(seg.text || "");
      const hay = text.toLowerCase();
      if (!hay.includes(needle)) return;
      let out = "", from = 0, i2;
      while ((i2 = hay.indexOf(needle, from)) >= 0) {
        out += text.slice(from, i2) + rep;
        from = i2 + needle.length;
      }
      out += text.slice(from);
      seg.text = out;
      seg.words = reflowWords(seg, out);
      affected.push(i);
    });
    computeSearchMatches();
    keepSearchIdxNear(-1, -1);
    rerenderForReplace(affected);
    const cur = state.search.matches[state.search.idx];
    if (cur) ensureCardVisible(cur.segIdx);
  }

  // ── 字幕存档：由内存中的 state.curSegs 在前端组档，下载到使用者指定位置 ──
  //   （浏览器沙箱不暴露来源档真实路径，无法写回来源文件夹，故走下载。
  //    SRT/TXT 格式比照 subtitle_lines，含「说话者N：」前缀。）
  function buildSrt(segs) {
    return segs.map((s, i) => {
      const spk = s.speaker ? `${spkLabel(s.speaker)}：` : "";
      return `${i + 1}\n${STTUtil.srtTs(s.start)} --> ${STTUtil.srtTs(s.end)}\n${spk}${s.text}\n`;
    }).join("\n");
  }
  function buildTxt(segs) {
    return segs.some(s => s.speaker)
      ? segs.map(s => (s.speaker ? `${spkLabel(s.speaker)}：` : "") + s.text).join("\n")
      : segs.map(s => s.text).join("");
  }
  // LRC 歌词：行首时间戳 [mm:ss.xx]，带说话者前缀
  function buildLrc(segs) {
    return segs.map(s =>
      `[${STTUtil.lrcTs(s.start)}]${s.speaker ? `${spkLabel(s.speaker)}：` : ""}${s.text}`
    ).join("\n") + "\n";
  }
  // ASS 卡拉OK：有字级 words 时输出 \k 逐字标签（厘秒拍点，播放器即卡拉OK高亮），
  // 无 words 退回整行显示。字间空隙并入前一字拍子，保持拍点连续。
  function buildAss(segs) {
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
      return `Dialogue: 0,${STTUtil.assTs(s.start)},${STTUtil.assTs(s.end)},Default,,0,0,0,,${text}`;
    });
    return header.concat(lines).join("\n") + "\n";
  }
  const _SAVE_EXTS = { srt: ".srt", txt: ".txt", lrc: ".lrc", ass: ".ass" };
  async function saveSubtitle(fmt) {
    if (!state.curSegs.length) return;
    if (!fmt || !_SAVE_EXTS[fmt]) {
      fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    }
    const text = fmt === "txt" ? buildTxt(state.curSegs)
      : fmt === "lrc" ? buildLrc(state.curSegs)
      : fmt === "ass" ? buildAss(state.curSegs)
      : buildSrt(state.curSegs);
    const base = (picked && picked.name) ? picked.name.replace(/\.[^.]+$/, "") : T("save.defaultName", "字幕");
    const filename = base + (_SAVE_EXTS[fmt] || ".srt");
    STTUtil.downloadBlob(filename, text);
    logLine("✓ " + T("file.saved", "已存档字幕：") + filename);
  }

  // ── 记录 ────────────────────────────────────────────────
  function logLine(msg) {
    const el = $("#file-log"); el.hidden = false;
    const d = document.createElement("div"); d.className = "ln"; d.textContent = msg;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  }
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
          o.value = l.value; o.textContent = l.value ? l.label : T("common.langAuto", "语言 自动");
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
    if (window.I18N) { window.I18N.setLang(uiLang); refreshViewTitle(); }
  }
  function segSet(sel, v) { $$(sel + " button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
  // 界面缩放：用 CSS zoom（Chromium/WebView2 支持）整体缩放，px 版面也能等比生效。
  function applyScale(pct) { document.body.style.zoom = (Math.max(50, Math.min(200, +pct || 100)) / 100); }

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
      el.innerHTML = `<span class="hn">${STTUtil.escapeHtml(it.name)}</span>
        <span class="hm">${when} · ${Math.max(1, Math.round(it.size / 1024))} KB</span>
        <span class="hx" role="button" tabindex="0" aria-label="${T("file.historyDel", "删除（到回收站）")}" title="${T("file.historyDel", "删除（到回收站）")}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/></svg>
        </span>`;
      // span[role=button] 无默认激活行为：补 Enter/Space 键盘删除
      el.addEventListener("keydown", e => {
        if (e.target.closest(".hx") && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault(); e.stopPropagation(); deleteHistoryItem(it, el);
        }
      });
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
        logLine("✕ " + T("file.historyDelFail", "删除失败：") + ((r && r.error) || T("common.unknownErr", "未知错误")));
      }
    } catch (err) {
      el.disabled = false;
      logLine("✕ " + T("file.historyDelFail", "删除失败：") + (err.message || err));
    }
  }

  // ── 共享工具 ────────────────────────────────────────────
  // escapeHtml / fmtClock（及 srtTs/lrcTs/assTs/downloadBlob）已抽至
  // util.js（window.STTUtil），调用处统一走 STTUtil.xxx。
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
  const SILENCE_MS = 2200, MIN_SEG_MS = 500, MAX_SEG_MS = 20000, VAD_THRESH = 0.014;

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

  async function startRec() {
    if (!recSupported()) { recNote(T("record.noApi", "此环境不支持录音 API。"), true); return; }
    // 依选定的麦克风建立音讯约束（空值 → 系统默认设备）。
    const micId = $("#rec-mic") ? $("#rec-mic").value : "";
    const audioConstraint = { echoCancellation: true, noiseSuppression: true };
    if (micId) audioConstraint.deviceId = { exact: micId };
    try {
      state.REC.stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
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
    state.REC.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = state.REC.ctx.createMediaStreamSource(state.REC.stream);
    state.REC.analyser = state.REC.ctx.createAnalyser(); state.REC.analyser.fftSize = 1024;
    src.connect(state.REC.analyser);
    state.REC.on = true; state.REC.sec = 0;
    state.REC.recStart = Date.now();      // 真实时间轴基准：段起止=录音内的真实秒数
    API.recordState(true);          // 同步后端：录音中 → 关窗需确认
    $("#rec-btn").classList.add("recording");
    const lw = $("#rec-live-wave"); lw.hidden = false;
    lw.innerHTML = Array.from({ length: 28 }, () => `<div class="b" style="height:4px"></div>`).join("");
    $("#rec-timer").textContent = "00:00";
    state.REC.timer = setInterval(() => { state.REC.sec++; $("#rec-timer").textContent = STTUtil.fmtClock(state.REC.sec); }, 1000);
    recNote(T("record.listening", "聆听中…说完停顿约 2 秒会自动识别；再按一次结束并完成最后一段。"));
    startSeg(); monitor();
  }
  function recElapsed() { return state.REC.recStart ? (Date.now() - state.REC.recStart) / 1000 : 0; }
  function startSeg() {
    state.REC.chunks = []; state.REC.speech = false; state.REC.silentSince = 0; state.REC.segStart = Date.now();
    state.REC.segStartElapsed = recElapsed();   // 本段在录音里的真实起点
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
      .find(m => MediaRecorder.isTypeSupported(m)) || "";
    state.REC.recorder = mime ? new MediaRecorder(state.REC.stream, { mimeType: mime })
                        : new MediaRecorder(state.REC.stream);
    state.REC.recorder.ondataavailable = e => { if (e.data && e.data.size) state.REC.chunks.push(e.data); };
    state.REC.recorder.onstop = onSegStop;
    state.REC.recorder.start();
  }
  function cutSeg() {
    if (state.REC.recorder && state.REC.recorder.state === "recording") {
      state.REC.segEndElapsed = recElapsed();   // 录制媒体止于 stop() 调用时刻
      state.REC.recorder.stop();
    }
  }
  async function onSegStop() {
    const dur = Date.now() - state.REC.segStart;
    const blob = new Blob(state.REC.chunks, { type: state.REC.chunks[0] ? state.REC.chunks[0].type : "audio/webm" });
    const ok = state.REC.speech && dur >= MIN_SEG_MS && blob.size > 1200;
    // 真实时间轴：本段起点=开录时刻、终点=切段时刻（静音间隔如实留白），
    // 导出的 SRT 可直接用于同步（旧版固定 5s 顺排，时间轴是假的）。
    const t0 = state.REC.segStartElapsed || 0;
    const t1 = Math.max(t0 + 0.5, state.REC.segEndElapsed || recElapsed());
    if (state.REC.on) startSeg(); else teardownRec();    // 先续录，避免上传期间漏话
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
    const buf = new Uint8Array(state.REC.analyser.fftSize);
    const bars = $$(".b", $("#rec-live-wave"));
    const tick = () => {
      if (!state.REC.on) return;
      state.REC.analyser.getByteTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / buf.length);
      const base = Math.min(34, rms * 260);
      bars.forEach(b => b.style.height = Math.max(4, base * (0.5 + Math.random() * 0.5)).toFixed(0) + "px");
      const now = Date.now();
      if (rms >= VAD_THRESH) { state.REC.speech = true; state.REC.silentSince = 0; }
      else if (state.REC.speech) {
        if (!state.REC.silentSince) state.REC.silentSince = now;
        else if (now - state.REC.silentSince >= SILENCE_MS) cutSeg();   // 停顿 → 切段
      }
      if (Date.now() - state.REC.segStart >= MAX_SEG_MS && state.REC.speech) cutSeg();
      state.REC.raf = requestAnimationFrame(tick);
    };
    state.REC.raf = requestAnimationFrame(tick);
  }
  function stopRec() {
    state.REC.on = false; $("#rec-btn").classList.remove("recording");
    API.recordState(false);         // 同步后端：录音已结束
    if (state.REC.raf) cancelAnimationFrame(state.REC.raf);
    clearInterval(state.REC.timer);
    $("#rec-live-wave").hidden = true;
    recNote(T("record.stopped", "已停止。"));
    cutSeg();    // 收尾段（onSegStop 会上传并 teardown）
  }
  function teardownRec() {
    try { state.REC.stream && state.REC.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { state.REC.ctx && state.REC.ctx.close(); } catch (e) {}
    state.REC.stream = state.REC.ctx = state.REC.analyser = state.REC.recorder = null;
  }
  function recNote(msg, warn) {
    const el = $(".rec-note");
    if (el) { el.textContent = msg; el.style.color = warn ? "var(--live)" : "var(--muted)"; }
  }
  function appendRecLine(text, t0, t1) {
    const host = $("#rec-subs");
    if (state.REC.firstResult) { host.innerHTML = ""; state.REC.firstResult = false; }
    // 真实时间轴：t0/t1 为该段在录音里的真实起止秒（onSegStop 传入）
    const start = Number.isFinite(t0) ? t0
      : (state.REC.segs.length ? state.REC.segs[state.REC.segs.length - 1].end + 0.1 : 0);
    const end = Number.isFinite(t1) ? Math.max(t1, start + 0.5) : start + 5;
    const seg = { start, end, speaker: null, text };
    state.REC.segs.push(seg);
    const el = document.createElement("div");
    el.className = "sub-card";
    el.innerHTML = `<span class="tc">${STTUtil.fmtClock(start)}</span>
      <div class="body"><div class="txt">${STTUtil.escapeHtml(text)}</div></div>`;
    host.appendChild(el);
    host.scrollTop = host.scrollHeight;
    $("#rec-save").disabled = false;
    if (state.REC.fileHandle) recAutosaveWrite();    // 实时存档：每段附加后即落盘
  }

  // ── 清除／存储／实时存档（对应 app.py 录制页的清除、存储 SRT、实时追加保存）──
  function recClear() {
    state.REC.segs = []; state.REC.firstResult = true;
    $("#rec-subs").innerHTML = `<div class="sub-card"><span class="tc">— · —</span>`
      + `<div class="body"><div class="txt" style="color:var(--muted)">${T("record.placeholder", "开始录音后，识别结果会逐段出现在这里。")}</div></div></div>`;
    $("#rec-save").disabled = true;
  }
  async function recSave() {
    if (!state.REC.segs.length) return;
    let fmt = "srt";
    try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    const text = fmt === "txt" ? buildTxt(state.REC.segs) : buildSrt(state.REC.segs);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "").slice(0, 14);
    STTUtil.downloadBlob(`realtime_${stamp}` + (fmt === "txt" ? ".txt" : ".srt"), text);
  }
  async function recAutosaveWrite() {
    if (!state.REC.fileHandle) return;
    try {
      let fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
      const text = fmt === "txt" ? buildTxt(state.REC.segs) : buildSrt(state.REC.segs);
      const w = await state.REC.fileHandle.createWritable();   // 截断重写（内容小，安全可靠）
      await w.write(text); await w.close();
    } catch (e) { recNote(T("record.autosaveFail", "实时存档写入失败：") + (e.message || e), true); state.REC.fileHandle = null; }
  }

// ══ boot()：原 IIFE 顶层的 listener 绑定 / API 事件订阅 / wire* / 初始调用 ══
//   由 main.js 在 DOM ready 后调用一次。语句按原 app.js 相对顺序移入；引用的
//   函数声明已提升至模块顶层、let/const 在模块求值期已初始化，故执行语义与
//   原先「IIFE 载入即执行顶层语句」完全一致。
export function boot() {
  state.gotoView = switchView;   // 注入视图切换回调：modelView「前往语音转文字」经此导航（零反向 import）

  $("#nav").addEventListener("click", e => {
    const btn = e.target.closest(".nav-item");
    if (btn) switchView(btn.dataset.view);
  });
  // 左下角状态点 → 直达「模型」页（状态与模型下载/热切换直接相关）
  $("#model-status").addEventListener("click", () => switchView("model"));
  $("#rec-latency-close").addEventListener("click", () => {
    _recLatencyClosed = true;
    const el = $("#rec-latency");
    if (el) el.hidden = true;
  });

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
  // 取消按钮：仅转录进行中可见；点击即禁用并改「正在取消…」，通知后端 /api/cancel
  $("#btn-cancel-tx") && $("#btn-cancel-tx").addEventListener("click", () => {
    const cancelBtn = $("#btn-cancel-tx");
    cancelBtn.disabled = true;
    cancelBtn.textContent = T("tx.cancelling", "正在取消…");
    API.cancel();
  });
  $("#btn-run").addEventListener("click", async () => {
    if (!picked || running) return;
    running = true; state.switchBusy = true;    // 转录中 → 模型页切换禁用
    const btn = $("#btn-run");
    const cancelBtn = $("#btn-cancel-tx");
    btn.disabled = true; btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>${T("file.running", "识别中…")}`;
    if (cancelBtn) { cancelBtn.hidden = false; cancelBtn.disabled = false; cancelBtn.textContent = T("tx.cancel", "取消转换"); }
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
      const msg = err && err.message ? err.message : String(err);
      if (/已取消|cancel/i.test(msg)) {
        // 用户主动取消 → 中性提示，不当错误弹窗处理
        logLine("ℹ " + T("tx.cancelled", "已取消"));
      } else {
        logLine("✕ " + msg);
      }
    } finally {
      running = false; state.switchBusy = false;
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>${T("file.run", "开始转换")}`;
      if (cancelBtn) cancelBtn.hidden = true;    // 完成/失败/取消后一律隐藏
    }
  });

  // 进度分派：模型就地加载中 → 模型页进度；批量执行中 → 当前任务列；否则 → 单文件进度条
  API.on("progress", ({ pct, status }) => {
    if (pct < 0) { connHint(status); return; }   // SSE 断线/恢复提示（桥接层自动重连）
    if (state.modelLoading) {
      setModelProg(pct, status);
    } else {
      showProgress(pct, status);
    }
  });
  // 文档级快捷键：Ctrl+Z 撤销、Ctrl+Y / Ctrl+Shift+Z 重做；仅在已渲染结果时生效；
  // 编辑器输入中（contenteditable）放行浏览器默认的文本撤销。
  document.addEventListener("keydown", e => {
    if (!state.curSegs.length || !(e.ctrlKey || e.metaKey) || e.altKey) return;
    const k = (e.key || "").toLowerCase();
    const isUndo = k === "z" && !e.shiftKey;
    const isRedo = k === "y" || (k === "z" && e.shiftKey);
    if (!isUndo && !isRedo) return;
    if (e.target && e.target.closest && e.target.closest("[contenteditable]")) return;
    e.preventDefault();
    isUndo ? doUndo() : doRedo();
  });
  window.addEventListener("resize", syncStickyOffset);
  window.addEventListener("resize", () => {
    if (!_virtualMode || !_vwin) return;
    _vwin.setViewport(subsViewport());
    mountWindow();
  });
  $("#btn-karaoke") && $("#btn-karaoke").addEventListener("click", () => setKaraoke(!karaokeOn));
  // 事件委托接线（绑定一次）：字幕卡交互 → #subs；波形区块点击 → #seg-strip
  wireSubsDelegation();
  (function wireSegStripDelegation() {
    const strip = $("#seg-strip");
    if (!strip || strip.dataset.delegated) return;
    strip.dataset.delegated = "1";
    strip.addEventListener("click", e => {
      const blk = e.target.closest(".seg-block");
      if (blk) selectSeg(+blk.dataset.seg);            // 点区块 → 选中该段区间
    });
  })();

  $("#btn-open-dir").addEventListener("click", () => API.openOutputDir());
  // 合并段落：进入/退出选择模式；确认/取消在 #merge-bar
  $("#btn-merge-mode") && $("#btn-merge-mode").addEventListener("click", () => setMergeMode(!state.mergeMode));
  $("#merge-ok") && $("#merge-ok").addEventListener("click", mergeSelectedSegments);
  $("#merge-cancel") && $("#merge-cancel").addEventListener("click", () => setMergeMode(false));
  $("#btn-undo") && $("#btn-undo").addEventListener("click", doUndo);
  $("#btn-redo") && $("#btn-redo").addEventListener("click", doRedo);
  // 搜索区事件：输入即重算（重置到首个匹配）；Enter/Shift+Enter 循环跳转；Esc 清空
  $("#search-q") && $("#search-q").addEventListener("input", e => {
    state.search.q = e.target.value;
    computeSearchMatches();
    state.search.idx = state.search.matches.length ? 0 : -1;
    updateSearchCount();
    _playingIdx = -1;                  // 先复位再渲染：虚拟挂载按当前值套用 playing
    renderSubs(state.curSegs);
    if (state.search.idx >= 0) ensureCardVisible(state.search.matches[0].segIdx);
  });
  $("#search-q") && $("#search-q").addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); e.shiftKey ? prevMatch() : nextMatch(); }
    else if (e.key === "Escape") { e.preventDefault(); clearSearch(); }
  });
  $("#search-prev") && $("#search-prev").addEventListener("click", prevMatch);
  $("#search-next") && $("#search-next").addEventListener("click", nextMatch);
  $("#search-replace") && $("#search-replace").addEventListener("click", replaceCurrentMatch);
  $("#search-replace-all") && $("#search-replace-all").addEventListener("click", replaceAllMatches);
  // 保存菜单：SRT（默认，按全局输出格式）/ ASS / LRC / TXT
  const saveMenu = $("#save-menu");
  $("#btn-save-sub").addEventListener("click", () => {
    if (!state.curSegs.length) return;
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
    if (window.I18N) { window.I18N.setLang(v); refreshViewTitle(); }
    if (state.curView === "model") renderModel();   // 重绘动态文字（核心卡描述等）
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
  $("#rec-btn").addEventListener("click", () => state.REC.on ? stopRec() : startRec());
  $("#rec-mic-refresh") && $("#rec-mic-refresh").addEventListener("click", enumerateMics);
  if (navigator.mediaDevices) {
    try { navigator.mediaDevices.addEventListener("devicechange", enumerateMics); } catch (e) {}
  }
  $("#rec-clear") && $("#rec-clear").addEventListener("click", recClear);
  $("#rec-save") && $("#rec-save").addEventListener("click", recSave);

  // 实时存档：用 File System Access API（WebView2/Edge 支持）持续把目前累积的
  // 字幕写入使用者选定的文件 —— 重现 app.py「实时追加保存，可随时中断不遗失」。
  // 不支持的环境隐藏此开关（仍可用「存储字幕」一次性导出）。
  const FS_SAVE_OK = typeof window.showSaveFilePicker === "function";
  if (FS_SAVE_OK) $("#rec-autosave-wrap").hidden = false;
  $("#rec-autosave") && $("#rec-autosave").addEventListener("change", async e => {
    if (!e.target.checked) { state.REC.fileHandle = null; return; }
    try {
      let fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (err) {}
      const ext = fmt === "txt" ? "txt" : "srt";
      state.REC.fileHandle = await window.showSaveFilePicker({
        suggestedName: `realtime.${ext}`,
        types: [{ description: T("save.fileDesc", "字幕档"), accept: { "text/plain": ["." + ext] } }],
      });
      await recAutosaveWrite();           // 立即写一次（含既有片段）
      recNote(T("record.autosaveOn", "实时存档已启用：每段识别完成后会自动写入所选文件。"));
    } catch (err) {                         // 使用者取消选档 → 还原开关
      state.REC.fileHandle = null; e.target.checked = false;
    }
  });
  // ── 启动 ────────────────────────────────────────────────
  // 关闭兜底（Edge --app / 浏览器访问本地端点时无原生确认框）：
  // 有任务进行中 → 触发浏览器原生「确定要离开吗」提示。
  // 桌面原生窗口（URL 带 ?native=1）的关闭确认由 app_webview 的 closing
  // 事件处理（原生 MessageBox），这里不注册，避免确认后二次弹窗。
  if (!new URLSearchParams(location.search).has("native")) {
    window.addEventListener("beforeunload", (e) => {
      const busy = running || state.REC.on || state.modelLoading;
      if (busy) { e.preventDefault(); e.returnValue = ""; }
    });
  }
  // 加载完成 → 更新就绪灯 + 语言清单（依引擎）；若为「就地下载并加载」流程 → 收尾
  API.on("status", async (s) => {
    refreshStatus(); loadLanguages();
    let ready = s && s.modelReady;
    if (ready == null) { try { ready = (await API.getStatus()).modelReady; } catch (e) {} }
    if (state.modelLoading && ready) {
      state.modelLoading = false;
      setModelProg(100, T("common.done", "完成"));
      setTimeout(() => { $("#model-progress").hidden = true; }, 800);
      modelMsg(T("model.readySwitch", "模型已就绪。已为你切换到「语音转文字」。"), "info");
      renderHealth();                 // 自检色点刷新
      switchView("file");             // 完成 → 进入转录页
    } else if (s && s.error && state.modelLoading) {
      state.modelLoading = false;
      $("#model-progress").hidden = true;
      modelMsg(T("model.loadFail", "加载失败：") + s.error, "warn");
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
}

/* ============================================================
   modelView.js — 「模型与设备」视图模块（自 app.js:1795-2064 迁出，ESM）
   迁移映射：
   - renderHealth/renderModelCheck/renderModel/coreObj/renderCoreCards/
     populateModels/updateArch/renderAccel/accelMsg/applyModel/_switchBlocked/
     updateLoadBtn/setModelProg/modelMsg + ICON_CPU/ICON_GPU → 本模块（实现逐字一致）
   - 模块局部状态 _modelOpt/_curCore（原 app.js:1799-1800）→ 本模块 let
   - _modelLoading/_switchBusy（原 app.js:2008-2009）→ ../state.js
   - listener 绑定（#btn-health-recheck/#accel-cards/#core-cards/#model-select/
     #btn-model-load）→ 本模块顶层（模块求值期绑定，时序等同原 IIFE 顶层；
     均在 DOM 解析完成后、任何用户交互之前）
   对 appCore 的反向需求（零反向 import）：
   - 「前往语音转文字」→ 经 state.gotoView 回调（appCore.boot() 注入 switchView）
   - appCore → 本模块为单向 import：renderModel/renderHealth/setModelProg/
     modelMsg/updateLoadBtn
   ============================================================ */
import { state } from "../state.js";
import * as STTUtil from "../util.js";
const { $, $$, T } = STTUtil;
const API = window.QwenAPI;   // 经典脚本 bridge.js 的全局（不得 import 不存在的模块）

let _modelOpt = null;     // 最近一次 getModelOptions 结果
let _curCore = null;      // 目前选中的核心标签

// ── 系统自检：每核心 × 每能力，显示 Green/Yellow/Red ──────────────
export async function renderHealth() {
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
    card.innerHTML = `<div class="hc-title">${STTUtil.escapeHtml(g.label)}` +
      (active ? ` <span class="chip chip-ok" style="font-size:10px">${T("model.inUse", "使用中")}</span>` : ``) + `</div>`;
    (g.items || []).forEach(it => {
      const row = document.createElement("div"); row.className = "hc-item";
      row.innerHTML = `${dot(it.status)}<span class="hi-label">${STTUtil.escapeHtml(it.label)}</span>` +
        `<span class="hi-detail">${STTUtil.escapeHtml(it.detail || "")}</span>`;
      card.appendChild(row);
    });
    host.appendChild(card);
  });
}
$("#btn-health-recheck") && $("#btn-health-recheck").addEventListener("click", renderHealth);

// ── 检查明细：「模型未下载」的判定依据 ────────────────────────────
//   后端 selectedDetail 只在所选模型不就绪时返回（就绪时 null → 整块隐藏）。
//   显示程序实际检查的目录与缺失文件——settings 残留 model_dir 指向别处、
//   目录里放的不是所选模型这类问题，一眼看出（用户反馈实录：ov_models
//   明明有模型，重启后仍提示重新下载）。
async function renderModelCheck() {
  const el = $("#model-check");
  if (!el) return;
  try {
    const st = await API.getStatus();
    const d = st && st.selectedDetail;
    if (!d) { el.hidden = true; el.innerHTML = ""; return; }
    let html = `<div class="mc-line">${STTUtil.escapeHtml(T("model.checkDir", "程序检查的模型目录："))}<code>${STTUtil.escapeHtml(d.modelDir)}</code></div>`;
    if (d.crispasrDir)
      html += `<div class="mc-line">${STTUtil.escapeHtml(T("model.checkCoreDir", "CrispASR 核心目录："))}<code>${STTUtil.escapeHtml(d.crispasrDir)}</code></div>`;
    if (d.coreIssue)
      html += `<div class="mc-line mc-warn">${STTUtil.escapeHtml(T("model.checkCoreIssue", "加速器：{issue}", { issue: d.coreIssue }))}</div>`;
    const miss = d.missing || [], missC = d.missingCrispasr || [];
    if (miss.length)
      html += `<div class="mc-line">${STTUtil.escapeHtml(T("model.checkMissing", "缺少 {n} 个文件：", { n: miss.length }))}<code>${STTUtil.escapeHtml(miss.slice(0, 4).join("、"))}${miss.length > 4 ? "…" : ""}</code></div>`;
    if (missC.length)
      html += `<div class="mc-line">${STTUtil.escapeHtml(T("model.checkMissingCore", "核心目录缺少 {n} 个文件：", { n: missC.length }))}<code>${STTUtil.escapeHtml(missC.join("、"))}</code></div>`;
    el.innerHTML = html;
    el.hidden = false;
  } catch (e) { el.hidden = true; }
}

export async function renderModel() {
  renderHealth();
  await renderModelCheck();
  // 核心卡 + 模型下拉
  try {
    _modelOpt = await API.getModelOptions();
    _curCore = _modelOpt.current.core;
    renderCoreCards();
    populateModels(_curCore, _modelOpt.current.model);
    modelMsg("", null);
  } catch (e) {}
  // 检测设备 + 诊断
  const d = await API.listDevices();
  const host = $("#dev-list"); host.innerHTML = "";
  d.devices.forEach(dev => {
    const cpu = dev.kind === "cpu";
    const el = document.createElement("div"); el.className = "dev";
    el.innerHTML = `<span class="ic">${cpu ? ICON_CPU : ICON_GPU}</span>
      <span class="nm">${dev.kind === "cpu" ? "CPU · " : "GPU · "}${STTUtil.escapeHtml(dev.name)}</span>
      <span class="vram">${STTUtil.escapeHtml(dev.note || "")}</span>`;
    host.appendChild(el);
  });
  const accelNeeds = await renderAccel();
  updateLoadBtn(null, accelNeeds);   // 按钮文案对齐加速器状态（提示让你下载时，按钮不能是「前往」）
  const diag = $("#gpu-diag");
  if (d.diag && d.diag.text) {
    diag.hidden = false;
    diag.className = "banner " + (d.diag.level === "warn" ? "banner-warn" : "banner-info");
    const txt = diag.querySelector("div"); if (txt) txt.innerHTML = STTUtil.escapeHtml(d.diag.text);
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
      <div class="t">${STTUtil.escapeHtml(core.label)}</div>
      <div class="d">${STTUtil.escapeHtml(T("model.nModels", `${core.models.length} 种模型可选`, { n: core.models.length }))}</div></div>`;
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
    if (op.recommended) tags.push(T("accel.recommended", "建议"));
    if (a.installed && a.installed.variant === op.key
        && a.installed.version === a.latest) tags.push(T("accel.installed", "已安装"));
    const tagHtml = tags.length
      ? ` <span class="chip chip-muted" style="font-weight:400">${tags.join(" · ")}</span>` : "";
    const noteHtml = op.note ? `　<span style="color:var(--warn,#b26a00)">${STTUtil.escapeHtml(op.note)}</span>` : "";
    card.innerHTML = `<span class="rd"></span><div class="info">
      <div class="t">${STTUtil.escapeHtml(op.label)}　<span style="font-weight:400;color:var(--muted)">${T("accel.sizeMb", "约 {n} MB", { n: op.size_mb })}</span>${tagHtml}</div>
      <div class="d">${STTUtil.escapeHtml(op.desc || "")}${noteHtml}</div></div>`;
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
  return !!a.needsDownload;   // 供 updateLoadBtn 对齐按钮文案（提示让你下载时，按钮不能再是「前往」）
}
function accelMsg(text, level) {
  const el = $("#accel-msg"); if (!el) return;
  if (!text) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false; el.style.marginTop = "10px";
  el.className = "banner " + (level === "warn" ? "banner-warn" : "banner-info");
  el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg><div>${STTUtil.escapeHtml(text)}</div>`;
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
// 转录/加载忙碌 → 引擎卡与模型下拉禁操作（热切换的并发保护）
function _switchBlocked() { return state.switchBusy; }
async function applyModel(core, model) {
  if (_switchBlocked()) return;   // 转录中忽略选择变更（下拉会回弹到当前值）
  try {
    const res = await API.setModel(core, model);
    if (res && res.message) modelMsg(res.message, "info");
    else modelMsg("", null);
    updateLoadBtn(res);
  } catch (err) { modelMsg(T("model.applyFail", "套用失败：") + (err.message || err), "warn"); }
}
// 依 setModel 结果 + 目前加载状态，决定「下载并加载」按钮的文字/可用性（热切换版）
// accelNeeds：加速器版本/变体与所选不符（renderAccel 的 needsDownload）——
// 此刻按钮必须是「下载并加载模型」（点它会连带取得加速器），不能显示「前往」。
export async function updateLoadBtn(res, accelNeeds) {
  const btn = $("#btn-model-load"); if (!btn) return;
  let st = {};
  try { st = await API.getStatus(); } catch (e) {}
  if (accelNeeds == null) {
    // 未传加速器状态时自行补齐（ applyModel 成功后加速器已就绪，可传 false 跳过）
    try { const a = await API.get_accel(); accelNeeds = !!(a && a.needsDownload); } catch (e) {}
  }
  btn.classList.remove("btn-ghost");
  if (state.modelLoading || st.loading) {
    btn.disabled = true; btn.textContent = T("model.loading", "下载／加载中…"); return;
  }
  if (st.transcribing) {
    btn.disabled = true; btn.textContent = T("model.busy", "转录进行中，无法切换"); btn.dataset.act = ""; return;
  }
  // 所选组合的加速器/文件不就绪（如加速器版本不符）时，即使旧引擎可用、
  // 选择未变，按钮也必须是「下载并加载模型」——点它才会升级加速器并加载，
  // 与上方「…点『下载并加载模型』时会一并取得」提示一致（旧实现只看
  // modelReady/selectionDiffers，与提示条的 needsDownload 口径脱节）。
  const selectedReady = (st.selectedReady !== false)
    && !(st.selectedDetail && st.selectedDetail.coreIssue);
  if (st.modelReady && selectedReady && !accelNeeds && !(res && res.identityChanged)) {
    // 已就绪且选择未变 → 引导前往转录
    btn.disabled = false; btn.textContent = T("model.goto", "前往语音转文字"); btn.dataset.act = "goto"; return;
  }
  // 选择有变化（或尚未加载）→ 热切换：就地下载并加载
  btn.disabled = false; btn.textContent = T("model.load", "下载并加载模型"); btn.dataset.act = "load";
}
$("#btn-model-load") && $("#btn-model-load").addEventListener("click", async () => {
  const btn = $("#btn-model-load");
  if (btn.dataset.act === "goto") { if (state.gotoView) state.gotoView("file"); return; }
  state.modelLoading = true; btn.disabled = true; btn.textContent = T("model.loading", "下载／加载中…");
  $("#model-progress").hidden = false;
  setModelProg(0, T("model.prepLoad", "准备加载…"));
  modelMsg(T("model.loadingHint", "正在下载并加载模型，请稍候——可留在本页观看进度。"), "info");
  try { await API.startLoad(); }
  catch (err) { state.modelLoading = false; modelMsg(T("model.loadFail", "加载失败：") + (err.message || err), "warn"); updateLoadBtn(); }
});
export function setModelProg(pct, status) {
  $("#model-prog-bar").style.width = pct + "%";
  $("#model-prog-pct").textContent = Math.round(pct) + "%";
  if (status != null) $("#model-prog-status").textContent = status;
}
export function modelMsg(text, level) {
  const el = $("#model-msg");
  if (!text) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false; el.style.marginTop = "10px";
  el.className = "banner " + (level === "warn" ? "banner-warn" : "banner-info");
  el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg><div>${STTUtil.escapeHtml(text)}</div>`;
}
const ICON_CPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>`;
const ICON_GPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 18v2M18 18v2"/></svg>`;

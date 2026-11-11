/* ============================================================
   util.js — 共享纯工具：DOM 查询 / i18n 取字 / HTML 转义 / 时间戳格式化
              / Blob 下载 / 编辑历史栈 / 虚拟滚动窗口 / 卡拉OK游标
   ES module（渐进迁移）：由 main.js → appCore.js / views/ 经 import 使用；
   文件尾保留 window.STTUtil 挂载（迁移期兼容，未经 import 的调用面仍可用）。
   $ / $$ / T 自原 app.js:8-16 抽入（实现逐字一致，遵循本项目
   「纯工具抽至 util.js」的既有惯例）。
   ============================================================ */

// ── DOM 查询助手（原 app.js:8-9，逐字一致）──────────────
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ── i18n 取字（含 {n} 等简单插值）；无字典或无此键时回退默认值 def。
//   （原 app.js:12-16，逐字一致；I18N 为经典脚本 i18n.js 挂到 window 的全局，
//    经典脚本先于 ESM 执行，故此处经全局作用域取用是安全的。）
export function T(key, def, vars) {
  let s = (window.I18N && window.I18N.t(key)) || def || key;
  if (vars) for (const k in vars) s = s.replace("{" + k + "}", vars[k]);
  return s;
}

// ── HTML 转义：innerHTML 模板拼接前必过 ─────────────────
export function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ── m:ss 时钟（波形时长／计时器显示；负值钳 0、四舍五入到秒）──
export function fmtClock(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = String(Math.floor(sec / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

// ── SRT 时间戳：HH:MM:SS,mmm ────────────────────────────
export function srtTs(s) {
  const ms = Math.round(s * 1000);
  const h = String(Math.floor(ms / 3600000)).padStart(2, "0");
  const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, "0");
  const sec = String(Math.floor((ms % 60000) / 1000)).padStart(2, "0");
  return `${h}:${m}:${sec},${String(ms % 1000).padStart(3, "0")}`;
}

// ── LRC 时间戳：mm:ss.xx ────────────────────────────────
export function lrcTs(sec) {
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60), ss = s - m * 60;
  return `${String(m).padStart(2, "0")}:${ss.toFixed(2).padStart(5, "0")}`;
}

// ── ASS 时间戳：H:MM:SS.cc（厘秒；小时不补零的 V+MS 格式）──
export function assTs(sec) {
  const cs = Math.max(0, Math.round(sec * 100));
  const h = Math.floor(cs / 360000), m = Math.floor(cs / 6000) % 60;
  const s = Math.floor(cs / 100) % 60, c = cs % 100;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(c).padStart(2, "0")}`;
}

// ── Blob 下载：text（字符串）或 Blob → objectURL → a.click → 1s 后 revoke
export function downloadBlob(filename, textOrBlob) {
  const blob = textOrBlob instanceof Blob
    ? textOrBlob
    : new Blob([textOrBlob], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ── 编辑历史栈（撤销/重做的纯数据结构）──────────────────
//   push(state) 存「编辑前」快照（只存引用；深拷贝由调用方做）；
//   undo(current) 弹出上一快照返回 { state }，并把 current 压入 redo 栈，
//   无可撤销返回 null；redo(current) 对称；新 push 清空 redo（编辑后旧分支作废）；
//   超出 limit 截断最旧；reset 清空两栈。不碰 DOM，可独立单测。
export function createHistory(limit) {
  const n = Math.floor(+limit);
  const cap = Math.max(1, Number.isFinite(n) ? n : 50);   // 无效/缺省 → 50；有限值钳到 ≥1
  const undoStack = [], redoStack = [];
  function trimUndo() { while (undoStack.length > cap) undoStack.shift(); }
  return {
    push(state) {
      undoStack.push(state);
      trimUndo();
      redoStack.length = 0;
    },
    undo(current) {
      if (!undoStack.length) return null;
      const state = undoStack.pop();
      redoStack.push(current);
      return { state };
    },
    redo(current) {
      if (!redoStack.length) return null;
      const state = redoStack.pop();
      undoStack.push(state);
      trimUndo();
      return { state };
    },
    canUndo() { return undoStack.length > 0; },
    canRedo() { return redoStack.length > 0; },
    reset() { undoStack.length = 0; redoStack.length = 0; },
  };
}

// ── 虚拟滚动窗口模型（纯数据结构，Node 可单测；DOM 粘合在 appCore.js）──
//   大字幕列表（数千段）只挂载可视窗口 ± 缓冲的卡片，上下用 spacer 撑高度。
//   本模型只负责「段索引 ↔ 纵向像素」换算与高度校正：
//   - 已渲染过的段记录实测高度（调用方传入时应含卡片间距，如 offsetHeight+gap）；
//   - 未渲染段用「实测均值（尚无实测时用 defaultHeight）」估算；
//   - rangeForScroll(scrollTop) 依估算/实测偏移算出应挂载 [first, last]。
export function createVirtualWindow(opts) {
  const o = opts || {};
  const viewport = Math.max(1, Math.round(+o.viewport) || 600);        // 可视高度 px
  let buffer = Math.max(0, Math.round(+o.buffer) || 0);                // 上下预渲染 px
  if (!buffer) buffer = viewport * 2;                                  // 缺省：窗口高 ×2
  const defaultHeight = Math.max(1, +o.defaultHeight || 64);           // 估算兜底 px
  let count = Math.max(0, Math.round(+o.count) || 0);
  let viewportCur = viewport, bufferCur = buffer;
  const heights = new Array(count);      // undefined = 未实测
  let sum = 0, n = 0;                    // 实测高度的合计 / 条数（均值分子分母）
  let cur = { first: 0, last: -1 };      // 最近一次 rangeForScroll 的窗口

  const avg = () => (n ? sum / n : defaultHeight);
  function hAt(i) { return (i >= 0 && i < count && heights[i] != null) ? heights[i] : avg(); }
  function addMeasured(v) { if (v != null) { sum += v; n++; } }
  function dropMeasured(v) { if (v != null) { sum -= v; n--; } }

  return {
    get count() { return count; },
    get viewport() { return viewportCur; },
    get buffer() { return bufferCur; },
    get first() { return cur.first; },
    get last() { return cur.last; },

    // 单段高度：实测值；未实测 → 均值估算
    getHeight(idx) { return hAt(idx); },

    // 记录实测高度；返回 true 表示与原估算差 > 0.5px（调用方可据此校正 spacer）
    setHeight(idx, h) {
      if (!(idx >= 0 && idx < count) || !(+h > 0)) return false;
      const prev = heights[idx];
      const before = prev != null ? prev : avg();
      if (prev != null) sum -= prev; else n++;
      heights[idx] = +h;
      sum += +h;
      return Math.abs(before - +h) > 0.5;
    },

    // 全列表估算总高（含实测 + 估算）
    estimateTotal() {
      let total = 0;
      for (let i = 0; i < count; i++) total += hAt(i);
      return total;
    },

    // 段 idx 顶边的累计偏移（估算/实测混算）；offsetOf(count) = 总高
    offsetOf(idx) {
      let acc = 0;
      const end = Math.max(0, Math.min(idx | 0, count));
      for (let i = 0; i < end; i++) acc += hAt(i);
      return acc;
    },

    // 依 scrollTop 算应挂载窗口 [first, last]（含上下 buffer 预渲染），
    // 并回填 top/bottom（上下 spacer 应有的高度）。结果同时存入 first/last。
    rangeForScroll(scrollTop) {
      const total = this.estimateTotal();
      const top = Math.max(0, scrollTop - bufferCur);
      const bottom = scrollTop + viewportCur + bufferCur;
      let first = 0, acc = 0;
      while (first < count && acc + hAt(first) <= top) { acc += hAt(first); first++; }
      let last = first, acc2 = acc;
      while (last < count && acc2 < bottom) { acc2 += hAt(last); last++; }
      // 钳制：滚动越过总高（或空列表）时窗口收在合法区间内
      cur = count === 0
        ? { first: 0, last: -1 }
        : (() => {
            const f = Math.min(first, count - 1);
            return { first: f, last: Math.max(f, Math.min(last - 1, count - 1)) };
          })();
      return {
        first: cur.first,
        last: cur.last,
        top: this.offsetOf(cur.first),
        bottom: total - this.offsetOf(cur.last + 1),
      };
    },

    // 结构变更（拆分/合并）：idx 处删 removeCount 段、插入 insertHeights
    // （null = 未实测占位，后续挂载时实测）。实测高度随索引平移。
    spliceAt(idx, removeCount, insertHeights) {
      const rm = Math.max(0, Math.min(removeCount | 0, count - Math.max(0, idx | 0)));
      const start = Math.max(0, idx | 0);
      for (let k = 0; k < rm; k++) dropMeasured(heights[start + k]);
      const ins = (insertHeights || []).map(v => (v == null ? undefined : +v));
      heights.splice(start, rm, ...ins);
      for (const v of ins) addMeasured(v);
      count = heights.length;
    },

    // 段数增减（新转录/撤销恢复等）；减小的尾部实测高度一并回收
    setCount(nCount) {
      nCount = Math.max(0, nCount | 0);
      if (nCount < count) {
        for (let i = nCount; i < count; i++) dropMeasured(heights[i]);
        heights.length = nCount;
      } else {
        while (heights.length < nCount) heights.push(undefined);
      }
      count = nCount;
    },

    // 视口/缓冲变化（窗口 resize）；下次 rangeForScroll 生效
    setViewport(v, b) {
      viewportCur = Math.max(1, Math.round(+v) || viewportCur);
      bufferCur = (b != null) ? Math.max(0, Math.round(+b) || 0) : viewportCur * 2;
    },

    // 清空实测高度（内容大改后重算）
    resetHeights() {
      heights.fill(undefined);
      sum = 0; n = 0;
    },
  };
}

// ── 卡拉OK逐字游标（纯逻辑，Node 可单测）───────────────────
//   维护「已唱游标」index（[0..index] 视为已唱）。每帧给目标游标：
//   - 前进：只返回新区间 [from, to]（调用方给这些 span 加 .sung）；
//   - 回退/seek：backward=true，调用方应先整段清 .sung，再补 [0..to]；
//   - 无变化返回 null（多数帧走这里 → 零 DOM 写）。
export function karaokeCursor(total) {
  const n = Math.max(0, total | 0);
  let idx = -1;
  return {
    get index() { return idx; },
    advance(target) {
      const t = Math.max(-1, Math.min(n - 1, target | 0));
      if (t === idx) return null;
      let r;
      if (t < idx) r = { from: 0, to: t, backward: true };      // 回退：整段重置
      else r = { from: idx + 1, to: t, backward: false };       // 前进：增量区间
      idx = t;
      return r;
    },
    // 手动重置（段落切换等）；有游标时返回待清区间，本来为空返回 null
    reset() {
      const prev = idx;
      idx = -1;
      return prev >= 0 ? { from: 0, to: prev, backward: true } : null;
    },
  };
}

// 目前应视为「已唱」的最后一字索引（words[i].end <= t；无 → -1）。
// hint = 上一帧游标：就地前后扫（播放连续时 O(1)），越界回退到全扫兜底。
export function karaokeSungIndex(words, t, hint) {
  const n = (words && words.length) || 0;
  if (!n) return -1;
  let i = Number.isInteger(hint) ? Math.min(n - 1, Math.max(-1, hint)) : -1;
  if (i >= 0 && words[i].end <= t) {
    while (i + 1 < n && words[i + 1].end <= t) i++;
    return i;
  }
  while (i >= 0 && words[i].end > t) i--;
  if (i >= 0) return i;
  let last = -1;                                    // words 无序等异常 → 全扫兜底
  for (let k = 0; k < n; k++) if (words[k].end <= t) last = k;
  return last;
}

// ── 迁移期兼容挂载：window.STTUtil 命名空间保持不变 ──────
//   appCore/views 经 import 使用本模块；挂载供迁移期未改写为 import 的
//   调用面（及未来经典脚本）继续可用。typeof 守卫仅为 Node 冒烟测试
//   服务（浏览器里 window 恒存在，行为与原先一致）。
if (typeof window !== "undefined") {
  window.STTUtil = { escapeHtml, fmtClock, srtTs, lrcTs, assTs, downloadBlob, createHistory,
                     createVirtualWindow, karaokeCursor, karaokeSungIndex, $, $$, T };
}

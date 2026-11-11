/* ============================================================
   main.js — ESM 入口（渐进迁移）：编排 import 求值顺序 + DOM ready 后 boot
   import 图（无环，单向）：
     main.js
     ├─ ./state.js            （零依赖：跨视图共享可变状态容器）
     ├─ ./util.js             （零依赖：纯工具；尾挂 window.STTUtil 兼容）
     ├─ ./views/modelView.js  （依赖 state/util：模型页视图，含自身 listener）
     └─ ./appCore.js          （依赖 state/util/modelView：应用核心 + boot()）
   ESM 按依赖后序求值：state → util → modelView → appCore，随后 boot()。
   经典脚本 bridge.js / i18n.js（index.html 前部同步引入）先于任何模块代码
   执行，故模块内经 window.QwenAPI / window.I18N 取用是安全的——不得 import
   不存在的模块。
   ============================================================ */
import "./state.js";              // 1) 共享状态（无依赖）
import "./util.js";               // 2) 纯工具（含 window.STTUtil 兼容挂载）
import "./views/modelView.js";    // 3) 模型页视图（依赖 state/util；appCore 亦 import 之，模块只求值一次）
import { boot } from "./appCore.js";   // 4) 应用核心（依赖前三者）

// 模块脚本为 deferred：执行时 DOM 已解析完成（readyState = "interactive"），
// 此分支恒走 else；DOMContentLoaded 兜底仅为防御性正确（语义等同原 app.js
// 在 body 尾部以经典脚本执行）。
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => boot());
} else {
  boot();
}

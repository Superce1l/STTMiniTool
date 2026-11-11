/* ============================================================
   state.js — 跨视图共享的可变状态容器（ES module，零依赖，Node 可加载）
   盘点自 app.js IIFE 顶层的可变状态：只收「跨视图/跨模块共享」者；
   纯局部状态（picked/running/audioEl/audioUrl/waveDur/_editHistory/
   _subEls/_blockEls/_playingIdx/_playedBars/_segHint/_stickyPad/
   _virtualMode/_vwin/_spacerTop/_spacerBot/_scrollHost/_scrollRaf/
   _measureRaf/_editCtx/playFollow/karaokeOn/karaokeRAF/_karaCache/
   _recLatencyClosed/_mqHandler/_modelOpt/_curCore 等）留在所属模块作用域。
   字段 → 原位置（app.js IIFE 顶层）：
     curView      ← :24    let _curView（当前视图名）
     lastStatus   ← 新增   最近一次 /api/status 快照（refreshStatus 写入；
                            视图模块读快照，禁反向 import appCore）
     gotoView     ← 新增   boot() 注入的 switchView 回调；供 views/modelView.js
                            「前往语音转文字」反向导航（不 import appCore）
     curSegs      ← :271   当前字幕段落（转录结果 / 编辑对象）
     speakerNames ← :274   说话者自定义名称 {编号: 名字}
     selRange     ← :342   选中片段 {start, end, idx}
     mergeMode    ← :338   合并选择模式（点卡片＝选择而非跳播）
     mergeSel     ← :339   合并已选段索引 Set
     search       ← :285   字幕搜索替换状态 {q, matches, bySeg, idx}
     modelLoading ← :2008  模型「就地下载并加载」进行中（进度导到模型页）
     switchBusy   ← :2009  转录进行中 → 模型切换整体禁用
     REC          ← :2259-2265 录制会话状态机（对象字面量逐字保留；运行期
                            动态补充 recStart/segStartElapsed/segEndElapsed）
   约定：直接属性读写（无 setter 样板）；引用替换（state.x = 新值）与原
   let 重赋值语义一致；REC 为共享引用，属性就地变更。
   ============================================================ */
export const state = {
  // ── 视图导航 ──────────────────────────────────────────
  curView: "file",
  lastStatus: null,     // {loading, modelReady, selectedReady, selectionDiffers, error, activeModel, selectedModel, backend, version, ...}（后端 /api/status 原样快照）
  gotoView: null,       // function(name)｜null：由 appCore.boot() 注入

  // ── 转录数据（音频页字幕编辑器；recordView 未来复用）──
  curSegs: [],
  speakerNames: {},
  selRange: null,

  // ── 字幕编辑器共享模式 ────────────────────────────────
  mergeMode: false,
  mergeSel: new Set(),
  search: { q: "", matches: [], bySeg: null, idx: -1 },   // matches: [{segIdx,start,length,k}]

  // ── 忙碌标志（转录 ↔ 模型热切换互斥；beforeunload 判busy）──
  modelLoading: false,
  switchBusy: false,

  // ── 录制会话（app.js:2259-2265 原样搬入）─────────────
  REC: {
    on: false, sec: 0, timer: null, raf: 0, firstResult: true,
    stream: null, ctx: null, analyser: null, recorder: null, chunks: [],
    speech: false, silentSince: 0, segStart: 0,
    segs: [],            // 累积识别结果（含合成时间轴），供清除／存档／实时存档
    fileHandle: null,    // 实时存档的 File System Access 文件 handle（若支持）
  },
};

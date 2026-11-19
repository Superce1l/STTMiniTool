## v2.4.0 — 发行整理 + 转录队列（核心不再同捆 / 双格式发行 / 对齐器归位）

对应 [issue #7](https://github.com/Superce1l/STTMiniTool/issues/7) 与用户反馈的「拖入新视频会替换/清除
当前展示、并让熄灭的按钮亮起」。安装包不再把 CrispASR 核心冻进 zip，改为首次使用时取上游最新版；
同时提供 **onedir** 与 **onefile** 两种发行包；转录改走任务队列，拖入文件不再顶替当前结果。

### 📁 时间轴对齐器 GGUF 改存模型目录
- `qwen3-forced-aligner-0.6b-q5_0.gguf`（q5 约 643 MB）此前下载到 `crispasr/`，跟核心混在一起：
  - 换加速版本／升版时，解压流程会先清掉该目录的 `.exe/.dll`，用户「卸载重装、保留配置」时也容易
    把这个大档一起删掉——删了就得重新下 643 MB。
- 现在统一下载到**模型目录**（默认 `ov_models/`），与 Whisper GGML、Qwen3-ASR GGUF 等模型同层：
  核心目录只放核心，模型一律在模型目录，清理核心不再误伤模型。
- 旧版已下载到 `crispasr/` 的档**照样认**（`downloader.find_aligner_gguf` 先查模型目录、再查旧位置），
  升级不会触发重新下载。

### 🧲 CrispASR 核心不再随包附带
- 打包不再拷贝 `crispasr/`（exe + dll + build marker）。首次使用 GPU 核心时，
  `downloader.download_crispasr_core()` 向 GitHub 取**最新 release** 的对应加速版本
  （Vulkan／CUDA 13／CUDA 12／CPU／CPU 兼容版）并解压到 `crispasr/`。
- 这解决了「同捆 0.8.53、上游已 0.8.54 → 一联网就判『版本不符 → 重新下载』」的循环；版本始终跟上游。
- 下载前的硬件检测本来就只走 OS 查询（CIM／nvidia-smi），没有本地 `crispasr.exe` 也能给出推荐加速版本。
- 核心仍是按需下载，不会让只用 CPU（OpenVINO）核心的用户白下几十 MB。

### 📦 onedir / onefile 双格式发行
`build_webview.bat` 一次构建产出两个包（共用同一份 PyInstaller 参数，避免两版漂移）：

| 发行格式 | 产物 | 说明 |
|----------|------|------|
| **onedir** | `dist2\STTMiniTool\` + `dist2\STTMiniTool-v<ver>-windows.zip` | EXE + `_internal/`，零解压、启动快、不占 `%TEMP%`（推荐） |
| **onefile** | `dist2\onefile\STTMiniTool.exe` + `dist2\STTMiniTool-v<ver>-onefile-windows.zip` | 单一 EXE；每次启动把运行时解压到 `%TEMP%\_MEIxxxx`，首次启动较慢、需 C 盘余量 |

两种包都不含 CrispASR 核心与任何模型（模型、ffmpeg、核心均按需下载）。

### 🧵 转录队列（音频页）

- **拖入只入队，不再顶替展示**：旧实现里结果渲染会回头读「当前选中的文件」，转录期间换选文件就会把
  A 的字幕配上 B 的波形/音轨（B 不是文件对象时干脆没有播放）——即反馈的「拖入新视频把画面前的东西
  替换/清除」。现在结果与产生它的那次任务**强绑定**（`renderResult(segments, file)`）。
- **按钮不再乱亮**：`#btn-run` 的可用性由单一 `syncRunButton()` 推导（选中文件 + 未在入队中），
  运行中拖入文件不会把「识别中…」的按钮重新点亮。
- **拖到页面空白处不再劫持界面**：新增 window 级 `dragover/dragenter/drop` 兜底 `preventDefault`，
  避免 WebView2 导航到被拖入的视频文件、把前端整页替换掉。
- **队列面板**：音频页新增「队列」，每行显示状态/进度，可取消、查看、移除；同一时刻只跑一个任务
  （空闲自动开跑、运行中只排队）。
- **结果展示策略**：新任务完成时若结果区为空才自动展示，否则标「查看」，不打断正在校对的结果。
- **暂停/恢复**：刷新或页面断连 → 后端暂停（运行中的转 `paused` 并保留输入暂存，排队的保留），
  回到页面点「继续」按原顺序接着跑。
- **持久化**：任务输入随任务留档——`queue-spool/<sid>/` 存未完成任务（重启后恢复为「排队中 · 队列已暂停」），
  `queue-archive/<jid>/` 存终态任务（input 文件 + 结果 JSON，保留 10 条 / 7 天）。因此刷新、关窗、重启后
  「查看」仍可播放原音轨（`GET /api/queue-audio`，支持 Range），失败/取消可「重试」（复用归档输入，不必重传）。
- **拖动排序**：仅「排队中」的任务可拖（`POST /api/queue-reorder {id, before}`），运行中/已暂停不可改。
- **录音也走队列**：`/api/transcribe` 内部改为「隐藏任务 + 同步等结果」，与批量任务串行执行，不再出现
  「已有识别任务进行中」把录音分段丢掉的情况；隐藏任务不占队列面板、不归档。
- **接口**：`POST /api/enqueue`（上传即入队，立即返回 job_id）、`GET /api/queue`（快照含 paused）、
  `GET /api/queue-result?id=`（最近 5 份结果）、`POST /api/queue-pause|queue-resume|queue-cancel|queue-clear`；
  进度与终态经 SSE（`progress` 带 `jobId`、`queue` 事件）推送。

> 说明：刷新/重启后「查看」会向后端取归档音轨（>300 MB 的文件只显示字幕与时间轴）；归档可随时用「清空已完成」删除。

### 升级说明
- onedir：覆盖安装即可，`settings.json`、模型目录与 `crispasr/` 沿用；已有对齐器 gguf 不必重下。
- onefile：解压得到单个 exe，可放任意英文目录；模型/核心会下载到 exe 旁。
- 首启会重新下载一次 CrispASR 核心（若使用 GPU 核心）——这也是本次改动的主要收益：拿到的是当前最新版。

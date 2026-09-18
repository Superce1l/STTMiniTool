---
name: qwenasr
description: 用本机「语音识别小工具」(QwenASR) 把音频／视频转成字幕，并在取得逐字稿后做校对、摘要或改写。当用户说「转录」「上字幕」「听打」「这段录音讲什么」「帮我做逐字稿」「校对字幕」「摘要这段音频」，或给了 .mp3/.wav/.m4a/.mp4 等文件要求处理内容时使用。全程在本机执行，音频不外传。
---

# QwenASR — 本机语音识别（Agent 用法）

`QwenASR-WebView.exe` 带子命令执行时是**无头 CLI**：不开窗口、执行完即退出、
stdout 只有结果、进度走 stderr。适合直接在 Bash 工具里调用。

## 找到可执行文件

按顺序查找，找到第一个就用（**请用绝对路径调用**）：

1. 用户明说的路径
2. `D:\Project\QwenASRMiniTool\dist2\QwenASR-WebView\QwenASR-WebView.exe`
3. 开发环境（没有 EXE 时）：`D:\Project\QwenASRMiniTool\.venv\Scripts\python.exe D:\Project\QwenASRMiniTool\app_webview.py`

找不到就问用户 EXE 在哪，**不要**自己去装别的 ASR 工具。

## 先确认状态

```bash
"<EXE>" status --json
```
返回已选核心、模型是否就绪、加速版本（`accel.selected` / `accel.recommended`）与
路径诊断（`paths.modelDir` 等）。`selectedReady:false` 代表模型还没下载——
第一次转录会自动下载（0.15～3 GB，视模型而定，需数分钟），先告知用户再继续。

## 转录

```bash
"<EXE>" transcribe "D:\path\audio.mp3" --json
```

stdout 是一份 JSON：

```json
{"input":"...","ok":true,"srtPath":"...\\subtitles\\audio.srt",
 "segments":[{"i":1,"start":0.48,"end":1.84,"text":"会议的第一项议程"}, ...],
 "text":"整段连续逐字稿"}
```

常用选项：

| 选项 | 说明 |
|------|------|
| `-l Chinese` / `Japanese` / `English` | 指定语言（留空＝自动检测；已知语言时指定较准） |
| `--profile zh\|ja\|whisper` | 按用途换模型，**只影响这次执行**，不会改到用户 GUI 的设置 |
| `--hint "人名、术语、背景说明"` | 识别提示，能明显改善专有名词 |
| `--diarize [--speakers N]` | 说话者分离（segments 会多 `speaker` 字段） |
| `--no-align` | 关闭字级时间轴对齐（默认开启） |
| `-o out.srt` / `-f srt\|txt\|json` | 指定输出文件与格式（仅单一输入时有效） |

`--profile` 选哪个（不确定就用默认，别乱换）：

- `zh` 中文标准 — 一般情况的默认，断句标点最完整（Qwen3-ASR-1.7B）
- `ja` 日本语 — 日语／动漫语音（Qwen3-ASR-1.7B 日语动漫特化）
- `whisper` OpenAI Whisper（通用）— 99 语言通用，多语言混合场景稳健

多文件：`transcribe a.mp3 b.mp3 c.mp3 --json` → `{"results":[...]}`。

## 长音频（超过 2 小时）

超过 2 小时的音频会自动用 FFmpeg 切片转录（30 分钟一片、15 秒重叠），转录完
自动合并时间轴，用法与普通文件完全一致，无需额外参数。前提是 ffmpeg 可用
（EXE 旁没有时会自动下载）。

## 校对字幕（重要：时间轴不可碰）

流程是**三步**，中间那步才是你做的事：

```bash
# ① 转录并存成 JSON
"<EXE>" transcribe "audio.mp3" -o work.json -f json --json > result.json

# ② 你读 work.json，只改每个对象的 "text"，存成 fixed.json
#    绝对不要改 i / start / end，也不要增删段落

# ③ 用原始文件当时间轴基准写回 SRT
"<EXE>" apply fixed.json --base work.json -o final.srt
```

`--base` 会**以原始文件的时间轴为准**，只从你的文件取 `text` 按 `i` 对应回填——
就算你不小心动到时间，也会被基准文件覆盖回去。**校对时一律加 `--base`。**

校对时该做与不该做：

- ✅ 错字、同音字、专有名词、人名、公司名、术语
- ✅ 明显的语音误判（「修醉」→「修碎」之类，按上下文判断）
- ❌ 不要润稿、不要改语气、不要删赘字 —— 除非用户明确要求
- ❌ 不要合并或拆分段落（会与时间轴脱节）

## 摘要 / 改写

EXE 只负责交出逐字稿，**摘要是你自己做**。用 `--json` 拿 `text` 字段
（整段连续逐字稿）或 `segments`（要引用时间点时用）直接处理即可，
不需要任何额外命令。要标时间点时引用 `start`，格式化成 `mm:ss` 给用户。

## 注意事项

- **第一次执行会下载模型**（0.15～3 GB，Whisper Large 最大约 3.1 GB）。长音频
  转录可能数分钟 —— 用后台运行或把超时放宽，不要中途重试（会重跑整个文件）。
- **stdout 是干净 JSON**；加载与进度消息都在 stderr。解析前不需要清洗。
- 退出码：`0` 成功／`1` 转录失败／`2` 参数错误／`130` 中断。
- 视频文件（mp4/mkv/…）可直接传入，会自动抽音轨（首次需下载 ffmpeg）。
- 产出的 SRT 默认落在 EXE 旁的 `subtitles\`，路径在 `srtPath`。
- 「未产生字幕（未检测到人声）」通常是素材本身没有语音，或语言选错，
  改用 `--profile whisper` 重试。
- 加速版本（CUDA / Vulkan）：NVIDIA 显卡默认推荐 CUDA；`status --json` 的
  `accel.needsDownload:true` 表示核心待下载，下次加载模型时自动处理。

## 查可用模型

```bash
"<EXE>" profiles --json     # 含本机硬件判断与各 profile 是否已下载（present 字段）
```

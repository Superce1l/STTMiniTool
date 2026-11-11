# [Feature Request] Persistent/streaming transcription mode for realtime use

> 给上游 CrispStrobe/CrispASR 的 feature request 草稿（可直接翻译提交到
> https://github.com/CrispStrobe/CrispASR/issues ）。附完整实测数据。

## Summary

STTMiniTool（本地实时录音转字幕）每句话说完都要为新音频段单独调用一次
`crispasr.exe`。实测发现**进程固定开销占实时出字延迟的 ~90%**，而
crispasr 目前没有可供宿主程序驻留调用的接口。请求提供以下任一形式：

1. **Server / stdin 模式**：进程常驻，模型只加载一次，经 stdin/stdout
   （按行 JSON：`{"file": "...", "id": ...}` → `{"id": ..., "text": ...}`）
   或本地 TCP 端口接受转录请求；
2. **稳定的 C ABI**：把内部的 `core_qwen3_stream` / `qwen3_asr_context`
   流式能力以一组稳定 C 函数导出（`crispasr_context_init / load_model /
   transcribe_samples / free`）；
3. 或者至少文档化 `crispasr.dll` 中已有的流式符号的调用约定。

## Motivation (measured)

环境：Windows 11 / NVIDIA GTX 1660 SUPER 4GB / driver 32.0.16.2114 /
crispasr 0.8.53 / qwen3-asr-1.7b-q8_0.gguf (2.5 GB) / `-nf -bo 1 -fa`。

| 输入 | 模型 | 总耗时 | 拆分 |
|---|---|---|---|
| 0.2 s 静音 | qwen3 Q8 | **2.35 s** | ≈ 全部为进程固定开销（Vulkan init + 权重上显存） |
| 0.2 s 静音 | whisper base | **0.45 s** | 同上比例 |
| 3 s 中文语音 | qwen3 Q8 | 2.5–2.6 s | 固定开销 ≈ 2.35 s，推理 ≈ 0.2 s |
| 3 s 中文语音 | whisper base | 0.45–0.47 s | 推理仅 ~0.02 s/秒音频（RTF≈0.09） |

实时录音场景每说完一句（停顿检测）就要出一次字，2.35 s 的固定开销让
qwen3 Q8 的实时出字滞后约 2.6 s/段——其中 90% 并非推理。冷/热（OS 文件
缓存）差异仅 ~0.28 s，即瓶颈在设备初始化与权重上传显存，而非磁盘。

## Workarounds we tried / rejected

- **多次调用预热**：仅省 ~0.28 s（文件缓存），无济于事；
- **多输入一次调用**（`file0 file1 ...`）：会牺牲「段结束即出字」的实时
  交互，只能作为批处理场景的优化；
- **调用 `crispasr.dll` 的流式符号**（`core_qwen3_stream`、
  `qwen3_asr_context` 等）：导出表 9951 个符号全部为 MSVC C++ mangled
  内部符号、无稳定 C ABI、无头文件——宿主语言（Python）调用需要复刻
  std::string 内存布局与 context 生命周期，随版本升级漂移，不可维护。

## Requested interface (sketch)

```
crispasr_server --model model.gguf --port 8321        # 或
crispasr_context *ctx = crispasr_init(opts);
crispasr_load_model(ctx, "model.gguf");
crispasr_result *r = crispasr_transcribe_samples(ctx, float32_pcm, n, opts);
```

只要模型驻留显存、免去每次 spawn 的 init+load，实时出字延迟可从
~2.6 s/段 降到 ~0.2 s/段（约 12 倍）。happy to test any branch/release.

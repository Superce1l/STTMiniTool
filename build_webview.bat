@echo off
REM =======================================================
REM  Qwen3 ASR - WebView edition build (PyInstaller, ONEDIR)
REM
REM  Builds app_webview.py into a directory-based exe:
REM    dist2\STTMiniTool\STTMiniTool.exe  +  dist2\STTMiniTool\_internal\
REM
REM  WHY ONEDIR (switched back from onefile in 2.2.9): onefile extracted
REM  the whole runtime (several hundred MB) into C:\ %TEMP%\_MEIxxxx at
REM  EVERY launch - users with small C: drives saw their disk fill up
REM  (issue #3). onedir keeps files next to the exe: zero extraction,
REM  faster startup, no temp litter.
REM
REM  WHY IT WAS ONEFILE BEFORE (MOTW): loose DLLs from a downloaded zip
REM  inherit Mark-of-the-Web (Zone.Identifier). .NET refuses to load
REM  MOTW-tagged assemblies, so pythonnet/clr fails and pywebview silently
REM  falls back to Edge. FIX (motw_unblock.py): at startup - BEFORE any
REM  clr load - the app deletes the Zone.Identifier ADS from every file in
REM  its own directory (user-writable, no admin needed). One downside
REM  remains: SmartScreen shows "Run anyway" on first launch of the MOTW-
REM  tagged exe (same as onefile did); the self-unblock clears the exe's
REM  own MOTW too, so later launches start clean.
REM
REM  The WebView UI = local stdlib HTTP server (webview_server.py) that
REM  serves the webview\ folder + /api, opened in a native WebView2 window
REM  via pywebview (URL only, no js_api). Same engine as the CTk edition
REM  (app_webview -> webview_backend -> import app), so ALL engine deps must
REM  be bundled too.
REM
REM  NOTE: ALL content in this .bat is English (Windows CP950 safety).
REM =======================================================

SET SRC=%~dp0
IF "%SRC:~-1%"=="\" SET SRC=%SRC:~0,-1%
IF EXIST "%SRC%\build_venv\Scripts\python.exe" (
    SET VENV=%SRC%\build_venv
) ELSE (
    SET VENV=%SRC%\.venv
)
SET PYTHON=%VENV%\Scripts\python.exe

echo === Step 1: Ensure PyInstaller + pywebview ===
%PYTHON% -m pip install pyinstaller pywebview --quiet

echo.
echo === Step 2: Locate dependency paths ===
FOR /F "delims=" %%i IN ('%PYTHON% -c "import openvino, os; print(os.path.dirname(openvino.__file__))"') DO SET OV_PKG=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import kaldi_native_fbank, os; print(os.path.dirname(kaldi_native_fbank.__file__))"') DO SET KNF_DIR=%%i

echo openvino          : %OV_PKG%
echo kaldi_native_fbank: %KNF_DIR%

IF "%OV_PKG%"==""     GOTO :dep_failed
IF "%KNF_DIR%"==""    GOTO :dep_failed

echo.
echo === Step 2b: Ensure silero_vad_v4.onnx present ===
IF NOT EXIST "%SRC%\ov_models\silero_vad_v4.onnx" (
    echo   downloading silero_vad_v4.onnx...
    %PYTHON% -c "from downloader import _download_file, _VAD_URL; from pathlib import Path; p=Path(r'%SRC%\ov_models'); p.mkdir(exist_ok=True); _download_file(_VAD_URL, p/'silero_vad_v4.onnx')"
)

echo.
echo === Step 3: Build WebView edition (onedir) ===
REM Clean the whole output folder so stale exe/_internal never lingers.
IF EXIST "%SRC%\dist2\STTMiniTool" rmdir /S /Q "%SRC%\dist2\STTMiniTool"
IF EXIST "%SRC%\dist2\STTMiniTool-v*.zip" del /Q "%SRC%\dist2\STTMiniTool-v*.zip"

%PYTHON% -m PyInstaller ^
    --windowed ^
    --name "STTMiniTool" ^
    --distpath "%SRC%\dist2" ^
    --icon "%SRC%\assets\icon.ico" ^
    --add-data "%SRC%\assets\icon.ico;." ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%KNF_DIR%;kaldi_native_fbank" ^
    --add-data "%SRC%\webview;webview_assets" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --add-data "%SRC%\ov_models\silero_vad_v4.onnx;ov_models" ^
    --add-data "%SRC%\app.py;." ^
    --add-data "%SRC%\webview_backend.py;." ^
    --add-data "%SRC%\webview_server.py;." ^
    --add-data "%SRC%\engine_types.py;." ^
    --add-data "%SRC%\settings_store.py;." ^
    --add-data "%SRC%\ffmpeg_utils.py;." ^
    --add-data "%SRC%\audio_io.py;." ^
    --add-data "%SRC%\version.py;." ^
    --add-data "%SRC%\fa_aligner.py;." ^
    --add-data "%SRC%\crisp_engine.py;." ^
    --add-data "%SRC%\subtitle_lines.py;." ^
    --add-data "%SRC%\cli_mode.py;." ^
    --add-data "%SRC%\chatllm_engine.py;." ^
    --add-data "%SRC%\diarize.py;." ^
    --add-data "%SRC%\processor_numpy.py;." ^
    --add-data "%SRC%\proc_guard.py;." ^
    --add-data "%SRC%\applog.py;." ^
    --add-data "%SRC%\motw_unblock.py;." ^
    --runtime-hook "%SRC%\runtime_hook_utf8.py" ^
    --collect-all webview ^
    --collect-all pythonnet ^
    --collect-all clr_loader ^
    --collect-data certifi ^
    --collect-all tokenizers ^
    --collect-all scipy ^
    --hidden-import certifi ^
    --hidden-import webview ^
    --hidden-import webview.platforms.edgechromium ^
    --hidden-import webview.platforms.winforms ^
    --hidden-import clr ^
    --hidden-import clr_loader ^
    --hidden-import pythonnet ^
    --hidden-import proxy_tools ^
    --hidden-import app ^
    --hidden-import webview_backend ^
    --hidden-import webview_server ^
    --hidden-import engine_types ^
    --hidden-import settings_store ^
    --hidden-import openvino ^
    --hidden-import openvino.runtime ^
    --hidden-import onnxruntime ^
    --hidden-import soundfile ^
    --hidden-import scipy ^
    --hidden-import scipy.cluster ^
    --hidden-import scipy.cluster.hierarchy ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.spatial.distance ^
    --hidden-import scipy._lib.messagestream ^
    --hidden-import crisp_engine ^
    --hidden-import subtitle_lines ^
    --hidden-import cli_mode ^
    --hidden-import version ^
    --hidden-import fa_aligner ^
    --hidden-import proc_guard ^
    --hidden-import applog ^
    --hidden-import motw_unblock ^
    --hidden-import diarize ^
    --hidden-import audio_io ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module transformers ^
    --exclude-module qwen_asr ^
    --exclude-module triton ^
    --exclude-module bitsandbytes ^
    --noconfirm ^
    %SRC%\app_webview.py

IF ERRORLEVEL 1 GOTO :build_failed
IF NOT EXIST "%SRC%\dist2\STTMiniTool\STTMiniTool.exe" GOTO :build_failed

echo.
echo ===================================================
echo  WebView build SUCCESS - copying crispasr core
echo ===================================================
REM --- chatllm: only main.exe needed at runtime --------------------------------
REM The chatllm core is NOT bundled; chatllm_engine.py is included because app.py
REM imports its diagnostic helpers (format_vad_diag / detect_degenerate_asr).
REM chatllm main.exe + FA .bin are still used at runtime by the OpenVINO
REM karaoke word-level aligner (fa_aligner.ChatLLMAligner).
REM
REM --- crispasr core: BUNDLED (exe + dll only, NO models) ----------------
REM We ship the CrispASR core binaries so hardware detection
REM (crispasr.exe --diagnostics) and the Vulkan GPU path work out of the box on
REM the model page, without a first-run 27MB core download. Only .exe + .dll are
REM copied; the large model/aligner files (*.bin, *.gguf) are NOT bundled - they
REM download on demand when a GPU model is selected.
IF NOT EXIST "%SRC%\crispasr\crispasr.exe" GOTO :after_crispasr
IF NOT EXIST "%SRC%\dist2\STTMiniTool\crispasr\" mkdir "%SRC%\dist2\STTMiniTool\crispasr\"
xcopy "%SRC%\crispasr\*.exe" "%SRC%\dist2\STTMiniTool\crispasr\" /Y /Q
xcopy "%SRC%\crispasr\*.dll" "%SRC%\dist2\STTMiniTool\crispasr\" /Y /Q
REM Build marker (crispasr_build.json) records version + acceleration variant.
REM Without it the app treats the bundled core as outdated and re-downloads it
REM on first model load. Keep the source folder in sync with the release.
IF EXIST "%SRC%\crispasr\crispasr_build.json" xcopy "%SRC%\crispasr\crispasr_build.json" "%SRC%\dist2\STTMiniTool\crispasr\" /Y /Q
echo  crispasr core (exe+dll+build marker) copied
:after_crispasr

REM --- ffmpeg: NOT bundled, downloaded on demand ------------------------
REM ffmpeg is fetched + unzipped on first video transcription from
REM https://huggingface.co/dseditor/Collection/resolve/main/ffmpeg.zip
REM (downloader.download_ffmpeg), so no ffmpeg.exe is copied here.

echo.
echo === Step 4: Package release zip (STTMiniTool-vVERSION-windows.zip) ===
REM Zip the app folder itself (top-level STTMiniTool\ inside the zip) with
REM PowerShell Compress-Archive. Users extract -\> STTMiniTool\ and run
REM STTMiniTool.exe inside; first launch self-unblocks MOTW (motw_unblock.py).
FOR /F "usebackq delims=" %%v IN (`%PYTHON% -c "import version; print(version.__version__)"`) DO SET VER=%%v
IF "%VER%"=="" GOTO :zip_failed
echo  version: %VER%
IF EXIST "%SRC%\dist2\STTMiniTool-v%VER%-windows.zip" del /Q "%SRC%\dist2\STTMiniTool-v%VER%-windows.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path '%SRC%\dist2\STTMiniTool' -DestinationPath '%SRC%\dist2\STTMiniTool-v%VER%-windows.zip' -CompressionLevel Optimal"
IF ERRORLEVEL 1 GOTO :zip_failed

echo.
echo  Done: dist2\STTMiniTool\STTMiniTool.exe
echo        dist2\STTMiniTool-v%VER%-windows.zip
GOTO :eof

:zip_failed
echo.
echo  ERROR: release zip packaging FAILED.
EXIT /B 1

:dep_failed
echo.
echo  ERROR: a dependency path is empty (import failed). Aborting.
EXIT /B 1

:build_failed
echo.
echo  ERROR: PyInstaller build FAILED.
EXIT /B 1

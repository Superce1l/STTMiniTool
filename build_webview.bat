@echo off
REM =======================================================
REM  STTMiniTool - WebView edition build (PyInstaller)
REM
REM  Produces BOTH release flavors in one run (issue #7 item 3):
REM    dist2\STTMiniTool\STTMiniTool.exe + _internal\   (ONEDIR)
REM    dist2\onefile\STTMiniTool.exe                    (ONEFILE)
REM  plus two zips:
REM    dist2\STTMiniTool-v<ver>-windows.zip          (ONEDIR, folder inside)
REM    dist2\STTMiniTool-v<ver>-onefile-windows.zip  (ONEFILE, single exe)
REM
REM  WHY BOTH:
REM    ONEDIR  - files sit next to the exe: zero extraction, fast start,
REM              no temp litter (issue #3: onefile used to unpack hundreds
REM              of MB into C:\%TEMP%\_MEIxxxx on EVERY launch).
REM    ONEFILE - one portable exe for users who want a single file; it
REM              still extracts the runtime to %TEMP% on each launch.
REM  Let the user pick and document the tradeoff (README).
REM
REM  CrispASR IS NOT BUNDLED (issue #7 item 2):
REM    The GPU core (crispasr.exe + dlls) is fetched at first use from the
REM    LATEST GitHub release by downloader.download_crispasr_core(), so
REM    users always run a current core instead of a copy frozen at build
REM    time (bundling 0.8.53 while 0.8.54 was out produced "version
REM    mismatch" re-downloads). Nothing from crispasr\ is copied here.
REM    Models (whisper / qwen3-asr / forced-aligner gguf) live in the
REM    model directory and are downloaded on demand - they are NOT shipped
REM    in the zip either.
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

REM ---------------------------------------------------------------------
REM  Shared PyInstaller arguments (same payload for both flavors).
REM  Kept in one variable so onedir / onefile can never drift apart.
REM ---------------------------------------------------------------------
SET SHARED=--windowed --name "STTMiniTool" --icon "%SRC%\assets\icon.ico" --add-data "%SRC%\assets\icon.ico;." --add-data "%OV_PKG%;openvino" --add-data "%KNF_DIR%;kaldi_native_fbank" --add-data "%SRC%\webview;webview_assets" --add-data "%SRC%\prompt_template.json;." --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" --add-data "%SRC%\ov_models\silero_vad_v4.onnx;ov_models" --add-data "%SRC%\app.py;." --add-data "%SRC%\webview_backend.py;." --add-data "%SRC%\webview_server.py;." --add-data "%SRC%\engine_types.py;." --add-data "%SRC%\settings_store.py;." --add-data "%SRC%\ffmpeg_utils.py;." --add-data "%SRC%\audio_io.py;." --add-data "%SRC%\version.py;." --add-data "%SRC%\fa_aligner.py;." --add-data "%SRC%\crisp_engine.py;." --add-data "%SRC%\subtitle_lines.py;." --add-data "%SRC%\cli_mode.py;." --add-data "%SRC%\chatllm_engine.py;." --add-data "%SRC%\diarize.py;." --add-data "%SRC%\processor_numpy.py;." --add-data "%SRC%\proc_guard.py;." --add-data "%SRC%\applog.py;." --add-data "%SRC%\motw_unblock.py;." --runtime-hook "%SRC%\runtime_hook_utf8.py" --collect-all webview --collect-all pythonnet --collect-all clr_loader --collect-data certifi --collect-all tokenizers --collect-all scipy --hidden-import certifi --hidden-import webview --hidden-import webview.platforms.edgechromium --hidden-import webview.platforms.winforms --hidden-import clr --hidden-import clr_loader --hidden-import pythonnet --hidden-import proxy_tools --hidden-import app --hidden-import webview_backend --hidden-import webview_server --hidden-import engine_types --hidden-import settings_store --hidden-import openvino --hidden-import openvino.runtime --hidden-import onnxruntime --hidden-import soundfile --hidden-import scipy --hidden-import scipy.cluster --hidden-import scipy.cluster.hierarchy --hidden-import scipy.spatial --hidden-import scipy.spatial.distance --hidden-import scipy._lib.messagestream --hidden-import crisp_engine --hidden-import subtitle_lines --hidden-import cli_mode --hidden-import version --hidden-import fa_aligner --hidden-import proc_guard --hidden-import applog --hidden-import motw_unblock --hidden-import diarize --hidden-import audio_io --exclude-module torch --exclude-module torchvision --exclude-module torchaudio --exclude-module transformers --exclude-module qwen_asr --exclude-module triton --exclude-module bitsandbytes

REM Clean the output folders so a stale exe/_internal never lingers.
IF EXIST "%SRC%\dist2\STTMiniTool" rmdir /S /Q "%SRC%\dist2\STTMiniTool"
IF EXIST "%SRC%\dist2\onefile" rmdir /S /Q "%SRC%\dist2\onefile"
IF EXIST "%SRC%\dist2\STTMiniTool-v*-windows.zip" del /Q "%SRC%\dist2\STTMiniTool-v*-windows.zip"
IF EXIST "%SRC%\dist2\STTMiniTool-v*-onefile-windows.zip" del /Q "%SRC%\dist2\STTMiniTool-v*-onefile-windows.zip"

echo.
echo === Step 3a: Build WebView edition (ONEDIR) ===
%PYTHON% -m PyInstaller %SHARED% ^
    --onedir ^
    --distpath "%SRC%\dist2" ^
    --workpath "%SRC%\build\pkg-onedir" ^
    --specpath "%SRC%\build\spec-onedir" ^
    --noconfirm ^
    "%SRC%\app_webview.py"

IF ERRORLEVEL 1 GOTO :build_failed
IF NOT EXIST "%SRC%\dist2\STTMiniTool\STTMiniTool.exe" GOTO :build_failed

echo.
echo === Step 3b: Build WebView edition (ONEFILE) ===
%PYTHON% -m PyInstaller %SHARED% ^
    --onefile ^
    --distpath "%SRC%\dist2\onefile" ^
    --workpath "%SRC%\build\pkg-onefile" ^
    --specpath "%SRC%\build\spec-onefile" ^
    --noconfirm ^
    "%SRC%\app_webview.py"

IF ERRORLEVEL 1 GOTO :build_failed
IF NOT EXIST "%SRC%\dist2\onefile\STTMiniTool.exe" GOTO :build_failed

REM --- guard: CrispASR core must NOT be inside the release -------------
REM  (it is downloaded at first use; a leftover copy would freeze the
REM   version and resurrect issue #7). Warning only: a dev tree may still
REM   keep crispasr\ next to the sources.
IF EXIST "%SRC%\dist2\STTMiniTool\crispasr" echo  WARNING: crispasr\ found in onedir output - release must not bundle the core!
IF EXIST "%SRC%\dist2\STTMiniTool\_internal\crispasr" echo  WARNING: crispasr\ found in _internal - release must not bundle the core!

echo.
echo ===================================================
echo  WebView build SUCCESS (onedir + onefile)
echo ===================================================
REM --- ffmpeg: NOT bundled, downloaded on demand -----------------------
REM ffmpeg is fetched + unzipped on first video transcription from
REM https://huggingface.co/dseditor/Collection/resolve/main/ffmpeg.zip
REM (downloader.download_ffmpeg), so no ffmpeg.exe is copied here.
REM --- CrispASR: NOT bundled, downloaded on demand ---------------------
REM downloader.download_crispasr_core() pulls the LATEST GitHub release
REM (variant: vulkan / cuda13 / cuda12 / cpu / cpu-legacy) on first model
REM load. Hardware detection before that download uses OS queries only
REM (downloader.recommend_crispasr_variant), so no local exe is needed.

echo.
echo === Step 4: Package release zips ===
FOR /F "delims=" %%v IN ('%PYTHON% -c "import version; print(version.__version__)"') DO SET VER=%%v
IF "%VER%"=="" GOTO :zip_failed
echo  version: %VER%

REM ONEDIR zip: folder STTMiniTool\ at the top -> run STTMiniTool\STTMiniTool.exe
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path '%SRC%\dist2\STTMiniTool' -DestinationPath '%SRC%\dist2\STTMiniTool-v%VER%-windows.zip' -CompressionLevel Optimal"
IF ERRORLEVEL 1 GOTO :zip_failed

REM ONEFILE zip: single exe at the top -> unzip and run STTMiniTool.exe
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path '%SRC%\dist2\onefile\STTMiniTool.exe' -DestinationPath '%SRC%\dist2\STTMiniTool-v%VER%-onefile-windows.zip' -CompressionLevel Optimal"
IF ERRORLEVEL 1 GOTO :zip_failed

echo.
echo  Done (ONEDIR) : dist2\STTMiniTool\STTMiniTool.exe
echo        zip   : dist2\STTMiniTool-v%VER%-windows.zip
echo  Done (ONEFILE): dist2\onefile\STTMiniTool.exe
echo        zip   : dist2\STTMiniTool-v%VER%-onefile-windows.zip
GOTO :eof

:dep_failed
echo.
echo  ERROR: could not locate openvino / kaldi_native_fbank in the venv.
GOTO :fail

:build_failed
echo.
echo  ERROR: PyInstaller build failed (see messages above).
GOTO :fail

:zip_failed
echo.
echo  ERROR: release zip could not be produced.
GOTO :fail

:fail
exit /b 1

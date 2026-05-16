@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%..\app.py" if exist "%SCRIPT_DIR%..\src\" (
    cd /d "%SCRIPT_DIR%.."
) else (
    cd /d "%SCRIPT_DIR%GGF-DramaBox"
)

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run install-GGF-DramaBox.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate
set "DRAMABOX_MODEL_DIR=%CD%\models"
set "LTX_DISTILLED_DIR=%DRAMABOX_MODEL_DIR%\ltx-distilled-1.1"
set "LTX_DISTILLED_FILE=ltx-2.3-22b-distilled-1.1.safetensors"
set "LTX_DISTILLED_PATH=%LTX_DISTILLED_DIR%\%LTX_DISTILLED_FILE%"

set "CURL_EXE="
for /f "usebackq delims=" %%C in (`where curl 2^>nul`) do if not defined CURL_EXE set "CURL_EXE=%%C"

echo.
echo GGF-DramaBox launch menu
echo   1. Default DramaBox
echo   2. Official LTX-2.3 distilled v1.1
echo   3. Custom local LTX / distilled model paths
set /p "MODEL_CHOICE=Choose profile [2]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=2"

set /p "DRAMABOX_PORT=Port [7862]: "
if "%DRAMABOX_PORT%"=="" set "DRAMABOX_PORT=7862"

if "%MODEL_CHOICE%"=="2" (
    set "DRAMABOX_MODEL_PROFILE=ltx-distilled"
    set "DRAMABOX_MODEL_TYPE=distilled"
    call :ensure_ltx_distilled
    if errorlevel 1 (
        echo [ERROR] LTX distilled download failed.
        pause
        exit /b 1
    )
) else if "%MODEL_CHOICE%"=="3" (
    set "DRAMABOX_MODEL_PROFILE=custom"
    echo.
    echo Paste local paths for the custom model.
    set /p "DRAMABOX_TRANSFORMER_PATH=Transformer/audio-only checkpoint: "
    set /p "DRAMABOX_AUDIO_COMPONENTS_PATH=Audio components or full checkpoint: "
    set /p "DRAMABOX_GEMMA_ROOT=Gemma root folder: "
    set /p "DRAMABOX_MODEL_TYPE=Model type [distilled/dev/dramabox/auto]: "
    if "!DRAMABOX_MODEL_TYPE!"=="" set "DRAMABOX_MODEL_TYPE=distilled"
) else (
    set "DRAMABOX_MODEL_PROFILE=dramabox"
    set "DRAMABOX_MODEL_TYPE=dramabox"
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [WARN] NVIDIA GPU was not detected. DramaBox is intended for CUDA GPUs.
)

echo.
echo Starting GGF-DramaBox on http://127.0.0.1:%DRAMABOX_PORT%
start "" "http://127.0.0.1:%DRAMABOX_PORT%"
python app.py --port %DRAMABOX_PORT%

pause
exit /b 0

:ensure_ltx_distilled
if exist "%LTX_DISTILLED_PATH%" (
    echo [DOWNLOAD] resuming/checking %LTX_DISTILLED_PATH%
    set "CURL_RESUME=-C -"
) else (
    set "CURL_RESUME="
)

if not defined CURL_EXE (
    echo [ERROR] curl.exe was not found. Cannot download LTX distilled.
    exit /b 1
)

if not exist "%LTX_DISTILLED_DIR%" mkdir "%LTX_DISTILLED_DIR%"
echo [DOWNLOAD] curl -Lo %LTX_DISTILLED_PATH%
"%CURL_EXE%" %CURL_RESUME% -Lo "%LTX_DISTILLED_PATH%" --fail --retry 12 --retry-delay 5 --retry-all-errors --connect-timeout 60 "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors"
if errorlevel 1 exit /b 1
if not exist "%LTX_DISTILLED_PATH%" exit /b 1
for %%F in ("%LTX_DISTILLED_PATH%") do if %%~zF EQU 0 exit /b 1
exit /b 0

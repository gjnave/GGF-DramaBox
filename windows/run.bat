@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0GGF-DramaBox"
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run install-GGF-DramaBox.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo GGF-DramaBox model profile
echo   1. Default DramaBox from Hugging Face
echo   2. Official LTX-2.3 distilled v1.1 from Hugging Face
echo   3. Custom local LTX / distilled model paths
set /p "MODEL_CHOICE=Choose profile [1]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"

if "%MODEL_CHOICE%"=="2" (
    set "DRAMABOX_MODEL_PROFILE=ltx-distilled"
    set "DRAMABOX_MODEL_TYPE=distilled"
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

set /p "DRAMABOX_PORT=Port [7862]: "
if "%DRAMABOX_PORT%"=="" set "DRAMABOX_PORT=7862"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [WARN] NVIDIA GPU was not detected. DramaBox is intended for CUDA GPUs.
)

echo.
echo Starting GGF-DramaBox on http://127.0.0.1:%DRAMABOX_PORT%
start "" "http://127.0.0.1:%DRAMABOX_PORT%"
python app.py --port %DRAMABOX_PORT%

pause

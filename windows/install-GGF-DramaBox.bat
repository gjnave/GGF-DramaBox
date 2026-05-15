@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0"

IF EXIST "disclaimer.md" ( TYPE "disclaimer.md" & pause )
IF EXIST "about.nfo" TYPE "about.nfo"

echo.
echo Optional: paste a Hugging Face token for higher rate limits.
echo Leave blank to skip.
set /p "HF_TOKEN=HF token (optional): "
if defined HF_TOKEN (
    set "HF_TOKEN=%HF_TOKEN%"
    set "HUGGINGFACE_HUB_TOKEN=%HF_TOKEN%"
)

if exist "GGF-DramaBox\.git" (
    echo.
    echo Existing GGF-DramaBox checkout found. Pulling latest...
    cd /d "%~dp0GGF-DramaBox"
    git pull
) else (
    echo.
    echo Cloning GGF-DramaBox...
    git clone https://github.com/gjnave/GGF-DramaBox
    if errorlevel 1 ( echo [ERROR] Git clone failed. & pause & exit /b 1 )
    cd /d "%~dp0GGF-DramaBox"
)

:: Find Python
set "BASE_PYTHON="
for /f "usebackq delims=" %%P in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do set "BASE_PYTHON=%%P"
if not defined BASE_PYTHON for /f "usebackq delims=" %%P in (`where python 2^>nul`) do set "BASE_PYTHON=%%P"
if not defined BASE_PYTHON ( echo [ERROR] Python 3 not found. Install Python 3.10+ and rerun. & pause & exit /b 1 )

:: Create venv
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    "%BASE_PYTHON%" -m venv venv
    if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
)

call venv\Scripts\activate

:: Deps
python -m pip install --upgrade pip wheel setuptools

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo NVIDIA GPU detected -- installing CUDA PyTorch...
    pip install --upgrade torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
) else (
    echo [WARN] NVIDIA GPU was not detected. Installing CPU PyTorch for setup only.
    echo [WARN] DramaBox expects CUDA. Runtime CPU launch requires DRAMABOX_ALLOW_CPU=1 and will be very slow.
    pip install --upgrade torch==2.8.0 torchaudio==2.8.0
)
if errorlevel 1 ( echo [ERROR] PyTorch install failed. & pause & exit /b 1 )

pip install --upgrade "huggingface_hub[hf_xet]>=0.35.0" "hf_xet>=1.1.0" "hf_transfer>=0.1.9"
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] Python dependency install failed. & pause & exit /b 1 )

:: Download default model assets
set "HF_HUB_ENABLE_HF_TRANSFER=1"
echo.
echo Downloading DramaBox model assets. This is large and can take a while.
python app.py --download-only
if errorlevel 1 ( echo [ERROR] Download failed. & pause & exit /b 1 )

echo.
echo Optional: also download the official LTX-2.3 distilled v1.1 checkpoint now.
echo This is large, but it makes run.bat option 2 ready without pasting paths.
set /p "INSTALL_DISTILLED=Download LTX distilled too? [y/N]: "
if /i "%INSTALL_DISTILLED%"=="Y" (
    set "DRAMABOX_MODEL_PROFILE=ltx-distilled"
    set "DRAMABOX_MODEL_TYPE=distilled"
    python app.py --download-only
    if errorlevel 1 ( echo [ERROR] Distilled download failed. & pause & exit /b 1 )
    set "DRAMABOX_MODEL_PROFILE=dramabox"
    set "DRAMABOX_MODEL_TYPE=dramabox"
)

:: Run
echo.
echo Starting GGF-DramaBox on http://127.0.0.1:7862
start "" "http://127.0.0.1:7862"
set "DRAMABOX_PORT=7862"
python app.py --port 7862

pause

@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0"

IF EXIST "disclaimer.md" ( TYPE "disclaimer.md" & pause )
IF EXIST "about.nfo" TYPE "about.nfo"

echo.
echo Optional: paste a Hugging Face token for private/gated/rate-limited downloads.
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

set "DRAMABOX_MODEL_DIR=%CD%\models"
set "DRAMABOX_DIR=%DRAMABOX_MODEL_DIR%\dramabox"
set "GEMMA_DIR=%DRAMABOX_MODEL_DIR%\gemma-3-12b-it-bnb-4bit"
set "LTX_DISTILLED_DIR=%DRAMABOX_MODEL_DIR%\ltx-distilled-1.1"

if exist "%~dp0aria2c.exe" (
    set "ARIA2C=%~dp0aria2c.exe"
) else (
    set "ARIA2C="
    for /f "usebackq delims=" %%A in (`where aria2c 2^>nul`) do if not defined ARIA2C set "ARIA2C=%%A"
)

set "CURL_EXE="
for /f "usebackq delims=" %%C in (`where curl 2^>nul`) do if not defined CURL_EXE set "CURL_EXE=%%C"

if defined ARIA2C (
    echo Downloader: aria2c
) else (
    if not defined CURL_EXE (
        echo [ERROR] Neither aria2c.exe nor curl.exe was found.
        pause
        exit /b 1
    )
    echo Downloader: curl
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

pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] Python dependency install failed. & pause & exit /b 1 )

echo.
echo Installing model files into:
echo   %DRAMABOX_MODEL_DIR%
echo.

call :download "https://huggingface.co/ResembleAI/Dramabox/resolve/main/dramabox-dit-v1.safetensors" "%DRAMABOX_DIR%" "dramabox-dit-v1.safetensors"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/ResembleAI/Dramabox/resolve/main/dramabox-audio-components.safetensors" "%DRAMABOX_DIR%" "dramabox-audio-components.safetensors"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/ResembleAI/Dramabox/resolve/main/assets/silence_latent_frame.pt" "%DRAMABOX_DIR%\assets" "silence_latent_frame.pt"
if errorlevel 1 goto :download_failed

call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/added_tokens.json" "%GEMMA_DIR%" "added_tokens.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/chat_template.jinja" "%GEMMA_DIR%" "chat_template.jinja"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/chat_template.json" "%GEMMA_DIR%" "chat_template.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/config.json" "%GEMMA_DIR%" "config.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/generation_config.json" "%GEMMA_DIR%" "generation_config.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/model-00001-of-00002.safetensors" "%GEMMA_DIR%" "model-00001-of-00002.safetensors"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/model-00002-of-00002.safetensors" "%GEMMA_DIR%" "model-00002-of-00002.safetensors"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/model.safetensors.index.json" "%GEMMA_DIR%" "model.safetensors.index.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/preprocessor_config.json" "%GEMMA_DIR%" "preprocessor_config.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/processor_config.json" "%GEMMA_DIR%" "processor_config.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/special_tokens_map.json" "%GEMMA_DIR%" "special_tokens_map.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/tokenizer.json" "%GEMMA_DIR%" "tokenizer.json"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/tokenizer.model" "%GEMMA_DIR%" "tokenizer.model"
if errorlevel 1 goto :download_failed
call :download "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main/tokenizer_config.json" "%GEMMA_DIR%" "tokenizer_config.json"
if errorlevel 1 goto :download_failed

echo.
echo Optional: also download the official LTX-2.3 distilled v1.1 checkpoint now.
set /p "INSTALL_DISTILLED=Download LTX distilled too? [y/N]: "
if /i "%INSTALL_DISTILLED%"=="Y" (
    call :download "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors" "%LTX_DISTILLED_DIR%" "ltx-2.3-22b-distilled-1.1.safetensors"
    if errorlevel 1 goto :download_failed
)

echo.
echo Install complete.
echo Models are installed under:
echo   %DRAMABOX_MODEL_DIR%
echo.
set /p "LAUNCH_NOW=Launch GGF-DramaBox now? [Y/n]: "
if /i not "%LAUNCH_NOW%"=="N" (
    echo Starting GGF-DramaBox on http://127.0.0.1:7862
    start "" "http://127.0.0.1:7862"
    set "DRAMABOX_PORT=7862"
    python app.py --port 7862
)

pause
exit /b 0

:download
set "URL=%~1"
set "DEST_DIR=%~2"
set "DEST_FILE=%~3"
set "OUT=%DEST_DIR%\%DEST_FILE%"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
if exist "%OUT%" (
    for %%F in ("%OUT%") do if %%~zF GTR 0 (
        echo [OK] %OUT%
        exit /b 0
    )
)
echo [DOWNLOAD] %DEST_FILE%
if defined ARIA2C (
    if defined HF_TOKEN (
        "%ARIA2C%" -c -x 8 -s 8 -k 1M --allow-overwrite=true --auto-file-renaming=false --header="Authorization: Bearer %HF_TOKEN%" -d "%DEST_DIR%" -o "%DEST_FILE%" "%URL%"
    ) else (
        "%ARIA2C%" -c -x 8 -s 8 -k 1M --allow-overwrite=true --auto-file-renaming=false -d "%DEST_DIR%" -o "%DEST_FILE%" "%URL%"
    )
) else (
    if defined HF_TOKEN (
        "%CURL_EXE%" --fail --retry 5 --retry-delay 2 -H "Authorization: Bearer %HF_TOKEN%" -Lo "%OUT%" "%URL%"
    ) else (
        "%CURL_EXE%" --fail --retry 5 --retry-delay 2 -Lo "%OUT%" "%URL%"
    )
)
if errorlevel 1 exit /b 1
if not exist "%OUT%" exit /b 1
for %%F in ("%OUT%") do if %%~zF EQU 0 exit /b 1
exit /b 0

:download_failed
echo.
echo [ERROR] Model download failed.
echo Check the URL/token/network above, then rerun this installer. Completed files will be skipped.
pause
exit /b 1

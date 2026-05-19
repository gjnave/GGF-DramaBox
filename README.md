# GGF-DramaBox

GGF-DramaBox is a local Windows/GPU wrapper for
[Resemble AI DramaBox](https://github.com/resemble-ai/DramaBox), an expressive
prompt-driven TTS and voice-cloning model built on LTX-2.3 audio.

The app keeps a warm `TTSServer` in GPU memory, exposes a Gradio interface for
scene prompting, and adds a model switch path for later LTX/distilled
checkpoint swaps.

## Hardware

- NVIDIA GPU strongly recommended.
- The default DramaBox stack can peak around 24 GB VRAM.
- CPU fallback is blocked by default because it is impractically slow. Set
  `DRAMABOX_ALLOW_CPU=1` only for testing startup behavior.

## Installed Layout

The standalone installer expects this shape:

```text
some-folder/
  install-GGF-DramaBox.bat
  run.bat
  GGF-DramaBox/
    app.py
    src/
```

The installer downloads model files into:

```text
GGF-DramaBox/
  models/
    dramabox/
    gemma-3-12b-it-bnb-4bit/
    ltx-distilled-1.1/   optional
```

The standalone `.bat` installer downloads those files directly. It uses
`aria2c.exe` from the installer folder or PATH when available; otherwise it
falls back to `curl -L -o`. If `aria2c` starts but fails on a TLS/network error,
the same file is retried with `curl -Lo`. The optional LTX distilled checkpoint
uses `curl -Lo` first because that Hugging Face redirect is more reliable with
curl on Windows.
`run.bat` also uses `curl -Lo` for option 2 before the app starts, so the app
does not fall back to Python's Hugging Face HEAD request for that checkpoint.
The Python runtime only resolves local files; it does not use the Hugging Face
download API for models.

The installer also refreshes the parent-level `run.bat` from the repo copy, so
rerunning the installer updates the member-facing launcher.

## Model Switching

Default launch downloads/uses:

- `ResembleAI/Dramabox` transformer and audio components
- `unsloth/gemma-3-12b-it-bnb-4bit` text encoder

`run.bat` defaults to the faster DramaBox profile. Option 2 downloads/runs the
official LTX-2.3 distilled v1.1 checkpoint
from `Lightricks/LTX-2.3`:

```bat
set DRAMABOX_MODEL_PROFILE=ltx-distilled
set DRAMABOX_MODEL_TYPE=distilled
```

The installer can pre-download this model too. It uses:

- `ltx-2.3-22b-distilled-1.1.safetensors`

To use a different local or experimental LTX model later, choose the custom
model option in `run.bat`, or set:

```bat
set DRAMABOX_MODEL_PROFILE=custom
set DRAMABOX_MODEL_TYPE=distilled
set DRAMABOX_TRANSFORMER_PATH=D:\models\ltx-distilled-audio-only.safetensors
set DRAMABOX_AUDIO_COMPONENTS_PATH=D:\models\ltx-2.3-22b-distilled.safetensors
set DRAMABOX_GEMMA_ROOT=D:\models\gemma-3-12b-it-bnb-4bit
```

The in-app **Model Switch** tab can also reload a local checkpoint without
editing files. Distilled mode uses the distilled sigma schedule and no guidance
by default; DramaBox mode uses the expressive guided defaults.

## LoRA Training

The app includes a **LoRA Training** tab that turns your dataset paths and
training settings into the exact preprocess, train, and test-inference commands.
It supports manifest JSONL, TSV, `gemini_synthetic`, and `libriheavy` indexes.
`src/preprocess.py` also writes a training-ready `index.txt` beside the
preprocessed latents, so manifest and TSV datasets can go straight into
`src/train.py`.

Keep trained LoRAs separate and load them with `--lora` during inference.
Pre-merging LoRAs into the checkpoint is not recommended.

## Direct Run

```bat
call venv\Scripts\activate
python app.py --port 7862
```

For a model-only prefetch:

```bat
windows\install-GGF-DramaBox.bat
```

## License

DramaBox is distributed under the LTX-2 Community License Agreement inherited
from the upstream project. See `LICENSE`.

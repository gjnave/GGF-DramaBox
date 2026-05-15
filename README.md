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

## Model Switching

Default launch downloads/uses:

- `ResembleAI/Dramabox` transformer and audio components
- `unsloth/gemma-3-12b-it-bnb-4bit` text encoder

To use a local or distilled LTX model later, choose the custom model option in
`run.bat`, or set:

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

## Direct Run

```bat
call venv\Scripts\activate
python app.py --port 7862
```

For a model-only prefetch:

```bat
python app.py --download-only
```

## License

DramaBox is distributed under the LTX-2 Community License Agreement inherited
from the upstream project. See `LICENSE`.

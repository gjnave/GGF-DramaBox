#!/usr/bin/env python3
"""GGF-DramaBox local GPU app."""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from inference_server import TTSServer, model_defaults  # noqa: E402
from model_downloader import get_all_paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROMPTS = [
    (
        "Cold villain",
        'A shadowy villain speaks with cold menace, "You have entered my domain, mortal." '
        'He chuckles darkly, "Such arrogance will be your undoing." '
        'His voice rises with fury, "Kneel, or be destroyed where you stand!"',
    ),
    (
        "Tender whisper",
        'A woman speaks tenderly, "It has been a long day, my love." '
        'She whispers, "Close your eyes. I am right here." '
        'She hums quietly, "Mmmm. Sleep now."',
    ),
    (
        "Radio host",
        'A late-night radio host clears his throat, "Good evening, dear listeners." '
        'He settles into a warm smoky tone, "The rain is tapping at my window like an old friend." '
        'He chuckles softly, "Stay right where you are."',
    ),
    (
        "Fraying patience",
        'An exhausted father speaks with fraying patience, "Sweetie, daddy is asking very nicely." '
        'He sighs deeply, "Ohhhh my goodness." '
        'Then he laughs helplessly, "Hahaha, I am losing my mind."',
    ),
]

VOICE_DIR = ROOT / "assets" / "voices"
VOICE_EXAMPLES = [
    [str(VOICE_DIR / "male_harvey_keitel.mp3"), PROMPTS[0][1]],
    [str(VOICE_DIR / "female_shadowheart.wav"), PROMPTS[1][1]],
    [str(VOICE_DIR / "male_old_movie.wav"), PROMPTS[2][1]],
    [str(VOICE_DIR / "male_petergriffin.wav"), PROMPTS[3][1]],
]

MODEL_LOCK = threading.Lock()
SERVER: TTSServer | None = None
PATHS: dict = {}


def _device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU fallback"


def _load_server(paths: dict | None = None, dtype: str | None = None, bnb_4bit: bool = True) -> TTSServer:
    global PATHS
    PATHS = paths or get_all_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and os.environ.get("DRAMABOX_ALLOW_CPU") != "1":
        raise RuntimeError(
            "CUDA was not detected. DramaBox is designed for NVIDIA GPU use. "
            "Install CUDA PyTorch or set DRAMABOX_ALLOW_CPU=1 to force a very slow CPU run."
        )
    return TTSServer(
        checkpoint=PATHS["transformer"],
        full_checkpoint=PATHS["audio_components"],
        gemma_root=PATHS["gemma_root"],
        device=device,
        dtype=dtype or os.environ.get("LTX_DTYPE", "bf16"),
        compile_model=os.environ.get("DRAMABOX_COMPILE", "0") == "1",
        bnb_4bit=bnb_4bit,
        model_type=PATHS.get("model_type", "dramabox"),
    )


def _ensure_server() -> TTSServer:
    global SERVER
    if SERVER is None:
        with MODEL_LOCK:
            if SERVER is None:
                SERVER = _load_server()
    return SERVER


def _status_markdown() -> str:
    server = _ensure_server()
    defaults = server.default_params
    return (
        f"**Loaded model:** `{server.model_type}`  \n"
        f"**Device:** `{_device_name()}`  \n"
        f"**Profile:** `{PATHS.get('profile', 'dramabox')}`  \n"
        f"**Transformer:** `{PATHS.get('transformer', '')}`  \n"
        f"**Audio/full checkpoint:** `{PATHS.get('audio_components', '')}`  \n"
        f"**Defaults:** cfg `{defaults['cfg_scale']}`, stg `{defaults['stg_scale']}`, "
        f"steps `{'distilled' if defaults['steps'] == 0 else defaults['steps']}`"
    )


def generate(
    prompt: str,
    audio_ref,
    cfg_scale: float,
    stg_scale: float,
    duration_multiplier: float,
    target_duration: float,
    ref_duration: float,
    seed: int,
    auto_rescale: bool,
    rescale_scale: float,
    modality_scale: float,
    steps: int,
    watermark: bool,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty.")
    server = _ensure_server()
    ref_path = str(audio_ref) if audio_ref and os.path.exists(str(audio_ref)) else None
    out_path = tempfile.mktemp(prefix="ggf_dramabox_", suffix=".wav", dir=str(OUTPUT_DIR))
    t0 = time.time()
    with MODEL_LOCK:
        result = server.generate_to_file(
            prompt=prompt.strip(),
            output=out_path,
            voice_ref=ref_path,
            cfg_scale=float(cfg_scale),
            stg_scale=float(stg_scale),
            duration_multiplier=float(duration_multiplier),
            gen_duration=float(target_duration),
            ref_duration=float(ref_duration),
            seed=int(seed),
            rescale_scale="auto" if auto_rescale else float(rescale_scale),
            modality_scale=float(modality_scale),
            steps=int(steps),
            watermark=bool(watermark),
        )
    elapsed = time.time() - t0
    return result, f"Generated in {elapsed:.1f}s: `{Path(result).name}`"


def load_custom_model(
    transformer_path: str,
    audio_components_path: str,
    gemma_root: str,
    model_type: str,
    dtype: str,
    bnb_4bit: bool,
):
    global SERVER, PATHS
    if not transformer_path or not audio_components_path or not gemma_root:
        raise gr.Error("Custom model switching needs transformer, audio/full checkpoint, and Gemma root paths.")

    paths = {
        "profile": "custom",
        "model_type": model_type,
        "transformer": transformer_path.strip().strip('"'),
        "audio_components": audio_components_path.strip().strip('"'),
        "gemma_root": gemma_root.strip().strip('"'),
    }
    for key in ("transformer", "audio_components"):
        if not os.path.isfile(paths[key]):
            raise gr.Error(f"{key} not found: {paths[key]}")
    if not os.path.isdir(paths["gemma_root"]):
        raise gr.Error(f"Gemma root not found: {paths['gemma_root']}")

    with MODEL_LOCK:
        SERVER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        SERVER = _load_server(paths=paths, dtype=dtype, bnb_4bit=bnb_4bit)

    defaults = model_defaults(SERVER.model_type)
    return (
        _status_markdown(),
        gr.update(value=defaults["cfg_scale"]),
        gr.update(value=defaults["stg_scale"]),
        gr.update(value=defaults["rescale_scale"] == "auto"),
        gr.update(value=0.0 if defaults["rescale_scale"] == "auto" else defaults["rescale_scale"]),
        gr.update(value=defaults["modality_scale"]),
        gr.update(value=defaults["steps"]),
    )


def apply_prompt(choice: str):
    for name, prompt in PROMPTS:
        if name == choice:
            return prompt
    return PROMPTS[0][1]


def _quote_cmd(value: str | Path | int | float) -> str:
    text = str(value).strip()
    if not text:
        return '""'
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace('"', r'\"')
    if any(ch in text for ch in (" ", "&", "(", ")", "[", "]", "{", "}", '"')):
        return f'"{text}"'
    return text


def build_lora_commands(
    dataset_type: str,
    index_path: str,
    audio_dir: str,
    preprocessed_dir: str,
    min_duration: float,
    max_duration: float,
    max_samples: int,
    train_output_dir: str,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    warmup_steps: int,
    save_every: int,
    use_validation: bool,
    validation_config: str,
    lora_path: str,
    voice_sample: str,
    test_prompt: str,
    test_output: str,
):
    dataset_type = dataset_type or "manifest"
    preprocessed_dir = preprocessed_dir.strip().strip('"') or str(ROOT / "training_data" / "preprocessed")
    train_output_dir = train_output_dir.strip().strip('"') or str(ROOT / "training_runs" / "dramabox_lora")
    validation_config = validation_config.strip().strip('"') or str(ROOT / "configs" / "val_config.example.yaml")
    lora_path = lora_path.strip().strip('"') or str(Path(train_output_dir) / f"lora_step_{int(save_every):05d}.safetensors")
    test_output = test_output.strip().strip('"') or str(ROOT / "output" / "lora_test.wav")
    test_prompt = test_prompt.strip() or 'A woman speaks warmly, "Hello from my trained LoRA."'

    checkpoint = ROOT / "models" / "dramabox" / "dramabox-audio-components.safetensors"
    audio_only = ROOT / "models" / "dramabox" / "dramabox-dit-v1.safetensors"
    gemma_root = ROOT / "models" / "gemma-3-12b-it-bnb-4bit"
    speaker_index = Path(preprocessed_dir) / "index.txt"
    example_index = "D:\\path\\to\\your_data.jsonl"
    example_audio_dir = "D:\\path\\to\\wavs"
    example_voice_sample = "D:\\path\\to\\reference.wav"

    preprocess_parts = [
        "python src\\preprocess.py",
        f"--dataset-type {dataset_type}",
        f"--index {_quote_cmd(index_path or example_index)}",
        f"--audio-dir {_quote_cmd(audio_dir or example_audio_dir)}",
        f"--output-dir {_quote_cmd(preprocessed_dir)}",
        f"--checkpoint {_quote_cmd(checkpoint)}",
        f"--audio-only-ckpt {_quote_cmd(audio_only)}",
        f"--gemma-root {_quote_cmd(gemma_root)}",
        f"--max-duration {float(max_duration):.1f}",
        f"--min-duration {float(min_duration):.1f}",
        "--skip-existing",
    ]
    if int(max_samples) > 0:
        preprocess_parts.append(f"--max-samples {int(max_samples)}")

    train_parts = [
        "accelerate launch src\\train.py",
        "--config configs\\training_args.example.yaml",
        f"--data-dir {_quote_cmd(preprocessed_dir)}",
        f"--speaker-index {_quote_cmd(speaker_index)}",
        f"--checkpoint {_quote_cmd(audio_only)}",
        f"--full-checkpoint {_quote_cmd(checkpoint)}",
        "--base-model dev",
        f"--output-dir {_quote_cmd(train_output_dir)}",
        f"--steps {int(steps)}",
        f"--lr {float(learning_rate):.2e}",
        f"--warmup-steps {int(warmup_steps)}",
        f"--save-every {int(save_every)}",
        f"--lora-rank {int(lora_rank)}",
        f"--lora-alpha {int(lora_alpha)}",
        f"--lora-dropout {float(lora_dropout):.2f}",
    ]
    if use_validation:
        train_parts.append(f"--val-config {_quote_cmd(validation_config)}")

    infer_parts = [
        "python src\\inference.py",
        f"--checkpoint {_quote_cmd(audio_only)}",
        f"--full-checkpoint {_quote_cmd(checkpoint)}",
        f"--gemma-root {_quote_cmd(gemma_root)}",
        f"--lora {_quote_cmd(lora_path)}",
        f"--lora-rank {int(lora_rank)}",
        f"--voice-sample {_quote_cmd(voice_sample or example_voice_sample)}",
        f"--prompt {_quote_cmd(test_prompt)}",
        f"--output {_quote_cmd(test_output)}",
    ]

    def join_parts(parts: list[str]) -> str:
        return " ^\n  ".join(parts)

    manifest = (
        '{"audio_filepath": "wavs/spk01_001.wav", "speaker": "spk01", '
        '"text": "A woman speaks warmly, \\"Hello, how are you today?\\""}\n'
        '{"audio_filepath": "wavs/spk01_002.wav", "speaker": "spk01", '
        '"text": "Hello, how are you today?", "duration": 3.8}'
    )
    notes = (
        f"Preprocess writes `{speaker_index}` for training. Keep at least two usable samples per speaker. "
        "Load the LoRA at inference time with `--lora`; do not pre-merge it into the checkpoint."
    )
    return join_parts(preprocess_parts), join_parts(train_parts), join_parts(infer_parts), manifest, notes


CSS = """
:root {
    --ggf-ink: #14120f;
    --ggf-muted: #5f594f;
    --ggf-line: #d8d0c3;
    --ggf-paper: #fff9ed;
    --ggf-cream: #f4ead7;
    --ggf-red: #d83b2a;
    --ggf-blue: #174c62;
    --ggf-gold: #d99b2b;
}
body, .gradio-container {
    background:
        linear-gradient(135deg, rgba(216, 59, 42, 0.08), transparent 34%),
        linear-gradient(315deg, rgba(23, 76, 98, 0.12), transparent 42%),
        #fbf6eb !important;
    color: var(--ggf-ink);
}
.gradio-container {
    max-width: 1240px !important;
}
.brand-hero {
    border: 1px solid rgba(20, 18, 15, 0.12);
    border-radius: 8px;
    padding: 28px 30px;
    background:
        linear-gradient(120deg, rgba(20, 18, 15, 0.94), rgba(23, 76, 98, 0.9)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.07) 0, rgba(255,255,255,0.07) 1px, transparent 1px, transparent 18px);
    color: #fff8ec;
    box-shadow: 0 18px 44px rgba(20, 18, 15, 0.16);
}
.brand-lockup {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.brand-badge {
    display: inline-flex;
    width: 44px;
    height: 44px;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: var(--ggf-red);
    color: #fff8ec;
    font-weight: 900;
    box-shadow: inset 0 -4px 0 rgba(0, 0, 0, 0.18);
}
.brand-hero h1 {
    margin: 0;
    font-size: 40px;
    line-height: 1.02;
    color: #fff8ec !important;
}
.brand-kicker {
    color: #f5c35d;
    font-weight: 800;
    letter-spacing: 0;
    margin: 0;
}
.brand-copy {
    max-width: 780px;
    color: #f4ead7;
    font-size: 17px;
    margin: 14px 0 12px;
}
.brand-links a {
    color: #ffcf68 !important;
    font-weight: 700;
}
.brand-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}
.brand-chip {
    border: 1px solid rgba(255, 248, 236, 0.22);
    border-radius: 999px;
    padding: 6px 10px;
    color: #fff8ec;
    background: rgba(255, 248, 236, 0.08);
    font-size: 13px;
}
.action-strip {
    border: 1px solid var(--ggf-line);
    border-radius: 8px;
    padding: 14px;
    background: rgba(255, 249, 237, 0.82);
}
.panel {
    border: 1px solid var(--ggf-line);
    border-radius: 8px;
    padding: 16px;
    background: rgba(255, 252, 245, 0.9);
}
.guide-panel {
    border: 1px solid #cfc3af;
    border-radius: 8px;
    padding: 22px 24px;
    background: linear-gradient(180deg, rgba(255,255,255,0.58), rgba(255,255,255,0.2)), var(--ggf-paper);
}
.guide-panel h2 {
    margin-top: 0;
    color: var(--ggf-blue);
}
.guide-panel code {
    background: #efe1c7;
    border-radius: 5px;
    padding: 2px 5px;
}
.prompt-note {
    margin: 8px 0 14px;
    color: var(--ggf-muted);
    font-size: 15px;
    font-style: italic;
}
.prompt-heading {
    margin: 0 0 4px;
    color: var(--ggf-ink);
    font-weight: 800;
}
.status-box textarea {
    min-height: 42px !important;
    max-height: 58px !important;
    overflow-y: auto !important;
    font-size: 13px !important;
}
.generate-button button, .generate-button, button.primary, .gradio-button.primary {
    background: linear-gradient(180deg, #e24a38, var(--ggf-red)) !important;
    border-color: #b92f22 !important;
    color: #fff8ec !important;
    font-weight: 900 !important;
    box-shadow: 0 12px 22px rgba(216, 59, 42, 0.22) !important;
}
.utility-button button, .utility-button {
    border: 1px solid #cfc3af !important;
    background: #fffaf0 !important;
    color: var(--ggf-ink) !important;
    font-weight: 800 !important;
}
.utility-button button:hover, .utility-button:hover {
    border-color: var(--ggf-gold) !important;
}
.tabs, .tab-nav, .gradio-tabs {
    border-color: var(--ggf-line) !important;
}
"""


def build_app() -> gr.Blocks:
    server = _ensure_server()
    defaults = server.default_params
    auto_rescale_default = defaults["rescale_scale"] == "auto"
    rescale_default = 0.0 if auto_rescale_default else float(defaults["rescale_scale"])

    with gr.Blocks(
        title="GGF-DramaBox",
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="stone"),
        css=CSS,
        analytics_enabled=False,
    ) as app:
        gr.HTML(
            "<div class='brand-hero'>"
            "<div class='brand-lockup'><div class='brand-badge'>GGF</div>"
            "<p class='brand-kicker'>GET GOING FAST</p></div>"
            "<h1>DramaBox</h1>"
            "<p class='brand-copy'>Expressive local GPU voice cloning with warm-server generation, "
            "stage-direction prompting, and fast model switching when you want to try distilled LTX.</p>"
            "<p class='brand-links'>Built around Resemble AI DramaBox and packaged in the Get Going Fast style.</p>"
            "<div class='brand-chips'>"
            "<span class='brand-chip'>NVIDIA GPU workflow</span>"
            "<span class='brand-chip'>Voice reference support</span>"
            "<span class='brand-chip'>Local models first</span>"
            "<span class='brand-chip'>Distilled swap ready</span>"
            "</div>"
            "</div>"
        )

        with gr.Tabs():
            with gr.Tab("Generate"):
                gr.Markdown(
                    "<p class='prompt-note'>Write the scene like direction plus dialogue. "
                    "Put spoken words inside quotes, keep actions outside, and use the voice reference when you want tighter character matching.</p>"
                )
                with gr.Group(elem_classes=["action-strip"]):
                    with gr.Row():
                        with gr.Column(scale=3):
                            prompt_choice = gr.Dropdown(
                                [name for name, _ in PROMPTS],
                                label="Prompt starter",
                                value=PROMPTS[0][0],
                            )
                            prompt_box = gr.Textbox(
                                label="Scene prompt",
                                value=PROMPTS[0][1],
                                lines=7,
                                placeholder='A narrator speaks warmly, "Hello from DramaBox."',
                            )
                            prompt_choice.change(apply_prompt, inputs=[prompt_choice], outputs=[prompt_box])
                            audio_ref = gr.Audio(label="Voice reference (optional, 10+ seconds)", type="filepath")
                            generate_btn = gr.Button("Generate", variant="primary", size="lg", elem_classes=["generate-button"])

                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["panel"]):
                                audio_out = gr.Audio(label="Generated audio", type="filepath")
                                run_status = gr.Textbox(
                                    value="Ready.",
                                    label="Status",
                                    interactive=False,
                                    elem_classes=["status-box"],
                                )
                            with gr.Group(elem_classes=["panel"]):
                                gr.Markdown("### Runtime")
                                gr.Markdown(_status_markdown())

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Generation Controls")
                    with gr.Row():
                        cfg_scale = gr.Slider(1.0, 10.0, value=defaults["cfg_scale"], step=0.25, label="CFG scale")
                        stg_scale = gr.Slider(0.0, 5.0, value=defaults["stg_scale"], step=0.25, label="STG scale")
                        steps = gr.Slider(0, 60, value=defaults["steps"], step=1, label="Steps (0 = distilled schedule)")
                    with gr.Row():
                        duration_multiplier = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Auto duration multiplier")
                        target_duration = gr.Slider(0.0, 60.0, value=0.0, step=1.0, label="Target duration, seconds (0 = auto)")
                        ref_duration = gr.Slider(3.0, 30.0, value=10.0, step=1.0, label="Reference seconds")
                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        modality_scale = gr.Slider(1.0, 5.0, value=defaults["modality_scale"], step=0.25, label="Modality scale")
                        auto_rescale = gr.Checkbox(value=auto_rescale_default, label="Auto CFG rescale")
                        rescale_scale = gr.Slider(0.0, 1.0, value=rescale_default, step=0.05, label="Manual rescale")
                        watermark = gr.Checkbox(value=True, label="Watermark")

                generate_btn.click(
                    generate,
                    inputs=[
                        prompt_box,
                        audio_ref,
                        cfg_scale,
                        stg_scale,
                        duration_multiplier,
                        target_duration,
                        ref_duration,
                        seed,
                        auto_rescale,
                        rescale_scale,
                        modality_scale,
                        steps,
                        watermark,
                    ],
                    outputs=[audio_out, run_status],
                )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Quick Starts")
                    gr.Examples(
                        examples=VOICE_EXAMPLES,
                        inputs=[audio_ref, prompt_box],
                        label="Voice + prompt examples",
                        cache_examples=False,
                    )

            with gr.Tab("Model Switch"):
                with gr.Group(elem_classes=["guide-panel"]):
                    gr.Markdown(
                        "## Model Switch\n"
                        "Swap in a local LTX or distilled checkpoint here. Reloading takes a bit because the current "
                        "GPU runtime is released before the new one is loaded warm."
                    )
                    custom_transformer = gr.Textbox(label="Transformer/audio-only checkpoint path")
                    custom_audio = gr.Textbox(label="Audio components or full checkpoint path")
                    custom_gemma = gr.Textbox(label="Gemma root directory")
                    with gr.Row():
                        custom_type = gr.Dropdown(
                            ["dramabox", "distilled", "dev", "auto"],
                            value="distilled",
                            label="Model type",
                        )
                        custom_dtype = gr.Dropdown(["bf16", "fp16"], value=os.environ.get("LTX_DTYPE", "bf16"), label="Dtype")
                        custom_bnb = gr.Checkbox(value=True, label="Gemma bnb 4-bit")
                    reload_btn = gr.Button("Load custom model", variant="primary", elem_classes=["generate-button"])
                    switch_status = gr.Markdown(_status_markdown())
                    reload_btn.click(
                        load_custom_model,
                        inputs=[
                            custom_transformer,
                            custom_audio,
                            custom_gemma,
                            custom_type,
                            custom_dtype,
                            custom_bnb,
                        ],
                        outputs=[
                            switch_status,
                            cfg_scale,
                            stg_scale,
                            auto_rescale,
                            rescale_scale,
                            modality_scale,
                            steps,
                        ],
                    )

            with gr.Tab("LoRA Training"):
                with gr.Group(elem_classes=["guide-panel"]):
                    gr.Markdown(
                        "## Train A DramaBox LoRA\n"
                        "Fine-tune a speaker, language flavor, or style on top of DramaBox. Prepare paired audio and text, "
                        "preprocess it into latents, then launch training with Accelerate. Keep the final LoRA separate and "
                        "load it at inference time."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            lora_dataset_type = gr.Dropdown(
                                ["manifest", "tsv", "gemini_synthetic", "libriheavy"],
                                value="manifest",
                                label="Index format",
                            )
                            lora_index = gr.Textbox(label="Index / manifest path")
                            lora_audio_dir = gr.Textbox(label="Audio folder")
                            lora_preprocessed = gr.Textbox(
                                label="Preprocessed output folder",
                                value=str(ROOT / "training_data" / "preprocessed"),
                            )
                        with gr.Column(scale=1):
                            lora_min_duration = gr.Number(value=2.0, label="Min seconds")
                            lora_max_duration = gr.Number(value=20.0, label="Max seconds")
                            lora_max_samples = gr.Number(value=0, label="Max samples (0 = all)", precision=0)
                            lora_train_output = gr.Textbox(
                                label="Training output folder",
                                value=str(ROOT / "training_runs" / "dramabox_lora"),
                            )

                    with gr.Row():
                        lora_steps = gr.Number(value=10000, label="Steps", precision=0)
                        lora_rank = gr.Number(value=128, label="Rank", precision=0)
                        lora_alpha = gr.Number(value=128, label="Alpha", precision=0)
                        lora_dropout = gr.Number(value=0.1, label="Dropout")
                    with gr.Row():
                        lora_lr = gr.Number(value=1e-4, label="Learning rate")
                        lora_warmup = gr.Number(value=500, label="Warmup steps", precision=0)
                        lora_save_every = gr.Number(value=500, label="Save every", precision=0)
                        lora_use_val = gr.Checkbox(value=False, label="Run validation at saves")
                    lora_val_config = gr.Textbox(
                        label="Validation config",
                        value=str(ROOT / "configs" / "val_config.example.yaml"),
                    )

                    with gr.Row():
                        lora_test_path = gr.Textbox(label="LoRA path to test after training")
                        lora_voice_sample = gr.Textbox(label="Reference voice sample")
                    lora_test_prompt = gr.Textbox(
                        label="Test prompt",
                        value='A woman speaks warmly, "Hello from my trained LoRA."',
                        lines=2,
                    )
                    lora_test_output = gr.Textbox(
                        label="Test output WAV",
                        value=str(ROOT / "output" / "lora_test.wav"),
                    )

                    lora_build_btn = gr.Button("Build commands", variant="primary", elem_classes=["generate-button"])
                    lora_notes = gr.Markdown()
                    lora_manifest_example = gr.Textbox(
                        label="Manifest JSONL example",
                        lines=3,
                        interactive=False,
                    )
                    lora_preprocess_cmd = gr.Textbox(
                        label="1. Preprocess command",
                        lines=9,
                        interactive=False,
                    )
                    lora_train_cmd = gr.Textbox(
                        label="2. Train command",
                        lines=12,
                        interactive=False,
                    )
                    lora_infer_cmd = gr.Textbox(
                        label="3. Test inference command",
                        lines=9,
                        interactive=False,
                    )
                    gr.Markdown(
                        "### Accepted Index Formats\n"
                        "**Manifest JSONL:** `{\"audio_filepath\": \"wavs/spk01_001.wav\", \"speaker\": \"spk01\", \"text\": \"A woman speaks warmly, \\\"Hello.\\\"\"}`\n\n"
                        "**TSV:** `wavs/spk01_001.wav<TAB>A woman speaks warmly, \"Hello.\"`\n\n"
                        "**gemini_synthetic:** `id~speaker~lang~sr~samples~dur~phonemes~text`\n\n"
                        "**libriheavy:** `id~speaker~lang~samples~dur_ms~phonemes~text`\n\n"
                        "Use a scene wrapper like `A woman speaks warmly, \"<transcript>\"` when you want the LoRA to learn the same prompted style used at inference."
                    )
                    lora_build_btn.click(
                        build_lora_commands,
                        inputs=[
                            lora_dataset_type,
                            lora_index,
                            lora_audio_dir,
                            lora_preprocessed,
                            lora_min_duration,
                            lora_max_duration,
                            lora_max_samples,
                            lora_train_output,
                            lora_steps,
                            lora_rank,
                            lora_alpha,
                            lora_dropout,
                            lora_lr,
                            lora_warmup,
                            lora_save_every,
                            lora_use_val,
                            lora_val_config,
                            lora_test_path,
                            lora_voice_sample,
                            lora_test_prompt,
                            lora_test_output,
                        ],
                        outputs=[
                            lora_preprocess_cmd,
                            lora_train_cmd,
                            lora_infer_cmd,
                            lora_manifest_example,
                            lora_notes,
                        ],
                    )

            with gr.Tab("Prompt Guide"):
                with gr.Group(elem_classes=["guide-panel"]):
                    gr.Markdown(
                        "## Prompt Guide\n"
                        "Use a speaker description, then put spoken words inside double quotes. Put actions outside quotes.\n\n"
                        "**Good:** `A tired detective mutters, \"This case is not over.\" He sighs deeply. \"Not by a long shot.\"`\n\n"
                        "**Inside quotes:** dialogue and phonetic sounds like `\"Hahaha\"`, `\"Mmmm\"`, `\"Ugh\"`.\n\n"
                        "**Outside quotes:** `She laughs nervously.`, `A long pause.`, `He clears his throat.`"
                    )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GGF-DramaBox local app")
    parser.add_argument("--download-only", action="store_true", help="Resolve/download model assets and exit.")
    parser.add_argument(
        "--profile",
        choices=["dramabox", "ltx-distilled", "all"],
        default=os.environ.get("DRAMABOX_MODEL_PROFILE", "dramabox"),
        help="Model profile to install when using --download-only.",
    )
    parser.add_argument("--host", default=os.environ.get("DRAMABOX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DRAMABOX_PORT", "7862")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.download_only:
        original_profile = os.environ.get("DRAMABOX_MODEL_PROFILE")
        profiles = ["dramabox", "ltx-distilled"] if args.profile == "all" else [args.profile]
        for profile in profiles:
            print(f"\nInstalling model profile: {profile}")
            os.environ["DRAMABOX_MODEL_PROFILE"] = profile
            os.environ["DRAMABOX_MODEL_TYPE"] = "distilled" if profile == "ltx-distilled" else "dramabox"
            get_all_paths()
        if original_profile is not None:
            os.environ["DRAMABOX_MODEL_PROFILE"] = original_profile
        print("Download/resolve complete.")
        raise SystemExit(0)

    _ensure_server()
    build_app().queue(max_size=12).launch(
        server_name=args.host,
        server_port=args.port,
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        show_api=False,
    )

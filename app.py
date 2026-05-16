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


CSS = """
.gradio-container { max-width: 1240px !important; }
.ggf-hero {
    border: 1px solid #24324a;
    background: #111827;
    color: #f8fafc;
    padding: 18px 20px;
    border-radius: 8px;
}
.ggf-hero h1 { margin: 0 0 6px 0; font-size: 28px; }
.ggf-hero p { margin: 0; color: #cbd5e1; }
.ggf-note { color: #94a3b8; font-size: 13px; }
"""


def build_app() -> gr.Blocks:
    server = _ensure_server()
    defaults = server.default_params
    auto_rescale_default = defaults["rescale_scale"] == "auto"
    rescale_default = 0.0 if auto_rescale_default else float(defaults["rescale_scale"])

    with gr.Blocks(
        title="GGF-DramaBox",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        analytics_enabled=False,
    ) as app:
        gr.HTML(
            "<div class='ggf-hero'>"
            "<h1>GGF-DramaBox</h1>"
            "<p>Local GPU expressive TTS with voice cloning, warm-server generation, "
            "and swappable DramaBox/LTX checkpoints.</p>"
            "</div>"
        )

        with gr.Tabs():
            with gr.Tab("Generate"):
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
                        generate_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        audio_out = gr.Audio(label="Generated audio", type="filepath")
                        run_status = gr.Markdown("Ready.")
                        with gr.Accordion("Model status", open=True):
                            gr.Markdown(_status_markdown())

                with gr.Accordion("Generation controls", open=True):
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

                gr.Examples(
                    examples=VOICE_EXAMPLES,
                    inputs=[audio_ref, prompt_box],
                    label="Voice + prompt examples",
                    cache_examples=False,
                )

            with gr.Tab("Model Switch"):
                gr.Markdown(
                    "Switch to a local LTX or distilled checkpoint here. The reload can take a while "
                    "because the current GPU model is released and the new one is loaded warm."
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
                reload_btn = gr.Button("Load custom model", variant="primary")
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

            with gr.Tab("Prompt Guide"):
                gr.Markdown(
                    "Use a speaker description, then put spoken words inside double quotes. "
                    "Put actions outside quotes.\n\n"
                    "Good: `A tired detective mutters, \"This case is not over.\" "
                    "He sighs deeply. \"Not by a long shot.\"`\n\n"
                    "Inside quotes: dialogue and phonetic sounds like `\"Hahaha\"`, `\"Mmmm\"`, `\"Ugh\"`.\n\n"
                    "Outside quotes: `She laughs nervously.`, `A long pause.`, `He clears his throat.`"
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

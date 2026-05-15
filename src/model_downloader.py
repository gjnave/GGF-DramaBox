#!/usr/bin/env python3
"""Download and resolve DramaBox/LTX model assets.

The default profile downloads ResembleAI/DramaBox from Hugging Face. For later
LTX swaps, set the DRAMABOX_* path variables or choose the custom option in
run.bat. That lets the app boot with a distilled/local checkpoint without
editing Python files.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent

DRAMABOX_REPO = "ResembleAI/Dramabox"
LTX_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"

DEFAULT_CACHE = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~")),
    ".cache",
    "dramabox",
)
DEFAULT_MODEL_DIR = APP_DIR / "models"

MODEL_FILES = {
    "transformer": "dramabox-dit-v1.safetensors",
    "audio_components": "dramabox-audio-components.safetensors",
    "silence_latent": "assets/silence_latent_frame.pt",
}

LTX_DISTILLED_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip().strip('"')
    return None


def _require_file(path: str, label: str) -> str:
    resolved = str(Path(path).expanduser())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _require_dir(path: str, label: str) -> str:
    resolved = str(Path(path).expanduser())
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _model_base_dir() -> Path:
    return Path(os.environ.get("DRAMABOX_MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()


def _use_hf_cache_only() -> bool:
    return os.environ.get("DRAMABOX_USE_HF_CACHE", "0") == "1"


def _download_file(repo_id: str, filename: str, subdir: str, cache_dir: str | None = None) -> str:
    """Download a Hugging Face file into the local standalone models folder."""
    if _use_hf_cache_only():
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir or os.environ.get("DRAMABOX_CACHE_DIR") or DEFAULT_CACHE,
            token=_token(),
        )

    local_dir = _model_base_dir() / subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=_token(),
    )


def _download_snapshot(repo_id: str, subdir: str, cache_dir: str | None = None) -> str:
    """Download a Hugging Face repository snapshot into the local models folder."""
    if _use_hf_cache_only():
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir or os.environ.get("DRAMABOX_CACHE_DIR") or DEFAULT_CACHE,
            token=_token(),
        )

    local_dir = _model_base_dir() / subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=_token(),
    )


def get_model_path(name: str, cache_dir: str | None = None) -> str:
    """Download one default DramaBox model file and return its local path."""
    cache_dir = cache_dir or os.environ.get("DRAMABOX_CACHE_DIR") or DEFAULT_CACHE

    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_FILES)}")

    repo_id = os.environ.get("DRAMABOX_HF_REPO", DRAMABOX_REPO)
    repo_path = os.environ.get(f"DRAMABOX_{name.upper()}_FILE", MODEL_FILES[name])
    logger.info("Fetching %s from %s/%s...", name, repo_id, repo_path)

    local_path = _download_file(repo_id, repo_path, "dramabox", cache_dir)
    logger.info("  -> %s", local_path)
    return local_path


def get_ltx_distilled_path(cache_dir: str | None = None) -> str:
    """Download the official LTX-2.3 distilled v1.1 checkpoint."""
    cache_dir = cache_dir or os.environ.get("DRAMABOX_CACHE_DIR") or DEFAULT_CACHE
    repo_id = os.environ.get("DRAMABOX_LTX_REPO", LTX_REPO)
    filename = os.environ.get("DRAMABOX_LTX_DISTILLED_FILE", LTX_DISTILLED_FILE)
    logger.info("Fetching official LTX distilled checkpoint from %s/%s...", repo_id, filename)
    local_path = _download_file(repo_id, filename, "ltx-distilled-1.1", cache_dir)
    logger.info("  -> %s", local_path)
    return local_path


def get_gemma_path(cache_dir: str | None = None) -> str:
    """Download or resolve the Gemma text encoder directory."""
    override = _first_env("DRAMABOX_GEMMA_ROOT", "GEMMA_DIR")
    if override:
        return _require_dir(override, "Gemma root")

    cache_dir = cache_dir or os.environ.get("DRAMABOX_CACHE_DIR") or DEFAULT_CACHE
    repo_id = os.environ.get("DRAMABOX_GEMMA_REPO", GEMMA_REPO)
    logger.info("Fetching Gemma from %s...", repo_id)

    local_dir = _download_snapshot(repo_id, "gemma-3-12b-it-bnb-4bit", cache_dir)
    logger.info("  -> %s", local_dir)
    return local_dir


def get_all_paths(cache_dir: str | None = None) -> dict:
    """Resolve all assets needed by TTSServer.

    Environment overrides:
      DRAMABOX_MODEL_PROFILE=dramabox|ltx-distilled|custom
      DRAMABOX_TRANSFORMER_PATH=/path/to/audio-only.safetensors
      DRAMABOX_AUDIO_COMPONENTS_PATH=/path/to/full-or-components.safetensors
      DRAMABOX_GEMMA_ROOT=/path/to/gemma
      DRAMABOX_MODEL_TYPE=dramabox|dev|distilled|auto
    """
    cache_dir = cache_dir or os.environ.get("DRAMABOX_CACHE_DIR") or DEFAULT_CACHE
    profile = os.environ.get("DRAMABOX_MODEL_PROFILE", "dramabox").strip().lower()

    transformer_override = _first_env(
        "DRAMABOX_TRANSFORMER_PATH",
        "LTX_TRANSFORMER_PATH",
        "LTX_CHECKPOINT",
    )
    audio_override = _first_env(
        "DRAMABOX_AUDIO_COMPONENTS_PATH",
        "DRAMABOX_FULL_CHECKPOINT_PATH",
        "LTX_FULL_CHECKPOINT_PATH",
        "LTX_FULL_CHECKPOINT",
    )

    paths: dict[str, str] = {
        "profile": profile,
        "model_type": os.environ.get("DRAMABOX_MODEL_TYPE", "dramabox").strip().lower(),
    }

    if profile in {"ltx-distilled", "distilled"}:
        distilled_path = get_ltx_distilled_path(cache_dir)
        paths["profile"] = "ltx-distilled"
        paths["model_type"] = "distilled"
        # The warm server needs both a transformer checkpoint and the audio
        # component/full checkpoint. The official distilled file is the full
        # LTX checkpoint, so use it for both slots.
        paths["transformer"] = distilled_path
        paths["audio_components"] = distilled_path
        paths["silence_latent"] = get_model_path("silence_latent", cache_dir)
        paths["gemma_root"] = get_gemma_path(cache_dir)
        return paths

    if profile == "custom" or transformer_override or audio_override:
        if not transformer_override or not audio_override:
            raise RuntimeError(
                "Custom model mode needs both DRAMABOX_TRANSFORMER_PATH and "
                "DRAMABOX_AUDIO_COMPONENTS_PATH."
            )
        paths["transformer"] = _require_file(transformer_override, "Transformer checkpoint")
        paths["audio_components"] = _require_file(audio_override, "Audio/full checkpoint")
        silence_override = _first_env("DRAMABOX_SILENCE_LATENT_PATH")
        if silence_override:
            paths["silence_latent"] = _require_file(silence_override, "Silence latent")
        else:
            paths["silence_latent"] = get_model_path("silence_latent", cache_dir)
        paths["gemma_root"] = get_gemma_path(cache_dir)
        return paths

    for name in MODEL_FILES:
        paths[name] = get_model_path(name, cache_dir)

    paths["gemma_root"] = get_gemma_path(cache_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    resolved = get_all_paths()
    print("\nAll model assets resolved:")
    for key, value in resolved.items():
        if key in {"profile", "model_type"}:
            print(f"  {key}: {value}")
            continue
        size = os.path.getsize(value) / 1e9 if os.path.isfile(value) else None
        if size is None:
            print(f"  {key}: {value} (directory)")
        else:
            print(f"  {key}: {value} ({size:.2f} GB)")

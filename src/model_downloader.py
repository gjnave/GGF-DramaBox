#!/usr/bin/env python3
"""Resolve locally installed DramaBox/LTX model assets.

Downloads are handled by the Windows BAT launchers with aria2c/curl. This module
only points the app at files that are already present under the local models
folder or explicit DRAMABOX_* path overrides.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR = APP_DIR / "models"

MODEL_FILES = {
    "transformer": "dramabox-dit-v1.safetensors",
    "audio_components": "dramabox-audio-components.safetensors",
    "silence_latent": "assets/silence_latent_frame.pt",
}

LTX_DISTILLED_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"
GEMMA_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


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


def _local_file(subdir: str, filename: str) -> str | None:
    path = _model_base_dir() / subdir / Path(filename)
    if path.is_file() and path.stat().st_size > 0:
        return str(path)
    return None


def _local_dir(subdir: str, required_file: str) -> str | None:
    path = _model_base_dir() / subdir
    if path.is_dir() and (path / required_file).is_file():
        return str(path)
    return None


def _missing_required_files(subdir: str, required_files: tuple[str, ...]) -> list[str]:
    path = _model_base_dir() / subdir
    if not path.is_dir():
        return list(required_files)
    missing: list[str] = []
    for name in required_files:
        file_path = path / name
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            missing.append(name)
    return missing


def get_model_path(name: str, cache_dir: str | None = None) -> str:
    """Resolve one locally installed DramaBox model file."""
    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_FILES)}")

    repo_path = os.environ.get(f"DRAMABOX_{name.upper()}_FILE", MODEL_FILES[name])
    logger.info("Resolving local %s model file %s...", name, repo_path)

    local_path = _local_file("dramabox", repo_path)
    if local_path:
        logger.info("  -> %s", local_path)
        return local_path

    expected = _model_base_dir() / "dramabox" / Path(repo_path)
    raise FileNotFoundError(
        f"{name} is not installed locally. Run install-GGF-DramaBox.bat to "
        f"download it with aria2c/curl -Lo: {expected}"
    )


def get_ltx_distilled_path(cache_dir: str | None = None) -> str:
    """Resolve the locally installed official LTX-2.3 distilled v1.1 checkpoint."""
    filename = os.environ.get("DRAMABOX_LTX_DISTILLED_FILE", LTX_DISTILLED_FILE)
    logger.info("Resolving local LTX distilled checkpoint %s...", filename)
    local_path = _local_file("ltx-distilled-1.1", filename)
    if local_path:
        logger.info("  -> %s", local_path)
        return local_path

    expected = _model_base_dir() / "ltx-distilled-1.1" / filename
    raise FileNotFoundError(
        "LTX distilled checkpoint is not installed locally. Run run.bat option 2 "
        f"or install-GGF-DramaBox.bat to download it with curl -Lo: {expected}"
    )


def get_gemma_path(cache_dir: str | None = None) -> str:
    """Resolve the locally installed Gemma text encoder directory."""
    override = _first_env("DRAMABOX_GEMMA_ROOT", "GEMMA_DIR")
    if override:
        return _require_dir(override, "Gemma root")

    logger.info("Resolving local Gemma model directory...")

    expected = _model_base_dir() / "gemma-3-12b-it-bnb-4bit"
    missing = _missing_required_files("gemma-3-12b-it-bnb-4bit", GEMMA_REQUIRED_FILES)
    if not missing:
        logger.info("  -> %s", expected)
        return str(expected)

    raise FileNotFoundError(
        "Gemma is missing required local files "
        f"({', '.join(missing)}). Run install-GGF-DramaBox.bat again to finish "
        f"downloading into: {expected}"
    )


def get_all_paths(cache_dir: str | None = None) -> dict:
    """Resolve all assets needed by TTSServer.

    Environment overrides:
      DRAMABOX_MODEL_PROFILE=dramabox|ltx-distilled|custom
      DRAMABOX_TRANSFORMER_PATH=/path/to/audio-only.safetensors
      DRAMABOX_AUDIO_COMPONENTS_PATH=/path/to/full-or-components.safetensors
      DRAMABOX_GEMMA_ROOT=/path/to/gemma
      DRAMABOX_MODEL_TYPE=dramabox|dev|distilled|auto
    """
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

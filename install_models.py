#!/usr/bin/env python3
"""Install GGF-DramaBox model assets without starting the app."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model_downloader import get_all_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install GGF-DramaBox model assets")
    parser.add_argument(
        "--profile",
        choices=["dramabox", "ltx-distilled", "all"],
        default="dramabox",
        help="Model profile to install.",
    )
    return parser.parse_args()


def install_profile(profile: str) -> None:
    print(f"\nInstalling model profile: {profile}")
    os.environ["DRAMABOX_MODEL_PROFILE"] = profile
    os.environ["DRAMABOX_MODEL_TYPE"] = "distilled" if profile == "ltx-distilled" else "dramabox"
    paths = get_all_paths()
    print("\nInstalled/resolved assets:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    profiles = ["dramabox", "ltx-distilled"] if args.profile == "all" else [args.profile]
    for item in profiles:
        install_profile(item)
    print("\nModel installation complete.")

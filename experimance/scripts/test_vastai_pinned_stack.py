#!/usr/bin/env python3
"""Smoke test the pinned VastAI model-server Python stack in an isolated virtualenv."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


STACK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "services"
    / "image_server"
    / "src"
    / "image_server"
    / "generators"
    / "vastai"
    / "server"
    / "pinned_stack.json"
)

IMPORT_CHECK = """
import accelerate
import diffusers
import fastapi
import huggingface_hub
import peft
import pydantic
import safetensors
import tokenizers
import transformers

from diffusers import StableDiffusionXLControlNetPipeline
from transformers import CLIPTextModel

print({
    'accelerate': accelerate.__version__,
    'diffusers': diffusers.__version__,
    'fastapi': fastapi.__version__,
    'huggingface_hub': huggingface_hub.__version__,
    'peft': peft.__version__,
    'pydantic': pydantic.__version__,
    'safetensors': safetensors.__version__,
    'tokenizers': tokenizers.__version__,
    'transformers': transformers.__version__,
    'pipeline': StableDiffusionXLControlNetPipeline.__name__,
    'clip_has_text_model_attr': hasattr(CLIPTextModel, 'text_model'),
})
""".strip()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create the test virtualenv",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the temporary virtualenv for debugging",
    )
    return parser


def load_stack_config() -> tuple[dict[str, str], list[str]]:
    with STACK_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config["pinned_packages"], config.get("post_install_no_deps", [])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pinned_packages, post_install_no_deps = load_stack_config()

    temp_dir = Path(tempfile.mkdtemp(prefix="vastai-pinned-stack-"))
    venv_dir = temp_dir / "venv"
    try:
        run([args.python, "-m", "venv", str(venv_dir)])

        python_bin = venv_dir / "bin" / "python"
        pip_bin = [str(python_bin), "-m", "pip"]

        run(pip_bin + ["install", "--upgrade", "pip"])
        run(
            pip_bin
            + [
                "install",
                *(f"{name}=={version}" for name, version in pinned_packages.items()),
            ]
        )
        for package in post_install_no_deps:
            run(pip_bin + ["install", "--no-deps", package])
        run([str(python_bin), "-c", IMPORT_CHECK])
        print("Pinned VastAI stack smoke test passed.")
        return 0
    finally:
        if args.keep_venv:
            print(f"Kept test virtualenv at {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Smoke test the VastAI model-server Python stack in an isolated virtualenv.

Verifies that the pinned package versions in pinned_stack.json install cleanly and
pass critical import checks (including the CLIPTextModel.text_model compatibility
check that caused the original breakage).

USAGE
-----

1. Verify the current pinned stack still works:

    uv run python scripts/test_vastai_pinned_stack.py

2. Check whether the latest published versions work yet:

    uv run python scripts/test_vastai_pinned_stack.py --latest

   If the test passes, the printed JSON block can be pasted directly into
   pinned_stack.json to advance the pins.

   If the test fails, the error tells you which package is incompatible.

3. Test latest-everything but hold one or more packages back:

    uv run python scripts/test_vastai_pinned_stack.py --latest --pin transformers==5.5.4
    uv run python scripts/test_vastai_pinned_stack.py --latest --pin transformers==5.5.4 --pin peft==0.15.0

   Useful for bisecting which package introduced a regression.

4. Keep the temp virtualenv for manual inspection after a failure:

    uv run python scripts/test_vastai_pinned_stack.py --latest --keep-venv

BACKGROUND
----------

The VastAI model server uses a pinned ML stack to prevent overnight breakage when
upstream packages release incompatible versions.  The single source of truth for
those pins is:

    services/image_server/src/image_server/generators/vastai/server/pinned_stack.json

Both this smoke test and the remote provisioning script (vast_provisioning.sh) read
from that file, so updating it is the only change needed to advance all pins.

Known incompatibility: some newer transformers releases changed CLIP internals,
and diffusers' StableDiffusionXLControlNetPipeline may break depending on the
diffusers + transformers pair.  Do not assume a whole major/minor line is safe;
validate specific pairs with this smoke test before updating pinned_stack.json.
"""

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

from transformers import CLIPTextConfig, CLIPTextModel

dummy_text_encoder = CLIPTextModel(CLIPTextConfig())

versions = {
    'accelerate': accelerate.__version__,
    'diffusers': diffusers.__version__,
    'fastapi': fastapi.__version__,
    'huggingface_hub': huggingface_hub.__version__,
    'peft': peft.__version__,
    'pydantic': pydantic.__version__,
    'safetensors': safetensors.__version__,
    'tokenizers': tokenizers.__version__,
    'transformers': transformers.__version__,
    # text_model is an instance attribute on CLIPTextModel, not a class attribute.
    'clip_has_text_model_attr': hasattr(dummy_text_encoder, 'text_model'),
}

try:
    from diffusers import StableDiffusionXLControlNetPipeline
    versions['pipeline'] = StableDiffusionXLControlNetPipeline.__name__
except Exception as exc:
    versions['pipeline_import_error'] = f"{type(exc).__name__}: {exc}"
    print(versions)
    raise AssertionError(
        "Failed to import StableDiffusionXLControlNetPipeline. "
        "This usually means diffusers and transformers are out of sync. "
        f"Observed diffusers={versions['diffusers']}, transformers={versions['transformers']}. "
        "If you pin transformers to 4.44.x, also pin diffusers to a compatible "
        "release (e.g. 0.30.0) instead of using latest diffusers."
    ) from exc

print(versions)
assert versions['clip_has_text_model_attr'], (
    f"CLIPTextModel missing 'text_model' attribute — "
    f"transformers {versions['transformers']} is incompatible. "
    "This transformers build is incompatible with SDXL ControlNet expectations "
    "for this stack. Try a different transformers pin and validate with this smoke test."
)
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
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Install latest versions (no pins) and print resolved versions as a pinned_stack.json update",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="PKG==VER",
        help="Override a specific package version when using --latest (e.g. --pin transformers==5.5.4). Can be repeated.",
    )
    return parser


def load_stack_config() -> tuple[dict[str, str], list[str]]:
    with STACK_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config["pinned_packages"], config.get("post_install_no_deps", [])


RESOLVED_VERSIONS_SCRIPT = """
import importlib.metadata, json, sys
pkgs = sys.argv[1:]
out = {}
for p in pkgs:
    try:
        out[p] = importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
        out[p] = None
print(json.dumps(out))
""".strip()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pinned_packages, post_install_no_deps = load_stack_config()

    mode = "latest" if args.latest else "pinned"
    prefix = f"vastai-{mode}-stack-"
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    venv_dir = temp_dir / "venv"
    try:
        run([args.python, "-m", "venv", str(venv_dir)])

        python_bin = venv_dir / "bin" / "python"
        pip_bin = [str(python_bin), "-m", "pip"]

        run(pip_bin + ["install", "--upgrade", "pip"])

        if args.latest:
            # Install by name only — let pip choose latest compatible versions
            install_specs = list(pinned_packages.keys())
            # Apply any --pin overrides (replace bare name with pinned spec)
            if args.pin:
                override_map = {}
                for spec in args.pin:
                    pkg = spec.split("==")[0].split(">=")[0].split("<=")[0].lower()
                    override_map[pkg] = spec
                install_specs = [
                    override_map.get(s.lower(), s) for s in install_specs
                ]
                # Add any overrides for packages not already in the list
                for spec in args.pin:
                    pkg = spec.split("==")[0].split(">=")[0].split("<=")[0].lower()
                    if pkg not in [s.lower() for s in pinned_packages.keys()]:
                        install_specs.append(spec)
            run(pip_bin + ["install"] + install_specs)
            for package in post_install_no_deps:
                run(pip_bin + ["install", "--no-deps", package])
        else:
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

        if args.latest:
            result = subprocess.run(
                [str(python_bin), "-c", RESOLVED_VERSIONS_SCRIPT]
                + list(pinned_packages.keys()),
                capture_output=True,
                text=True,
                check=True,
            )
            resolved = json.loads(result.stdout)
            print("\n" + "=" * 60)
            print("RESOLVED LATEST VERSIONS — paste into pinned_stack.json:")
            print("=" * 60)
            new_config = {
                "pinned_packages": {k: v for k, v in resolved.items() if v},
                "post_install_no_deps": post_install_no_deps,
            }
            print(json.dumps(new_config, indent=2))
            print("=" * 60)
            print("\nLatest VastAI stack smoke test passed.")
        else:
            print("Pinned VastAI stack smoke test passed.")
        return 0
    finally:
        if args.keep_venv:
            print(f"Kept test virtualenv at {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
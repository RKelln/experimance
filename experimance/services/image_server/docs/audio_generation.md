# Audio Generation

## Overview

The image server can generate environmental audio via the prompt-to-audio generator (TangoFlux). Audio requests are sent over ZMQ and returned as `AudioReady` messages when the project schema includes audio types.

Environment assumptions:

- Linux or macOS (audio dependencies are not supported on Windows)
- Python 3.11 with `uv`
- CUDA-capable GPU recommended for faster generation

When to use:

- You need ambient audio clips tied to prompts
- You want cached audio generation with metadata

When not to use:

- Your project schema does not include `AudioRenderRequest` and `AudioReady`

Files touched:

- `services/image_server/src/image_server/generators/audio/`
- `services/image_server/src/image_server/image_service.py`
- `services/image_server/src/image_server/cli.py`
- `projects/fire/schemas.py` (project-specific audio schemas)

## Setup

```bash
uv sync --package image-server --extra audio_gen
```

Pre-download audio models:

```bash
uv run python services/image_server/download_models.py
```

## Configuration

Configure audio generation in `projects/<project>/image_server.toml`.

```toml
[audio_generator]
strategy = "prompt2audio"
timeout = 120
enabled = true

[prompt2audio]
model_name = "declare-lab/TangoFlux"
duration_s = 8
sample_rate = 44100
steps = 30
guidance_scale = 4.5
```

## Usage

### CLI direct audio mode

```bash
uv run -m image_server.cli --audio -i
uv run -m image_server.cli --audio --audio-prompt "gentle rain"
```

### Direct test script

```bash
uv run python services/image_server/tests/test_audio_direct.py -i
```

## Testing

See `services/image_server/docs/testing.md` for test suites and ZMQ tools.

## Troubleshooting

- If `AudioRenderRequest` is missing, confirm your project schema (e.g., `projects/fire/schemas.py`).
- If model downloads fail, verify disk space and network access.
- If generation is slow, confirm CUDA is available or lower duration/steps.

## Integrations

- Generator system: `services/image_server/docs/generators.md`

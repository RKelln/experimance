# Image Server Overview

## Overview

The Image Server consumes render requests over ZMQ, generates images (and optional audio), and publishes results back to core services. It supports multiple generator backends, including mock, local SDXL, FAL.AI, and VastAI.

Environment assumptions:

- Linux or macOS host with Python 3.11
- `uv` installed
- ZMQ ports available (default ports in `experimance_common.constants.DEFAULT_PORTS`)

When to use:

- You need image generation for Experimance scenes
- You want the canonical ZMQ flow for `RenderRequest` and `ImageReady`

When not to use:

- You only need the model server API; use the VastAI model server instead

Files touched:

- `services/image_server/src/image_server/image_service.py`
- `services/image_server/src/image_server/__main__.py`
- `services/image_server/src/image_server/cli.py`

## Setup

```bash
uv sync --package image-server
```

## Configuration

Project configs live in `projects/<project>/image_server.toml`.

```toml
[generator]
strategy = "mock"
timeout = 12

[mock]
use_existing_images = true
existing_images_dir = "media/images/generated"
```

## Usage

### Start the service

```bash
uv run -m image_server
```

### CLI tool

```bash
uv run -m image_server.cli -i
uv run -m image_server.cli --prompt "A forest scene" --era wilderness --biome forest
```

### Audio generation via CLI

Audio CLI mode requires `AudioRenderRequest` and `AudioReady` schemas (for example, in `projects/fire/schemas.py`).

```bash
uv run -m image_server.cli --audio -i
uv run -m image_server.cli --audio --audio-prompt "gentle rain"
uv run -m image_server.cli --list-audio-prompts
```

## Testing

See `services/image_server/docs/testing.md`.

## Troubleshooting

- Ensure ports 5555, 5564, and 5565 are free.
- Check logs for generator initialization failures.
- For ZMQ debugging, run `services/image_server/tests/validate_zmq_addresses.py`.

## Integrations

- Message schemas: `libs/common/src/experimance_common/schemas_base.py` and project-specific schemas.
- Generator system: `services/image_server/docs/generators.md`.

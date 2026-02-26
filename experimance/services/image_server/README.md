# Image Server Service

## Overview

The Image Server consumes render requests over ZMQ, generates images (and optional audio), and publishes results back to the core service.

Environment assumptions:

- Linux or macOS host with Python 3.11
- `uv` installed
- ZMQ ports available (defaults in `experimance_common.constants.DEFAULT_PORTS`)

## Quick Start

```bash
# Set the active project (do this once)
uv run set-project experimance

# Start the image server
uv run -m image_server

# Use the mock generator (recommended for development)
# In projects/experimance/image_server.toml:
# [generator]
# strategy = "mock"
# timeout = 12

# Or switch to VastAI by setting:
# strategy = "vastai"
```

Generator strategies are configured in `projects/<project>/image_server.toml`. The mock generator can reuse existing images:

```toml
[mock]
use_existing_images = true
existing_images_dir = "media/images/generated"
```

## Configuration

- Project config: `projects/<project>/image_server.toml`
- Defaults for this service: `services/image_server/config.toml`

## Usage

- Start the service: `uv run -m image_server`
- CLI for test requests: `uv run -m image_server.cli -i`

## Message Types (Schemas)

- RenderRequest (work item)
  ```json
  {
    "type": "RenderRequest",
    "request_id": "uuid-string",
    "era": "wilderness",
    "biome": "forest",
    "prompt": "A forest scene with tall trees",
    "negative_prompt": "blurry, low quality",
    "seed": 12345,
    "width": 1024,
    "height": 1024,
    "depth_map": {
      "image_data": "base64-encoded-png",
      "uri": null
    }
  }
  ```

- ImageReady (result)
  ```json
  {
    "type": "ImageReady",
    "request_id": "uuid-string",
    "uri": "file:///path/to/image.png",
    "era": "wilderness",
    "biome": "forest",
    "prompt": "A forest scene with tall trees"
  }
  ```

  Note: Era and Biome fields are provided via project-specific schema extensions.

- AudioRenderRequest (work item)
  ```json
  {
    "type": "AudioRenderRequest",
    "request_id": "uuid-string",
    "generator": "tangoflux",
    "prompt": "gentle rain falling on forest leaves",
    "duration_s": 10,
    "style": "ambient",
    "seed": 12345,
    "metadata": {},
    "clear_queue": false
  }
  ```

- AudioReady (result)
  ```json
  {
    "type": "AudioReady",
    "request_id": "uuid-string",
    "uri": "file:///path/to/audio.wav",
    "prompt": "gentle rain falling on forest leaves",
    "duration_s": 10.0,
    "is_loop": true,
    "metadata": {
      "clap_similarity": 0.85,
      "cache_hit": false,
      "generator": "tangoflux"
    }
  }
  ```

## Additional Docs

- `services/image_server/docs/service_overview.md` - Service architecture, CLI usage, and integration points.
- `services/image_server/docs/dynamic_generators.md` - Per-request generator routing and image-to-image handling.
- `services/image_server/docs/generators.md` - Generator capabilities, configuration, and extension points.
- `services/image_server/docs/fal_lightning_i2i.md` - FAL Lightning I2I configuration and CLI usage.
- `services/image_server/docs/vastai.md` - VastAI generator setup, provisioning, and CLI management.
- `services/image_server/docs/vastai_model_server.md` - Model server API, parameters, and health endpoints.
- `services/image_server/docs/testing.md` - Test suites, ZMQ utilities, and audio direct testing.
- `services/image_server/docs/audio_generation.md` - Audio setup, schemas, and usage notes.
- `services/image_server/docs/zmq_messaging.md` - Port mappings and message flow.
- `services/image_server/docs/roadmap.md` - Near-term documentation goals and known gaps.

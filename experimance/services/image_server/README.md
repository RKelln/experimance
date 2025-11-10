# Image Server Service

The Image Server generates images in response to RenderRequest messages. It supports multiple generation strategies including Mock (uses existing images), VastAI/Comfy, local SDXL, and others. It started as purely an image generation service but now supports audio and can be extended to video as well.

## Overview

The service:

- Pulls `RenderRequest` work items from the worker queue (push/pull)
- Generates images based on prompts, era/biome and optional depth maps
- Pushes `ImageReady` results back to the core service
- Optionally subscribes to the unified events channel for coordination messages

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

# Or switch back to VastAI/Comfy by setting:
# strategy = "vastai"
```

Generator strategies are configured in `projects/<project>/image_server.toml`. The mock generator can reuse existing images:

```toml
[mock]
use_existing_images = true
existing_images_dir = "media/images/generated"
```

## Architecture

Image Server follows the same ControllerService + worker pattern as other services:

- Worker pull: receives `RenderRequest` on port 5564
- Worker push: sends `ImageReady` on port 5565
- Unified events pub/sub: 5555 (for coordination/status if needed)

Ports are defined in `experimance_common.constants.DEFAULT_PORTS`.

## ZMQ Communication

### Addresses and Ports

- Worker Pull (Core -> Image Server work): `tcp://localhost:5564`
- Worker Push (Image Server -> Core results): `tcp://*:5565`
- Events (Unified Pub/Sub): `tcp://*:5555` (publish), `tcp://localhost:5555` (subscribe)

### Message Types (Schemas)

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

## CLI Tool

Use the built-in CLI to send test requests and test generation:

```bash
# Interactive image mode
uv run -m image_server.cli -i

# Direct command
uv run -m image_server.cli --prompt "A forest scene" --era wilderness --biome forest

# Audio test mode (Fire project only)
uv run -m image_server.cli --audio -i
uv run -m image_server.cli --audio --audio-prompt "gentle rain"
uv run -m image_server.cli --list-audio-prompts
```

## Testing

The service includes ZeroMQ test utilities and pytest suites:

```bash
# Run all tests
uv run -m pytest services/image_server/tests

# Run test runner script
services/image_server/tests/run_zmq_tests.py

# Validate ZMQ addresses
services/image_server/tests/validate_zmq_addresses.py
```

Individual tests:

- `tests/test_zmq_messaging.py`
- `tests/test_zmq_render_request.py`
- `tests/test_image_server_service.py`
- `tests/test_local_sdxl.py`
- `tests/test_audio_direct.py`

## Troubleshooting

If you encounter communication issues:

1. Ensure the service is running and bound to the expected ports
2. Validate addresses with `services/image_server/tests/validate_zmq_addresses.py`
3. Confirm no port conflicts (5555 events, 5564/5565 worker)
4. Check logs for ZMQ connection or generator initialization errors
5. Run the test suite to isolate messaging problems

Common pitfalls:

- Mixing up bind/connect semantics (`tcp://*:` for bind, `tcp://localhost:` for connect)
- ZMQ slow joiner: add a short delay after socket creation
- Mismatched message schemas (ensure depth_map object, not raw PNG string)


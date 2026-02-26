# ZMQ Messaging

## Overview

The image server uses ZMQ worker sockets for request/response handling. Render requests arrive on the worker pull socket and results are returned on the worker push socket.

Environment assumptions:

- ZMQ ports open and free (defaults in `experimance_common.constants.DEFAULT_PORTS`)
- Core services configured with matching ports

When to use:

- You need to confirm the message flow or port configuration
- You are debugging message schema mismatches

When not to use:

- You only need CLI examples; see `services/image_server/docs/service_overview.md`

Files touched:

- `services/image_server/src/image_server/image_service.py`
- `services/image_server/src/image_server/cli.py`
- `libs/common/src/experimance_common/constants.py`

## Setup

No special setup is required beyond running the service and ensuring ports are free.

## Configuration

Ports are defined in `experimance_common.constants.DEFAULT_PORTS` and can be overridden in `projects/<project>/image_server.toml` via ZMQ config if needed.

## Usage

### Addresses and Ports

- Worker Pull (Core -> Image Server work): `tcp://localhost:5564`
- Worker Push (Image Server -> Core results): `tcp://*:5565`

### Message Types

Render requests and responses use project-specific schema extensions:

- `RenderRequest`
- `ImageReady`

Audio messages are project-specific (for example, `projects/fire/schemas.py` defines `AudioRenderRequest` and `AudioReady`).

Send a request using the CLI:

```bash
uv run -m image_server.cli -i
```

## Testing

```bash
uv run python services/image_server/tests/validate_zmq_addresses.py
uv run python services/image_server/tests/run_zmq_tests.py
```

## Troubleshooting

- If you see timeouts, verify the image server is running and ports are free.
- If messages are ignored, confirm schema `type` values and project-specific schemas are loaded.

## Integrations

- Service overview: `services/image_server/docs/service_overview.md`

# VastAI Generator and Model Server

## Overview

The VastAI generator provisions or connects to VastAI GPU instances to run the Experimance model server. It supports ControlNet depth conditioning, era-specific LoRAs, and managed instance lifecycle.

Environment assumptions:

- Linux host with network access to VastAI
- `uv` installed
- VastAI account with API key or CLI auth

When to use:

- You need remote GPU image generation without local GPU hardware
- You want scalable generation with automatic instance management

When not to use:

- You have reliable local GPU capacity (`local_sdxl` may be simpler)

Files touched:

- `services/image_server/src/image_server/generators/vastai/vastai_generator.py`
- `services/image_server/src/image_server/generators/vastai/vastai_manager.py`
- `services/image_server/src/image_server/generators/vastai/server/`

## Setup

### Install VastAI CLI tool

```bash
uv tool install vastai
uv tool run vastai set api-key <VASTAI_API_KEY>
```

### Provisioning script

Use the automatic provisioning workflow in the VastAI dashboard.

```bash
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/RKelln/experimance/refs/heads/main/experimance/services/image_server/src/image_server/generators/vastai/server/vast_provisioning.sh
```

This script is hosted in the repository and runs on the instance during provisioning.

## Configuration

Set the strategy and VastAI configuration in `projects/<project>/image_server.toml`:

```toml
[generator]
strategy = "vastai"

[vastai]
model_name = "lightning"
pre_warm = true
create_if_none = true
wait_for_ready = true
control_guidance_end = 0.8
```

## Usage

### Instance management (Python)

```python
from image_server.generators.vastai.vastai_manager import VastAIManager

manager = VastAIManager()
endpoint = manager.find_or_create_instance()
```

### Instance management (CLI)

The maintained CLI is exposed as a console entry point.

```bash
uv run vastai list
uv run vastai provision
uv run vastai health
```

## Testing

Run the test harness locally:

```bash
uv run python services/image_server/src/image_server/generators/vastai/test_vastai_manager.py
```

## Troubleshooting

### Exclusion list

The generator maintains an exclusion list for failed offers and instances.

```bash
uv run python services/image_server/src/image_server/generators/vastai/vastai_manager.py --exclusion-list-stats
uv run python services/image_server/src/image_server/generators/vastai/vastai_manager.py --clear-exclusion-list
uv run python services/image_server/src/image_server/generators/vastai/vastai_manager.py --exclude-offer OFFER_ID
```

### Provisioning issues

If VastAI ignores `PROVISIONING_SCRIPT`, the manager can fallback to SCP provisioning when creating instances.

## Integrations

- Model server API: `services/image_server/docs/vastai_model_server.md`
- Generator system: `services/image_server/docs/generators.md`

## Model server

The model server is implemented in `services/image_server/src/image_server/generators/vastai/server/` and exposes:

- `POST /generate`
- `GET /healthcheck`
- `GET /models`

See `services/image_server/docs/vastai_model_server.md` for server parameters and API details.

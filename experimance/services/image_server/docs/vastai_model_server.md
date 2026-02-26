# VastAI Model Server API

## Overview

The VastAI model server runs on GPU instances and provides a FastAPI API for ControlNet depth-conditioned SDXL generation. It supports multiple SDXL variants and era-specific LoRAs.

Environment assumptions:

- Python 3.11+ and CUDA-capable GPU on the VastAI instance
- Models downloaded on first use

When to use:

- You need the HTTP API for remote image generation
- You are debugging model server behavior or parameters

When not to use:

- You only need the image server's ZMQ flow; use `image_server` directly

Files touched:

- `services/image_server/src/image_server/generators/vastai/server/model_server.py`
- `services/image_server/src/image_server/generators/vastai/server/data_types.py`

## Setup

The server is installed and started by the provisioning script. For development on an instance:

```bash
python model_server.py
```

## Usage

### Health check

```bash
curl http://localhost:8000/healthcheck
```

### Generate image

```bash
curl -X POST http://localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "a serene lake at sunset", "mock_depth": true}'
```

### List models

```bash
curl http://localhost:8000/models
```

## Configuration

Environment variables:

| Variable            | Default                            | Description                 |
| ------------------- | ---------------------------------- | --------------------------- |
| `MODEL_SERVER_URL`  | `http://0.0.0.0:5001`              | Model server endpoint       |
| `MODEL_SERVER_HOST` | `0.0.0.0`                          | Model server bind address   |
| `MODEL_SERVER_PORT` | `5001`                             | Model server port           |
| `MODELS_DIR`        | `/workspace/models`                | Model storage directory     |
| `MODEL_LOG`         | `/workspace/logs/model_server.log` | Log file path               |
| `PRELOAD_MODEL`     | `lightning`                        | Model to preload on startup |
| `LOG_LEVEL`         | `info`                             | Logging level               |

## Parameters

Request fields for `POST /generate`:

| Parameter             | Type    | Default     | Description                                    |
| --------------------- | ------- | ----------- | ---------------------------------------------- |
| `prompt`              | string  | required    | Text description of desired image              |
| `negative_prompt`     | string  | optional    | What to avoid in the image                     |
| `depth_map_b64`       | string  | optional    | Base64-encoded depth map                       |
| `mock_depth`          | boolean | false       | Generate mock depth map for testing            |
| `model`               | string  | "lightning" | Model type: lightning, hyper, base             |
| `era`                 | string  | optional    | LoRA theme: wilderness, pre_industrial, future |
| `steps`               | integer | auto        | Inference steps (model-dependent)              |
| `cfg`                 | float   | auto        | Guidance scale (model-dependent)               |
| `seed`                | integer | random      | Random seed for reproducibility                |
| `lora_strength`       | float   | 1.0         | LoRA application strength                      |
| `controlnet_strength` | float   | 0.8         | ControlNet conditioning strength               |
| `scheduler`           | string  | "auto"      | Scheduler: auto, euler, dpm_sde                |
| `width`               | integer | 1024        | Output width (multiple of 64)                  |
| `height`              | integer | 1024        | Output height (multiple of 64)                 |

## Testing

Use the VastAI manager test harness:

```bash
uv run python services/image_server/src/image_server/generators/vastai/test_vastai_manager.py
```

## Troubleshooting

- Slow generation: use `lightning` model and lower step counts.
- Out of memory: reduce resolution or disable extra features.
- Connection issues: ensure the model server is running and port 8000 is exposed on the instance.

## Integrations

- VastAI generator: `services/image_server/docs/vastai.md`

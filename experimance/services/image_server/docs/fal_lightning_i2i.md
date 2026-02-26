# FAL Lightning Image-to-Image

## Overview

FAL Lightning I2I provides fast image-to-image transformations using the `fal-ai/fast-lightning-sdxl/image-to-image` endpoint.

Environment assumptions:

- Python 3.11
- Network access to FAL.AI
- `fal_client` installed (`uv sync --package image-server`)

When to use:

- You need rapid image-to-image transformations with minimal latency

When not to use:

- You require ControlNet depth conditioning or LoRA support; use VastAI or FAL ComfyUI instead

Files touched:

- `services/image_server/src/image_server/generators/fal/fal_lightning_i2i_generator.py`
- `services/image_server/src/image_server/generators/fal/fal_lightning_i2i_config.py`

## Setup

```bash
uv sync --package image-server
```

## Configuration

```toml
[generator]
strategy = "falai_lightning_i2i"

[falai_lightning_i2i]
endpoint = "fal-ai/fast-lightning-sdxl/image-to-image"
strength = 0.7
num_inference_steps = 4
```

## Usage

### Python

```python
from image_server.generators.fal.fal_lightning_i2i_generator import FalLightningI2IGenerator
from image_server.generators.fal.fal_lightning_i2i_config import FalLightningI2IConfig

config = FalLightningI2IConfig(
    strategy="falai_lightning_i2i",
    endpoint="fal-ai/fast-lightning-sdxl/image-to-image",
    strength=0.7,
    num_inference_steps=4,
    format="jpeg",
)

generator = FalLightningI2IGenerator(config=config)
result_path = await generator.generate_image(
    prompt="a beautiful oil painting in Van Gogh style",
    image_url="https://example.com/input-image.jpg",
)
```

### CLI example

```bash
uv run -m image_server.generators.fal.fal_lightning_i2i_generator \
    --image_url "https://example.com/image.jpg" \
    --strength 0.8 \
    --seed 42 \
    --output_dir "/tmp/generated"
```

## Parameters

Required:

- `image_url`: Source image URL or base64 data URI
- `prompt`: Transformation prompt

Optional:

- `strength` (0.0-1.0, default 0.4 in config)
- `num_inference_steps` (default 4)
- `seed`
- `format` (`jpeg` or `png`)
- `image_size` (custom width/height)

## Testing

Example script:

```bash
uv run python services/image_server/src/image_server/generators/fal/example_lightning_i2i.py
```

## Troubleshooting

- Missing `image_url`: the generator requires a source image and will raise an error.
- Timeouts: adjust the generator timeout in `projects/<project>/image_server.toml`.

## Integrations

- Dynamic selection: `services/image_server/docs/dynamic_generators.md`
- Generator system: `services/image_server/docs/generators.md`

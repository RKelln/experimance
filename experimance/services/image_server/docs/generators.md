# Generators Guide

## Overview

The image server uses a modular generator system for both image and audio generation. Generators are selected by strategy and expose capabilities like image-to-image, ControlNet depth conditioning, and negative prompts.

Environment assumptions:

- Python 3.11 with `uv` available
- Network access for remote generators (FAL.AI, VastAI)
- NVIDIA GPU with CUDA for `local_sdxl` or audio generation (optional)

When to use:

- You need to add or extend image/audio generation backends
- You want consistent capability checks and lifecycle handling across generators

When not to use:

- You only need mock images and do not plan to integrate new backends

Files touched:

- `services/image_server/src/image_server/generators/generator.py`
- `services/image_server/src/image_server/generators/config.py`
- `services/image_server/src/image_server/generators/factory.py`
- `services/image_server/src/image_server/generators/audio/`

## Setup

Install optional dependencies when needed.

```bash
uv sync --package image-server --extra local_gen
uv sync --package image-server --extra audio_gen
```

For audio model downloads:

```bash
uv run python services/image_server/download_models.py
```

## Configuration

Generator strategies are configured in `projects/<project>/image_server.toml` with a default strategy and per-strategy overrides.

```toml
[generator]
strategy = "mock"
timeout = 120

[mock]
use_existing_images = true
existing_images_dir = "media/images/generated"
```

Available strategies in the factory:

- `mock`
- `fal_comfy`
- `falai_lightning_i2i`
- `vastai`
- `local_sdxl`
- `subprocess` (advanced wrapper)

## Usage

### Generator capabilities

Generators declare supported capabilities with `supported_capabilities` and can be checked before sending optional parameters:

```python
if generator.supports_capability(GeneratorCapabilities.IMAGE_TO_IMAGE):
    render_request.reference_image = reference_image
```

Capabilities include:

- `IMAGE_TO_IMAGE`
- `CONTROLNET`
- `NEGATIVE_PROMPTS`
- `SEEDS`
- `CUSTOM_SCHEDULERS`
- `LORA`
- `UPSCALING`

Current capability matrix:

| Generator     | IMG2IMG | ControlNet | Negative | Seeds | Schedulers | LoRA | Upscaling |
| ------------- | ------- | ---------- | -------- | ----- | ---------- | ---- | --------- |
| Mock          | Yes     | No         | Yes      | Yes   | No         | No   | No        |
| Local SDXL    | Yes     | Yes        | Yes      | Yes   | Yes        | No   | No        |
| VastAI        | Yes     | Yes        | Yes      | Yes   | Yes        | Yes  | No        |
| FAL ComfyUI   | Yes     | Yes        | Yes      | Yes   | No         | Yes  | No        |
| FAL Lightning | Yes     | No         | Yes      | Yes   | No         | No   | No        |

### Audio generation

The audio generator (TangoFlux) provides text-to-audio generation, metadata, and semantic caching. Install `audio_gen` and pre-download models as shown above. See `services/image_server/docs/audio_generation.md` for configuration and schema notes.

## Testing

See `services/image_server/docs/testing.md` for generator and ZMQ test runners.

## Troubleshooting

- Missing generator dependencies: install the relevant `uv` extras.
- GPU errors: confirm CUDA-compatible drivers for `local_sdxl` or audio generation.
- Remote failures: verify network access and credentials for FAL.AI or VastAI.

## Integrations

- Dynamic per-request generator selection: `services/image_server/docs/dynamic_generators.md`
- VastAI provisioning and model server: `services/image_server/docs/vastai.md`

## Creating New Generators

### 1) Define configuration

```python
from image_server.generators.config import BaseGeneratorConfig

class YourServiceConfig(BaseGeneratorConfig):
    api_key: str
    model_name: str = "default-model"
```

### 2) Implement generator

```python
from image_server.generators.generator import ImageGenerator, GeneratorCapabilities

class YourServiceGenerator(ImageGenerator):
    supported_capabilities = {
        GeneratorCapabilities.IMAGE_TO_IMAGE,
        GeneratorCapabilities.NEGATIVE_PROMPTS,
        GeneratorCapabilities.SEEDS,
    }

    async def _generate_image_impl(self, prompt: str, **kwargs) -> str:
        self._validate_prompt(prompt)
        output_path = self._get_output_path("png", request_id=kwargs.get("request_id"))
        return output_path
```

### 3) Register in factory

```python
from .your_service.your_service_generator import YourServiceGenerator
from .your_service.your_service_config import YourServiceConfig

GENERATORS["your_service"] = {
    "config_class": YourServiceConfig,
    "generator_class": YourServiceGenerator,
}
```

### 4) Add tests

```python
import pytest
from .your_service_generator import YourServiceGenerator
from .your_service_config import YourServiceConfig

@pytest.mark.asyncio
async def test_your_service_generation():
    config = YourServiceConfig(strategy="your_service", api_key="test-key")
    generator = YourServiceGenerator(config, output_dir="/tmp")
    await generator.start()
    try:
        result = await generator.generate_image("test prompt")
        assert result.endswith(".png")
    finally:
        await generator.stop()
```

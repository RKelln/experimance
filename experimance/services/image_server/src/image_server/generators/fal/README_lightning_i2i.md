
# FAL Lightning Image-to-Image Generator

This generator provides high-speed image-to-image transformation using the FAL.AI Lightning SDXL endpoint.

## Features

- **Fast Generation**: Uses Lightning SDXL for rapid image-to-image transformations (typically 4 inference steps)
- **Base64 & URL Support**: Accepts both base64 data URIs and hosted image URLs as input
- **Configurable Strength**: Control how much the output resembles the input image (0.0-1.0)
- **Multiple Formats**: Support for JPEG and PNG output
- **Safety Filtering**: Optional NSFW content detection
- **Async Support**: Full async/await support with proper cancellation

## Quick Start

```python
from image_server.generators.fal.fal_lightning_i2i_generator import FalLightningI2IGenerator
from image_server.generators.fal.fal_lightning_i2i_config import FalLightningI2IConfig

# Create configuration
config = FalLightningI2IConfig(
    strategy="falai_lightning_i2i",
    endpoint="fal-ai/fast-lightning-sdxl/image-to-image", 
    strength=0.7,  # Moderate transformation
    num_inference_steps=4,
    format="jpeg"
)

# Create generator
generator = FalLightningI2IGenerator(config=config)

# Generate image
result_path = await generator.generate_image(
    prompt="a beautiful oil painting in Van Gogh style",
    image_url="https://example.com/input-image.jpg"
)
```

## Command Line Usage

```bash
# Run example with default settings
uv run -m image_server.generators.fal.fal_lightning_i2i_generator

# Run with custom parameters
uv run -m image_server.generators.fal.fal_lightning_i2i_generator \
    --image_url "https://example.com/image.jpg" \
    --strength 0.8 \
    --seed 42 \
    --output_dir "/tmp/generated"
```

## Configuration Parameters

### Required
- `image_url`: Source image URL or base64 data URI
- `prompt`: Text description for the transformation

### Optional
- `strength` (0.0-1.0): How much output resembles input (default: 0.95)
- `num_inference_steps`: Number of denoising steps (default: 4)
- `seed`: Random seed for reproducible results
- `format`: Output format ("jpeg" or "png", default: "jpeg")
- `image_size`: Output dimensions ("square_hd", custom object, etc.)
- `enable_safety_checker`: Enable NSFW detection (default: true)

## Era-Based Adjustments

The generator automatically adjusts parameters based on the `era` parameter:

- **Wilderness/Pre-Industrial**: Reduces strength by 10-20% for subtle transformations
- **Future**: Increases strength by 10% for more dramatic changes

## API Reference

### FalLightningI2IConfig

Configuration class inheriting from `SDXLConfig` with Lightning I2I specific parameters.

### FalLightningI2IGenerator 

Main generator class with the following key methods:

- `generate_image(prompt, image_url, **kwargs)`: Generate transformed image
- `stop()`: Cancel ongoing generation
- `falai_image_url_generator(response)`: Extract URLs from API response

## Error Handling

The generator provides detailed error messages for common issues:

- Missing `image_url` parameter
- Invalid prompts
- Timeout errors
- API failures
- Network issues

## Dependencies

- `fal_client`: FAL.AI Python client library
- `experimance_common`: Common utilities and schemas
- Standard async/await support

## Examples

See `example_lightning_i2i.py` for a complete working example.
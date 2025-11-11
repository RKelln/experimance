# Dynamic Generator Selection and Image-to-Image Support

This document describes the new dynamic generator functionality added to the Experimance Image Server, which allows for flexible generator selection at request time and improved support for image-to-image generation.

## Overview

The image server now supports:

1. **Dynamic Generator Selection**: Choose different generators per request using the `RenderRequest.generator` field
2. **Generator Manager**: Efficient caching and management of multiple generator instances
3. **Enhanced Image-to-Image**: Better support for reference images in generation requests
4. **Concurrent Processing**: Multiple generators can run simultaneously

## Key Components

### GeneratorManager

The `GeneratorManager` class handles dynamic creation and caching of generator instances:

```python
from image_server.generators.factory import GeneratorManager

# Initialize with default configuration
manager = GeneratorManager(
    default_strategy="falai",
    cache_dir="/tmp/images",
    timeout=60,
    default_configs={
        "mock": {"strategy": "mock"},
        "falai": {"strategy": "falai", "endpoint": "fal-ai/fast-lightning-sdxl"},
        "falai_lightning_i2i": {"strategy": "falai_lightning_i2i", "strength": 0.7}
    }
)

# Get generator (creates if needed, uses cache if available)
generator = manager.get_generator("falai")

# Check capabilities
is_i2i = manager.is_image_to_image_generator("falai_lightning_i2i")  # True

# List available strategies
strategies = manager.get_available_strategies()
```

### Dynamic Generator Selection

Specify the generator to use in the `RenderRequest`:

```python
from experimance_common.schemas_base import RenderRequest

# Use specific generator
request = RenderRequest(
    request_id="unique_id",
    generator="falai_lightning_i2i",  # Dynamic selection!
    prompt="A beautiful landscape",
    reference_image=reference_image_source
)

# Use default generator (omit generator field)
request = RenderRequest(
    request_id="unique_id",
    prompt="A futuristic city"
    # Will use default generator configured in service
)
```

### Image-to-Image Generation

The service now properly handles reference images for image-to-image generation:

```python
from experimance_common.schemas_base import ImageSource

# Using file URI
reference = ImageSource(uri="file:///path/to/image.png")

# Using base64 data
reference = ImageSource(image_data="data:image/png;base64,iVBORw0KGgo...")

# Create I2I request
request = RenderRequest(
    request_id="i2i_example",
    generator="falai_lightning_i2i",
    prompt="Transform into a magical landscape",
    reference_image=reference
)
```

## Configuration Changes

### Image Server Service

The image server now uses `GeneratorManager` internally while maintaining backward compatibility:

```python
# In ImageServerService.__init__()
self.generator_manager = GeneratorManager(
    default_strategy=config.generator.strategy,
    cache_dir=config.cache_dir,
    timeout=config.generator.timeout,
    default_configs=strategy_configs
)

# Legacy single generator still available
self.generator = create_generator_from_config(...)
```

### Generator Strategy Configuration

Configure multiple strategies in your service config:

```toml
[generator]
strategy = "falai"  # Default strategy
timeout = 120

# Strategy-specific configurations
[falai]
endpoint = "fal-ai/fast-lightning-sdxl"
num_inference_steps = 4

[falai_lightning_i2i]
endpoint = "fal-ai/fast-lightning-sdxl/image-to-image"
strength = 0.7
num_inference_steps = 4

[mock]
use_existing_images = true
existing_images_dir = "/path/to/images"
```

## Request Processing Flow

1. **Request Received**: Service receives `RenderRequest` via ZMQ
2. **Generator Selection**: Extract `generator` field or use default
3. **Generator Acquisition**: `GeneratorManager.get_generator()` returns cached or new instance
4. **Image Processing**: Handle reference images for I2I generation
5. **Generation**: Call appropriate generator with processed parameters
6. **Response**: Publish `ImageReady` message with result

## Implementation Details

### Reference Image Handling

The service automatically converts reference images for generators:

```python
# In _process_render_request()
reference_image = None
if hasattr(request, 'reference_image') and request.reference_image:
    if request.reference_image.uri:
        reference_image = request.reference_image.uri
    elif request.reference_image.image_data:
        # Convert base64 to proper data URI
        if not request.reference_image.image_data.startswith('data:'):
            reference_image = f"data:image/png;base64,{request.reference_image.image_data}"
        else:
            reference_image = request.reference_image.image_data

# Pass to generator
await generator.generate_image(
    prompt=prompt,
    image_url=reference_image,  # For I2I generators
    **other_params
)
```

### Generator Caching

Generators are cached based on strategy and configuration:

```python
# Cache key includes config overrides
cache_key = strategy
if config_overrides:
    override_key = "_".join(f"{k}={v}" for k, v in sorted(config_overrides.items()))
    cache_key = f"{strategy}_{override_key}"

# Reuse cached generator or create new one
if cache_key in self._generators:
    return self._generators[cache_key]
```

## Supported Generators

### Text-to-Image Generators

- **mock**: Testing and development
- **falai**: FAL.AI SDXL for production text-to-image
- **openai**: OpenAI DALL-E (when enabled)
- **local**: Local SDXL pipeline (when configured)

### Image-to-Image Generators

- **falai_lightning_i2i**: FAL.AI Lightning SDXL I2I for fast transformations

## Usage Examples

### Basic Dynamic Selection

```python
# Multiple requests with different generators
requests = [
    RenderRequest(
        request_id="landscape_1",
        generator="falai",
        prompt="Mountain landscape at sunset"
    ),
    RenderRequest(
        request_id="portrait_1", 
        generator="mock",
        prompt="Portrait of a wise elder"
    )
]
```

### Image-to-Image Workflow

```python
# 1. Generate base image
base_request = RenderRequest(
    request_id="base_image",
    generator="falai",
    prompt="A simple house in a field"
)

# 2. Transform with I2I
transform_request = RenderRequest(
    request_id="transformed",
    generator="falai_lightning_i2i",
    prompt="The same house but in winter with snow",
    reference_image=ImageSource(uri=base_image_uri)
)
```

### Configuration Overrides

```python
# Generator manager allows runtime config overrides
generator = manager.get_generator(
    "falai", 
    config_overrides={"num_inference_steps": 8, "strength": 0.9}
)
```

## Error Handling

The system gracefully handles various error conditions:

- **Unknown Generator**: Returns error with available strategies
- **Missing Reference Image**: Warns and falls back to text-to-image
- **Generation Failure**: Properly propagates errors with context
- **Timeout**: Respects per-generator timeout settings

## Migration Guide

### From Single Generator

**Old:**
```python
# Fixed generator at startup
self.generator = create_generator_from_config(strategy, config)

# Always use same generator
image_path = await self.generator.generate_image(prompt)
```

**New:**
```python
# Dynamic generator manager
self.generator_manager = GeneratorManager(default_strategy, configs)

# Select generator per request
generator = self.generator_manager.get_generator(request.generator)
image_path = await generator.generate_image(prompt)
```

### Configuration Updates

Update your service configuration to include multiple generator strategies and their specific settings.

## Performance Considerations

- **Generator Caching**: Reduces initialization overhead
- **Concurrent Processing**: Multiple generators can run simultaneously
- **Memory Management**: Generators are stopped and cleaned up properly
- **Image Transport**: Efficient handling of reference images using existing utilities

## Future Enhancements

- **Auto-scaling**: Dynamic creation/destruction based on load
- **Load Balancing**: Distribute requests across multiple instances
- **Strategy Selection**: Automatic generator selection based on prompt analysis
- **Fallback Chains**: Automatic fallback to alternative generators on failure

# Generators Guide

This document provides a comprehensive guide to the generation systems in the Experimance project, covering both image and audio generators and how to create new ones.

## Overview

The generation system uses a modular architecture supporting both **image** and **audio** generation with the following components:

- **Base Generator Classes**: Provide common functionality including thread-safe queuing, request tracking, and lifecycle management
- **Generator Configurations**: Pydantic models for type-safe configuration
- **Generator Factory**: Creates and manages generator instances
- **Specific Generators**: Individual implementations for different backends and modalities

### Image Generation System

The image generation system includes:

- **Base Generator Class** (`generator.py`): Provides common functionality including thread-safe queuing, request tracking, and lifecycle management
- **Generator Configurations** (`config.py`): Pydantic models for type-safe configuration
- **Generator Factory** (`factory.py`): Creates and manages generator instances
- **Specific Generators**: Individual implementations for different backends

## Architecture

### Generator Capabilities System

The generator system includes a capabilities framework that allows services to dynamically discover what features each generator supports. This enables intelligent routing and fallback strategies.

#### Available Capabilities

Generators can declare support for the following capabilities (defined in `GeneratorCapabilities`):

- **IMAGE_TO_IMAGE**: Supports img2img generation with reference images and strength parameters
- **CONTROLNET**: Supports ControlNet conditioning (depth maps, poses, etc.)
- **NEGATIVE_PROMPTS**: Supports negative prompting for excluding unwanted elements
- **SEEDS**: Supports reproducible generation with seed values
- **CUSTOM_SCHEDULERS**: Supports different sampling schedulers beyond defaults
- **LORA**: Supports LoRA (Low-Rank Adaptation) model fine-tuning
- **UPSCALING**: Supports image upscaling/super-resolution

#### Declaring Capabilities

Generators declare their capabilities using a class-level `supported_capabilities` set:

```python
class YourGenerator(ImageGenerator):
    # Declare what this generator supports
    supported_capabilities = {
        GeneratorCapabilities.IMAGE_TO_IMAGE,
        GeneratorCapabilities.NEGATIVE_PROMPTS,
        GeneratorCapabilities.SEEDS,
    }
```

#### Checking Capabilities

Services can check generator capabilities before making requests:

```python
# Check if generator supports a specific capability
if generator.supports_capability(GeneratorCapabilities.IMAGE_TO_IMAGE):
    # Safe to send img2img requests
    render_request.reference_image = reference_image
    render_request.strength = 0.6

# Get all supported capabilities
capabilities = generator.get_supported_capabilities()
```

#### Current Generator Capabilities

| Generator     | IMG2IMG | ControlNet | Negative | Seeds | Schedulers | LoRA | Upscaling |
| ------------- | ------- | ---------- | -------- | ----- | ---------- | ---- | --------- |
| Mock          | ✓       | ✗          | ✓        | ✓     | ✗          | ✗    | ✗         |
| Local SDXL    | ✓       | ✓          | ✓        | ✓     | ✓          | ✗    | ✗         |
| VastAI        | ✓       | ✓          | ✓        | ✓     | ✓          | ✓    | ✗         |
| FAL ComfyUI   | ✓       | ✓          | ✓        | ✓     | ✗          | ✓    | ✗         |
| FAL Lightning | ✓       | ✗          | ✓        | ✓     | ✗          | ✗    | ✗         |

### Thread-Safe Queuing System

All generators inherit from the base `ImageGenerator` class, which provides:

- **Asynchronous request queuing**: Ensures thread-safe processing of concurrent requests
- **Configurable concurrency**: Control how many requests process simultaneously (default: 1)
- **Request tracking**: Monitor pending requests and queue state
- **Queue management**: Clear queued requests during failures/recovery
- **Graceful lifecycle**: Proper startup/shutdown with cleanup

### Generation Flow

1. **Request Reception**: `generate_image()` queues the request with a unique ID
2. **Queue Processing**: Background task processes requests through the queue
3. **Implementation**: Calls `_generate_image_impl()` - the method subclasses implement
4. **Response**: Returns the generated image path to the caller

## Existing Generators

### Mock Generator (`mock/`)
**Purpose**: Testing and development  
**Features**: 
- Generates simple placeholder images with prompt text overlay
- Option to use existing images from a directory for realistic testing
- Configurable image size, colors, and text rendering
- No external dependencies

**Use Cases**: Unit testing, development when external services are unavailable

### Local SDXL Generator (`local/`)
**Purpose**: Local GPU-based image generation  
**Features**:
- Uses Diffusers library with SDXL Lightning model
- ControlNet depth conditioning support
- Configurable steps, guidance scale, scheduler
- Model compilation and optimization options
- Warmup functionality for faster subsequent generations

**Use Cases**: Production local generation, full control over model and parameters

### Audio Generation System

The audio generation system provides text-to-audio capabilities with the following components:

#### TangoFlux Generator (`audio/`)
**Purpose**: High-quality text-to-audio synthesis using TangoFlux model  
**Features**:
- Text-to-audio generation using TangoFlux (declare-lab/TangoFlux)
- CLAP-based audio similarity scoring for quality assessment
- BGE embeddings for enhanced text understanding
- Configurable model storage with centralized MODELS_DIR support
- Audio format normalization and processing
- PyTorch 2.4.0 with CUDA 12.1.x support

**Configuration Options**:
- `model_name`: TangoFlux model variant (default: "declare-lab/TangoFlux")
- `clap_model`: CLAP model for audio similarity (default: "laion/clap-htsat-unfused")
- `bge_model`: BGE embedding model (default: "BAAI/bge-large-en-v1.5")
- `models_dir`: Directory for model storage (default: uses MODELS_DIR environment variable)
- `sample_rate`: Output audio sample rate (default: 16000)
- `duration`: Audio duration in seconds (default: 10.0)

**Use Cases**: 
- Environmental soundscapes for installations
- Dynamic audio generation based on text descriptions
- AI-generated background audio for interactive experiences

#### Audio Dependencies

The audio generation system requires the following dependencies (Linux/macOS only):
- `soundfile>=0.12.1`: Audio file I/O
- `librosa>=0.10.0`: Audio analysis and processing
- `ffmpeg-normalize>=1.28.0`: Audio normalization
- `torch==2.4.0`: PyTorch with CUDA 12.1.x (required by TangoFlux)
- `torchaudio>=2.1.0`: Audio tensor operations
- `transformers>=4.39.0`: HuggingFace model loading
- `sentence-transformers>=2.2.2`: Text embeddings

**Installation**:
```bash
# Install audio generation dependencies
cd services/image_server
uv sync --extra audio_gen

# Download models to centralized location
uv run python src/image_server/generators/audio/download_models.py
```

**CUDA Compatibility Note**: 
TangoFlux requires exactly PyTorch 2.4.0, which uses CUDA 12.1.x runtime libraries. This is a requirement of the TangoFlux model and cannot be changed without breaking compatibility.

### Audio Generator Architecture

Audio generators extend the base generator pattern with audio-specific functionality:

```python
class AudioGenerator:
    """Base class for audio generators."""
    
    def __init__(self, config: BaseAudioGeneratorConfig, **kwargs):
        self.config = config
        self.models_dir = config.models_dir
    
    async def generate_audio(self, prompt: str, **kwargs) -> str:
        """Generate audio from text prompt. Returns path to audio file."""
        pass
    
    def setup_model_cache(self, models_dir: Path):
        """Configure HuggingFace cache for centralized model storage."""
        pass
```

**Key Features**:
- Centralized model storage in `MODELS_DIR`
- Lazy model loading with caching
- Audio format standardization
- Quality assessment with CLAP similarity scoring
- Thread-safe generation with proper cleanup

### VastAI Generator (`vastai/`)
**Purpose**: Remote GPU instances via VastAI  
**Features**:
- Automatic instance management and health monitoring
- Recovery and fallback mechanisms for failed instances
- ControlNet and LoRA support
- Background instance provisioning
- Comprehensive error handling and retry logic
- Queue integration for instance recovery scenarios

**Use Cases**: Cost-effective scalable generation, handling variable loads

### FAL.AI Generators (`fal/`)

#### ComfyUI Generator (`fal_comfy_generator.py`)
**Purpose**: ComfyUI workflows via FAL.AI  
**Features**:
- ControlNet depth conditioning
- LoRA integration with era-based mappings
- Workflow-based generation approach
- Support for complex multi-node pipelines

#### Lightning I2I Generator (`fal_lightning_i2i_generator.py`)
**Purpose**: Fast image-to-image transformations  
**Features**:
- Lightning SDXL for rapid I2I (typically 4 inference steps)
- Accepts base64 data URIs and hosted image URLs
- Configurable transformation strength
- NSFW content filtering
- High-speed transformations

**Use Cases**: Quick image modifications, style transfers

### OpenAI Generator (`openai/`)
**Purpose**: DALL-E API integration  
**Status**: Currently commented out in factory, needs API integration work

## Creating New Generators

### Step 1: Define Configuration

Create a configuration class extending `BaseGeneratorConfig`:

```python
# generators/your_service/your_service_config.py
from image_server.generators.config import BaseGeneratorConfig

class YourServiceConfig(BaseGeneratorConfig):
    api_key: str
    model_name: str = "default-model"
    custom_param: int = 10
    # Add other service-specific parameters
```

### Step 2: Implement Generator Class

Create the generator class extending `ImageGenerator`:

```python
# generators/your_service/your_service_generator.py
import logging
from typing import Optional
from image_server.generators.generator import ImageGenerator, GeneratorCapabilities
from .your_service_config import YourServiceConfig

logger = logging.getLogger(__name__)

class YourServiceGenerator(ImageGenerator):
    """Your custom image generator implementation."""
    
    # Declare what capabilities this generator supports
    supported_capabilities = {
        GeneratorCapabilities.IMAGE_TO_IMAGE,
        GeneratorCapabilities.NEGATIVE_PROMPTS,
        GeneratorCapabilities.SEEDS,
        # Add other capabilities your generator supports
    }
    
    def _configure(self, config: YourServiceConfig, **kwargs):
        """Configure generator-specific settings."""
        self.config = config
        # Initialize any service-specific setup
        logger.info(f"Configured {self.__class__.__name__} with model: {config.model_name}")
    
    async def start(self):
        """Start the generator - call parent first for queue system. Put warm-up code here."""
        await super().start()
        # Add any service-specific startup logic
        logger.info(f"{self.__class__.__name__} started")
    
    async def _generate_image_impl(self, prompt: str, **kwargs) -> str:
        """
        Implement the actual image generation logic.
        
        This is the core method you need to implement.
        Should return the path to the generated image file.
        
        Check capabilities before using optional parameters:
        - reference_image/image_b64 + strength: IMAGE_TO_IMAGE capability
        - depth_map/depth_map_b64: CONTROLNET capability  
        - negative_prompt: NEGATIVE_PROMPTS capability
        - seed: SEEDS capability
        """
        self._validate_prompt(prompt)
        logger.info(f"Generating image: {prompt[:50]}...")
        
        # Example: Check for img2img parameters
        reference_image = kwargs.get('image') or kwargs.get('image_b64')
        if reference_image and self.supports_capability(GeneratorCapabilities.IMAGE_TO_IMAGE):
            strength = kwargs.get('strength', 0.8)
            logger.info(f"Using img2img with strength {strength}")
            # Handle img2img generation
        
        try:
            # Your generation logic here
            # 1. Call your service API
            # 2. Process the response
            # 3. Save the image using self._get_output_path()
            # 4. Return the file path
            
            output_path = self._get_output_path("png", request_id=kwargs.get('request_id'))
            
            # Example: Save your generated image
            # generated_image.save(output_path, "PNG")
            
            logger.info(f"Image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Your service generation failed: {e}")
    
    async def stop(self):
        """Stop the generator - call parent for queue cleanup."""
        logger.info(f"Stopping {self.__class__.__name__}")
        # Add any service-specific cleanup
        await super().stop()
```

### Step 3: Register in Factory

Add your generator to the appropriate factory registry:

**For Image Generators:**
```python
# generators/factory.py
from .your_service.your_service_generator import YourServiceGenerator
from .your_service.your_service_config import YourServiceConfig

GENERATORS = {
    # ... existing generators ...
    "your_service": {
        "config_class": YourServiceConfig,
        "generator_class": YourServiceGenerator
    },
}
```

**For Audio Generators:**
```python
# generators/audio/factory.py (if separate audio factory exists)
# or include in main factory with audio_ prefix
from .your_audio_service.your_audio_generator import YourAudioGenerator
from .your_audio_service.your_audio_config import YourAudioConfig

AUDIO_GENERATORS = {
    # ... existing audio generators ...
    "your_audio_service": {
        "config_class": YourAudioConfig,
        "generator_class": YourAudioGenerator
    },
}
```

### Step 4: Add Configuration Support

Update the configuration union type:

```python
# generators/config.py
GeneratorConfigType = Union[
    MockGeneratorConfig,
    # ... other configs ...
    YourServiceConfig,  # Add your config here
]
```

### Step 5: Create Tests

Create tests for your generator:

```python
# generators/your_service/test_your_service.py
import pytest
import asyncio
from .your_service_generator import YourServiceGenerator
from .your_service_config import YourServiceConfig

@pytest.mark.asyncio
async def test_your_service_generation():
    config = YourServiceConfig(
        strategy="your_service",
        api_key="test-key",
        model_name="test-model"
    )
    
    generator = YourServiceGenerator(config, output_dir="/tmp")
    
    try:
        await generator.start()
        result = await generator.generate_image("test prompt")
        assert result.endswith(".png")
    finally:
        await generator.stop()
```

**For Audio Generators:**
```python
# generators/audio/test_audio_generator.py
import pytest
from pathlib import Path
from .prompt2audio import TangoFluxGenerator
from .audio_config import TangoFluxConfig

@pytest.mark.asyncio
async def test_audio_generation():
    config = TangoFluxConfig(
        model_name="declare-lab/TangoFlux",
        duration=5.0,  # Shorter for testing
        sample_rate=16000
    )
    
    generator = TangoFluxGenerator(config)
    audio_path = await generator.generate_audio("gentle rain")
    
    assert Path(audio_path).exists()
    assert audio_path.endswith(".wav")
```

## Best Practices

### Capability Declaration
- Always declare your generator's capabilities using the `supported_capabilities` class attribute
- Only declare capabilities your generator actually supports and implements properly
- Check capabilities before using optional parameters in `_generate_image_impl()`:
  ```python
  # Good: Check capability before using
  if self.supports_capability(GeneratorCapabilities.IMAGE_TO_IMAGE):
      strength = kwargs.get('strength', 0.8)
      # Use img2img parameters
  
  # Bad: Assume parameters are always available
  strength = kwargs.get('strength', 0.8)  # May not be supported
  ```
- Document any capability limitations in your generator's docstring

### Error Handling
- Always catch and re-raise with context: `raise RuntimeError(f"Service failed: {e}")` 
- Log errors with appropriate levels (ERROR for failures, WARNING for retries)
- Implement retry logic for transient failures using `tenacity`

### Concurrency and Threading
- The base class handles thread-safety - you only implement `_generate_image_impl()`
- Use `asyncio.to_thread()` for blocking operations to avoid blocking the event loop
- Set appropriate `max_concurrent` in constructor (1 for thread-unsafe operations, higher for parallelizable work)

### Queue Management for Complex Generators
- For generators with recovery/fallback logic (like VastAI), use `clear_pending_requests()` during failures
- Call `restart_queue_processor()` after recovery for clean state
- Use `get_queue_stats()` for monitoring and debugging

### Configuration
- Use Pydantic models for type-safe configuration
- Provide sensible defaults for optional parameters
- Document all configuration options with descriptions

### Resource Management
- Always call `await super().start()` and `await super().stop()` for proper lifecycle
- Clean up external resources in the `stop()` method
- Use context managers for temporary resources

### Output Handling
- Use `self._get_output_path()` for consistent file naming and organization
- Support subdirectories for organized storage: `self._get_output_path("png", sub_dir="era/biome")`
- Include `request_id` in filenames for traceability when provided

### Logging
- Use structured logging with consistent formats
- Log key events: start/stop, generation requests, errors, performance metrics
- Use appropriate log levels (DEBUG for detailed flow, INFO for key events, WARNING for recoverable issues)

## Integration Examples

### Adding to Image Server Service
The service automatically discovers and uses registered generators:

```bash
# Use your image generator
uv run -m image_server --generator-strategy your_generator

# Use audio generation features
uv run -m image_server --enable-audio-generation
```

### Using Audio Generation
```python
# Example usage in code
from image_server.generators.audio.prompt2audio import TangoFluxGenerator
from image_server.generators.audio.audio_config import TangoFluxConfig

config = TangoFluxConfig(
    model_name="declare-lab/TangoFlux",
    duration=10.0,
    sample_rate=16000
)

generator = TangoFluxGenerator(config)
audio_path = await generator.generate_audio("ocean waves crashing on a rocky shore")
```

### Configuration in TOML files
```toml
[generator]
strategy = "your_generator"

[your_generator]
model_name = "premium-model"
custom_param = 25

# Audio generation configuration
[audio_generator]
model_name = "declare-lab/TangoFlux"
clap_model = "laion/clap-htsat-unfused"
bge_model = "BAAI/bge-large-en-v1.5"
models_dir = "/path/to/models"  # or use MODELS_DIR env var
sample_rate = 16000
duration = 10.0
```

Place API keys needed in `projects/<project_name>/.env`.


## Advanced Features

### Health Monitoring
Implement health checking for robust generators:

```python
async def test_connection(self) -> Dict[str, Any]:
    """Test service connectivity and return status."""
    try:
        # Test your service
        return {"status": "healthy", "service_version": "1.0"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Recovery and Fallback
For complex generators with external dependencies:

```python
async def _handle_service_failure(self):
    """Handle service failures with recovery logic."""
    # Clear pending requests to fail fast
    await self.clear_pending_requests("Service recovering")
    
    # Implement recovery logic
    await self._recover_service()
    
    # Restart queue processor for clean state
    await self.restart_queue_processor()
```

### Performance Monitoring
Track and log performance metrics:

```python
# In _generate_image_impl()
start_time = time.time()
# ... generation logic ...
total_time = time.time() - start_time

logger.info(f"Generated in {total_time:.2f}s")
if self._total_requests % 10 == 0:
    success_rate = self._successful_requests / self._total_requests * 100
    logger.info(f"Success rate: {success_rate:.1f}%")
```

## Troubleshooting

### Common Issues
1. **Queue hanging**: Ensure `_generate_image_impl()` handles all exceptions
2. **Resource leaks**: Always call `super().stop()` and clean up in `stop()` method
3. **Thread safety**: Never modify shared state in `_generate_image_impl()` without proper synchronization
4. **Configuration errors**: Use Pydantic validation and provide clear error messages

### Debugging
- Enable debug logging: `--log-level debug`
- Use queue stats: `generator.get_queue_stats()`
- Check service logs for external API issues
- Test generators independently with direct calls

## Future Considerations

- **Multi-GPU Support**: Increase `max_concurrent` parameter for parallel processing
- **Load Balancing**: Implement multiple instance management for high-throughput scenarios  
- **Caching**: Add result caching for repeated requests
- **Streaming**: Support for streaming generation progress
- **Plugin System**: Dynamic generator loading for extensibility

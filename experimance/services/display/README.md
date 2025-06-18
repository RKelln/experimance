# Experimance Display Service

The Display Service is the visual rendering component of the Experimance installation. It handles all visual output including satellite landscape images, masked video overlays, text communication from the AI agent, and custom transitions between scenes.

## Features

- **Satellite Landscape Images**: Displays background images with smooth crossfade transitions
- **Masked Video Overlays**: Dynamic video overlays that respond to sand interaction using grayscale masks
- **Text Overlays**: Multiple concurrent text displays with speaker-specific styling (agent, system, debug)
- **Custom Transitions**: Support for custom transition videos between scenes
- **Real-time Performance**: Maintains 30fps at various resolutions (tested at 1920x1080 and 3840x2560)
- **ZMQ Integration**: Subscribes to image and text channels for real-time updates
- **Testable Interface**: Direct API for testing without ZMQ infrastructure

## Installation

The display service is part of the main Experimance project. From the project root:

```bash
# Install all dependencies
uv sync

# Or install just the display service
cd services/display
uv sync
```

## Usage

### Running the Service

```bash
# From project root - windowed mode for development
uv run -m experimance_display --windowed

# With debug logging
uv run -m experimance_display --windowed --log-level DEBUG

# From display service directory
cd services/display
uv run python -m experimance_display --windowed

# Production mode (fullscreen)
uv run -m experimance_display

# See all options
uv run -m experimance_display --help
```

### Command Line Options

- `--config, -c`: Path to configuration file (default: `services/display/config.toml`)
- `--name, -n`: Service instance name (default: `display-service`)
- `--log-level, -l`: Log level (DEBUG, INFO, WARNING, ERROR)
- `--windowed, -w`: Run in windowed mode (overrides config fullscreen setting)
- `--debug`: Enable debug overlay

### Keyboard Controls

When the display window has focus:

- **ESC** or **Q**: Exit the service gracefully
- **F11**: Toggle fullscreen mode
- **F1**: Toggle debug overlay (shows FPS and layer info)
- **Ctrl+C**: Signal-based graceful shutdown (works from terminal)

## Configuration

The service is configured via `config.toml`. Key settings:

```toml
# Display settings (must be inside [display] section)
[display]
fullscreen = false              # Whether to run in fullscreen mode
resolution = [1920, 1080]      # Window resolution (if not fullscreen)
vsync = true                   # Enable vertical sync
debug_overlay = false          # Show debug information

# ZeroMQ addresses
[zmq]
images_sub_address = "tcp://localhost:5555"  # Unified events channel
events_sub_address = "tcp://localhost:5555"  # Event messages

# Performance
[transitions]
crossfade_duration = 1.0       # Duration of image crossfades
preload_frames = true          # Preload transition frames
max_preload_mb = 500          # Memory limit for preloading

# Rendering backend
[rendering]
backend = "opengl"            # Rendering backend
shader_path = "shaders/"      # Path to custom shaders
```

> **Important**: All settings must be placed in their appropriate sections. For example, `fullscreen` must be inside the `[display]` section, not at the root level.

## Message Types

The display service subscribes to these ZMQ message types:

### Events Channel (port 5555)
- **DisplayMedia**: Primary display content (images, sequences, videos) with transition control
- **ImageReady**: Legacy image messages (deprecated, use DisplayMedia)
- **TransitionReady**: Custom transition videos
- **LoopReady**: Animated loop videos (future enhancement)

### Text & Video Control
- **TextOverlay**: Display text with speaker styling
- **RemoveText**: Remove specific text overlays
- **VideoMask**: Update video overlay mask for sand interaction

### System Events
- **EraChanged**: Notification of era transitions

## API Reference

### Direct Testing Interface

For testing without ZMQ infrastructure:

```python
from experimance_display import DisplayService, DisplayServiceConfig

# Create service
config = DisplayServiceConfig.from_file("config.toml")
service = DisplayService(config)

# Start service
await service.start()

# Trigger display updates directly
service.trigger_display_update("text_overlay", {
    "text_id": "test-1",
    "content": "Hello, World!",
    "speaker": "agent",
    "duration": 5.0
})

service.trigger_display_update("image_ready", {
    "image_id": "landscape-1",
    "uri": "file:///path/to/image.jpg",
    "image_type": "satellite_landscape"
})

# Clean up
await service.stop()
```

### Message Schemas

#### DisplayMedia Message (Primary)
```python
{
    "type": "DisplayMedia",
    "content_type": "image",             # Required: "image", "image_sequence", "video"
    
    # For IMAGE content_type
    "image_data": "<image_data>",        # Image data (file path, base64, numpy, PIL)
    "uri": "file:///path/to/image.png",  # Alternative: file reference
    
    # For IMAGE_SEQUENCE content_type  
    "sequence_path": "/path/to/frames/", # Directory with numbered images
    
    # For VIDEO content_type
    "video_path": "/path/to/video.mp4",  # Path to video file
    
    # Display properties
    "duration": 3.0,                     # Duration in seconds (sequences/videos)
    "loop": false,                       # Whether to loop content
    "fade_in": 0.5,                     # Fade in duration in seconds
    "fade_out": 0.5,                    # Fade out duration in seconds
    
    # Context information
    "era": "ai_future",                  # Current era context
    "biome": "coastal",                  # Current biome context
    "source_request_id": "<uuid>"        # Links to original RenderRequest
}
```

#### TextOverlay Message
```python
{
    "text_id": "unique-identifier",      # Required: unique ID for this text
    "content": "Text to display",        # Required: text content
    "speaker": "agent",                  # Optional: "agent", "system", "debug"
    "duration": 5.0,                     # Optional: auto-remove after seconds
    "position": "bottom_center",         # Optional: text positioning
    "replace": True                      # Optional: replace existing text with same ID
}
```

#### ImageReady Message
```python
{
    "image_id": "unique-identifier",     # Required: unique ID for this image
    "uri": "file:///path/image.jpg",     # Required: file path or URL
    "image_type": "satellite_landscape", # Optional: image type hint
    "transition_duration": 1.0          # Optional: custom transition timing
}
```

#### VideoMask Message
```python
{
    "mask_data": "base64_encoded_image", # Required: grayscale mask as base64
    "fade_in_duration": 0.5,            # Optional: fade in timing
    "fade_out_duration": 0.5            # Optional: fade out timing
}
```

## Architecture

The display service uses a layered rendering approach:

1. **Background Layer**: Satellite landscape images with crossfade transitions
2. **Video Overlay Layer**: Masked video responding to sand interaction
3. **Text Overlay Layer**: Multiple concurrent text items
4. **Debug Layer**: Performance metrics and system information

### Key Components

- **DisplayService**: Main service coordinating ZMQ and rendering
- **LayerManager**: Z-order rendering coordination
- **ImageRenderer**: Background image display with crossfades
- **VideoOverlayRenderer**: Masked video overlay with dynamic updates
- **TextOverlayManager**: Multiple text overlays with styling
- **ResourceCache**: Texture and resource management (future)

## Performance

### Targets
- **30fps minimum** at 1920x1080 resolution
- **60fps target** for smooth animations
- **Support for 4K** displays (3840x2560 tested)
- **10+ concurrent text overlays** without performance impact
- **<1GB GPU memory** usage under normal operation

### Optimization Features
- Texture caching and reuse
- Layer visibility optimization
- Frame pacing for consistent timing
- Efficient text rendering with batching
- Shader-based effects for GPU acceleration

## Development

### Running Tests

```bash
# Run basic functionality test
cd services/display
uv run python test_display_service.py

# Run with mock ZMQ messages
uv run python test_display.py
```

### Adding New Features

1. **New Message Type**: Add to `experimance_common.zmq_utils.MessageType`
2. **New Renderer**: Create in `renderers/` directory, register with LayerManager
3. **New Shader**: Add to `shaders/` directory, load in appropriate renderer
4. **Configuration**: Update `config.py` schema and `config.toml`

### Debugging

Enable debug mode for development:

```bash
# Debug logging + overlay
uv run -m experimance_display --windowed --debug --log-level DEBUG

# Performance profiling
uv run -m experimance_display --windowed --log-level INFO
# Press F1 to toggle debug overlay for FPS monitoring
```

## Troubleshooting

### Common Issues

**Service starts but window doesn't appear**
- Check if running in fullscreen on wrong monitor
- Try `--windowed` flag for development
- Verify OpenGL support: `glxinfo | grep OpenGL`

**Poor performance/low FPS**
- Disable vsync in config: `vsync = false`
- Reduce resolution: `resolution = [1280, 720]`
- Check GPU memory usage with system monitor
- Enable debug overlay (F1) to monitor FPS

**ZMQ connection issues**
- Verify port configuration matches other services
- Check if ZMQ services are running: `ss -tlnp | grep 5555`
- Review log output for connection errors

**Configuration issues**
- Make sure settings are in the correct TOML sections (e.g., `fullscreen` must be inside `[display]`)
- Use the `--log-level DEBUG` flag to see detailed configuration loading messages
- Check the path to your config file with `--config /path/to/config.toml`
- Use `--windowed` flag to override fullscreen setting for debugging

**Text not displaying**
- Verify message format matches schema
- Check text overlay layer is enabled
- Test with direct interface instead of ZMQ

**Images not loading**
- Verify file paths are absolute
- Check file permissions and format support
- Monitor log output for loading errors
- Test with known good image files

### Log Analysis

Key log messages to monitor:

```
INFO - DisplayService started on 1920x1080     # Service ready
INFO - Window initialized: 1920x1080           # Display setup
ERROR - Error handling ImageReady              # Message processing issues
DEBUG - FPS: 29.8                             # Performance monitoring
INFO - Exit key pressed, shutting down        # Clean shutdown
```

## Dependencies

### Core Dependencies
- **pyglet**: OpenGL rendering and window management
- **experimance_common**: ZMQ communication and configuration
- **pydantic**: Configuration validation
- **asyncio**: Concurrent event loop management

### Optional Dependencies
- **Pillow**: Additional image format support
- **numpy**: Advanced image processing
- **opencv-python**: Computer vision features (future)

### System Requirements
- **Python 3.11+**
- **OpenGL 3.3+** compatible graphics
- **4GB+ RAM** (8GB recommended for 4K)
- **Dedicated GPU** recommended for best performance
- **Linux/macOS/Windows** (tested on Linux)

## Future Enhancements

See [TODO.md](TODO.md) for planned features:

- **Loop Animation Support**: Animated loops from still images
- **Advanced Transitions**: Custom shader effects
- **Multi-Display**: Support for multiple monitors/projectors
- **Interactive Features**: Mouse/touch interaction
- **Performance Analytics**: Real-time performance monitoring
- **Resource Streaming**: Dynamic loading for large installations

## Contributing

1. Follow the [coding standards](../../README.md#coding-standards)
2. Add tests for new features
3. Update documentation for API changes
4. Profile performance impact of changes
5. Test on target hardware when possible

## License

Part of the Experimance project. See project root for license information.

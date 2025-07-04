# Image Transport and ZMQ Utilities Documentation

This document describes the modernized image transport utilities for the Experimance project, featuring enum-based transport modes and robust message handling.

## Architecture Overview

The image transport system is built around clean separation of concerns:

- **`experimance_common.image_utils`**: Pure image processing and format conversion
- **`experimance_common.zmq.zmq_utils`**: ZMQ communication and transport decisions  
- **`experimance_common.constants`**: Transport modes and configuration
- **`experimance_display.utils.pyglet_utils`**: Display-specific image loading

## Core Components

### ImageLoadFormat Enum

The `ImageLoadFormat` enum in `experimance_common.constants` replaces old boolean flags:

```python
from experimance_common.constants import ImageLoadFormat

# Available transport modes
ImageLoadFormat.FILE_PATH    # Load from file system path
ImageLoadFormat.BASE64       # Embedded base64-encoded data  
ImageLoadFormat.NUMPY        # Raw numpy array data
ImageLoadFormat.PIL          # PIL Image object
```

### IMAGE_TRANSPORT_MODES

Predefined transport configurations for different scenarios:

```python
from experimance_common.constants import IMAGE_TRANSPORT_MODES

# Available modes
IMAGE_TRANSPORT_MODES = {
    "local_file": ImageLoadFormat.FILE_PATH,     # Local file system
    "remote_base64": ImageLoadFormat.BASE64,     # Remote/network transfer
    "direct_numpy": ImageLoadFormat.NUMPY,      # In-memory processing
    "direct_pil": ImageLoadFormat.PIL           # PIL workflows
}
```

## Usage Patterns

### 1. Preparing ZMQ Messages

```python
from experimance_common.zmq.zmq_utils import prepare_image_message
from experimance_common.constants import ImageLoadFormat

# Automatic transport mode selection
message = prepare_image_message(
    image_data="/path/to/image.png",  # Source: file, PIL, numpy, or base64
    target_address="tcp://localhost:5555",
    transport_format=ImageLoadFormat.FILE_PATH,  # Explicit format
    mask_id="unique_id"
)

# The message will contain:
# - Optimized image data in the specified format
# - Transport metadata
# - Size and format information
```

### 2. Loading Images from Messages  

```python
from experimance_common.image_utils import load_image_from_message
from experimance_common.constants import ImageLoadFormat

# Universal image loading with enum-based format
pil_image = load_image_from_message(
    message=zmq_message,
    load_format=ImageLoadFormat.PIL,     # Desired output format
    image_field="image_data"             # Field containing image data
)

# Works with any input format, converts to requested output format
numpy_array = load_image_from_message(
    message=zmq_message,
    load_format=ImageLoadFormat.NUMPY    # Convert to numpy array
)
```

### 3. Display Service Integration

```python
from experimance_display.utils.pyglet_utils import load_pyglet_image_from_message

# Direct pyglet image loading for display service
pyglet_image, temp_file = load_pyglet_image_from_message(
    message=display_media_message,
    image_field="image_data",           # Field in DisplayMedia message
    image_id="display_id"               # For temp file naming
)

try:
    sprite = pyglet.sprite.Sprite(pyglet_image)
    # Use sprite for rendering...
finally:
    # Clean up temporary files
    if temp_file:
        cleanup_temp_file(temp_file)
```

## Transport Mode Selection

### Automatic Selection Logic

The system automatically chooses the best transport mode based on:

1. **Connection Type**: Local vs remote addresses
2. **Image Size**: Large images prefer file paths, small ones use direct transport
3. **Processing Context**: In-memory operations use numpy/PIL

```python
from experimance_common.zmq.zmq_utils import choose_image_transport_mode

# Automatic mode selection
transport_mode = choose_image_transport_mode(
    target_address="tcp://localhost:5555",  # Local connection
    image_size_bytes=2048000,               # 2MB image
    force_mode=None                         # Let system decide
)
# Returns: ImageLoadFormat.FILE_PATH (local + large)

transport_mode = choose_image_transport_mode(
    target_address="tcp://192.168.1.100:5555",  # Remote connection
    image_size_bytes=512000,                     # 512KB image  
    force_mode=None
)
# Returns: ImageLoadFormat.BASE64 (remote + medium size)
```

### Manual Override

```python
# Force specific transport mode regardless of conditions
message = prepare_image_message(
    image_data=pil_image,
    target_address="tcp://localhost:5555",
    transport_format=ImageLoadFormat.BASE64,  # Force base64 encoding
    force_mode=True
)
```

## Error Handling and Robustness

### Graceful Degradation

The transport system includes multiple fallback mechanisms:

```python
try:
    # Try primary transport mode
    image = load_image_from_message(message, ImageLoadFormat.PIL)
except (FileNotFoundError, PermissionError):
    # Fallback to base64 if file access fails
    image = load_image_from_message(message, ImageLoadFormat.BASE64)
except Exception as e:
    # Ultimate fallback
    logger.warning(f"Image loading failed: {e}")
    image = get_fallback_image()
```

### Temporary File Management

For file-based transport, the system automatically manages temporary files:

- **Automatic Cleanup**: Temp files are cleaned up after use
- **Collision Prevention**: Unique filenames prevent conflicts
- **Error Safety**: Cleanup happens even if processing fails

## Integration with DISPLAY_MEDIA

The DISPLAY_MEDIA message type uses these transport utilities:

```python
# Core service sending to display
display_media = {
    "type": "DisplayMedia",
    "content_type": "image",
    "image_data": "<transport_optimized_data>",  # Uses transport utilities
    "transport_format": "file_path",             # Enum value as string
    "fade_in": 0.5
}

# Display service receiving
pyglet_image, temp_file = load_pyglet_image_from_message(
    display_media, 
    image_field="image_data"
)
```

## Benefits of the New Design

1. **Type Safety**: Enum-based modes prevent string typos and invalid combinations
2. **Performance**: Automatic transport optimization based on context
3. **Maintainability**: Clear separation between transport logic and image processing
4. **Testability**: Enum values can be easily mocked and tested
5. **Extensibility**: New transport modes can be added without breaking existing code

## Migration from Legacy API

### Old Approach (Deprecated)
```python
# Old boolean-based API
message = prepare_image_message(
    image_data=data,
    use_file_transport=True,    # Boolean flag
    embed_image=False           # Another boolean flag
)
```

### New Approach
```python  
# New enum-based API
message = prepare_image_message(
    image_data=data,
    transport_format=ImageLoadFormat.FILE_PATH  # Clear, explicit mode
)
```

## Testing and Validation

The transport utilities include comprehensive test coverage:

- **Round-trip tests**: Ensure data integrity across all transport modes
- **Size optimization tests**: Verify transport mode selection logic
- **Error handling tests**: Validate graceful degradation
- **Integration tests**: Test full core→display message flow

See `utils/tests/test_image_utils.py` and `utils/tests/test_image_message_integration.py` for examples.

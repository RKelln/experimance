# Display Service Design Document

## Overview

The Display Service is responsible for rendering the visual output of the Experimance installation. It subscribes to multiple ZMQ channels to receive images, masks, videos, transitions, and text overlays, then composites them into a real-time display using OpenGL/Pyglet.

## Architecture

### Service Pattern
- **Type**: ZmqSubscriberService (no publishing required)
- **Subscription Channels**:
  - `images` (tcp://localhost:5558) - Images, transitions, loops from image_server
  - `events` (tcp://localhost:5555) - Control events from experimance core
  - `display_ctrl` (tcp://localhost:5560) - Text overlays and display commands

### Message Types

#### Images Channel Messages
```json
{
  "type": "ImageReady",
  "request_id": "uuid",
  "image_id": "uuid", 
  "uri": "file:///path/to/image.png",
  "metadata": {
    "image_type": "satellite_landscape" | "mask",
    "transition_video": "file:///path/to/transition.mp4" // optional
  }
}

{
  "type": "TransitionReady", 
  "request_id": "uuid",
  "transition_id": "uuid",
  "uri": "file:///path/to/transition.mp4",
  "from_image": "image_uuid",
  "to_image": "image_uuid"
}

{
  "type": "LoopReady",
  "loop_id": "uuid", 
  "uri": "file:///path/to/loop.mp4",
  "metadata": {
    "loop_type": "idle_animation"
  }
}
```

#### Display Control Messages
```json
{
  "type": "TextOverlay",
  "text_id": "uuid",
  "speaker": "agent" | "system",
  "content": "Hello, welcome to Experimance",
  "duration": 5.0, // seconds, or null for infinite
  "style": {
    "font_size": 32,
    "color": [255, 255, 255, 255],
    "position": "bottom_center",
    "background": true
  }
}

{
  "type": "RemoveText",
  "text_id": "uuid"
}

{
  "type": "VideoMask",
  "mask_id": "uuid", 
  "uri": "file:///path/to/mask.png",
  "fade_in_duration": 0.2,
  "fade_out_duration": 1.0
}
```

## Core Components

### 1. DisplayService (Main Service)
- Inherits from ZmqSubscriberService
- Manages the main render loop and window
- Coordinates all visual components
- Handles ZMQ message routing to appropriate handlers

### 2. LayerManager
- Manages rendering layers in z-order
- Layers: background_image, video_overlay, text_overlay, debug_overlay
- Handles cross-fading between states
- Optimizes rendering by skipping invisible layers

### 3. ImageRenderer
- Adapts ImageCycler from pyglet_test.py
- Handles satellite landscape image display
- Manages crossfade transitions between images
- Supports custom transition videos
- Caches recently used textures
- **Future Enhancement**: Current image can be replaced by animated loop video
  - LoopReady messages contain video of the same scene but animated (subtle movement, clouds, etc.)
  - Seamlessly switches between still image and loop video without user-visible transition
  - Falls back to still image if loop video unavailable or fails to load
  - Loop videos repeat indefinitely until next image/transition

### 4. VideoOverlayRenderer  
- Adapts VideoOverlay from pyglet_test.py
- Renders masked video overlay
- Handles dynamic mask updates from sand interaction
- Manages fade in/out timing based on interaction state

### 5. TextOverlayManager
- Renders text overlays with speaker-specific styling
- Manages multiple concurrent text items by ID
- Handles automatic expiration and manual removal
- Supports different positioning and styling options

### 6. TransitionManager
- Handles custom transition videos between images
- Falls back to shader-based crossfade when no custom transition
- Manages transition timing and state

### 7. ResourceCache
- Caches loaded textures, shaders, and fonts
- Implements LRU eviction policy  
- Manages GPU memory usage
- Preloads commonly used resources

## Rendering Pipeline

### Frame Render Order
1. **Background Layer**: Current satellite landscape image / video loop
2. **Video Layer**: Masked video overlay (when active)
3. **Text Layer**: All active text overlays
4. **Debug Layer**: FPS, performance metrics (if enabled)

### State Management
- **Current Image**: Currently displayed satellite landscape (or video loop)
- **Next Image**: Queued image waiting for transition
- **Video State**: Active/inactive, current mask, fade timers
- **Text State**: Dictionary of active text overlays by ID
- **Transition State**: Current transition type and progress

### Performance Optimizations
- Use sprite batching for multiple text overlays
- Implement texture streaming for large images
- Skip rendering invisible layers
- Use shader-based effects instead of CPU blending where possible
- Implement frame pacing to maintain consistent 60fps

## Configuration

### Display Settings
```toml
[display]
fullscreen = true
monitor = 0
resolution = [1920, 1080]
fps_limit = 60
vsync = true
debug_overlay = false

[zmq]
images_sub_address = "tcp://localhost:5558"
events_sub_address = "tcp://localhost:5555" 
display_ctrl_sub_address = "tcp://localhost:5560"

[rendering]
max_texture_cache_mb = 512
shader_path = "shaders/"
font_path = "fonts/"
preload_common_resources = true

[transitions]
default_crossfade_duration = 1.0
video_fade_in_duration = 0.2
video_fade_out_duration = 1.0
text_fade_duration = 0.3

[text_styles]
[text_styles.agent]
font_size = 28
color = [255, 255, 255, 255]
position = "bottom_center"
background = true
background_color = [0, 0, 0, 128]

[text_styles.system]
font_size = 24  
color = [200, 200, 200, 255]
position = "top_right"
background = false
```

## Error Handling

### File Loading Errors
- Log missing files but continue rendering
- Use placeholder textures for missing images
- Gracefully handle corrupted video files

### ZMQ Communication Errors
- Implement reconnection logic with exponential backoff
- Continue rendering with last known state during disconnections
- Log communication failures for debugging

### Rendering Errors
- Catch OpenGL errors and recover gracefully
- Fall back to software rendering if GPU unavailable
- Implement safe mode with minimal rendering

## Testing Strategy

### Unit Tests
- Test individual components (ImageRenderer, TextOverlayManager, etc.)
- Mock ZMQ messages for isolated testing
- Test resource caching and memory management

### Integration Tests  
- Test complete message flow from ZMQ to rendered output
- Test performance under load (multiple rapid updates)
- Test recovery from various error conditions

### Performance Tests
- Measure frame rate stability under various loads
- Test memory usage with large numbers of cached resources
- Benchmark GPU memory usage and texture streaming

## Future Enhancements

### Advanced Transitions
- Support for more complex transition effects
- Particle systems for special effects
- Dynamic lighting effects

### Interactive Elements
- Click/touch interaction support
- Hover effects for UI elements
- Gesture recognition integration

### Multiple Display Support
- Support for multiple monitors/projectors
- Different content per display
- Synchronized playback across displays

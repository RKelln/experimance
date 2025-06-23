# Display Service TODO

**Status Update (January 2025)**: Phase 1 and most of Phase 2 core infrastructure is now complete! The display service is fully functional with working signal handling, keyboard controls, and all major rendering components implemented. The service successfully handles all message types, provides comprehensive testing tools, and maintains target performance. See [README.md](README.md) for usage instructions.

## Phase 1: Core Infrastructure ✅ = Done, 🚧 = In Progress, ⏳ = Todo

### Service Foundation
- ✅ Create DisplayService class inheriting from ZmqSubscriberService
- ✅ Implement basic configuration loading using experimance_common patterns
- ✅ Set up main event loop with pyglet window management
- ✅ Implement graceful shutdown and cleanup
- ✅ Add testable interface for non-ZMQ control (dev/testing)

### Message Handling
- ✅ Add new MessageType enums to experimance_common.zmq_utils:
  - TEXT_OVERLAY
  - REMOVE_TEXT  
  - CHANGE_MAP
- ✅ Implement message routing to appropriate handlers
- ✅ Add validation for incoming messages

### Basic Rendering (30fps target)
- ✅ Extract and adapt ImageCycler from pyglet_test.py
- ✅ Extract and adapt VideoOverlay from pyglet_test.py  
- ✅ Create LayerManager for z-order rendering
- ✅ Implement basic crossfade between images

### Text Overlays (Phase 1 - Essential)
- ✅ TextOverlayManager for multiple concurrent text items
- ✅ Handle TextOverlay and RemoveText messages
- ✅ Support text replacement with same ID (streaming text)
- ✅ Speaker-specific styling (agent vs system)
- ✅ Automatic expiration based on duration

### Video Overlay (Phase 1 - Responsiveness)
- ✅ VideoOverlayRenderer with dynamic mask support
- ✅ Handle VideoMask messages to update overlay mask
- ✅ Implement fade in/out timing based on sand interaction
- ✅ Basic shader-based masking using grayscale masks (basic implementation)

### Testing & Validation
- ✅ Basic syntax validation and import testing
- ✅ Service initialization testing (window creation, component loading)
- ✅ Keyboard controls (ESC/Q keys for exit)
- ✅ Signal handling (Ctrl+C graceful shutdown)
- ✅ Test text overlay functionality with ZMQ messages
- ✅ Test video overlay with mask updates (CLI tool available)
- ✅ Test image loading and crossfade transitions

## Phase 2: Core Features

### Image Display & Loop Animation 
- ✅ ImageRenderer class with texture caching
- ✅ Handle ImageReady messages from images channel
- ✅ Support for satellite_landscape and mask image types
- ✅ Automatic crossfade on new image receipt
- ⏳ **Future**: Handle LoopReady messages for animated loops of still images
- ⏳ **Future**: Seamless switching between still images and looping videos
- ⏳ **Future**: Loop video playback with automatic restart

### Video Overlay
- ✅ VideoOverlayRenderer with dynamic mask support
- ✅ Handle VideoMask messages to update overlay mask
- ✅ Implement fade in/out timing based on sand interaction
- ✅ Basic shader-based masking using grayscale masks

### Text Overlays
- ✅ TextOverlayManager for multiple concurrent text items
- ✅ Handle TextOverlay and RemoveText messages
- ✅ Speaker-specific styling (agent vs system)
- ✅ Automatic expiration based on duration
- ✅ Different positioning options (bottom_center, top_right, etc.)

### Core Rendering Infrastructure
- ✅ LayerManager for z-order rendering coordination
- ✅ DebugOverlayRenderer for performance metrics and system info
- ✅ Window management (fullscreen/windowed modes)
- ✅ Frame timing and FPS management (30fps target achieved)
- ✅ Headless mode support for testing
- ✅ Configuration system with TOML support

### Testing & Development Tools
- ✅ CLI tool for manual testing of all message types
- ✅ Direct interface for non-ZMQ testing
- ✅ Comprehensive test suite with mock data
- ✅ Title screen functionality
- ✅ Debug text display for all positions and speakers

## Phase 3: Advanced Features

### Custom Transitions
- ⏳ TransitionManager for custom transition videos
- ⏳ Handle TransitionReady messages
- ⏳ Fall back to shader crossfade when no custom transition
- ⏳ Smooth timing coordination between transitions and images

### Resource Management
- ⏳ ResourceCache with LRU eviction
- ⏳ GPU memory monitoring and management
- ⏳ Texture streaming for large images
- ⏳ Preloading of common resources

### Performance Optimization
- ⏳ Frame pacing for consistent 60fps
- ⏳ Sprite batching for text overlays
- ⏳ Layer visibility optimization
- ⏳ Shader compilation caching

## Phase 4: Polish & Testing

### Error Handling
- ✅ Graceful handling of missing files
- ⏳ ZMQ reconnection with exponential backoff
- ⏳ OpenGL error recovery
- ⏳ Safe mode fallback rendering

### Configuration
- ✅ Complete configuration schema validation
- ⏳ Runtime configuration updates
- ⏳ Multi-monitor support configuration
- ⏳ Performance tuning options

### Testing
- ✅ Unit tests for all core components
- ✅ Integration tests with mock ZMQ messages
- ⏳ Performance benchmarking
- ⏳ Memory leak testing
- ⏳ Long-running stability tests

### Documentation
- ✅ API documentation for all public interfaces
- ✅ Configuration reference
- ✅ Troubleshooting guide
- ⏳ Performance tuning guide

## Phase 5: Advanced Features (Future)

### Enhanced Transitions
- ⏳ More transition effects (wipe, dissolve, zoom, etc.)
- ⏳ Particle system integration
- ⏳ Dynamic lighting effects

### Interactive Features
- ⏳ Mouse/touch interaction support
- ⏳ UI hover effects
- ⏳ Gesture recognition hooks

### Multi-Display
- ⏳ Multiple monitor/projector support
- ⏳ Different content per display
- ⏳ Synchronized playback

## File Structure

```
services/display/src/experimance_display/
├── __init__.py
├── display_service.py          # Main service class
├── config.py                   # Configuration schema
├── renderers/
│   ├── __init__.py
│   ├── image_renderer.py       # Satellite landscape images
│   ├── video_overlay_renderer.py  # Masked video overlay
│   ├── text_overlay_manager.py    # Text overlays
│   └── layer_manager.py        # Z-order layer management
├── transitions/
│   ├── __init__.py
│   ├── transition_manager.py   # Custom transition videos
│   └── crossfade.py           # Shader-based crossfade
├── resources/
│   ├── __init__.py
│   ├── resource_cache.py      # Texture/resource caching
│   └── shader_manager.py      # Shader compilation & caching
├── utils/
│   ├── __init__.py
│   ├── gl_utils.py           # OpenGL utilities
│   └── timing.py             # Frame timing utilities
└── shaders/
    ├── crossfade.vert
    ├── crossfade.frag
    ├── video_mask.vert
    └── video_mask.frag
```

## Dependencies

### Required
- pyglet (OpenGL rendering)
- experimance_common (ZMQ, config, logging)
- pydantic (config validation)
- numpy (image processing)

### Optional
- Pillow (additional image format support)
- opencv-python (advanced image processing)

## Implementation Notes

### Performance Targets
- Maintain 60fps at 1920x1080 resolution
- Support up to 10 concurrent text overlays
- Handle image updates every 2-3 seconds without frame drops
- Keep GPU memory usage under 1GB

### Key Technical Decisions
- Use pyglet for proven performance from prototype
- Implement sprite batching for text overlays
- Use shader-based effects for better GPU utilization
- Cache textures in GPU memory with LRU eviction
- Separate resource loading from rendering thread where possible

### Testing Priority
1. Core rendering stability
2. ZMQ message handling reliability  
3. Memory management over long runs
4. Performance under load
5. Error recovery scenarios

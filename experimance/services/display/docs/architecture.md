# Display Service Architecture

## Overview

The Display Service renders the visual output of the Experimance installation. It subscribes to a ZMQ events channel, receives messages describing images, masks, text overlays, and transitions, then composites them in real time using OpenGL via pyglet.

**Files touched by this subsystem:**
- `src/experimance_display/display_service.py` — main service class and event loop
- `src/experimance_display/config.py` — Pydantic configuration schema
- `src/experimance_display/renderers/` — all renderer components
- `src/experimance_display/utils/pyglet_utils.py` — OpenGL/pyglet helpers

## Service Pattern

- **Type**: ZMQ subscriber (no publishing)
- **Subscription channel**: `events` on `tcp://localhost:5555` (unified inter-service channel)
- **Subscribed message topics**: `DISPLAY_MEDIA`, `DISPLAY_TEXT`, `REMOVE_TEXT`, `CHANGE_MAP`

> The older `display_ctrl` channel (port 5560) is no longer used. All messages arrive on the single `events` channel.

## Message Types

### Display Content

```json
{
  "type": "DisplayMedia",
  "request_id": "uuid",
  "content_type": "image",
  "uri": "file:///path/to/image.png",
  "position": [1200, 300],
  "fade_in": 2.0
}
```

`content_type` values: `"image"`, `"image_sequence"`, `"video"`, `"clear"`

### Text Overlays

```json
{
  "type": "DisplayText",
  "text_id": "uuid",
  "speaker": "agent",
  "content": "Hello, welcome to Experimance",
  "duration": 5.0,
  "style": {
    "position": "bottom_center"
  }
}
```

```json
{
  "type": "RemoveText",
  "text_id": "uuid"
}
```

### Map / Video Mask Update

```json
{
  "type": "ChangeMap",
  "mask_id": "uuid",
  "uri": "file:///path/to/mask.png",
  "fade_in_duration": 0.2,
  "fade_out_duration": 1.0
}
```

> In code, this message type is `MessageType.CHANGE_MAP` (previously called `VideoMask` in older docs).

### Other

```json
{
  "type": "TransitionReady",
  "transition_id": "uuid",
  "uri": "file:///path/to/transition.mp4",
  "from_image": "image_uuid",
  "to_image": "image_uuid"
}
```

```json
{
  "type": "LoopReady",
  "loop_id": "uuid",
  "uri": "file:///path/to/loop.mp4",
  "still_uri": "file:///path/to/still.png"
}
```

## Core Components

### DisplayService
- Inherits from `BaseService`
- Owns the pyglet window and main event loop
- Routes incoming ZMQ messages to renderer methods
- Supports windowed, fullscreen, and headless (test) modes

### LayerManager
- Manages rendering layers in Z-order
- Layers: `background_image`, `panorama`, `video_overlay`, `text_overlay`, `debug_overlay`
- Skips invisible layers for performance

### ImageRenderer
- Displays satellite landscape images
- Manages crossfade transitions between images
- Texture caching for recently used images

### PanoramaRenderer
- Wide-aspect panoramic display with base images and positioned tiles
- GPU-accelerated Gaussian blur with σ→0 transition
- Horizontal mirroring support
- See [panorama.md](panorama.md) for full details

### VideoOverlayRenderer
- Renders masked video overlay
- Handles dynamic mask updates (`ChangeMap` messages)
- Manages fade-in/out timing

### TextOverlayManager
- Multiple concurrent text items keyed by `text_id`
- Speaker-specific styling (`agent`, `system`, `debug`, `title`)
- Automatic expiration by duration
- Streaming text (replace existing item with same ID)

### TransitionManager
- Custom transition videos between images
- Falls back to shader-based crossfade when no video provided

### DebugOverlayRenderer
- FPS counter and layer info
- Toggled via F1 key or `--debug` flag

### ShaderRenderer
- Runs configurable full-screen GLSL fragment shader effects
- See [shaders.md](shaders.md) for details

## Rendering Pipeline

Frame render order (back to front):

1. Background image / panorama base
2. Panorama tiles
3. Video overlay (when active)
4. Text overlays
5. Shader effects (vignette, sparks, etc.)
6. Debug overlay (when enabled)

## State Management

| State | Description |
|-------|-------------|
| Current image | The currently displayed background image |
| Next image | Queued image awaiting crossfade |
| Video state | Active/inactive, current mask, fade timers |
| Text state | Dict of active text overlays by `text_id` |
| Transition state | Current transition type and progress |
| Panorama state | Base image, tile list, blur progress |

## Error Handling

- **Missing files**: Logged and skipped; rendering continues with last known state
- **ZMQ disconnect**: Service continues rendering; reconnection is handled by the subscriber infrastructure
- **OpenGL errors**: Caught and logged; rendering falls through gracefully
- **Headless mode**: Window creation is skipped entirely; used for unit tests

## Testing Strategy

See [testing.md](testing.md) for full details.

- **Headless tests** (`tests/test_display_headless*.py`): Mock the window and OpenGL; suitable for CI
- **Integration tests** (`tests/test_integration.py`): Full message-to-render flow
- **Window tests** (`tests/test_display*.py`): Require a live display; run manually
- **Direct interface**: `DisplayService.trigger_display_update()` accepts messages without ZMQ

## Future Enhancements

- TransitionManager: custom transition videos (Phase 3, not yet implemented)
- ResourceCache: LRU GPU texture eviction (Phase 3)
- Loop animation: still-to-video seamless switch (see [loop-animation.md](loop-animation.md))
- Multi-display: multiple projector/monitor support

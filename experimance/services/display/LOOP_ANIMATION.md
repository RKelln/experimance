# Loop Animation Architecture

## Overview

Loop animation is a future enhancement to the Display Service that allows still satellite landscape images to be replaced with subtle animated loops. This creates a more engaging visual experience during idle periods while maintaining the core image-based narrative.

## Technical Flow

### 1. Loop Generation (External Service)
```
LoopRequest → animate_worker → LoopReady
```

**LoopRequest** (sent by experimance core):
```json
{
  "type": "LoopRequest", 
  "request_id": "uuid",
  "still_uri": "file:///path/to/generated_image.png",
  "style": "subtle_clouds" | "gentle_water" | "ambient_movement",
  "duration_s": 10.0,
  "loop_seamlessly": true
}
```

**LoopReady** (sent by animate_worker):
```json
{
  "type": "LoopReady",
  "request_id": "uuid", 
  "loop_id": "uuid",
  "still_uri": "file:///path/to/generated_image.png",
  "video_uri": "file:///path/to/animated_loop.mp4",
  "duration_s": 10.0,
  "is_seamless": true
}
```

### 2. Display Service Integration

The Display Service receives LoopReady messages on the `images` channel and needs to:

1. **Associate Loop with Current Image**: Match the `still_uri` to determine if this loop applies to the currently displayed image
2. **Preload Video**: Load the loop video into memory/GPU
3. **Seamless Transition**: Switch from still image to looping video without visible interruption
4. **Loop Management**: Continuously loop the video until replaced by new content

## Implementation Strategy

### Phase 1: Basic Loop Support
- Extend ImageRenderer to handle both still images and loop videos
- Add video playback capability using pyglet's media player
- Implement seamless switching between image texture and video texture
- Handle loop restart when video reaches end

### Phase 2: Advanced Features
- Crossfade between still image and loop video (subtle fade-in of animation)
- Multiple loop styles per image (choose best match)
- Memory management for multiple preloaded loops
- Graceful fallback when loops fail to load or play

## Code Architecture

### ImageRenderer Enhancements

```python
class ImageRenderer:
    def __init__(self):
        self.current_image = None
        self.current_video_loop = None
        self.video_player = None
        self.loop_fade_timer = 0.0
        self.state = "image"  # "image", "fading_to_loop", "loop"
    
    def handle_image_ready(self, message):
        """Handle new still image."""
        # Load new image, crossfade as normal
        pass
    
    def handle_loop_ready(self, message):
        """Handle loop video for current image."""
        if self._is_loop_for_current_image(message):
            self._prepare_loop_transition(message)
    
    def _prepare_loop_transition(self, loop_message):
        """Prepare to transition from still to loop."""
        # Load video, prepare for seamless switch
        pass
    
    def update(self, dt):
        """Update animation state."""
        if self.state == "fading_to_loop":
            self._update_loop_fade(dt)
        elif self.state == "loop":
            self._update_loop_playback(dt)
    
    def render(self):
        """Render current state."""
        if self.state == "image":
            self._render_image()
        elif self.state == "fading_to_loop":
            self._render_image_to_loop_crossfade()
        elif self.state == "loop":
            self._render_loop()
```

### Loop Management

```python
class LoopManager:
    def __init__(self):
        self.active_loops = {}  # image_id -> loop_info
        self.preloaded_videos = {}  # loop_id -> video_player
    
    def register_loop(self, loop_message):
        """Register a loop for future use."""
        pass
    
    def get_loop_for_image(self, image_id):
        """Get best loop for given image."""
        pass
    
    def preload_loop_video(self, video_uri):
        """Preload video into memory."""
        pass
    
    def cleanup_unused_loops(self):
        """Clean up loops no longer needed."""
        pass
```

## Configuration

### Loop-Specific Settings
```toml
[loops]
enabled = true                    # Enable loop animation feature
fade_in_duration = 2.0           # Time to fade from image to loop
preload_timeout = 10.0           # Max time to wait for loop preload
max_cache_loops = 5              # Maximum loops to keep in memory
fallback_on_failure = true       # Fall back to still image on loop failure

[loops.styles]
# Style preferences for different loop types
subtle_clouds = { priority = 1, fade_duration = 3.0 }
gentle_water = { priority = 2, fade_duration = 2.0 }
ambient_movement = { priority = 3, fade_duration = 1.5 }
```

## Error Handling

### Loop Loading Failures
- **Video Load Error**: Log error, continue with still image
- **Video Codec Unsupported**: Log warning, fallback to still
- **Memory Exhaustion**: Clean up old loops, retry with reduced cache

### Playback Issues
- **Video Corruption**: Stop loop, return to still image
- **Performance Degradation**: Disable loops temporarily
- **Audio Sync Issues**: Mute video audio (loops should be silent anyway)

## Performance Considerations

### Memory Management
- Limit concurrent loaded videos (max 3-5 loops)
- Use video compression optimized for looping (H.264 with GOP=1)
- Preload loops asynchronously to avoid frame drops

### GPU Usage
- Share texture memory between image and video rendering
- Use hardware video decoding when available
- Implement frame prediction to smooth playback

### Timing Synchronization
- Ensure loop restart is frame-perfect (no visible seam)
- Handle variable frame rates gracefully
- Synchronize with main render loop timing

## Testing Strategy

### Unit Tests
- Test loop association with images
- Test video loading and playback
- Test seamless transitions
- Test memory cleanup

### Integration Tests
- Test complete flow: LoopRequest → LoopReady → Display
- Test error recovery scenarios
- Test performance under load
- Test memory usage over time

### Performance Tests
- Measure frame rate impact of video loops
- Test GPU memory usage with multiple loops
- Benchmark loop transition smoothness

## Future Enhancements

### Advanced Animation
- Multiple loop layers (clouds + water movement)
- Parallax effects for depth
- Seasonal variations (snow, rain, fog effects)

### Interactive Loops
- Loops that respond to user interaction intensity
- Different loop speeds based on era/biome
- Loops that incorporate depth map data

### AI-Generated Loops
- Integration with video generation models
- Style transfer from still image to loop
- Procedural animation generation

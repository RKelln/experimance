# Loop Animation (Future Feature)

Loop animation allows a still satellite landscape image to be seamlessly replaced with a subtle animated video loop — giving the appearance of gentle movement (clouds drifting, water rippling) during idle periods.

This feature is **not yet implemented**. This document captures the intended design.

**Files that will be touched when implemented:**
- `src/experimance_display/renderers/image_renderer.py` — `ImageRenderer` extension
- `src/experimance_display/display_service.py` — `LoopReady` message handler

## Message Flow

```
[external animate service]
    LoopRequest  →  animate_worker  →  LoopReady
                                            ↓
                                    DisplayService
                                    (match to current image,
                                     preload, seamless switch)
```

### `LoopRequest` (sent by experimance core)

```json
{
  "type": "LoopRequest",
  "request_id": "uuid",
  "still_uri": "file:///path/to/generated_image.png",
  "style": "subtle_clouds",
  "duration_s": 10.0,
  "loop_seamlessly": true
}
```

`style` values: `"subtle_clouds"`, `"gentle_water"`, `"ambient_movement"`

### `LoopReady` (sent by animate_worker)

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

## Display Service Behaviour

On receipt of `LoopReady`:

1. Match `still_uri` to the currently displayed image
2. Preload the video into GPU memory asynchronously
3. Transition from still texture → video texture without a visible seam
4. Loop the video indefinitely until replaced by new `DisplayMedia` content

## Planned Configuration

```toml
[loops]
enabled = true
fade_in_duration = 2.0      # Seconds to crossfade still → loop
preload_timeout = 10.0      # Max preload wait time (seconds)
max_cache_loops = 5
fallback_on_failure = true  # Stay on still image if loop fails
```

## Error Handling

| Failure | Behaviour |
|---------|-----------|
| Video load error | Log and continue showing still image |
| Unsupported codec | Log warning, stay on still |
| Memory exhaustion | Evict oldest loop, retry |
| Playback corruption | Stop loop, return to still |

## Performance Notes

- Keep concurrent loaded videos to 3–5 maximum
- Use H.264 with GOP=1 for smooth looping
- Preload asynchronously to avoid frame drops
- Loop restart should be frame-perfect (no visible seam)

## Phased Implementation Plan

**Phase 1 (basic)**
- Extend `ImageRenderer` to hold a video player alongside the still texture
- Handle `LoopReady` in `DisplayService`
- Seamless still → video switch, automatic restart at end

**Phase 2 (polish)**
- Crossfade from still to loop (subtle fade-in of motion)
- Multiple loop styles per image
- Memory management for preloaded loops
- Graceful fallback

## Future Ideas

- Multiple loop layers (clouds + water simultaneously)
- Parallax depth effects
- Seasonal variations (snow, rain, fog)
- Loops that vary speed with era/biome context

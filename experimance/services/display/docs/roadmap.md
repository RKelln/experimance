# Display Service Roadmap

Current implementation status and near-term goals.

**Last reviewed**: January 2025

## Status Summary

Core rendering is fully functional. The service handles all primary message types, maintains target performance at 1920×1080, and has a comprehensive test suite. The items below reflect what is still missing or known to be rough.

## Done

- DisplayService class, ZMQ subscriber, graceful shutdown
- Fullscreen, windowed, and headless modes
- Keyboard controls (ESC/Q exit, F11 fullscreen toggle, F1 debug overlay)
- Signal handling (Ctrl+C)
- ImageRenderer with crossfade transitions
- PanoramaRenderer with base images, tiles, blur-to-sharp, and mirroring
- VideoOverlayRenderer with dynamic mask updates
- TextOverlayManager — multiple concurrent items, speaker styles, streaming text
- DebugOverlayRenderer — FPS, layer info
- ShaderRenderer — turbulent vignette, rising sparks effects
- Configuration system (Pydantic + TOML)
- CLI testing tool (`experimance-display-cli`)
- Headless test suite and direct (non-ZMQ) interface

## Near-Term Goals

### TransitionManager (Phase 3)
Custom transition videos between images (TransitionReady messages).
Falls back to shader crossfade when no video is provided.
- `src/experimance_display/transitions/` is scaffolded but empty

### ResourceCache (Phase 3)
LRU texture eviction to keep GPU memory under control during long-running sessions.
Currently textures accumulate until process restart.

### ZMQ reconnection (Phase 4)
Exponential backoff reconnection when the events channel goes away.
Currently the service continues with last state but does not attempt to reconnect.

### Loop animation (future)
Seamless still-image → animated-video switching when a `LoopReady` message arrives.
See [loop-animation.md](loop-animation.md) for full design.

### Performance: frame pacing (Phase 3)
Consistent 60fps frame pacing — currently hitting 30fps minimum but not reliably 60fps.

### Runtime config reload
Apply config changes without restarting the service.

## Known Gaps

| Area | Gap |
|------|-----|
| `transitions/` | Directory and `__init__.py` exist but `transition_manager.py` and `crossfade.py` are not yet written |
| `resources/` | `resource_cache.py` not yet written |
| `utils/` | `gl_utils.py` and `timing.py` referenced in TODO but not yet written |
| `config.toml` | `fade_out_duration` key under `[title_screen]` is not a recognised field — should be `fade_duration` |
| Root test files | `test_config.py`, `test_display.py`, `test_display_service.py`, `test_integration.py` at the service root are empty stubs and can be removed |
| Multi-display | No support for multiple monitors or projectors from a single service instance |
| Performance benchmarks | No automated benchmarking or memory-leak detection yet |
| Long-running stability | No test coverage for multi-hour operation |

## Phase 5 (Long-Term)

- More transition effects: wipe, dissolve, zoom, particles
- Dynamic lighting effects
- Mouse/touch interaction
- Multi-display: different content per projector, synchronised playback
- Fog and fairy lights projection system (see [magic-effects.md](magic-effects.md))

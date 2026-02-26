# tune_detector.py

Launcher for the interactive HOG+MOG2 audience detector tuning tool. Provides a live webcam feed with real-time parameter adjustment via trackbars, so you can tune detection for a specific venue lighting and layout without modifying code.

See `scripts/tune_detector.py` (thin launcher — actual implementation is in `services/agent/src/agent/vision/interactive_detector_tuning.py`).

## Quick Start

```bash
# Start with default profile (indoor_office)
uv run python scripts/tune_detector.py

# List available profiles
uv run python scripts/tune_detector.py --list-profiles

# Start with a specific profile
uv run python scripts/tune_detector.py --profile gallery_dim

# Use camera index 1 (if multiple cameras)
uv run python scripts/tune_detector.py --camera 1

# Verbose logging
uv run python scripts/tune_detector.py --verbose
```

## Interactive Controls

| Key | Action |
|---|---|
| `q` / `ESC` | Quit |
| `s` | Save current settings to `{profile_name}_tuned.toml` |
| `r` | Reset to original profile values |
| `h` | Toggle HOG detection visualization (green boxes) |
| `m` | Toggle motion detection overlay (red contours) |
| `i` | Toggle parameter info display |

## Tunable Parameters

| Parameter | Description |
|---|---|
| Detection Scale Factor | Frame downscale before detection (smaller = faster, less accurate) |
| Min Person Height | Minimum bounding-box height in pixels for a valid person |
| Motion Threshold | Minimum contour area to count as motion |
| Motion Intensity | Sensitivity threshold for motion detection |
| Stability | Majority-vote window size for temporal smoothing |
| HOG Threshold | Person detection confidence threshold |
| HOG Scale | Pyramid scale factor for multi-scale HOG |
| Win Stride | HOG window stride (larger = faster, less accurate) |
| MOG2 Var Threshold | Background subtraction variance threshold |
| MOG2 History | Number of frames used to build background model |
| Person Base Confidence | Base score for each detected person |
| Motion Weight | Weight of motion confidence in the final combined score |

## Workflow

1. Pick the closest existing profile: `--list-profiles`
2. Start the tool with that profile and watch the live feed
3. Adjust parameters gradually — observe the effect on green boxes and red contours
4. Test under different lighting conditions and with real audience scenarios
5. Press `s` to save; the file is written next to your project's config
6. Save multiple profiles for different environments (e.g. `gallery_bright`, `gallery_dim`)

## Tips

- Start with a profile that roughly matches your environment before fine-tuning
- Use `--verbose` to see numeric detection scores in the terminal
- HOG works best with reasonable person height (at least 60–80px); adjust camera framing first
- MOG2 history affects how quickly the background model adapts to new people standing still

## Requirements

- Webcam connected to the system
- OpenCV (`cv2`) — included in agent service dependencies
- No pipecat dependencies required (loads only core detection modules)

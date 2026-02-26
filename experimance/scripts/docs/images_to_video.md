# images_to_video.py

Converts a folder of timestamped images into a video with optional crossfade transitions. Useful for turning generated gallery images into a timelapse or review reel.

See `scripts/images_to_video.py`.  
Related: [`combine_triptychs.md`](combine_triptychs.md)

## Quick Start

```bash
# Basic: all images in a directory → output.mp4
uv run scripts/images_to_video.py media/images/generated -o output.mp4

# Filter by time range
uv run scripts/images_to_video.py media/images/generated -o output.mp4 \
    --start-time "2025-07-27 20:00:00" \
    --end-time "2025-07-27 22:00:00"

# Preview: list matched images without creating video
uv run scripts/images_to_video.py media/images/generated --list-only
```

## Options

| Flag | Default | Description |
|---|---|---|
| `DIR` | (required) | Directory containing images |
| `-o / --output` | `output.mp4` | Output video file path |
| `--start-time DATETIME` | — | Include images on or after this time (`YYYY-MM-DD HH:MM:SS`) |
| `--end-time DATETIME` | — | Include images on or before this time |
| `--frames-per-image N` | 3 | How many video frames each image is held |
| `--blend-frames N` | 1 | Frames used for the crossfade transition |
| `--fps N` | 30 | Output video frame rate |
| `--list-only` | false | Print matched images and exit, no video created |

## Filename Format

Images must follow the naming convention:
```
<prefix>_YYYYMMDD_HHMMSS_<uuid>.jpg
```
Example: `gen_20250727_201523_a3f9b1c2.jpg`

Files that don't match this pattern are skipped. Images are sorted by embedded timestamp before processing.

## Requirements

```bash
uv add ffmpegcv numpy pillow
```

- `ffmpegcv` — video encoding
- `numpy` — frame blending
- `Pillow (PIL)` — image loading
- `ffmpeg` must be installed system-wide (used by ffmpegcv)

# combine_triptychs.py

Stitches sets of triptych tile images into seamless full-width panoramas with linear blending in the overlap regions. The output can then be fed into `images_to_video.py` to create a video.

See `scripts/combine_triptychs.py`.  
Related: [`images_to_video.md`](images_to_video.md)

## Quick Start

```bash
# Combine all triptychs in a directory
uv run scripts/combine_triptychs.py /path/to/images -o /path/to/combined

# Filter by time range
uv run scripts/combine_triptychs.py /path/to/images -o /path/to/combined \
    --start-time "2025-09-10 14:00:00" \
    --end-time   "2025-09-10 18:00:00"

# Preview: list groups without combining
uv run scripts/combine_triptychs.py /path/to/images --list-only

# Then make a video from the combined images
uv run scripts/images_to_video.py /path/to/combined -o output.mp4
```

## Image Group Format

The script groups images by timestamp prefix. Each group consists of:

| File | Size | Role |
|---|---|---|
| Preview | 2240×424 | Panorama thumbnail — used to identify the group |
| Left tile | 1344×760 | Left panel |
| Middle tile | 1480×760 | Middle panel (overlaps the left on its left side) |
| Right tile | 1480×760 | Right panel (overlaps the middle on its left side) |

Tiles are stitched left→middle→right with linear blending in the overlap zones to eliminate seams.

## Options

| Flag | Default | Description |
|---|---|---|
| `DIR` | (required) | Input directory containing triptych images |
| `-o / --output` | (required) | Output directory for combined panoramas |
| `--start-time DATETIME` | — | Process groups at or after this time |
| `--end-time DATETIME` | — | Process groups at or before this time |
| `--list-only` | false | List groups and exit without stitching |
| `--tile-width N` | 1344 | Override expected left tile width |
| `--tile-height N` | 760 | Override expected tile height |

## Requirements

```bash
uv add numpy pillow
```

- `numpy` — overlap blending
- `Pillow (PIL)` — image loading and stitching

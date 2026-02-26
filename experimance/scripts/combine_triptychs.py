#!/usr/bin/env python3
"""
Combine triptych tile images into seamless panoramas with blended overlaps.

Processes groups of: preview (2240x424) + left tile (1344x760) +
middle tile (1480x760) + right tile (1480x760).

Usage:
    uv run scripts/combine_triptychs.py /path/to/images -o /path/to/combined
    uv run scripts/combine_triptychs.py /path/to/images --list-only
    # Then create a video:
    uv run scripts/images_to_video.py /path/to/combined -o output.mp4

See docs/combine_triptychs.md for full option reference.

Requirements: numpy, Pillow
"""

import argparse
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: uv add numpy pillow")
    sys.exit(1)


# Default tile dimensions from core.toml
DEFAULT_TILE_WIDTH = 1344
DEFAULT_TILE_HEIGHT = 760


def parse_filename_timestamp(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from filename in format: prefix_YYYYMMDD_HHMMSS.ext

    Args:
        filename: The image filename

    Returns:
        datetime object if parsed successfully, None otherwise
    """
    pattern = r"_(\d{8})_(\d{6})\.\w+$"
    match = re.search(pattern, filename)

    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


class TriptychGroup:
    """Represents a group of images forming a triptych."""

    def __init__(self, preview: Path, left: Path, middle: Path, right: Path, timestamp: datetime):
        self.preview = preview
        self.left = left
        self.middle = middle
        self.right = right
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"TriptychGroup({self.timestamp}, left={self.left.name})"


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Get image dimensions (width, height) without loading full image."""
    with Image.open(img_path) as img:
        return img.size


def classify_image(size: tuple[int, int]) -> str:
    """
    Classify an image by its dimensions.

    Returns:
        'preview' - panorama preview (wider, shorter)
        'left' - left tile (base width)
        'extended' - middle or right tile (extended width)
        'unknown' - doesn't match expected patterns
    """
    width, height = size

    # Preview is panoramic (wider than tall, specifically around 2240x424)
    if width > 2000 and height < 500:
        return "preview"

    # Tiles are taller (around 760 height)
    if 700 < height < 900:
        if 1300 < width < 1400:  # Base tile width ~1344
            return "left"
        elif 1400 < width < 1600:  # Extended tile width ~1480
            return "extended"

    return "unknown"


def collect_image_groups(
    image_dir: Path,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> list[TriptychGroup]:
    """
    Collect images and group them into triptych sets.

    Args:
        image_dir: Directory containing images
        start_time: Optional start time filter
        end_time: Optional end time filter
        extensions: Tuple of valid image extensions

    Returns:
        List of TriptychGroup objects sorted by timestamp
    """
    # Collect all images with timestamps and dimensions
    images: list[tuple[Path, datetime, tuple[int, int], str]] = []

    for img_path in image_dir.iterdir():
        if not img_path.is_file() or img_path.is_symlink():
            continue

        if img_path.suffix.lower() not in extensions:
            continue

        timestamp = parse_filename_timestamp(img_path.name)
        if timestamp is None:
            continue

        if start_time and timestamp < start_time:
            continue
        if end_time and timestamp > end_time:
            continue

        try:
            size = get_image_size(img_path)
            img_type = classify_image(size)
            images.append((img_path, timestamp, size, img_type))
        except Exception as e:
            print(f"Warning: Could not read image {img_path.name}: {e}")
            continue

    images.sort(key=lambda x: x[1])

    # Group images by looking for the pattern: preview(s), left, middle, right
    groups: list[TriptychGroup] = []

    i = 0
    while i < len(images):
        path, timestamp, size, img_type = images[i]

        if img_type != "preview":
            i += 1
            continue

        # Found a preview, skip any additional consecutive previews
        preview_path = path
        preview_timestamp = timestamp
        i += 1

        while i < len(images) and images[i][3] == "preview":
            preview_path = images[i][0]
            preview_timestamp = images[i][1]
            i += 1

        # Look for left, middle, right in sequence
        if i >= len(images) or images[i][3] != "left":
            continue
        left_path = images[i][0]
        i += 1

        if i >= len(images) or images[i][3] != "extended":
            continue
        middle_path = images[i][0]
        i += 1

        if i >= len(images) or images[i][3] != "extended":
            continue
        right_path = images[i][0]
        i += 1

        groups.append(
            TriptychGroup(
                preview=preview_path,
                left=left_path,
                middle=middle_path,
                right=right_path,
                timestamp=preview_timestamp,
            )
        )

    return groups


def create_blend_mask(overlap_width: int, tile_height: int) -> np.ndarray:
    """
    Create a linear blend mask for the overlap region.

    Args:
        overlap_width: Width of the overlap region in pixels
        tile_height: Height of the tiles

    Returns:
        Blend mask array with values from 0.0 to 1.0
    """
    gradient = np.linspace(0.0, 1.0, overlap_width)
    mask = np.tile(gradient, (tile_height, 1))
    return mask[:, :, np.newaxis].astype(np.float32)


def combine_triptych(
    group: TriptychGroup,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
) -> np.ndarray:
    """
    Combine a triptych group into a single panoramic image with blended overlaps.

    Args:
        group: TriptychGroup containing paths to the images
        tile_width: Base tile width (left tile width)
        tile_height: Tile height

    Returns:
        Combined panoramic image as numpy array (RGB)
    """
    left_img = np.array(Image.open(group.left).convert("RGB"))
    middle_img = np.array(Image.open(group.middle).convert("RGB"))
    right_img = np.array(Image.open(group.right).convert("RGB"))

    extended_width = middle_img.shape[1]
    overlap = extended_width - tile_width

    panorama_width = tile_width * 3
    panorama = np.zeros((tile_height, panorama_width, 3), dtype=np.uint8)

    blend_mask = create_blend_mask(overlap, tile_height)

    # Place left tile
    panorama[:, :tile_width] = left_img[:, :tile_width]

    # Blend left-middle overlap
    left_overlap_start = tile_width - overlap
    left_overlap = left_img[:, left_overlap_start:tile_width]
    middle_overlap = middle_img[:, :overlap]

    blended = (
        left_overlap.astype(np.float32) * (1.0 - blend_mask)
        + middle_overlap.astype(np.float32) * blend_mask
    ).astype(np.uint8)
    panorama[:, left_overlap_start:tile_width] = blended

    # Place middle tile (non-overlap part)
    panorama[:, tile_width : 2 * tile_width] = middle_img[:, overlap:extended_width]

    # Blend middle-right overlap
    middle_overlap_start = 2 * tile_width - overlap
    middle_right_overlap = middle_img[:, extended_width - overlap : extended_width]
    right_overlap = right_img[:, :overlap]

    blended = (
        middle_right_overlap.astype(np.float32) * (1.0 - blend_mask)
        + right_overlap.astype(np.float32) * blend_mask
    ).astype(np.uint8)
    panorama[:, middle_overlap_start : 2 * tile_width] = blended

    # Place right tile (non-overlap part)
    panorama[:, 2 * tile_width : 3 * tile_width] = right_img[:, overlap:extended_width]

    return panorama


def save_combined_triptychs(
    groups: list[TriptychGroup],
    output_dir: Path,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    prefix: str = "triptych",
    img_format: str = "png",
    quality: int = 90,
) -> list[Path]:
    """
    Combine and save all triptych images.

    Output filenames are compatible with images_to_video.py:
        prefix_YYYYMMDD_HHMMSS_uuid.ext

    Args:
        groups: List of TriptychGroup objects
        output_dir: Directory to save combined images
        tile_width: Base tile width
        tile_height: Tile height
        prefix: Filename prefix
        img_format: Output format (png, jpg, webp)
        quality: Quality for jpg/webp (1-100)

    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map format to extension
    ext = "jpg" if img_format == "jpg" else img_format

    results = []
    for idx, group in enumerate(groups):
        print(f"Combining {idx + 1}/{len(groups)}: {group.left.stem}")

        try:
            combined = combine_triptych(group, tile_width, tile_height)

            # Create filename compatible with images_to_video.py
            timestamp_str = group.timestamp.strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            output_path = output_dir / f"{prefix}_{timestamp_str}_{unique_id}.{ext}"

            img = Image.fromarray(combined)

            if img_format == "png":
                img.save(output_path)
            elif img_format == "jpg":
                img.save(output_path, quality=quality, optimize=True)
            elif img_format == "webp":
                img.save(output_path, quality=quality)

            results.append(output_path)

        except Exception as e:
            print(f"Error combining {group.left.name}: {e}")
            continue

    return results


def list_groups(groups: list[TriptychGroup]) -> None:
    """Print list of triptych groups."""
    if not groups:
        print("No triptych groups found")
        return

    print(f"\nFound {len(groups)} triptych groups:\n")
    print(f"{'Index':<6} {'Timestamp':<20} {'Left Tile':<45}")
    print("-" * 75)

    for idx, group in enumerate(groups):
        print(
            f"{idx + 1:<6} {group.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{group.left.name:<45}"
        )


def parse_datetime(date_str: str) -> datetime:
    """Parse datetime string in various formats."""
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d_%H%M%S", "%Y%m%d"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse datetime: {date_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine triptych tile images into seamless panoramas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("image_dir", type=Path, help="Directory containing triptych images")

    parser.add_argument("-o", "--output", type=Path, help="Output directory for combined images")

    parser.add_argument(
        "--prefix",
        type=str,
        default="triptych",
        help="Filename prefix for output images (default: triptych)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)",
    )

    parser.add_argument(
        "--quality", type=int, default=90, help="Quality for jpg/webp (1-100, default: 90)"
    )

    parser.add_argument(
        "--tile-width",
        type=int,
        default=DEFAULT_TILE_WIDTH,
        help=f"Base tile width (default: {DEFAULT_TILE_WIDTH})",
    )

    parser.add_argument(
        "--tile-height",
        type=int,
        default=DEFAULT_TILE_HEIGHT,
        help=f"Tile height (default: {DEFAULT_TILE_HEIGHT})",
    )

    parser.add_argument(
        "--start-time", type=str, help="Filter: only include images after this time"
    )

    parser.add_argument("--end-time", type=str, help="Filter: only include images before this time")

    parser.add_argument(
        "--list-only", action="store_true", help="Just list triptych groups without combining"
    )

    args = parser.parse_args()

    if not args.image_dir.exists():
        print(f"Error: Image directory does not exist: {args.image_dir}")
        sys.exit(1)

    if not args.list_only and not args.output:
        print("Error: Output directory required (use -o/--output)")
        sys.exit(1)

    # Parse time filters
    start_time = None
    end_time = None

    if args.start_time:
        try:
            start_time = parse_datetime(args.start_time)
            print(f"Filtering images after: {start_time}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if args.end_time:
        try:
            end_time = parse_datetime(args.end_time)
            print(f"Filtering images before: {end_time}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Collect triptych groups
    print(f"Scanning for triptych images in: {args.image_dir}")
    groups = collect_image_groups(args.image_dir, start_time, end_time)

    if not groups:
        print("No valid triptych groups found!")
        sys.exit(1)

    print(f"Found {len(groups)} triptych groups")

    if args.list_only:
        list_groups(groups)
        return

    # Combine and save
    print(f"\nSaving combined triptychs to: {args.output}")
    results = save_combined_triptychs(
        groups,
        args.output,
        args.tile_width,
        args.tile_height,
        args.prefix,
        args.format,
        args.quality,
    )

    print(f"\nDone! Created {len(results)} combined images")
    print(f"\nTo create a video, run:")
    print(f"  uv run scripts/images_to_video.py {args.output} -o output.mp4")


if __name__ == "__main__":
    main()

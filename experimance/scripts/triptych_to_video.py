#!/usr/bin/env python3
"""
Combine triptych images with proper overlap blending and create a video.

The script processes images in groups of 4:
    - Low-res preview (2240x424) - used to identify groups but not in final output
    - Left tile (1344x760)
    - Middle tile (1480x760) - has overlap on left side
    - Right tile (1480x760) - has overlap on left side

The tiles are combined with linear blending in the overlap regions to create
seamless panoramic images, then assembled into a video with crossfade transitions.

Usage examples:
    # Basic usage
    uv run scripts/triptych_to_video.py /path/to/images -o output.mp4

    # Save combined triptychs as well
    uv run scripts/triptych_to_video.py /path/to/images -o output.mp4 --save-combined /path/to/combined

    # Custom video settings
    uv run scripts/triptych_to_video.py /path/to/images -o output.mp4 \\
        --frames-per-image 90 --blend-frames 30 --fps 30

    # Preview mode - just list groups without processing
    uv run scripts/triptych_to_video.py /path/to/images --list-only

Requirements:
    - ffmpegcv
    - numpy
    - Pillow (PIL)
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import ffmpegcv
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: uv add ffmpegcv numpy pillow")
    sys.exit(1)


# Default tile dimensions from core.toml
DEFAULT_TILE_WIDTH = 1344
DEFAULT_TILE_HEIGHT = 760
DEFAULT_MIN_OVERLAP_PERCENT = 10


def parse_filename_timestamp(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from filename in format: prefix_YYYYMMDD_HHMMSS.ext
    
    Args:
        filename: The image filename
        
    Returns:
        datetime object if parsed successfully, None otherwise
    """
    # Match pattern: anything_YYYYMMDD_HHMMSS.ext
    pattern = r'_(\d{8})_(\d{6})\.\w+$'
    match = re.search(pattern, filename)
    
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    
    return None


class TriptychGroup:
    """Represents a group of 4 images forming a triptych."""
    
    def __init__(
        self,
        preview: Path,
        left: Path,
        middle: Path,
        right: Path,
        timestamp: datetime
    ):
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
        return 'preview'
    
    # Tiles are taller (around 760 height)
    if 700 < height < 900:
        if 1300 < width < 1400:  # Base tile width ~1344
            return 'left'
        elif 1400 < width < 1600:  # Extended tile width ~1480
            return 'extended'
    
    return 'unknown'


def collect_image_groups(
    image_dir: Path,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> list[TriptychGroup]:
    """
    Collect images and group them into triptych sets.
    
    Images are expected to come in groups with the pattern:
    - preview (2240x424) - one or more preview images
    - left (1344x760) - exactly one left tile
    - middle (1480x760) - exactly one middle tile
    - right (1480x760) - exactly one right tile
    
    The grouping is based on image dimensions, not strict 4-image groups,
    to handle cases where there are duplicate previews or incomplete groups.
    
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
        if not img_path.is_file():
            continue
        
        # Skip symlinks
        if img_path.is_symlink():
            continue
            
        if img_path.suffix.lower() not in extensions:
            continue
            
        timestamp = parse_filename_timestamp(img_path.name)
        if timestamp is None:
            continue
            
        # Apply time filters
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
    
    # Sort by timestamp
    images.sort(key=lambda x: x[1])
    
    # Group images by looking for the pattern: preview(s), left, middle, right
    groups: list[TriptychGroup] = []
    
    i = 0
    while i < len(images):
        path, timestamp, size, img_type = images[i]
        
        # Look for a preview to start a group
        if img_type != 'preview':
            # Skip non-preview images that aren't part of a group
            i += 1
            continue
        
        # Found a preview, save it and skip any additional previews
        preview_path = path
        preview_timestamp = timestamp
        i += 1
        
        # Skip any additional consecutive preview images
        while i < len(images) and images[i][3] == 'preview':
            # Use the last preview in the sequence
            preview_path = images[i][0]
            preview_timestamp = images[i][1]
            i += 1
        
        # Now look for left, middle, right in sequence
        if i >= len(images) or images[i][3] != 'left':
            continue  # No left tile, skip this group
        
        left_path = images[i][0]
        i += 1
        
        if i >= len(images) or images[i][3] != 'extended':
            continue  # No middle tile, skip this group
        
        middle_path = images[i][0]
        i += 1
        
        if i >= len(images) or images[i][3] != 'extended':
            continue  # No right tile, skip this group
        
        right_path = images[i][0]
        i += 1
        
        # Found a complete group!
        group = TriptychGroup(
            preview=preview_path,
            left=left_path,
            middle=middle_path,
            right=right_path,
            timestamp=preview_timestamp
        )
        groups.append(group)
    
    return groups


def calculate_overlap(
    tile_width: int,
    extended_tile_width: int
) -> int:
    """
    Calculate the overlap in pixels between tiles.
    
    Args:
        tile_width: Width of the base tile (left tile)
        extended_tile_width: Width of the extended tiles (middle/right)
        
    Returns:
        Overlap in pixels
    """
    return extended_tile_width - tile_width


def create_blend_mask(overlap_width: int, tile_height: int) -> np.ndarray:
    """
    Create a linear blend mask for the overlap region.
    
    Args:
        overlap_width: Width of the overlap region in pixels
        tile_height: Height of the tiles
        
    Returns:
        Blend mask array with values from 0.0 to 1.0
    """
    # Create linear gradient from 0 to 1 across the overlap width
    gradient = np.linspace(0.0, 1.0, overlap_width)
    
    # Expand to full height
    mask = np.tile(gradient, (tile_height, 1))
    
    # Add channel dimension for broadcasting
    mask = mask[:, :, np.newaxis]
    
    return mask.astype(np.float32)


def combine_triptych(
    group: TriptychGroup,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT
) -> np.ndarray:
    """
    Combine a triptych group into a single panoramic image with blended overlaps.
    
    Args:
        group: TriptychGroup containing paths to the 4 images
        tile_width: Base tile width (left tile width)
        tile_height: Tile height
        
    Returns:
        Combined panoramic image as numpy array (RGB)
    """
    # Load the three tiles
    left_img = np.array(Image.open(group.left).convert('RGB'))
    middle_img = np.array(Image.open(group.middle).convert('RGB'))
    right_img = np.array(Image.open(group.right).convert('RGB'))
    
    # Calculate overlap
    extended_width = middle_img.shape[1]
    overlap = calculate_overlap(tile_width, extended_width)
    
    # Calculate final panorama dimensions
    # Each tile contributes tile_width to the final image (minus overlaps that are blended)
    panorama_width = tile_width * 3
    panorama_height = tile_height
    
    # Create output array
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    
    # Create blend mask for overlap regions
    blend_mask = create_blend_mask(overlap, tile_height)
    
    # Place left tile (no overlap on left side of panorama)
    panorama[:, :tile_width] = left_img[:, :tile_width]
    
    # Blend left-middle overlap
    # The overlap region in the final image is at: tile_width - overlap to tile_width
    # Left tile contributes: tile_width - overlap to tile_width (the rightmost 'overlap' pixels)
    # Middle tile contributes: 0 to overlap (the leftmost 'overlap' pixels, which is the extended part)
    
    left_overlap_start = tile_width - overlap
    left_overlap_region = left_img[:, left_overlap_start:tile_width]
    middle_overlap_region = middle_img[:, :overlap]
    
    # Linear blend: left fades out (1->0), middle fades in (0->1)
    blended_left_middle = (
        left_overlap_region.astype(np.float32) * (1.0 - blend_mask) +
        middle_overlap_region.astype(np.float32) * blend_mask
    ).astype(np.uint8)
    
    # Place blended region
    panorama[:, left_overlap_start:tile_width] = blended_left_middle
    
    # Place middle tile (non-overlap part)
    # Middle tile after overlap region: overlap to extended_width
    # This goes to: tile_width to tile_width + (extended_width - overlap) = tile_width to tile_width + tile_width = 2*tile_width
    middle_non_overlap = middle_img[:, overlap:extended_width]
    panorama[:, tile_width:2*tile_width] = middle_non_overlap
    
    # Blend middle-right overlap
    middle_overlap_start = 2*tile_width - overlap
    # Middle tile contributes: extended_width - overlap to extended_width
    middle_right_overlap = middle_img[:, extended_width - overlap:extended_width]
    # Right tile contributes: 0 to overlap
    right_overlap_region = right_img[:, :overlap]
    
    blended_middle_right = (
        middle_right_overlap.astype(np.float32) * (1.0 - blend_mask) +
        right_overlap_region.astype(np.float32) * blend_mask
    ).astype(np.uint8)
    
    panorama[:, middle_overlap_start:2*tile_width] = blended_middle_right
    
    # Place right tile (non-overlap part)
    right_non_overlap = right_img[:, overlap:extended_width]
    panorama[:, 2*tile_width:3*tile_width] = right_non_overlap
    
    return panorama


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two images using alpha blending.
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blend factor (0.0 = all img1, 1.0 = all img2)
        
    Returns:
        Blended image
    """
    blended = (img1.astype(np.float32) * (1.0 - alpha) + 
               img2.astype(np.float32) * alpha).astype(np.uint8)
    return blended


def save_combined_triptychs(
    groups: list[TriptychGroup],
    output_dir: Path,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT
) -> list[tuple[Path, np.ndarray]]:
    """
    Save all combined triptych images to a directory.
    
    Args:
        groups: List of TriptychGroup objects
        output_dir: Directory to save combined images
        tile_width: Base tile width
        tile_height: Tile height
        
    Returns:
        List of (path, image_array) tuples
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for idx, group in enumerate(groups):
        print(f"Combining triptych {idx + 1}/{len(groups)}: {group.left.stem}")
        
        try:
            combined = combine_triptych(group, tile_width, tile_height)
            
            # Create output filename based on original timestamp
            timestamp_str = group.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"triptych_{timestamp_str}.png"
            
            # Save the combined image
            Image.fromarray(combined).save(output_path)
            
            results.append((output_path, combined))
            
        except Exception as e:
            print(f"Error combining triptych {group.left.name}: {e}")
            continue
    
    return results


def create_video(
    groups: list[TriptychGroup],
    output_path: Path,
    frames_per_image: int = 90,
    blend_frames: int = 30,
    fps: int = 30,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    codec: str = 'h264',
    quality: int = 18
) -> None:
    """
    Create video from triptych groups with crossfade transitions.
    
    Args:
        groups: List of TriptychGroup objects
        output_path: Output video file path
        frames_per_image: Number of frames to hold each image (at 30fps, 90 = 3 seconds)
        blend_frames: Number of frames for crossfade transition (at 30fps, 30 = 1 second)
        fps: Frames per second for output video
        tile_width: Base tile width
        tile_height: Tile height
        codec: Video codec to use
        quality: CRF quality (0-51, lower is better)
    """
    if not groups:
        print("Error: No triptych groups to process")
        return
    
    print(f"Processing {len(groups)} triptych groups...")
    
    # Calculate output dimensions
    panorama_width = tile_width * 3
    panorama_height = tile_height
    
    print(f"Output resolution: {panorama_width}x{panorama_height}")
    print(f"Output FPS: {fps}")
    print(f"Frames per image: {frames_per_image} ({frames_per_image/fps:.1f}s)")
    print(f"Blend frames: {blend_frames} ({blend_frames/fps:.1f}s)")
    print(f"Using codec: {codec}, CRF: {quality}")
    
    # Initialize video writer
    try:
        # For h264 output with BGR input, use bgr24 pix_fmt
        # ffmpegcv will convert to yuv420p for the output automatically
        out = ffmpegcv.VideoWriter(
            str(output_path),
            codec=codec,
            fps=fps,
            pix_fmt='bgr24'
        )
    except Exception as e:
        print(f"Error: Could not create video writer: {e}")
        return
    
    try:
        prev_img = None
        frame_count = 0
        
        for idx, group in enumerate(groups):
            print(f"Processing {idx + 1}/{len(groups)}: {group.left.stem}")
            
            try:
                # Combine the triptych
                curr_img = combine_triptych(group, tile_width, tile_height)
                
                # Convert RGB to BGR for ffmpegcv
                curr_img_bgr = curr_img[:, :, ::-1]
                
            except Exception as e:
                print(f"Error combining triptych {group.left.name}: {e}")
                continue
            
            # Crossfade from previous image
            if prev_img is not None and blend_frames > 0:
                for blend_idx in range(blend_frames):
                    alpha = (blend_idx + 1) / (blend_frames + 1)
                    blended = blend_images(prev_img, curr_img_bgr, alpha)
                    out.write(blended)
                    frame_count += 1
            
            # Write frames for current image
            for _ in range(frames_per_image):
                out.write(curr_img_bgr)
                frame_count += 1
            
            prev_img = curr_img_bgr
        
        print(f"\nVideo created successfully: {output_path}")
        print(f"Total frames written: {frame_count}")
        
        # Calculate video duration
        duration = frame_count / fps
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
    finally:
        out.release()


def list_groups(groups: list[TriptychGroup]) -> None:
    """Print list of triptych groups."""
    if not groups:
        print("No triptych groups found")
        return
    
    print(f"\nFound {len(groups)} triptych groups:\n")
    print(f"{'Index':<6} {'Timestamp':<20} {'Preview':<40} {'Left':<40}")
    print("-" * 110)
    
    for idx, group in enumerate(groups):
        print(f"{idx + 1:<6} {group.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} "
              f"{group.preview.name:<40} {group.left.name:<40}")


def parse_datetime(date_str: str) -> datetime:
    """Parse datetime string in various formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y%m%d_%H%M%S",
        "%Y%m%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse datetime: {date_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine triptych images and create video with crossfade transitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing triptych images"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("triptych_video.mp4"),
        help="Output video file (default: triptych_video.mp4)"
    )
    
    parser.add_argument(
        "--save-combined",
        type=Path,
        metavar="DIR",
        help="Save combined triptych images to this directory"
    )
    
    parser.add_argument(
        "--frames-per-image",
        type=int,
        default=90,
        help="Frames to hold each image (default: 90, 3s at 30fps)"
    )
    
    parser.add_argument(
        "--blend-frames",
        type=int,
        default=30,
        help="Frames for crossfade transition (default: 30, 1s at 30fps)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second (default: 30)"
    )
    
    parser.add_argument(
        "--tile-width",
        type=int,
        default=DEFAULT_TILE_WIDTH,
        help=f"Base tile width (default: {DEFAULT_TILE_WIDTH})"
    )
    
    parser.add_argument(
        "--tile-height",
        type=int,
        default=DEFAULT_TILE_HEIGHT,
        help=f"Tile height (default: {DEFAULT_TILE_HEIGHT})"
    )
    
    parser.add_argument(
        "--codec",
        type=str,
        default="h264",
        help="Video codec (default: h264)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=18,
        help="CRF quality 0-51, lower is better (default: 18)"
    )
    
    parser.add_argument(
        "--start-time",
        type=str,
        help="Filter: only include images after this time (YYYY-MM-DD HH:MM:SS)"
    )
    
    parser.add_argument(
        "--end-time",
        type=str,
        help="Filter: only include images before this time (YYYY-MM-DD HH:MM:SS)"
    )
    
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Just list triptych groups without creating video"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.image_dir.exists():
        print(f"Error: Image directory does not exist: {args.image_dir}")
        sys.exit(1)
    
    # Parse time filters
    start_time = None
    end_time = None
    
    if args.start_time:
        try:
            start_time = parse_datetime(args.start_time)
            print(f"Filtering images after: {start_time}")
        except ValueError as e:
            print(f"Error: Invalid start time: {e}")
            sys.exit(1)
    
    if args.end_time:
        try:
            end_time = parse_datetime(args.end_time)
            print(f"Filtering images before: {end_time}")
        except ValueError as e:
            print(f"Error: Invalid end time: {e}")
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
    
    # Save combined images if requested
    if args.save_combined:
        print(f"\nSaving combined triptychs to: {args.save_combined}")
        save_combined_triptychs(
            groups,
            args.save_combined,
            args.tile_width,
            args.tile_height
        )
    
    # Create video
    print(f"\nCreating video: {args.output}")
    create_video(
        groups,
        args.output,
        frames_per_image=args.frames_per_image,
        blend_frames=args.blend_frames,
        fps=args.fps,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        codec=args.codec,
        quality=args.quality
    )


if __name__ == "__main__":
    main()

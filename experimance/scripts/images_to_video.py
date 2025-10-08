#!/usr/bin/env python3
"""
Convert a sequence of timestamped images into a video with crossfade transitions.

The script processes images with filenames in the format:
    prefix_YYYYMMDD_HHMMSS_uuid.jpg

Images are sorted by their timestamp and can be filtered by time range.
Each image is held for a specified number of frames, with optional crossfade
transitions between images.

Usage examples:
    # Basic usage - all images with defaults
    uv run scripts/images_to_video.py ../../fmc/images -o output.mp4

    # Specify time range
    uv run scripts/images_to_video.py ../../fmc/images -o output.mp4 \\
        --start-time "2025-07-27 20:00:00" \\
        --end-time "2025-07-27 22:00:00"

    # Custom frame settings
    uv run scripts/images_to_video.py ../../fmc/images -o output.mp4 \\
        --frames-per-image 5 \\
        --blend-frames 2 \\
        --fps 30

    # List images without creating video
    uv run scripts/images_to_video.py ../../fmc/images --list-only

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
from typing import List, Optional, Tuple

try:
    import ffmpegcv
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: uv add ffmpegcv numpy pillow")
    sys.exit(1)


def parse_filename_timestamp(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from filename in format: prefix_YYYYMMDD_HHMMSS_uuid.ext
    
    Args:
        filename: The image filename
        
    Returns:
        datetime object if parsed successfully, None otherwise
    """
    # Match pattern: anything_YYYYMMDD_HHMMSS_uuid.ext
    pattern = r'_(\d{8})_(\d{6})_[a-f0-9\-]+\.\w+$'
    match = re.search(pattern, filename)
    
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    
    return None


def collect_images(
    image_dir: Path,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> List[Tuple[Path, datetime]]:
    """
    Collect and sort images by timestamp, recursively searching subdirectories.
    
    Args:
        image_dir: Directory containing images (searches recursively)
        start_time: Optional start time filter
        end_time: Optional end time filter
        extensions: Tuple of valid image extensions
        
    Returns:
        List of (Path, datetime) tuples sorted by timestamp
    """
    images = []
    
    # Recursively search all subdirectories
    for img_path in image_dir.rglob('*'):
        if not img_path.is_file():
            continue
            
        if img_path.suffix.lower() not in extensions:
            continue
            
        timestamp = parse_filename_timestamp(img_path.name)
        if timestamp is None:
            # Silently skip files that don't match the pattern
            continue
            
        # Apply time filters
        if start_time and timestamp < start_time:
            continue
        if end_time and timestamp > end_time:
            continue
            
        images.append((img_path, timestamp))
    
    # Sort by timestamp
    images.sort(key=lambda x: x[1])
    
    return images


def load_and_resize_image(img_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Load image and resize to target dimensions.
    
    Args:
        img_path: Path to image file
        target_size: (width, height) tuple
        
    Returns:
        NumPy array in BGR format (for ffmpegcv compatibility)
    """
    # Load with PIL for better compatibility
    img = Image.open(img_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array (RGB)
    img_array = np.array(img)
    
    # Convert RGB to BGR for ffmpegcv
    img_bgr = img_array[:, :, ::-1]
    
    return img_bgr


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
    # Simple weighted blend using numpy
    blended = (img1 * (1.0 - alpha) + img2 * alpha).astype(np.uint8)
    return blended


def create_video(
    images: List[Tuple[Path, datetime]],
    output_path: Path,
    frames_per_image: int = 2,
    blend_frames: int = 1,
    fps: int = 30,
    resolution: Optional[Tuple[int, int]] = None,
    codec: str = 'h264',
    quality: Optional[int] = None
) -> None:
    """
    Create video from image sequence with crossfade transitions.
    
    Args:
        images: List of (Path, datetime) tuples
        output_path: Output video file path
        frames_per_image: Number of frames to hold each image
        blend_frames: Number of frames for crossfade transition
        fps: Frames per second for output video
        resolution: Optional (width, height) tuple, otherwise uses first image size
        codec: Video codec to use (default: 'h264')
        quality: Optional quality setting (0-51 for h264, lower is better)
    """
    if not images:
        print("Error: No images to process")
        return
    
    print(f"Processing {len(images)} images...")
    
    # Determine resolution from first image if not specified
    if resolution is None:
        first_img = Image.open(images[0][0])
        resolution = first_img.size  # PIL returns (width, height)
        first_img.close()
    
    width, height = resolution
    print(f"Output resolution: {width}x{height}")
    print(f"Output FPS: {fps}")
    print(f"Frames per image: {frames_per_image}")
    print(f"Blend frames: {blend_frames}")
    print(f"Using codec: {codec}")
    
    # Initialize video writer with ffmpegcv
    try:
        # Set up video writer options
        vcodec_params = {}
        if quality is not None and codec == 'h264':
            vcodec_params['crf'] = quality  # 0-51, lower is better quality
        
        out = ffmpegcv.VideoWriter(
            str(output_path),
            codec=codec,
            fps=fps,
            **vcodec_params
        )
        
    except Exception as e:
        print(f"Error: Could not create video writer: {e}")
        return
    
    try:
        prev_img = None
        frame_count = 0
        skipped_duplicates = 0
        
        for idx, (img_path, timestamp) in enumerate(images):
            print(f"Processing {idx + 1}/{len(images)}: {img_path.name}")
            
            # Load and resize current image
            try:
                curr_img = load_and_resize_image(img_path, (width, height))
                
                # Validate image dimensions
                if curr_img.shape[0] != height or curr_img.shape[1] != width:
                    print(f"Warning: Image size mismatch for {img_path.name}, skipping...")
                    continue
                    
            except Exception as e:
                print(f"Error loading {img_path.name}: {e}, skipping...")
                continue
            
            # Check if this image is identical to the previous one
            if prev_img is not None:
                # Compare images using numpy array comparison
                if np.array_equal(curr_img, prev_img):
                    print(f"  -> Duplicate of previous image, skipping...")
                    skipped_duplicates += 1
                    continue
            
            if prev_img is not None and blend_frames > 0:
                # Create crossfade transition
                for blend_idx in range(blend_frames):
                    alpha = (blend_idx + 1) / (blend_frames + 1)
                    blended = blend_images(prev_img, curr_img, alpha)
                    out.write(blended)
                    frame_count += 1
            
            # Write frames for current image
            for _ in range(frames_per_image):
                out.write(curr_img)
                frame_count += 1
            
            prev_img = curr_img
        
        print(f"\nVideo created successfully: {output_path}")
        print(f"Total frames written: {frame_count}")
        if skipped_duplicates > 0:
            print(f"Skipped {skipped_duplicates} duplicate images")
        
        # Calculate video duration
        duration = frame_count / fps
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
    finally:
        out.release()


def list_images(images: List[Tuple[Path, datetime]]) -> None:
    """Print list of images with timestamps."""
    if not images:
        print("No images found matching criteria")
        return
    
    print(f"\nFound {len(images)} images:\n")
    print(f"{'Index':<6} {'Timestamp':<20} {'Filename'}")
    print("-" * 80)
    
    for idx, (img_path, timestamp) in enumerate(images):
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{idx:<6} {timestamp_str:<20} {img_path.name}")
    
    if images:
        print(f"\nTime range: {images[0][1]} to {images[-1][1]}")


def parse_datetime(date_string: str) -> datetime:
    """Parse datetime string in various formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y%m%d_%H%M%S",
        "%Y%m%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse datetime: {date_string}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert timestamped images to video with crossfade transitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ../../fmc/images -o output.mp4
  %(prog)s ../../fmc/images -o output.mp4 --start-time "2025-07-27 20:00:00"
  %(prog)s ../../fmc/images --list-only
        """
    )
    
    parser.add_argument(
        'image_dir',
        type=Path,
        help='Directory containing timestamped images'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output video file path (default: output.mp4)',
        default=Path('output.mp4')
    )
    
    parser.add_argument(
        '--start-time',
        type=str,
        help='Start time filter (e.g., "2025-07-27 20:00:00")'
    )
    
    parser.add_argument(
        '--end-time',
        type=str,
        help='End time filter (e.g., "2025-07-27 22:00:00")'
    )
    
    parser.add_argument(
        '--frames-per-image',
        type=int,
        default=2,
        help='Number of frames to hold each image (default: 2)'
    )
    
    parser.add_argument(
        '--blend-frames',
        type=int,
        default=1,
        help='Number of frames for crossfade transition (default: 1)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output video (default: 30)'
    )
    
    parser.add_argument(
        '--resolution',
        type=str,
        help='Output resolution as WIDTHxHEIGHT (e.g., 1920x1080), default: use first image size'
    )
    
    parser.add_argument(
        '--codec',
        type=str,
        default='h264',
        help='Video codec to use (default: h264). Options: h264, h265, vp9, mpeg4'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        help='Video quality for h264 codec (0-51, lower is better, default: 23)'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List images without creating video'
    )
    
    args = parser.parse_args()
    
    # Validate image directory
    if not args.image_dir.exists():
        print(f"Error: Image directory does not exist: {args.image_dir}")
        sys.exit(1)
    
    if not args.image_dir.is_dir():
        print(f"Error: Not a directory: {args.image_dir}")
        sys.exit(1)
    
    # Parse time filters
    start_time = None
    end_time = None
    
    if args.start_time:
        try:
            start_time = parse_datetime(args.start_time)
        except ValueError as e:
            print(f"Error parsing start time: {e}")
            sys.exit(1)
    
    if args.end_time:
        try:
            end_time = parse_datetime(args.end_time)
        except ValueError as e:
            print(f"Error parsing end time: {e}")
            sys.exit(1)
    
    # Parse resolution
    resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            print(f"Error: Invalid resolution format: {args.resolution}")
            print("Use format: WIDTHxHEIGHT (e.g., 1920x1080)")
            sys.exit(1)
    
    # Collect images
    print(f"Scanning directory: {args.image_dir}")
    if start_time:
        print(f"Start time filter: {start_time}")
    if end_time:
        print(f"End time filter: {end_time}")
    
    images = collect_images(args.image_dir, start_time, end_time)
    
    if args.list_only:
        list_images(images)
    else:
        if not images:
            print("No images found. Use --list-only to see what's available.")
            sys.exit(1)
        
        create_video(
            images,
            args.output,
            frames_per_image=args.frames_per_image,
            blend_frames=args.blend_frames,
            fps=args.fps,
            resolution=resolution,
            codec=args.codec,
            quality=args.quality
        )


if __name__ == '__main__':
    main()

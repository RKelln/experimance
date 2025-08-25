#!/usr/bin/env python3
"""
Tiling system for panorama image generation.

This module handles the intelligent tiling of panorama images into
overlapping segments for high-resolution generation while maintaining
seamless blending between tiles.
"""

import math
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageFilter
import numpy as np

logger = logging.getLogger(__name__)


def round_to_multiple_of_8(value: int) -> int:
    """
    Round a value to the nearest multiple of 8.
    
    Many image generation models require dimensions divisible by 8.
    
    Args:
        value: The value to round
        
    Returns:
        The nearest multiple of 8
    """
    return ((value + 4) // 8) * 8


@dataclass
class TileSpec:
    """Specification for a single tile."""
    
    # Display positioning (final layout coordinates)
    display_x: int  # X position in display panorama
    display_y: int  # Y position in display panorama
    display_width: int  # Width in display panorama (without overlap)
    display_height: int  # Height in display panorama
    overlap: int # Overlap in pixels (0 if no overlap)

    # Generation parameters (for render requests - includes overlap)
    generated_width: int  # Width to generate (includes overlap extension)
    generated_height: int  # Height to generate 
    
    # Tile metadata
    tile_index: int  # Tile number (0-based)
    total_tiles: int  # Total number of tiles


class PanoramaTiler:
    """
    Manages tiling strategy for panorama images.
    
    Given a base image and constraints, calculates optimal tiling to:
    - Minimize number of tiles
    - Keep each tile under max megapixel limit
    - Maintain minimum overlap for seamless blending
    - Generate proper positioning for display service
    """
    
    def __init__(
        self,
        display_tile_width: int = 1920,
        display_tile_height: int = 1080,
        generated_tile_width: int = 1344,
        generated_tile_height: int = 768,
        min_overlap_percent: float = 5.0,
        max_megapixels: float = 1.0
    ):
        """
        Initialize tiler with configuration.
        
        Args:
            display_tile_width: Base tile width for display positioning
            display_tile_height: Base tile height for display positioning
            generated_tile_width: Base tile width for generation (before overlap extension)
            generated_tile_height: Base tile height for generation
            min_overlap_percent: Minimum overlap between tiles (5-50%)
            max_megapixels: Maximum megapixels per tile (0.5-5.0)
        """
        self.display_tile_width = display_tile_width
        self.display_tile_height = display_tile_height
        self.generated_tile_width = generated_tile_width
        self.generated_tile_height = generated_tile_height
        self.min_overlap_percent = min_overlap_percent
        self.max_megapixels = max_megapixels
        
        # Calculate effective max pixels
        self.max_pixels = int(max_megapixels * 1_000_000)
        
        # Verify aspect ratios are consistent
        self._validate_aspect_ratios()
        
        logger.info(
            f"Tiler initialized: display={display_tile_width}x{display_tile_height}, "
            f"generated={generated_tile_width}x{generated_tile_height}, "
            f"min_overlap={min_overlap_percent}%, max_mp={max_megapixels}"
        )
    
    def _validate_aspect_ratios(self):
        """Validate that display and generated aspect ratios are consistent."""
        display_aspect = self.display_tile_width / self.display_tile_height
        generated_aspect = self.generated_tile_width / self.generated_tile_height
        
        aspect_diff = abs(display_aspect - generated_aspect) / display_aspect
        
        if aspect_diff > 0.05:  # Allow 5% tolerance
            logger.warning(
                f"Aspect ratio mismatch: display={display_aspect:.3f}, "
                f"generated={generated_aspect:.3f}, diff={aspect_diff*100:.1f}%"
            )
        else:
            logger.debug(
                f"Aspect ratios consistent: display={display_aspect:.3f}, "
                f"generated={generated_aspect:.3f}"
            )
    
    def calculate_tiles(
        self, 
        panorama_display_width: int, 
        panorama_display_height: int
    ) -> List[TileSpec]:
        """
        Calculate optimal tiling strategy for a panorama.
        
        Args:
            panorama_display_width: Display width of the panorama to tile
            panorama_display_height: Display height of the panorama to tile
            
        Returns:
            List of TileSpec objects describing each tile
        """
        logger.info(f"Calculating tiles for {panorama_display_width}x{panorama_display_height} display panorama")
        
        # Check if we can fit the entire panorama in a single tile
        if (panorama_display_width <= self.display_tile_width and 
            panorama_display_height <= self.display_tile_height):
            
            logger.info("Panorama fits in single tile")
            return [TileSpec(
                display_x=0, 
                display_y=0,
                display_width=panorama_display_width,
                display_height=panorama_display_height,
                overlap=0,
                generated_width=round_to_multiple_of_8(self.generated_tile_width),
                generated_height=round_to_multiple_of_8(self.generated_tile_height),
                tile_index=0,
                total_tiles=1,
            )]
        
        # Calculate how many tiles we need horizontally
        num_tiles = math.ceil(panorama_display_width / self.display_tile_width)
        
        logger.info(f"Tiler input: panorama={panorama_display_width}px, tile_width={self.display_tile_width}px → {num_tiles} tiles needed")
        
        # Calculate overlap needed in display space
        if num_tiles == 1:
            overlap_display = 0
        else:
            # Calculate minimum overlap from fitting constraint
            # total_width = num_tiles * tile_width - (num_tiles-1) * overlap
            # So: overlap = (num_tiles * tile_width - total_width) / (num_tiles - 1)
            fitting_overlap = (num_tiles * self.display_tile_width - panorama_display_width) / (num_tiles - 1)
            
            # Calculate minimum overlap from percentage constraint
            min_overlap_required = (self.min_overlap_percent / 100.0) * self.display_tile_width
            
            # Use the larger of the two requirements
            overlap_display = max(fitting_overlap, min_overlap_required)
            overlap_display = int(overlap_display)
            
            logger.info(
                f"Overlap calculation: fitting={fitting_overlap:.1f}px, "
                f"min_required={min_overlap_required:.1f}px, using={overlap_display}px"
            )
        
        # Calculate overlap needed in generated space (scale proportionally)
        display_to_generated_ratio = self.generated_tile_width / self.display_tile_width
        overlap_generated = int(overlap_display * display_to_generated_ratio)
        
        logger.info(
            f"Tiling strategy: {num_tiles} tiles, "
            f"display_overlap={overlap_display}px, generated_overlap={overlap_generated}px"
        )
        
        # Generate tile specifications
        tiles = []
        
        for i in range(num_tiles):
            # Calculate display position and dimensions
            if i == 0:
                # First tile: no overlap, starts at 0
                display_x = 0
                display_width = self.display_tile_width
            else:
                # Subsequent tiles: positioned at tile_width intervals, but shifted left by overlap
                display_x = (i * self.display_tile_width) - overlap_display
                display_width = self.display_tile_width + overlap_display  # Base width + left overlap
            
            logger.info(f"Tile {i}: raw_position={i * self.display_tile_width}, "
                       f"overlap_shift={overlap_display}, final_x={display_x}, width={display_width}")
            
            # Clamp to panorama bounds for the last tile
            if display_x + display_width > panorama_display_width:
                old_width = display_width
                display_width = panorama_display_width - display_x
                logger.info(f"Tile {i}: clamped width from {old_width} to {display_width} "
                           f"(x={display_x} + width={display_width} = {display_x + display_width} ≤ {panorama_display_width})")
            
            # Calculate generated dimensions (includes overlap extension)
            # With left-edge-only fading, only add left overlap for non-first tiles
            generated_width = self.generated_tile_width
            generated_height = self.generated_tile_height
            
            if i > 0:  # All tiles except first get left overlap
                # Add overlap while maintaining aspect ratio
                generated_width += overlap_generated
                # Adjust height proportionally to maintain aspect ratio
                aspect_ratio = self.generated_tile_width / self.generated_tile_height
                generated_height = int(generated_width / aspect_ratio)
            
            # Ensure dimensions are divisible by 8 (required by most image generation models)
            generated_width = round_to_multiple_of_8(generated_width)
            generated_height = round_to_multiple_of_8(generated_height)
            
            # Check megapixel constraint
            if generated_width * generated_height > self.max_pixels:
                logger.warning(
                    f"Tile {i} exceeds megapixel limit: "
                    f"{generated_width}x{generated_height} = "
                    f"{generated_width * generated_height / 1_000_000:.2f}MP"
                )
            
            tiles.append(TileSpec(
                display_x=display_x,
                display_y=0,  # Panoramas are typically single row
                display_width=display_width,
                display_height=panorama_display_height,
                overlap=overlap_display if i > 0 else 0,  # No overlap for first tile
                generated_width=generated_width,
                generated_height=generated_height,
                tile_index=i,
                total_tiles=num_tiles,
            ))
        
        return tiles
    
    def get_overlap_pixels(self, panorama_display_width: int) -> tuple[int, int]:
        """
        Calculate overlap pixels for display and generated resolutions.
        
        Args:
            panorama_display_width: Display width of the panorama
            
        Returns:
            Tuple of (display_overlap_pixels, generated_overlap_pixels)
        """
        num_tiles = math.ceil(panorama_display_width / self.display_tile_width)
        
        if num_tiles <= 1:
            return 0, 0
        
        # Calculate overlap in display space
        overlap_display = (num_tiles * self.display_tile_width - panorama_display_width) / (num_tiles - 1)
        overlap_display = int(overlap_display)
        
        # Calculate overlap in generated space
        display_to_generated_ratio = self.generated_tile_width / self.display_tile_width
        overlap_generated = int(overlap_display * display_to_generated_ratio)
        
        return overlap_display, overlap_generated
    
    def crop_tile_from_image(self, image: Image.Image, tile_spec: TileSpec) -> Image.Image:
        """
        Crop a tile from the base image using display coordinates.
        
        Args:
            image: Source panorama image (should be at display resolution)
            tile_spec: Specification for the tile to crop
            
        Returns:
            Cropped tile image
        """
        return image.crop((
            tile_spec.display_x,
            tile_spec.display_y,
            tile_spec.display_x + tile_spec.display_width,
            tile_spec.display_y + tile_spec.display_height
        ))
    
    def prepare_tile_reference_image(
        self, 
        base_image: Image.Image, 
        tile_spec: TileSpec,
        display_width: int,
        display_height: int,
        output_dir: str = "/tmp"
    ) -> str:
        """
        Create a reference image for tile generation by cropping and scaling the base image.
        
        Args:
            base_image: Base panorama image (PIL Image object)
            tile_spec: Tile specification with positioning info
            display_width: Total display panorama width
            display_height: Total display panorama height
            output_dir: Directory to save the reference image
            
        Returns:
            Path to the created reference image file
        """
        import os
        
        # Calculate scaling factors from display coordinates to base image coordinates
        base_width, base_height = base_image.size
        
        x_scale = base_width / display_width
        y_scale = base_height / display_height
        
        logger.debug(f"Base image: {base_width}x{base_height}, Display: {display_width}x{display_height}")
        logger.debug(f"Scale factors: x={x_scale:.3f}, y={y_scale:.3f}")
        
        # Calculate crop coordinates in display space first
        crop_x = tile_spec.display_x
        crop_width = tile_spec.display_width
        
        # For tiles with overlap, extend the crop to include overlap region
        if tile_spec.overlap > 0:
            crop_x = max(0, crop_x - tile_spec.overlap)
            crop_width += tile_spec.overlap
        
        # Scale coordinates to base image space
        base_crop_x = int(crop_x * x_scale)
        base_crop_y = int(tile_spec.display_y * y_scale)
        base_crop_width = int(crop_width * x_scale)
        base_crop_height = int(tile_spec.display_height * y_scale)
        
        # Clamp coordinates to image boundaries
        base_crop_x = max(0, base_crop_x)
        base_crop_y = max(0, base_crop_y)
        base_crop_width = min(base_crop_width, base_width - base_crop_x)
        base_crop_height = min(base_crop_height, base_height - base_crop_y)
        
        # Ensure we have positive dimensions
        if base_crop_width <= 0 or base_crop_height <= 0:
            logger.warning(f"Invalid crop dimensions for tile {tile_spec.tile_index}: {base_crop_width}x{base_crop_height}")
            # Fall back to a proportional section of the image
            base_crop_x = int((tile_spec.tile_index / tile_spec.total_tiles) * base_width * 0.8)  # 80% to avoid edge
            base_crop_y = 0
            base_crop_width = max(1, int(base_width / tile_spec.total_tiles))
            base_crop_height = base_height
        
        logger.debug(f"Tile {tile_spec.tile_index} crop: ({base_crop_x}, {base_crop_y}) {base_crop_width}x{base_crop_height}")
        
        # Crop the relevant section from the base image
        cropped = base_image.crop((
            base_crop_x,
            base_crop_y,
            base_crop_x + base_crop_width,
            base_crop_y + base_crop_height
        ))
        
        # Scale to match generated dimensions for better reference
        reference = cropped.resize(
            (tile_spec.generated_width, tile_spec.generated_height),
            Image.Resampling.LANCZOS
        )
        
        # Save reference image
        ref_filename = f"tile_{tile_spec.tile_index}_ref.jpg"
        ref_path = os.path.join(output_dir, ref_filename)
        reference.save(ref_path, "JPEG", quality=85)
        
        logger.debug(f"Created reference image for tile {tile_spec.tile_index}: {ref_path} ({reference.size})")
        return ref_path
    
    def apply_edge_blending(self, tile_image: Image.Image | str, tile_spec: TileSpec) -> Image.Image:
        """
        Apply edge blending to a tile for seamless composition.
        
        Creates a transparency mask that fades out the left edge for tiles
        that have left fade enabled (all tiles except the first).
        The blend covers the entire overlap width.
        
        Args:
            tile_image: Tile image to blend
            tile_spec: Tile specification
            overlap_pixels: Width of overlap region to blend (in pixels)
            
        Returns:
            Tile image with alpha channel for blending
        """
        if isinstance(tile_image, str):
            # Load image from file path
            tile_image = Image.open(tile_image).convert('RGBA')
        else:
            # Convert to RGBA if needed
            if tile_image.mode != 'RGBA':
                tile_image = tile_image.convert('RGBA')
        
        # Only apply left edge fading for tiles that need it
        if tile_spec.overlap <= 0:
            return tile_image

        # Create alpha mask
        width, height = tile_image.size
        blend_px = min(tile_spec.overlap, width // 2)  # Don't blend more than half the image
        
        # Use NumPy for efficient alpha mask creation
        alpha_array = np.full((height, width), 255, dtype=np.uint8)
        
        # Create linear gradient for left edge fade
        if blend_px > 0:
            gradient = np.linspace(0, 255, blend_px, dtype=np.uint8)
            alpha_array[:, :blend_px] = gradient[np.newaxis, :]
        
        # Convert back to PIL Image
        alpha = Image.fromarray(alpha_array, mode='L')
        
        # Apply alpha mask
        tile_image.putalpha(alpha)

        logger.debug(
            f"Applied left edge fade to tile {tile_spec.tile_index} with {blend_px}px overlap, {width}x{height}"
        )
        
        return tile_image
    
    def get_tile_info_summary(self, tiles: List[TileSpec]) -> dict:
        """
        Get summary information about the tiling strategy.
        
        Args:
            tiles: List of tile specifications
            
        Returns:
            Dictionary with tiling information
        """
        if not tiles:
            return {}
        
        total_display_pixels = sum(t.display_width * t.display_height for t in tiles)
        total_generated_pixels = sum(t.generated_width * t.generated_height for t in tiles)
        max_generated_pixels = max(t.generated_width * t.generated_height for t in tiles)
        
        return {
            'total_tiles': len(tiles),
            'total_display_pixels': total_display_pixels,
            'total_display_megapixels': total_display_pixels / 1_000_000,
            'total_generated_pixels': total_generated_pixels,
            'total_generated_megapixels': total_generated_pixels / 1_000_000,
            'max_tile_generated_pixels': max_generated_pixels,
            'max_tile_generated_megapixels': max_generated_pixels / 1_000_000,
            'tiles': [
                {
                    'index': t.tile_index,
                    'display_position': (t.display_x, t.display_y),
                    'display_size': (t.display_width, t.display_height),
                    'generated_size': (t.generated_width, t.generated_height),
                    'generated_megapixels': (t.generated_width * t.generated_height) / 1_000_000,
                    'overlap': t.overlap
                }
                for t in tiles
            ]
        }


def create_tiler_from_config(tile_config) -> PanoramaTiler:
    """
    Create a PanoramaTiler from a TileConfig object.
    
    Args:
        tile_config: TileConfig instance from service configuration
        
    Returns:
        Configured PanoramaTiler instance
    """
    return PanoramaTiler(
        display_tile_width=tile_config.display_width,
        display_tile_height=tile_config.display_height,
        generated_tile_width=tile_config.generated_width,
        generated_tile_height=tile_config.generated_height,
        min_overlap_percent=tile_config.min_overlap_percent,
        max_megapixels=tile_config.max_megapixels
    )

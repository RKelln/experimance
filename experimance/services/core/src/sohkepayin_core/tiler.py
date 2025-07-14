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


@dataclass
class TileSpec:
    """Specification for a single tile."""
    
    x: int  # X position in base image
    y: int  # Y position in base image  
    width: int  # Tile width
    height: int  # Tile height
    tile_index: int  # Tile number (0-based)
    total_tiles: int  # Total number of tiles
    
    # Position in final panorama (after any transformations)
    final_x: int = 0
    final_y: int = 0
    final_width: int = 0
    final_height: int = 0


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
        max_tile_width: int = 1920,
        max_tile_height: int = 1080,
        min_overlap_percent: float = 15.0,
        max_megapixels: float = 1.0,
        edge_blend_pixels: int = 50
    ):
        """
        Initialize tiler with configuration.
        
        Args:
            max_tile_width: Maximum width for generated tiles
            max_tile_height: Maximum height for generated tiles  
            min_overlap_percent: Minimum overlap between tiles (5-50%)
            max_megapixels: Maximum megapixels per tile (0.5-5.0)
            edge_blend_pixels: Pixels to blend at tile edges
        """
        self.max_tile_width = max_tile_width
        self.max_tile_height = max_tile_height
        self.min_overlap_percent = min_overlap_percent
        self.max_megapixels = max_megapixels
        self.edge_blend_pixels = edge_blend_pixels
        
        # Calculate effective max pixels
        self.max_pixels = int(max_megapixels * 1_000_000)
        
        logger.info(
            f"Tiler initialized: max_size={max_tile_width}x{max_tile_height}, "
            f"min_overlap={min_overlap_percent}%, max_mp={max_megapixels}, "
            f"blend_px={edge_blend_pixels}"
        )
    
    def calculate_tiles(self, panorama_width: int, panorama_height: int) -> List[TileSpec]:
        """
        Calculate optimal tiling strategy for a panorama.
        
        Args:
            panorama_width: Width of the panorama to tile
            panorama_height: Height of the panorama to tile
            
        Returns:
            List of TileSpec objects describing each tile
        """
        logger.info(f"Calculating tiles for {panorama_width}x{panorama_height} panorama")
        
        # Check if we can fit the entire panorama in a single tile
        if (panorama_width <= self.max_tile_width and 
            panorama_height <= self.max_tile_height and
            panorama_width * panorama_height <= self.max_pixels):
            
            logger.info("Panorama fits in single tile")
            return [TileSpec(
                x=0, y=0,
                width=panorama_width,
                height=panorama_height,
                tile_index=0,
                total_tiles=1,
                final_x=0, final_y=0,
                final_width=panorama_width,
                final_height=panorama_height
            )]
        
        # Calculate horizontal tiling
        tiles_x, tile_width, overlap_x = self._calculate_dimension_tiling(
            panorama_width, self.max_tile_width
        )
        
        # Calculate vertical tiling (usually 1 for panoramas)
        tiles_y, tile_height, overlap_y = self._calculate_dimension_tiling(
            panorama_height, self.max_tile_height
        )
        
        total_tiles = tiles_x * tiles_y
        
        logger.info(
            f"Tiling strategy: {tiles_x}x{tiles_y} = {total_tiles} tiles, "
            f"tile_size={tile_width}x{tile_height}, "
            f"overlap={overlap_x}x{overlap_y}px"
        )
        
        # Generate tile specifications
        tiles = []
        tile_index = 0
        
        for y_idx in range(tiles_y):
            for x_idx in range(tiles_x):
                # Calculate position in base image
                x_pos = x_idx * (tile_width - overlap_x) if x_idx > 0 else 0
                y_pos = y_idx * (tile_height - overlap_y) if y_idx > 0 else 0
                
                # Clamp to image bounds
                actual_width = min(tile_width, panorama_width - x_pos)
                actual_height = min(tile_height, panorama_height - y_pos)
                
                # Verify tile stays under megapixel limit
                if actual_width * actual_height > self.max_pixels:
                    logger.warning(
                        f"Tile {tile_index} exceeds megapixel limit: "
                        f"{actual_width}x{actual_height} = "
                        f"{actual_width * actual_height / 1_000_000:.2f}MP"
                    )
                
                tiles.append(TileSpec(
                    x=x_pos,
                    y=y_pos,
                    width=actual_width,
                    height=actual_height,
                    tile_index=tile_index,
                    total_tiles=total_tiles,
                    final_x=x_pos,  # Same as source for now
                    final_y=y_pos,
                    final_width=actual_width,
                    final_height=actual_height
                ))
                
                tile_index += 1
        
        return tiles
    
    def _calculate_dimension_tiling(
        self, 
        dimension: int, 
        max_tile_size: int
    ) -> Tuple[int, int, int]:
        """
        Calculate tiling for a single dimension.
        
        Args:
            dimension: Size of the dimension to tile
            max_tile_size: Maximum size per tile
            
        Returns:
            Tuple of (num_tiles, tile_size, overlap_pixels)
        """
        if dimension <= max_tile_size:
            return 1, dimension, 0
        
        # Calculate minimum number of tiles needed
        min_tiles = math.ceil(dimension / max_tile_size)
        
        # Try different tile counts to find optimal solution
        best_solution = None
        best_score = float('inf')
        
        for num_tiles in range(min_tiles, min_tiles + 3):  # Try a few options
            tile_size, overlap = self._calculate_tile_size_and_overlap(
                dimension, max_tile_size, num_tiles
            )
            
            if tile_size is None or overlap is None:  # Invalid solution
                continue
                
            # Score solution (prefer fewer tiles, reasonable overlap)
            overlap_percent = (overlap / tile_size) * 100 if tile_size > 0 else 100
            if overlap_percent < self.min_overlap_percent:
                continue  # Skip if overlap too small
            
            # Score: balance number of tiles vs overlap efficiency
            score = num_tiles + (overlap_percent / 100) * 0.5
            
            if score < best_score:
                best_score = score
                best_solution = (num_tiles, tile_size, overlap)
        
        if best_solution is None:
            # Fallback: force minimum tiles with whatever overlap we get
            num_tiles = min_tiles
            tile_size, overlap = self._calculate_tile_size_and_overlap(
                dimension, max_tile_size, num_tiles
            )
            if tile_size is None or overlap is None:
                logger.error(
                    f"Failed to calculate valid tiling for {dimension}px dimension "
                    f"with max tile size {max_tile_size}px"
                )
                return num_tiles, max_tile_size, 0
            logger.warning(
                f"Using fallback tiling: {num_tiles} tiles, {overlap}px overlap "
                f"({overlap/tile_size*100:.1f}%)"
            )
            return num_tiles, tile_size or max_tile_size, overlap or 0
        
        return best_solution
    
    def _calculate_tile_size_and_overlap(
        self, 
        dimension: int, 
        max_tile_size: int, 
        num_tiles: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Calculate tile size and overlap for a given number of tiles.
        
        Args:
            dimension: Total dimension size
            max_tile_size: Maximum allowed tile size
            num_tiles: Number of tiles to use
            
        Returns:
            Tuple of (tile_size, overlap_pixels) or (None, None) if invalid
        """
        if num_tiles == 1:
            return dimension, 0
        
        # For n tiles with overlap: dimension = n * tile_size - (n-1) * overlap
        # Rearranging: overlap = (n * tile_size - dimension) / (n - 1)
        
        # Start with maximum tile size and work down
        for tile_size in range(max_tile_size, max_tile_size // 2, -1):
            overlap = (num_tiles * tile_size - dimension) / (num_tiles - 1)
            
            if overlap < 0:
                continue  # Need positive overlap
            
            if overlap > tile_size * 0.5:
                continue  # Too much overlap
            
            # Check megapixel constraint (assuming worst case height)
            if tile_size * self.max_tile_height > self.max_pixels:
                continue
            
            return tile_size, int(overlap)
        
        return None, None
    
    def crop_tile_from_image(self, image: Image.Image, tile_spec: TileSpec) -> Image.Image:
        """
        Crop a tile from the base image.
        
        Args:
            image: Source panorama image
            tile_spec: Specification for the tile to crop
            
        Returns:
            Cropped tile image
        """
        return image.crop((
            tile_spec.x,
            tile_spec.y,
            tile_spec.x + tile_spec.width,
            tile_spec.y + tile_spec.height
        ))
    
    def apply_edge_blending(self, tile_image: Image.Image, tile_spec: TileSpec) -> Image.Image:
        """
        Apply edge blending to a tile for seamless composition.
        
        Creates a transparency mask that fades out at the edges where
        this tile will overlap with adjacent tiles.
        
        Args:
            tile_image: Tile image to blend
            tile_spec: Tile specification
            
        Returns:
            Tile image with alpha channel for blending
        """
        # Convert to RGBA if needed
        if tile_image.mode != 'RGBA':
            tile_image = tile_image.convert('RGBA')
        
        # Create alpha mask
        alpha = Image.new('L', tile_image.size, 255)
        
        # Apply edge fading based on tile position
        width, height = tile_image.size
        blend_px = min(self.edge_blend_pixels, width // 4, height // 4)
        
        if tile_spec.tile_index > 0:  # Not leftmost tile
            # Fade left edge
            for x in range(min(blend_px, width)):
                alpha_value = int(255 * (x / blend_px))
                for y in range(height):
                    alpha.putpixel((x, y), alpha_value)
        
        if tile_spec.tile_index < tile_spec.total_tiles - 1:  # Not rightmost tile
            # Fade right edge
            for x in range(max(0, width - blend_px), width):
                alpha_value = int(255 * ((width - 1 - x) / blend_px))
                for y in range(height):
                    alpha.putpixel((x, y), alpha_value)
        
        # Apply alpha mask
        tile_image.putalpha(alpha)
        
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
        
        total_pixels = sum(t.width * t.height for t in tiles)
        max_pixels = max(t.width * t.height for t in tiles)
        
        return {
            'total_tiles': len(tiles),
            'total_pixels': total_pixels,
            'total_megapixels': total_pixels / 1_000_000,
            'max_tile_pixels': max_pixels,
            'max_tile_megapixels': max_pixels / 1_000_000,
            'tiles': [
                {
                    'index': t.tile_index,
                    'position': (t.x, t.y),
                    'size': (t.width, t.height),
                    'megapixels': (t.width * t.height) / 1_000_000
                }
                for t in tiles
            ]
        }

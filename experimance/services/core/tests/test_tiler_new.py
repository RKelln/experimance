#!/usr/bin/env python3
"""
Unit tests for the updated PanoramaTiler class.

Tests the simplified tiling approach with:
- Display vs generated resolution handling
- Left-edge-only fading
- Overlap extension calculations
"""

import pytest
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sohkepayin_core.tiler import PanoramaTiler, TileSpec, create_tiler_from_config
from sohkepayin_core.config import TileConfig


class TestPanoramaTilerNew:
    """Test the updated PanoramaTiler class."""
    
    def test_single_tile_scenario(self):
        """Test when panorama fits in a single tile."""
        tiler = PanoramaTiler(
            display_tile_width=1920,
            display_tile_height=1080,
            generated_tile_width=1344,
            generated_tile_height=768,
            max_megapixels=2.0
        )
        
        # Small panorama that fits in one tile
        tiles = tiler.calculate_tiles(1600, 900)
        
        assert len(tiles) == 1
        tile = tiles[0]
        assert tile.display_x == 0
        assert tile.display_y == 0
        assert tile.display_width == 1600
        assert tile.display_height == 900
        assert tile.generated_width == 1344  # Uses base generated size
        assert tile.generated_height == 768
        assert tile.tile_index == 0
        assert tile.total_tiles == 1
        assert tile.has_left_fade == False
    
    def test_horizontal_tiling_two_tiles(self):
        """Test horizontal tiling with two tiles."""
        tiler = PanoramaTiler(
            display_tile_width=1920,
            display_tile_height=1080,
            generated_tile_width=1344,
            generated_tile_height=768,
            min_overlap_percent=10.0
        )
        
        # Panorama that needs 2 tiles: 3000px wide
        tiles = tiler.calculate_tiles(3000, 1080)
        
        assert len(tiles) == 2
        
        # First tile
        tile0 = tiles[0]
        assert tile0.display_x == 0
        assert tile0.display_y == 0
        assert tile0.display_width == 1920
        assert tile0.display_height == 1080
        assert tile0.tile_index == 0
        assert tile0.total_tiles == 2
        assert tile0.has_left_fade == False
        
        # Generated width should include right overlap for first tile
        assert tile0.generated_width > 1344  # Should have overlap added
        assert tile0.generated_height == 768
        
        # Second tile
        tile1 = tiles[1]
        assert tile1.tile_index == 1
        assert tile1.total_tiles == 2
        assert tile1.has_left_fade == True  # All tiles except first have left fade
        
        # Generated width should include left overlap for last tile
        assert tile1.generated_width > 1344  # Should have overlap added
        
        # Display positions should be calculated correctly
        assert tile1.display_x > 0  # Should start after some offset
        assert tile1.display_x + tile1.display_width >= 3000  # Should reach the end
    
    def test_horizontal_tiling_three_tiles(self):
        """Test horizontal tiling with three tiles."""
        tiler = PanoramaTiler(
            display_tile_width=1920,
            display_tile_height=1080,
            generated_tile_width=1344,
            generated_tile_height=768,
            min_overlap_percent=15.0
        )
        
        # Panorama that needs 3 tiles: 5760px wide (3 * 1920)
        tiles = tiler.calculate_tiles(5760, 1080)
        
        assert len(tiles) == 3
        
        # Check fade properties
        assert tiles[0].has_left_fade == False  # First tile
        assert tiles[1].has_left_fade == True   # Middle tile
        assert tiles[2].has_left_fade == True   # Last tile
        
        # Check generated widths include appropriate overlaps
        assert tiles[0].generated_width > 1344  # First: base + right overlap
        assert tiles[1].generated_width > tiles[0].generated_width  # Middle: base + both overlaps
        assert tiles[2].generated_width > 1344  # Last: base + left overlap
        
        # Middle tile should have the largest generated width (left + right overlap)
        middle_width = tiles[1].generated_width
        first_width = tiles[0].generated_width
        last_width = tiles[2].generated_width
        assert middle_width > first_width
        assert middle_width > last_width
    
    def test_overlap_calculation(self):
        """Test that overlaps are calculated correctly."""
        tiler = PanoramaTiler(
            display_tile_width=1000,
            display_tile_height=1080,
            generated_tile_width=800,
            generated_tile_height=768,
            min_overlap_percent=20.0
        )
        
        # Test with exact 2-tile scenario: 2000px / 1000px = 2 tiles
        tiles = tiler.calculate_tiles(2000, 1080)
        
        assert len(tiles) == 2
        
        # Calculate expected overlap in display space
        # For 2 tiles: 2000 = 2 * 1000 - (2-1) * overlap
        # So: overlap = (2000 - 2000) / 1 = 0 (no overlap needed for exact fit)
        
        # But let's test with a case that needs overlap
        tiles = tiler.calculate_tiles(1800, 1080)  # Needs 2 tiles with overlap
        
        assert len(tiles) == 2
        # In this case, we'd need overlap to cover the gap efficiently
    
    def test_create_from_config(self):
        """Test creating tiler from configuration."""
        config = TileConfig(
            display_width=1920,
            display_height=1080,
            generated_width=1344,
            generated_height=768,
            min_overlap_percent=20.0,
            max_megapixels=2.0
        )
        
        tiler = create_tiler_from_config(config)
        
        assert tiler.display_tile_width == 1920
        assert tiler.display_tile_height == 1080
        assert tiler.generated_tile_width == 1344
        assert tiler.generated_tile_height == 768
        assert tiler.min_overlap_percent == 20.0
        assert tiler.max_megapixels == 2.0
    
    def test_tile_info_summary(self):
        """Test the tile information summary."""
        tiler = PanoramaTiler(
            display_tile_width=1920,
            display_tile_height=1080,
            generated_tile_width=1344,
            generated_tile_height=768
        )
        
        tiles = tiler.calculate_tiles(3000, 1080)
        summary = tiler.get_tile_info_summary(tiles)
        
        assert 'total_tiles' in summary
        assert 'total_display_pixels' in summary
        assert 'total_generated_pixels' in summary
        assert 'tiles' in summary
        assert summary['total_tiles'] == len(tiles)
        
        # Check individual tile info
        for i, tile_info in enumerate(summary['tiles']):
            assert tile_info['index'] == i
            assert 'display_position' in tile_info
            assert 'display_size' in tile_info
            assert 'generated_size' in tile_info
            assert 'has_left_fade' in tile_info
            
            # Verify fade property matches tile
            assert tile_info['has_left_fade'] == tiles[i].has_left_fade
    
    def test_edge_blending_simplified(self):
        """Test the simplified edge blending (left edge only)."""
        from PIL import Image
        
        test_image = Image.new('RGB', (1000, 500), color='blue')
        
        # Test first tile (no fade)
        tile_spec_first = TileSpec(
            display_x=0, display_y=0,
            display_width=1000, display_height=500,
            generated_width=1200, generated_height=500,
            tile_index=0, total_tiles=3,
            has_left_fade=False
        )
        
        tiler = PanoramaTiler(edge_blend_pixels=50)
        blended_first = tiler.apply_edge_blending(test_image, tile_spec_first)
        
        # First tile should not be modified (no fade)
        assert blended_first.mode == 'RGBA'
        # Should return the image unchanged since has_left_fade is False
        
        # Test middle tile (with left fade)
        tile_spec_middle = TileSpec(
            display_x=500, display_y=0,
            display_width=1000, display_height=500,
            generated_width=1400, generated_height=500,
            tile_index=1, total_tiles=3,
            has_left_fade=True
        )
        
        blended_middle = tiler.apply_edge_blending(test_image, tile_spec_middle)
        
        # Should have alpha channel with left fade
        assert blended_middle.mode == 'RGBA'
        alpha_channel = blended_middle.split()[-1]
        
        # Left edge should be faded
        left_edge_alpha = alpha_channel.getpixel((0, 250))
        assert left_edge_alpha < 255  # Should be faded
        
        # Center should be fully opaque
        center_alpha = alpha_channel.getpixel((500, 250))
        assert center_alpha == 255  # Should be fully opaque
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        tiler = PanoramaTiler(
            display_tile_width=1920,
            display_tile_height=1080,
            generated_tile_width=1344,
            generated_tile_height=768
        )
        
        # Exact multiple of tile width
        tiles = tiler.calculate_tiles(1920, 1080)
        assert len(tiles) == 1
        assert tiles[0].has_left_fade == False
        
        # Just over tile width
        tiles = tiler.calculate_tiles(1921, 1080)
        assert len(tiles) == 2
        assert tiles[0].has_left_fade == False
        assert tiles[1].has_left_fade == True
        
        # Very wide panorama
        tiles = tiler.calculate_tiles(10000, 1080)
        assert len(tiles) > 2
        
        # Verify all tiles have proper fade settings
        for i, tile in enumerate(tiles):
            if i == 0:
                assert tile.has_left_fade == False
            else:
                assert tile.has_left_fade == True
    
    def test_display_vs_generated_scaling(self):
        """Test that display and generated resolutions are handled correctly."""
        # Set up different scaling ratios
        tiler = PanoramaTiler(
            display_tile_width=1920,      # Display resolution
            display_tile_height=1080,
            generated_tile_width=1344,    # Lower generated resolution
            generated_tile_height=768,
            min_overlap_percent=15.0
        )
        
        tiles = tiler.calculate_tiles(3840, 1080)  # 2 display tiles wide
        
        assert len(tiles) == 2
        
        # Display positioning should use display resolution
        assert tiles[0].display_width == 1920
        assert tiles[1].display_width == 1920
        
        # Generated sizes should be based on generated resolution + overlap
        assert tiles[0].generated_width > 1344  # Base + overlap
        assert tiles[1].generated_width > 1344  # Base + overlap
        
        # Generated height should match the base generated height
        assert tiles[0].generated_height == 768
        assert tiles[1].generated_height == 768


if __name__ == '__main__':
    pytest.main([__file__])

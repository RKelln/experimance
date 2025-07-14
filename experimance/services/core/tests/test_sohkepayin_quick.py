#!/usr/bin/env python3
"""
Quick integration test for Sohkepayin core service.

This is a simple test to verify the basic components work together
without requiring full ZMQ infrastructure.
"""

import sys
import tempfile
import asyncio
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sohkepayin_core.config import SohkepayinCoreConfig, LLMConfig, PanoramaConfig, TileConfig
from sohkepayin_core.tiler import PanoramaTiler
from sohkepayin_core.llm_prompt_builder import LLMPromptBuilder
from sohkepayin_core.llm import MockLLMProvider


async def test_basic_components():
    """Test basic component functionality."""
    print("🧪 Testing basic Sohkepayin components...")
    
    # Test 1: Configuration loading
    print("1. Testing configuration...")
    
    # Create config directly using proper config classes
    config = SohkepayinCoreConfig(
        service_name="test_sohkepayin_core",
        llm=LLMConfig(
            provider="mock",
            model="test-model"
        ),
        panorama=PanoramaConfig(
            width=2400,
            height=1080
        ),
        tiles=TileConfig(
            max_width=1200,
            max_height=1080,
            min_overlap_percent=15.0,
            max_megapixels=1.0
        )
    )
    print(f"   ✅ Config loaded: {config.service_name}")
    
    # Test 2: Tiler functionality
    print("2. Testing tiler...")
    tiler = PanoramaTiler(
        max_tile_width=config.tiles.max_width,
        max_tile_height=config.tiles.max_height,
        min_overlap_percent=config.tiles.min_overlap_percent,
        max_megapixels=config.tiles.max_megapixels
    )
    
    tiles = tiler.calculate_tiles(config.panorama.width, config.panorama.height)
    print(f"   ✅ Tiler calculated {len(tiles)} tiles for {config.panorama.width}x{config.panorama.height}")
    
    # Print tile details
    for i, tile in enumerate(tiles):
        print(f"      Tile {i}: {tile.width}x{tile.height} at ({tile.x}, {tile.y})")
    
    # Test tile summary
    summary = tiler.get_tile_info_summary(tiles)
    print(f"   📊 Total megapixels: {summary['total_megapixels']:.2f}MP")
    
    # Test 3: LLM prompt builder
    print("3. Testing LLM prompt builder...")
    mock_llm = MockLLMProvider()
    
    prompt_builder = LLMPromptBuilder(
        llm=mock_llm,
        system_prompt_or_file="You are a helpful assistant that creates image prompts."
    )
    
    test_story = "A serene mountain landscape with snow-capped peaks reflected in a crystal-clear alpine lake."
    
    try:
        base_prompt = await prompt_builder.build_prompt(test_story)
        print(f"   ✅ Base prompt generated: {len(base_prompt.prompt)} characters")
        print(f"      Prompt preview: {base_prompt.prompt[:100]}...")
        
        # Test panorama prompt conversion
        panorama_prompt = prompt_builder.base_prompt_to_panorama_prompt(base_prompt)
        print(f"   ✅ Panorama prompt: {len(panorama_prompt.prompt)} characters")
        
        # Test tile prompt conversion
        tile_prompt = prompt_builder.base_prompt_to_tile_prompt(base_prompt)
        print(f"   ✅ Tile prompt: {len(tile_prompt.prompt)} characters")
        
    except Exception as e:
        print(f"   ❌ LLM prompt builder failed: {e}")
        return False
    
    # Test 4: Edge cases
    print("4. Testing edge cases...")
    
    # Very small panorama (should be single tile)
    small_tiles = tiler.calculate_tiles(800, 600)
    assert len(small_tiles) == 1, f"Expected 1 tile for small panorama, got {len(small_tiles)}"
    print("   ✅ Small panorama handled correctly")
    
    # Very large panorama (should create multiple tiles)
    large_tiles = tiler.calculate_tiles(4800, 1080)
    assert len(large_tiles) > 1, f"Expected multiple tiles for large panorama, got {len(large_tiles)}"
    print(f"   ✅ Large panorama creates {len(large_tiles)} tiles")
    
    # Verify megapixel constraints
    for tile in large_tiles:
        pixels = tile.width * tile.height
        megapixels = pixels / 1_000_000
        assert megapixels <= config.tiles.max_megapixels * 1.1, f"Tile exceeds megapixel limit: {megapixels:.2f}MP"
    print("   ✅ All tiles respect megapixel constraints")
    
    print("\n🎉 All basic component tests passed!")
    return True


async def test_service_lifecycle():
    """Test service creation without ZMQ."""
    print("\n🔄 Testing service lifecycle (without ZMQ)...")
    
    # Create config directly using proper config classes
    config = SohkepayinCoreConfig(
        service_name="test_lifecycle",
        llm=LLMConfig(
            provider="mock",
            model="test-model"
        ),
        panorama=PanoramaConfig(
            width=1920,
            height=1080
        ),
        tiles=TileConfig(
            max_width=1920,
            max_height=1080
        )
    )
    
    # Import here to avoid circular imports
    from sohkepayin_core.sohkepayin_core import SohkepayinCoreService, CoreState
    
    # Create service (but don't start ZMQ)
    service = SohkepayinCoreService(config)
    
    print(f"   ✅ Service created: {service.config.service_name}")
    print(f"   📊 Initial state: {service.core_state}")
    
    # Test state transitions
    await service._transition_to_state(CoreState.LISTENING)
    assert service.core_state == CoreState.LISTENING
    print("   ✅ State transition to LISTENING")
    
    await service._transition_to_state(CoreState.BASE_IMAGE)
    assert service.core_state == CoreState.BASE_IMAGE
    print("   ✅ State transition to BASE_IMAGE")
    
    await service._transition_to_state(CoreState.TILES)
    assert service.core_state == CoreState.TILES
    print("   ✅ State transition to TILES")
    
    print("   🎉 Service lifecycle test passed!")
    return True


if __name__ == "__main__":
    async def main():
        print("🚀 Running Sohkepayin Core Service Quick Tests\n")
        
        try:
            success1 = await test_basic_components()
            success2 = await test_service_lifecycle()
            
            if success1 and success2:
                print("\n✨ All tests passed! Core service components are working correctly.")
                return 0
            else:
                print("\n❌ Some tests failed!")
                return 1
                
        except Exception as e:
            print(f"\n💥 Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

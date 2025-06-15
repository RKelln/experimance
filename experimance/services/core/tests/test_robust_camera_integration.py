#!/usr/bin/env python3
"""
Test the integration of robust_camera.py into experimance_core.py.

This test validates that the new robust camera system is properly integrated
and replaces the old depth_finder functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import DEFAULT_CONFIG_PATH

async def test_core_service_initialization():
    """Test that the core service initializes with robust camera."""
    
    print("🧪 Testing Experimance Core Service with Robust Camera Integration")
    print("=" * 60)
    
    try:
        # Create core service
        print("📦 Creating Experimance Core Service...")
        service = ExperimanceCoreService(config_path=DEFAULT_CONFIG_PATH)
        
        # Check that it has the new depth processor instead of old generator
        print("🔍 Checking service attributes...")
        assert hasattr(service, '_depth_processor'), "Service should have _depth_processor attribute"
        assert hasattr(service, '_camera_state'), "Service should have _camera_state attribute"
        assert not hasattr(service, 'depth_generator'), "Service should not have old depth_generator attribute"
        
        print("✅ Service created with correct attributes")
        
        # Test camera config creation
        print("⚙️  Testing camera config creation...")
        camera_config = service._create_camera_config()
        print(f"   Resolution: {camera_config.resolution}")
        print(f"   FPS: {camera_config.fps}")
        print(f"   Min depth: {camera_config.min_depth}")
        print(f"   Max depth: {camera_config.max_depth}")
        print(f"   Change threshold: {camera_config.change_threshold}")
        print(f"   Output resolution: {camera_config.output_resolution}")
        
        print("✅ Camera config created successfully")
        
        # Test that initialization methods exist
        print("🔧 Checking initialization methods...")
        assert hasattr(service, '_initialize_depth_processor'), "Should have _initialize_depth_processor method"
        assert hasattr(service, '_initialize_depth_processor_with_retry'), "Should have retry method"
        
        print("✅ All required methods exist")
        
        print("🎉 Integration test passed! Robust camera is properly integrated.")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_depth_processor_mock():
    """Test that we can create a mock depth processor."""
    
    print("\n🎭 Testing Mock Depth Processor Integration")
    print("=" * 60)
    
    try:
        # Create core service 
        service = ExperimanceCoreService(config_path=DEFAULT_CONFIG_PATH)
        
        # Create camera config
        camera_config = service._create_camera_config()
        
        # Test that we can create a mock processor
        from experimance_core.depth_factory import create_depth_processor
        
        # Use mock mode with example images
        mock_path = str(Path(__file__).parent / "media" / "images" / "mocks")
        if Path(mock_path).exists():
            print(f"📁 Using mock images from: {mock_path}")
            mock_processor = create_depth_processor(camera_config, mock_path=mock_path)
            print("✅ Mock depth processor created successfully")
        else:
            print(f"⚠️  Mock images directory not found at {mock_path}")
            print("   Creating processor without mock path...")
            processor = create_depth_processor(camera_config, mock_path=None)
            print("✅ Real depth processor created successfully")
        
    except Exception as e:
        print(f"❌ Mock processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    async def main():
        print("🚀 Starting Robust Camera Integration Tests")
        print("=" * 70)
        
        success1 = await test_core_service_initialization()
        success2 = await test_depth_processor_mock()
        
        print("\n" + "=" * 70)
        if success1 and success2:
            print("🎉 All integration tests passed!")
            print("   The robust camera system is properly integrated into experimance_core.py")
            print("   The old depth_finder dependencies have been successfully replaced.")
        else:
            print("❌ Some integration tests failed!")
            sys.exit(1)
    
    asyncio.run(main())

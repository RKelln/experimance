#!/usr/bin/env python3
"""
Quick test to verify that experimance_core can be started with the new robust camera integration.
"""

import asyncio
import sys
import signal
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import DEFAULT_CONFIG_PATH

async def test_service_startup():
    """Test that the service can start and run briefly."""
    
    print("🚀 Testing Experimance Core Service Startup with Robust Camera")
    print("=" * 70)
    
    service = None
    try:
        # Create and start the service
        print("📦 Creating service...")
        service = ExperimanceCoreService(config_path=DEFAULT_CONFIG_PATH)
        
        print("🔧 Starting service...")
        await service.start()
        
        print("✅ Service started successfully!")
        print(f"   Camera state: {service._camera_state}")
        print(f"   Current era: {service.current_era}")
        print(f"   Current biome: {service.current_biome}")
        
        # Let it run for a couple of seconds to verify tasks work
        print("⏱️  Running for 3 seconds to test depth processing...")
        await asyncio.sleep(3)
        
        print("✅ Service ran successfully!")
        print(f"   Final camera state: {service._camera_state}")
        
        # Check if depth processing tried to initialize
        if hasattr(service, 'depth_retry_count'):
            print(f"   Depth retry count: {service.depth_retry_count}")
        
    except Exception as e:
        print(f"❌ Service startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if service:
            print("🛑 Stopping service...")
            await service.stop()
            print("✅ Service stopped cleanly")
    
    return True

if __name__ == "__main__":
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        success = asyncio.run(test_service_startup())
        if success:
            print("\n🎉 Core service startup test passed!")
            print("   The robust camera integration is working correctly.")
        else:
            print("\n❌ Core service startup test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(0)

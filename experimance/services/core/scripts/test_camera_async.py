#!/usr/bin/env python3
"""
Test Camera Recovery

Simple test to verify the async camera utilities work correctly.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_core.camera_utils import (
    get_camera_diagnostics_async, 
    reset_realsense_camera_async,
    kill_camera_processes_async
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_async_functions():
    """Test all async camera functions."""
    print("🧪 Testing Async Camera Functions")
    print("=" * 40)
    
    # Test 1: Get diagnostics
    print("\n1️⃣ Testing camera diagnostics...")
    try:
        diagnostics = await asyncio.wait_for(get_camera_diagnostics_async(), timeout=10.0)
        print(f"   ✅ Diagnostics: {len(diagnostics['devices'])} devices found")
        print(f"   📊 Processes: {len(diagnostics['processes'])} camera-related processes")
        print(f"   🔌 USB devices: {len(diagnostics['usb_devices'])} Intel/RealSense USB devices")
    except asyncio.TimeoutError:
        print("   ⏰ Diagnostics timed out (10s)")
    except Exception as e:
        print(f"   ❌ Diagnostics error: {e}")
    
    # Test 2: Process killing (only if there are processes to kill)
    print("\n2️⃣ Testing process killing...")
    try:
        killed = await asyncio.wait_for(kill_camera_processes_async(), timeout=5.0)
        if killed:
            print("   ✅ Some camera processes were killed")
        else:
            print("   ℹ️ No camera processes needed killing")
    except asyncio.TimeoutError:
        print("   ⏰ Process killing timed out (5s)")
    except Exception as e:
        print(f"   ❌ Process killing error: {e}")
    
    # Test 3: Camera reset (only if there are cameras)
    print("\n3️⃣ Testing camera reset...")
    try:
        # First check if there are cameras
        diagnostics = await get_camera_diagnostics_async()
        if len(diagnostics['devices']) > 0:
            print("   🔄 Attempting camera reset...")
            success = await asyncio.wait_for(reset_realsense_camera_async(aggressive=False), timeout=30.0)
            if success:
                print("   ✅ Camera reset successful")
            else:
                print("   ⚠️ Camera reset failed (but no error)")
        else:
            print("   ℹ️ No cameras found to reset")
    except asyncio.TimeoutError:
        print("   ⏰ Camera reset timed out (30s)")
    except asyncio.CancelledError:
        print("   🛑 Camera reset was cancelled")
    except Exception as e:
        print(f"   ❌ Camera reset error: {e}")
    
    print("\n🎉 Async function testing completed!")

async def test_cancellation():
    """Test that functions can be cancelled with Ctrl+C."""
    print("\n🛑 Testing Cancellation (press Ctrl+C during reset)")
    print("   Starting a camera reset that you can cancel...")
    
    try:
        success = await reset_realsense_camera_async(aggressive=True)
        print(f"   Reset completed: {success}")
    except asyncio.CancelledError:
        print("   ✅ Reset was properly cancelled!")
        raise

async def main():
    """Main test function."""
    try:
        await test_async_functions()
        
        # Ask if user wants to test cancellation
        try:
            test_cancel = input("\n❓ Test cancellation? (y/N): ").strip().lower()
            if test_cancel == 'y':
                await test_cancellation()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Testing cancelled by user")
            
    except KeyboardInterrupt:
        print("\n✅ Properly caught Ctrl+C during testing!")
        return True
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n👋 Final Ctrl+C caught at top level")
        sys.exit(130)  # Standard exit code for Ctrl+C

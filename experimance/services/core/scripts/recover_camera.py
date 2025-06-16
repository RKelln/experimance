#!/usr/bin/env python3
"""
Camera Recovery Script

Quick script to recover stuck RealSense cameras.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_core.camera_utils import reset_realsense_camera_async, get_camera_diagnostics_async

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main recovery function."""
    print("🔧 RealSense Camera Recovery Tool")
    print("=" * 40)
    
    try:
        # Show current state
        print("\n📊 Current camera state:")
        diagnostics = await asyncio.wait_for(get_camera_diagnostics_async(), timeout=10.0)
        print(f"   Devices found: {len(diagnostics['devices'])}")
        print(f"   Context created: {diagnostics['realsense_info'].get('context_created', False)}")
        
        if diagnostics['realsense_info'].get('error'):
            print(f"   Error: {diagnostics['realsense_info']['error']}")
        
        # Attempt aggressive reset
        print("\n🔄 Attempting aggressive camera reset...")
        success = await asyncio.wait_for(
            reset_realsense_camera_async(aggressive=True), 
            timeout=60.0
        )
        
        if success:
            print("✅ Camera reset successful!")
            
            # Verify recovery
            print("\n🔍 Verifying recovery...")
            await asyncio.sleep(2)
            post_diagnostics = await asyncio.wait_for(get_camera_diagnostics_async(), timeout=10.0)
            print(f"   Devices found after reset: {len(post_diagnostics['devices'])}")
            
            if len(post_diagnostics['devices']) > 0:
                print("🎉 Camera recovery completed successfully!")
                return True
            else:
                print("⚠️  Reset completed but no devices detected")
                return False
        else:
            print("❌ Camera reset failed")
            return False
            
    except asyncio.TimeoutError:
        print("⏰ Recovery operation timed out")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Recovery cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Recovery failed with error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)  # Standard exit code for Ctrl+C
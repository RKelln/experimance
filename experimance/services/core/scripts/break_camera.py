#!/usr/bin/env python3
"""
Camera Breaking Script

This script intentionally creates bad camera states for testing recovery mechanisms.
It starts the camera but doesn't properly clean up, leaving it in a stuck state.
"""
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pyrealsense2 as rs  # type: ignore
from experimance_core.camera_utils import get_camera_diagnostics_async

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to hold camera resources
pipelines = []
contexts = []
devices = []

def signal_handler(signum, frame):
    """Handle signals by exiting without cleanup."""
    print(f"\n💥 Received signal {signum} - exiting WITHOUT cleanup!")
    print("🔥 Camera should now be in a broken state for testing...")
    sys.exit(1)  # Exit without cleanup

async def break_camera_method_1():
    """Break camera by starting pipeline and exiting without stopping."""
    print("🚨 Method 1: Start pipeline and exit without stopping")
    
    try:
        # Create pipeline but don't store it for cleanup
        pipeline = rs.pipeline()  # type: ignore
        config = rs.config()  # type: ignore
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # type: ignore
        
        print("   Starting camera pipeline...")
        profile = pipeline.start(config)
        
        # Get a few frames to ensure camera is fully started
        print("   Capturing frames to ensure camera is active...")
        for i in range(5):
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if depth:
                print(f"   Frame {i+1}: Got depth frame with {depth.get_width()}x{depth.get_height()} resolution")
            time.sleep(0.1)
        
        # Store pipeline globally so it doesn't get garbage collected
        pipelines.append(pipeline)
        
        print("✅ Camera pipeline started successfully")
        print("💥 NOW EXITING WITHOUT CLEANUP - Camera should be stuck!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start pipeline: {e}")
        return False

async def break_camera_method_2():
    """Break camera by creating multiple contexts without cleanup."""
    print("🚨 Method 2: Create multiple contexts without cleanup")
    
    try:
        print("   Creating multiple RealSense contexts...")
        for i in range(3):
            ctx = rs.context()  # type: ignore
            devices_list = ctx.query_devices()
            print(f"   Context {i+1}: Found {len(devices_list)} devices")
            
            # Store contexts globally
            contexts.append(ctx)
            
            if len(devices_list) > 0:
                dev = devices_list[0]
                print(f"   Device info: {dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else 'Unknown'}")  # type: ignore
                devices.append(dev)
        
        print("✅ Multiple contexts created")
        print("💥 NOW EXITING WITHOUT CLEANUP - Contexts should be leaked!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create contexts: {e}")
        return False

async def break_camera_method_3():
    """Break camera by starting and abruptly terminating during frame capture."""
    print("🚨 Method 3: Start capture and terminate during active streaming")
    
    try:
        pipeline = rs.pipeline()  # type: ignore
        config = rs.config()  # type: ignore
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # type: ignore
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # type: ignore
        
        print("   Starting dual-stream pipeline...")
        profile = pipeline.start(config)
        
        # Start aggressive frame capture
        print("   Starting aggressive frame capture (will exit mid-stream)...")
        pipelines.append(pipeline)
        
        # Capture frames aggressively then exit
        for i in range(10):
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            print(f"   Frame {i+1}: depth={depth is not None}, color={color is not None}")
            
            if i == 5:
                print("💥 TERMINATING MID-CAPTURE!")
                break
                
        return True
        
    except Exception as e:
        print(f"❌ Failed during aggressive capture: {e}")
        return False

async def main():
    """Main function to break the camera."""
    print("💥 RealSense Camera Breaking Tool")
    print("=" * 40)
    print("⚠️  WARNING: This will intentionally put your camera in a bad state!")
    print("   Use the recover_camera.py script to fix it afterwards.")
    print()
    
    # Set up signal handlers to exit without cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Show current camera state
        print("📊 Current camera state before breaking:")
        diagnostics = await get_camera_diagnostics_async()
        print(f"   Devices found: {len(diagnostics['devices'])}")
        print(f"   Context created: {diagnostics['realsense_info'].get('context_created', False)}")
        
        if len(diagnostics['devices']) == 0:
            print("❌ No cameras found - cannot break what isn't there!")
            return False
        
        # Choose breaking method
        print("\n🔧 Choose breaking method:")
        print("1. Start pipeline and exit without stopping")
        print("2. Create multiple contexts without cleanup")
        print("3. Start capture and terminate during streaming")
        print("4. All methods (maximum chaos)")
        
        try:
            choice = input("\nEnter method (1-4) or 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Cancelled by user")
            return False
        
        if choice == 'q':
            print("👋 Exiting safely without breaking anything")
            return False
        
        success = False
        
        if choice == '1':
            success = await break_camera_method_1()
        elif choice == '2':
            success = await break_camera_method_2()
        elif choice == '3':
            success = await break_camera_method_3()
        elif choice == '4':
            print("🎆 MAXIMUM CHAOS MODE - Running all breaking methods!")
            success1 = await break_camera_method_1()
            await asyncio.sleep(1)
            success2 = await break_camera_method_2()
            await asyncio.sleep(1)
            success3 = await break_camera_method_3()
            success = success1 or success2 or success3
        else:
            print("❌ Invalid choice")
            return False
        
        if success:
            print("\n🎯 Camera breaking complete!")
            print("💡 The camera should now be in a problematic state.")
            print("🔧 Use 'uv run python services/core/scripts/recover_camera.py' to fix it.")
            print("\n💀 Exiting without cleanup in 3 seconds...")
            
            # Give user a chance to Ctrl+C for immediate exit
            try:
                await asyncio.sleep(3)
            except KeyboardInterrupt:
                print("\n💥 Immediate exit via Ctrl+C!")
            
            # Exit without any cleanup
            sys.exit(1)
        else:
            print("❌ Failed to break camera")
            return False
            
    except KeyboardInterrupt:
        print("\n💥 Interrupted - camera may be in broken state!")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Camera might still be broken though...")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        # If we get here, something went wrong
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n💥 Final Ctrl+C - guaranteed broken camera state!")
        sys.exit(130)  # Standard exit code for Ctrl+C

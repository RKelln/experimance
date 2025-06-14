#!/usr/bin/env python3
"""
Minimal RealSense camera test with validated settings.

This script uses only camera settings that we know work for the D415.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time


def reset_camera():
    """Reset the camera hardware if possible."""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found")
            return False
        
        dev = devices[0]
        device_name = dev.get_info(rs.camera_info.name)
        print(f"Found device: {device_name}")
        print("Attempting hardware reset...")
        dev.hardware_reset()
        print("Hardware reset successful")
        time.sleep(2)  # Wait for device to reinitialize
        return True
        
    except Exception as e:
        print(f"Reset failed: {e}")
        return False


def test_camera_settings():
    """Test different camera settings to find what works."""
    
    # Known working settings based on rs-enumerate-devices output
    test_configs = [
        {"name": "640x480@30Hz", "width": 640, "height": 480, "fps": 30},
        {"name": "640x480@15Hz", "width": 640, "height": 480, "fps": 15},
        {"name": "848x480@10Hz", "width": 848, "height": 480, "fps": 10},
        {"name": "1280x720@6Hz", "width": 1280, "height": 720, "fps": 6},
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        
        try:
            # Create pipeline and config
            pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Enable only depth stream (no color to keep it simple)
            rs_config.enable_stream(
                rs.stream.depth, 
                config['width'], 
                config['height'], 
                rs.format.z16, 
                config['fps']
            )
            
            # Try to start
            print(f"Starting pipeline: {config['width']}x{config['height']} @ {config['fps']}Hz")
            profile = pipeline.start(rs_config)
            print("✓ Pipeline started successfully")
            
            # Get a few frames to make sure it's working
            frames_captured = 0
            for i in range(5):
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                depth_frame = frames.get_depth_frame()
                
                if depth_frame:
                    frames_captured += 1
                    depth_data = np.asanyarray(depth_frame.get_data())
                    print(f"  Frame {i+1}: {depth_data.shape}, min={depth_data.min()}, max={depth_data.max()}")
                else:
                    print(f"  Frame {i+1}: No depth data")
                
                time.sleep(0.1)
            
            pipeline.stop()
            print(f"✓ Successfully captured {frames_captured}/5 frames")
            
            if frames_captured >= 3:
                print(f"*** {config['name']} WORKS WELL ***")
                return config
            else:
                print(f"! {config['name']} partially works but unreliable")
                
        except Exception as e:
            print(f"✗ {config['name']} failed: {e}")
            try:
                pipeline.stop()
            except:
                pass
        
        time.sleep(1)  # Brief pause between tests
    
    return None


def main():
    """Main test function."""
    print("RealSense Camera Minimal Test")
    print("=" * 40)
    
    # First, try to reset the camera
    print("Step 1: Reset camera")
    reset_success = reset_camera()
    if not reset_success:
        print("Warning: Camera reset failed, continuing anyway...")
    
    print("\nStep 2: Test camera settings")
    working_config = test_camera_settings()
    
    if working_config:
        print(f"\n🎉 Found working configuration: {working_config['name']}")
        print("You can use these settings in your application:")
        print(f"  Width: {working_config['width']}")
        print(f"  Height: {working_config['height']}")
        print(f"  FPS: {working_config['fps']}")
    else:
        print("\n❌ No working configuration found")
        print("Try the following:")
        print("1. Unplug and replug the camera")
        print("2. Restart the computer")
        print("3. Check for conflicting processes using the camera")
    
    print("\nTest complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
List available webcams and test their capabilities.

This script helps identify which webcams are available on the system
and tests their basic functionality for use with the Experimance Agent Service.
"""

import cv2
import sys
from pathlib import Path

def list_webcams():
    """List all available webcam devices and their capabilities."""
    print("Scanning for available webcams...")
    print("=" * 60)
    
    available_cameras = []
    
    # Test camera indices 0-9 (covers most systems)
    for camera_id in range(10):
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            # Get camera name if possible
            camera_name = f"Camera {camera_id}"
            
            # Try to capture a frame to verify it's working
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✓ Camera {camera_id}: {camera_name}")
                print(f"    Resolution: {width}x{height}")
                print(f"    FPS: {fps}")
                print(f"    Backend: {cap.getBackendName()}")
                
                # Test if we can set different resolutions
                test_resolutions = [(640, 480), (1280, 720), (1920, 1080)]
                supported_resolutions = []
                
                for test_w, test_h in test_resolutions:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_h)
                    
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if actual_w == test_w and actual_h == test_h:
                        supported_resolutions.append(f"{test_w}x{test_h}")
                
                if supported_resolutions:
                    print(f"    Supported resolutions: {', '.join(supported_resolutions)}")
                else:
                    print(f"    Resolution testing failed - may support custom resolutions")
                
                available_cameras.append({
                    'id': camera_id,
                    'name': camera_name,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': cap.getBackendName()
                })
                
                print()
            else:
                print(f"✗ Camera {camera_id}: Detected but cannot capture frames")
                print()
        
        cap.release()
    
    print("=" * 60)
    
    if available_cameras:
        print(f"Found {len(available_cameras)} working camera(s)")
        print("\nRecommended configuration:")
        
        # Recommend the first working camera
        recommended = available_cameras[0]
        print(f"""
[agent.vision]
webcam_enabled = true
webcam_device_id = {recommended['id']}  # {recommended['name']}
webcam_width = {recommended['width']}
webcam_height = {recommended['height']}
webcam_fps = 30
""")
        
        if len(available_cameras) > 1:
            print("Alternative cameras:")
            for cam in available_cameras[1:]:
                print(f"  Device ID {cam['id']}: {cam['name']} ({cam['width']}x{cam['height']})")
    else:
        print("No working cameras found!")
        print("\nTroubleshooting tips:")
        print("- Check that your camera is connected and not in use by another application")
        print("- Try running as administrator/sudo if on Linux")
        print("- Install camera drivers if needed")
        print("- Check system privacy settings (camera permissions)")
    
    return available_cameras


def test_camera_capture(camera_id, duration=5):
    """Test continuous capture from a specific camera."""
    print(f"Testing camera {camera_id} for {duration} seconds...")
    print("Press 'q' to quit early or wait for automatic stop")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Cannot open camera {camera_id}")
        return False
    
    # Set up capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = cv2.getTickCount()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Show frame (optional - comment out if running headless)
            try:
                cv2.imshow(f'Camera {camera_id} Test', frame)
                
                # Check for 'q' key press or timeout
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopped by user")
                    break
            except:
                # Running headless, just count frames
                pass
            
            # Check timeout
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed >= duration:
                break
    
    except KeyboardInterrupt:
        print("Stopped by user (Ctrl+C)")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Calculate stats
    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"✓ Captured {frame_count} frames in {elapsed:.1f} seconds")
    print(f"✓ Actual FPS: {actual_fps:.1f}")
    
    return True


def main():
    """Main function to list cameras and optionally test one."""
    print("Experimance Agent - Webcam Detection Tool")
    print("=" * 60)
    
    # List all available cameras
    cameras = list_webcams()
    
    # If cameras found, offer to test one
    if cameras and len(sys.argv) > 1:
        try:
            test_id = int(sys.argv[1])
            if any(cam['id'] == test_id for cam in cameras):
                print(f"\nTesting camera {test_id}...")
                test_camera_capture(test_id)
            else:
                print(f"Camera {test_id} not found in available cameras")
        except ValueError:
            print("Invalid camera ID provided")
    elif cameras:
        print(f"\nTo test a specific camera, run:")
        print(f"python {Path(__file__).name} <camera_id>")
        print(f"Example: python {Path(__file__).name} 0")


if __name__ == "__main__":
    main()

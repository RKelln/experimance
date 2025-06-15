#!/usr/bin/env python3
"""
RealSense Camera Diagnostic and Recovery Tool

This script provides comprehensive diagnostics and recovery options for RealSense cameras
that are stuck in "Device or resource busy" or other error states.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experimance_core.robust_camera import (
    get_camera_diagnostics, 
    reset_realsense_camera,
    kill_camera_processes,
    usb_reset_device
)
import pyrealsense2 as rs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_diagnostics():
    """Print comprehensive camera diagnostics."""
    print("🔍 RealSense Camera Diagnostics")
    print("=" * 50)
    
    diagnostics = get_camera_diagnostics()
    
    # RealSense devices
    print(f"\n📷 RealSense Devices ({len(diagnostics['devices'])} found):")
    if diagnostics['devices']:
        for i, device in enumerate(diagnostics['devices']):
            print(f"  Device {i}:")
            print(f"    Name: {device['name']}")
            print(f"    Serial: {device['serial']}")
            print(f"    Firmware: {device['firmware']}")
            print(f"    USB Type: {device['usb_type']}")
            print(f"    Sensors: {len(device['sensors'])}")
            for j, sensor in enumerate(device['sensors']):
                print(f"      Sensor {j}: {sensor['name']} ({len(sensor['profiles'])} profiles)")
    else:
        print("  ❌ No RealSense devices found")
    
    # USB devices
    print(f"\n🔌 USB Devices (Intel/RealSense):")
    if diagnostics['usb_devices']:
        for device in diagnostics['usb_devices']:
            print(f"  {device}")
    else:
        print("  ❌ No Intel USB devices found")
    
    # Processes
    print(f"\n🔄 Potentially Interfering Processes ({len(diagnostics['processes'])} found):")
    if diagnostics['processes']:
        for proc in diagnostics['processes']:
            if 'error' in proc:
                print(f"  ⚠️  Error: {proc['error']}")
            else:
                print(f"  PID {proc['pid']}: {proc['name']}")
                if proc['cmdline']:
                    print(f"    Command: {proc['cmdline']}")
    else:
        print("  ✅ No interfering processes found")


def run_rs_enumerate():
    """Run rs-enumerate to get detailed device information."""
    print("\n🔧 Running rs-enumerate...")
    print("-" * 30)
    
    try:
        result = subprocess.run(['rs-enumerate'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ rs-enumerate failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ rs-enumerate not found. Install Intel RealSense SDK tools.")
    except subprocess.TimeoutExpired:
        print("❌ rs-enumerate timed out")
    except Exception as e:
        print(f"❌ Error running rs-enumerate: {e}")


def run_lsusb():
    """Run lsusb to show USB device information."""
    print("\n🔧 USB Device Information (lsusb)...")
    print("-" * 40)
    
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Filter for Intel devices
            intel_devices = [line for line in result.stdout.split('\n') 
                           if 'intel' in line.lower() or '8086' in line]
            if intel_devices:
                for device in intel_devices:
                    print(f"  {device}")
            else:
                print("  ❌ No Intel USB devices found")
        else:
            print(f"❌ lsusb failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ lsusb not found")
    except Exception as e:
        print(f"❌ Error running lsusb: {e}")


def run_dmesg_camera():
    """Check dmesg for camera-related messages."""
    print("\n🔧 Recent kernel messages (camera-related)...")
    print("-" * 50)
    
    try:
        result = subprocess.run(['dmesg', '-T'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Filter for camera/USB related messages
            camera_msgs = []
            for line in result.stdout.split('\n'):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in 
                       ['realsense', 'intel', 'usb', 'camera', 'uvc', 'video']):
                    camera_msgs.append(line)
            
            # Show last 10 relevant messages
            for msg in camera_msgs[-10:]:
                print(f"  {msg}")
                
            if not camera_msgs:
                print("  ℹ️  No recent camera-related kernel messages")
        else:
            print(f"❌ dmesg failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running dmesg: {e}")


def test_basic_connection():
    """Test basic RealSense connection."""
    print("\n🔧 Testing Basic RealSense Connection...")
    print("-" * 40)
    
    try:
        # Test context creation
        ctx = rs.context()
        devices = ctx.query_devices()
        
        print(f"✅ RealSense context created successfully")
        print(f"✅ Found {len(devices)} devices")
        
        if len(devices) == 0:
            print("❌ No devices found - check connections")
            return False
        
        # Test device access
        for i, dev in enumerate(devices):
            try:
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                print(f"✅ Device {i}: {name} (Serial: {serial})")
                
                # Test sensor enumeration
                sensors = dev.query_sensors()
                print(f"✅ Found {len(sensors)} sensors")
                
            except Exception as e:
                print(f"❌ Error accessing device {i}: {e}")
                return False
        
        # Test pipeline creation (but don't start)
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            print("✅ Pipeline configuration successful")
            
            # Test pipeline start/stop
            profile = pipeline.start(config)
            print("✅ Pipeline started successfully")
            
            # Try to get one frame
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            
            if depth_frame:
                print("✅ Frame capture successful")
            else:
                print("⚠️  Frame capture returned no depth frame")
            
            pipeline.stop()
            print("✅ Pipeline stopped successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Basic connection test failed: {e}")
        return False


def recovery_sequence():
    """Run comprehensive recovery sequence."""
    print("\n🚀 Starting Recovery Sequence...")
    print("=" * 40)
    
    # Step 1: Kill interfering processes
    print("\n1. Killing interfering processes...")
    if kill_camera_processes():
        print("✅ Killed camera processes")
        time.sleep(2)
    else:
        print("ℹ️  No processes to kill")
    
    # Step 2: Hardware reset
    print("\n2. Hardware reset...")
    if reset_realsense_camera(aggressive=False):
        print("✅ Hardware reset successful")
    else:
        print("❌ Hardware reset failed")
    
    # Step 3: Test connection
    print("\n3. Testing connection...")
    if test_basic_connection():
        print("✅ Recovery successful!")
        return True
    
    # Step 4: Aggressive recovery
    print("\n4. Attempting aggressive recovery...")
    if reset_realsense_camera(aggressive=True):
        print("✅ Aggressive reset successful")
        
        # Final test
        time.sleep(3)
        if test_basic_connection():
            print("✅ Aggressive recovery successful!")
            return True
    
    print("❌ Recovery failed - manual intervention required")
    print("\nTry:")
    print("1. Unplug and replug the USB cable")
    print("2. Restart the computer")
    print("3. Check USB cable and connections")
    print("4. Try a different USB port")
    
    return False


def main():
    parser = argparse.ArgumentParser(description="RealSense Camera Diagnostic and Recovery Tool")
    parser.add_argument('--diagnostics', '-d', action='store_true', help='Show comprehensive diagnostics')
    parser.add_argument('--enumerate', '-e', action='store_true', help='Run rs-enumerate')
    parser.add_argument('--lsusb', '-l', action='store_true', help='Show USB devices')
    parser.add_argument('--dmesg', '-m', action='store_true', help='Show kernel messages')
    parser.add_argument('--test', '-t', action='store_true', help='Test basic connection')
    parser.add_argument('--recover', '-r', action='store_true', help='Run recovery sequence')
    parser.add_argument('--all', '-a', action='store_true', help='Run all diagnostics and recovery')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        args.all = True
    
    print("🎯 RealSense Camera Debug Tool")
    print("=" * 50)
    
    if args.all or args.diagnostics:
        print_diagnostics()
    
    if args.all or args.enumerate:
        run_rs_enumerate()
    
    if args.all or args.lsusb:
        run_lsusb()
    
    if args.all or args.dmesg:
        run_dmesg_camera()
    
    if args.all or args.test:
        test_basic_connection()
    
    if args.all or args.recover:
        recovery_sequence()


if __name__ == "__main__":
    main()

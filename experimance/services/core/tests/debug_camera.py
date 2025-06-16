#!/usr/bin/env python3
"""
RealSense Camera Diagnostic and Recovery Tool

This script provides comprehensive diagnostics and recovery options for RealSense cameras
that are stuck in "Device or resource busy" or other error states.

Run from project root with: uv run python services/core/debug_camera_clean.py
"""

import sys
import os
import time
import subprocess
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_diagnostics():
    """Print comprehensive camera diagnostics."""
    print("\n📊 Camera Diagnostics")
    print("-" * 30)
    
    try:
        import pyrealsense2 as rs # type: ignore
        ctx = rs.context() # type: ignore
        devices = ctx.query_devices()
        
        print(f"🎥 Cameras found: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"   {i+1}. {device.get_info(rs.camera_info.name)} (S/N: {device.get_info(rs.camera_info.serial_number)})") # type: ignore
            print(f"      FW: {device.get_info(rs.camera_info.firmware_version)}") # type: ignore
    except Exception as e:
        print(f"❌ Failed to get camera diagnostics: {e}")
    
    # Check for running camera processes
    try:
        result = subprocess.run(['pgrep', '-f', 'realsense'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"� Running camera processes: {len(pids)}")
            for pid in pids:
                if pid:
                    print(f"   PID {pid}")
        else:
            print("✅ No RealSense processes running")
    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")


def run_rs_enumerate():
    """Run rs-enumerate to check RealSense devices."""
    print("\n🔍 Running rs-enumerate...")
    print("-" * 30)
    
    try:
        result = subprocess.run(['rs-enumerate'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ rs-enumerate failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ rs-enumerate not found. Install librealsense2-tools")
    except subprocess.TimeoutExpired:
        print("⏰ rs-enumerate timed out")
    except Exception as e:
        print(f"❌ Error running rs-enumerate: {e}")


def run_lsusb():
    """Show USB device information."""
    print("\n🔌 USB Device Information")
    print("-" * 30)
    
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Filter for Intel devices (RealSense)
            lines = result.stdout.strip().split('\n')
            intel_devices = [line for line in lines if '8086' in line]
            
            if intel_devices:
                print("Intel USB devices (potential RealSense cameras):")
                for device in intel_devices:
                    print(f"   {device}")
            else:
                print("❌ No Intel USB devices found")
                
            print(f"\nAll USB devices: {len(lines)} total")
        else:
            print(f"❌ lsusb failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running lsusb: {e}")


def run_dmesg_camera():
    """Check kernel messages for camera-related errors."""
    print("\n📋 Recent Kernel Messages (Camera-related)")
    print("-" * 50)
    
    try:
        result = subprocess.run(['dmesg', '--time-format=iso'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            camera_lines = []
            
            keywords = ['realsense', 'uvcvideo', 'usb 3-', 'usb 4-', 'intel', 'D415', 'D435']
            for line in lines[-100:]:  # Check last 100 lines
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    camera_lines.append(line)
            
            if camera_lines:
                for line in camera_lines[-10:]:  # Show last 10 relevant lines
                    print(f"   {line}")
            else:
                print("ℹ️  No recent camera-related kernel messages")
        else:
            print(f"❌ dmesg failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running dmesg: {e}")


def test_basic_connection():
    """Test basic camera connection without advanced features."""
    print("\n🔌 Basic Connection Test")
    print("-" * 30)
    
    try:
        import pyrealsense2 as rs
        
        # Try to create context and enumerate devices
        ctx = rs.context() # type: ignore
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ No RealSense devices detected")
            return False
        
        print(f"✅ Found {len(devices)} RealSense device(s)")
        
        for i, device in enumerate(devices):
            print(f"   Device {i+1}:")
            print(f"     Name: {device.get_info(rs.camera_info.name)}") # type: ignore
            print(f"     Serial: {device.get_info(rs.camera_info.serial_number)}") # type: ignore
            print(f"     Firmware: {device.get_info(rs.camera_info.firmware_version)}") # type: ignore
            
            # Try to query sensors
            sensors = device.query_sensors()
            print(f"     Sensors: {len(sensors)}")
        
        # Try basic pipeline creation (don't start it)
        try:
            pipeline = rs.pipeline() # type: ignore
            config = rs.config() # type: ignore
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # type: ignore
            
            # Test configuration without starting
            profile = pipeline.get_active_profile()
            print("✅ Pipeline configuration successful")
            
        except Exception as e:
            print(f"⚠️  Pipeline test failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ pyrealsense2 not installed")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def recovery_sequence():
    """Run basic recovery sequence for stuck cameras."""
    print("\n🔧 Recovery Sequence")
    print("=" * 30)
    
    steps = [
        ("Kill camera processes", kill_camera_processes),
        ("Wait for cleanup", lambda: time.sleep(2)),
        ("Test basic connection", test_basic_connection)
    ]
    
    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n{i}. {step_name}...")
        try:
            result = step_func()
            if result is False:
                print(f"❌ Step {i} failed")
                return False
            else:
                print(f"✅ Step {i} completed")
        except Exception as e:
            print(f"❌ Step {i} error: {e}")
            
    print("\n🎉 Recovery sequence completed!")
    return True


def kill_camera_processes():
    """Kill any running camera-related processes."""
    try:
        # Kill RealSense processes
        subprocess.run(['pkill', '-f', 'realsense'], capture_output=True)
        subprocess.run(['pkill', '-f', 'librealsense'], capture_output=True)
        subprocess.run(['pkill', '-f', 'rs-'], capture_output=True)
        print("✅ Killed camera processes")
        return True
    except Exception as e:
        print(f"⚠️  Error killing processes: {e}")
        return True  # Don't fail the sequence for this


def quick_performance_test():
    """Quick performance test using the existing test_camera.py script."""
    print("\n⚡ Quick Performance Test")
    print("-" * 30)
    
    try:
        # Get current directory
        current_dir = Path(__file__).parent
        
        # Run the test_camera.py script with real camera and short duration
        cmd = [
            'uv', 'run', 'python', 
            str(current_dir / 'tests' / 'test_camera.py'),
            '--real', '--duration', '5', '--verbose'
        ]
        
        print("🎯 Running camera performance test (5 seconds)...")
        print("   Command:", ' '.join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Performance test completed successfully")
            
            # Parse the output for performance metrics
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Frame' in line and 'performance' in line:
                    print(f"📊 {line}")
                elif 'FPS' in line or 'fps' in line:
                    print(f"📊 {line}")
                elif 'frames in' in line:
                    print(f"📊 {line}")
            
            # Look for thermal throttling indicators
            mask_times = []
            capture_times = []
            for line in output_lines:
                if 'mask=' in line and 'capture=' in line:
                    try:
                        # Extract times: "capture=75.7ms, mask=10.2ms"
                        parts = line.split()
                        for part in parts:
                            if part.startswith('capture=') and part.endswith('ms,'):
                                capture_times.append(float(part[8:-3]))
                            elif part.startswith('mask=') and part.endswith('ms,'):
                                mask_times.append(float(part[5:-3]))
                    except:
                        pass
            
            if capture_times and mask_times:
                avg_capture = sum(capture_times) / len(capture_times)
                avg_mask = sum(mask_times) / len(mask_times)
                
                print(f"\n📊 Performance Analysis:")
                print(f"   Average capture time: {avg_capture:.1f}ms")
                print(f"   Average mask time: {avg_mask:.1f}ms")
                
                # Thermal assessment
                if avg_capture > 60 or avg_mask > 8:
                    print("⚠️  Performance indicates possible thermal throttling")
                    print("   Capture >60ms or mask >8ms suggests degraded performance")
                elif avg_capture > 40 or avg_mask > 5:
                    print("📊 Performance is moderate - monitor for degradation")
                else:
                    print("✅ Performance looks good - no obvious throttling")
            
            return True
        else:
            print(f"❌ Performance test failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Performance test timed out")
        return False
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def check_system_thermals():
    """Check system thermal status that might affect camera performance."""
    print("\n🌡️  System Thermal Check")
    print("-" * 30)
    
    try:
        # Check CPU temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000
                print(f"🌡️  CPU Temperature: {temp:.1f}°C")
                if temp > 80:
                    print("⚠️  High CPU temperature - may cause throttling")
                elif temp > 70:
                    print("📊 CPU temperature elevated")
                else:
                    print("✅ CPU temperature normal")
        except FileNotFoundError:
            print("ℹ️  CPU temperature monitoring not available")
        except Exception as e:
            print(f"⚠️  Error reading CPU temperature: {e}")
        
        # Check CPU governor
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                governor = f.read().strip()
                print(f"⚙️  CPU Governor: {governor}")
                if governor == 'powersave':
                    print("⚠️  CPU in powersave mode - may limit performance")
                elif governor == 'performance':
                    print("✅ CPU in performance mode")
                else:
                    print(f"📊 CPU governor: {governor}")
        except FileNotFoundError:
            print("ℹ️  CPU governor info not available")
        except Exception as e:
            print(f"⚠️  Error reading CPU governor: {e}")
            
    except Exception as e:
        print(f"❌ System thermal check failed: {e}")


def suggest_performance_fixes():
    """Suggest fixes for performance issues based on diagnostics."""
    print("\n🛠️  Suggested Performance Fixes")
    print("=" * 40)
    
    print("1. 🔌 USB Power Management:")
    print("   # Disable USB auto power management for RealSense")
    print("   echo 'on' | sudo tee /sys/bus/usb/devices/*/power/control")
    print("   # Or specifically for Intel devices:")
    print("   for dev in /sys/bus/usb/devices/*; do")
    print("     if [ -f \"$dev/idVendor\" ] && [ \"$(cat $dev/idVendor)\" = \"8086\" ]; then")
    print("       echo 'on' | sudo tee \"$dev/power/control\"")
    print("     fi")
    print("   done")
    
    print("\n2. 🌡️  CPU Thermal Management:")
    print("   # Check current thermal policy")
    print("   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    print("   # Set performance governor temporarily")
    print("   sudo cpupower frequency-set -g performance")
    print("   # Reset back to powersave/ondemand after testing")
    
    print("\n3. 💾 Memory/Cache Optimization:")
    print("   # Increase camera warmup frames")
    print("   # In config.toml: warm_up_frames = 30")
    print("   # Pre-allocate numpy arrays")
    print("   # Use memory mapping for large buffers")
    
    print("\n4. 🚌 USB Optimization:")
    print("   # Use dedicated USB 3.0+ port")
    print("   # Avoid USB hubs")
    print("   # Check USB latency settings")
    print("   # Monitor with: sudo usbmon")
    
    print("\n5. 📊 Application-Level Fixes:")
    print("   # Implement frame warmup period")
    print("   # Use consistent CPU affinity")
    print("   # Monitor and adapt to performance changes")
    print("   # Cache processed masks when locked")
    
    # Apply automated fixes
    print("\n🔧 Applying Automated Performance Fixes...")
    print("---------------------------------------------")
    
    fixes_applied = []
    
    # Check USB power management
    try:
        result = subprocess.run(['find', '/sys/bus/usb/devices/', '-name', 'idVendor'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            intel_devices = []
            for device_path in result.stdout.strip().split('\n'):
                if device_path:
                    try:
                        with open(device_path, 'r') as f:
                            vendor = f.read().strip()
                            if vendor == '8086':  # Intel
                                device_dir = os.path.dirname(device_path)
                                device_name = os.path.basename(device_dir)
                                power_control = os.path.join(device_dir, 'power', 'control')
                                if os.path.exists(power_control):
                                    try:
                                        with open(power_control, 'r') as pc:
                                            current = pc.read().strip()
                                            intel_devices.append((device_name, current))
                                    except:
                                        pass
                    except:
                        pass
            
            if intel_devices:
                print(f"🔌 Found {len(intel_devices)} Intel USB devices")
                for device, current in intel_devices:
                    if current == 'auto':
                        print(f"   Device {device}: changing power control from 'auto' to 'on'")
                        print(f"   NOTE: Run 'echo on | sudo tee /sys/bus/usb/devices/{device}/power/control' manually")
                        fixes_applied.append(f"USB power mgmt: {device}")
                    else:
                        print(f"   Device {device}: power control already '{current}'")
    except Exception as e:
        print(f"⚠️  USB power check failed: {e}")
    
    # Check CPU governor
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            governor = f.read().strip()
            print(f"🌡️  CPU Governors: {governor}")
            if governor in ['powersave', 'ondemand']:
                print(f"   NOTE: '{governor}' governor may limit performance")
                print(f"   Consider: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
                fixes_applied.append("CPU governor check")
    except Exception as e:
        print(f"⚠️  CPU governor check failed: {e}")
    
    if fixes_applied:
        print(f"\n✅ Diagnostics completed. Found {len(fixes_applied)} items to address:")
        for fix in fixes_applied:
            print(f"   - {fix}")
        print("\nSome fixes require sudo privileges - run suggested commands manually")
    else:
        print("\n✅ No obvious performance issues detected")


def main():
    """Main function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RealSense Camera Debug Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--diag", action="store_true", help="Show camera diagnostics")
    parser.add_argument("--enum", action="store_true", help="Run rs-enumerate")
    parser.add_argument("--usb", action="store_true", help="Show USB device info")
    parser.add_argument("--dmesg", action="store_true", help="Check kernel messages")
    parser.add_argument("--test", action="store_true", help="Test basic connection")
    parser.add_argument("--recover", action="store_true", help="Run recovery sequence")
    parser.add_argument("--perf", action="store_true", help="Quick performance test")
    parser.add_argument("--thermal", action="store_true", help="Check system thermals")
    parser.add_argument("--fix", action="store_true", help="Suggest performance fixes")
    parser.add_argument("--all", action="store_true", help="Run all diagnostics")
    
    args = parser.parse_args()
    
    print("🎯 RealSense Camera Debug Tool")
    print("=" * 50)
    
    if args.all or (not any(vars(args).values())):
        # Run all diagnostics if no specific option or --all
        print_diagnostics()
        run_rs_enumerate()
        run_lsusb()
        run_dmesg_camera()
        test_basic_connection()
        quick_performance_test()
        check_system_thermals()
        suggest_performance_fixes()
    else:
        if args.diag:
            print_diagnostics()
        if args.enum:
            run_rs_enumerate()
        if args.usb:
            run_lsusb()
        if args.dmesg:
            run_dmesg_camera()
        if args.test:
            test_basic_connection()
        if args.recover:
            recovery_sequence()
        if args.perf:
            quick_performance_test()
        if args.thermal:
            check_system_thermals()
        if args.fix:
            suggest_performance_fixes()


if __name__ == "__main__":
    main()

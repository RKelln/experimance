#!/usr/bin/env python3
"""
Audio diagnostic and recovery script for the Experimance project.

This script helps diagnose and recover from audio issues, particularly with USB audio devices.
"""

import logging
import sys
import argparse
from pathlib import Path

# Add the common library to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "common" / "src"))

from experimance_common.audio_utils import (
    list_audio_devices, reset_audio_device_by_name, cleanup_audio_resources,
    force_audio_system_reset, validate_audio_device_index
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_devices():
    """List all available audio devices."""
    print("\n=== Available Audio Devices ===")
    devices = list_audio_devices()
    
    if not devices:
        print("No audio devices found or error occurred")
        return
    
    for device in devices:
        input_str = f"IN:{device['max_input_channels']}" if device['max_input_channels'] > 0 else "IN:0"
        output_str = f"OUT:{device['max_output_channels']}" if device['max_output_channels'] > 0 else "OUT:0"
        print(f"[{device['index']:2d}] {device['name']} ({input_str}, {output_str}) @ {device['default_sample_rate']}Hz")
    
    print(f"\nTotal: {len(devices)} devices")


def test_yealink_device():
    """Test specifically for Yealink devices."""
    print("\n=== Yealink Device Test ===")
    devices = list_audio_devices()
    
    yealink_devices = [d for d in devices if 'yealink' in d['name'].lower()]
    
    if not yealink_devices:
        print("No Yealink devices found")
        return False
    
    for device in yealink_devices:
        print(f"Found Yealink device: [{device['index']}] {device['name']}")
        input_ok = device['max_input_channels'] > 0
        output_ok = device['max_output_channels'] > 0
        print(f"  Input channels: {device['max_input_channels']} {'✓' if input_ok else '✗'}")
        print(f"  Output channels: {device['max_output_channels']} {'✓' if output_ok else '✗'}")
        print(f"  Sample rate: {device['default_sample_rate']}Hz")
        
        if input_ok and output_ok:
            print("  Status: Device appears functional ✓")
            return True
        else:
            print("  Status: Device may have issues ✗")
    
    return False


def reset_yealink():
    """Attempt to reset Yealink USB audio device."""
    print("\n=== Attempting Yealink Device Reset ===")
    success = reset_audio_device_by_name("Yealink")
    
    if success:
        print("Reset attempt completed. Please wait a few seconds for the device to reinitialize.")
        print("You may need to restart the agent service after this.")
        
        # Test the device after reset
        print("\nTesting device after reset...")
        import time
        time.sleep(3)
        test_yealink_device()
    else:
        print("Reset attempt failed or not available.")
        print("Manual recovery options:")
        print("1. Unplug and replug the USB device")
        print("2. Restart audio services: sudo systemctl restart alsa-state")
        print("3. Kill and restart PulseAudio: pulseaudio --kill && pulseaudio --start")
        print("4. Reboot the system if issues persist")
    
    return success


def force_reset():
    """Perform comprehensive audio system reset."""
    print("\n=== Comprehensive Audio System Reset ===")
    print("This will attempt multiple reset methods...")
    
    success = force_audio_system_reset()
    
    if success:
        print("Audio system reset completed.")
        print("Testing devices after reset...")
        import time
        time.sleep(3)
        list_devices()
    else:
        print("Audio system reset had limited success.")
        print("Consider manual intervention or system reboot.")
    
    return success


def validate_device():
    """Validate that device index 6 (Yealink) is accessible."""
    print("\n=== Validating Device Index 6 ===")
    
    if validate_audio_device_index(6):
        print("✓ Device index 6 is valid and accessible")
        return True
    else:
        print("✗ Device index 6 is not accessible")
        print("This may indicate the device is stuck or has changed index")
        return False


def main():
    parser = argparse.ArgumentParser(description='Audio diagnostic and recovery tool')
    parser.add_argument('action', choices=['list', 'test', 'reset', 'force-reset', 'validate'], 
                       help='Action to perform: list devices, test Yealink, reset Yealink, force comprehensive reset, or validate device index')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'list':
            list_devices()
        elif args.action == 'test':
            test_yealink_device()
        elif args.action == 'reset':
            reset_yealink()
        elif args.action == 'force-reset':
            force_reset()
        elif args.action == 'validate':
            validate_device()
            
    except Exception as e:
        logger.error(f"Error during {args.action}: {e}")
        return 1
    finally:
        # Clean up audio resources
        cleanup_audio_resources()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

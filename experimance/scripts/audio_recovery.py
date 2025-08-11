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


def stop_audio_services() -> bool:
    """
    Stop all audio services to prepare for device reset.
    
    This stops PipeWire, JACK, and other audio services to ensure
    clean device reinitialization.
    
    Returns:
        True if services were stopped successfully
    """
    try:
        import subprocess
        import time
        
        logger.info("Stopping audio services for clean device reset...")
        success = False
        
        # Stop user-level PipeWire services
        try:
            logger.info("Stopping PipeWire services...")
            subprocess.run(['systemctl', '--user', 'stop', 
                          'pipewire-pulse.socket', 'pipewire.socket', 
                          'pipewire-pulse', 'pipewire'], 
                         capture_output=True, timeout=10)
            time.sleep(1)
            success = True
            logger.info("PipeWire services stopped")
        except Exception as e:
            logger.debug(f"PipeWire stop failed: {e}")
        
        # Stop JACK
        try:
            logger.info("Stopping JACK...")
            subprocess.run(['jack_control', 'stop'], 
                         capture_output=True, timeout=10)
            time.sleep(1)
            success = True
            logger.info("JACK stopped")
        except Exception as e:
            logger.debug(f"JACK stop failed: {e}")
        
        # Kill any remaining audio processes
        try:
            logger.info("Cleaning up remaining audio processes...")
            subprocess.run(['pkill', '-f', 'pulseaudio'], 
                         capture_output=True, timeout=5)
            subprocess.run(['pkill', '-f', 'pipewire'], 
                         capture_output=True, timeout=5)
            time.sleep(1)
        except Exception as e:
            logger.debug(f"Process cleanup failed: {e}")
        
        if success:
            logger.info("Audio services stopped successfully")
        else:
            logger.warning("Audio service stop had limited success")
            
        return success
        
    except Exception as e:
        logger.error(f"Error stopping audio services: {e}")
        return False


def restart_audio_services() -> bool:
    """
    Restart audio services after device reset.
    
    Returns:
        True if services were restarted successfully
    """
    try:
        import subprocess
        import time
        
        logger.info("Restarting audio services...")
        success = False
        
        # Start JACK first
        try:
            logger.info("Starting JACK...")
            subprocess.run(['jack_control', 'start'], 
                         capture_output=True, timeout=10)
            time.sleep(2)
            success = True
            logger.info("JACK started")
        except Exception as e:
            logger.debug(f"JACK start failed: {e}")
        
        # Start PipeWire services
        try:
            logger.info("Starting PipeWire services...")
            subprocess.run(['systemctl', '--user', 'start', 
                          'pipewire.socket', 'pipewire-pulse.socket'], 
                         capture_output=True, timeout=10)
            time.sleep(2)
            success = True
            logger.info("PipeWire services started")
        except Exception as e:
            logger.debug(f"PipeWire start failed: {e}")
        
        if success:
            logger.info("Audio services restarted successfully")
        else:
            logger.warning("Audio service restart had limited success")
            
        return success
        
    except Exception as e:
        logger.error(f"Error restarting audio services: {e}")
        return False


def comprehensive_usb_audio_reset(device_name: str) -> bool:
    """
    Comprehensive USB audio device reset including audio service management.
    
    This function:
    1. Stops all audio services
    2. Attempts USB device reset via multiple methods
    3. Restarts audio services
    
    Args:
        device_name: Name of the USB audio device to reset
        
    Returns:
        True if reset was successful
    """
    logger.info(f"Starting comprehensive reset for USB audio device: {device_name}")
    
    # Step 1: Stop audio services
    logger.info("Step 1: Stopping audio services...")
    stop_audio_services()
    
    # Step 2: Reset the USB device
    logger.info("Step 2: Resetting USB device...")
    device_reset_success = reset_audio_device_by_name(device_name)
    
    # Step 3: Give device time to reinitialize
    logger.info("Step 3: Waiting for device reinitialization...")
    import time
    time.sleep(3)
    
    # Step 4: Restart audio services
    logger.info("Step 4: Restarting audio services...")
    service_restart_success = restart_audio_services()
    
    # Step 5: Additional wait for full system initialization
    logger.info("Step 5: Final initialization wait...")
    time.sleep(2)
    
    success = device_reset_success or service_restart_success
    
    if success:
        logger.info(f"Comprehensive reset for {device_name} completed successfully")
    else:
        logger.warning(f"Comprehensive reset for {device_name} had limited success")
    
    return success


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


def test_icusbaudio7d_device():
    """Test specifically for ICUSBAUDIO7D devices."""
    print("\n=== ICUSBAUDIO7D Device Test ===")
    devices = list_audio_devices()
    
    icusb_devices = [d for d in devices if 'icusbaudio7d' in d['name'].lower()]
    
    if not icusb_devices:
        print("No ICUSBAUDIO7D devices found")
        return False
    
    for device in icusb_devices:
        print(f"Found ICUSBAUDIO7D device: [{device['index']}] {device['name']}")
        input_channels = device['max_input_channels']
        output_channels = device['max_output_channels']
        print(f"  Input channels: {input_channels}")
        print(f"  Output channels: {output_channels}")
        print(f"  Sample rate: {device['default_sample_rate']}Hz")
        
        # For a 5.1 surround sound device, we expect 6 output channels minimum
        expected_outputs = 6
        output_ok = output_channels >= expected_outputs
        
        print(f"  Expected output channels: {expected_outputs} (5.1 surround)")
        
        if output_ok:
            print("  Status: Device appears functional ✓")
            print("  ✓ Sufficient channels for 5.1 surround sound")
            return True
        else:
            print("  Status: Device has issues ✗")
            print(f"  ✗ Only {output_channels} output channels detected (need {expected_outputs} for 5.1)")
            print("  This indicates the device may be stuck in a limited mode or needs reset")
    
    return False


def reset_icusbaudio7d():
    """Attempt to reset ICUSBAUDIO7D USB audio device with comprehensive approach."""
    print("\n=== Comprehensive ICUSBAUDIO7D Device Reset ===")
    print("This will stop audio services, reset the device, and restart services.")
    
    success = comprehensive_usb_audio_reset("ICUSBAUDIO7D")
    
    if success:
        print("Comprehensive reset completed!")
        print("The device should now show proper 5.1 surround sound capabilities (6+ output channels).")
        
        # Test the device after reset
        print("\nTesting device after comprehensive reset...")
        import time
        time.sleep(5)  # Give more time for full system reinitialization
        test_icusbaudio7d_device()
    else:
        print("Comprehensive reset had limited success.")
        print("Manual recovery options for ICUSBAUDIO7D:")
        print("1. Unplug and replug the USB device")
        print("2. Check USB power management: sudo sh -c 'echo on > /sys/bus/usb/devices/*/power/control'")
        print("3. Try a different USB port (preferably USB 3.0)")
        print("4. Restart the entire system")
        print("5. Check if device firmware needs updating")
    
    return success


def reset_yealink():
    """Attempt to reset Yealink USB audio device with comprehensive approach."""
    print("\n=== Comprehensive Yealink Device Reset ===")
    print("This will stop audio services, reset the device, and restart services.")
    
    success = comprehensive_usb_audio_reset("Yealink")
    
    if success:
        print("Comprehensive reset completed!")
        print("You may need to restart the agent service after this.")
        
        # Test the device after reset
        print("\nTesting device after comprehensive reset...")
        import time
        time.sleep(3)
        test_yealink_device()
    else:
        print("Comprehensive reset had limited success.")
        print("Manual recovery options:")
        print("1. Unplug and replug the USB device")
        print("2. Restart audio services: sudo systemctl restart alsa-state")
        print("3. Kill and restart PulseAudio: pulseaudio --kill && pulseaudio --start")
        print("4. Reboot the system if issues persist")
    
    return success


def force_reset():
    """Perform comprehensive audio system reset for all devices."""
    print("\n=== Comprehensive Audio System Reset ===")
    print("This will reset all audio devices and services...")
    
    # Stop all audio services first
    print("Stopping all audio services...")
    stop_audio_services()
    
    # Use the original force reset from common library
    success = force_audio_system_reset()
    
    # Restart services
    print("Restarting audio services...")
    restart_success = restart_audio_services()
    
    if success or restart_success:
        print("Comprehensive audio system reset completed.")
        print("Testing devices after reset...")
        import time
        time.sleep(5)
        list_devices()
    else:
        print("Comprehensive audio system reset had limited success.")
        print("Consider manual intervention or system reboot.")
    
    return success or restart_success


def validate_device():
    """Validate that expected audio devices are accessible by name."""
    print("\n=== Validating Expected Audio Devices ===")
    
    devices = list_audio_devices()
    icusb_found = False
    yealink_found = False
    icusb_valid = False
    yealink_valid = False
    
    # Look for devices by name
    for device in devices:
        name_lower = device['name'].lower()
        
        if 'icusbaudio7d' in name_lower:
            icusb_found = True
            print(f"✓ Found ICUSBAUDIO7D device: [{device['index']}] {device['name']}")
            if validate_audio_device_index(device['index']):
                print("  ✓ Device is accessible")
                icusb_valid = True
            else:
                print("  ✗ Device is not accessible")
            
            # Check capabilities
            if device['max_output_channels'] >= 6:
                print(f"  ✓ Output channels: {device['max_output_channels']} (sufficient for 5.1)")
            else:
                print(f"  ⚠️  Output channels: {device['max_output_channels']} (need 6 for 5.1)")
        
        elif 'yealink' in name_lower:
            yealink_found = True
            print(f"✓ Found Yealink device: [{device['index']}] {device['name']}")
            if validate_audio_device_index(device['index']):
                print("  ✓ Device is accessible")
                yealink_valid = True
            else:
                print("  ✗ Device is not accessible")
            
            # Check capabilities
            if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
                print(f"  ✓ I/O channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
            else:
                print(f"  ⚠️  I/O channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
    
    if not icusb_found:
        print("✗ ICUSBAUDIO7D device not found")
    if not yealink_found:
        print("✗ Yealink device not found")
    
    # Show all devices for reference
    print("\nAll available devices:")
    for device in devices:
        device_type = ""
        if 'icusbaudio7d' in device['name'].lower():
            device_type = " (ICUSBAUDIO7D - 5.1 Surround)"
        elif 'yealink' in device['name'].lower():
            device_type = " (Yealink - Communications)"
        print(f"[{device['index']:2d}] {device['name']}{device_type}")
    
    return icusb_valid and yealink_valid


def fix_jack_shared_memory():
    """
    Fix JACK shared memory issues that cause JackShmReadWritePtr errors.
    
    These errors commonly occur when JACK server starts but clients can't
    connect due to shared memory initialization problems.
    
    Returns:
        True if fix was attempted successfully
    """
    print("\n=== Fixing JACK Shared Memory Issues ===")
    
    try:
        import subprocess
        import time
        
        logger.info("Attempting to fix JACK shared memory issues...")
        
        # Step 1: Stop JACK completely
        print("Step 1: Stopping JACK server...")
        try:
            subprocess.run(['jack_control', 'stop'], 
                         capture_output=True, timeout=10)
            time.sleep(2)
        except Exception as e:
            logger.debug(f"JACK stop failed: {e}")
        
        # Step 2: Clean up any remaining JACK processes
        print("Step 2: Cleaning up JACK processes...")
        try:
            subprocess.run(['pkill', '-f', 'jackd'], capture_output=True, timeout=5)
            subprocess.run(['pkill', '-f', 'jackdbus'], capture_output=True, timeout=5)
            time.sleep(1)
        except Exception as e:
            logger.debug(f"Process cleanup failed: {e}")
        
        # Step 3: Clear JACK temporary files and shared memory
        print("Step 3: Clearing JACK shared memory...")
        try:
            # Clear JACK temporary directory
            subprocess.run(['rm', '-rf', '/dev/shm/jack*'], capture_output=True, timeout=5)
            subprocess.run(['rm', '-rf', '/tmp/jack*'], capture_output=True, timeout=5)
            
            # Clear user's JACK runtime directory
            import os
            jack_runtime_dir = f"/run/user/{os.getuid()}/jack"
            if os.path.exists(jack_runtime_dir):
                subprocess.run(['rm', '-rf', jack_runtime_dir], capture_output=True, timeout=5)
                
        except Exception as e:
            logger.debug(f"Shared memory cleanup failed: {e}")
        
        # Step 4: Reset ALSA to ensure clean hardware state
        print("Step 4: Resetting ALSA...")
        try:
            subprocess.run(['alsactl', 'restore'], capture_output=True, timeout=5)
        except Exception as e:
            logger.debug(f"ALSA reset failed: {e}")
        
        # Step 5: Restart JACK with fresh initialization
        print("Step 5: Restarting JACK with clean state...")
        try:
            subprocess.run(['jack_control', 'start'], 
                         capture_output=True, timeout=15)
            time.sleep(3)  # Give JACK more time to initialize properly
        except Exception as e:
            logger.debug(f"JACK restart failed: {e}")
            
        print("JACK shared memory fix completed.")
        print("Testing JACK connectivity...")
        
        # Test if the fix worked
        time.sleep(2)
        return test_jack_system()
        
    except Exception as e:
        logger.error(f"Error fixing JACK shared memory: {e}")
        return False


def test_jack_system():
    """Test JACK audio system status and capabilities."""
    print("\n=== JACK Audio System Test ===")
    
    try:
        import subprocess
        
        # Check if JACK is running
        result = subprocess.run(['jack_control', 'status'], 
                              capture_output=True, text=True, timeout=5)
        
        if 'started' in result.stdout:
            print("✓ JACK server is running")
        else:
            print("✗ JACK server is not running")
            print("  Try: uv run scripts/audio_recovery.py restart-services")
            return False
            
    except Exception as e:
        print(f"✗ Cannot check JACK status: {e}")
        return False
    
    # Get JACK parameters
    try:
        result = subprocess.run(['jack_control', 'dp'], 
                              capture_output=True, text=True, timeout=5)
        
        print("\n--- JACK Configuration ---")
        for line in result.stdout.split('\n'):
            if 'device:' in line and 'hw:' in line:
                print(f"Device: {line.split(':')[-1].strip()}")
            elif 'rate:' in line and 'set:' in line:
                rate = line.split(':')[-1].strip()
                print(f"Sample Rate: {rate}Hz")
            elif 'outchannels:' in line and 'set:' in line:
                channels = line.split(':')[-1].strip()
                print(f"Output Channels: {channels}")
                
    except Exception as e:
        print(f"Could not get JACK parameters: {e}")
    
    # List JACK ports
    try:
        result = subprocess.run(['jack_lsp'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n--- JACK Ports ---")
            print(f"✗ Cannot connect to JACK server for port listing")
            print(f"   Error: {result.stderr.strip()}")
            print(f"   This suggests JACK server is not properly accessible")
            print(f"   Try: uv run scripts/audio_recovery.py restart-services")
            return False
        
        all_ports = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        playback_ports = [port for port in all_ports if port.startswith('system:playback_')]
        capture_ports = [port for port in all_ports if port.startswith('system:capture_')]
        
        print(f"\n--- JACK Ports ---")
        print(f"✓ Playback ports: {len(playback_ports)} ({', '.join(playback_ports)})")
        print(f"✓ Capture ports: {len(capture_ports)} ({', '.join(capture_ports)})")
        
        # Check for 5.1/7.1 surround capability
        if len(playback_ports) >= 6:
            print(f"✓ Sufficient channels for 5.1 surround sound")
        if len(playback_ports) >= 8:
            print(f"✓ Sufficient channels for 7.1 surround sound")
        elif len(playback_ports) > 0:
            print(f"⚠️  Only {len(playback_ports)} channels available (need 6 for 5.1)")
        
        return len(playback_ports) >= 6
        
    except Exception as e:
        print(f"Could not list JACK ports: {e}")
        return False


def test_jack_audio_playback(test_file=None):
    """Test audio playback through JACK using a test file."""
    print("\n=== JACK Audio Playback Test ===")
    
    if not test_file:
        # Use a default test file
        test_file = "media/audio/music/modern_loop.wav"
    
    # Check if test file exists
    from pathlib import Path
    if not Path(test_file).exists():
        print(f"✗ Test file not found: {test_file}")
        print("Available test files:")
        try:
            for f in Path("media/audio/music").glob("*.wav"):
                print(f"  {f}")
        except:
            pass
        return False
    
    print(f"Testing playback of: {test_file}")
    print("This will attempt to play audio through JACK...")
    print("Gallery staff should listen for audio on the surround sound system.")
    
    try:
        import subprocess
        
        # Method 1: Try with ffplay using JACK
        print("\n--- Method 1: ffplay with JACK ---")
        try:
            print("Starting audio playback (will play for 10 seconds)...")
            result = subprocess.run([
                'timeout', '10s', 
                'ffplay', '-nodisp', '-autoexit', 
                '-f', 'jack', test_file
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 or result.returncode == 124:  # 124 = timeout success
                print("✓ ffplay with JACK completed")
                return True
            else:
                print(f"✗ ffplay with JACK failed: {result.stderr}")
        except Exception as e:
            print(f"✗ ffplay with JACK failed: {e}")
        
        # Method 2: Try with aplay through JACK bridge
        print("\n--- Method 2: aplay direct to hardware ---")
        try:
            print("Testing direct hardware playback (5 seconds)...")
            result = subprocess.run([
                'timeout', '5s',
                'aplay', '-D', 'hw:3,0', test_file
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 or result.returncode == 124:
                print("✓ Direct hardware playback completed")
                return True
            else:
                print(f"✗ Direct hardware playback failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Direct hardware playback failed: {e}")
        
        # Method 3: Try speaker test on specific channels
        print("\n--- Method 3: Speaker test on rear channels ---")
        try:
            print("Testing rear speakers (channels 5&6 for music)...")
            result = subprocess.run([
                'timeout', '5s',
                'speaker-test', '-D', 'hw:3,0', '-c', '6', '-t', 'wav', '-s', '5,6'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 or result.returncode == 124:
                print("✓ Rear speaker test completed")
                return True
            else:
                print(f"✗ Rear speaker test failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Rear speaker test failed: {e}")
        
        print("\n⚠️  All playback methods failed.")
        print("This may indicate hardware issues or incorrect audio routing.")
        return False
        
    except Exception as e:
        print(f"✗ Audio playback test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Audio diagnostic and recovery tool')
    parser.add_argument('action', choices=[
        'list', 'test', 'test-yealink', 'test-icusb', 'test-jack', 'test-audio', 'fix-jack',
        'reset', 'reset-yealink', 'reset-icusb', 'force-reset', 
        'validate', 'stop-services', 'restart-services'
    ], help='Action to perform: list devices, test Yealink/ICUSBAUDIO7D/JACK, play test audio, fix JACK shared memory, reset devices, validate devices by name, or manage audio services')
    
    parser.add_argument('--file', '-f', type=str, 
                       help='Audio file to use for testing (default: media/audio/music/modern_loop.wav)')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'list':
            list_devices()
        elif args.action == 'test':
            # Test both devices
            print("Testing all supported devices...")
            yealink_ok = test_yealink_device()
            icusb_ok = test_icusbaudio7d_device()
            jack_ok = test_jack_system()
            
            print("\n=== Summary ===")
            if not yealink_ok and not icusb_ok:
                print("⚠️  No supported devices found or all have issues")
            elif not yealink_ok:
                print("⚠️  Yealink device has issues")
            elif not icusb_ok:
                print("⚠️  ICUSBAUDIO7D device has issues")
            else:
                print("✓ All supported devices appear functional")
                
            if jack_ok:
                print("✓ JACK audio system is working")
            else:
                print("⚠️  JACK audio system has issues")
                
        elif args.action == 'test-yealink':
            test_yealink_device()
        elif args.action == 'test-icusb':
            test_icusbaudio7d_device()
        elif args.action == 'test-jack':
            test_jack_system()
        elif args.action == 'fix-jack':
            print("=== JACK Shared Memory Fix ===")
            print("This will fix common JACK connectivity issues including:")
            print("- JackShmReadWritePtr errors")
            print("- 'Cannot connect to server socket' errors")
            print("- SuperCollider JACK connection problems\n")
            
            success = fix_jack_shared_memory()
            if success:
                print("✓ JACK shared memory fix completed successfully")
            else:
                print("⚠️  JACK fix had limited success - may need manual intervention")
        elif args.action == 'test-audio':
            # Comprehensive audio test for gallery staff
            print("=== Comprehensive Audio Test for Gallery Staff ===")
            print("This test will check JACK and attempt audio playback.")
            print("Listen for audio on the surround sound system.\n")
            
            jack_ok = test_jack_system()
            if jack_ok:
                test_jack_audio_playback(args.file)
            else:
                print("⚠️  JACK system not ready. Try running: uv run scripts/audio_recovery.py restart-services")
                
        elif args.action == 'reset':
            # Reset both devices with comprehensive approach
            print("Attempting comprehensive reset of all supported devices...")
            reset_yealink()
            reset_icusbaudio7d()
        elif args.action == 'reset-yealink':
            reset_yealink()
        elif args.action == 'reset-icusb':
            reset_icusbaudio7d()
        elif args.action == 'force-reset':
            force_reset()
        elif args.action == 'validate':
            validate_device()
        elif args.action == 'stop-services':
            print("=== Stopping Audio Services ===")
            success = stop_audio_services()
            if success:
                print("Audio services stopped successfully")
            else:
                print("Audio service stop had limited success")
        elif args.action == 'restart-services':
            print("=== Restarting Audio Services ===")
            success = restart_audio_services()
            if success:
                print("Audio services restarted successfully")
            else:
                print("Audio service restart had limited success")
            
    except Exception as e:
        logger.error(f"Error during {args.action}: {e}")
        return 1
    finally:
        # Clean up audio resources
        cleanup_audio_resources()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

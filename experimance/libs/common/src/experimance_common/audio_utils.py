import logging
import os
import sys
from typing import Optional, Dict, List, Tuple
import contextlib

# Set environment variables to suppress audio system errors BEFORE importing pyaudio
os.environ.setdefault('ALSA_PCM_CARD', 'default')
os.environ.setdefault('ALSA_PCM_DEVICE', '0')
# Suppress JACK error messages
os.environ.setdefault('JACK_NO_AUDIO_RESERVATION', '1')
# Try to redirect JACK errors to /dev/null
os.environ.setdefault('JACK_DEFAULT_SERVER', 'dummy')

import pyaudio

logger = logging.getLogger(__name__)

# Cache for device enumeration to avoid repeated slow operations
_device_cache: Optional[List[Dict]] = None
_pyaudio_instance: Optional[pyaudio.PyAudio] = None

@contextlib.contextmanager
def suppress_audio_errors():
    """Context manager to suppress ALSA/JACK error output."""
    # Suppress both stderr and stdout temporarily to hide ALSA/JACK errors
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    
    # Also try to suppress ALSA error handler if possible
    alsa_error_handler_set = False
    try:
        # Try to set ALSA error handler to null (if ctypes is available)
        try:
            import ctypes
            import ctypes.util
            
            # Find ALSA library
            alsa_lib = ctypes.util.find_library('asound')
            if alsa_lib:
                libasound = ctypes.CDLL(alsa_lib)
                # Set error handler to null function
                c_error_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, 
                                                 ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
                null_error_handler = c_error_handler(lambda *args: None)
                libasound.snd_lib_error_set_handler(null_error_handler)
                alsa_error_handler_set = True
        except Exception:
            pass  # Couldn't set ALSA error handler, continue with file suppression
        
        # Suppress file descriptors
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            sys.stdout = devnull
            yield
            
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        
        # Reset ALSA error handler if we set it
        if alsa_error_handler_set:
            try:
                # Reset to default handler
                libasound.snd_lib_error_set_handler(None)
            except Exception:
                pass


def _get_pyaudio_instance() -> pyaudio.PyAudio:
    """Get a cached PyAudio instance to avoid repeated initialization."""
    global _pyaudio_instance
    if _pyaudio_instance is None:
        with suppress_audio_errors():
            _pyaudio_instance = pyaudio.PyAudio()
    return _pyaudio_instance


def _get_device_list() -> List[Dict]:
    """Get cached device list to avoid repeated enumeration."""
    global _device_cache
    if _device_cache is None:
        logger.debug("Enumerating audio devices (this may take a moment)...")
        _device_cache = []
        p = _get_pyaudio_instance()
        
        with suppress_audio_errors():
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    _device_cache.append({
                        'index': i,
                        'name': str(device_info['name']),
                        'max_input_channels': int(device_info['maxInputChannels']),
                        'max_output_channels': int(device_info['maxOutputChannels']),
                        'default_sample_rate': device_info.get('defaultSampleRate', 44100)
                    })
                except Exception as e:
                    logger.debug(f"Error checking device {i}: {e}")
                    continue
        
        logger.debug(f"Found {len(_device_cache)} audio devices")
    
    return _device_cache


def cleanup_audio_resources():
    """Clean up cached audio resources. Call this at shutdown."""
    global _pyaudio_instance, _device_cache
    if _pyaudio_instance:
        try:
            with suppress_audio_errors():
                _pyaudio_instance.terminate()
        except Exception as e:
            logger.debug(f"Error terminating PyAudio: {e}")
        finally:
            _pyaudio_instance = None
    _device_cache = None


def find_audio_device_by_name(device_name: str, input_device: bool = True) -> Optional[int]:
    """
    Find an audio device index by partial name matching.
    
    Args:
        device_name: Partial name to search for (case-insensitive)
        input_device: If True, look for devices with input channels; if False, look for output channels
        
    Returns:
        Device index if found, None otherwise
    """
    if not pyaudio:
        logger.error("PyAudio not available for device name resolution")
        return None
        
    try:
        device_name_lower = device_name.lower()
        devices = _get_device_list()
        
        for device in devices:
            name = device['name'].lower()
            max_inputs = device['max_input_channels']
            max_outputs = device['max_output_channels']
            
            # Check if the device name contains our search term
            if device_name_lower in name:
                # Check if device has the right capabilities
                if input_device and max_inputs > 0:
                    logger.info(f"Found input device '{device['name']}' at index {device['index']}")
                    return device['index']
                elif not input_device and max_outputs > 0:
                    logger.info(f"Found output device '{device['name']}' at index {device['index']}")
                    return device['index']
                    
        logger.warning(f"No {'input' if input_device else 'output'} device found matching '{device_name}'")
        return None
            
    except Exception as e:
        logger.error(f"Error searching for audio device '{device_name}': {e}")
        return None


def resolve_audio_device_index(
    device_index: Optional[int], 
    device_name: Optional[str], 
    input_device: bool = True,
    fast_mode: bool = False
) -> Optional[int]:
    """
    Resolve audio device index from either explicit index or device name.
    
    Args:
        device_index: Explicit device index (takes precedence if both are provided)
        device_name: Device name to search for
        input_device: Whether this is for input (True) or output (False)
        fast_mode: If True, skip device enumeration and use index directly when available
        
    Returns:
        Resolved device index, or None for default device
    """
    # In fast mode, if we have a device index, use it directly without validation
    if fast_mode and device_index is not None:
        logger.debug(f"Fast mode: using device index {device_index} directly")
        return device_index
    
    # Device name takes precedence over index if both are provided
    if device_name:
        resolved_index = find_audio_device_by_name(device_name, input_device)
        if resolved_index is not None:
            return resolved_index
        else:
            logger.warning(f"Device name '{device_name}' not found, falling back to device index")
    
    # Fall back to explicit index
    if device_index is not None:
        return device_index
        
    # Use default device
    return None


def validate_audio_device_index(device_index: int) -> bool:
    """
    Validate that a device index exists and is accessible.
    
    Args:
        device_index: Device index to validate
        
    Returns:
        True if device is valid and accessible
    """
    try:
        with suppress_audio_errors():
            p = pyaudio.PyAudio()
            try:
                device_info = p.get_device_info_by_index(device_index)
                return device_info is not None
            finally:
                p.terminate()
    except Exception:
        return False


def list_audio_devices() -> List[Dict]:
    """
    List all available audio devices with their capabilities.
    
    Returns:
        List of device info dictionaries
    """
    try:
        devices = _get_device_list()
        return devices
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        return []


def reset_audio_device_by_name(device_name: str) -> bool:
    """
    Attempt to reset/reinitialize a USB audio device by name.
    
    This is a best-effort function that tries to help with stuck USB audio devices.
    
    Args:
        device_name: Partial name of the device to reset
        
    Returns:
        True if reset was attempted, False otherwise
    """
    try:
        import subprocess
        import os
        import time
        
        logger.info(f"Attempting to reset USB audio device matching '{device_name}'")
        
        # Method 1: Try to find and reset USB device via sysfs (no sudo required)
        try:
            # Find USB devices in sysfs
            usb_devices_path = "/sys/bus/usb/devices"
            if os.path.exists(usb_devices_path):
                for device_dir in os.listdir(usb_devices_path):
                    device_path = os.path.join(usb_devices_path, device_dir)
                    
                    # Look for device product name
                    product_file = os.path.join(device_path, "product")
                    if os.path.exists(product_file):
                        try:
                            with open(product_file, 'r') as f:
                                product_name = f.read().strip()
                            
                            if device_name.lower() in product_name.lower():
                                logger.info(f"Found USB device: {product_name} at {device_path}")
                                
                                # Try to unbind and rebind the device
                                authorized_file = os.path.join(device_path, "authorized")
                                if os.path.exists(authorized_file):
                                    try:
                                        # Deauthorize
                                        with open(authorized_file, 'w') as f:
                                            f.write('0')
                                        time.sleep(1)
                                        
                                        # Reauthorize
                                        with open(authorized_file, 'w') as f:
                                            f.write('1')
                                        
                                        logger.info(f"Successfully reset USB device: {product_name}")
                                        time.sleep(2)  # Give device time to reinitialize
                                        return True
                                        
                                    except PermissionError:
                                        logger.debug(f"No permission to reset {product_name} via sysfs")
                                        
                        except Exception as e:
                            logger.debug(f"Error checking device {device_dir}: {e}")
                            continue
                            
        except Exception as e:
            logger.debug(f"sysfs reset method failed: {e}")
        
        # Method 2: Try udev/systemd approach
        try:
            logger.info("Attempting udev-based device reset...")
            # Trigger udev rules to reload USB audio devices
            result = subprocess.run(['udevadm', 'trigger', '--subsystem-match=usb', '--attr-match=bInterfaceClass=01'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("udev trigger completed")
                time.sleep(2)
                return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"udev reset method failed: {e}")
        
        # Method 3: Module reload approach (if available)  
        try:
            logger.info("Attempting to reload USB audio modules...")
            # Try to reload snd-usb-audio module (may require sudo)
            modules_to_reload = ['snd_usb_audio', 'snd_usbmidi_lib']
            
            for module in modules_to_reload:
                try:
                    # Check if module is loaded
                    result = subprocess.run(['lsmod'], capture_output=True, text=True, timeout=5)
                    if module in result.stdout:
                        # Try to reload (this usually requires sudo, but we'll try)
                        subprocess.run(['sudo', '-n', 'modprobe', '-r', module], 
                                     capture_output=True, timeout=5)
                        time.sleep(0.5)
                        subprocess.run(['sudo', '-n', 'modprobe', module], 
                                     capture_output=True, timeout=5)
                        logger.info(f"Reloaded module: {module}")
                except Exception:
                    pass  # Module reload failed, continue
            
            time.sleep(2)
            return True
            
        except Exception as e:
            logger.debug(f"Module reload failed: {e}")
        
        # Method 4: PulseAudio/ALSA restart (user-level)
        try:
            logger.info("Attempting to restart user audio services...")
            
            # Kill PulseAudio (it will auto-restart)
            subprocess.run(['pulseaudio', '--kill'], capture_output=True, timeout=5)
            time.sleep(1)
            
            # Restart PulseAudio
            subprocess.run(['pulseaudio', '--start'], capture_output=True, timeout=5)
            time.sleep(1)
            
            # Force ALSA to re-scan devices
            subprocess.run(['alsactl', 'restore'], capture_output=True, timeout=5)
            
            logger.info("Audio services restart completed")
            time.sleep(2)
            return True
            
        except Exception as e:
            logger.debug(f"Audio service restart failed: {e}")
        
        # Method 5: Clear PyAudio cache and force re-enumeration
        try:
            logger.info("Clearing audio device cache...")
            cleanup_audio_resources()
            time.sleep(1)
            
            # Force re-enumeration by creating a new PyAudio instance
            with suppress_audio_errors():
                test_pa = pyaudio.PyAudio()
                device_count = test_pa.get_device_count()
                test_pa.terminate()
                logger.info(f"Re-enumerated {device_count} audio devices")
            
            return True
            
        except Exception as e:
            logger.debug(f"Cache clear failed: {e}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error attempting to reset audio device: {e}")
        return False


def force_audio_system_reset() -> bool:
    """
    Perform a comprehensive audio system reset.
    
    This function tries multiple methods to reset the audio system
    when devices are stuck or causing delays.
    
    Returns:
        True if any reset method was attempted successfully
    """
    logger.info("Performing comprehensive audio system reset...")
    
    success = False
    
    # 1. Clear our caches first
    cleanup_audio_resources()
    
    # 2. Try to reset Yealink device specifically
    if reset_audio_device_by_name("Yealink"):
        success = True
    
    # 3. General audio system reset
    try:
        import subprocess
        import time
        
        # Kill all audio processes
        audio_processes = ['pulseaudio', 'pipewire', 'jack']
        for process in audio_processes:
            try:
                subprocess.run(['pkill', '-f', process], capture_output=True, timeout=5)
            except Exception:
                pass
        
        time.sleep(2)
        
        # Restart PulseAudio
        subprocess.run(['pulseaudio', '--start'], capture_output=True, timeout=10)
        time.sleep(2)
        
        success = True
        
    except Exception as e:
        logger.debug(f"System audio reset failed: {e}")
    
    # 4. Force PyAudio reinitialization
    try:
        with suppress_audio_errors():
            test_pa = pyaudio.PyAudio()
            test_pa.terminate()
        success = True
    except Exception:
        pass
    
    if success:
        logger.info("Audio system reset completed")
    else:
        logger.warning("Audio system reset had limited success")
    
    return success

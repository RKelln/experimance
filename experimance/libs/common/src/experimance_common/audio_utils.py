import logging
from typing import Optional
import pyaudio

logger = logging.getLogger(__name__)

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
        p = pyaudio.PyAudio()
        
        try:
            device_name_lower = device_name.lower()
            
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    name = str(device_info['name']).lower()
                    max_inputs = int(device_info['maxInputChannels'])
                    max_outputs = int(device_info['maxOutputChannels'])
                    
                    # Check if the device name contains our search term
                    if device_name_lower in name:
                        # Check if device has the right capabilities
                        if input_device and max_inputs > 0:
                            logger.info(f"Found input device '{device_info['name']}' at index {i}")
                            return i
                        elif not input_device and max_outputs > 0:
                            logger.info(f"Found output device '{device_info['name']}' at index {i}")
                            return i
                            
                except Exception as e:
                    logger.debug(f"Error checking device {i}: {e}")
                    continue
                    
            logger.warning(f"No {'input' if input_device else 'output'} device found matching '{device_name}'")
            return None
            
        finally:
            p.terminate()
            
    except Exception as e:
        logger.error(f"Error searching for audio device '{device_name}': {e}")
        return None


def resolve_audio_device_index(
    device_index: Optional[int], 
    device_name: Optional[str], 
    input_device: bool = True
) -> Optional[int]:
    """
    Resolve audio device index from either explicit index or device name.
    
    Args:
        device_index: Explicit device index (takes precedence if both are provided)
        device_name: Device name to search for
        input_device: Whether this is for input (True) or output (False)
        
    Returns:
        Resolved device index, or None for default device
    """
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

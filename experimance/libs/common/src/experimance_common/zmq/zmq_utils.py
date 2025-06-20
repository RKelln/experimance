"""
ZMQ utility enhancement module that avoids hanging on socket operations.
"""

import asyncio
import json
import logging
import os
import socket
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union, cast

import zmq
import zmq.asyncio

# Re-export MessageType for backward compatibility
from experimance_common.zmq.config import MessageType


# Note: Logging is configured by the CLI or service entry point
logger = logging.getLogger(__name__)

from experimance_common.constants import (
    DEFAULT_TIMEOUT,
    HEARTBEAT_INTERVAL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RECV_TIMEOUT,
    HEARTBEAT_TOPIC,
    IMAGE_TRANSPORT_MODES,
    DEFAULT_IMAGE_TRANSPORT_MODE,
    IMAGE_TRANSPORT_SIZE_THRESHOLD,
    TEMP_FILE_PREFIX,
    TEMP_FILE_CLEANUP_AGE,
    TEMP_FILE_CLEANUP_INTERVAL,
    DEFAULT_TEMP_DIR,
    FILE_URI_PREFIX,
    BASE64_PNG_PREFIX,
)

from experimance_common.schemas import MessageBase

# Image Transport Utilities for ZMQ Communication

def is_local_address(address: str) -> bool:
    """Check if a ZMQ address is local (same machine).
    
    Args:
        address: ZMQ address (e.g., "tcp://localhost:5555" or "tcp://192.168.1.100:5555")
        
    Returns:
        True if address is local, False otherwise
    """
    try:
        if "localhost" in address or "127.0.0.1" in address:
            return True
        
        # Extract hostname/IP from address
        if "://" in address:
            host_port = address.split("://")[1]
            host = host_port.split(":")[0]
            
            # Get local IP addresses
            local_ips = [socket.gethostbyname(socket.gethostname())]
            try:
                # Also check all local interfaces
                hostname = socket.gethostname()
                local_ips.extend([ip for ip in socket.gethostbyname_ex(hostname)[2]])
            except:
                pass
                
            return host in local_ips or host in ["localhost", "127.0.0.1"]
            
    except Exception:
        pass
    
    return False


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, TypeError):
        return 0


def choose_image_transport_mode(
    file_path: Optional[Union[str, Path]] = None,
    target_address: Optional[str] = None,
    transport_mode: str = DEFAULT_IMAGE_TRANSPORT_MODE,
    force_mode: Optional[str] = None
) -> str:
    """Choose the best image transport mode based on ZMQ configuration and context.
    
    Args:
        file_path: Path to image file (if available)
        target_address: ZMQ target address (e.g., "tcp://localhost:5555")
        transport_mode: Configured transport mode from IMAGE_TRANSPORT_MODES
        force_mode: Force specific mode, overrides all other logic
        
    Returns:
        Transport mode to use: "file_uri", "base64", or "hybrid"
    """
    if force_mode and force_mode in IMAGE_TRANSPORT_MODES.values():
        return force_mode
    
    if transport_mode == IMAGE_TRANSPORT_MODES["FILE_URI"]:
        return IMAGE_TRANSPORT_MODES["FILE_URI"]
    elif transport_mode == IMAGE_TRANSPORT_MODES["BASE64"]:
        return IMAGE_TRANSPORT_MODES["BASE64"]
    elif transport_mode == IMAGE_TRANSPORT_MODES["HYBRID"]:
        return IMAGE_TRANSPORT_MODES["HYBRID"]
    elif transport_mode == IMAGE_TRANSPORT_MODES["AUTO"]:
        # Auto-detect based on target and file size
        is_local = target_address and is_local_address(target_address)
        file_size = get_file_size(file_path) if file_path else 0
        
        if is_local and file_size > 0:
            # Local target with existing file - prefer URI for large files
            if file_size > IMAGE_TRANSPORT_SIZE_THRESHOLD:
                return IMAGE_TRANSPORT_MODES["FILE_URI"]
            else:
                # Small file - base64 is fine and more reliable
                return IMAGE_TRANSPORT_MODES["BASE64"]
        else:
            # Remote target or no file - use base64
            return IMAGE_TRANSPORT_MODES["BASE64"]
    
    # Default fallback
    return IMAGE_TRANSPORT_MODES["BASE64"]


def prepare_image_message(
    image_data: Optional[Union[str, Path, Any]] = None,  # Any for PIL Image, encdoed string or nd.array
    target_address: Optional[str] = None,
    transport_mode: str = DEFAULT_IMAGE_TRANSPORT_MODE,
    **message_kwargs
) -> Dict[str, Any]:
    """Prepare a ZMQ message with appropriate image transport method.
    
    Args:
        image_data: Image to send (file path, PIL Image, numpy array, or base64 string)
        target_address: ZMQ target address
        transport_mode: Transport mode configuration
        **message_kwargs: Additional message fields
        
    Returns:
        Message dict with uri and/or image_data fields set appropriately.
        
    Note:
        When using file_uri or hybrid modes with numpy arrays or PIL images,
        temporary files are created and automatically scheduled for perodic cleanup. 
        The message includes a '_temp_file' key for awareness but cleanup MUST be handled 
        automatically by the publisher. See `periodic_temp_file_cleanup()`.
    """
    # Import here to avoid circular imports
    from experimance_common.image_utils import (
        png_to_base64url, 
        ndarray_to_base64url,
        save_ndarray_as_tempfile,
        save_pil_image_as_tempfile,
    )
    
    message = dict(message_kwargs)
    
    if image_data is None:
        return message
    
    # Determine what we're working with
    file_path = None
    pil_image = None
    numpy_array = None
    base64_data = None
    
    if isinstance(image_data, (str, Path)):
        if isinstance(image_data, str) and image_data.startswith(BASE64_PNG_PREFIX):
            # Already base64 encoded
            base64_data = image_data
        else:
            # File path
            file_path = str(image_data)
    else:
        # Check if it's a numpy array
        try:
            import numpy as np
            if isinstance(image_data, np.ndarray):
                numpy_array = image_data
            else:
                # Assume it's a PIL Image or similar object with a save method
                if hasattr(image_data, 'save'):
                    pil_image = image_data
        except ImportError:
            # numpy not available, check for PIL Image
            if hasattr(image_data, 'save'):
                pil_image = image_data
    
    # Choose transport mode
    chosen_mode = choose_image_transport_mode(
        file_path=file_path,
        target_address=target_address,
        transport_mode=transport_mode
    )
    
    # Prepare message based on chosen mode
    if chosen_mode == IMAGE_TRANSPORT_MODES["FILE_URI"]:
        if file_path and os.path.exists(file_path):
            message["uri"] = f"{FILE_URI_PREFIX}{os.path.abspath(file_path)}"
        elif numpy_array is not None:
            # Save numpy array as temporary file
            temp_path = save_ndarray_as_tempfile(numpy_array)
            message["uri"] = f"{FILE_URI_PREFIX}{os.path.abspath(temp_path)}"
            # Mark as temp file for caller awareness (cleanup is service responsibility)
            message["_temp_file"] = temp_path
        elif pil_image:
            # Save PIL image as temporary file
            temp_path = save_pil_image_as_tempfile(pil_image)
            message["uri"] = f"{FILE_URI_PREFIX}{os.path.abspath(temp_path)}"
            # Mark as temp file for caller awareness (cleanup is service responsibility)
            message["_temp_file"] = temp_path
        else:
            # Fallback to base64 if no valid source
            chosen_mode = IMAGE_TRANSPORT_MODES["BASE64"]
    
    if chosen_mode in [IMAGE_TRANSPORT_MODES["BASE64"], IMAGE_TRANSPORT_MODES["HYBRID"]]:
        if base64_data:
            message["image_data"] = base64_data
        elif numpy_array is not None:
            message["image_data"] = ndarray_to_base64url(numpy_array)
        elif pil_image:
            message["image_data"] = png_to_base64url(pil_image)
        elif file_path and os.path.exists(file_path):
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    message["image_data"] = png_to_base64url(img)
            except Exception as e:
                logger.warning(f"Could not load image for base64 encoding: {e}")
    
    if chosen_mode == IMAGE_TRANSPORT_MODES["HYBRID"]:
        # Include both URI and base64 data when possible
        if file_path and os.path.exists(file_path) and "uri" not in message:
            message["uri"] = f"{FILE_URI_PREFIX}{os.path.abspath(file_path)}"
        elif numpy_array is not None and "uri" not in message:
            # Create temporary file for URI in hybrid mode
            temp_path = save_ndarray_as_tempfile(numpy_array)
            message["uri"] = f"{FILE_URI_PREFIX}{os.path.abspath(temp_path)}"
            # Mark as temp file for caller awareness (cleanup is service responsibility)
            message["_temp_file"] = temp_path
        elif pil_image and "uri" not in message:
            # Create temporary file for URI in hybrid mode
            temp_path = save_pil_image_as_tempfile(pil_image)
            message["uri"] = f"{FILE_URI_PREFIX}{os.path.abspath(temp_path)}"
            # Mark as temp file for caller awareness (cleanup is service responsibility)
            message["_temp_file"] = temp_path
    
    return message


def cleanup_temp_image_file(message: Dict[str, Any]) -> None:
    """Manually clean up temporary image file created by prepare_image_message().
    
    Note: Temporary files are automatically cleaned up after 5 minutes.
    This function is provided for immediate cleanup if needed.
    
    Args:
        message: Message dict that may contain '_temp_file' key
    """
    temp_file = message.get("_temp_file")
    if temp_file and os.path.exists(temp_file):
        try:
            os.unlink(temp_file)
            logger.debug(f"Cleaned up temporary image file: {temp_file}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary image file {temp_file}: {e}")
        finally:
            # Remove the temp file marker regardless
            message.pop("_temp_file", None)


def cleanup_old_temp_files(max_age_seconds: int = TEMP_FILE_CLEANUP_AGE, 
                         temp_dir: str = DEFAULT_TEMP_DIR, 
                         pattern: str = f"{TEMP_FILE_PREFIX}*") -> int:
    """Clean up old temporary image files.
    
    This is a utility function that services can call periodically to clean up
    old temporary files created by prepare_image_message().
    
    Args:
        max_age_seconds: Maximum age of files to keep (default: 5 minutes)
        temp_dir: Directory to search for temp files (default: system temp directory)
        pattern: File pattern to match (default: experimance_img_*)
        
    Returns:
        Number of files cleaned up
    """
    import glob
    import time
    
    search_pattern = os.path.join(temp_dir, pattern)
    now = time.time()
    cleaned_count = 0
    
    try:
        for file_path in glob.glob(search_pattern):
            try:
                # Check file age
                if os.path.getmtime(file_path) < (now - max_age_seconds):
                    os.unlink(file_path)
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old temp file: {file_path}")
            except OSError:
                # File already deleted, permission error, etc.
                pass
    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} old temp files")
    
    return cleaned_count


async def periodic_temp_file_cleanup(interval_seconds: int = TEMP_FILE_CLEANUP_INTERVAL,
                                   max_age_seconds: int = TEMP_FILE_CLEANUP_AGE):
    """Async task for periodic cleanup of old temp files.
    
    Services can add this as a background task:
        asyncio.create_task(periodic_temp_file_cleanup())
    
    Args:
        interval_seconds: How often to run cleanup (default: 1 minute)
        max_age_seconds: Maximum age of files to keep (default: 5 minutes)
    """
    while True:
        try:
            cleanup_old_temp_files(max_age_seconds)
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            # Final cleanup on shutdown
            cleanup_old_temp_files(max_age_seconds=0)  # Clean all temp files
            break
        except Exception as e:
            logger.error(f"Error in periodic temp file cleanup: {e}")
            await asyncio.sleep(interval_seconds)

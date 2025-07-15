import io
import logging
import os
import tempfile
import glob
import time
import base64
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union, Tuple
from pathlib import Path

from experimance_common.schemas import ImageSource

if TYPE_CHECKING:
    import numpy as np

import cv2
from PIL import Image

from .constants import (
    TEMP_FILE_PREFIX, 
    TEMP_FILE_SUFFIX, 
    TEMP_FILE_CLEANUP_AGE,
    BASE64_PNG_PREFIX,
    DATA_URL_PREFIX,
    FILE_URI_PREFIX
)

# Get logger for this module
logger = logging.getLogger(__name__)


class ImageLoadFormat(Enum):
    """Format options for loading images from messages."""
    PIL = "pil"           # Return PIL Image object
    NUMPY = "numpy"       # Return numpy array
    FILEPATH = "filepath" # Return file path (may create temp file)
    ENCODED = "encoded"   # Return base64 encoded string


def cv2_img_to_base64url(img):
    image_base64 = base64.b64encode(cv2.imencode('.png', img)[1].tobytes()).decode('utf-8')
    return f"{BASE64_PNG_PREFIX}{image_base64}"


def png_to_base64url(image, format="PNG"):
    if image is None: return None
    format = format.strip(" .")
    buffered = io.BytesIO()
    image.save(buffered, format=format.upper())
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"{DATA_URL_PREFIX}{format.lower()};base64,{image_base64}"


def base64url_to_png(base64url):
    """
    Convert a base64url string to a PNG image.
    
    Args:
        base64url (str): The base64url string representing the image.
        
    Returns:
        PIL.Image: The decoded image.
    """
    # Remove the data URL prefix if present
    if base64url.startswith(DATA_URL_PREFIX):
        base64url = base64url.split(",")[1]
    
    # Decode the base64url string
    image_data = base64.b64decode(base64url)
    
    # Create a BytesIO stream and open it as an image
    image_stream = io.BytesIO(image_data)
    return Image.open(image_stream)

def ndarray_to_base64url(ndarray):
    """Convert numpy array to base64 URL string.
    
    Args:
        ndarray: Numpy array representing an image
        
    Returns:
        str: Base64 encoded image data URL
    """
    # Convert numpy array to PIL Image
    # Handle different array formats (grayscale, RGB, RGBA, BGR)
    import numpy as np
    
    if len(ndarray.shape) == 2:
        # Grayscale image
        pil_image = Image.fromarray(ndarray, mode='L')
    elif len(ndarray.shape) == 3:
        if ndarray.shape[2] == 3:
            # Assume RGB or BGR - OpenCV uses BGR by default
            if hasattr(ndarray, 'dtype') and ndarray.dtype == np.uint8:
                # For OpenCV BGR images, convert to RGB
                rgb_array = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_array, mode='RGB')
            else:
                # Assume already RGB
                pil_image = Image.fromarray(ndarray, mode='RGB')
        elif ndarray.shape[2] == 4:
            # RGBA image
            pil_image = Image.fromarray(ndarray, mode='RGBA')
        else:
            raise ValueError(f"Unsupported array shape: {ndarray.shape}")
    else:
        raise ValueError(f"Unsupported array dimensions: {len(ndarray.shape)}")
    
    return png_to_base64url(pil_image)



def convert_images_to_mp4(input_path, output_path, filename, fps, remove_images=False, input_format="png"):
    # using ffmpeg convert the png files to mp4 and delete the png files if requested
    os.system(f"ffmpeg -framerate {fps} -i {input_path}/%d.{input_format} -c:v libx264 -pix_fmt yuv420p {output_path}/{filename}")
    if remove_images:
        os.system(f"rm {input_path}/*.{input_format}")


def get_mock_images(path: Union[str, Path]) -> list:
    # use the images in the folder for mock generation
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mock images folder not found: {path}")
    mock_images = []
    for file in path.iterdir():
        if file.is_file() and file.suffix in [".jpg", ".png", ".webp", ".jpeg"]:
            mock_images.append(file)
    if len(mock_images) == 0: 
        raise FileNotFoundError(f"No images found in mock images folder: {path}")
    return mock_images

# input cv2 image, output cv2 image and bounds
def crop_to_content(image, size:tuple=(1024, 1024), bounds:Optional[tuple]=None):
    """
    Crop the image to the content area, expanding it to fit the target size while preserving aspect ratio.
    If bounds are provided, they are used as the content area. If not, the content area is determined by finding non-zero pixels.
    The resulting image will be resized to the target size, with letterboxing or pillarboxing applied as needed.
    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR).
        size (tuple): Target size for the output image (width, height).
        bounds (tuple, optional): Predefined bounds for the content area (x, y, width, height). If None, will calculate from non-zero pixels.
    Returns:
        tuple: A tuple containing the resized image and the bounds of the content area (x, y, width, height).
    """
    if bounds is None:
        # Find the bounding box of the content
        non_zero_pixels = cv2.findNonZero(image)
        if non_zero_pixels is None:
            return cv2.resize(image, size), (0, 0, image.shape[1], image.shape[0])
        
        x, y, w, h = cv2.boundingRect(non_zero_pixels)
        bounds = (x, y, w, h)

    x, y, w, h = bounds
    if w <= 0 or h <= 0:
        return cv2.resize(image, size), (0, 0, image.shape[1], image.shape[0])
    
    # Get image dimensions
    img_h, img_w = image.shape[:2]
    target_w, target_h = size
    
    # Calculate target aspect ratio
    target_aspect = target_w / target_h
    content_aspect = w / h
    
    # Expand the crop area around the content to match target aspect ratio
    # This preserves the content's original proportions within the target size
    if content_aspect > target_aspect:
        # Content is wider than target - need to add height (letterboxing top/bottom)
        new_h = int(w / target_aspect)
        h_expand = (new_h - h) // 2
        new_y = y - h_expand
        new_x = x
        new_w = w
        new_h = new_h
    else:
        # Content is taller than target - need to add width (pillarboxing left/right)
        new_w = int(h * target_aspect)
        w_expand = (new_w - w) // 2
        new_x = x - w_expand
        new_y = y
        new_w = new_w
        new_h = h
    
    # Ensure the expanded crop area stays within image bounds
    new_x = max(0, min(new_x, img_w - 1))
    new_y = max(0, min(new_y, img_h - 1))
    new_w = min(new_w, img_w - new_x)
    new_h = min(new_h, img_h - new_y)
    
    # If we hit image boundaries, adjust the other dimension to maintain aspect ratio
    actual_aspect = new_w / new_h
    if abs(actual_aspect - target_aspect) > 0.01:  # Small tolerance for floating point
        if actual_aspect > target_aspect:
            # Too wide, reduce width
            new_w = int(new_h * target_aspect)
            # Re-center horizontally if possible
            if new_x + new_w <= img_w:
                pass  # Already good
            else:
                new_x = max(0, img_w - new_w)
        else:
            # Too tall, reduce height
            new_h = int(new_w / target_aspect)
            # Re-center vertically if possible
            if new_y + new_h <= img_h:
                pass  # Already good
            else:
                new_y = max(0, img_h - new_h)
    
    bounds = (new_x, new_y, new_w, new_h)

    # Crop the image to the calculated area (which includes padding around content)
    cropped_image = image[new_y:new_y+new_h, new_x:new_x+new_w]
    # Resize to target size (this preserves the letterboxing/pillarboxing)
    resized_image = cv2.resize(cropped_image, size)

    return resized_image, bounds

def save_ndarray_as_tempfile(ndarray, suffix=TEMP_FILE_SUFFIX, prefix=TEMP_FILE_PREFIX, request_id=None):
    """Save numpy array as a temporary image file.
    
    Args:
        ndarray: Numpy array representing an image
        suffix: File extension (default: .png)
        prefix: Filename prefix for temp file
        request_id: Optional request ID to include in filename for traceability
        
    Returns:
        str: Path to the saved temporary file
    """
    # Convert numpy array to PIL Image using existing function
    import numpy as np
    from PIL import Image
    
    if len(ndarray.shape) == 2:
        # Grayscale image
        pil_image = Image.fromarray(ndarray, mode='L')
    elif len(ndarray.shape) == 3:
        if ndarray.shape[2] == 3:
            # Assume RGB or BGR - OpenCV uses BGR by default
            if hasattr(ndarray, 'dtype') and ndarray.dtype == np.uint8:
                # For OpenCV BGR images, convert to RGB
                rgb_array = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_array, mode='RGB')
            else:
                # Assume already RGB
                pil_image = Image.fromarray(ndarray, mode='RGB')
        elif ndarray.shape[2] == 4:
            # RGBA image
            pil_image = Image.fromarray(ndarray, mode='RGBA')
        else:
            raise ValueError(f"Unsupported array shape: {ndarray.shape}")
    else:
        raise ValueError(f"Unsupported array dimensions: {len(ndarray.shape)}")
    
    # Create filename with timestamp and request_id if provided
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if request_id:
        file_prefix = f"{prefix}{timestamp}_{request_id}_"
    else:
        file_prefix = f"{prefix}{timestamp}_"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=file_prefix) as temp_file:
        temp_path = temp_file.name
        pil_image.save(temp_path)
    
    return temp_path


def save_pil_image_as_tempfile(pil_image, suffix=TEMP_FILE_SUFFIX, prefix=TEMP_FILE_PREFIX, request_id=None):
    """Save PIL Image as a temporary file.
    
    Args:
        pil_image: PIL Image object
        suffix: File extension (default: .png)
        prefix: Filename prefix for temp file
        request_id: Optional request ID to include in filename for traceability
        
    Returns:
        str: Path to the saved temporary file
    """
    # Create filename with timestamp and request_id if provided
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if request_id:
        file_prefix = f"{prefix}{timestamp}_{request_id}_"
    else:
        file_prefix = f"{prefix}{timestamp}_"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=file_prefix) as temp_file:
        temp_path = temp_file.name
        pil_image.save(temp_path)
    
    return temp_path

def ndarray_to_temp_file(ndarray, prefix=TEMP_FILE_PREFIX, suffix=TEMP_FILE_SUFFIX, request_id=None) -> str:
    """Save numpy array to a temporary file and return the file path.
    
    Args:
        ndarray: Numpy array representing an image
        prefix: Prefix for temporary filename
        suffix: File extension (default: .png)
        request_id: Optional request ID to include in filename for traceability
        
    Returns:
        str: Path to the temporary file
    """
    import tempfile
    import numpy as np
    
    # Convert numpy array to PIL Image first
    if len(ndarray.shape) == 2:
        # Grayscale image
        pil_image = Image.fromarray(ndarray, mode='L')
    elif len(ndarray.shape) == 3:
        if ndarray.shape[2] == 3:
            # Assume RGB or BGR - OpenCV uses BGR by default
            if hasattr(ndarray, 'dtype') and ndarray.dtype == np.uint8:
                # For OpenCV BGR images, convert to RGB
                rgb_array = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_array, mode='RGB')
            else:
                # Assume already RGB
                pil_image = Image.fromarray(ndarray, mode='RGB')
        elif ndarray.shape[2] == 4:
            # RGBA image
            pil_image = Image.fromarray(ndarray, mode='RGBA')
        else:
            raise ValueError(f"Unsupported array shape: {ndarray.shape}")
    else:
        raise ValueError(f"Unsupported array dimensions: {len(ndarray.shape)}")
    
    # Create filename with timestamp and request_id if provided
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if request_id:
        file_prefix = f"{prefix}{timestamp}_{request_id}_"
    else:
        file_prefix = f"{prefix}{timestamp}_"
    
    # Create temporary file
    fd, temp_path = tempfile.mkstemp(prefix=file_prefix, suffix=suffix)
    try:
        # Close the file descriptor since PIL will open its own
        os.close(fd)
        # Save the image
        pil_image.save(temp_path)
        return temp_path
    except Exception as e:
        # Clean up on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


def pil_to_temp_file(pil_image, prefix=TEMP_FILE_PREFIX, suffix=TEMP_FILE_SUFFIX, request_id=None) -> str:
    """Save PIL Image to a temporary file and return the file path.
    
    Args:
        pil_image: PIL Image object
        prefix: Prefix for temporary filename
        suffix: File extension (default: .png)
        request_id: Optional request ID to include in filename for traceability
        
    Returns:
        str: Path to the temporary file
    """
    import tempfile
    
    # Create filename with timestamp and request_id if provided
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if request_id:
        file_prefix = f"{prefix}{timestamp}_{request_id}_"
    else:
        file_prefix = f"{prefix}{timestamp}_"
    
    # Create temporary file
    fd, temp_path = tempfile.mkstemp(prefix=file_prefix, suffix=suffix)
    try:
        # Close the file descriptor since PIL will open its own
        os.close(fd)
        # Save the image
        pil_image.save(temp_path)
        return temp_path
    except Exception as e:
        # Clean up on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


def cleanup_temp_file(file_path: str | Path) -> bool:
    """Clean up a temporary file.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        bool: True if successful, False if file doesn't exist or error
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            return True
        return False
    except Exception:
        return False

# Image message receiver utilities
def extract_image_from_message(message: dict|ImageSource, 
                              prefer_uri: bool = True) -> Optional[Union[str, Image.Image]]:
    """Extract image data from a ZMQ message created by prepare_image_message.
    
    This is the receiver-side counterpart to prepare_image_message() and handles
    all the transport modes: FILE_URI, BASE64, and HYBRID.
    
    Args:
        message: ZMQ message dict containing image data
        prefer_uri: If True and both uri/image_data exist, prefer URI (more efficient for local files)
        
    Returns:
        - File path string (if using URI transport and file exists)
        - PIL Image object (if using base64 transport or loading from file)
        - None if no valid image data found
        
    Example:
        message = {
            "uri": "file:///tmp/image.png",
            "image_data": "data:image/png;base64,iVBORw0KGgoAAAA...",
            "mask_id": "test_mask"
        }
        
        # Get file path (if prefer_uri=True and file exists)
        image = extract_image_from_message(message, prefer_uri=True)
        
        # Or get PIL Image object
        image = extract_image_from_message(message, prefer_uri=False)
    """
    uri = message.get("uri")
    image_data = message.get("image_data")
    
    if not uri and not image_data:
        return None
    
    # Try URI first if preferred and available
    if prefer_uri and uri:
        file_path = uri_to_file_path(uri)
        if file_path and os.path.exists(file_path):
            return file_path
    
    # Try base64 data if available
    if image_data and image_data.startswith(DATA_URL_PREFIX):
        try:
            return base64url_to_png(image_data)
        except Exception:
            pass
    
    # Fall back to URI if base64 failed or wasn't preferred
    if uri:
        file_path = uri_to_file_path(uri)
        if file_path and os.path.exists(file_path):
            return file_path
    
    return None


def load_image_from_message(message: dict|ImageSource, 
                           format: ImageLoadFormat = ImageLoadFormat.PIL) -> Optional[Union[Image.Image, 'np.ndarray', str, tuple]]:
    """Load image from ZMQ message in the specified format.
    
    Args:
        message: ZMQ message dict containing image data
        format: ImageLoadFormat enum specifying desired return format
        
    Returns:
        - PIL Image if format=ImageLoadFormat.PIL
        - numpy array if format=ImageLoadFormat.NUMPY
        - tuple (file_path, is_temp_file) if format=ImageLoadFormat.FILEPATH
        - base64 encoded string if format=ImageLoadFormat.ENCODED
        - None if loading failed
        
    Example:
        # Load as PIL Image
        pil_image = load_image_from_message(message)
        
        # Load as numpy array
        np_array = load_image_from_message(message, ImageLoadFormat.NUMPY)
        
        # Load for systems that need file paths (like pyglet)
        file_path, is_temp = load_image_from_message(message, ImageLoadFormat.FILEPATH)
        try:
            # Use file_path with your system
            pass
        finally:
            if is_temp:
                cleanup_temp_file(file_path)
    """
    if format == ImageLoadFormat.FILEPATH:
        return _ensure_file_path(message)
    
    # Standard path: extract image data and convert as needed
    image_data = extract_image_from_message(message, prefer_uri=True)
    
    if image_data is None:
        return None
    
    try:
        if isinstance(image_data, str):
            # It's a file path, load from file
            from PIL import Image
            pil_image = Image.open(image_data)
        else:
            # It's already a PIL Image
            pil_image = image_data
        
        if format == ImageLoadFormat.NUMPY:
            try:
                import numpy as np
                return np.array(pil_image)
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.error("NumPy is not available but NUMPY format was requested")
                return None
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to convert PIL Image to numpy array: {e}")
                return None
        elif format == ImageLoadFormat.ENCODED:
            return png_to_base64url(pil_image) # FIXME: how do we know this is a png?
        else:  # format == ImageLoadFormat.PIL
            return pil_image
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load image from message: {e}")
        return None


def _ensure_file_path(message: dict|ImageSource) -> Optional[tuple]:
    """Ensure we get a file path from message, creating temp file if needed.
    
    Args:
        message: ZMQ message dict containing image data
        
    Returns:
        Tuple of (file_path, is_temp_file) or None if failed
        Caller must cleanup temp file if is_temp_file is True
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # First check if we have a URI with an existing file
        if "uri" in message:
            file_path = uri_to_file_path(message["uri"])
            if file_path and os.path.exists(file_path):
                return (file_path, False)  # Not a temp file
        
        # No existing file, need to create one from image data
        # This will trigger a warning since we're forced to create temp files
        logger.warning("No existing file path found in message - creating temporary file. "
                      "Consider using file_uri transport mode for better performance.")
        
        image_data = extract_image_from_message(message, prefer_uri=False)
        
        if image_data is None:
            return None
            
        if isinstance(image_data, str):
            # It's a file path, but the file doesn't exist - this is an error
            logger.error(f"File path in message does not exist: {image_data}")
            return None
        else:
            # It's a PIL Image, save to temp file
            temp_path = save_pil_image_as_tempfile(image_data)
            return (temp_path, True)  # Is a temp file
            
    except Exception as e:
        logger.error(f"Failed to ensure file path from message: {e}")
        return None


def uri_to_file_path(uri: str) -> Optional[str]:
    """Convert URI to local file path.
    
    Args:
        uri: URI string (e.g., "file:///path/to/image.png" or just "/path/to/image.png")
        
    Returns:
        Local file path or None if invalid
        
    Example:
        path = uri_to_file_path("file:///tmp/image.png")  # Returns "/tmp/image.png"
        path = uri_to_file_path("/tmp/image.png")         # Returns "/tmp/image.png"
    """
    try:
        from urllib.parse import urlparse
        
        if uri.startswith(FILE_URI_PREFIX):
            # Remove file:// prefix
            return uri[len(FILE_URI_PREFIX):]
        elif "://" not in uri:
            # Assume it's already a file path
            return uri
        else:
            # Try to parse as URI
            parsed = urlparse(uri)
            if parsed.scheme == "file":
                return parsed.path
            else:
                return None
                
    except Exception:
        return None


def is_temp_file_message(message: dict|ImageSource) -> bool:
    """Check if message contains temporary file that should be cleaned up.
    
    Args:
        message: ZMQ message dict
        
    Returns:
        True if message contains temporary file marker
    """
    return "_temp_file" in message


def cleanup_message_temp_file(message: dict|ImageSource) -> bool:
    """Clean up temporary file from message if present.
    
    Args:
        message: ZMQ message dict that may contain temp file
        
    Returns:
        True if cleanup was successful or not needed
    """
    temp_file_path = message.get("_temp_file")
    if temp_file_path:
        return cleanup_temp_file(temp_file_path)
    return True


def file_path_to_base64url(file_path: str, format: str = "PNG") -> str:
    """Convert a local file path to a base64 data URL.
    
    Args:
        file_path: Local file path to an image
        format: Output format for the base64 encoding (PNG, JPEG, etc.)
        
    Returns:
        Base64 data URL string (e.g., "data:image/png;base64,...")
        
    Raises:
        ValueError: If file path is invalid or empty
        RuntimeError: If file doesn't exist or image processing fails
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    # Handle file:// URI prefix
    if file_path.startswith(FILE_URI_PREFIX):
        file_path = file_path[len(FILE_URI_PREFIX):]
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise RuntimeError(f"File not found: {file_path}")
    
    try:
        # Load image with PIL and convert to base64
        with Image.open(file_path) as img:
            result = png_to_base64url(img, format=format)
            if result is None:
                raise RuntimeError("Failed to convert image to base64")
            return result
    except Exception as e:
        raise RuntimeError(f"Failed to process image file: {e}")


def extract_image_as_base64(image_source: Optional[Union[dict, 'ImageSource']], image_name: str = "image") -> Optional[str]:
    """Extract image data from ImageSource and convert to base64 data URL.
    
    This is a utility function that handles various ImageSource formats and converts them
    to a standardized base64 data URL format for cloud service compatibility.
    
    Args:
        image_source: ImageSource object or dict containing image data
        image_name: Name for logging purposes (e.g., "depth_map", "reference_image")
        
    Returns:
        Base64 data URL string or None if no valid image data
    """
    if image_source is None:
        return None
        
    try:
        # Handle dict format (from deserialization)
        if isinstance(image_source, dict):
            # Check if already has base64 data
            if image_source.get('image_data'):
                image_data = image_source['image_data']
                # Ensure it's a proper data URL
                if image_data.startswith(DATA_URL_PREFIX):
                    return image_data
                elif not image_data.startswith('data:'):
                    return f"data:image/png;base64,{image_data}"
                else:
                    return image_data
            
            # Check if has URI (file path)
            elif image_source.get('uri'):
                uri = image_source['uri']
                # Convert file path to base64
                if not uri.startswith(('http://', 'https://')):
                    return file_path_to_base64url(uri)
                else:
                    # Return HTTP URLs as-is for now
                    return uri
        
        # Handle ImageSource object
        else:
            # Check if already has base64 data
            if hasattr(image_source, 'image_data') and image_source.image_data:
                image_data = image_source.image_data
                # Ensure it's a proper data URL
                if image_data.startswith(DATA_URL_PREFIX):
                    return image_data
                elif not image_data.startswith('data:'):
                    return f"data:image/png;base64,{image_data}"
                else:
                    return image_data
            
            # Check if has URI (file path)
            elif hasattr(image_source, 'uri') and image_source.uri:
                uri = image_source.uri
                # Convert file path to base64
                if not uri.startswith(('http://', 'https://')):
                    return file_path_to_base64url(uri)
                else:
                    # Return HTTP URLs as-is for now
                    return uri
                    
            # Try to load using the standard message format
            else:
                base64_data = load_image_from_message(image_source, ImageLoadFormat.ENCODED)
                if base64_data and isinstance(base64_data, str):
                    return base64_data
        
        logger.debug(f"No valid {image_name} data found in image source")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract {image_name} as base64: {e}")
        return None
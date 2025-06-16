import io
import os
from pathlib import Path
import base64
from typing import Optional, Union

import cv2
from PIL import Image

def cv2_img_to_base64url(img):
    image_base64 =  base64.b64encode(cv2.imencode('.png', img)[1]).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


def png_to_base64url(image, format="PNG"):
    format = format.strip(" .")
    buffered = io.BytesIO()
    image.save(buffered, format=format.upper())
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{image_base64}"

# cv2 image to base64url
# def ndarray_to_base64url(ndarray):
#     return png_to_base64url(Image.fromarray(ndarray))


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
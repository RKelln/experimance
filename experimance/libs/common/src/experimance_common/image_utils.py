import io
import os
from pathlib import Path
import base64

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


def get_mock_images(path:str) -> list:
    # use the images in the folder for mock generation
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
def crop_to_content(image, size:tuple=(1024, 1024), bounds:tuple=None):

    if bounds is None:
        # Find the bounding box of the content
        non_zero_pixels = cv2.findNonZero(image)
        x, y, w, h = cv2.boundingRect(non_zero_pixels)
        bounds = (x, y, w, h)

    # ensure equal width and height and update center
    x, y, w, h = bounds
    if w <= 0 or h <= 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    
    if w > h:
        y = max(0, y - (w - h) // 2)
        h = w
    else:
        x = max(0, x - (h - w) // 2)
        w = h
    bounds = (x, y, w, h)

    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Resize the image to the desired size
    resized_image = cv2.resize(cropped_image, size)

    return resized_image, bounds
import argparse
import json
import os
import requests
import time
from random import randint
from pathlib import Path
import base64
from io import BytesIO
from itertools import cycle
from typing import Generator
import re
import multiprocessing as mp

import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image
from blessed import Terminal

from experimance_common.image_utils import get_mock_images, convert_images_to_mp4, crop_to_content


base_save_path = Path("saved_data")
change_threshold_resolution = (128,128)
# Morphological operations kernel (smaller for reduced image size)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

DEFAULT_DEPTH_RESOLUTION = (1280, 720)
DEFAULT_FPS = 6
DEFAULT_OUTPUT_RESOLUTION = (1024, 1024)

term = Terminal()
def print_status(message:str, style:str='info'):
    if style == 'info':
        color = term.lightskyblue
    elif style == 'warning':
        color = term.orange
    elif style == 'error':
        color = term.red3
    else:
        color = term.snow
    m = message.split(":", maxsplit=1)
    if len(m) == 2:
        message = f"{term.normal}{m[0]}:{term.bold}{m[1]}{term.normal}" 
    message = f"{message}{term.clear_eol}"
    print(color(message), sep='', end='\r', flush=True)


def mask_bright_area(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the center of the image
    center = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)

    # Apply a threshold to find bright areas
    _, thresholded_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Thresholded', thresholded_image)

    # Create a mask initialized to zeros (black)
    mask = np.zeros_like(gray_image)

    # Use floodFill to find the contiguous bright area from the center
    # Note: The mask used by floodFill needs to be 2 pixels larger than the source image
    flood_fill_mask = np.zeros((gray_image.shape[0] + 2, gray_image.shape[1] + 2), np.uint8)
    cv2.floodFill(thresholded_image, flood_fill_mask, center, (255,), loDiff=(20,), upDiff=(20,), flags=cv2.FLOODFILL_MASK_ONLY)

    # The actual mask used for flood fill is 1 pixel larger all around, so we need to crop it
    cropped_mask = flood_fill_mask[1:-1, 1:-1]

    # Find the contours of the bright area
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the contours
    cv2.drawContours(cropped_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return cropped_mask


def circle_mask(image):
    # Calculate the center of the image
    image_center = (image.shape[1] / 2, image.shape[0] / 2)

    # mask
    mask = np.zeros_like(image)

    # Detect circles
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 1, param1=50, param2=30, minRadius=50, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Initialize variables to keep track of the largest circle nearest to the center
        max_radius = 0
        min_radius = int(image.shape[1] / 2)
        best_circle = None
        nearest_distance = float('inf')

        # Iterate through all found circles
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            # Calculate distance from the center of the image to the circle's center
            distance = np.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)

            # Update if this circle is smaller and nearer to the center than the ones before
            if distance < nearest_distance:
                best_circle = i
                min_radius = radius
                nearest_distance = distance

        # Proceed only if a nearest circle is found
        if best_circle is not None:
            # Create a mask with the same dimensions as the original image, initialized to black
            mask = np.zeros_like(image)

            # Draw the nearest and largest circle on the mask
            cv2.circle(mask, (best_circle[0], best_circle[1]), best_circle[2], (255, 255, 255), thickness=-1)

    return mask


# return true /false if obsctruction detected or None if test fails
def simple_obstruction_detect(image, size=(32,32), pixel_threshold=0):
    # look for pixels that are black (0) inside the central circle

    # rescale for faster processing
    image = cv2.resize(image, size)

    if is_blank_frame(image):
        return None

    thickness_multiplier = 0.3
    circle_diameter = int(size[0] * (1.0 + thickness_multiplier))
    circle_radius = circle_diameter // 2
    circle_center = (size[0] // 2, size[1] // 2)

    # turn everything outside the circle into white
    cv2.circle(image, circle_center, circle_radius, (255, 255, 255), thickness=int(size[0] * thickness_multiplier))

    #cv2.imshow('Obstruction', image)

    # count pure black pixels inside the circle in the image
    not_black_pixels = cv2.countNonZero(image)
    black_pixels = size[0] * size[1] - not_black_pixels
    #print("Black pixels:", black_pixels, pixel_threshold)
    return black_pixels > pixel_threshold


def depth_to_contour_map(ndarray_image, step=10):
    # Create empty destination matrix with same size and type as input
    normalized_depth = np.zeros_like(ndarray_image)
    cv2.normalize(ndarray_image, normalized_depth, 0, 255, cv2.NORM_MINMAX)
    normalized_depth = normalized_depth.astype(np.uint8)
    
    # Apply color map
    colored_image = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    
    # Find contours
    contours, _ = cv2.findContours(normalized_depth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    cv2.drawContours(colored_image, contours, -1, (0, 0, 0), 1)
    
    return colored_image


def depth_to_contour_map_matplot(ndarray_image, step=10):
    """
    Converts a grayscale depth map to an elevation contour map with false coloring.

    Parameters:
    - ndarray_image: np.ndarray, the grayscale depth map.
    - step: int, the step size that controls the number of contour lines.

    Returns:
    - contour_map: np.ndarray, the contour map as an image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Ensure the input image is grayscale
    if len(ndarray_image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")
    
    # Normalize the depth map and apply the color map
    normalized_depth_map = cv2.normalize(ndarray_image, None, 0, 255, cv2.NORM_MINMAX)
    colored_image = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_JET)

    # Get the dimensions of the input image
    height, width = ndarray_image.shape

    # Create a figure and axis to draw on, matching the size of the input image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(colored_image, alpha=0.8)
    ax.contour(ndarray_image, levels=np.arange(0, 256, step), colors='black', linewidths=0.5)
    ax.axis('off')

    # FIXME: must be a better way to draw on to a ndarray?!

    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Convert the BytesIO object to a numpy array
    contour_map_pil = Image.open(buf)
    contour_map = np.array(contour_map_pil)

    return contour_map


# Function to detect blank images
def is_blank_frame(image, threshold = 1.0):
    if np.std(image) < threshold:
        print_status("blank frame detected", style='warning')
        return True
    return False


# Function to clip and normalize the image
# def clip_and_normalize(image, low_clip:int=15, high_clip:int=240, alpha:int=0, beta:int=255):
#     if low_clip == 0 and high_clip == 255: # no mask needed
#         return cv2.normalize(image, None, alpha, beta, cv2.NORM_MINMAX)
#     mask = cv2.inRange(image, low_clip, high_clip)
#     return cv2.normalize(image, None, alpha, beta, cv2.NORM_MINMAX, mask=mask)


# returns amount of difference between two images
# returns difference and frame to use for next comparison
def detect_difference(image1, image2, threshold=60):
    # no previous image yet
    if image1 is None:
        return threshold+1, image2

    # ignore blank frames
    if is_blank_frame(image2):
        return 0, image1
    
    # Normalize the image to account for luminance changes
    # (assume image1 is already normalized)
    # EDIT: all mages should be normalized already
    #image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX)
    #image2 = clip_and_normalize(image2)

    # Calculate the absolute difference between current and previous frames
    depth_diff_1 = cv2.absdiff(np.asanyarray(image1), np.asanyarray(image2))
    
    _, depth_diff_2 = cv2.threshold(depth_diff_1, threshold, 255, cv2.THRESH_BINARY)

    # Apply adaptive thresholding to the difference to detect changes
    #depth_diff_2 = cv2.adaptiveThreshold(depth_diff_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 7)

    # Morphological operations to remove noise
    depth_diff_3 = cv2.morphologyEx(depth_diff_2, cv2.MORPH_CLOSE, kernel)
    depth_diff_3 = cv2.morphologyEx(depth_diff_3, cv2.MORPH_OPEN, kernel)

    #cv2.imshow('Depth Difference', np.hstack((image1, image2, depth_diff_1, depth_diff_2, depth_diff_3)))

    # Count non-zero pixels
    diff = cv2.countNonZero(depth_diff_3)

    #print("Difference:", diff)

    return diff, image2

def depth_camera_frame_generator(json_config=None,
                   size:tuple=DEFAULT_DEPTH_RESOLUTION, 
                   fps:int=DEFAULT_FPS, 
                   align:bool=True,
                   min_depth:float=0,
                   max_depth:float=50,
                   colorizer_flag=2):
    
    # Configure the streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, fps)
    if align:
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, fps)
    
    # Try to start pipeline with auto-reset on "Couldn't resolve requests" error
    max_retries = 3
    for attempt in range(max_retries):
        try:
            profile = pipeline.start(config)
            break  # Success, exit retry loop
        except RuntimeError as e:
            print(f"Camera initialization failed (attempt {attempt + 1}/{max_retries}): {e}")
            #print_status(f"Camera initialization failed (attempt {attempt + 1}/{max_retries}): {e}", style='warning')
            if attempt < max_retries - 1:  # Don't reset on last attempt
                print_status("Attempting camera reset...", style='info')
                time.sleep(3)  # Wait a moment before retrying
                if reset_realsense_camera():
                    print_status("Retrying camera initialization...", style='info')
                    continue
                else:
                    print_status("Camera reset failed, retrying anyway...", style='warning')
                    continue
            else:
                # the last attempt, re-raise
                raise

    if json_config is not None:
        # check config path exists
        config_path = Path(json_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        jsonObj = json.load(open(config_path))
        json_string = str(jsonObj).replace("'", '\"')

        device = profile.get_device()
        advnc_mode = rs.rs400_advanced_mode(device)
        advnc_mode.load_json(json_string)

    # define the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()

    # what are the preset options?
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    current_preset = depth_sensor.get_option(rs.option.visual_preset)
    print('preset range: '+str(preset_range)+str(current_preset))

    # set the visual reset to high accuracy
    for i in range(int(preset_range.max)):
        visualpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        print('%02d: %s'%(i,visualpreset))
        if visualpreset == "High Accuracy":
            depth_sensor.set_option(rs.option.visual_preset, i)
            current_preset = depth_sensor.get_option(rs.option.visual_preset)
            print('current preset: '+str(current_preset))    

    # Create colorizer object
    colorizer = rs.colorizer(colorizer_flag) # FIXME: doesn't work?
    # colorize using white to black (near to far)
    # 0 - Jet
    # 1 - Classic
    # 2 - WhiteToBlack
    # 3 - BlackToWhite
    # 4 - Bio
    # 5 - Cold
    # 6 - Warm
    # 7 - Quantized
    # 8 - Pattern
    # 9 - Hue
    
    # set min and max distances in colorizer
    colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
    colorizer.set_option(rs.option.min_distance, min_depth)
    colorizer.set_option(rs.option.max_distance, max_depth)

    colorizer.set_option(rs.option.color_scheme, colorizer_flag) # NOTE: seems like this needs to be done after the visual preset is set


    # Alignment and clipping distance
    # See: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

    # if max_depth > 0:
    #     # Getting the depth sensor's depth scale (see rs-align example for explanation)
    #     depth_sensor = profile.get_device().first_depth_sensor()
    #     depth_scale = depth_sensor.get_depth_scale()
    #     print("Depth Scale is: " , depth_scale)

    #     # We will be removing the background of objects more than
    #     #  clipping_distance_in_meters meters away
    #     clipping_distance = max_depth / depth_scale
    # else:
    #     clipping_distance = 0

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    if align:
        align_to = rs.stream.color
        align = rs.align(align_to)

    try:
        color_image = None
        depth_image = None

        while True:
            frames = pipeline.wait_for_frames()

            if align:
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
            else:
                depth_frame = frames.get_depth_frame()
                color_image = None
            
            if not depth_frame:
                yield None, color_image

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)
            
            # Convert depth_frame to numpy array to render image in opencv
            depth_colormap = np.asanyarray(depth_color_frame.get_data())

            # expand the range of colors to the full range
            # RK: disabled for now because it causes excessive flickering
            #depth_colormap = cv2.normalize(depth_colormap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #depth_colormap = clip_and_normalize(depth_colormap, alpha=15, beta=240)

            # Convert to grayscale if it's not already (in case it's a color image)
            if len(depth_colormap.shape) == 3 and depth_colormap.shape[2] == 3:
                depth_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            else:
                depth_image = depth_colormap
            
            yield depth_image, color_image

            time.sleep(1/fps)  # Optional: Sleep for a short duration before capturing the next frame
    except Exception as e:
        print("Error in pipeline:", e)
        raise
    finally:
        print("Stopping pipeline")
        pipeline.stop()
        #time.sleep(0.25) # FIXME: help pipeline stop properly


def mock_depth_frame_generator(json_config=None, size=(640, 480), fps=30, align=True, min_depth=0.0, max_depth=10.0):
    # Placeholder for the actual implementation of the generator
    obstruction = False
    while True:
        # Simulate depth frames
        depth_image = np.random.randint(0, 256, size, dtype=np.uint8)
        #color_image = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        cv2.imshow('Mock', depth_image)
        yield depth_image, obstruction


def depth_generator( json_config=None,
                   size:tuple=DEFAULT_DEPTH_RESOLUTION, 
                   fps:int=30, 
                   recording:bool=True,
                   align:bool=True,
                   min_depth:float=0.0,
                   max_depth:float=10.0,
                   change_threshold:int=0,
                   detect_hands:bool=False,
                   crop:bool=False,
                   output_size:tuple=DEFAULT_OUTPUT_RESOLUTION,
                   test=False,
                   warm_up_period=10,
                   mock=None,
                   ):
    
    # output settings
    print("\n### Depth camera settings ###")
    if test: print("Test mode enabled")
    print("Depth resolution:", size)
    print("Output resolution:", output_size)
    print("Depth cam config:", json_config)
    print("FPS:", fps)
    print("Min depth:", min_depth)
    print("Max depth:", max_depth)
    print("Change threshold:", change_threshold)
    print("Detect hands:", detect_hands)
    print("Crop:", crop)
    print("Warm-up period:", warm_up_period)
    print("")


    # Create directories to save data
    if recording:
        save_path = base_save_path / time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    if mock is not None:
        if mock == True:
            depth_frame_gen = mock_depth_frame_generator()
        elif isinstance(mock, str):
            mock_images = get_mock_images(mock)
            depth_frame_gen = mock_depth_generator(mock_images)
        elif isinstance(mock, list):
            depth_frame_gen = mock_depth_generator(mock)
        elif callable(mock):
            depth_frame_gen = mock()
        else:
            raise ValueError("Invalid mock parameter")
    else:
        depth_frame_gen = depth_camera_frame_generator(
            json_config=json_config, 
            size=size, 
            fps=fps, 
            align=align, 
            min_depth=min_depth, 
            max_depth=max_depth,
            colorizer_flag=0 if test else 2) 

    changed = False
    obstruction = False
    crop_bounds = None
    prev_masked_image = None  # for change detection
    prev_crop_bounds = None   # for cropping, to prefer to use the same center
    warm_up = True

    try:
        frame_number = 0
        while True:
            depth_image, _ = next(depth_frame_gen)
                
            # Validate that both frames are valid
            if depth_image is None:
                continue

            frame_number += 1
            #print("Captured frame: ", frame_number, "warmup", warm_up_period, end="\n")
            
            importance_mask = mask_bright_area(depth_image)
            #cv2.imshow('Importance Mask', importance_mask)

            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(depth_image, depth_image, mask=importance_mask)

            # crop and resize the masked image (remove excess black, resize to 1024 width & height)
            if crop:
                output, crop_bounds = crop_to_content(masked_image, size=output_size, bounds=prev_crop_bounds)
            else:
                if output_size != (masked_image.shape[1], masked_image.shape[0]):
                    output = cv2.resize(masked_image, output_size)
                else:
                    output = masked_image

            if detect_hands:
                # detect hands in the masked image
                obs = simple_obstruction_detect(output, pixel_threshold=1)
                if obs is not None: # if valid result
                    if obs != obstruction: # update on changed value only
                        obstruction = obs
                        print_status(f"Obstruction: {obstruction}")
                        
                # no change while obstructed
                if obstruction:
                    changed = False
                    if warm_up: # extend warm-up period until hands are gone
                        warm_up_period += 1

            if warm_up or change_threshold <= 0 or test:
                changed = True # always change
            # set changed based on change threshold, unless we are detecting hands and obstructed
            elif (not detect_hands or not obstruction): # no change dectection if obstruction detected
                # rescale the resolution for faster processing
                small_masked_image = cv2.resize(output, change_threshold_resolution)

                # detect changes between frames
                diff, prev_masked_image = detect_difference(prev_masked_image, small_masked_image)
                #if test and prev_masked_image is not None:
                    #cv2.imshow('Depth Difference threshold', np.hstack((prev_masked_image, small_masked_image)))

                if diff > change_threshold:
                    print_status(f"Change detected: {diff}")
                    changed = True
                else:
                    changed = False
            else: # blocked by obstruction
                changed = False
            
            if test:
                # Display the original and masked image at the same time, side by side
                cv2.imshow('Depth', np.hstack((depth_image, masked_image)))
                # Display cropped depth
                cv2.imshow('Cropped', output)

            #print("warmup", warm_up, "changed", changed, "obstruction", obstruction)
            # if detect_hands and obstruction:
            #     # no change while obstructed
            #     changed = False 

            # warm-up period
            if warm_up:
                changed = True
                if frame_number >= warm_up_period: # always update on first non-warm-up frame
                    prev_crop_bounds = crop_bounds  # lock the crop bounds after warm-up
                    warm_up = False
                    print("Depth cam warm-up period complete\n")
                else:
                    # FIXME: hack to avoid using bad depth images until warm-up is done
                    obstruction = True
            
            if recording:
                pass # TODO

            if changed: # only yield if changed (FIXME?)
                if test:
                    yield depth_to_contour_map(output), obstruction
                else:
                    yield output, obstruction

            time.sleep(0.01)  # Optional: Sleep for a short duration before capturing the next frame
            #time.sleep(1/fps)  # Optional: Sleep for a short duration before capturing the next frame
    except Exception as e:
        print("Error in depth generator:", e)
        raise
    finally:
        print("Stopping depth generator")
        depth_frame_gen.close()
        if test:
            cv2.destroyAllWindows()


def depth_pipeline(output_queue:mp.Queue, depth_factory):

    # we need to create the client inside the process otherwize the ZMQ context will be shared / not work
    depth_gen = depth_factory()
    try:
        # TODO: send changes to detect hands regardless of changs in depth??? 
        for result in depth_gen:
            if result is None:
                break
            depth_image, detected_hands = result
            if depth_image is not None:
                #print("queuing depth image")
                output_queue.put((depth_image, detected_hands))
    finally:
        depth_gen.close()


def mock_depth_generator(mock_depth_images, delay=0.1):
    # use the images in the folder for mock generation
    # 1% chance of fail, 1% chance of black frame, 1% new random frame
    # loop forever

    image = cv2.imread(mock_depth_images[0], cv2.IMREAD_GRAYSCALE)
    yield image, False # return the first image

    while True:
        i = randint(0, 100)
        if i == 0:
            yield None, False
        elif i == 1:
            yield np.zeros_like(image), None
        elif i < 99: # no change
            yield None, False
        else: # new depth image
            image = cv2.imread(mock_depth_images[randint(0,len(mock_depth_images)-1)], cv2.IMREAD_GRAYSCALE)
            yield image, False
        #time.sleep(delay)


def reset_realsense_camera():
    """
    Attempt to reset the RealSense camera hardware.
    
    Returns:
        bool: True if reset was successful, False otherwise
    """
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print('No RealSense devices found for reset')
            return False
        
        dev = devices[0]
        device_name = dev.get_info(rs.camera_info.name)
        print(f'Found device: {device_name}')
        print('Attempting hardware reset...')
        dev.hardware_reset()
        print('Hardware reset successful')
        
        # Wait a moment for the device to reinitialize
        import time
        time.sleep(2)
        return True
        
    except Exception as e:
        print(f'Camera reset failed: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='Capture and send depth frames from a RealSense camera')
    parser.add_argument('-n', '--name', type=str, default=time.strftime("%Y%m%d-%H%M%S"), help='Name of the capture')
    parser.add_argument('-c','--config', type=str, default=None, help='Path to the depth camerma json configuration file')
    # NOTE: reccomended depth resolution:
    # 1280x720 for intel realsense D415 with min operating range of 43.8cm
    # but to reduce the minimum operating range you can reduce resolution
    # https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
    parser.add_argument('-s', '--size', type=int, nargs=2, default=DEFAULT_DEPTH_RESOLUTION, help='Size of the capture in pixels (width height)')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Frames per second')
    parser.add_argument('-r','--recording', action='store_true', help='Record the frames')
    parser.add_argument('-m','--min-depth', '--min_depth', type=float, default=0, help='Minimum depth in meters')
    parser.add_argument('-M','--max-depth', '--max_depth', type=float, default=10, help='Maximum depth in meters')
    parser.add_argument('-a','--align', action='store_true', help='Align depth and color frames')
    parser.add_argument('--change-threshold', '--change_threshold', type=int, default=0, help='Threshold for detecting depth changes')
    parser.add_argument('--hand_detection', '--hand-detection', '--detect_hands', '--detect-hands', action='store_true', help='Detect hands in the depth image')
    parser.add_argument('--crop', action='store_true', help='Crop the image to the content')
    parser.add_argument('--output-size', '--output_size', type=int, nargs=2, default=DEFAULT_OUTPUT_RESOLUTION, help='Size of the output image in pixels (width height)')
    parser.add_argument('--test', action='store_true', help="Testing mode")
    parser.add_argument('--mock-depth', '--mock_depth', type=str, default=None, help="Path to a folder of images to use for mock depth maps")
    args = parser.parse_args()

    depth_gen = None
    if args.mock_depth is not None:
        mock_images = get_mock_images(args.mock_depth)
        depth_gen = mock_depth_generator(mock_images)
    else:
        print("Starting depth generator")
        print("Deth cam config:", args.config)
        print("Depth cam resolution (--size):", args.size)
        print("FPS:", args.fps)
        print("Output:", args.output_size)

        depth_gen = depth_generator(
                        json_config=args.config,
                        size=tuple(args.size), 
                        fps=args.fps, 
                        recording=args.recording, 
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        align=args.align,
                        change_threshold=args.change_threshold,
                        detect_hands=args.hand_detection,
                        crop=args.crop,
                        test=args.test,
                        output_size=tuple(args.output_size),
                    )
    try:
        for result in depth_gen:
            if result is None:
                break
            depth_image, detect_hands = result
            #print("Detect hands", detect_hands, end="\r")
            if depth_image is not None:
            #     # display cv image
                cv2.imshow('Final', depth_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Stopping depth generator")
        depth_gen.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

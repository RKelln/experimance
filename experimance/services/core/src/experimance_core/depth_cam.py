import argparse
import os
import time
from pathlib import Path
from collections import deque
import time

from outputs import NullOutput, ShmData, VirtualWebcam, OutputChain, MidiOutput, OSCOutput

import numpy as np
import cv2
import pyrealsense2 as rs


# IMPORTANT NOTE:
#
# start the virtual webcam first! e.g.
# $ sudo modprobe v4l2loopback card_label="VirtualCam"
#
# check with device is created:
# $ v4l2-ctl --list-devices
#
# To remove:
# $ sudo modprobe -r v4l2loopback

# Basic controls (while debug window active):
#
CONTROLS = """
 q or escape key to quit
 c key to toggle clean mode (morphological operations to clean up the thresholded image)
 r key to set reference image (for background subtraction)
 d toggle debug mode (display the mask image)
 -+ keys to decrease / incr`ease threshold
 e key to toggle EMA (Exponential Moving Average) mode
 0-9 keys to set frame queue size (0 to disable multi-frame calculations)
 <> keys to decrease / increase EMA alpha
 [] keys to decrease / increase kernel size
 :' keys to decrease / increase clipping distance
 s toggle segementation mode (pose detection)
 p key to print debug info
"""

# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARKS = {
    "nose": 0,
    "left eye (inner)": 1,
    "left eye": 2,
    "left eye (outer)": 3,
    "right eye (inner)": 4, 
    "right eye": 5, 
    "right eye (outer)": 6,
    "left ear": 7, 
    "right ear": 8, 
    "mouth (left)": 9, 
    "mouth (right)": 10,
    "left shoulder": 11,
    "right shoulder": 12,
    "left elbow": 13,
    "right elbow": 14, 
    "left wrist": 15,
    "right wrist": 16, 
    "left pinky": 17,
    "right pinky": 18,
    "left index": 19,
    "right index": 20,
    "left thumb": 21,
    "right thumb": 22,
    "left hip": 23,
    "right hip": 24,
    "left knee": 25,
    "right knee": 26,
    "left ankle": 27,
    "right ankle": 28,
    "left heel": 29,
    "right heel": 30,
    "left foot index": 31, 
    "right foot index": 32,
}


def get_patch(array, x, y, patch_size):
    start_x = max(0, x - patch_size // 2)
    start_y = max(0, y - patch_size // 2)
    end_x = start_x + patch_size
    end_y = start_y + patch_size
    patch = array[start_y:end_y, start_x:end_x]
    return patch

def capture_frames(name="capture", 
                   depth=True,
                   color=True,
                   depth_size=(640, 480),
                   color_size=(640, 480),
                   fps=30,
                   clip_distance=0, # meters
                   align=True,
                   background_color=0,
                   foreground_color=255,
                   debug=False,
                   outputter=None,
                   clean=False,
                   reference_depth=None, 
                   multi=0,
                   ema_alpha=0,
                   kernel_size=5,
                   threshold=25,
                   input=None,
                   pose_detection=False,
                   segment=True,
                   temporal_filter=True,
                   spatial_filter=True,
                   num_people=3,
                   landmark_filter=None,
                   perf=0,
                   combine=True, # combine depth and color for pose & segmentation
                   gamma=1.0,
                   ):

    if outputter is None:
        outputter = NullOutput()

    if multi > 0:
        frame_queue = deque(maxlen=multi)
    else:
        frame_queue = None

    ema_color = None
    def init_ema(size):
        nonlocal ema_color
        ema_color = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if ema_alpha > 0:
        init_ema(depth_size)

    # Configure the streams
    pipeline = rs.pipeline()
    config = rs.config()
    if input is not None and input != "":
        print("Reading from file:", input)
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/read_bag_example.py
        rs.config.enable_device_from_file(config, input, repeat_playback=True)
        # start the pipeline so we can get the data format
        profile = pipeline.start(config)
        try:
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()
            print("Depth info:", depth_intrinsics)
            depth_size = (depth_intrinsics.width, depth_intrinsics.height)
            fps = depth_profile.fps()
        except RuntimeError:
            print("No depth stream in bag file")
            depth = False
            align = False # need depth to align
        try:
            rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            rgb_intrinsics = rgb_profile.get_intrinsics()
            print("RGB info:", rgb_intrinsics)
            color_size = (rgb_intrinsics.width, rgb_intrinsics.height)
            fps = rgb_profile.fps()
        except RuntimeError:
            print("No color stream in bag file")
            color = False
            align = False # need color to align
    else:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No camera detected")
            return
        
        config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)
        config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps)
        profile = pipeline.start(config)

        


    # Create colorizer object
    colorizer = rs.colorizer(2)

    # Alignment and clipping distance
    # See: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance = 0
    
    # FIXME? use rs2_project_color_pixel_to_depth_pixel
    # also see: https://github.com/IntelRealSense/librealsense/issues/5603#issuecomment-574019008
    # NOTE: the methods maybe reversed because of this bug:
    # https://github.com/IntelRealSense/librealsense/commit/b831660f50d9e54b8a2ec7873bf491aad14fb100
    # currently using simple version from: 
    # https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
    def get_depth(aligned_depth_frame, x, y, patch_size=5, min_d=0.0, max_d=0.0):
        nonlocal depth_scale

        # if x and y are floats then treat as percentage of width and height
        if isinstance(x, float):
            x = int(x * color_size[0])
        if isinstance(y, float):
            y = int(y * color_size[1])

        # clamp x and y to color image size
        x = max(0, min(x, color_size[0] - 1))
        y = max(0, min(y, color_size[1] - 1))

        # retrieve a patch_size depth patch around the given pixel
        depth_patch = get_patch(aligned_depth_frame, x, y, patch_size)
        if depth_patch is None or depth_patch.size == 0:
            return 0
        depth_patch = depth_patch * depth_scale
        dist = np.mean(depth_patch)

        if dist > 0 and max_d > 0:
            # clamp to min and max distance
            dist = max(min_d, min(max_d, dist))
            # normalize to 0-1
            dist = (dist - min_d) / (max_d - min_d)

        return dist

    def get_depth_fn(aligned_depth_frame, patch_size=5, min_d=0.0, max_d=0.0):
        return lambda x, y: get_depth(aligned_depth_frame, x, y, patch_size, min_d=min_d, max_d=max_d)

    def set_clip(d):
        nonlocal clipping_distance
        nonlocal clip_distance
        if clip_distance > 0:
            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            clipping_distance = clip_distance / depth_scale
        else:
            clipping_distance = 0
    set_clip(clip_distance)

    foreground_mask = None
    color_mask = None
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    if depth and color and align:
        align_to = rs.stream.color
        align = rs.align(align_to)

    # pose detection setup
    # see: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md#python-solution-api
    # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose.py 
    if debug or pose_detection or segment:
        from mediapipe_pose import PoseLandmarker
        pose = PoseLandmarker(segment=segment, num_poses=num_people)

    start_time_ns = time.monotonic_ns()
    time_ns = start_time_ns
    frame_number = 0
    sec_per_frame = 1 / fps

    # https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md
    temporal_filter = rs.temporal_filter() if temporal_filter else None
    spatial_filter = rs.spatial_filter() if spatial_filter else None
    #decimation_filter = rs.decimation_filter()
    #disparity_transform = rs.disparity_transform()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    

    lookUpTable = np.empty((1,256), np.uint8)
    def set_gamma(gamma):
        nonlocal lookUpTable
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    if gamma != 1.0:
        set_gamma(gamma)

    try:
        outputter.start()

        while True:
            frame_number += 1
            frames = pipeline.wait_for_frames()

            if depth and align:
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
            else:
                if depth:
                    depth_frame = frames.get_depth_frame()
                if color:
                    color_frame = frames.get_color_frame()

            # Validate that both frames are valid
            if (depth and not depth_frame) or (color and not color_frame):
                continue

            if depth:
                # depth post processing
                if spatial_filter is not None or temporal_filter is not None:
                    depth_frame = depth_to_disparity.process(depth_frame)
                    if spatial_filter is not None:
                        depth_frame = spatial_filter.process(depth_frame)
                    if temporal_filter is not None:
                        depth_frame = temporal_filter.process(depth_frame)
                    depth_frame = disparity_to_depth.process(depth_frame)

                depth_color_frame = colorizer.colorize(depth_frame)
                depth_image = np.asanyarray(depth_color_frame.get_data())
                depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
                depth_image.flags.writeable = False # speed optimization
                if not color: # grayscale depth to color
                    color_image = np.dstack((depth_image, depth_image, depth_image)).astype(np.uint8)

            if color:
                color_image = np.asanyarray(color_frame.get_data())
                if gamma != 1.0:
                    color_image = cv2.LUT(color_image, lookUpTable)
            
            color_image.flags.writeable = False # speed optimization
                
            # pose dectection and segmentation
            if pose_detection or segment:
                if combine:
                    # use depth as pseudo alpha channel on color to try to improve color image contrast
                    if depth:
                        alpha = np.dstack((depth_image, depth_image, depth_image)).astype(float)/255
                        color_image = cv2.multiply(alpha, color_image.astype(float)).astype(color_image.dtype)
                # NOTE: can't use get_timestamp() because of looping inputs
                # frame_time_ms = depth_frame.get_timestamp() # ms since camera started
                frame_time_ms = (time.monotonic_ns() - start_time_ns) / 1e6 # ms
                pose_results = pose.detect(color_image, frame_time_ms)

                if segment and pose_results is not None:
                    # segmentation mask not always available every frame
                    seg_mask = pose.get_segmentation_mask(pose_results)
                    if seg_mask is not None:
                        seg_mask = seg_mask * foreground_color
                        seg_mask = cv2.GaussianBlur(seg_mask, (kernel_size, kernel_size), 0)
                        foreground_mask = np.dstack((seg_mask, seg_mask, seg_mask)).astype(np.uint8)

            if foreground_mask is None and foreground_color is not None:
                foreground_mask = np.ones(color_image.shape, dtype=np.uint8) * foreground_color

            if foreground_mask is not None:
                foreground_mask.flags.writeable = False # speed optimization

            if depth:              
                if reference_depth is not None and threshold > 0:
                    # subtract reference from current depth image, min 0
                    diff = cv2.subtract(depth_image, reference_depth)

                    # Smooth the difference image
                    #diff = cv2.GaussianBlur(diff, (kernel_size, kernel_size), 0)

                    # Apply a threshold to the difference image
                    _, depth_image = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                    depth_image.flags.writeable = False # speed optimization
            
                if clipping_distance > 0:
                    df = np.asanyarray(depth_frame.get_data())
                    depth_frame_3d = np.dstack((df, df, df)) # depth frame is 1 channel, color is 3 channels
                    
                    if foreground_mask is not None:
                        color_mask = np.where((depth_frame_3d > clipping_distance) | (depth_frame_3d <= 0), 
                                                background_color, foreground_mask)
                    else:
                        color_mask = np.where((depth_frame_3d > clipping_distance) | (depth_frame_3d <= 0), 
                                                background_color, color_image)
                else:
                    # mask the depth image with the foreground mask
                    if foreground_mask is not None:
                        #color_mask = np.where(foreground_mask > 0, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR), background_color)
                        # FIXME: add background color?
                        dst = foreground_mask.astype(float) / 255. * cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                        color_mask = dst.astype(color_image.dtype)
                    else:
                        color_mask = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

            else: # no depth
                if foreground_mask is not None:
                    #color_mask = np.where(foreground_mask > 0, color_image, background_color)
                    # mask the color image with the greyscale foreground_mask
                    # FIXME: use background and foreground colors
                    dst = foreground_mask.astype(float) / 255. * color_image
                    color_mask = dst.astype(color_image.dtype)
                else:
                    color_mask = np.copy(color_image)

            if ema_alpha > 0 and ema_color is not None:
                # Update the Exponential Moving Average (EMA) with the current depth frame
                ema_color = (1 - ema_alpha) * ema_color + ema_alpha * color_mask
                # add EMA to color image (take max value)
                color_mask = np.maximum(color_mask, ema_color).astype(color_mask.dtype)

            if frame_queue is not None:
                # Add the current frame to the queue
                frame_queue.append(np.copy(color_mask))

                # Compute the intersection of the frames in the queue
                #color_mask = np.min(frame_queue, axis=0).astype(color_mask.dtype)
                #color_mask = np.average(frame_queue, axis=0).astype(color_mask.dtype)
                #color_mask = np.median(frame_queue, axis=0).astype(color_mask.dtype)
                color_mask = np.max(frame_queue, axis=0).astype(color_mask.dtype)
                
            if clean:
                # Apply morphological operations to clean up the thresholded image
                # circular kernel:
                #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            if pose_detection and pose_results is not None:
                if debug:
                    color_mask = pose.draw_landmarks_on_image(color_mask, pose_results)
                
                depth_data = np.asanyarray(depth_frame.get_data())
                if clip_distance > 0:
                    depth_fn = get_depth_fn(depth_data, min_d=1.0, max_d=clip_distance)
                else:
                    depth_fn = get_depth_fn(depth_data)
                outputter.send(image=color_mask, 
                               pose_data=pose.get_world_landmarks(pose_results, 
                                                                  depth_fn=get_depth_fn(depth_data)))
            else:
                outputter.send(image=color_mask)

            if debug:
                if color and segment and foreground_mask is not None:
                    cv2.imshow('Mask', np.vstack((color_mask, foreground_mask)))
                elif reference_depth is not None:
                    # horizontally stack reference and color mask
                    cv2.imshow('Mask', np.hstack((color_mask, cv2.cvtColor(reference_depth, cv2.COLOR_GRAY2BGR))))
                elif combine:
                    cv2.imshow('Mask', np.hstack((color_image, color_mask)))
                else:
                    cv2.imshow('Mask', color_mask)

            # if s key pressed, toggle recording, otherwise if q or escape key pressed, then stop
            k = cv2.waitKey(1) & 0xFF

            if k > 0:
                clear = " " * 20
                if k == ord('q') or k == 27:
                    break
                elif k == ord('c'):
                    clean = not clean
                    print("Clean mode:", clean, clear, end="\r")
                elif k == ord('r'):
                    if reference_depth is not None:
                        reference_depth = None
                        ema_color = None
                        print("Reference image cleared", clear, end="\r")
                    else:
                        # take reference image (for background subtraction)
                        reference_depth = cv2.GaussianBlur(depth_image, (5, 5), 0)
                        #ema_depth = np.float32(reference_depth)
                        ema_color = np.copy(color_image)
                        print("Reference image set", clear, end="\r")
                elif k == ord('d'):
                    debug = not debug
                    if debug:
                        # show the debug window
                        cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
                    if not debug:
                        # hide the debug window
                        cv2.destroyWindow('Mask')
                    print("Debug mode:", clear, debug, end="\r")
                elif k == ord('e'):
                    if ema_color is not None:
                        ema_color = None
                        print("EMA off", clear, end="\r")
                    else:
                        init_ema(depth_size)
                        print("EMA alpha:", ema_alpha, clear, end="\r")
                elif k == ord('-') or k == ord('_'):
                    threshold = max(0, threshold - 1)
                    print("- Threshold:", threshold, clear, end="\r")
                elif k == ord('+') or k == ord('='):
                    threshold = min(255, threshold + 1)
                    print("+ Threshold:", threshold, clear, end="\r")
                elif k >= ord('0') and k <= ord('9'):
                    # set frame queue size
                    multi = k - ord('0')
                    if multi <= 1:
                        frame_queue = None
                        print("Multi-frame disabled", clear, end="\r")
                    else:
                        frame_queue = deque(maxlen=multi)
                        print("Multi-frame size:", multi, clear, end="\r")
                elif k == ord(',') or k == ord('<'):
                    ema_alpha = max(0, ema_alpha - 0.01)
                    print("- EMA alpha:", ema_alpha, clear, end="\r")
                elif k == ord('.') or k == ord('>'):
                    ema_alpha = min(1, ema_alpha + 0.01)
                    print("+ EMA alpha:", ema_alpha, clear, end="\r")
                elif k == ord('[') or k == ord('{'):
                    kernel_size = max(1, kernel_size - 2)
                    print("- Kernel size:", kernel_size, clear, end="\r")
                elif k == ord(']') or k == ord('}'):
                    kernel_size = min(255, kernel_size + 2)
                    print("+ Kernel size:", kernel_size, clear, end="\r")
                elif k == ord(';') or k == ord(":"):
                    clip_distance = max(0, clip_distance - 0.1)
                    set_clip(clip_distance)
                    print("- Clip distance:", clip_distance, clear, end="\r")
                elif k == ord("'") or k == ord('"'):
                    clip_distance = min(20, clip_distance + 0.1)
                    set_clip(clip_distance)
                    print("+ Clip distance:", clip_distance, clear, end="\r")
                elif k == ord('s'):
                    segment = not segment
                    pose = PoseLandmarker(segment=segment, num_poses=num_people)
                    foreground_mask = None
                    print("Segmentation mode:", segment, clear, end="\r")
                elif k == ord('h'):
                    gamma = max(0.1, gamma - 0.1)
                    set_gamma(gamma)
                    print(f"- Gamma: {gamma:.1f}", clear, end="\r")
                elif k == ord('g'):
                    gamma = min(5.0, gamma + 0.1)
                    set_gamma(gamma)
                    print(f"+ Gamma: {gamma:.1f}", clear, end="\r")
                elif k == ord('m'):
                    combine = not combine
                    print("Combine mode:", combine, clear, end="\r")
                elif k == ord('p'):
                    print(clear)
                    print("Debug mode:", debug)
                    print("Depth:", depth)
                    print("Size:", depth_size)
                    print("FPS:", fps)
                    print("Clip distance:", clip_distance)
                    print("Reference image:", reference_depth is not None)
                    print("Clean mode:", clean)
                    print("Threshold:", threshold)
                    print("Frame queue:", len(frame_queue) if frame_queue is not None else 0)
                    print("EMA alpha:", ema_alpha)
                    print("Kernel size:", kernel_size)
                    print("Pose detection:", pose_detection)
                    print("Segmentation mode:", segment)
                    print("Temporal filter:", temporal_filter is not None)
                    print("Spatial filter:", spatial_filter is not None)
                    print("Number of people:", num_people)
                    print("Landmark filter:", landmark_filter)
                    print("Background color:", background_color)
                    print("Foreground color:", foreground_color)
                    print("Gamma:", gamma)

            frame_number += 1
            duration = time.monotonic_ns() - time_ns # in nanoseconds
            time_ns += duration
            duration = duration / 1e9 # to seconds
            if perf > 0 and frame_number % perf == 0:
                print(" " * 20, f"{duration * 1000:.2f} ms/frame", end="\r")
            if duration < sec_per_frame:
                time.sleep(sec_per_frame - duration)  # Optional: Sleep for a short duration before capturing the next frame

    finally:
        pipeline.stop()
        outputter.stop()
        if debug:
            cv2.destroyAllWindows()
            # print settings as CLI arguments
            cli_args = f"-s {depth_size[0]} {depth_size[1]}"
            cli_args += f" --fps {fps}"
            cli_args += f" -d {clip_distance:.1f}"
            if clean:
                cli_args += " --clean"
            if ema_alpha > 0:
                cli_args += f" -e {ema_alpha:.2f}"
            if multi > 0:
                cli_args += f" -m {multi}"
            if pose_detection:
                cli_args += " --pose_detection"
            if segment:
                cli_args += " --segment"
            if temporal_filter:
                cli_args += " -tf"
            if spatial_filter:
                cli_args += " -sf"
            cli_args += f" -th {threshold}"
            cli_args += f" -k {kernel_size}"
            cli_args += f" -b {background_color}"
            cli_args += f" -f {foreground_color}"
            cli_args += f" --num_people {num_people}"
            if landmark_filter is not None:
                cli_args += f" --landmarks {' '.join([str(l) for l in landmark_filter])}"
            if gamma != 1.0:
                cli_args += f" -g {gamma:.1f}"
            if combine:
                cli_args += " --combine"
            # if smoothing > 0.0: # FIXME: update smoothing on the fly in outputter
            #     cli_args += f" --smooth {smoothing}"
            print(f"python3 mask.py {cli_args}")


def main():
    parser = argparse.ArgumentParser(description='Get image and depth from RealSense camera, send to output')
    parser.add_argument('-s', '--size', type=int, nargs=2, default=[1280, 720], help='Size of the capture in pixels (width height)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('-d','--clip_distance', type=float, default=0.0, help='Clip distance in meters')
    parser.add_argument('--debug', action='store_true', help='Display the mask image')
    parser.add_argument('-c', '--clean', action='store_true', help='Apply morphological operations to clean up the thresholded image')
    parser.add_argument('--device', type=str, default="", help='Virtual webcam device')
    parser.add_argument('--shmdata', type=str, default="", help='shmdata pipe path')
    parser.add_argument('-e','--ema_alpha', type=float, default=0, help='Exponential Moving Average alpha (0 to disable averaging)')
    parser.add_argument('-m','--multi', type=int, default=0, help='Number of frames to consolidate (0 to disable multi-frame calculations)')
    parser.add_argument("-i", "--input", type=str, default="", help="Path to the bag or mp4 file")
    parser.add_argument('-p','--pose_detect', action='store_true', help='Enable pose detection')
    parser.add_argument('-sg','--segment', action='store_true', help='Enable segmentation (human masking)')
    parser.add_argument('-n','--num_people', type=int, default=1, help='Max number of people to detect')
    parser.add_argument('-tf','--temporal_filter', action='store_true', help='Enable temporal filter')
    parser.add_argument('-sf','--spatial_filter', action='store_true', help='Enable spatial filter')
    parser.add_argument('-k','--kernel_size', type=int, default=5, help='Kernel size for morphological operations')
    parser.add_argument('-r','--reference_depth', type=str, default="", help='Reference depth image')
    parser.add_argument('-th','--threshold', type=int, default=25, help='Threshold for background subtraction')
    parser.add_argument('-b','--background_color', type=int, default=0, help='Background color')
    parser.add_argument('-f','--foreground_color', type=int, default=255, help='Foreground color')
    parser.add_argument('--no-depth', action='store_true', help='Disable depth processing')
    parser.add_argument('--no-color', action='store_true', help='Disable color processing')
    parser.add_argument('--midi', type=str, dest='midi_address', default=None,
                        help='Midi output address PORT/CHANNEL (e.g.: 1/1)')
    parser.add_argument('--osc', type=str, dest='osc_address', default=None,
                        help='Open Sound Control address (e.g.: 127.0.0.1:4000)')
    parser.add_argument('--landmarks', nargs='*', default=[], choices=LANDMARKS.values(), help='Landmarks to send for output (e.g.: 1 2 3)')
    parser.add_argument('--landmark_names', nargs='*', default=[], choices=LANDMARKS.keys(), help='Landmark names to send for output (e.g.: "nose" "left eye (inner)" "left wrist")')
    parser.add_argument('--perf', type=int, default=0, help='Print performance info (every N frames)')
    parser.add_argument('--smooth', type=float, default=0.0, help='Smooth pose values (0 to disable smoothing)')
    parser.add_argument('-g','--gamma', type=float, default=1.0, help='Gamma correction (0.1 to 5.0)')
    parser.add_argument('--combine', action='store_true', help='Combine depth and color for pose & segmentation')
    args = parser.parse_args()

    # size choices:
    if args.size not in [[640, 480], [848,480], [1280, 720]]:
        print("Invalid size:", args.size)
        return

    # landmarks
    landmarks = None
    if len(args.landmarks) > 0:
        landmarks = set([int(l) for l in args.landmarks])
    if len(args.landmark_names) > 0:
        if landmarks is None:
            landmarks = set()
        for name in args.landmark_names:
            landmarks.add(LANDMARKS[name.lower()])
        landmarks = list(landmarks).sort()
    if landmarks is not None and len(landmarks) > 0 and not args.pose_detect:
        print("Landmarks specified but pose detection not enabled")
        return
    
    
    output = None

    # setup shmdata
    if args.shmdata is not None and args.shmdata != "":
        output = ShmData(
            args.shmdata, #'/tmp/shmdata_pipe' for example
            size=args.size,
        )

    # create virtual webcam
    if output is None:
        if args.device is None or args.device == "":
            print("Virtual webcam device not specified")
            print("Use --device to specify virtual webcam device")
            print('$ sudo modprobe v4l2loopback exclusive_caps=1 card_label="VirtualCam"')
            print('$ v4l2-ctl --list-devices')
        else:
            output = VirtualWebcam(
                args.device, #'/dev/video2' for example
                size=args.size,
            )
    if args.midi_address:
        port, channel = args.midi_address.split("/")
        output = OutputChain(
            output,
            MidiOutput(port=port, channel=channel, landmarks=landmarks, smoothing=args.smooth),
        )
    elif args.osc_address:
        host, port = args.osc_address.split(":")
        host = host.strip()
        port = int(port.strip())
        output = OutputChain(
            output,
            OSCOutput(host=host, port=port, landmarks=landmarks, smoothing=args.smooth),
        )
    if output is None:
        output = NullOutput()

    if args.debug:
        print(CONTROLS)

    reference_depth = None
    if args.reference_depth is not None and args.reference_depth != "":
        reference_depth = cv2.imread(reference_depth, cv2.IMREAD_GRAYSCALE)
        if reference_depth is None:
            print("Invalid reference depth image")
            return

    capture_frames(depth=not args.no_depth, color=not args.no_color,
                depth_size=args.size, color_size=args.size, fps=args.fps, 
                outputter=output, input=args.input,
                debug=args.debug,
                # pre and post processing
                temporal_filter=args.temporal_filter, spatial_filter=args.spatial_filter,
                multi=args.multi, ema_alpha=args.ema_alpha, clean=args.clean, 
                clip_distance=args.clip_distance,
                kernel_size=args.kernel_size, threshold=args.threshold, 
                # pose
                pose_detection=args.pose_detect, segment=args.segment, num_people=args.num_people,
                landmark_filter=landmarks,
                reference_depth=reference_depth,
                background_color=args.background_color, foreground_color=args.foreground_color,
                perf=args.perf,
                combine=args.combine,
                gamma=args.gamma,
                )

if __name__ == '__main__':
    main()

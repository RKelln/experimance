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
import queue

import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image
from screeninfo import get_monitors
import lunar_tools as lt  # pip install git+https://github.com/lunarring/lunar_tools
from blessed import Terminal

from image_server.image_generation_client import ImageGenerationClient, image_pipeline
from .depth_finder import depth_generator, depth_pipeline, depth_to_contour_map
from experimance_common.image_utils import cv2_img_to_base64url
from experimance_display.pygame_display import OpenGLImageShaderDisplay

def fullscreen(image, resolution=(1920, 1080)):
    if image is None:
        return None
    #print("Image shape:", image.shape)
    aspect_ratio = image.shape[1] / image.shape[0]  # width/height
    channels = image.shape[2] if len(image.shape) == 3 else 1

    # black background
    bkgd = np.zeros((resolution[1], resolution[0], channels), dtype=np.uint8)
    if channels == 4:
        bkgd[:, :, 3] = 255  # Set alpha channel to 255 (fully opaque)

    # Resize the image to fit the window size (maintain aspect ratio)
    # Assuming porttrait screens where height is shortest
    w = int(resolution[1] * aspect_ratio)
    resized_image = cv2.resize(image, (w, resolution[1]), interpolation=cv2.INTER_AREA)

    # composite the image in the center on background
    x = (resolution[0] - w) // 2
    bkgd[:, x:x+w, :channels] = resized_image  # Ensure to copy all channels

    return bkgd


def get_monitor_size(monitor_number=0):
    """Get the size of the specified monitor (default is the first monitor)."""
    monitors = get_monitors()
    if monitor_number >= len(monitors):
        raise ValueError("Monitor number out of range")
    monitor = monitors[monitor_number]
    return monitor.width, monitor.height


def create_window_on_monitor(window_name, requested_monitor=0, is_fullscreen=False):
    """Create a cv2 window on a specific monitor."""
    monitors = get_monitors()
    monitor_number = requested_monitor
    if requested_monitor >= len(monitors):
        monitor_number = 0 # default to first monitor
    monitor = monitors[monitor_number]

    # Create a named window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Move the window to the monitor
    cv2.moveWindow(window_name, monitor.x, monitor.y)

    # Optionally, resize the window to match the monitor size for fullscreen
    if is_fullscreen:
        cv2.resizeWindow(window_name, monitor.width, monitor.height)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    return (monitor.width, monitor.height)




def experimance(depth_factory, 
                image_server_factory, image_params_factory, 
                prompt_generator, 
                display_server, 
                audio_server,
                test=False,
                warm_up_period:float=5.0, # seconds
                start_seed:int=-1,
                ):

    # setup generators to run as workers that place things into queues
    # setup display server to run as a worker that places things into a queue

    # depth worker
    depth_queue = mp.Queue()
    depth_process = mp.Process(target=depth_pipeline, args=(depth_queue, depth_factory))
    depth_process.start()

    # image worker
    image_request_queue = mp.Queue()
    image_output_queue = mp.Queue()
    image_process = mp.Process(target=image_pipeline, args=(image_request_queue, image_output_queue, image_server_factory))
    image_process.start()

    # prompt generator
    prompt = next(prompt_generator)
    print("Prompt:", prompt)

    depth_image = None
    start_time = time.monotonic()
    warm_up = warm_up_period > 0
    prompt_timing = 20.0
    prompt_timer = time.monotonic()
    last_depth_time = time.monotonic()
    min_time_between_images = 5.0
    seed = start_seed

    def image_request(p, d, s=-1) -> bool:
        nonlocal last_depth_time, min_time_between_images, image_params_factory, image_request_queue
        now = time.monotonic()
        if now - last_depth_time > min_time_between_images:
            last_depth_time = now
            image_params = image_params_factory(p, d, s)
            image_request_queue.put(image_params, block=False)
            return True
        return False

    try:
        while True:
            
            #print('starting loop')
            if warm_up and time.monotonic() - start_time > warm_up_period:
                print("Warm up period complete")
                warm_up = False
                prompt_timer = 0 # force prompt to be sent

            if not test and time.monotonic() - prompt_timer > prompt_timing:
                prompt = next(prompt_generator)
                print("Prompt:", prompt)
                prompt_timer = time.monotonic()
                seed += 1
                if image_request(prompt, depth_image, seed):
                    prompt_timer = time.monotonic()
                else: #failed to send image request
                    seed -= 1 # revert seed change

            # get depth image from queue, non-blocking
            try:
                #print("Getting depth image")
                result = depth_queue.get(block=False)
                if result is None:
                    print("depth image is None")
                    break
                depth_image, detected_hands = result
                # if we have a depth image then make a new image
                if depth_image is not None:
                    if test:
                        image = depth_to_contour_map(depth_image)
                        display_server.send_img(image)
                    elif not warm_up:
                        if detected_hands is False:
                            print("\nNew depth image", seed, "\n")
                            image_request(prompt, depth_image, seed)
            except queue.Empty:
                #print("No depth image ready yet")
                pass
            
            if warm_up or test:
                continue

            # check for completed generated images:
            try:
                image = None
                image_paths = image_output_queue.get(block=False)
                if image_paths is None:
                    print("image paths is None")
                    break

                print("Got image", image_paths)

                if len(image_paths) == 0:
                    print("No image paths")
                    continue

                if len(image_paths) > 1:
                    print("More than one image path")
                    continue

                if not os.path.exists(image_paths[0]):
                    print("Image path does not exist")
                    continue

                # got new image, handle it
                image = cv2.imread(image_paths[0])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if display_server is not None:
                    display_server.send_img(image)

            except queue.Empty:
                #print("No image ready yet")
                pass

            # wait for key press
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("quitting")
                break
            elif key == ord('p'):
                prompt = next(prompt_generator)
                print("Prompt:", prompt)

            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    except Exception as e:
        print("Exception:", e)
    finally:
        # cleanup
        print("Cleaning up experimance")
        depth_queue.put(None)
        image_request_queue.put(None)
    
        depth_process.join()
        image_process.join()
    
        image_process.terminate()
        depth_process.terminate()
    
        if display_server is not None:
            display_server.stop()
    
        cv2.destroyAllWindows()


class LundarRenderer:
    def __init__(self, size=(1920, 1080)):
        self.renderer = lt.Renderer(width=size[0], height=size[1])

    def start(self):
        pass

    def stop(self):
        pass

    def send_img(self, image):
        resized_image = fullscreen(image, (self.renderer.width, self.renderer.height))
        self.renderer.render(resized_image)


class OpenGLRenderer:
    def __init__(self, size=(1024, 1024)):
        self.renderer = OpenGLImageShaderDisplay(width=size[0], height=size[1], 
                                    window_title="Experimance", 
                                    do_fullscreen=False,
                                    display_index=0,
                                    show_fps=True, circular_mask=False)
    def start(self):
        self.renderer.start()

    def stop(self):
        self.renderer.stop()

    def send_img(self, image):
        self.renderer.render(image)


def main():
    parser = argparse.ArgumentParser(description='Experimance')
    parser.add_argument('-n', '--name', type=str, default=time.strftime("%Y%m%d-%H%M%S"), help='Name of the capture')
    parser.add_argument('-s', '--size', type=int, nargs=2, default=(1920, 1080), help='Size of the output in pixels (width height)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--test', action='store_true', help="Testing mode")
    parser.add_argument('-d', '--display-server', '--display_server', type=str, default=None, help="ZMQ server address:port to send images to")
    parser.add_argument('--seed', type=int, default=-1, help="Seed to start for random generation")
    parser.add_argument('--prompts', type=str, default=None, help="Path to a file of prompts to use for text generation")
    parser.add_argument('--gen_server', '--gen-server', type=str, default=None, help="ZMQ server address:port for generative server")
    parser.add_argument('--mock-depth', '--mock_depth', type=str, default=None, help="Path to a folder of images to use for mock depth maps")
    args = parser.parse_args()

    display_server = None
    if args.display_server is not None:
        if args.display_server == "default" or args.display_server == "":
            ip = '127.0.0.1'
            port = '5556'
        else:
            if ":" not in args.display_server:
                raise ValueError("display_server must be in the format 'ip:port'")
            ip = args.display_server.split(":")[0]
            port = args.display_server.split(":")[1]
        display_server = lt.ZMQPairEndpoint(is_server=False, ip=ip, port=port)
    else:
        display_server = OpenGLRenderer()

    # prompt generator
    prompt_generator = None
    if args.prompts is not None:
        # load prompts from file
        with open(args.prompts, "r") as f:
            prompts = f.readlines()
        # convert --prompt "positive prompt" --negative_prompt "negative prompt" format to tuples
        prompts = [tuple(p.strip().split(" --negative_prompt ")) for p in prompts]
        prompts = [(p[0].removeprefix("--prompt ").strip('"'), p[1].strip('"')) for p in prompts]
        # remove loras (ad anything between <>) eg: <lora:xl_more_art-full_v1:0.6>
        prompts = [(re.sub(r'<.*?>', '', p[0]), p[1]) for p in prompts]
        print("Prompts:", prompts[0])
        prompt_generator = cycle(prompts)
    else:
        prompt_generator = cycle([
            ("colorful satellite image in the style of experimance, (dense urban:1.2) dramatic landscape, buildings, farmland, (industrial:1.1), (rivers, lakes:1.1), busy highways, hills, vibrant hyper detailed photorealistic maximum detail, 32k, high resolution ultra HD",
            "distorted, warped, blurry, text, cartoon, illustration")
            #("(vibrant colorful:1.1) top down aerial photo, busy urban density, city infrastructure, housing, architecture, machinery, roads, highways, (buildings:1.1), vehicles, farm, industry, river, field, forest, park, rock strata, garbage, pollution, (Edward Burtynsky:1.1), landscape, highly detailed, meticulous hyper detail, photorealistic maximum detail, 32k, ultra HD", 
            #"blurry, sketch, cartoon, illustration, CGI, 3D render, unreal engine, worst quality, low quality, compression artifacts, deformed, lowres, ugly, oversaturated, fish-eye lens, monochromatic, flat, smooth, haze, grainy")
        ])

    # setup the depth and image generation factories
    depth_factory = lambda: depth_generator(
        json_config="high_accuracy_white-to_black_40-50cm_v2.json",
        #fps=10,
        min_depth=0.49,
        max_depth=0.56,
        align=True,
        change_threshold=50,
        detect_hands=True,
        crop=True,
        output_size=(1024,1024),
        test=False,
        mock=args.mock_depth,
    )

    image_server_factory = lambda: ImageGenerationClient(service="fal")

    image_params_factory = lambda prompt, depth_image, seed: {
        "endpoint": "comfy/RKelln/experimancexilightningdepth",
        "arguments": {
            "ksampler_seed": randint(1, 1000000) if seed < 0 else seed,
            "prompt": prompt[0],
            "negative_prompt": prompt[1],
            "lora_url": "https://storage.googleapis.com/experimance_models/civitai_experimance_sdxl_lora_step_1000_1024x1024.safetensors",
            "lora_strength": 1.0,
            "model_url": "https://civitai.com/api/download/models/471120?type=Model&format=SafeTensor&size=full&fp=fp16",
            "depth_map": cv2_img_to_base64url(depth_image),
        },
    }

    # Juggernaut hyper with depth
    #     "endpoint": "comfy/RKelln/experimance_hyper_depth_v5",
    #     "arguments": {
    #         "lora_url": "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor",
    #         "lora_strength": 0.8,
    #         "model_url": "https://civitai.com/api/download/models/471120?type=Model&format=SafeTensor&size=full&fp=fp16",

    experimance(depth_factory, 
                image_server_factory, 
                image_params_factory, 
                prompt_generator, 
                display_server, 
                None,
                test=args.test,
                warm_up_period=10)




if __name__ == '__main__':
    main()

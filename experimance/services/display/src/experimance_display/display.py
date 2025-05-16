import argparse
import time
import concurrent.futures
import queue
import threading
import os
from typing import Tuple, Generator, Union, Optional

from experimance_transition.transition import SimilarityTransition

import lunar_tools as lt
from screeninfo import get_monitors

from PIL import Image
import numpy as np
import cv2
import sdl2

import multiprocessing as mp

from ffmpegcv import VideoWriter

from experimance_transition.transition_worker import (
    TransitionWorker, 
    FlowTransitionWorker, 
    BlendTransitionWorker, 
    mpTransitioner,
    start_transition_worker
)
from experimance_display.pygame_display import FullscreenImageDisplay, OpenGLImageDisplay, OpenGLImageShaderDisplay

# display images sent to it over ZMQ
# each image is sent as a multipart message, with the first part being the image data
# and the second part being the image metadata (width, height, channels, timing info)


def get_monitor_size(monitor_number=0):
    """Get the size of the specified monitor (default is the first monitor)."""
    monitors = get_monitors()
    if monitor_number >= len(monitors):
        raise ValueError("Monitor number out of range")
    monitor = monitors[monitor_number]
    return monitor.width, monitor.height

class NullWriter:
    def write(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class ImageWriter:
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.index = 0

    def write(self, image:np.ndarray):
        filename = f"{self.index:05d}.webp"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, image)
        self.index += 1

    def __enter__(self):
        # ensure path exists
        os.makedirs(self.output_dir, exist_ok=True)
        return self

    def __exit__(self, *args):
        pass

class DisplayServer:
    def __init__(self, size=(1920,1080), ip='127.0.0.1', port='5556', 
                 transitioner=None, transition_duration=0.5,
                 fullscreen=True, display_index=0, fps=30, 
                 loop=False, output=None):
        self.server = lt.ZMQPairEndpoint(is_server=True, ip=ip, port=port)
        
        # get screen resolution for fullscreen mode
        if fullscreen:
            size = get_monitor_size(display_index)
        
        # self.renderer = lt.Renderer(width=size[0], height=size[1], 
        #                             window_title="Experimance", 
        #                             do_fullscreen=fullscreen,
        #                             display_index=display_index)
        self.renderer = OpenGLImageShaderDisplay(width=size[0], height=size[1], 
                                    window_title="Experimance", 
                                    do_fullscreen=fullscreen,
                                    display_index=display_index,
                                    show_fps=True, circular_mask=True)
        self.renderer.start()

        self.images = []
        self.current = None # index of current image
        self.next = None    # index of next image
        self.current_image = None  # image data of current image
        self.next_image = None     # image data of next image
        self.transition_start = None
        self.transition_end = None
        if transitioner is None:
            transitioner = TransitionWorker()
        self.transitioner = transitioner
        self.transition_duration = transition_duration
        self.fullscreen = fullscreen
        self.fps = fps
        self.loop = loop

        if output is not None:
            if output.endswith(".mp4"):
                self.writer = VideoWriter(args.output, codec='libx264', fps=args.fps, preset="slow")
            else: # writer is a directory to place images
                self.writer = ImageWriter(args.output)
        else:
            self.writer = NullWriter()
        
        
    def run(self):
        image_updated:bool = False
        image  = None
        transition : Optional[Generator] = None
        dur:float = 1.0

        try:
            with self.writer as writer:
                while True:
                    t = time.monotonic()

                    # message check loop
                    while not self.server.messages.empty():
                        message_type, message_data = self.server.messages.get()
                        if message_type == "json":
                            if self.handle_message(message_data):
                                return
                        elif message_type == "img":
                            if message_data is not None:
                                self.handle_image(message_data)

                    # image render loop 
                    if len(self.images) == 0:
                        time.sleep(0.25)
                        continue

                    image_updated = False

                    if self.current_image is None and self.current is not None:
                        self.current_image = self.images[self.current]
                        image = self.current_image
                        image_updated = True
                        self.transitioner.add_image(self.current_image)

                    if self.next_image is None and self.next is not None:
                        if self.current != self.next:
                            self.next_image = self.images[self.next]
                            self.transitioner.add_image(self.next_image)
                            if self.transition_start is None:
                                self.start_transition_countdown(delay=4.0)

                    if self.transition_start is not None and self.transition_end is not None:
                        now = sdl2.SDL_GetTicks()
                        if self.transition_start > now:
                            # transition hasn't started yet
                            continue # FIXME: display current image?
                        
                        if transition is None:
                            transition = self.transitioner.new_transition() # returns generator

                        #start_time = time.monotonic()
                        step, total_steps, image = next(transition)
                        #print("transition time:", time.monotonic() - start_time)

                        if image is None or total_steps == 0:
                            #print("transition not ready")
                            continue
                        
                        image_updated = True

                        #print("transition step", step, total_steps)

                        if step >= total_steps:
                            # transition is done
                            transition.close()
                            transition = None
                            self.current = self.next
                            if self.next_image is not None: # we could have been transitioning from black to current
                                self.current_image = self.next_image
                            self.next_image = None
                            self.transition_start = None
                            self.transition_end = None
                            # set up next transition, but don't actualy start it until next image available
                            if len(self.images) > 1:
                                if self.loop:
                                    self.next = (self.current + 1) % len(self.images)
                                else:
                                    self.next = min(self.current + 1, len(self.images) - 1)
                                #print("done transition to", self.current, "next is", self.next)
                                if self.next != self.current:
                                    self.start_transition_countdown(delay=4.0)

                    if image is not None:
                        if image_updated:
                            writer.write(image)
                        self.renderer.render(image, fps=(1.0/dur))

                    dur = time.monotonic() - t
                    #print("fps", 1.0/dur)
                    delay = (1.0/self.fps) - dur
                    if delay > 0:
                        time.sleep(delay)

        except Exception as e:
            import traceback
            traceback.print_exc()

            print("DisplayServer error", e)
        finally:
            self.transitioner.stop()
            if transition is not None:
                transition.close()
            self.server.stop()
            self.renderer.stop()


    def next_index(self, delay:float=0.0):
        if len(self.images) == 0:
            return
        
        if self.loop:
            self.next = (self.current + 1) % len(self.images)
        else:
            self.next = self.current + 1
            if self.next >= len(self.images):
                self.next = len(self.images) - 1

        if self.next != self.current:
            self.start_transition_countdown(delay=delay)


    def set_next_index(self, index:int|None, delay:float=0.0):
        if index is None:
            self.next_image()
            return
        if index < 0:
            index = max(0, len(self.images) - index)
        index = index % len(self.images)
        self.next = index
        self.next_image = self.images[index]
        if self.next != self.current:
            self.start_transition_countdown(delay=delay)

    def start_transition_countdown(self, delay:float=0.0, duration:float=None): # duration in seconds
        if self.transition_end is None:
            if duration is None:
                duration = self.transition_duration
            self.transition_start = sdl2.SDL_GetTicks() + delay * 1000
            self.transition_end = self.transition_start + duration * 1000
            #print("start transition countdown", self.transition_start, self.transition_end)


    def resize_image(self, image):
        # resize image to fit screen and center, black background
        if image.shape[0] > self.renderer.height or image.shape[1] > self.renderer.width:
            # image is larger than screen, resize
            scale = min(self.renderer.height / image.shape[0], self.renderer.width / image.shape[1])
            image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        elif image.shape[0] < self.renderer.height or image.shape[1] < self.renderer.width:
            # image is smaller than screen, resize
            scale = min(self.renderer.height / image.shape[0], self.renderer.width / image.shape[1])
            image = cv2.resize(image, (0,0), fx=scale, fy=scale)

        # center and place on background
        bg = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)
        y0 = (self.renderer.height - image.shape[0]) // 2
        x0 = (self.renderer.width - image.shape[1]) // 2
        bg[y0:y0+image.shape[0], x0:x0+image.shape[1]] = image
        return bg
        

    def handle_image(self, image):
        #resized = self.resize_image(image)
        # convert to BGR if needed

        resized = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #resized = image
        # add to queue
        self.images.append(resized)
        print("got image:", image.shape, "resized:", resized.shape, "images:", len(self.images))
        print("current:", self.current, "next:", self.next, "transition:", self.transition_start, self.transition_end)
        if self.current is None: # first image received
            self.current = 0
            self.next = 0
            self.start_transition_countdown() # force transition (usuaully impossible when current == next)
        elif self.transition_end is None: # no ongoing transition
            self.next_index()
            print("transitioning from", self.current, "to", self.next, "of", len(self.images))


    def handle_message(self, message):
        print("got message", message)
        if 'command' in message:
            message = message['command']
        
        # FIXME: use new match syntax

        if message == "exit":
            return True
        elif message == "next":
            if len(self.images) > 0:
                self.current = (self.current + 1) % len(self.images)
        elif message == "prev":
            if len(self.images) > 0:
                self.current = (self.current - 1) % len(self.images)
        elif message == "clear": # clear all images
            self.images = []
            self.current = None
            self.next = None
            self.transition_start = None
            self.transition_end = None
        elif message == "new": # clear image queue, retain current and next images, finish transition if needed
            self.images = []
            self.current = None
            self.next = None
            if self.transition_end is None: # no active transition
                self.next_image = None
        elif message == "pause":
            self.transition_start = None
            self.transition_end = None
        elif message == "play":
            self.transition_start = sdl2.SDL_GetTicks()
            self.transition_end = self.transition_start + self.transition_duration * 1000
        elif message.startswith("set_duration"):
            self.transition_duration = float(message.split(" ")[1])
        elif message.startswith("set_transition"):
            self.transition_type = message.split(" ")[1]
            #self.init_transition()
            # TODO
        elif message == "get_size":
            # send the monitor size back
            self.server.send_json({"width": self.renderer.width, "height": self.renderer.height})

        return False


def random_image(size:Tuple[int,int]): # (W,H)
    # gaussian noise combined with a major color and then blurred
    noise = np.random.normal(128, 64, (size[1],size[0],3)).clip(0,255).astype(np.uint8)
    color = np.random.rand(3) * 255
    image = np.zeros((size[1],size[0],3), dtype=np.uint8)
    image[:,:,:] = color
    image += noise
    image = cv2.GaussianBlur(image, (13,13), 0)
    image = image.clip(0,255).astype(np.uint8)
    return image


def title_image(size:Tuple[int,int], text:str="Experimance"):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_color = (255, 255, 255)
    line_type = cv2.LINE_AA

    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # if text is larger than image size then rescale text
    image_width, image_height = size
    horizontal_padding = 20 # padding on left and right
    text_width += horizontal_padding*2
    if text_width > image_width:
        font_scale = font_scale * image_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the center position
    x = (image_width - text_width) // 2
    y = (image_height + text_height) // 2

    # Put the text on the image (note np is (H,W) and cv2 is (W,H))
    title_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    title_image = cv2.putText(title_image, text, (x, y), font, font_scale, text_color, font_thickness, line_type)

    return title_image


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Display server for images sent over ZMQ')
    parser.add_argument('-d', '--transition-duration', '--transition_duration', type=float, default=0.5, help='Image transition duration in seconds')
    parser.add_argument('-t', '--transition-type', '--transition_type', type=str, default='blend', choices=['none','blend', 'flow'], help='Image transition type')   
    parser.add_argument('-s', '--size', type=str, default='1920x1080', help='Display size (WxH)')
    parser.add_argument('-i', '--ip', type=str, default='localhost', help='IP address to bind to')
    parser.add_argument('-p', '--port', type=str, default='5556', help='Port to bind to')
    parser.add_argument('-w', '--windowed', action='store_true', help='Run in windowed mode')
    parser.add_argument('--display-index', '--display_index', type=int, default=0, help='Index of display to use')
    parser.add_argument('--loop', action='store_true', help='Loop through images')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--output', type=str, default=None, help='Output video path')

    args = parser.parse_args()

    # (W,H)
    size = tuple(map(int, args.size.split('x')))

    # create Transitioner
    steps = max(1, int(args.fps * args.transition_duration))
    transition_class = TransitionWorker
    if args.transition_type == "blend":
        transition_class = BlendTransitionWorker
    elif args.transition_type == "flow":
        transition_class = FlowTransitionWorker
 
    transition_params = {
        "steps": steps,
        "blur_amount": 5,
        "blend_steps": 0,
        "size": 256,
        "weight1": 1.0,
        "weight2": 2.0,
        "distance_weight": 1.0,
    }
    transitioner = mpTransitioner(
        input_queue=mp.Queue(), 
        target=start_transition_worker, 
        transition_class=transition_class,
        transition_params=transition_params,
        )

    display = DisplayServer(
        size=size, 
        ip=args.ip, 
        port=args.port,
        transitioner=transitioner,
        transition_duration=args.transition_duration,
        fullscreen=not args.windowed,
        display_index=args.display_index,
        loop=args.loop,
        fps=args.fps,
        output=args.output,
    )

    display.handle_image(title_image((1024,1024), "Experimance"))

    # if test then create 5 test images and send them to the display in a separate process
    if args.test:
        for i in range(5):
            image = random_image(size)
            # add number to image
            image = cv2.putText(image, str(i), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv2.LINE_AA)
            display.handle_image(image)

    print(f"Display server: {size}, {args.ip}:{args.port}, {args.transition_duration:0.2f}s {args.transition_type}, loop: {args.loop} fps: {args.fps}")

    display.run()

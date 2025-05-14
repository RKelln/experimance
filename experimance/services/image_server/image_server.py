import argparse
import io
import json
import os
import time
from multiprocessing import Process, Queue
from threading import Thread
from pathlib import Path
import random
from pprint import pprint
import re

import fal_client
import openai
import requests
import zmq
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

HEARTBEAT_ID = "_HEARTBEAT"
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
SERVICES=["fal", "openai", "local"]
DEFAULT_ZMQ_ADDRESS = "ipc:///tmp/zmq.ipc"

def falai_image_url_generator(response):
    # determine the type of response (differs based on the endpoint)

    # model endpoint
    # {'images': [{'url': 'https://fal.media/files/panda/uQsLdbOPox-ntaMxeDzsy.png', 'width': 1024, 'height': 768, 'content_type': 'image/jpeg'}], 'timings': {'inference': 0.3568768650002312}, 'seed': 852971348, 'has_nsfw_concepts': [False], 'prompt': 'test'}
    if 'images' in response:
        for image_result in response['images']:
            yield image_result['url']
        return
    
    # comfy workflow endpoint:
    # {
    #     "outputs": {
    #         "14": {
    #           "images": [
    #               {
    #               "url": "https://fal.media/files/panda/YB89Jz_d95I7_ZeBFhslU_ComfyUI_00003_.png",
    #               "type": "output",
    #               "filename": "ComfyUI_00003_.png",
    #               "subfolder": ""
    #               }
    #           ]
    #         }
    #     }, ...
    if 'outputs' in response:
        # the next key could change based on workflow, but it will always be first index
        first_output_key = list(response['outputs'].keys())[0]
        for image_result in response['outputs'][first_output_key]['images']:
            if 'image_output' not in image_result:
                yield image_result['url']
            else:
                yield image_result['image_output']
        return
    raise ValueError(f"Unknown response type: {response}")

def falai_gen_images(**kwargs):
    if kwargs is None:
        raise ValueError("falai_gen_images: No arguments provided")
    
    endpoint = kwargs.get("endpoint", None)
    arguments = kwargs.get("arguments", None)

    # check if endpoint and arguments are provided
    if endpoint is None or arguments is None:
        raise ValueError(f"'endpoint' and 'arguments' must be provided: {kwargs}")
    
    try:
        # print("Generating falai images with endpoint", endpoint)
        # pprint(clip_value_strings(arguments))

        # submit the request
        handler = fal_client.submit(
            endpoint,
            arguments=arguments
        )
        return handler.get()
    except Exception as e:
        raise ValueError(f"Failed to generate images: {e}")
    return None


def openai_gen_image(client, **kwargs):
    # check if prompt is provided
    if 'prompt' not in kwargs:
        raise ValueError(f"'prompt' must be provided: {kwargs}")
    
    return client.images.generate(
        model="dall-e-3",
        prompt=kwargs.get("prompt"),
        n=kwargs.get("n", 1),  # Number of images to generate
        size=kwargs.get("size", "1024x1024"),  # Size of the generated image
        quality=kwargs.get("quality", "standard")
    )

def openai_image_url_generator(response):
    for data in response.data:
        yield data.url


def local_image(base_path, **kwargs):
    path = kwargs.get("path", ".")
    match = kwargs.get("match")
    n = kwargs.get("n", 1)
    choice = kwargs.get("choice", "oldest")
    if choice not in ["oldest", "first", "last", "random"]:
        raise ValueError(f"'choice' must be one of: oldest, first, last, random. Got: {choice}")
    
    dir = Path(base_path) / Path(path)
    if not dir.exists():
        raise ValueError(f"Invalid 'path' ({path} abs: {dir})")
    
    # all images in dir
    files = [file for ext in IMAGE_EXTENSIONS for file in dir.glob(ext)]
    if len(files) == 0:
        raise ValueError(f"No images found in {dir}")
    
    if match:
        # find all files in directory that match regex
        files = [f for f in files if match in f.name]

    if choice == "oldest": # by access time
        files = sorted(files, key=lambda f: os.path.getatime(f))
    elif choice == "first":
        files = sorted(files, key=lambda f: f.name)
    elif choice == "last":
        files = sorted(files, key=lambda f: f.name, reverse=True)
    elif choice == "random":
        random.shuffle(files)

    files = files[:n]

    # touch all files to update their access time (not modified time)
    for f in files:
        stat_info = os.stat(f)
        os.utime(f, (time.time(), stat_info.st_mtime))

    return [str(f) for f in files]

# recursively limit the string length of dict values and return a copy
def clip_value_strings(d:dict, length:int=50) -> dict:
    dc = d.copy()
    for k,v in d.items():
        if isinstance(v, dict):
            dc[k] = clip_value_strings(v, length)
        elif isinstance(v, str) and len(v) > length:
            dc[k] = v[:length] + "..."
    return dc

class ImageServer:
    def __init__(self, output_dir, 
                 zmq_address=DEFAULT_ZMQ_ADDRESS,
                 default_service='local', 
                 allowed_services=['local'],
                 timeout=10.0,
                 image_format='webp',
                 heartbeat_interval=5.0):
        
        self.output_dir = output_dir
        self.zmq_address = zmq_address

        self.task_queue = Queue()
        self.response_queue = Queue()
        self.default_service = default_service
        self.timeout = timeout
        self.image_format = image_format.strip().strip(".").lower()
        self.heartbeat_interval = heartbeat_interval

        self.services = {}
        self.image_url_generators = {}

        # import services here
        for service in allowed_services:
            if service == 'fal':
                self.services['fal'] = falai_gen_images
                self.image_url_generators['fal'] = falai_image_url_generator
            
            elif service == 'openai':
                openaiclient = openai.Client()
                # bind the client to the function
                def openai_gen_image_bound(**kwargs):
                    return openai_gen_image(openaiclient, **kwargs)

                self.services['openai'] = openai_gen_image_bound
                self.image_url_generators['openai'] = openai_image_url_generator

            elif service == "local":
                def local_image_bound(**kwargs):
                    return local_image(self.output_dir, **kwargs)
                self.services["local"] = local_image_bound
                self.image_url_generators['local'] = None
            else:
                raise ValueError(f"Service {service} is not supported")


    def image_generation_worker(self):
        while True:
            task = self.task_queue.get()
            if task is None:  # Shutdown signal
                print("Shutting down worker...")
                break
        
            # must have ids
            if 'task_id' not in task or 'client_id' not in task:
                print("ERROR: Task must have task and client ID", task)
                continue

            # Extract information from task
            client_id = task['client_id']  # Unique ID for the client
            task_id = task['task_id']  # Unique ID for the task
            service = task.get('service', self.default_service)  # The text-to-image service to use
            params = task.get('params', {})  # Additional parameters for the service
            if service not in self.services:
                print("ERROR: Service not found", service, "must be one of", SERVICES)
                continue

            if params is None:
                print("ERROR: Params must be a dictionary")
                continue

            if len(params) == 0:
                print("WARNING: No parameters provided for task", task_id)

            print("Processing task", task_id, "with service", service)

            response_data = {
                'client_id': client_id,
                'task_id': task_id,
            }

            # Call the service to generate the image
            try:
                response = self.services[service](**params)
                if response is None:
                    print("Response is None")
                    self.response_queue.put(response_data | {
                        'status': 'failed',
                        'error': f"Failed to generate images from {service}"
                    })
                    continue
                image_paths = []
                if service not in self.image_url_generators or self.image_url_generators[service] is None:
                    # special case: the response are the paths
                    image_paths = response
                else:
                    for image_url in self.image_url_generators[service](response):
                        # get the image
                        r = requests.get(image_url, timeout=self.timeout)
                        if r.status_code == 200:
                            image = Image.open(io.BytesIO(r.content))
                            image_path = f"{self.output_dir}/{service}_{task_id}_{len(image_paths)+1}.{self.image_format}"
                            image.save(image_path)
                            image_paths.append(image_path)
                        else:
                            self.response_queue.put(response_data | {
                                'status': 'failed',
                                'error': f"Failed to download images from {response}"
                            })
                            continue
                self.response_queue.put(response_data | {
                    'status': 'completed',
                    'image_paths': image_paths
                })
                print("Finished task", task_id, "with service", service)
            except Exception as e:
                # get line number and context of error
                import traceback
                traceback.print_exc()

                self.response_queue.put(response_data | {
                    'status': 'failed',
                    'error': str(e)
                })
                continue


    def heartbeat(self):
        while True:
            time.sleep(self.heartbeat_interval)
            self.response_queue.put_nowait({
                'task_id': HEARTBEAT_ID,
                'time': time.time(),
                'interval': self.heartbeat_interval,
            })


    def request_listener(self):
        try:
            while True:
                # Message structure: In a ROUTER/DEALER pattern, messages are usually multipart:
                #     First part: The client ID (used for routing).
                #     Second part: An empty frame (which is part of the ZMQ protocol when using ROUTER/DEALER).
                #     Third part: The actual data (JSON or whatever the message content is).
                
                # Receive a message from any client
                client_id, message = self.router_socket.recv_multipart()
                if message is None:
                    print("Received None message")
                    continue
                task = json.loads(message)

                task_id = task.get('task_id')
                service = task.get('service')
                params = task.get('params', {})

                print(f"Received task {task_id} from client {client_id}")
                pprint(clip_value_strings(params), compact=True)

                self.task_queue.put({
                    'client_id': client_id,
                    'task_id': task_id,
                    'params': params,
                    'service': service
                })
        except KeyboardInterrupt:
            print("Shutting down listener...")
        except Exception as e:
            print("Error in request listener:", e)
        finally:
            self.task_queue.put(None)  # Signal the worker to exit
            self.router_socket.close


    def response_sender(self):
        while True:
            result = self.response_queue.get()
            # remove client id from response dict
            client_id = result.pop('client_id', None)
            if client_id is None:
                print("response_sender: Missing client_id")
                continue
            print("Sending response to client", str(client_id))
            self.router_socket.send_multipart([client_id, json.dumps(result).encode()])


    def start(self):
        # Setup ZeroMQ context and sockets
        context = zmq.Context()
        self.router_socket = context.socket(zmq.ROUTER)
        self.router_socket.bind(self.zmq_address)

        os.makedirs(self.output_dir, exist_ok=True)

        # Start worker processes
        worker = Process(target=self.image_generation_worker)
        worker.start()
        
        # Start the heartbeat process
        #heartbeat = Thread(target=self.heartbeat, daemon=True)
        #heartbeat.start()

        # Start request listener
        listener_thread = Thread(target=self.request_listener, daemon=True)
        listener_thread.start()

        # Start response sender
        sender_thread = Thread(target=self.response_sender, daemon=True)
        sender_thread.start()

        try:
            listener_thread.join()
            sender_thread.join()
        except KeyboardInterrupt:
            self.task_queue.put(None)  # Signal the worker to exit
            worker.join()
            #heartbeat.join()


def validate_zmq_address(value):
    # Define regex patterns for different ZeroMQ transport protocols
    patterns = [
        r'^tcp://\*:\d+$',  # TCP
        r'^ipc:///.+$',     # IPC
        r'^inproc://.+$',   # In-process
        r'^pgm://.+$',      # PGM
        r'^epgm://.+$',     # EPGM
        r'^udp://.+$',      # UDP
    ]
    
    if not any(re.match(pattern, value) for pattern in patterns):
        raise argparse.ArgumentTypeError(f"Invalid address format: '{value}'. Expected formats: 'tcp://*:port', 'ipc:///path', 'inproc://name', 'pgm://address', 'epgm://address', 'udp://address'")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Server')
    parser.add_argument('--zmq', type=validate_zmq_address, default=DEFAULT_ZMQ_ADDRESS, help='Address to listen for requests')
    parser.add_argument('--output-dir', '--output_dir', type=str, default='.', help='Path to save generated images')
    parser.add_argument('--services', type=str, default='all', help='Comma-separated list of services to use')
    parser.add_argument('--timeout', type=float, default=10.0, help='Timeout for downloading images')
    parser.add_argument('--image-format', '--image_format', type=str, default='webp', help='Image format to save the generated images')
    parser.add_argument('--default-service', '--default_service', type=str, default='local', help='Default service to use for image generation')
    parser.add_argument('--heartbeat-interval', '--heartbeat_interval', type=float, default=5.0, help='Heartbeat interval for the server')
    args = parser.parse_args()

    services = []
    if args.services == 'all':
        services = SERVICES
    else:
        for service in args.services.split(","):
            if service not in SERVICES:
                raise ValueError(f"Invalid service: {service}. Must be one of {SERVICES}")
            services.append(service)

    server = ImageServer(args.output_dir,
                         zmq_address=args.zmq,
                         allowed_services=services, 
                         timeout=args.timeout, 
                         image_format=args.image_format, 
                         default_service=args.default_service,
                         heartbeat_interval=args.heartbeat_interval)
    print("Starting image server with services:", args.services, "on address", args.zmq)
    server.start()
    
    
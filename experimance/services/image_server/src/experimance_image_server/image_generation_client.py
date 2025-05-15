import zmq
import json
import uuid
import threading
import time
import base64
import io
from PIL import Image
import multiprocessing as mp
import queue

from experimance_common.image_utils import png_to_base64url


class ImageGenerationClient:
    def __init__(self, 
                 zmq_address="ipc:///tmp/zmq.ipc", 
                 service=None,
                 heartbeat_timeout=None):
        self.service = service
        self.heartbeat_timeout = heartbeat_timeout
        self.zmq_address = zmq_address

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect(self.zmq_address)

        self.callbacks = {}
        self.callbacks_lock = threading.Lock()  # Lock for synchronizing access to callbacks

        self._stop_polling = threading.Event()
        self.poll_thread = threading.Thread(target=self._poll_responses)
        self.poll_thread.daemon = True
        self.poll_thread.start()
        

    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def create_request(self, params, service=None, task_id=None):
        """Create an image generation request and return the task ID."""
        if service is None:
            service = self.service
        if service is None:
            raise ValueError("Service must be provided")
        
        if task_id is None:
            task_id = str(uuid.uuid4())
        message = {
            'task_id': task_id,
            'service': service,
            'params': params,
        }
        print(self.zmq_address, task_id, service)  # Debugging print
        self.socket.send_json(message)
        return task_id


    def poll(self, task_id=None):
        """Poll to check if a specific task is completed or return any completed task."""
        with self.callbacks_lock:  # Acquire the lock before accessing callbacks
            if self.callbacks is None or len(self.callbacks) == 0: return None
            if task_id:
                # Check if the task_id exists and has a response ready
                if task_id in self.callbacks:
                    return task_id, self.callbacks.pop(task_id)
                return None  # Return None only if the task is not yet completed
            else:
                # Return the first completed task found
                if self.callbacks:
                    task_id, image_path = self.callbacks.popitem()
                    return task_id, image_path
        return None
    
    def register_callback(self, task_id, callback):
        """Register a callback function to be called when a task is completed."""
        with self.callbacks_lock:  # Acquire the lock before modifying callbacks
            self.callbacks[task_id] = callback


    def _poll_responses(self):
        """Internal method to poll for responses from the server."""
        while not self._stop_polling.is_set():
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
                # Ignore messages that are not responses
                if 'task_id' not in message:
                    print("Received invalid message", message)
                    continue

                task_id = message['task_id']

                if task_id == "_HEARTBEAT":
                    # TODO: Implement heartbeat timeout handling
                    continue

                if 'image_paths' not in message:
                    # likely an error or status message
                    continue

                image_paths = message['image_paths']

                print(f"Received response for task {task_id}: {image_paths}")
                
                with self.callbacks_lock:  # Ensure exclusive access to callbacks
                    if task_id in self.callbacks:
                        callback = self.callbacks.pop(task_id)
                        if callable(callback):
                            callback(image_paths)
                    else:
                        self.callbacks[task_id] = image_paths
            except zmq.Again:
                # No message received, continue the loop
                pass
            
            time.sleep(0.1)  # Add a small sleep to avoid tight loop


    def close(self):
        """Close the sockets and context."""
        # Signal the polling thread to stop
        self._stop_polling.set()
        
        # Ensure the poll thread exits cleanly
        if self.poll_thread.is_alive():
            self.poll_thread.join()
        
        # Clear callbacks
        with self.callbacks_lock:
            self.callbacks.clear()
        
        # Close the sockets and terminate the context
        self.socket.close()
        self.context.term()


def image_pipeline(request_queue:mp.Queue, output_queue:mp.Queue, client_factory):
    # start a ImageGneratorClient that is fed from a Queue and outputs to a Queue
    # the ImageGeneratorClient will poll the server for responses
    # and put the responses in the output Queue
    # the ImageGeneratorClient will also poll the request_queue for new requests
    # and send the requests to the server

    # we need to create the client inside the process otherwize the ZMQ context will be shared / not work
    client = client_factory()

    def output(image_paths):
        print("image pipeline callback queued", image_paths)
        output_queue.put(image_paths)

    with client:
        while True:
            try:
                params = request_queue.get(block=True)
                if params is None:
                    output_queue.put(None)
                    break
                task_id = client.create_request(params)
                client.register_callback(task_id, output)
            except queue.Empty:
                continue
        # close queues
        request_queue.close()
        output_queue.close()



if __name__ == "__main__":
    # Example usage
    request_queue = mp.Queue()
    output_queue = mp.Queue()

    def client_factory():
        return ImageGenerationClient(service="local")
    
    pipeline_process = mp.Process(target=image_pipeline, args=(request_queue, output_queue, client_factory))
    pipeline_process.start()

    # Example usage
    falai_experimance_depth_params = {
        "endpoint": "comfy/RKelln/experimance_hyper_depth_v5",
        "arguments": {
            "ksampler_seed": 1,
            "prompt": "colorful RAW photo modern masterpiece, overhead top down aerial shot, in the style of (Edward Burtynsky:1.2) and (Gerhard Richter:1.2), (dense urban:1.2) dramatic landscape, buildings, farmland, (industrial:1.1), (rivers, lakes:1.1), busy highways, hills, vibrant hyper detailed photorealistic maximum detail, 32k, high resolution ultra HD",
            "negative_prompt": "distorted, warped, blurry, text, cartoon",
            "lora_url": "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor",
            "lora_strength": 0.8,
            "model_url": "https://civitai.com/api/download/models/471120?type=Model&format=SafeTensor&size=full&fp=fp16",
            "depth_map": png_to_base64url(Image.open("depth_map.png")),
        },
    }

    local_params = {
        "path": "mock/gen/",
        "choice": "oldest"
    }

    # task_id = client.create_request(local_params)
    # result = None
    # while result is None:
    #     time.sleep(0.2)
    #     result = client.poll(task_id)    

    print("Sending request")
    request_queue.put(local_params, block=False)
    print("Waiting for response")
    image_paths = output_queue.get()
    print("Received response:", image_paths)
    for image_path in image_paths:
        image = Image.open(image_path)
        image.show()
    print("Terminating pipeline process")
    request_queue.put(None, block=False)
    pipeline_process.join()
    print("Pipeline process terminated")
    


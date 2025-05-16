import multiprocessing as mp
import time
import queue
from typing import Tuple, Generator, Any, Optional

import numpy as np

from experimance_transition.transition import SimilarityTransition

class TransitionWorker:
    def __init__(self, transition_params: Optional[dict] = None):
        self.image = None

    def run(self, input_queue: mp.Queue, output_queue: mp.Queue):
        while True:
            try:
                image, _, _ = input_queue.get(block=True)
                if image is None:
                    break
                self.image = image
                output_queue.put((1, 1, image))
            except queue.Empty:
                continue
            except StopIteration:
                break
            except TypeError: # if non tuple is sent
                break


class BlendTransitionWorker(TransitionWorker):
    def __init__(self, transition_params: dict):
        self.imageA = transition_params.get("imageA", None)
        self.imageB = transition_params.get("imageB", None)
        self.steps = transition_params.get("steps", 1)

    def run(self, input_queue: mp.Queue, output_queue: mp.Queue):
        while True:
            try:
                image, _, steps = input_queue.get(block=True)
                if image is None:
                    break
                #print("blend image", image.shape, steps)
                if self.imageA is None:
                    #print("blend from black")
                    self.imageA = np.zeros(image.shape, dtype=image.dtype)
                elif self.imageB is not None: # shift previous blend target to source
                    #print("blend imageB -> imageA")
                    self.imageA = self.imageB.copy()
                self.imageB = image
                if steps is None or steps <= 0:
                    steps = self.steps
                # generate all the blended images
                for step in range(1, steps):
                    t = step / (self.steps + 1)
                    blended_image = self.imageA * (1.0 - t) + self.imageB * t
                    output_queue.put((step, steps, blended_image), block=False)
                # output final image
                output_queue.put((steps, steps, self.imageB))
            except queue.Empty:
                #print("blend empty input_queue")
                continue
            except StopIteration:
                break
            except TypeError: # if non-tuple is sent
                break
            



class FlowTransitionWorker(TransitionWorker):
    def __init__(self, transition_params: dict):
        self.transition = SimilarityTransition(**transition_params)
        self.steps = transition_params.get("steps", 1)

    def run(self, input_queue: mp.Queue, output_queue: mp.Queue):
        prev_image = None

        while True:
            try:
                image, location, steps, flowmap = input_queue.get(block=True)
                if image is None:
                    break
                if steps is None or steps <= 0:
                    steps = self.steps
                # location and flowmap can be None

                if prev_image is None:
                    prev_image = image
                    # transition from black
                    black_image = np.zeros(image.shape, dtype=image.dtype)
                    null_flowmap = np.zeros(image.shape[:2], dtype=np.float32)
                    for step, total_steps, frame in self.transition.transition(
                        black_image, image, 
                        location=location, steps=steps, 
                        flowmapA=null_flowmap, flowmapB=flowmap):
                        output_queue.put((step, total_steps, frame))
                else:
                    for step, total_steps, frame in self.transition.transition_to(image, 
                                                                location=location,
                                                                steps=steps,
                                                                flowmap=flowmap):
                        output_queue.put((step, total_steps, frame))
                # output final image
                output_queue.put((total_steps, total_steps, image))

            except queue.Empty:
                continue
            except StopIteration:
                break
            except TypeError: # if non tuple is sent
                break


def start_transition_worker(worker_class:type[TransitionWorker], transition_params:dict, 
                            input_queue: mp.Queue, output_queue: mp.Queue):
    #print("Creating transition worker", worker_class, transition_params)
    worker = worker_class(transition_params)
    worker.run(input_queue, output_queue)


# Multiprocessing transitioner
# This class is used to create a transitioner that runs in a separate process
# and sends the transitioned images to the output queue.
# The transitioner is started by send()ing the first and subsequent images to the 
# input queue. The transitioner will then transition between the images and send
# the transitioned images to the output queue retrived through next().
# Note that next() will not block but always return either None if no images 
# have ever been transitioned to or the last frame transitioned to.
# The transitioner will run until the transition is complete and then return the
# total number of transitions (not the number of steps in the transition) it has done.
class mpTransitioner:
    def __init__(self, 
                 input_queue:mp.Queue, 
                 target,
                 transition_class:type[TransitionWorker],  # Update type hint to specify we expect a class type
                 transition_params) -> None:
        self.input_queue = input_queue  # input queue for incoming images that need to be transitioned
        self.output_queue = mp.Queue()  # output queue for transitioned images
        self.transition_params = transition_params
        self.worker_process = mp.Process(target=target, 
                                         args=(transition_class, transition_params, self.input_queue, self.output_queue))
        self.worker_process.start()

        self.total_transitions = 0
        self.steps = self.transition_params['steps']

    def new_transition(self) -> Generator[Tuple[int, int, np.ndarray], Tuple[np.ndarray, Any, Any], int]:
        step = 0
        frame = None
        last_step = 0
        last_total_steps = 0
        last_frame = np.ndarray(0)

        while True:
            try:
                # FIXME: blaock True or False?
                step, total_steps, frame = self.output_queue.get(block=True)
                last_step, last_total_steps, last_frame = step, total_steps, frame
                yield step, total_steps, frame
                #print("mpTransitioner: after yield frame", step, total_steps, frame.shape)
                if total_steps > 0 and step >= total_steps:
                    self.total_transitions += 1
                    break # done with this generator
            except queue.Empty:
                #print("mpTransitioner: empty output_queue")
                yield last_step, last_total_steps, last_frame
                #print("mpTransitioner: after yield last frame")
                # read new image (using send() to send the image)
                # although images can be added using the input_queue directly
                # try:
                #     new_image, new_location, new_steps = yield
                #     print("mpTransitioner: new image", new_image.shape, new_location, new_steps)
                #     if new_image is None: # must receive image
                #         continue
                #     # NOTE: new_location and new_steps are optional
                #     if new_steps is None:
                #         new_steps = self.steps
                #     self.input_queue.put((new_image, new_location, new_steps))
                # except TypeError:
                #     continue
            except StopIteration:
                break
        return self.total_transitions

    def add_image(self, image, location=None, steps=None):
        try:
            self.input_queue.put((image, location, steps), block=False)
        except queue.Full:
            print("mpTransitioner: input_queue full")
            pass
        except Exception as e:  # catch all exceptions
            print("mpTransitioner: add_image", e)

    def stop(self):
        self.input_queue.put(None)
        self.input_queue.close()
        self.output_queue.close()
        self.worker_process.join()
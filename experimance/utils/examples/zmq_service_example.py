#!/usr/bin/env python3
"""
Example implementation using the new base service classes from experimance_common.service.

This example demonstrates how to:
1. Create services using the base service classes
2. Implement message and task handlers
3. Configure and initialize the communication patterns
4. Use standard service lifecycle methods (start, stop, run)
5. Implement proper error handling and shutdown

Usage:
    # Run as controller
    uv run -m utils.examples.zmq_service_example --controller
    
    # In another terminal, run as worker
    uv run -m utils.examples.zmq_service_example --worker
"""

import argparse
import asyncio
import logging
import random
import signal
import sys
import time
from typing import Dict, Any, List, Optional

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq_utils import MessageType, ZmqTimeoutError
from experimance_common.service import (
    ZmqControllerService, 
    ZmqWorkerService,
    ServiceState
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define ports for our example
CONTROLLER_PUB_PORT = DEFAULT_PORTS["example_pub"]
CONTROLLER_PUSH_PORT = DEFAULT_PORTS["example_pull"]
WORKER_PUSH_PORT = CONTROLLER_PUSH_PORT + 1000  # Use a different port for worker responses

# Topics for PUB/SUB communication
CONTROL_TOPIC = "example.control"
WORKER_RESPONSE_TOPIC = "example.response"

# Generate a random port for worker responses to avoid conflicts
def get_random_worker_port():
    return WORKER_PUSH_PORT + random.randint(1, 1000)


class ExampleController(ZmqControllerService):
    """Example controller service using the base ZmqControllerService class.
    
    This controller:
    1. Publishes heartbeats and control messages
    2. Listens for worker responses
    3. Pushes tasks to workers
    4. Processes worker results
    """
    
    def __init__(self, name: str = "example-controller"):
        """Initialize the controller service.
        
        Args:
            name: Name of this controller instance
        """
        # Initialize with just the publish and push sockets
        # We'll add the subscriber and pull sockets dynamically when workers connect
        super().__init__(
            service_name=name,
            pub_address=f"tcp://*:{CONTROLLER_PUB_PORT}",
            # Use None for now - we'll update these when workers connect
            sub_address=f"tcp://localhost:{CONTROLLER_PUB_PORT}",  # dummy value
            push_address=f"tcp://*:{CONTROLLER_PUSH_PORT}",
            pull_address=f"tcp://localhost:{CONTROLLER_PUSH_PORT}",  # dummy value
            topics=[WORKER_RESPONSE_TOPIC],
            heartbeat_topic=CONTROL_TOPIC,
            service_type="example-controller"
        )
        self.tasks_generated = 0
        self.tasks_completed = 0
        self.connected_workers = {}
        
    async def start(self):
        """Start the controller service with additional tasks."""
        await super().start()
        
        # Register message handlers
        self.register_handler(WORKER_RESPONSE_TOPIC, self.handle_worker_response)
        
        # Save handler for worker results
        self.task_handler = self.handle_worker_result
        
        # Add pull_tasks method for the controller
        self._register_task(self.pull_tasks())
        
        # Add task to register workers
        self._register_task(self.connect_to_workers())
        
        # Additional custom tasks
        self._register_task(self.generate_tasks())
        
        logger.info("Controller service started with all handlers registered")
    
    async def connect_to_workers(self):
        """Connect to new workers as they register."""
        while self.running:
            try:
                # Check all received messages for worker registrations
                for worker_id, info in list(self.connected_workers.items()):
                    if not info.get('connected', False):
                        port = info.get('port')
                        if port:
                            # Connect to the worker's push socket
                            logger.info(f"Connecting to worker {worker_id} on port {port}")
                            
                            # Update the worker info
                            self.connected_workers[worker_id]['connected'] = True
                            
                            # No need to create new sockets, we just track the connection
                
            except Exception as e:
                logger.error(f"Error connecting to workers: {e}")
                self.errors += 1
            
            await asyncio.sleep(1.0)  # Check for new workers periodically
    
    async def pull_tasks(self):
        """Pull results from workers."""
        if not self.pull_socket:
            logger.error("Cannot pull results: pull socket not initialized")
            return
            
        while self.running:
            try:
                result = await self.pull_socket.pull_async()
                if result:
                    # Check if the result is just an empty dict (error case)
                    if not result or len(result) == 0:
                        logger.debug("Received empty result, ignoring")
                        continue
                        
                    task_id = result.get("task_id", "unknown")
                    worker_name = result.get("service", "unknown")
                    
                    # Only process results that have a proper task_id and service name
                    if task_id == "unknown" or worker_name == "unknown":
                        logger.warning(f"Received incomplete task result, missing task_id or service: {result}")
                        continue
                        
                    logger.debug(f"Received result for task: {task_id} from {worker_name}")
                    self.messages_received += 1
                    
                    # Process result with registered handler if any
                    if self.task_handler:
                        try:
                            await self.task_handler(result)
                        except Exception as e:
                            logger.error(f"Error in result handler: {e}")
                            self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error pulling result: {e}")
                self.errors += 1
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
    
    def handle_worker_response(self, message: Dict[str, Any]):
        """Handle worker response messages (via PUB/SUB)."""
        worker_name = message.get("service", "unknown")
        message_type = message.get("type", "unknown")
        
        if message_type == "WorkerRegistration":
            # Handle worker registration
            port = message.get("port")
            if port:
                logger.info(f"Worker {worker_name} registered with port {port}")
                self.connected_workers[worker_name] = {
                    'port': port,
                    'connected': False,
                    'last_seen': time.time()
                }
        else:
            logger.info(f"Received response from {worker_name}: {message}")
    
    async def handle_worker_result(self, result: Dict[str, Any]):
        """Handle worker task results (via PUSH/PULL)."""
        task_id = result.get("task_id", "unknown")
        worker_name = result.get("service", "unknown")
        status = result.get("status", "unknown")
        
        # Validate that we have a real result from a worker
        if worker_name == "unknown" or task_id == "unknown":
            logger.warning(f"Received incomplete or invalid task result: {result}")
            return
            
        # Check if this worker is in our connected workers list
        if worker_name not in self.connected_workers:
            logger.warning(f"Received result from unknown worker '{worker_name}' for task {task_id}")
            # Add to connected workers for future tasks
            self.connected_workers[worker_name] = {
                'connected': True,
                'last_seen': time.time()
            }
            
        logger.info(f"Task {task_id} completed by {worker_name} with status: {status}")
        self.tasks_completed += 1
    
    async def generate_tasks(self):
        """Generate and distribute tasks to workers."""
        while self.running:
            # Wait a bit before generating a new task
            await asyncio.sleep(random.uniform(2.0, 5.0))
            
            # Check if any workers are connected
            worker_count = len([w for w in self.connected_workers.values() if w.get('connected', False)])
            
            # Generate a new task
            self.tasks_generated += 1
            task_id = f"task-{self.tasks_generated}"
            
            # Create the task message
            task = {
                "type": MessageType.RENDER_REQUEST,
                "id": task_id,
                "timestamp": time.time(),
                "task_id": task_id,
                "parameters": {
                    "complexity": random.randint(1, 5),
                    "priority": random.choice(["low", "normal", "high"])
                }
            }
            
            # Push the task to workers
            success = await self.push_task(task)
            if success:
                if worker_count > 0:
                    logger.info(f"Generated and pushed task {task_id} to {worker_count} worker(s)")
                else:
                    logger.warning(f"Generated task {task_id}, but no workers are connected. Task may be lost or never processed.")
                
                # Also publish a notification about the new task
                notification = {
                    "type": MessageType.RENDER_REQUEST,
                    "timestamp": time.time(),
                    "task_id": task_id,
                    "message": f"New task {task_id} is available"
                }
                await self.publish_message(notification)
            else:
                logger.warning(f"Failed to push task {task_id}")


class ExampleWorker(ZmqWorkerService):
    """Example worker service using the base ZmqWorkerService class.
    
    This worker:
    1. Subscribes to controller heartbeats and control messages
    2. Pulls tasks from the controller
    3. Processes tasks and sends results back
    """
    
    def __init__(self, name: str = "example-worker"):
        """Initialize the worker service.
        
        Args:
            name: Name of this worker instance
        """
        # Use a random worker port to avoid conflicts
        worker_port = get_random_worker_port()
        
        super().__init__(
            service_name=name,
            sub_address=f"tcp://localhost:{CONTROLLER_PUB_PORT}",
            pull_address=f"tcp://localhost:{CONTROLLER_PUSH_PORT}",
            push_address=f"tcp://*:{worker_port}",  # Bind to this address for sending responses
            topics=[CONTROL_TOPIC],
            service_type="example-worker"
        )
        self.tasks_processed = 0
        self.worker_port = worker_port
        
    async def start(self):
        """Start the worker service with additional handlers."""
        await super().start()
        
        # Register message handlers
        self.register_handler(CONTROL_TOPIC, self.handle_control_message)
        
        # Save task handler
        self.task_handler = self.process_task
        
        # Register with controller after starting
        self._register_task(self.register_with_controller_task())
        
        logger.info("Worker service started with all handlers registered")
    
    async def register_with_controller_task(self):
        """Periodically attempt to register with the controller."""
        # Wait a moment for connections to establish
        await asyncio.sleep(1.0)
        
        # Initial registration
        await self.register_with_controller()
        
        # Periodically re-register
        while self.running:
            await asyncio.sleep(10.0)  # Register every 10 seconds
            await self.register_with_controller()
    
    async def register_with_controller(self):
        """Send registration information to the controller."""
        registration = {
            "type": "WorkerRegistration",
            "timestamp": time.time(),
            "service": self.service_name,
            "port": self.worker_port,
            "state": self.state
        }
        
        # Send via PUSH/PULL for direct communication
        success = await self.send_response(registration)
        if success:
            logger.info(f"Registered with controller (port {self.worker_port})")
        else:
            logger.warning("Failed to register with controller")
    
    def handle_control_message(self, message: Dict[str, Any]):
        """Handle control messages from the controller."""
        message_type = message.get("type", "unknown")
        
        if message_type == MessageType.HEARTBEAT:
            logger.debug(f"Received heartbeat from controller")
            
            # Send a status update to the controller
            asyncio.create_task(self.send_status_update())
        elif message_type == MessageType.RENDER_REQUEST:
            # This is just a notification, actual task comes through PULL socket
            task_id = message.get("task_id", "unknown")
            logger.info(f"Received notification about new task: {task_id}")
        else:
            logger.info(f"Received control message: {message}")
    
    async def send_status_update(self):
        """Send a status update to the controller."""
        status = {
            "type": MessageType.IDLE_STATUS,
            "timestamp": time.time(),
            "service": self.service_name,
            "tasks_processed": self.tasks_processed,
            "state": self.state
        }
        
        # Send via PUSH/PULL for direct communication
        success = await self.send_response(status)
        if success:
            logger.debug(f"Sent status update via PUSH/PULL")
        else:
            logger.warning("Failed to send status update via PUSH/PULL")
    
    async def process_task(self, task: Dict[str, Any]):
        """Process a task received from the controller."""
        task_id = task.get("id", "unknown")
        complexity = task.get("parameters", {}).get("complexity", 1)
        priority = task.get("parameters", {}).get("priority", "normal")
        
        logger.info(f"Processing task {task_id} (complexity: {complexity}, priority: {priority})")
        
        # Simulate processing time based on complexity
        await asyncio.sleep(complexity * 0.5)
        
        # Update task counter
        self.tasks_processed += 1
        
        # Send result back to controller
        result = {
            "type": MessageType.IMAGE_READY,
            "timestamp": time.time(),
            "service": self.service_name,
            "task_id": task_id,
            "status": "completed",
            "result": {
                "processing_time": complexity * 0.5,
                "priority": priority
            }
        }
        
        # Send result via PUSH/PULL
        success = await self.send_response(result)
        if success:
            logger.info(f"Sent result for task {task_id}")
        else:
            logger.warning(f"Failed to send result for task {task_id}")


async def main():
    """Main entry point for the example service."""
    parser = argparse.ArgumentParser(description="ZMQ Service Example using Base Service Classes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--controller", action="store_true", help="Run as controller")
    group.add_argument("--worker", action="store_true", help="Run as worker")
    parser.add_argument("--name", type=str, help="Service instance name")
    args = parser.parse_args()
    
    # Create the appropriate service based on arguments
    if args.controller:
        name = args.name or "controller-1"
        service = ExampleController(name=name)
    else:
        name = args.name or f"worker-{random.randint(1, 1000)}"
        service = ExampleWorker(name=name)
    
    # Flag to track if we've started stopping the service
    stopping = False
        
    try:
        # Start and run the service
        await service.start()
        await service.run()
    except asyncio.CancelledError:
        logger.info("Service was cancelled")
        stopping = True
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        stopping = True
    except Exception as e:
        logger.error(f"Error running service: {e}", exc_info=True)
    finally:
        try:
            # Skip cleanup if we're already exiting or system is finalizing
            if sys.is_finalizing():
                logger.debug("System is finalizing, skipping explicit cleanup")
                return
                    
            # Make sure we stop the service with a timeout to prevent hanging
            if service and service.state != ServiceState.STOPPED and not stopping:
                logger.debug("Stopping service from main's finally block")
                stop_task = asyncio.create_task(service.stop())
                try:
                    await asyncio.wait_for(stop_task, timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for service to stop")
                except asyncio.CancelledError:
                    logger.debug("Stop task was cancelled")
                except Exception as e:
                    logger.warning(f"Error during stop: {e}")
        except Exception as e:
            logger.error(f"Error during service cleanup: {e}")
            # Don't re-raise, we're in cleanup mode


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt at the top level
        print("\nShutdown requested by keyboard interrupt")
        sys.exit(0)
    except RuntimeError as e:
        if "Event loop stopped before Future completed" in str(e):
            # This is expected during shutdown, exit gracefully
            sys.exit(0)
        else:
            logger.error(f"Unhandled RuntimeError: {e}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

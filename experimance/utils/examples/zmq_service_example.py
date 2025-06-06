#!/usr/bin/env python3
"""
Example implementation using the new base service classes from experimance_common.service

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
import time
from typing import Dict, Any, List, Optional

from experimance_common.constants import TICK
from experimance_common.zmq.zmq_utils import (
    MessageType, 
    ZmqTimeoutError, 
    ZmqPullSocket,
    ZmqPublisher,
    ZmqSubscriber,
    ZmqPushSocket,
    ZmqBindingPullSocket,
    ZmqConnectingPushSocket
)
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.controller import ZmqControllerService
from experimance_common.zmq.worker import ZmqWorkerService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define ports for our example
CONTROLLER_PUB_PORT = 5567    # 5567 - Controller publishes heartbeats/commands
CONTROLLER_PUSH_PORT = 5568   # 5568 - Controller pushes tasks 
CONTROLLER_PULL_PORT = CONTROLLER_PUSH_PORT + 10     # 5578 - Controller pulls results from workers
WORKER_RESPONSE_PUB_PORT = CONTROLLER_PUB_PORT + 20  # 5587 - Workers publish responses

# Topics for PUB/SUB communication
CONTROL_TOPIC = "example.control"
WORKER_RESPONSE_TOPIC = "example.response"
    

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
        # Initialize with just the publish and push sockets for sending to workers
        # and pull socket for receiving from workers
        super().__init__(
            service_name=name,
            pub_address=f"tcp://*:{CONTROLLER_PUB_PORT}",
            sub_address=f"tcp://localhost:99999",  # Dummy address - not used since no subscription needed
            push_address=f"tcp://*:{CONTROLLER_PUSH_PORT}",
            pull_address=f"tcp://*:{CONTROLLER_PULL_PORT}",  # Bind to receive results from workers
            topics=[],  # No subscription topics needed
            heartbeat_topic=CONTROL_TOPIC,
            service_type="example-controller"
        )
        self.tasks_generated = 0
        self.tasks_completed = 0
        self.connected_workers = {}
        
    async def start(self):
        """Start the controller service with additional tasks."""
        # Initialize publisher for broadcasting
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber for receiving responses (dummy since not used)
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize push socket for distributing tasks
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        # Initialize custom binding pull socket for receiving worker responses
        logger.info(f"Initializing binding pull socket on {self.pull_address}")
        self.pull_socket = ZmqBindingPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)
        
        # Save handler for worker results
        self.task_handler = self.handle_worker_result
        
        # Add task to register workers
        self.add_task(self.connect_to_workers())
        
        # Additional custom tasks
        self.add_task(self.generate_tasks())
        
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
                    
                    # Process result with registered handler if any
                    if self.task_handler:
                        try:
                            await self.task_handler(result)
                        except Exception as e:
                            logger.error(f"Error in result handler: {e}")
                            self.errors += 1
                    
                    self.messages_received += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error pulling result: {e}")
                self.errors += 1
            
            await asyncio.sleep(TICK)  # Small delay to prevent CPU spinning
    
    async def handle_worker_result(self, result: Dict[str, Any]):
        """Handle worker task results and registrations (via PUSH/PULL)."""
        message_type = result.get("type", "unknown")
        worker_name = result.get("service", "unknown")
        
        if worker_name == "unknown":
            logger.warning(f"Received incomplete task result, missing task_id or service: {result}")
            return
        
        # Handle worker registration messages
        if message_type == "WorkerRegistration":
            logger.info(f"Worker {worker_name} registered")
            self.connected_workers[worker_name] = {
                'connected': True,
                'last_seen': time.time()
            }
            return
        
        if worker_name not in self.connected_workers:
            logger.warning(f"Received result from unknown worker '{worker_name}'")
            return

        # Handle worker status updates
        if message_type == MessageType.IDLE_STATUS:
            tasks_processed = result.get("tasks_processed", 0)
            logger.debug(f"Status update from worker {worker_name}: {tasks_processed} tasks processed")
            
            # Update worker status in connected workers list
            if worker_name not in self.connected_workers:
                self.connected_workers[worker_name] = {
                    'connected': True,
                    'last_seen': time.time(),
                    'tasks_processed': tasks_processed
                }
            else:
                self.connected_workers[worker_name]['last_seen'] = time.time()
                self.connected_workers[worker_name]['tasks_processed'] = tasks_processed
                
            return
        
        # For task results, we need a task_id
        task_id = result.get("task_id", "unknown")
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
        super().__init__(
            service_name=name,
            sub_address=f"tcp://localhost:{CONTROLLER_PUB_PORT}",
            pull_address=f"tcp://localhost:{CONTROLLER_PUSH_PORT}",
            push_address=f"tcp://localhost:{CONTROLLER_PULL_PORT}",  # Connect to controller's pull socket
            topics=[CONTROL_TOPIC],
            service_type="example-worker"
        )
        self.tasks_processed = 0
        
    async def start(self):
        """Start the worker service with additional handlers."""
        # Initialize subscriber for receiving control messages
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize pull socket for receiving tasks
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Initialize custom connecting push socket for sending responses back
        if self.push_address:
            logger.info(f"Initializing connecting push socket on {self.push_address}")
            self.push_socket = ZmqConnectingPushSocket(self.push_address)
            self.register_socket(self.push_socket)
        
        # Register tasks
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)
        
        # Register message handlers
        self.register_handler(CONTROL_TOPIC, self.handle_control_message)
        
        # Save task handler
        self.task_handler = self.process_task
        
        # Register with controller after starting
        self.add_task(self.register_with_controller_task())
        
        logger.info("Worker service started with all handlers registered")
    
    async def register_with_controller_task(self):
        """Periodically attempt to register with the controller."""
        # Wait a moment for connections to establish
        await asyncio.sleep(1.0)
        
        # Initial registration
        await self.register_with_controller()
        
        # Periodically re-register
        while self.running:
            if await self._sleep_if_running(10.0): break # Re-register every 10 seconds
            await self.register_with_controller()
    
    async def register_with_controller(self):
        """Send registration information to the controller."""
        registration = {
            "type": "WorkerRegistration",
            "timestamp": time.time(),
            "service": self.service_name,
            "state": self.state.value if hasattr(self.state, 'value') else str(self.state)
        }
        
        # Send via PUSH socket to controller
        success = await self.send_response(registration)
        if success:
            logger.info(f"Registered with controller via PUSH socket")
        else:
            logger.warning("Failed to register with controller")
    
    def handle_control_message(self, message: Dict[str, Any]):
        """Handle control messages from the controller."""
        message_type = message.get("type", "unknown")
        
        if message_type == MessageType.HEARTBEAT:
            logger.debug(f"Received heartbeat from controller")
            
            # Send a status update to the controller
            asyncio.create_task(self.send_status_update())
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
    
    async def send_response(self, response: Dict[str, Any]) -> bool:
        """Send a response back to the controller.
        
        Args:
            response: Response message to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.push_socket:
            logger.error(f"Cannot send response: push socket not initialized")
            return False
            
        try:
            # Ensure we have our service name in the response
            if "service" not in response:
                response["service"] = self.service_name
                
            # Add timestamp if not present
            if "timestamp" not in response:
                response["timestamp"] = time.time()
                
            # Send the message using the push socket
            result = await self.push_socket.push_async(response)
            self.messages_sent += 1
            return True
        except ZmqTimeoutError:
            logger.warning(f"Timeout sending response to controller")
            self.errors += 1
            return False
        except Exception as e:
            logger.error(f"Error sending response to controller: {e}")
            self.errors += 1
            return False


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
    
    # Start and run the service
    await service.start()
    logger.info(f"{name} service is started")
    await service.run()
    logger.info(f"{name} service is complete")

if __name__ == "__main__":
    asyncio.run(main())


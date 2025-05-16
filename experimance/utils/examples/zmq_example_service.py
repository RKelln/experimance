#!/usr/bin/env python3
"""
Example implementation of ZeroMQ communication patterns using the experimance_common ZMQ utilities.

This example demonstrates a complete service architecture with:
1. Proper publisher-subscriber (PUB/SUB) pattern for broadcasting messages
2. Efficient push-pull (PUSH/PULL) pattern for task distribution
3. Bidirectional communication between controller and workers
4. Graceful shutdown handling
5. Proper error handling with timeouts

Usage:
    # Run as controller
    python -m utils.examples.zmq_example_service --controller --name controller-1
    
    # In another terminal, run as worker
    python -m utils.examples.zmq_example_service --name worker-1

Key concepts demonstrated:
- ZMQ socket initialization and cleanup
- Handling socket timeouts gracefully
- Properly structuring concurrent operations with asyncio
- Signal handling for graceful termination
- Bidirectional communication patterns
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from typing import Dict, Any, Optional, List

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq_utils import (
    ZmqPublisher,
    ZmqSubscriber,
    ZmqPushSocket,
    ZmqPullSocket,
    MessageType,
    ZmqTimeoutError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test ports
TEST_PUB_PORT = DEFAULT_PORTS["example_pub"]  # 5567
TEST_PULL_PORT = DEFAULT_PORTS["example_pull"]  # 5568
TEST_WORKER_RESPONSE_PORT = TEST_PULL_PORT + 1000  # 6568

# Topics
HEARTBEAT_TOPIC = "service.heartbeat"  # Topic for controller heartbeat messages
CONTROL_TOPIC = "service.control"      # Topic for control messages from workers

# ZMQ communication patterns:
# 1. PUB/SUB: Controller broadcasts heartbeats to all workers
#    - Controller binds to PUB socket on TEST_PUB_PORT
#    - Workers connect to SUB socket on TEST_PUB_PORT
#
# 2. PUSH/PULL: Controller distributes tasks to workers
#    - Controller binds to PUSH socket on TEST_PULL_PORT
#    - Workers connect to PULL socket on TEST_PULL_PORT
#
# 3. PUSH/PULL (reverse): Workers send status back to controller
#    - Workers bind to PUSH socket on TEST_WORKER_RESPONSE_PORT
#    - Controller connects to PULL socket on TEST_WORKER_RESPONSE_PORT


class ExampleService:
    """Example service demonstrating ZMQ communication patterns."""

    def __init__(self, service_name: str, is_controller: bool = False):
        """Initialize the example service.
        
        Args:
            service_name: Name of this service instance
            is_controller: Whether this instance acts as a controller
        
        Notes:
            The ZMQ socket initialization follows these patterns:
            - Controllers bind to publisher and task push sockets
            - Workers connect to these sockets to receive messages
            - Workers bind to their own push socket for sending status back
            - Controller connects to worker status pull socket
        """
        self.service_name = service_name
        self.is_controller = is_controller
        self.running = False
        self.tasks_processed = 0
        
        # Statistics tracking
        self.start_time = time.time()
        self.messages_sent = 0
        self.messages_received = 0
        self.last_stats_time = self.start_time
        
        # Initialize ZMQ components
        if is_controller:
            # Controller publishes heartbeats and pushes tasks
            self.publisher = ZmqPublisher(f"tcp://*:{TEST_PUB_PORT}", HEARTBEAT_TOPIC)
            self.subscriber = ZmqSubscriber(f"tcp://localhost:{TEST_PUB_PORT}", [CONTROL_TOPIC])
            self.push_socket = ZmqPushSocket(f"tcp://*:{TEST_PULL_PORT}")
            self.pull_socket = ZmqPullSocket(f"tcp://localhost:{TEST_WORKER_RESPONSE_PORT}")
        else:
            # Worker subscribes to heartbeats and pulls tasks
            # Note: Workers don't bind to publisher port, they only connect to the controller's
            self.publisher = None  # Workers don't need to publish via PUB/SUB
            self.subscriber = ZmqSubscriber(f"tcp://localhost:{TEST_PUB_PORT}", [HEARTBEAT_TOPIC])
            self.push_socket = ZmqPushSocket(f"tcp://*:{TEST_WORKER_RESPONSE_PORT}")  # Use a different port for worker responses
            self.pull_socket = ZmqPullSocket(f"tcp://localhost:{TEST_PULL_PORT}")
        
        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals.
        
        This ensures proper cleanup of ZMQ sockets on service termination.
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), shutting down gracefully...")
        self.running = False
    
    async def send_heartbeat(self):
        """Send periodic heartbeat messages."""
        # Only controllers send heartbeats via publisher
        if not self.is_controller or not self.publisher:
            return
            
        while self.running:
            heartbeat = {
                "type": MessageType.HEARTBEAT,
                "timestamp": time.time(),
                "service": self.service_name,
                "tasks_processed": self.tasks_processed
            }
            
            success = await self.publisher.publish_async(heartbeat)
            if success:
                logger.info(f"Sent heartbeat: {self.service_name}")
                self.messages_sent += 1
            else:
                logger.warning("Failed to send heartbeat")
            
            await asyncio.sleep(2)  # Send heartbeat every 2 seconds
    
    async def listen_for_messages(self):
        """Listen for incoming messages on subscribed topics."""
        if not self.subscriber:
            return
            
        while self.running:
            try:
                topic, message = await self.subscriber.receive_async()
                logger.info(f"Received message on {topic}: {message}")
                self.messages_received += 1
                
                # Respond to heartbeat messages if we're a worker (using push socket)
                if not self.is_controller and topic == HEARTBEAT_TOPIC and self.push_socket:
                    # Send a control message acknowledging the heartbeat
                    response = {
                        "type": MessageType.IDLE_STATUS,
                        "timestamp": time.time(),
                        "service": self.service_name,
                        "status": "ready",
                        "tasks_processed": self.tasks_processed
                    }
                    await self.push_socket.push_async(response)
                    logger.info(f"Sent worker status: {self.service_name}")
                    self.messages_sent += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
    
    async def push_tasks(self):
        """Push tasks for workers (controller only)."""
        if not self.is_controller or not self.push_socket:
            return
        
        task_id = 0
        while self.running:
            # Generate a new task
            task_id += 1
            task = {
                "type": MessageType.RENDER_REQUEST,
                "id": f"task-{task_id}",
                "timestamp": time.time(),
                "parameters": {
                    "complexity": task_id % 5 + 1,
                    "priority": "normal"
                }
            }
            
            success = await self.push_socket.push_async(task)
            if success:
                logger.info(f"Pushed task: {task['id']}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to push task: {task['id']}")
            
            await asyncio.sleep(5)  # Push a new task every 5 seconds
    
    async def pull_tasks(self):
        """Pull and process tasks (worker only)."""
        if self.is_controller or not self.pull_socket:
            return
        
        while self.running:
            try:
                task = await self.pull_socket.pull_async()
                if task:
                    self.messages_received += 1
                    logger.info(f"Received task: {task['id']}")
                    
                    # Simulate processing
                    complexity = task.get("parameters", {}).get("complexity", 1)
                    await asyncio.sleep(complexity)
                    logger.info(f"Completed task: {task['id']}")
                    self.tasks_processed += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error processing task: {e}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
    
    async def check_worker_responses(self):
        """Check for worker status responses (controller only)."""
        if not self.is_controller or not self.pull_socket:
            return
            
        while self.running:
            try:
                response = await self.pull_socket.pull_async()
                if response:
                    self.messages_received += 1
                    logger.info(f"Received worker response: {response}")
                    
                    # Process worker status
                    if response.get("type") == MessageType.IDLE_STATUS:
                        worker_name = response.get("service", "unknown")
                        tasks_processed = response.get("tasks_processed", 0)
                        logger.info(f"Worker {worker_name} is ready, processed {tasks_processed} tasks")
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error checking worker responses: {e}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
    
    async def display_stats(self):
        """Periodically display service statistics."""
        while self.running:
            await asyncio.sleep(10)  # Update stats every 10 seconds
            
            now = time.time()
            elapsed = now - self.start_time
            elapsed_since_last = now - self.last_stats_time
            
            # Calculate message rates
            sent_rate = self.messages_sent / elapsed if elapsed > 0 else 0
            received_rate = self.messages_received / elapsed if elapsed > 0 else 0
            
            # Format uptime as hours:minutes:seconds
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            
            # Prepare and log statistics
            stats = {
                "uptime": uptime_str,
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "tasks_processed": self.tasks_processed,
                "msg_send_rate": f"{sent_rate:.2f}/s",
                "msg_recv_rate": f"{received_rate:.2f}/s"
            }
            
            logger.info(f"Stats for {self.service_name}: {stats}")
            self.last_stats_time = now
    
    async def run(self):
        """Run the service."""
        self.running = True
        logger.info(f"Starting {self.service_name} as {'controller' if self.is_controller else 'worker'}")
        
        # Create and gather tasks
        tasks = [
            self.listen_for_messages(),
            self.display_stats()  # Always show statistics
        ]
        
        # Controllers send heartbeats, push tasks, and check worker responses
        if self.is_controller:
            tasks.append(self.send_heartbeat())
            tasks.append(self.push_tasks())
            tasks.append(self.check_worker_responses())
        # Workers pull tasks
        else:
            tasks.append(self.pull_tasks())
        
        await asyncio.gather(*tasks)
    
    def close(self):
        """Close all ZMQ sockets and perform cleanup.
        
        This method ensures all sockets are properly closed,
        preventing resource leaks and socket errors on restart.
        """
        logger.info(f"Closing ZMQ sockets for {self.service_name}...")
        
        # Close sockets in reverse order of initialization
        if self.pull_socket:
            logger.debug("Closing pull socket")
            self.pull_socket.close()
            self.pull_socket = None
        
        if self.push_socket:
            logger.debug("Closing push socket")
            self.push_socket.close() 
            self.push_socket = None
        
        if self.subscriber:
            logger.debug("Closing subscriber")
            self.subscriber.close()
            self.subscriber = None
        
        if self.publisher:
            logger.debug("Closing publisher")
            self.publisher.close()
            self.publisher = None
            
        logger.info("All sockets closed")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ZMQ Example Service")
    parser.add_argument("--controller", action="store_true", help="Run as controller")
    parser.add_argument("--name", type=str, default="example-service", help="Service name")
    args = parser.parse_args()
    
    service = ExampleService(args.name, args.controller)
    
    try:
        # Register asyncio-specific signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(service, s))
            )
        
        await service.run()
    except asyncio.CancelledError:
        logger.info("Main task was cancelled")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        service.close()
        logger.info("Service shutdown complete")


async def shutdown(service, sig):
    """Perform graceful shutdown when receiving a signal."""
    signal_name = signal.Signals(sig).name
    logger.info(f"Received signal {signal_name} in asyncio event loop")
    
    # Set service to not running state
    service.running = False
    
    # Give pending tasks a moment to notice the running flag change
    await asyncio.sleep(0.5)
    
    # Get the current task (main task) and cancel it
    loop = asyncio.get_running_loop()
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    
    if tasks:
        logger.info(f"Cancelling {len(tasks)} pending tasks")
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop the event loop
    loop.stop()


if __name__ == "__main__":
    asyncio.run(main())

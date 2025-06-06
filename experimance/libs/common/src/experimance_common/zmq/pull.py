"""
ZeroMQ Pull Service for Experimance.

This module provides the ZmqPullService class for receiving
tasks from pushers using the ZeroMQ PULL pattern.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict

from experimance_common.constants import TICK
from experimance_common.service import BaseZmqService
from experimance_common.service_state import ServiceState
from experimance_common.zmq_utils import ZmqPullSocket, ZmqTimeoutError

logger = logging.getLogger(__name__)

class ZmqPullService(BaseZmqService):
    """Service that pulls tasks from pushers.
    
    This service type establishes a ZeroMQ PULL socket to receive
    tasks from pushing services.
    """
    
    def __init__(self, service_name: str, 
                 pull_address: str,
                 service_type: str = "pull"):
        """Initialize a pull service.
        
        Args:
            service_name: Unique name for this service instance
            pull_address: ZeroMQ address to connect pull socket to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.pull_address = pull_address
        self.pull_socket = None
        self.task_handler = None
    
    async def start(self):
        """Start the pull service."""
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register message listening task
        self.add_task(self.pull_tasks())
        
        await super().start()
    
    def register_task_handler(self, handler: Callable[[Dict[str, Any]], Coroutine]):
        """Register a handler for incoming tasks.
        
        Args:
            handler: Async function to call with task data
        """
        self.task_handler = handler
    
    async def pull_tasks(self):
        """Pull and process tasks."""
        if not self.pull_socket:
            logger.error("Cannot pull tasks: pull socket not initialized")
            return
            
        while self.state == ServiceState.RUNNING:
            try:
                task = await self.pull_socket.pull_async()
                if task:
                    logger.debug(f"Received task: {task.get('id', 'unknown')}")
                    self.messages_received += 1
                    
                    # Process task with registered handler if any
                    if self.task_handler:
                        try:
                            await self.task_handler(task)
                        except Exception as e:
                            logger.error(f"Error in task handler: {e}")
                            self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error pulling task: {e}")
                self.errors += 1
            
            await asyncio.sleep(TICK)  # Small delay to prevent CPU spinning

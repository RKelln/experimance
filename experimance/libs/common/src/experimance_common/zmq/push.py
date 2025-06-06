"""
ZeroMQ Push Service for Experimance.

This module provides the ZmqPushService class for sending
tasks to workers using the ZeroMQ PUSH pattern.
"""

import logging
from typing import Any, Dict

from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.zmq_utils import ZmqPushSocket

logger = logging.getLogger(__name__)

class ZmqPushService(BaseZmqService):
    """Service that pushes tasks to workers.
    
    This service type establishes a ZeroMQ PUSH socket to distribute
    tasks to pulling workers.
    """
    
    def __init__(self, service_name: str, 
                 push_address: str,
                 service_type: str = "push"):
        """Initialize a push service.
        
        Args:
            service_name: Unique name for this service instance
            push_address: ZeroMQ address to bind push socket to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.push_address = push_address
        self.push_socket = None
    
    async def start(self):
        """Start the push service."""
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        await super().start()
    
    async def push_task(self, task: Dict[str, Any]) -> bool:
        """Push a task to workers.
        
        Args:
            task: Task data to send
            
        Returns:
            True if task was sent successfully, False otherwise
        """
        if not self.push_socket:
            logger.error("Cannot push task: push socket not initialized")
            self.errors += 1
            return False
        
        try:
            success = await self.push_socket.push_async(task)
            if success:
                logger.debug(f"Pushed task: {task.get('id', 'unknown')}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to push task: {task.get('id', 'unknown')}")
                self.errors += 1
            return success
        except Exception as e:
            logger.error(f"Error pushing task: {e}")
            self.errors += 1
            return False

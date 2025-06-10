"""
ZeroMQ Worker Service for Experimance.

This module provides the ZmqWorkerService class which combines
subscriber, pull, and push patterns for implementing worker
services that process tasks and report results.
"""

import logging
from typing import Any, Dict, List, Optional

from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.zmq_utils import ZmqSubscriber, ZmqPullSocket, ZmqPushSocket
from experimance_common.zmq.subscriber import ZmqSubscriberService
from experimance_common.zmq.pull import ZmqPullService
from experimance_common.zmq.push import ZmqPushService

logger = logging.getLogger(__name__)

class ZmqWorkerService(ZmqSubscriberService, ZmqPullService, ZmqPushService):
    """Worker service that subscribes to events and pulls tasks.
    
    This combined service is suitable for worker services that
    need to listen for control messages and receive tasks to process.
    """
    
    def __init__(self, service_name: str,
                 sub_address: str,
                 pull_address: str,
                 push_address: Optional[str] = None,
                 topics: List[str] = [],
                 service_type: str = "worker"):
        """Initialize a worker service.
        
        Args:
            service_name: Unique name for this service instance
            sub_address: ZeroMQ address to connect subscriber to
            pull_address: ZeroMQ address to connect pull socket to
            push_address: Optional ZeroMQ address to bind push socket for responses
            topics: List of topics to subscribe to
            service_type: Type of service (for logging and monitoring)
        """
        ZmqSubscriberService.__init__(self, service_name, sub_address, topics, service_type)
        ZmqPullService.__init__(self, service_name, pull_address, service_type)

        self.push_address:Optional[str] = push_address
        if self.push_address is not None:
            ZmqPushService.__init__(self, service_name, self.push_address, service_type)
    
    async def start(self):
        """Start the worker service."""
        # Initialize subscriber for receiving control messages
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.subscribe_topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.subscribe_topics)
        self.register_socket(self.subscriber)
        
        # Initialize pull socket for receiving tasks
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Initialize push socket for sending responses back (if address provided)
        if self.push_address:
            logger.info(f"Initializing push socket on {self.push_address}")
            self.push_socket = ZmqPushSocket(self.push_address)
            self.register_socket(self.push_socket)
        
        # Register tasks
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)
    
    async def send_response(self, response: Dict[str, Any]) -> bool:
        """Send a response back to the controller.
        
        Args:
            response: Response data to send
            
        Returns:
            True if response was sent successfully, False otherwise
        """
        if not self.push_socket:
            logger.error("Cannot send response: push socket not initialized")
            self.errors += 1
            return False
        
        try:
            success = await self.push_socket.push_async(response)
            if success:
                logger.debug(f"Sent response: {response.get('type', 'unknown')}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to send response: {response.get('type', 'unknown')}")
                self.errors += 1
            return success
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.errors += 1
            return False

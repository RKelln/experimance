"""
ZeroMQ Publisher Service for Experimance.

This module provides the ZmqPublisherService class for broadcasting
messages to subscribers on specific topics using ZeroMQ.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from experimance_common.constants import HEARTBEAT_INTERVAL, HEARTBEAT_TOPIC
from experimance_common.service_state import ServiceState
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.zmq_utils import MessageType, ZmqPublisher, topic_to_str, topics_to_strs, MessageDataType, TopicType
from experimance_common.schemas import MessageBase

logger = logging.getLogger(__name__)

class ZmqPublisherService(BaseZmqService):
    """Service that publishes messages on specific topics.
    
    This service type establishes a ZeroMQ PUB socket to broadcast
    messages to subscribing services.
    """
    
    def __init__(self, service_name: str, 
                 pub_address: str, 
                 topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "publisher"):
        """Initialize a publisher service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            topic: Topic for messages
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.pub_address = pub_address
        self.topic = topic_to_str(topic)
        self.publisher:Optional[ZmqPublisher] = None
    
    async def start(self):
        """Start the publisher service."""
        self.publisher = ZmqPublisher(self.pub_address, self.topic)
        logger.info(f"Initialized {self.publisher}")
        self.register_socket(self.publisher)
        
        # Register heartbeat task - _register_task will automatically create a Task
        self.add_task(self.send_heartbeat())
        
        await super().start()
    
    async def send_heartbeat(self, interval: float = HEARTBEAT_INTERVAL):
        """Send periodic heartbeat messages.
        
        Args:
            interval: Time between heartbeats in seconds
        """
        while self.state == ServiceState.RUNNING:
            try:
                heartbeat = {
                    "type": MessageType.HEARTBEAT,
                    "timestamp": time.time(),
                    "service": self.service_name,
                    "state": self.state
                }
                
                if self.publisher:
                    success = await self.publisher.publish_async(heartbeat)
                    if success:
                        logger.debug(f"Sent heartbeat: {self.service_name}")
                        self.messages_sent += 1
                    else:
                        logger.warning("Failed to send heartbeat")
                        self.errors += 1
                else:
                    logger.warning("Cannot send heartbeat: publisher not initialized")
                    self.errors += 1
            
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                self.errors += 1
            
            await asyncio.sleep(interval)
    
    async def publish_message(self, message: MessageDataType, topic: Optional[TopicType] = None) -> bool:
        """Publish a message to subscribers.
        
        Args:
            message: Message to publish (dict or Pydantic model)
            topic: Topic to publish on (if None, uses the default heartbeat topic)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.publisher:
            logger.error("Cannot publish message: publisher not initialized")
            self.errors += 1
            return False
        
        # get topic from message if provided, either in dict or as a Pydantic model
        if topic is None:
            if isinstance(message, MessageBase):
                topic = message.type
            elif isinstance(message, dict) and "topic" in message:
                topic = message["topic"]

        if topic is not None:
            topic = topic_to_str(topic)
            logger.debug(f"Publishing message to custom topic: {topic}")
        
        # If topic provided, create a new publisher or use existing one with that topic
        if topic is not None and topic != self.topic:
            logger.warning(f"Publishing to custom topic {topic}, not the default {self.topic}")
            temp_publisher = ZmqPublisher(self.pub_address, topic)
            try:
                success = await temp_publisher.publish_async(message)
                if success:
                    self.messages_sent += 1
                else:
                    self.errors += 1
                return success
            finally:
                temp_publisher.close()
        else:
            # Use the default publisher
            try:
                success = await self.publisher.publish_async(message)
                if success:
                    self.messages_sent += 1
                else:
                    self.errors += 1
                return success
            except Exception as e:
                logger.error(f"Error publishing message: {e}")
                self.errors += 1
                return False

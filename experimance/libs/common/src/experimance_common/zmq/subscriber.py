"""
ZeroMQ Subscriber Service for Experimance.

This module provides the ZmqSubscriberService class for receiving
messages from publishers on specific topics using ZeroMQ.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Union

from experimance_common.constants import TICK
from experimance_common.service_state import ServiceState
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.zmq_utils import ZmqSubscriber, ZmqTimeoutError

logger = logging.getLogger(__name__)

class ZmqSubscriberService(BaseZmqService):
    """Service that subscribes to messages on specific topics.
    
    This service type establishes a ZeroMQ SUB socket to receive
    broadcasts from publishing services.
    """
    
    def __init__(self, service_name: str, 
                 sub_address: str, 
                 topics: List[str],
                 service_type: str = "subscriber"):
        """Initialize a subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.sub_address = sub_address
        self.subscribe_topics = topics
        self.subscriber = None
        self.message_handlers = {}
    
    async def start(self):
        """Start the subscriber service."""
        self.subscriber = ZmqSubscriber(self.sub_address, self.subscribe_topics)
        logger.info(f"Initialized {self.subscriber}")
        self.register_socket(self.subscriber)
        
        # Register message listening task - _register_task will automatically create a Task
        self.add_task(self.listen_for_messages())
        
        await super().start()
    
    def register_handler(self, topic: str, 
                        handler: Union[Callable[[Dict[str, Any]], None], 
                                      Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]]):
        """Register a handler for a specific topic.
        
        Accepts both synchronous and asynchronous handler functions.
        
        Args:
            topic: Topic to handle messages for
            handler: Function to call with message data (can be sync or async)
        """
        if topic not in self.subscribe_topics:
            logger.warning(f"Registering handler for topic {topic} which is not in subscription list")
        
        self.message_handlers[topic] = handler
    
    async def listen_for_messages(self):
        """Listen for incoming messages on subscribed topics."""
        if not self.subscriber:
            logger.error("Cannot listen for messages: subscriber not initialized")
            return
            
        while self.state == ServiceState.RUNNING:
            try:
                topic, message = await self.subscriber.receive_async()
                logger.debug(f"Received message on {topic}: {message}")
                self.messages_received += 1
                
                # Process message with registered handler if any
                if topic in self.message_handlers:
                    try:
                        handler = self.message_handlers[topic]
                        # Check if the handler is a coroutine function and await if it is
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            # Call synchronous handler directly
                            handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler for topic {topic}: {e}")
                        self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                self.errors += 1
            
            await asyncio.sleep(TICK)  # Small delay to prevent CPU spinning

"""
ZeroMQ Publisher-Subscriber Service for Experimance.

This module provides the ZmqPublisherSubscriberService class which
combines the publisher and subscriber patterns for bidirectional
communication using ZeroMQ.
"""

import logging

from experimance_common.constants import HEARTBEAT_TOPIC
from experimance_common.service import BaseZmqService
from experimance_common.zmq_utils import ZmqPublisher, ZmqSubscriber
from experimance_common.zmq.publisher import ZmqPublisherService
from experimance_common.zmq.subscriber import ZmqSubscriberService

logger = logging.getLogger(__name__)

class ZmqPublisherSubscriberService(ZmqPublisherService, ZmqSubscriberService):
    """Service that both publishes and subscribes to messages.
    
    This combined service type is suitable for services that need to
    both broadcast their state and listen for events from other services.
    """
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 topics: list,
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "pubsub"):
        """Initialize a publisher-subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type of service (for logging and monitoring)
        """
        BaseZmqService.__init__(self, service_name, service_type)
        self.pub_address = pub_address
        self.sub_address = sub_address
        self.topics = topics
        self.heartbeat_topic = heartbeat_topic
        self.publisher = None
        self.subscriber = None
        self.message_handlers = {}
    
    async def start(self):
        """Start the publisher-subscriber service."""
        # Initialize publisher
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        
        await BaseZmqService.start(self)

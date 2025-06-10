"""
ZeroMQ Publisher-Subscriber Service for Experimance.

This module provides the ZmqPublisherSubscriberService class which
combines the publisher and subscriber patterns for bidirectional
communication using ZeroMQ.
"""

import logging

from experimance_common.constants import HEARTBEAT_TOPIC
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.zmq_utils import ZmqPublisher, ZmqSubscriber, topics_to_strs, topic_to_str
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
                 subscribe_topics: list,
                 publish_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "pubsub"):
        """Initialize a publisher-subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            topic: Topic for messages
            service_type: Type of service (for logging and monitoring)
        """
        BaseZmqService.__init__(self, service_name, service_type)
        self.pub_address = pub_address
        self.sub_address = sub_address
        self.subscribe_topics = topics_to_strs(subscribe_topics)
        self.publish_topic = topic_to_str(publish_topic)
        self.publisher = None
        self.subscriber = None
        self.message_handlers = {}
    
    async def start(self):
        """Start the publisher-subscriber service."""
        # Initialize publisher
        self.publisher = ZmqPublisher(self.pub_address, self.publish_topic)
        logger.info(f"Initialized {self.publisher}")
        self.register_socket(self.publisher)
        
        # Initialize subscriber
        self.subscriber = ZmqSubscriber(self.sub_address, self.subscribe_topics)
        logger.info(f"Initialized {self.subscriber}")
        self.register_socket(self.subscriber)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        
        await BaseZmqService.start(self)

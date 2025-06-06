"""
ZeroMQ Controller Service for Experimance.

This module provides the ZmqControllerService class which combines
publisher, subscriber, push, and pull patterns for centralized
control of a distributed system using ZeroMQ.
"""

import logging

from experimance_common.constants import HEARTBEAT_TOPIC
from experimance_common.service import BaseZmqService
from experimance_common.zmq_utils import ZmqPublisher, ZmqSubscriber, ZmqPullSocket, ZmqPushSocket
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.push import ZmqPushService
from experimance_common.zmq.pull import ZmqPullService

logger = logging.getLogger(__name__)

class ZmqControllerService(ZmqPublisherSubscriberService, ZmqPushService, ZmqPullService):
    """Controller service that publishes events, listens for responses, and pushes tasks.
    
    This combined service is suitable for central coordinator services that
    need to broadcast messages, listen for responses, and distribute tasks.
    It also pulls responses from workers.
    """
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 push_address: str,
                 pull_address: str,
                 topics: list,
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "controller"):
        ZmqPublisherSubscriberService.__init__(
            self,
            service_name=service_name,
            pub_address=pub_address,
            sub_address=sub_address,
            topics=topics,
            heartbeat_topic=heartbeat_topic,
            service_type=service_type
        )
        ZmqPushService.__init__(self, service_name=service_name, push_address=push_address, service_type=service_type)
        ZmqPullService.__init__(self, service_name=service_name, pull_address=pull_address, service_type=service_type)

        # Register the handler for messages from the PULL socket
        #self.register_task_handler(self._handle_worker_response)
    
    async def start(self):
        """Start the controller service."""
        # Initialize publisher for broadcasting
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber for receiving responses
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize push socket for distributing tasks
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        # Initialize pull socket for receiving worker responses
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)

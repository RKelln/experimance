"""
Experimance common package initialization.
"""

from experimance_common.constants import (
    DEFAULT_PORTS,
)

from experimance_common.zmq.zmq_utils import (
    ZmqPublisher,
    ZmqSubscriber,
    ZmqPushSocket,
    ZmqPullSocket,
    ZmqBindingPullSocket,
    ZmqConnectingPushSocket,
    MessageType,
    HEARTBEAT_INTERVAL,
)

from experimance_common.schemas import (
    Era,
    Biome,
    TransitionStyle,
    MessageBase,
    EraChanged,
    RenderRequest,
    IdleStatus,
    ImageReady,
    TransitionReady,
    LoopReady,
    AgentControlEvent,
    TransitionRequest,
    LoopRequest,
)

from experimance_common.config import (
    load_config_with_overrides,
    Config,
    ConfigError,
)

# Import service base classes
from experimance_common.base_service import (
    BaseService,
    ServiceState,
)

# Import ZMQ service classes
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.publisher import ZmqPublisherService
from experimance_common.zmq.subscriber import ZmqSubscriberService
from experimance_common.zmq.push import ZmqPushService
from experimance_common.zmq.pull import ZmqPullService
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.controller import ZmqControllerService
from experimance_common.zmq.worker import ZmqWorkerService

# logging
from experimance_common.logger import configure_external_loggers

__all__ = [
    # ZMQ utilities
    "ZmqPublisher",
    "ZmqSubscriber",
    "ZmqPushSocket",
    "ZmqPullSocket",
    "ZmqBindingPullSocket",
    "ZmqConnectingPushSocket",
    "MessageType",
    "DEFAULT_PORTS",
    "HEARTBEAT_INTERVAL",
    
    # Base Service
    "BaseService",

    # ZMQWservices
    "BaseZmqService",
    "ServiceState",
    "ZmqPublisherService",
    "ZmqSubscriberService",
    "ZmqPushService",
    "ZmqPullService",
    "ZmqPublisherSubscriberService",
    "ZmqControllerService",
    "ZmqWorkerService",
    
    # Schema definitions
    "Era",
    "Biome",
    "TransitionStyle",
    "MessageBase",
    "EraChanged",
    "RenderRequest",
    "IdleStatus",
    "ImageReady",
    "TransitionReady",
    "LoopReady",
    "AgentControlEvent",
    "TransitionRequest",
    "LoopRequest",
    
    # Configuration utilities
    "load_config_with_overrides",
    "Config",
    "ConfigError",

    # logging utilities
    "configure_external_loggers",
    
    # Test utilities - imported separately to avoid dependencies in production
    # Use: from experimance_common.test_utils import active_service, wait_for_service_state
]

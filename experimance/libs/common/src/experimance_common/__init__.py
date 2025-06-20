"""
Experimance common package initialization.
"""

# Core constants
from experimance_common.constants import (
    DEFAULT_PORTS,
    HEARTBEAT_INTERVAL,
    DEFAULT_IMAGE_TRANSPORT_MODE,
)

# ZMQ configuration and utilities
from experimance_common.zmq.config import (
    MessageType,
    MessageDataType,
    TopicType,
    ZmqSocketConfig,
    PublisherConfig,
    SubscriberConfig,
    PushConfig,
    PullConfig,
    PubSubServiceConfig,
    WorkerServiceConfig,
    ControllerServiceConfig,
)

from experimance_common.zmq.components import (
    BaseZmqComponent,
    PublisherComponent,
    SubscriberComponent,
    PushComponent,
    PullComponent,
    ZmqComponentError,
    ComponentNotRunningError,
)

from experimance_common.zmq.services import (
    BaseZmqService,
    PubSubService,
    WorkerService,
    ControllerService,
)

from experimance_common.zmq.zmq_utils import (
    MessageType,  # Re-export for backward compatibility
    image_ready_to_display_media,
    create_display_media_message,
    choose_image_transport_mode,
    is_local_address,
    prepare_image_message,
    cleanup_temp_image_file,
)

# Schema definitions
from experimance_common.schemas import (
    Era,
    Biome,
    TransitionStyle,
    DisplayContentType,
    DisplayTransitionType,
    ContentType,
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
    DisplayMedia,
)

# Configuration utilities
from experimance_common.config import (
    load_config_with_overrides,
    BaseConfig,
    ConfigError,
)

# Service base classes
from experimance_common.base_service import (
    BaseService,
    ServiceState,
)

# Logging utilities
from experimance_common.logger import configure_external_loggers

__all__ = [
    # Core constants
    "DEFAULT_PORTS",
    "HEARTBEAT_INTERVAL", 
    "DEFAULT_IMAGE_TRANSPORT_MODE",
    
    # ZMQ configuration and utilities
    "MessageType",
    "MessageDataType",
    "TopicType",
    "ZmqSocketConfig",
    "PublisherConfig",
    "SubscriberConfig", 
    "PushConfig",
    "PullConfig",
    "PubSubServiceConfig",
    "WorkerServiceConfig",
    "ControllerServiceConfig",
    
    # ZMQ components
    "BaseZmqComponent",
    "PublisherComponent",
    "SubscriberComponent",
    "PushComponent", 
    "PullComponent",
    "ZmqComponentError",
    "ComponentNotRunningError",
    
    # ZMQ services
    "BaseZmqService",
    "PubSubService",
    "WorkerService",
    "ControllerService",
    
    # ZMQ utilities
    "image_ready_to_display_media",
    "create_display_media_message",
    "choose_image_transport_mode",
    "is_local_address",
    "prepare_image_message",
    "cleanup_temp_image_file",
    
    # Schema definitions
    "Era",
    "Biome",
    "TransitionStyle", 
    "DisplayContentType",
    "DisplayTransitionType",
    "ContentType",
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
    "DisplayMedia",
    
    # Configuration utilities
    "load_config_with_overrides",
    "BaseConfig",
    "ConfigError",
    
    # Base Service
    "BaseService",
    "ServiceState",
    
    # Logging utilities
    "configure_external_loggers",
    
    # Test utilities - imported separately to avoid dependencies in production
    # Use: from experimance_common.test_utils import active_service, wait_for_service_state
]

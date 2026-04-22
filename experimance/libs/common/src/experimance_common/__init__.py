"""
Experimance common package initialization.
"""

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

# 1. load whichever .env was pointed to by UV_ENV_FILE or docker 'ENV' line
load_dotenv(find_dotenv(), override=False)

# 2. Detect and set project from .project file if PROJECT_ENV is not already set
from experimance_common.project_utils import ensure_project_env_set
ensure_project_env_set()

# 3. Load project (and variant) .env — PROJECT_ENV is preserved across all loads
from experimance_common.project_utils import load_project_dotenv
load_project_dotenv()


# Core constants
from experimance_common.constants_base import (
    DEFAULT_PORTS,
    DEFAULT_IMAGE_TRANSPORT_MODE,
)

# ZMQ configuration and utilities
from experimance_common.zmq.config import (
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

# Schema definitions
from experimance_common.schemas_base import (
    TransitionStyle,
    DisplayContentType,
    DisplayTransitionType,
    ContentType,
    MessageType,
    MessageBase,
    SpaceTimeUpdate,
    RenderRequest,
    PresenceStatus,
    ImageReady,
    TransitionReady,
    LoopReady,
    AudiencePresent,
    SpeechDetected,
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
    
    # Schema definitions (project-specific Era/Biome available via direct import)
    "TransitionStyle", 
    "DisplayContentType",
    "DisplayTransitionType",
    "ContentType",
    "MessageBase",
    "SpaceTimeUpdate",
    "RenderRequest",
    "PresenceStatus",
    "ImageReady",
    "TransitionReady",
    "LoopReady",
    "RequestBiome",
    "AudiencePresent",
    "SpeechDetected",
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

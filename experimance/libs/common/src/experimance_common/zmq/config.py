"""
ZMQ Configuration Schemas

Pydantic configuration models for the composition-based ZMQ architecture.
These schemas integrate seamlessly with the existing config.py system while
providing type-safe configuration for ZMQ components and services.

Key Features:
- Extends the existing Config base class for compatibility
- Supports all existing override and file loading capabilities  
- Type-safe configuration with validation
- Clean separation from ZMQ implementation
"""

from typing import Dict, List, Optional, Any, TypeAlias, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator

from experimance_common.config import BaseConfig
from experimance_common.schemas import MessageBase
from experimance_common.constants import (
    DEFAULT_PORTS, HEARTBEAT_TOPIC, DEFAULT_TIMEOUT, HEARTBEAT_INTERVAL,
    ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX
)

class MessageType(str, Enum):
    """Message types used in the Experimance system."""
    ERA_CHANGED = "EraChanged"
    RENDER_REQUEST = "RenderRequest"
    IDLE_STATUS = "IdleStatus"
    IMAGE_READY = "ImageReady"
    TRANSITION_READY = "TransitionReady"
    LOOP_READY = "LoopReady"
    AGENT_CONTROL_EVENT = "AgentControlEvent"
    TRANSITION_REQUEST = "TransitionRequest"
    LOOP_REQUEST = "LoopRequest"
    HEARTBEAT = "Heartbeat"
    ALERT = "Alert"
    # Display service message types
    DISPLAY_MEDIA = "DisplayMedia"
    TEXT_OVERLAY = "TextOverlay"
    REMOVE_TEXT = "RemoveText"
    CHANGE_MAP = "ChangeMap"
    # Add more message types as needed

    def __str__(self) -> str:
        """Return the string representation of the message type."""
        return self.value
    
    # allow for comparison with strings
    def __eq__(self, other: Any) -> bool:
        """Allow comparison with string values."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

class ZmqException(Exception):
    """Base exception for ZMQ-related errors."""
    pass


class ZmqTimeoutError(ZmqException):
    """Exception raised when a ZMQ operation times out."""
    pass

TopicType: TypeAlias = str | MessageType
MessageDataType: TypeAlias = Union[Dict[str, Any], MessageBase]

# =============================================================================
# CORE ZMQ CONFIGURATION CLASSES
# =============================================================================

class ZmqSocketConfig(BaseModel):
    """Configuration for a single ZMQ socket."""
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    address: str = Field(..., description="ZMQ address (e.g., tcp://*)")
    port: int = Field(..., ge=1024, le=65535, description="Port number")
    bind: bool = Field(default=True, description="True to bind, False to connect")
    socket_options: Dict[str, Any] = Field(default_factory=dict, description="Additional ZMQ socket options")
    
    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Validate ZMQ address format."""
        if not v.startswith(('tcp://', 'inproc://', 'ipc://')):
            raise ValueError("Address must use tcp://, inproc://, or ipc:// protocol")
        return v
    
    @property
    def full_address(self) -> str:
        """Get the complete ZMQ address with port."""
        return f"{self.address}:{self.port}"


class PublisherConfig(ZmqSocketConfig):
    """Configuration for Publisher component."""
    bind: bool = Field(default=True, description="Publishers typically bind")
    default_topic: Optional[TopicType] = Field(default=None, description="Default topic for publishing messages")
    
    @field_validator('default_topic', mode='before')
    @classmethod
    def validate_default_topic(cls, v):
        """Convert MessageType enum to string if needed."""
        if isinstance(v, MessageType):
            return v.value
        return v
    

class SubscriberConfig(ZmqSocketConfig):
    """Configuration for Subscriber component."""
    bind: bool = Field(default=False, description="Subscribers typically connect")
    topics: List[str] = Field(default_factory=list, description="Topics to subscribe to")
    

class PushConfig(ZmqSocketConfig):
    """Configuration for Push component."""
    bind: bool = Field(default=False, description="Push sockets typically connect to workers")
    

class PullConfig(ZmqSocketConfig):
    """Configuration for Pull component."""
    bind: bool = Field(default=True, description="Pull sockets typically bind (workers wait for work)")


# =============================================================================
# SPECIALIZED CONFIGS FOR CONTROLLER/WORKER PATTERNS
# =============================================================================

class ControllerPushConfig(PushConfig):
    """Push configuration for controllers that distribute work to workers."""
    bind: bool = Field(default=True, description="Controllers bind to distribute work to workers")
    address: str = Field(default=ZMQ_TCP_BIND_PREFIX, description="Controllers use bind addresses")


class ControllerPullConfig(PullConfig):
    """Pull configuration for controllers that collect results from workers."""
    bind: bool = Field(default=True, description="Controllers bind to collect results from workers")
    address: str = Field(default=ZMQ_TCP_BIND_PREFIX, description="Controllers use bind addresses")


class WorkerPushConfig(PushConfig):
    """Push configuration for workers that send results to controllers."""
    bind: bool = Field(default=False, description="Workers connect to controllers to send results")
    address: str = Field(default=ZMQ_TCP_CONNECT_PREFIX, description="Workers use connect addresses")


class WorkerPullConfig(PullConfig):
    """Pull configuration for workers that receive work from controllers."""
    bind: bool = Field(default=False, description="Workers connect to controllers to receive work")
    address: str = Field(default=ZMQ_TCP_CONNECT_PREFIX, description="Workers use connect addresses")


# =============================================================================
# WORKER CONFIGURATION
# =============================================================================

class WorkerConfig(BaseModel):
    """Configuration for a worker (PUSH/PULL pair) as defined by a controller."""
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    name: str = Field(..., description="Worker name")
    push_config: ControllerPushConfig = Field(..., description="Controller's PUSH socket configuration for this worker")
    pull_config: ControllerPullConfig = Field(..., description="Controller's PULL socket configuration for this worker")
    message_types: List[str] = Field(default_factory=list, description="Message types this worker handles")
    max_queue_size: int = Field(default=1000, ge=1, description="Maximum queue size")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate worker name."""
        if not v or not v.strip():
            raise ValueError("Worker name cannot be empty")
        return v.strip()


# =============================================================================
# SERVICE CONFIGURATION CLASSES
# =============================================================================

class PubSubServiceConfig(BaseConfig):
    """
    Configuration for PubSubService.
    Extends your existing Config class for full integration.
    """
    
    # Service identification
    name: str = Field(default="pubsub", description="Service name")
    log_level: str = Field(default="INFO", description="Logging level")
    timeout: float = Field(default=DEFAULT_TIMEOUT/1000, gt=0, description="Operation timeout in seconds")
    
    # ZMQ configuration
    publisher: Optional[PublisherConfig] = Field(default=None, description="Publisher configuration (optional)")
    subscriber: Optional[SubscriberConfig] = Field(default=None, description="Subscriber configuration (optional)")
    
    # Service settings
    heartbeat_interval: float = Field(default=HEARTBEAT_INTERVAL, gt=0, description="Heartbeat interval in seconds")


class WorkerServiceConfig(BaseConfig):
    """
    Configuration for WorkerService.
    Combines PubSub with Push/Pull functionality.
    """
    
    # Service identification
    name: str = Field(default="worker", description="Service name")
    log_level: str = Field(default="INFO", description="Logging level")
    timeout: float = Field(default=DEFAULT_TIMEOUT/1000, gt=0, description="Operation timeout in seconds")
    
    # PubSub configuration
    publisher: PublisherConfig = Field(..., description="Publisher configuration")
    subscriber: SubscriberConfig = Field(..., description="Subscriber configuration")
    
    # Worker configuration - uses worker-specific configs with correct bind defaults
    push: WorkerPushConfig = Field(..., description="Push configuration for sending results")
    pull: WorkerPullConfig = Field(..., description="Pull configuration for receiving work")
    
    # Service settings
    heartbeat_interval: float = Field(default=HEARTBEAT_INTERVAL, gt=0, description="Heartbeat interval in seconds")
    work_timeout: float = Field(default=60.0, gt=0, description="Work processing timeout")
    max_concurrent_tasks: int = Field(default=10, gt=0, description="Maximum concurrent work tasks")


class ControllerServiceConfig(BaseConfig):
    """
    Configuration for ControllerService.
    Manages PubSub plus multiple workers.
    """
    
    # Service identification
    name: str = Field(default="controller", description="Service name")
    log_level: str = Field(default="INFO", description="Logging level")
    timeout: float = Field(default=DEFAULT_TIMEOUT/1000, gt=0, description="Operation timeout in seconds")
    
    # PubSub configuration
    publisher: PublisherConfig = Field(..., description="Publisher configuration")
    subscriber: SubscriberConfig = Field(..., description="Subscriber configuration")
    
    # Workers configuration
    workers: Dict[str, WorkerConfig] = Field(default_factory=dict, description="Worker configurations by name")
    max_workers: int = Field(default=4, ge=1, description="Maximum number of concurrent workers")
    
    # Service settings
    heartbeat_interval: float = Field(default=HEARTBEAT_INTERVAL, gt=0, description="Heartbeat interval in seconds")
    worker_timeout: float = Field(default=60.0, gt=0, description="Worker timeout in seconds")
    
    @field_validator('workers')
    @classmethod
    def validate_workers(cls, v: Dict[str, WorkerConfig]) -> Dict[str, WorkerConfig]:
        """Validate worker configurations."""
        if not v:
            return v
            
        # Check for port conflicts
        used_ports = set()
        for worker_name, worker_config in v.items():
            push_port = worker_config.push_config.port
            pull_port = worker_config.pull_config.port
            
            if push_port in used_ports:
                raise ValueError(f"Port conflict: {push_port} used by multiple workers")
            if pull_port in used_ports:
                raise ValueError(f"Port conflict: {pull_port} used by multiple workers")
                
            used_ports.add(push_port)
            used_ports.add(pull_port)
            
        return v


# =============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# =============================================================================

def create_local_pubsub_config(
    name: str = "pubsub",
    pub_port: Optional[int] = DEFAULT_PORTS["events"],
    sub_port: Optional[int] = DEFAULT_PORTS["events"],
    sub_topics: List[TopicType] = [HEARTBEAT_TOPIC],
    default_pub_topic: Optional[TopicType] = HEARTBEAT_TOPIC
) -> PubSubServiceConfig:
    """
    Create a local PubSub configuration for quick setup and convenience.
    
    ⚠️  USAGE GUIDANCE:
    - **Production services**: Use PubSubServiceConfig.from_overrides() with proper config files
    - **Unit/integration tests**: Use mocks or minimal inline configs
    - **Quick setup/examples/prototypes**: Use this factory function for convenience
    
    This factory provides sensible defaults for local development, examples, one-off scripts,
    and prototyping. For production services, use the full BaseConfig integration for 
    proper configuration management with files, overrides, and validation.
    
    Args:
        name: Service name
        pub_port: Publisher port (None to disable publisher)
        sub_port: Subscriber port (None to disable subscriber)  
        sub_topics: Topics to subscribe to
        default_pub_topic: Default topic for publishing
        
    Returns:
        PubSubServiceConfig with local TCP configuration
        
    Example:
        # Quick setup/examples - OK
        config = create_local_pubsub_config("example-service", pub_port=5555)
        
        # Production - PREFER BaseConfig integration instead:
        config = PubSubServiceConfig.from_overrides(
            default_config={...},
            config_file="service.toml"
        )
        
        # Testing - PREFER mocks instead:
        config = Mock(spec=PubSubServiceConfig)
    """

    # Create publisher config if pub_port is provided
    publisher_config = None
    if pub_port is not None:
        publisher_config = PublisherConfig(
            address=ZMQ_TCP_BIND_PREFIX,
            port=pub_port,
            default_topic=default_pub_topic
        )
    
    # Create subscriber config if sub_port is provided
    subscriber_config = None
    if sub_port is not None:
        subscriber_config = SubscriberConfig(
            address=ZMQ_TCP_CONNECT_PREFIX,
            port=sub_port,
            topics=sub_topics
        )

    return PubSubServiceConfig(
        name=name,
        publisher=publisher_config,
        subscriber=subscriber_config
    )


def create_local_controller_config(
    name: str = "controller",
    pub_port: int = DEFAULT_PORTS["events"],
    sub_port: int = DEFAULT_PORTS["events"],
    worker_configs: Dict[str, Dict[str, int]] = {},
    default_topic: Optional[TopicType] = HEARTBEAT_TOPIC
) -> ControllerServiceConfig:
    """
    Create a local Controller configuration for quick setup and convenience.
    
    ⚠️  USAGE GUIDANCE:
    - **Production services**: Use ControllerServiceConfig.from_overrides() with proper config files
    - **Unit/integration tests**: Use mocks or minimal inline configs
    - **Quick setup/examples/prototypes**: Use this factory function for convenience
    
    This factory handles the complexity of setting up controller + worker socket
    configurations with proper port assignments and binding patterns. Use for
    examples, prototypes, and quick development setups.
    
    Args:
        name: Controller name
        pub_port: Publisher port (defaults to events port)
        sub_port: Subscriber port (defaults to events port)
        worker_configs: Dict like {"image": {"push": image_requests, "pull": image_results}}
        default_topic: Default topic for publishing
        
    Returns:
        ControllerServiceConfig with workers configured
        
    Example:
        # Production use - OK
        config = create_local_controller_config(
            name="image-controller",
            worker_configs={"image": {"push": 5564, "pull": 5565}}
        )
        
        # Testing use - NOT RECOMMENDED, use mocks instead:
        # mock_config = Mock(spec=ControllerServiceConfig)
        # mock_config.workers = {}
    """
    # Create worker configurations
    workers = {}
    for worker_name, ports in worker_configs.items():
        workers[worker_name] = WorkerConfig(
            name=worker_name,
            push_config=ControllerPushConfig(
                port=ports["push"]
            ),
            pull_config=ControllerPullConfig(
                port=ports["pull"]
            ),
            message_types=[f"{worker_name}.generate", f"{worker_name}.process"]
        )
    
    return ControllerServiceConfig(
        name=name,
        publisher=PublisherConfig(
            address=ZMQ_TCP_BIND_PREFIX,
            port=pub_port,
            default_topic=default_topic
        ),
        subscriber=SubscriberConfig(
            address=ZMQ_TCP_CONNECT_PREFIX,
            port=sub_port,
            topics=[HEARTBEAT_TOPIC, "image.ready", "transition.complete"]
        ),
        workers=workers
    )


def create_worker_service_config(
    name: str = "worker",
    work_pull_port: int = DEFAULT_PORTS["image_requests"],
    result_push_port: int = DEFAULT_PORTS["image_results"],
    pub_port: int = DEFAULT_PORTS["events"],
    sub_port: int = DEFAULT_PORTS["events"],
    sub_topics: List[str] = [],
    default_pub_topic: Optional[TopicType] = HEARTBEAT_TOPIC
) -> WorkerServiceConfig:
    """
    Create a WorkerService configuration for quick setup and convenience.
    
    ⚠️  USAGE GUIDANCE:
    - **Production services**: Use WorkerServiceConfig.from_overrides() with proper config files
    - **Unit/integration tests**: Use mocks or minimal inline configs
    - **Quick setup/examples/prototypes**: Use this factory function for convenience
    
    This factory handles the complexity of worker socket configuration, ensuring
    proper binding patterns (worker PULL binds, PUSH connects). Use for examples,
    prototypes, and quick development setups.
    
    Worker pattern:
    - PULL socket BINDS to receive work from controllers
    - PUSH socket CONNECTS to send results back to controllers
    - PUB/SUB for general communication
    
    Args:
        name: Worker service name
        work_pull_port: Port to bind PULL socket for receiving work
        result_push_port: Port to connect PUSH socket for sending results  
        pub_port: Publisher port for events
        sub_port: Subscriber port for events
        sub_topics: Topics to subscribe to
        default_pub_topic: Default publishing topic
        
    Returns:
        WorkerServiceConfig with proper socket binding patterns
        
    Example:
        # Production use - OK
        config = create_worker_service_config(
            name="image-worker",
            work_pull_port=5564,
            result_push_port=5565
        )
        
        # Testing use - NOT RECOMMENDED, use mocks instead:
        # mock_config = Mock(spec=WorkerServiceConfig)
        # mock_config.pull = Mock()
        # mock_config.push = Mock()
    """  
    return WorkerServiceConfig(
        name=name,
        publisher=PublisherConfig(
            address=ZMQ_TCP_BIND_PREFIX,
            port=pub_port,
            default_topic=default_pub_topic
        ),
        subscriber=SubscriberConfig(
            address=ZMQ_TCP_CONNECT_PREFIX,
            port=sub_port,
            topics=sub_topics
        ),
        pull=WorkerPullConfig(
            port=work_pull_port
        ),
        push=WorkerPushConfig(
            port=result_push_port
        )
    )


# =============================================================================
# CONFIGURATION VALIDATION HELPERS
# =============================================================================

def validate_no_port_conflicts(*configs: ZmqSocketConfig) -> None:
    """Validate that multiple socket configs don't have port conflicts."""
    used_ports = set()
    
    for config in configs:
        if config.port in used_ports:
            raise ValueError(f"Port conflict: {config.port} used by multiple sockets")
        used_ports.add(config.port)


def get_all_ports_from_config(config: ControllerServiceConfig) -> Dict[str, int]:
    """Extract all ports from a controller configuration for debugging."""
    ports = {
        "publisher": config.publisher.port,
        "subscriber": config.subscriber.port
    }
    
    for worker_name, worker_config in config.workers.items():
        ports[f"{worker_name}_push"] = worker_config.push_config.port
        ports[f"{worker_name}_pull"] = worker_config.pull_config.port
        
    return ports


if __name__ == "__main__":
    # Example usage showing proper BaseConfig integration
    print("=== ZMQ Configuration Examples ===")
    
    # Simple factory method for basic configs
    print("\n1. Simple PubSub Config:")
    pubsub_config = create_local_pubsub_config("test_pubsub", sub_topics=["image.ready"])
    assert pubsub_config.publisher is not None, "Publisher should be configured"
    assert pubsub_config.subscriber is not None, "Subscriber should be configured"
    print(f"   Publisher: {pubsub_config.publisher.full_address}")
    print(f"   Subscriber: {pubsub_config.subscriber.full_address}")
    print(f"   Topics: {pubsub_config.subscriber.topics}")
    
    # Controller config with workers
    print("\n2. Controller Config with Workers:")
    controller_config = create_local_controller_config("test_controller")
    print(f"   Publisher: {controller_config.publisher.full_address}")
    print(f"   Workers: {list(controller_config.workers.keys())}")
    
    ports = get_all_ports_from_config(controller_config)
    print("   Ports:", ports)
    
    # Proper integration with BaseConfig.from_overrides()
    print("\n3. Proper BaseConfig Integration:")
    default_config = {
        "name": "production_controller",
        "log_level": "INFO",
        "publisher": {"address": "tcp://*", "port": 5555},
        "subscriber": {"address": "tcp://localhost", "port": 5556, "topics": ["heartbeat"]},
        "workers": {}
    }
    
    override_config = {
        "log_level": "DEBUG",
        "publisher": {"port": 6000}  # Override just the port
    }
    
    # Use BaseConfig.from_overrides() directly - no wrapper functions needed
    config = ControllerServiceConfig.from_overrides(
        default_config=default_config,
        override_config=override_config
    )
    print(f"   Final config - Name: {config.name}, Log: {config.log_level}")
    print(f"   Final config - Pub port: {config.publisher.port}")
    
    print("\n✅ Configuration system ready for ZMQ components!")

# ZMQ Architecture Guide for Experimance

**✅ VALIDATED ARCHITECTURE**: This guide describes a fully tested and validated ZMQ architecture that has been proven to work correctly with real examples.

This guide explains how to build robust ZMQ services using the Experimance ZMQ architecture, which combines composition-based ZMQ components with BaseService lifecycle management.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Building Your First ZMQ Service](#building-your-first-zmq-service)
4. [Configuration System](#configuration-system)
5. [Message Schemas](#message-schemas)
6. [Communication Patterns](#communication-patterns)
7. [BaseService Integration](#baseservice-integration)
8. [Error Handling and Best Practices](#error-handling-and-best-practices)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Examples](#examples)
11. [Validation and Testing](#validation-and-testing)

## Architecture Overview

The Experimance ZMQ architecture uses **composition over inheritance** to create flexible, robust messaging services. The key principle is:

**BaseService + ZMQ Services = Production-Ready Service**

```
┌─────────────────┐    ┌─────────────────────────────────┐
│   BaseService   │    │         ZMQ Services            │
│                 │    │                                 │
│ • Lifecycle     │ +  │ • PubSubService                 │ = Your Service
│ • Signals       │    │ • WorkerService                 │
│ • Statistics    │    │ • ControllerService             │
│ • Error Recovery│    │   ┌───────────────────────────┐ │
└─────────────────┘    │   │     ZMQ Components        │ │
                       │   │ • Publisher • Subscriber  │ │
                       │   │ • Push      • Pull        │ │
                       │   │ • Async Context           │ │
                       │   └───────────────────────────┘ │
                       └─────────────────────────────────┘
```

### Key Features

- **Type-safe configuration** with Pydantic schemas
- **Graceful shutdown** with signal handling
- **Automatic cleanup** of ZMQ resources
- **Error recovery** and logging
- **Flexible topic handling** (strings or enums)
- **Integration** with existing config system

## Core Components

### 1. Configuration (`config.py`)

Pydantic schemas for type-safe, validated configuration:

```python
from experimance_common.zmq.config import (
    PubSubServiceConfig, PublisherConfig, SubscriberConfig
)

# Simple direct configuration
config = PubSubServiceConfig(
    name="my-service",
    publisher=PublisherConfig(
        address="tcp://*",
        port=5555,
        default_topic="general"
    ),
    subscriber=SubscriberConfig(
        address="tcp://localhost", 
        port=5556,
        topics=["heartbeat", "status"]
    )
)
```

### 2. Message Schemas (`schemas.py`)

Type-safe message definitions:

```python
from experimance_common.schemas import (
    MessageBase, EraChanged, Era, Biome
)

# Schema-based messages
era_message = EraChanged(era=Era.CURRENT, biome=Biome.RAINFOREST)

# Simple dict messages for testing
heartbeat = {
    "type": "heartbeat",
    "service": "my-service", 
    "timestamp": time.time()
}
```

### 3. Components (`components.py`)

Low-level ZMQ socket management:

```python
from experimance_common.zmq.components import (
    PublisherComponent, SubscriberComponent,
    PushComponent, PullComponent
)

# Components handle individual socket types
async with PublisherComponent(pub_config) as publisher:
    await publisher.publish(message, "topic")
```

### 4. Services (`services.py`)

High-level composed services:

```python
from experimance_common.zmq.services import (
    PubSubService, WorkerService, ControllerService
)

# Services combine multiple components
service = PubSubService(config)
```

### 5. Constants (`constants.py`)

Centralized configuration values:

```python
from experimance_common.constants import DEFAULT_PORTS, HEARTBEAT_TOPIC

# Use predefined ports and topics
pub_port = DEFAULT_PORTS["events"]  # 5555
topic = HEARTBEAT_TOPIC  # "heartbeat"
```

## Building Your First ZMQ Service

### Step 1: Create Your Service Cla

```python
from experimance_common.base_service import BaseService
from experimance_common.service_state import ServiceState
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.zmq.services import PubSubService
from experimance_common.constants import DEFAULT_PORTS

class MyZmqService(BaseService):
    """Example ZMQ service using BaseService + PubSubService."""
    
    def __init__(self, name: str = "my-zmq-service"):
        super().__init__(service_name=name, service_type="zmq-example")
        
        # Create ZMQ configuration with simple direct instantiation
        self.zmq_config = PubSubServiceConfig(
            name=self.service_name,
            publisher=PublisherConfig(
                address="tcp://*",
                port=DEFAULT_PORTS["events"],
                default_topic="general"
            ),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=DEFAULT_PORTS["events"],
                topics=["heartbeat", "status"]
            )
        )
        
        # Create ZMQ service
        self.zmq_service = PubSubService(self.zmq_config)
```

### Step 2: Implement Lifecycle Methods

**⚠️ IMPORTANT**: Do NOT manually set states when using BaseService. Let BaseService handle state transitions automatically through its decorators.

```python
    async def start(self):
        """Start the service - integrate BaseService with ZMQ service."""
        logger.info(f"Starting {self.service_name}")
        
        try:
            # Set up message handlers BEFORE starting ZMQ service
            self.zmq_service.add_message_handler("heartbeat", self._handle_heartbeat)
            self.zmq_service.add_message_handler("status", self._handle_status)
            self.zmq_service.set_default_handler(self._handle_general)
            
            # Start ZMQ service FIRST
            await self.zmq_service.start()
            
            # Add background tasks to BaseService
            self.add_task(self._publishing_loop())
            
            # Call BaseService start at the end (this handles state transitions automatically)
            await super().start()
            
        except Exception as e:
            self.record_error(e, is_fatal=True)
            raise
    
    async def stop(self):
        """Stop the service - ensure proper cleanup."""
        logger.info(f"Stopping {self.service_name}")
        
        try:
            # Stop ZMQ service FIRST
            await self.zmq_service.stop()
            
            # Call BaseService stop (this handles state transitions and task cleanup)
            await super().stop()
            
        except Exception as e:
            self.record_error(e)
            raise
```

### Step 3: Implement Message Handling

```python
    async def _handle_heartbeat(self, message_data):
        """Handle heartbeat messages."""
        try:
            service = message_data.get("service", "unknown")
            self.messages_received += 1
            logger.info(f"❤️ Heartbeat from {service}")
        except Exception as e:
            self.record_error(e)
    
    async def _handle_status(self, message_data):
        """Handle status messages."""
        try:
            service = message_data.get("service", "unknown")
            state = message_data.get("state", "unknown")
            self.messages_received += 1
            logger.info(f"📊 Status from {service}: {state}")
        except Exception as e:
            self.record_error(e)
    
    async def _handle_general(self, topic: str, message_data):
        """Handle messages without specific handlers."""
        msg_type = message_data.get("type", "unknown")
        logger.info(f"📝 General message on '{topic}': {msg_type}")
        self.messages_received += 1
```

### Step 4: Implement Background Tasks

```python
    async def _publishing_loop(self):
        """Background task for publishing messages."""
        counter = 0
        
        while self.running:  # BaseService provides this property
            try:
                counter += 1
                
                # Publish heartbeat
                heartbeat = {
                    "type": "heartbeat",
                    "service": self.service_name,
                    "sequence": counter,
                    "timestamp": time.time()
                }
                await self.zmq_service.publish(heartbeat, "heartbeat")
                
                # Publish status
                status = {
                    "type": "status", 
                    "service": self.service_name,
                    "state": self.state.value,
                    "uptime": time.time() - self.start_time
                }
                await self.zmq_service.publish(status, "status")
                
                self.messages_sent += 1
                
                # Use BaseService sleep method for proper shutdown handling
                await self._sleep_if_running(2.0)
                
            except Exception as e:
                self.record_error(e)
                await self._sleep_if_running(1.0)  # Brief pause before retry
```

## Configuration System

### Simple Configuration

For clear, readable examples and simple services, use direct instantiation:

```python
from experimance_common.zmq.config import (
    PubSubServiceConfig, PublisherConfig, SubscriberConfig,
    WorkerServiceConfig, PullConfig, PushConfig,
    ControllerServiceConfig, WorkerConfig
)
from experimance_common.constants import DEFAULT_PORTS

# PubSub service configuration
pubsub_config = PubSubServiceConfig(
    name="my-pubsub-service",
    publisher=PublisherConfig(
        address="tcp://*",
        port=DEFAULT_PORTS["events"],
        default_topic="heartbeat"
    ),
    subscriber=SubscriberConfig(
        address="tcp://localhost",
        port=DEFAULT_PORTS["events"],
        topics=["heartbeat", "status"]
    )
)

# Worker service configuration
worker_config = WorkerServiceConfig(
    name="my-worker",
    pull=PullConfig(
        address="tcp://localhost",
        port=DEFAULT_PORTS["work"]
    ),
    push=PushConfig(
        address="tcp://*",
        port=DEFAULT_PORTS["results"]
    )
)

# Controller service configuration
controller_config = ControllerServiceConfig(
    name="my-controller",
    publisher=PublisherConfig(
        address="tcp://*",
        port=DEFAULT_PORTS["events"]
    ),
    workers=[
        WorkerConfig(
            name="image-worker",
            push_port=DEFAULT_PORTS["images"],
            pull_port=DEFAULT_PORTS["image_results"]
        ),
        WorkerConfig(
            name="audio-worker", 
            push_port=DEFAULT_PORTS["audio"],
            pull_port=DEFAULT_PORTS["audio_results"]
        )
    ]
)
```

### Testing Approaches (Recommended)

**For unit tests**, use mocks instead of factory functions:

```python
from unittest.mock import Mock
from experimance_common.zmq.config import PubSubServiceConfig

# Method 1: Mock the entire config
mock_config = Mock(spec=PubSubServiceConfig)
mock_config.name = "test-service"
mock_config.publisher = Mock()
mock_config.subscriber = Mock()

# Method 2: Use BaseConfig.from_overrides() with minimal data
test_config = PubSubServiceConfig.from_overrides(
    default_config={
        "name": "test",
        "publisher": {"address": "tcp://*", "port": 5555},
        "subscriber": None
    }
)

# Method 3: Pytest fixtures
@pytest.fixture
def mock_pubsub_config():
    return Mock(spec=PubSubServiceConfig)
```

**Use BaseConfig integration for production services** and mocks for testing.

### Configuration Summary

| Use Case                  | Approach                      | Example                             |
| ------------------------- | ----------------------------- | ----------------------------------- |
| **Production Services**   | `BaseConfig.from_overrides()` | Full config file + override support |
| **Examples & Prototypes** | Direct instantiation          | `PubSubServiceConfig(...)`          |
| **Unit Testing**          | Mocks                         | `Mock(spec=PubSubServiceConfig)`    |
| **Integration Testing**   | Minimal BaseConfig            | Simple override configs             |

### Production Configuration (Recommended)

For production services, use BaseConfig integration with proper config files and overrides:

```python
from experimance_common.zmq.config import (
    PubSubServiceConfig, WorkerServiceConfig, ControllerServiceConfig
)

# PubSub service (bidirectional communication)
default_config = {
    "name": "my-service",
    "publisher": {
        "address": "tcp://*",
        "port": 5555,
        "default_topic": "general"
    },
    "subscriber": {
        "address": "tcp://localhost",
        "port": 5556,
        "topics": ["heartbeat", "status", "era_events"]
    }
}

config = PubSubServiceConfig.from_overrides(
    default_config=default_config,
    config_file="service.toml"
)

# Worker service (receives work, sends results)
worker_default_config = {
    "name": "image-worker",
    "work_pull": {
        "address": "tcp://localhost",
        "port": 5564  # DEFAULT_PORTS["image_requests"]
    },
    "result_push": {
        "address": "tcp://*", 
        "port": 5565  # DEFAULT_PORTS["image_results"]
    },
    "publisher": {
        "address": "tcp://*",
        "port": 5567,
        "default_topic": "worker_status"
    },
    "subscriber": {
        "address": "tcp://localhost",
        "port": 5555,  # DEFAULT_PORTS["events"]
        "topics": ["control", "heartbeat"]
    }
}

worker_config = WorkerServiceConfig.from_overrides(
    default_config=worker_default_config,
    config_file="worker.toml"
)

# Controller service (distributes work, collects results)
controller_default_config = {
    "name": "image-controller",
    "publisher": {
        "address": "tcp://*",
        "port": 5555,  # DEFAULT_PORTS["events"]
        "default_topic": "control"
    },
    "subscriber": {
        "address": "tcp://localhost", 
        "port": 5555,  # DEFAULT_PORTS["events"]
        "topics": ["worker_status", "heartbeat"]
    },
    "workers": {
        "image": {
            "push": {
                "address": "tcp://*",
                "port": 5564  # DEFAULT_PORTS["image_requests"]
            },
            "pull": {
                "address": "tcp://localhost",
                "port": 5565  # DEFAULT_PORTS["image_results"] 
            }
        }
    }
}

controller_config = ControllerServiceConfig.from_overrides(
    default_config=controller_default_config,
    config_file="controller.toml"
)
```

### Custom Configuration

For advanced use cases, create configurations directly:

```python
from experimance_common.zmq.config import (
    PubSubServiceConfig, PublisherConfig, SubscriberConfig
)

config = PubSubServiceConfig(
    name="custom-service",
    publisher=PublisherConfig(
        address="tcp://*",
        port=5555,
        default_topic="custom"
    ),
    subscriber=SubscriberConfig(
        address="tcp://localhost", 
        port=5556,
        topics=["custom", "heartbeat"]
    )
)
```

### Configuration Mutability and State Management

The ZMQ configuration system uses a **frozen config + mutable service state** pattern for robust state management:

#### Frozen Configuration Objects

All ZMQ configuration objects (sockets, workers, services) are **immutable after creation**:

```python
config = PubSubServiceConfig.from_overrides(default_config)

# ❌ This will raise an error - configs are frozen
config.timeout = 30.0  # ValidationError: Instance is frozen

# ✅ Create a new config if changes are needed
new_config = PubSubServiceConfig.from_overrides(
    default_config=default_config,
    override_config={"timeout": 30.0}
)
```

#### Mutable Service State Pattern

When services need runtime-mutable state, copy config values to instance variables:

```python
class MyZmqService(BaseService):
    def __init__(self, config: PubSubServiceConfig):
        super().__init__(service_name=config.name)
        
        # Store immutable config
        self.config = config
        
        # Copy values that need to be mutable during runtime
        self.timeout = config.timeout          # Can modify during runtime
        self.retry_count = 0                   # Runtime state
        self.connection_attempts = 0           # Runtime counters
        self.last_heartbeat = None             # Runtime timestamps
        
        # Create ZMQ service with original config
        self.zmq_service = PubSubService(config.pusub)
    
    async def handle_connection_error(self):
        # ✅ Can modify runtime state
        self.retry_count += 1
        self.timeout *= 1.5  # Exponential backoff
        
        if self.retry_count > 5:
            # ✅ Can create new config if needed
            new_config = self.config.model_copy(
                update={"timeout": self.timeout}
            )
            # Recreate service with new config if necessary
```

#### Benefits of This Pattern

- **🔒 Configuration Integrity**: Configs can't be accidentally modified
- **🔄 Runtime Flexibility**: Services can adapt behavior as needed  
- **🧪 Testing Reliability**: Frozen configs ensure consistent test conditions
- **📝 Clear Separation**: Configuration vs runtime state is explicit
- **⚡ Performance**: Immutable configs can be safely shared between services

#### Configuration vs Runtime State Guidelines

| **Configuration**         | **Runtime State**                   |
| ------------------------- | ----------------------------------- |
| Port numbers, addresses   | Connection status, retry counts     |
| Topic subscriptions       | Message statistics, timestamps      |
| Socket options, timeouts  | Dynamic timeout adjustments         |
| Service names, log levels | Current error counts, health status |

Use this pattern to keep your services both configurable and adaptable! 🎯

## Message Schemas

### Using Schema-Based Messages

For production code, use typed message schemas:

```python
from experimance_common.schemas import EraChanged, Era, Biome

# Create typed message
era_message = EraChanged(era=Era.CURRENT, biome=Biome.RAINFOREST)

# Publish schema message
await service.publish(era_message.model_dump(), "era_events")

# Handle schema message
async def handle_era_events(self, message_data):
    message = MessageBase.from_dict(message_data)
    if isinstance(message, EraChanged):
        logger.info(f"Era changed: {message.era} in {message.biome}")
```

### Using Simple Dict Messages

For testing and simple use cases, use dict messages:

```python
# Simple heartbeat message
heartbeat = {
    "type": "heartbeat",
    "service": self.service_name,
    "timestamp": time.time(),
    "sequence": counter
}

# Publish dict message
await service.publish(heartbeat, "heartbeat")
```

## Communication Patterns

### PubSub Pattern (Many-to-Many)

Use for status updates, events, coordination:

```python
# Configuration using BaseConfig integration
default_config = {
    "name": "status-service",
    "publisher": {
        "address": "tcp://*",
        "port": 5555,  # DEFAULT_PORTS["events"] - All services use this
        "default_topic": "status"
    },
    "subscriber": {
        "address": "tcp://localhost",
        "port": 5555,  # DEFAULT_PORTS["events"]
        "topics": ["heartbeat", "status", "era_events"]
    }
}

config = PubSubServiceConfig.from_overrides(
    default_config=default_config,
    config_file="status_service.toml"
)

# Service
service = PubSubService(config)
```

### Worker Pattern (Distributed Work Processing)

Use for distributing work across multiple workers:

```python
# Worker side - BaseConfig integration
worker_default_config = {
    "name": "image-worker-1",
    "work_pull": {
        "address": "tcp://localhost",
        "port": 5564  # DEFAULT_PORTS["image_requests"]
    },
    "result_push": {
        "address": "tcp://*",
        "port": 5565  # DEFAULT_PORTS["image_results"]
    }
}

worker_config = WorkerServiceConfig.from_overrides(
    default_config=worker_default_config,
    config_file="worker.toml"
)
worker = WorkerService(worker_config)

# Set work handler
async def process_image(work_data):
    # Process the work
    return {"result": "processed", "work_id": work_data["id"]}

worker.set_work_handler(process_image)
```

### Controller Pattern (Work Coordination)

Use for coordinating multiple workers:

```python
# Controller side - BaseConfig integration
controller_default_config = {
    "name": "image-controller",
    "workers": {
        "image": {
            "push": {
                "address": "tcp://*",
                "port": 5564  # DEFAULT_PORTS["image_requests"]
            },
            "pull": {
                "address": "tcp://localhost", 
                "port": 5565  # DEFAULT_PORTS["image_results"]
            }
        }
    }
}

controller_config = ControllerServiceConfig.from_overrides(
    default_config=controller_default_config,
    config_file="controller.toml"
)
controller = ControllerService(controller_config)

# Send work to workers
work_item = {"id": "task_001", "image_path": "/path/to/image.jpg"}
await controller.send_work_to_all_workers(work_item)
```

## BaseService Integration

### Key Integration Points

1. **State Management**: Use `ServiceState` enum for lifecycle tracking
2. **Signal Handling**: BaseService handles SIGINT/SIGTERM automatically
3. **Task Management**: Use `add_task()` for background coroutines
4. **Error Handling**: Use `record_error()` for error tracking
5. **Statistics**: BaseService tracks messages sent/received

### Best Practices

```python
class MyZmqService(BaseService):
    async def start(self):
        # ❌ WRONG: Don't manually set states
        # self.state = ServiceState.STARTING
        
        try:
            # ✅ CORRECT: Configure handlers before starting ZMQ
            self.zmq_service.add_message_handler("topic", self._handler)
            
            # ✅ CORRECT: Start ZMQ service first
            await self.zmq_service.start()
            
            # ✅ CORRECT: Add background tasks to BaseService
            self.add_task(self._background_task())
            
            # ✅ CORRECT: Call parent start (handles state transitions)
            await super().start()
            
            # ✅ CORRECT: Record successful service start
            self.record_health_check("service_start", HealthStatus.HEALTHY, "Service started successfully")
            
        except Exception as e:
            # ✅ CORRECT: Record errors properly
            self.record_error(e, is_fatal=True)
            raise
    
    async def stop(self):
        # ❌ WRONG: Don't manually set states
        # self.state = ServiceState.STOPPING
        
        try:
            # ✅ CORRECT: Stop ZMQ first
            await self.zmq_service.stop()
            
            # ✅ CORRECT: Call parent stop (handles states and cleanup)
            await super().stop()
            
        except Exception as e:
            self.record_error(e)
    
    async def _background_task(self):
        while self.running:  # BaseService property
            try:
                # Do work
                self.messages_sent += 1  # Update statistics
                
                # ✅ CORRECT: Use BaseService sleep for proper shutdown
                await self._sleep_if_running(1.0)
                
            except Exception as e:
                self.record_error(e)  # Non-fatal error
```

## Common Issues and Solutions

### Issue 1: State Management Conflicts

**Problem**: `RuntimeError: Cannot call start() when service is in ServiceState.STARTING`

**Cause**: Manually setting `self.state = ServiceState.STARTING` conflicts with BaseService's automatic state management.

**Solution**: Remove all manual state assignments. BaseService handles states automatically.

```python
# ❌ WRONG
async def start(self):
    self.state = ServiceState.STARTING  # Don't do this
    await super().start()
    self.state = ServiceState.RUNNING   # Don't do this

# ✅ CORRECT  
async def start(self):
    await self.zmq_service.start()
    await super().start()               # BaseService handles states
    
    # Record successful service start
    self.record_health_check("service_start", HealthStatus.HEALTHY, "Service started successfully")
```

### Issue 2: Handler Signature Mismatches

**Problem**: `TypeError: argument type mismatch` when setting handlers

**Cause**: Handler signatures must match the expected types.

**Solution**: Use correct signatures:

```python
# ✅ CORRECT: Topic handlers take only message_data
async def _handle_heartbeat(self, message_data):
    pass

# ✅ CORRECT: Default handlers take topic and message_data  
async def _handle_default(self, topic: str, message_data):
    pass

# ✅ CORRECT: Both sync and async handlers are supported
def _sync_handler(self, message_data):  # Sync
    pass

async def _async_handler(self, message_data):  # Async
    pass
```

### Issue 3: Double Stop Calls

**Problem**: `RuntimeError: Cannot call stop() when service is in ServiceState.STOPPING`

**Cause**: BaseService handles signals automatically, but test code also calls stop().

**Solution**: Let BaseService handle cleanup automatically:

```python
# ❌ WRONG: Manual cleanup can conflict with signal handling
async def main():
    service = MyService()
    try:
        await service.start()
        await service.run()
    finally:
        await service.stop()  # Can cause double-stop

# ✅ CORRECT: Let BaseService handle signals
async def main():
    service = MyService()
    try:
        await service.start()
        await service.run()  # Handles signals and calls stop() automatically
    except KeyboardInterrupt:
        pass  # BaseService already handled cleanup
```

### Issue 4: Wrong Method Names

**Problem**: `AttributeError: 'PubSubService' object has no attribute 'set_topic_handler'`

**Cause**: Using non-existent method names.

**Solution**: Use correct method names:

```python
# ❌ WRONG: This method doesn't exist
service.set_topic_handler("topic", handler)

# ✅ CORRECT: Use add_message_handler
service.add_message_handler("topic", handler)
service.set_default_handler(default_handler)
```

### Issue 5: Wrong Publish Parameter Order

**Problem**: Type errors when publishing messages

**Cause**: Wrong parameter order in publish() calls. Topic is optional.
           (It can be set by the message type or the default topic of the publisher.)

**Solution**: Use correct parameter order:

```python
# ❌ WRONG: data first, topic second
await service.publish("topic", message_data)

# ✅ CORRECT: data first, topic second  
await service.publish(message_data, "topic")
await service.publish(message_data)  # Uses default topic
```

### Issue 6: Configuration Validation Errors

**Problem**: `ValidationError: port Input should be a valid integer`

**Cause**: Incorrect configuration structure or missing required fields.

**Solution**: Use proper BaseConfig integration with correct structure:

```python
# ❌ WRONG: Missing required configuration structure
config = {"name": "test", "publisher": None}  # Missing proper structure

# ✅ CORRECT: Complete configuration structure
# Publisher only
default_config = {
    "name": "test-service",
    "publisher": {
        "address": "tcp://*",
        "port": 5555,
        "default_topic": "general"
    },
    "subscriber": None  # Explicitly None for publisher-only
}

config = PubSubServiceConfig.from_overrides(default_config=default_config)

# Subscriber only  
default_config = {
    "name": "test-service", 
    "publisher": None,  # Explicitly None for subscriber-only
    "subscriber": {
        "address": "tcp://localhost",
        "port": 5556,
        "topics": ["heartbeat"]
    }
}

config = PubSubServiceConfig.from_overrides(default_config=default_config)
```

## Error Handling and Best Practices

### Error Handling Patterns

```python
# 1. Handler error handling
async def _handle_message(self, message_data):
    try:
        # Process message
        self.messages_received += 1
    except Exception as e:
        self.record_error(e)  # Log and count error
        logger.error(f"Error handling message: {e}")

# 2. Publishing error handling  
async def _publish_message(self, data, topic):
    try:
        await self.zmq_service.publish(data, topic)
        self.messages_sent += 1
    except Exception as e:
        self.record_error(e)
        logger.error(f"Error publishing to {topic}: {e}")

# 3. Fatal vs non-fatal errors
try:
    await self.zmq_service.start()
except Exception as e:
    self.record_error(e, is_fatal=True)  # Fatal: can't continue
    raise

try:
    await self._process_work(item)
except Exception as e:
    self.record_error(e, is_fatal=False)  # Non-fatal: continue running
```

### Best Practices

1. **Always set message handlers before starting ZMQ services**
2. **Use BaseService task management for background work**
3. **Handle errors gracefully in message handlers**
4. **Update statistics in handlers and publishers**
5. **Use `_sleep_if_running()` in loops for proper shutdown**
6. **Stop ZMQ services before calling super().stop()**

## Examples

### Simple Publisher Service

```python
from experimance_common.base_service import BaseService
from experimance_common.service_state import ServiceState
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig
from experimance_common.zmq.services import PubSubService
import time

class SimplePublisher(BaseService):
    def __init__(self, name: str = "simple-publisher"):
        super().__init__(service_name=name, service_type="publisher")
        
        # Simple direct configuration for examples
        self.zmq_config = PubSubServiceConfig(
            name=self.service_name,
            publisher=PublisherConfig(
                address="tcp://*",
                port=5555,
                default_topic="general"
            ),
            subscriber=None  # Publisher only
        )
        self.zmq_service = PubSubService(self.zmq_config)
        self.counter = 0
    
    async def start(self):
        # Start ZMQ service first
        await self.zmq_service.start()
        
        # Add publishing task to BaseService
        self.add_task(self._publish_loop())
        
        # Call BaseService start (this handles state transitions)
        await super().start()
    
    async def stop(self):
        # Stop ZMQ service first
        await self.zmq_service.stop()
        
        # Call BaseService stop (this handles state transitions and task cleanup)
        await super().stop()
    
    async def _publish_loop(self):
        while self.running:
            self.counter += 1
            
            message = {
                "type": "heartbeat",
                "service": self.service_name,
                "sequence": self.counter,
                "timestamp": time.time()
            }
            
            await self.zmq_service.publish(message, "heartbeat")
            self.messages_sent += 1
            
            await self._sleep_if_running(2.0)

# Usage
async def main():
    service = SimplePublisher()
    try:
        await service.start()
        await service.run()  # Runs until signal
    except KeyboardInterrupt:
        pass
    finally:
        await service.stop()
```

### Simple Subscriber Service

```python
class SimpleSubscriber(BaseService):
    def __init__(self, name: str = "simple-subscriber"):
        super().__init__(service_name=name, service_type="subscriber")
        
        self.zmq_config = PubSubServiceConfig(
            name=self.service_name,
            publisher=None,  # Subscriber only
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5555,
                topics=["heartbeat", "status"]
            )
        )
        self.zmq_service = PubSubService(self.zmq_config)
    
    async def start(self):
        # Set up handlers before starting
        self.zmq_service.add_message_handler("heartbeat", self._handle_heartbeat)
        self.zmq_service.add_message_handler("status", self._handle_status)
        self.zmq_service.set_default_handler(self._handle_general)
        
        await self.zmq_service.start()
        await super().start()
    
    async def stop(self):
        await self.zmq_service.stop()
        await super().stop()
    
    async def _handle_heartbeat(self, message_data):
        service = message_data.get("service", "unknown")
        sequence = message_data.get("sequence", 0)
        self.messages_received += 1
        logger.info(f"❤️ Heartbeat #{sequence} from {service}")
    
    async def _handle_status(self, message_data):
        service = message_data.get("service", "unknown")
        state = message_data.get("state", "unknown")
        self.messages_received += 1
        logger.info(f"📊 Status from {service}: {state}")
    
    async def _handle_general(self, topic: str, message_data):
        msg_type = message_data.get("type", "unknown")
        self.messages_received += 1
        logger.info(f"📝 Message on '{topic}': {msg_type}")
```

### Running Services

```python
import asyncio
import argparse

async def run_publisher():
    service = SimplePublisher()
    try:
        await service.start()
        await service.run()
    except KeyboardInterrupt:
        logger.info("Shutting down publisher")
    finally:
        await service.stop()

async def run_subscriber():
    service = SimpleSubscriber()
    try:
        await service.start()
        await service.run()
    except KeyboardInterrupt:
        logger.info("Shutting down subscriber")
    finally:
        await service.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher", action="store_true")
    parser.add_argument("--subscriber", action="store_true")
    args = parser.parse_args()
    
    if args.publisher:
        asyncio.run(run_publisher())
    elif args.subscriber:
        asyncio.run(run_subscriber())

if __name__ == "__main__":
    main()
```

## Testing Your Services

1. **Start the subscriber** in one terminal:
   ```bash
   uv run your_service.py --subscriber
   ```

2. **Start the publisher** in another terminal:
   ```bash
   uv run your_service.py --publisher
   ```

3. **Test graceful shutdown** with Ctrl+C (SIGINT)

4. **Check logs** for proper message flow and cleanup

## Validation and Testing

This architecture has been extensively tested and validated with real working examples. The key validation included:

### Infrastructure Testing

✅ **Publisher/Subscriber Integration**: Tested bidirectional communication between publisher and subscriber services with proper BaseService lifecycle management.

✅ **Signal Handling**: Validated graceful shutdown with SIGINT (Ctrl+C) and SIGTERM signals, ensuring proper cleanup of ZMQ resources.

✅ **Error Recovery**: Tested error handling in message handlers, publishing, and connection scenarios.

✅ **Configuration Validation**: Verified Pydantic schema validation for all configuration types (PubSub, Worker, Controller).

✅ **State Management**: Confirmed BaseService state transitions work correctly without manual intervention.

### Unit Testing with Mocks

For unit testing ZMQ services without actual ZMQ sockets, use the mock system in `experimance_common.zmq.mocks`:

#### Mock Services Available

- **`MockPubSubService`**: Drop-in replacement for PubSubService
- **`MockWorkerService`**: Mock for WorkerService with Push/Pull functionality  
- **`MockControllerService`**: Mock for ControllerService managing multiple workers
- **`mock_environment()`**: Context manager for clean test isolation

#### Basic Usage

```python
from experimance_common.zmq.mocks import MockPubSubService, mock_environment
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig

async def test_my_service():
    # Use simple inline config for testing
    config = PubSubServiceConfig(
        name="test_service",
        publisher=PublisherConfig(address="tcp://*", port=5555),
        subscriber=SubscriberConfig(
            address="tcp://localhost", 
            port=5556, 
            topics=["test.topic"]
        )
    )
    
    async with mock_environment():  # Clean test environment
        async with MockPubSubService(config) as service:
            # Test publishing
            await service.publish("test.topic", {"message": "hello"})
            
            # Assert messages were published
            published = service.get_published_messages()
            assert len(published) == 1
            assert published[0].topic == "test.topic"
            assert published[0].content == {"message": "hello"}

# Run with: uv run -m pytest
```

#### Message Handler Testing

```python
async def test_message_handling():
    config = PubSubServiceConfig(
        name="handler_test",
        subscriber=SubscriberConfig(
            address="tcp://localhost",
            port=5556,
            topics=["events"]
        )
    )
    
    received_messages = []
    
    async with mock_environment():
        async with MockPubSubService(config) as service:
            # Set up handler
            def message_handler(topic, message):
                received_messages.append((topic, message))
            
            service.set_message_handler(message_handler)
            
            # Simulate receiving a message (via global message bus)
            from experimance_common.zmq.mocks import mock_message_bus
            await mock_message_bus.publish("events", {"event": "test"}, "external_service")
            
            await asyncio.sleep(0.1)  # Allow processing
            
            # Assert handler was called
            assert len(received_messages) == 1
            assert received_messages[0] == ("events", {"event": "test"})
```

#### Controller and Worker Testing

```python
async def test_controller_workers():
    from experimance_common.zmq.config import ControllerServiceConfig, WorkerConfig
    
    config = ControllerServiceConfig(
        name="test_controller",
        publisher=PublisherConfig(address="tcp://*", port=5557),
        subscriber=SubscriberConfig(address="tcp://localhost", port=5558, topics=["status"]),
        workers={
            "image_worker": WorkerConfig(
                name="image_worker",
                push_config=PushConfig(address="tcp://localhost", port=5559),
                pull_config=PullConfig(address="tcp://*", port=5560),
                message_types=["image.process"]
            )
        }
    )
    
    work_received = []
    
    async with mock_environment():
        async with MockControllerService(config) as controller:
            # Set up worker handler
            def work_handler(message):
                work_received.append(message)
            
            controller.set_worker_handler("image_worker", work_handler)
            
            # Test worker communication
            await controller.push_to_worker("image_worker", {"task": "process_image"})
            await asyncio.sleep(0.1)
            
            # Assert work was received
            assert len(work_received) == 1
            assert work_received[0] == {"task": "process_image"}
            
            # Test publishing
            await controller.publish("status", {"status": "ready"})
            published = controller.published_messages
            assert len(published) == 1
```

#### Key Testing Principles

1. **Use Inline Configs**: For tests, create configs directly instead of using factory functions
2. **Clean Environment**: Always use `mock_environment()` context manager
3. **Message Assertions**: Use `get_published_messages()` and `get_received_messages()` for verification
4. **Async Handling**: Add small delays (`await asyncio.sleep(0.1)`) for message processing
5. **Frozen Config Pattern**: Configs are immutable, runtime state is tracked in service properties

#### Mock Features

- **Global Message Bus**: Cross-service communication for integration testing
- **Message History**: Complete tracking of published/received messages with timestamps
- **Async Compatible**: All handlers support both sync and async functions
- **Error Simulation**: Can inject errors for robustness testing
- **Clean Isolation**: Each test gets a fresh message bus state

### Validated Examples

The following working examples validate the architecture:

1. **utils/examples/readme_zmq_test.py**: Complete publisher/subscriber example with BaseService integration
2. **utils/examples/zmq_baseservice_test.py**: BaseService integration testing
3. **utils/examples/new_zmq_test_service_example.py**: Alternative service implementation patterns

### Test Results

```bash
# Publisher output (validated working)
2025-01-15 21:17:59,474 - INFO - Starting readme-publisher
2025-01-15 21:17:59,478 - INFO - ❤️ Published heartbeat #1
2025-01-15 21:18:01,481 - INFO - ❤️ Published heartbeat #2
2025-01-15 21:18:03,484 - INFO - ❤️ Published heartbeat #3

# Subscriber output (validated working)  
2025-01-15 21:17:59,487 - INFO - Starting readme-subscriber
2025-01-15 21:17:59,490 - INFO - ❤️ Heartbeat #1 from readme-publisher
2025-01-15 21:18:01,483 - INFO - ❤️ Heartbeat #2 from readme-publisher
2025-01-15 21:18:03,486 - INFO - ❤️ Heartbeat #3 from readme-publisher

# Graceful shutdown (validated working)
^C2025-01-15 21:18:05,123 - INFO - Stopping readme-subscriber
2025-01-15 21:18:05,124 - INFO - Stopped readme-subscriber
```

### Key Fixes Validated

1. **Configuration Factory Functions**: Fixed to handle None values for optional publisher/subscriber components
2. **Handler Registration**: Verified both sync and async handlers work correctly
3. **Method Names**: Confirmed correct usage of `add_message_handler()` and `set_default_handler()` 
4. **Publish Parameter Order**: Validated correct `publish(data, topic)` parameter order
5. **BaseService Integration**: Confirmed automatic state management without manual state setting
6. **Resource Cleanup**: Verified proper ZMQ socket cleanup on shutdown

## Summary

The Experimance ZMQ architecture provides:

- ✅ **Type-safe configuration** with Pydantic
- ✅ **Robust lifecycle management** with BaseService
- ✅ **Automatic signal handling** and graceful shutdown
- ✅ **Flexible messaging** (strings or schemas)
- ✅ **Error recovery** and logging
- ✅ **Production-ready** components

Start with the simple examples above, then extend them for your specific use cases. The architecture handles the complexity of ZMQ lifecycle management, letting you focus on your business logic.

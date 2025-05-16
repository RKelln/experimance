# Experimance Common Library

This library provides common utilities, schemas, and constants for all Experimance services. It serves as the foundation for inter-service communication and shared functionality across the Experimance distributed system.

## Features

- **ZeroMQ Communication Utilities**: Non-hanging, timeout-aware ZMQ sockets for reliable inter-service communication
- **Message Type Definitions**: Standard message types used throughout the system
- **Configuration Utilities**: Shared configuration management
- **Constants**: Default values for ports, timeouts, and other settings
- **Schemas**: Data models and validation using Pydantic

## Installation

The common library is typically installed as part of the Experimance project:

```bash
# From the project root
uv sync
```

For development:

```bash
# From the common library directory
cd libs/common
uv pip install -e .
```

## ZeroMQ Utilities Usage Guide

The `experimance_common.zmq_utils` module provides enhanced ZeroMQ socket wrappers that avoid the common pitfalls of ZMQ such as hanging receives and difficult cleanup.

### Communication Patterns

The library implements two primary ZeroMQ communication patterns:

1. **Publisher-Subscriber (PUB/SUB)**: One-to-many, fire-and-forget messaging
2. **Push-Pull (PUSH/PULL)**: Load-balanced, one-way task distribution

### Message Types

Messages are categorized by their purpose using the `MessageType` enum:

```python
from experimance_common.zmq_utils import MessageType

# Available message types
# MessageType.ERA_CHANGED
# MessageType.RENDER_REQUEST
# MessageType.IDLE_STATUS
# MessageType.IMAGE_READY
# MessageType.TRANSITION_READY
# MessageType.LOOP_READY
# MessageType.AGENT_CONTROL_EVENT
# MessageType.TRANSITION_REQUEST
# MessageType.LOOP_REQUEST
# MessageType.HEARTBEAT
```

### Port Configuration

Standard ports for services are defined in `experimance_common.constants.DEFAULT_PORTS`:

```python
from experimance_common.constants import DEFAULT_PORTS

# Access standard ports
coordinator_pub_port = DEFAULT_PORTS["coordinator_pub"]  # 5555
display_pull_port = DEFAULT_PORTS["display_pull"]        # 5560
```

### Publisher-Subscriber Pattern

#### Synchronous Usage

```python
from experimance_common.zmq_utils import ZmqPublisher, ZmqSubscriber, MessageType, ZmqTimeoutError
import time

# Publisher (usually in one service)
publisher = ZmqPublisher("tcp://*:5555", "status-updates", use_asyncio=False)
try:
    # Publish a message
    message = {
        "type": MessageType.HEARTBEAT,
        "timestamp": time.time(),
        "service": "example-service"
    }
    success = publisher.publish(message)
    if success:
        print("Message published successfully")
finally:
    publisher.close()

# Subscriber (usually in another service)
subscriber = ZmqSubscriber("tcp://localhost:5555", ["status-updates"], use_asyncio=False)
try:
    # Receive a message (with built-in timeout)
    try:
        topic, message = subscriber.receive()
        print(f"Received message on topic {topic}: {message}")
    except ZmqTimeoutError:
        print("No message received within timeout period")
finally:
    subscriber.close()
```

#### Asynchronous Usage

```python
import asyncio
from experimance_common.zmq_utils import ZmqPublisher, ZmqSubscriber, MessageType, ZmqTimeoutError
import time

async def publish_example():
    publisher = ZmqPublisher("tcp://*:5555", "status-updates")
    try:
        # Publish a message asynchronously
        message = {
            "type": MessageType.HEARTBEAT,
            "timestamp": time.time(),
            "service": "example-service"
        }
        success = await publisher.publish_async(message)
        if success:
            print("Message published successfully")
    finally:
        publisher.close()

async def subscribe_example():
    subscriber = ZmqSubscriber("tcp://localhost:5555", ["status-updates"])
    try:
        # Receive a message asynchronously (with built-in timeout)
        try:
            topic, message = await subscriber.receive_async()
            print(f"Received message on topic {topic}: {message}")
        except ZmqTimeoutError:
            print("No message received within timeout period")
    finally:
        subscriber.close()

# Run the examples
asyncio.run(publish_example())
asyncio.run(subscribe_example())
```

### Push-Pull Pattern

#### Synchronous Usage

```python
from experimance_common.zmq_utils import ZmqPushSocket, ZmqPullSocket, MessageType, ZmqTimeoutError

# Push socket (distributes tasks)
push_socket = ZmqPushSocket("tcp://*:5556", use_asyncio=False)
try:
    # Push a task
    task = {
        "type": MessageType.RENDER_REQUEST,
        "id": "task-123",
        "parameters": {"resolution": [800, 600]}
    }
    success = push_socket.push(task)
    if success:
        print("Task pushed successfully")
finally:
    push_socket.close()

# Pull socket (receives tasks)
pull_socket = ZmqPullSocket("tcp://localhost:5556", use_asyncio=False)
try:
    # Pull a task (with built-in timeout)
    try:
        task = pull_socket.pull()
        print(f"Received task: {task}")
    except ZmqTimeoutError:
        print("No task received within timeout period")
finally:
    pull_socket.close()
```

#### Asynchronous Usage

```python
import asyncio
from experimance_common.zmq_utils import ZmqPushSocket, ZmqPullSocket, MessageType, ZmqTimeoutError

async def push_example():
    push_socket = ZmqPushSocket("tcp://*:5556")
    try:
        # Push a task asynchronously
        task = {
            "type": MessageType.RENDER_REQUEST,
            "id": "task-123",
            "parameters": {"resolution": [800, 600]}
        }
        success = await push_socket.push_async(task)
        if success:
            print("Task pushed successfully")
    finally:
        push_socket.close()

async def pull_example():
    pull_socket = ZmqPullSocket("tcp://localhost:5556")
    try:
        # Pull a task asynchronously (with built-in timeout)
        try:
            task = await pull_socket.pull_async()
            print(f"Received task: {task}")
        except ZmqTimeoutError:
            print("No task received within timeout period")
    finally:
        pull_socket.close()

# Run the examples
asyncio.run(push_example())
asyncio.run(pull_example())
```

### Best Practices

1. **Always Close Sockets**: Use try/finally blocks to ensure sockets are closed properly
2. **Handle Timeouts**: Catch `ZmqTimeoutError` to handle cases where no message is received
3. **Use Asyncio When Possible**: The asyncio versions provide better performance for concurrent operations
4. **Proper Addressing**:
   - For binding (servers): Use `tcp://*:PORT`
   - For connecting (clients): Use `tcp://localhost:PORT` or the actual host IP
5. **Standard Ports**: Use the constants from `DEFAULT_PORTS` for consistency
6. **Topic Naming**: Use clear, hierarchical topic names (e.g., "service.event-type")
7. **Error Handling**: Check return values (`success`) on publish/push operations
8. **Graceful Shutdown**: Implement signal handlers to close sockets on service shutdown

### Error Handling

The ZMQ utilities include specific exceptions:

- `ZmqException`: Base exception for all ZMQ-related errors
- `ZmqTimeoutError`: Raised when a receive or pull operation times out

```python
from experimance_common.zmq_utils import ZmqTimeoutError

try:
    message = subscriber.receive()
except ZmqTimeoutError:
    # Handle timeout case
    pass
except Exception as e:
    # Handle other errors
    print(f"Error: {e}")
```

### Testing

To verify your ZMQ communication, refer to the test examples in the utils/tests directory:

```bash
# Run ZMQ utility tests
uv run pytest -v utils/tests/test_zmq_utils.py
```

## Contributing

When extending or modifying the common library:

1. Add comprehensive docstrings
2. Include type hints
3. Add tests for new functionality
4. Update this README with usage examples

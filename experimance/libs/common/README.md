# Experimance Common Library

This library provides common utilities, schemas, and constants for all Experimance services. It serves as the foundation for inter-service communication and shared functionality across the Experimance distributed system.

## Features

- **ZeroMQ Communication Utilities**: Non-hanging, timeout-aware ZMQ sockets for reliable inter-service communication
- **Service Base Classes**: Standardized service classes for common service patterns (with and without ZMQ)
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
from experimance_common.schemas import MessageType

# Available message types
# MessageType.SPACE_TIME_UPDATE
# MessageType.RENDER_REQUEST
# MessageType.PRESENCE_STATUS
# MessageType.SPACE_TIME_UPDATE
# MessageType.IMAGE_READY
# MessageType.TRANSITION_READY
# MessageType.LOOP_READY
# MessageType.AUDIENCE_PRESENT
# MessageType.SPEECH_DETECTED
# MessageType.TRANSITION_REQUEST
# MessageType.LOOP_REQUEST
```

### Port Configuration

Standard ports for services are defined in `experimance_common.constants.DEFAULT_PORTS`:

```python
from experimance_common.constants import DEFAULT_PORTS

# Access standard ports - all services use unified events channel
events_port = DEFAULT_PORTS["events"]  # 5555
depth_port = DEFAULT_PORTS["depth_pub"]  # 5556 (high-bandwidth depth data)
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

## Service Base Classes

The `experimance_common.service` module provides a hierarchy of service classes that standardize service behavior across the Experimance system. These classes handle common concerns like lifecycle management, graceful shutdown, error handling, and statistics tracking.

### Service Class Hierarchy

The service classes are organized as follows:

1. **BaseService**: Core functionality for all services (with or without ZMQ)
   - **BaseZmqService**: Adds ZeroMQ-specific functionality
     - **PubSubService**: For services that both publish and subscribe
     - **ControllerService**: For controller services (publish + listen + push + pull)
     - **WorkerService**: For worker services (subscribe + pull + push responses)

See [README_SERVICE.md](README_SERVICE.md) for detailed documentation on using these service classes.

### Example Usage: Non-ZMQ Service

```python
from experimance_common.service import BaseService
import asyncio

class MyService(BaseService):
    def __init__(self):
        super().__init__(
            service_name="my-service",
            service_type="custom-service"
        )
        
    async def start(self):
        await super().start()
        # Register a custom task
        self._register_task(self.periodic_task())
        
    async def periodic_task(self):
        while self.running:
            print("Performing work...")
            await asyncio.sleep(1)

async def main():
    service = MyService()
    await service.start()
    await service.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Usage: ZMQ Service

```python
from experimance_common.zmq.publisher import ZmqPublisherService
from experimance_common.constants import DEFAULT_PORTS
import asyncio

class MyService(ZmqPublisherService):
    def __init__(self):
        super().__init__(
            service_name="my-service",
            pub_address=f"tcp://*:{DEFAULT_PORTS['example_pub']}",
        )
    
    async def start(self):
        await super().start()
        # Register additional tasks
        self._register_task(self.my_background_task())
    
    async def my_background_task(self):
        while self.running:
            # Do something periodically
            await asyncio.sleep(5)

# Usage
async def main():
    service = MyService()
    await service.start()
    await service.run()  # Runs until terminated with Ctrl+C

if __name__ == "__main__":
    asyncio.run(main())
```

For a complete example of controller-worker communication, see `utils/examples/zmq_service_example.py`.

For more details on service classes, see [README_SERVICE.md](./README_SERVICE.md).

## Contributing

When extending or modifying the common library:

1. Add comprehensive docstrings
2. Include type hints
3. Add tests for new functionality
4. Update this README with usage examples

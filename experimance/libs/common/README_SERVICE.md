# Experimance Common Services

This document describes the base service classes provided in `experimance_common.service` for building distributed applications with ZeroMQ.

## Core Concepts

The `experimance_common.service` module provides a set of base classes designed to simplify the creation of services that communicate using ZeroMQ. These classes handle common patterns such as:

- **Asynchronous Operations**: Built on `asyncio` for non-blocking I/O.
- **Lifecycle Management**: Standard `start()`, `stop()`, and `run()` methods.
- **Graceful Shutdown**: Signal handlers for `SIGINT` and `SIGTERM`.
- **Heartbeating**: Automatic heartbeat messages for service discovery and monitoring (for publisher services).
- **Statistics Tracking**: Basic statistics like messages sent/received and uptime.
- **Configurable Logging**: Consistent logging across services.
- **Error Handling**: Base error handling and cleanup mechanisms.

## Base Service Classes

### 1. `BaseService`
The fundamental base class for all services. It provides:
- Basic lifecycle management (`start`, `stop`, `run`).
- Signal handling for graceful shutdown.
- Statistics tracking (uptime, status).
- Task management for background operations.

### 2. `ZmqService`
Inherits from `BaseService` and adds common ZMQ functionalities:
- ZMQ context management.
- Socket creation and configuration utilities.
- Service name and type.

### 3. `ZmqPublisherService`
Inherits from `ZmqService`. A base class for services that publish messages using a ZMQ PUB socket.
- Manages a PUB socket.
- Sends periodic heartbeat messages on a configurable topic.
- Provides a `publish_message()` method.

### 4. `ZmqSubscriberService`
Inherits from `ZmqService`. A base class for services that subscribe to messages using a ZMQ SUB socket.
- Manages a SUB socket.
- Connects to a publisher and subscribes to specified topics.
- Registers a message handler to process received messages.
- Runs a listener task to receive and process messages.

### 5. `ZmqPushService`
Inherits from `ZmqService`. A base class for services that send tasks using a ZMQ PUSH socket.
- Manages a PUSH socket.
- Provides a `push_task()` method to send messages.

### 6. `ZmqPullService`
Inherits from `ZmqService`. A base class for services that receive tasks or results using a ZMQ PULL socket.
- Manages a PULL socket.
- Registers a task handler to process received messages.
- Runs a puller task to receive and process messages.

## Combined Service Classes

These classes combine functionalities from the base ZMQ service classes.

### 1. `ZmqPublisherSubscriberService`
Combines `ZmqPublisherService` and `ZmqSubscriberService`.
- Suitable for services that need to both publish and subscribe to messages.
- Example: A service that broadcasts its status and listens for commands.

### 2. `ZmqControllerService`
Inherits from `ZmqPublisherSubscriberService`, `ZmqPushService`, and `ZmqPullService`.
- Designed for central coordinator or controller services.
- **Publishes** events or state updates.
- **Subscribes** to responses or data from other services.
- **Pushes** tasks to worker services.
- **Pulls** results or acknowledgments from worker services.
- Implements a `_handle_worker_response()` method (meant to be overridden by subclasses) to process messages received on the PULL socket.

## Usage

To create a new service:
1. Choose the appropriate base class (e.g., `ZmqPublisherService`, `ZmqControllerService`).
2. Inherit from the chosen class.
3. Implement the `__init__` method to configure ZMQ addresses, topics, etc.
4. Override message/task handlers (e.g., `_handle_message` for subscribers, `_handle_task` for pullers, `_handle_worker_response` for `ZmqControllerService`).
5. Implement any custom logic within the `run()` method or as separate async tasks managed by `add_task()`.
6. Ensure `super().__init__(...)` and `await super().start()` (if overriding `start`) are called.

### Example: Basic Publisher

```python
# In your service module (e.g., my_publisher_service.py)
import asyncio
from experimance_common.service import ZmqPublisherService
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq_utils import MessageType

class MyPublisher(ZmqPublisherService):
    def __init__(self):
        super().__init__(
            service_name="MyPublisher",
            pub_address=f"tcp://*:{DEFAULT_PORTS[MessageType.STATE_UPDATE]}",
            heartbeat_topic="mypub.heartbeat"
        )

    async def run_custom_logic(self):
        # Example of publishing a custom message
        message = {"type": "CUSTOM_EVENT", "data": "hello world"}
        await self.publish_message(message)
        self.log_info(f"Published custom message: {message}")

    async def run(self):
        # Add custom logic to the service's tasks
        self.add_task(self.run_custom_logic())
        # The base run() method will keep the service alive
        # and manage other tasks like heartbeating.
        # If you don't call super().run(), you need to manage the service loop.
        await super().run()

async def main():
    service = MyPublisher()
    await service.start()
    # Keep it running until shutdown (e.g., Ctrl+C)
    # The service's signal handlers will manage cleanup.

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: `ZmqControllerService` Outline

```python
# In your controller service module
import asyncio
import logging
from experimance_common.service import ZmqControllerService
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq_utils import MessageType

logger = logging.getLogger(__name__)

class MyController(ZmqControllerService):
    def __init__(self):
        super().__init__(
            service_name="MyController",
            pub_address=f"tcp://*:{DEFAULT_PORTS[MessageType.COMMAND]}",      # For publishing commands
            sub_address=f"tcp://localhost:{DEFAULT_PORTS[MessageType.STATE_UPDATE]}", # For subscribing to state updates
            push_address=f"tcp://*:{DEFAULT_PORTS[MessageType.TASK]}",        # For pushing tasks to workers
            pull_address=f"tcp://*:{DEFAULT_PORTS[MessageType.RESULT]}",      # For pulling results from workers
            topics=["worker.status", "sensor.data"], # Topics to subscribe to
            heartbeat_topic="controller.heartbeat",
            service_type="controller"
        )
        # Register the specific handler for messages from the PULL socket
        # self.register_task_handler(self._handle_worker_response) # This is done in ZmqControllerService base class

    async def _handle_message(self, topic: str, message: dict):
        """Handles messages received on the SUB socket."""
        logger.info(f"Received subscribed message on topic '{topic}': {message}")
        # Process subscribed messages (e.g., worker status, sensor data)

    async def _handle_worker_response(self, message: dict):
        """Handles messages received on the PULL socket (from workers)."""
        logger.info(f"Received worker response: {message}")
        # Process responses/results from worker services
        # Example: Update internal state, trigger new commands, etc.

    async def perform_control_action(self):
        # Example: Publish a command
        command = {"type": MessageType.COMMAND.value, "action": "START_PROCESS", "param": "X"}
        await self.publish_message(command)
        logger.info(f"Published command: {command}")

        # Example: Push a task to a worker
        task = {"type": MessageType.TASK.value, "task_id": "123", "payload": "do_something"}
        await self.push_task(task)
        logger.info(f"Pushed task: {task}")

    async def run(self):
        self.add_task(self.perform_control_action())
        await super().run() # Manages listener, puller, heartbeats, etc.

async def main():
    service = MyController()
    await service.start()
    # Service runs until shutdown signal

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

## ZMQ Configuration

- **Ports**: Defined in `experimance_common.constants.DEFAULT_PORTS`.
- **Addresses**:
  - Binding: `tcp://*:{port}`
  - Connecting: `tcp://localhost:{port}` (or specific IP if remote)
- **Message Types**: Defined in `experimance_common.zmq_utils.MessageType`.

## Error Handling and Cleanup

- Services implement `try...finally` blocks to ensure ZMQ sockets and contexts are closed.
- Signal handlers trigger the `stop()` method for graceful shutdown.
- The `stop()` method cancels all running tasks and performs cleanup.

## Statistics

Services track:
- `start_time`: Timestamp when the service started.
- `messages_sent`: Count of messages published or pushed.
- `messages_received`: Count of messages subscribed or pulled.
- `status`: Current status (e.g., "running", "stopped").
- `uptime`: Calculated from `start_time`.

These can be accessed via service properties (e.g., `service.stats`). A `display_stats` task can be enabled to periodically log these statistics.

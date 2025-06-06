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
- **State Management**: Consistent service lifecycle state handling across inheritance hierarchies.

## Service State Management

Services in Experimance follow a well-defined lifecycle with the following states:

- `INITIALIZING`: Service is in the process of initialization
- `INITIALIZED`: Service has been fully instantiated
- `STARTING`: Service is in the process of starting up
- `STARTED`: Service has completed startup but not yet running
- `RUNNING`: Service is fully operational
- `STOPPING`: Service is in the process of shutting down
- `STOPPED`: Service has been fully stopped

The state management system ensures consistent state transitions across class inheritance hierarchies:

1. **State Validation**: Each lifecycle method (`start()`, `stop()`, `run()`) validates the current state before execution
2. **Automatic Transitions**: States change from `STATE` → `STATEing` → `STATEed` during lifecycle operations  
3. **Inheritance Support**: Base classes set "in progress" states at the beginning of a method and derived classes complete the transitions
4. **Event-Based Observability**: Services expose events for state transitions to enable waiting for specific states
5. **Early State Validation**: State validation happens before any code runs in lifecycle methods

The service lifecycle methods follow this pattern:
- `start()`: INITIALIZED → STARTING → STARTED 
- `run()`: STARTED → RUNNING (remains RUNNING until stopped)
- `stop()`: any state → STOPPING → STOPPED

### Implementation Details

The state management system consists of two main components:

1. **`StateManager` class**: Responsible for managing states, transitions, and events
   - `validate_and_begin_transition()`: Validates the current state and sets the "in progress" state
   - `complete_transition()`: Sets the completed state at the end of a method
   - `wait_for_state()`: Asynchronously waits for a specific state
   - `observe_state_change()`: Context manager for observing state transitions

2. **`@lifecycle_service` decorator**: Class decorator that automatically wraps the service's lifecycle methods
   - Wraps `start()`, `stop()`, and `run()` methods at class definition time
   - Handles state validation and transition at the beginning and end of each method call
   - Preserves proper behavior across inheritance chains

### Using State Management in Custom Services

When building custom services by extending `BaseService` or its ZMQ-specific subclasses, the state management system works automatically. The service moves through the proper state transitions during startup, execution, and shutdown without requiring any additional code.

For custom methods or advanced use cases, you can access the state management system directly:

```python
# Validate the current state and set the "in progress" state
self._state_manager.validate_and_begin_transition(
    'my_method',
    {ServiceState.RUNNING},  # Valid states
    ServiceState.STOPPING    # Progress state
)

# Your method implementation here

# Complete the transition at the end
self._state_manager.complete_transition(
    'my_method',
    ServiceState.STOPPING,   # Progress state
    ServiceState.STOPPED     # Completed state
)
```

You can wait for specific states in tests or in custom logic:

```python
# Wait for a service to reach the RUNNING state
await service._state_manager.wait_for_state(ServiceState.RUNNING, timeout=5.0)

# Use context manager for observing transitions
async with service._state_manager.observe_state_change(ServiceState.STOPPED):
    # This code should cause the service to stop
    await service.stop()
```

### Debugging State Transitions

The state management system provides tools for debugging state transitions:

```python
# Get the history of all state transitions with timestamps
state_history = service._state_manager.get_state_history()
for state, timestamp in state_history:
    print(f"State: {state}, Timestamp: {timestamp}")

# Current state is always available as a property
current_state = service.state
print(f"Current state: {current_state}")

# Enable debug logging to see detailed state transition information
import logging
logging.getLogger("experimance_common.service_state").setLevel(logging.DEBUG)
```

### Extending the State Management System

For specialized services with unique state requirements, you can extend the state management system:

```python
# Register a callback for a specific state transition
def on_running():
    print("Service is now running!")
    
service.register_state_callback(ServiceState.RUNNING, on_running)

# Create a custom wrapper method with state transitions
async def my_custom_operation(self):
    # Validate current state and set in-progress state
    self._state_manager.validate_and_begin_transition(
        'my_custom_operation',
        {ServiceState.STARTED, ServiceState.RUNNING},  # Valid states
        ServiceState.CUSTOM_STATE  # Custom progress state
    )
    
    try:
        # Implement custom operation
        await some_async_operation()
    finally:
        # Always set the completed state in finally block
        self._state_manager.complete_transition(
            'my_custom_operation',
            ServiceState.CUSTOM_STATE,  # Progress state
            ServiceState.RUNNING  # Completed state
        )
```

### Best Practices for Service State Management

1. **Follow the Lifecycle Pattern**: 
   - Always call `await super().start()` in your overridden `start()` method
   - Always call `await super().stop()` in your overridden `stop()` method
   - Always call `await super().run()` in your overridden `run()` method

2. **Handle States in Base Classes**:
   - Base classes should validate current state and set the "in progress" state
   - Derived classes generally don't need to manage state themselves

3. **Use Finally Blocks for Cleanup**:
   - State transitions to error or completed states should happen in `finally` blocks
   - This ensures proper state transitions even when exceptions occur

4. **Order of Operations**:
   - In `start()`: Initialize resources first, then call `super().start()`
   - In `stop()`: Call `super().stop()` first, then clean up resources
   - In `run()`: Register tasks first, then call `super().run()`

## Base Service Classes

### 1. `BaseService`
The fundamental base class for all services. It provides:
- Basic lifecycle management (`start`, `stop`, `run`).
- Signal handling for graceful shutdown.
- Statistics tracking (uptime, status).
- Task management for background operations.
- State management across inheritance hierarchies.

### 2. `BaseZmqService`
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

### 3. `ZmqWorkerService`
Inherits from `ZmqSubscriberService`, `ZmqPullService`, and `ZmqPushService`.
- Designed for worker services that process tasks.
- **Subscribes** to broadcast notifications.
- **Pulls** tasks from a controller service.
- **Pushes** results back to the controller.
- Override `_handle_task()` to implement custom task processing logic.

## Usage

To create a new service:
1. Choose the appropriate base class (e.g., `ZmqPublisherService`, `ZmqControllerService`).
2. Inherit from the chosen class.
3. Implement the `__init__` method to configure ZMQ addresses, topics, etc.
4. Override message/task handlers (e.g., `_handle_message` for subscribers, `_handle_task` for pullers, `_handle_worker_response` for `ZmqControllerService`).
5. Implement any custom logic within the `run()` method or as separate async tasks managed by `_register_task()`.
6. Ensure `super().__init__(...)` and `await super().start()` (if overriding `start`) are called.

### Example: Basic Publisher

```python
# In your service module (e.g., my_publisher_service.py)
import asyncio
from experimance_common.zmq.publisher import ZmqPublisherService
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
        self._register_task(self.run_custom_logic())
        # The base run() method will keep the service alive
        # and manage other tasks like heartbeating.
        # If you don't call super().run(), you need to manage the service loop.
        await super().run()

async def main():
    service = MyPublisher()
    await service.start()
    # Keep it running until shutdown (e.g., Ctrl+C)
    # The service's signal handlers will manage cleanup.
    await service.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: `ZmqControllerService` Outline

```python
# In your controller service module
import asyncio
import logging
from experimance_common.zmq.controller import ZmqControllerService
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
        self._register_task(self.perform_control_action())
        await super().run() # Manages listener, puller, heartbeats, etc.

async def main():
    service = MyController()
    await service.start()
    # Service runs until shutdown signal
    await service.run()

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

## Testing Services

The state management system enables efficient testing of services. Here's an example test pattern:

```python
import asyncio
import pytest
from experimance_common import ServiceState

async def test_service_lifecycle():
    # Create the service
    service = MyService(name="test-service")
    
    # Start the service and wait for it to transition to STARTED state
    await service.start()
    assert service.state == ServiceState.STARTED
    
    # Create a task to run the service
    run_task = asyncio.create_task(service.run())
    
    # Wait for the service to transition to RUNNING state
    await service._state_manager.wait_for_state(ServiceState.RUNNING, timeout=1.0)
    assert service.state == ServiceState.RUNNING
    
    # Test service functionality here
    # ...
    
    # Stop the service and wait for it to transition to STOPPED state
    async with service._state_manager.observe_state_change(ServiceState.STOPPED):
        await service.stop()
        
    # Clean up the run task
    if not run_task.done():
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
    
    assert service.state == ServiceState.STOPPED
```


## Tips

Generally in tasks you want to do a while loop:
```python
    from experimance_common.constants import

    # in task:
    while self.running:
        # do work
        await asyncio.sleep(TICK) # Small delay to prevent CPU spinning
```

However if you have work then a delay and more work, use:
```python
    while self.running:
        # do work
        # break if not running at start or end of sleep
        if await self._sleep_if_running(5.0): break 
        # more work
```
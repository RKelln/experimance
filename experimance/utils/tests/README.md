# Experimance Testing Utilities

This directory contains utility scripts and Pytest tests for the Experimance package.

## Testing Documentation

For more detailed information about testing in the Experimance project:

- [Service Testing Best Practices](./README_SERVICE_TESTING.md) - How to use `active_service()` and other test utilities (now available from `experimance_common.test_utils`)
- [ZMQ Testing Guide](./README_ZMQ_TESTS.md) - Details about testing ZMQ services
- [Service Architecture](../../libs/common/README_SERVICE.md) - General information about the service base classes

### Test Utilities Import

The shared test utilities are now part of the common library for easy importing:

```python
from experimance_common.test_utils import (
    active_service,           # Context manager for service lifecycle
    wait_for_service_state,   # Wait for specific service states
    MockZmqPublisher,         # Mock ZMQ sockets
    MockZmqSubscriber,
    # ... other utilities
)
```

## General utilities

You can find useful utilities to build your tests in `libs/common/src/experimance_common/`:
  1. `test_utils.py`: utilities for integration testing or common to all service testing
  2. `zmq/zmq_utils.py`: ZMQ related test utilities
  3. `image_utils.py`: image loading, saving and manipulation utilities
  4. `schemas.py`: Pydantic schemas used throughout the project
  5. `constants.py`: Don't hardcode strings, get from or add to this constants file
  6. `config.py`: How data is loaded into services. See 

Also look under the `service/NAME/tests/` directory for service specific tests and service specific mocks in `mocks.py`.

Please use the `schemas.py`, `constants.py` and `config.py` in the common library as well instead of hard-coding values 
or creating new configuration.


## Example 

The `utils/examples` directory contains example implementations that demonstrate proper usage patterns of the common library components, including the service classes.

- **`zmq_service_example.py`**: Demonstrates how to use the base service classes (`ZmqPublisherService`, `ZmqSubscriberService`, `ZmqPushService`, `ZmqPullService`, `ZmqControllerService`).
  ```bash
  # Run as controller
  uv run -m utils.examples.zmq_service_example --controller

  # In another terminal, run as worker
  uv run -m utils.examples.zmq_service_example --worker
  ```
  This example showcases:
  - Using the ZMQ service base classes for quick implementation.
  - Handling message and task processing with registered handlers.
  - Standard service lifecycle management (`start`, `stop`, `run`).
  - Proper error handling and recovery.
  - Implementing controller and worker patterns using the service classes.

- **Basic Service Example (`basic_service_example.py`)**: A non-ZMQ service implementation.
  ```bash
  uv run -m utils.examples.basic_service_example
  ```
  This example demonstrates:
  - Using the `BaseService` class for services that don't need ZMQ.
  - Implementing simple periodic tasks.
  - Standard service lifecycle management.
  - Error handling and recovery.

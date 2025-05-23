# Experimance Testing Utilities

This directory contains utility scripts and Pytest tests for the Experimance package.

## Available Test Scripts & Modules

### 1. `simple_test.py`
A minimal script that checks basic imports without requiring any extras. Useful for quick environment checks.

```bash
# Run from the experimance root directory
uv run python utils/tests/simple_test.py
```

This test:
- Verifies Python version and environment.
- Checks if `experimance` and `experimance_common` can be imported.
- Lists installed `experimance` packages.

### 2. `check_env.py`
Checks the Python environment and system configurations in more detail.

```bash
# Run from the experimance root directory
uv run python utils/tests/check_env.py
```

This test:
- Verifies Python version compatibility.
- Checks for required system libraries.
- Displays environment variables and paths.

### 3. `test_<component>.py`
Pytest module for testing the service base classes in `experimance_common.service`.

```bash
# Run from the experimance root directory
uv run -m pytest -v

# for debug logging
uv run -m pytest -v --log-cli-level=DEBUG -s 
```

For more detailed information about the ZMQ service base classes and their testing, see [README_ZMQ_TESTS.md](./README_ZMQ_TESTS.md) and `libs/common/README_SERVICE.md`.


## Example Code

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

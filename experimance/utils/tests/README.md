# Experimance Testing Utilities

This directory contains utility scripts for testing and troubleshooting the Experimance package installation.

## Available Test Scripts

### 1. `simple_test.py`
A minimal script that checks basic imports without requiring any extras.

```bash
# Run from the experimance root directory
uv run python utils/tests/simple_test.py
```

This test:
- Verifies Python version and environment
- Checks if experimance and experimance_common can be imported
- Lists installed experimance packages


### 2. `check_env.py`
Checks the Python environment and system configurations.

```bash
# Run from the experimance root directory
uv run python utils/tests/check_env.py
```

This test:
- Verifies Python version compatibility
- Checks for required system libraries
- Displays environment variables and paths


### 3. `test_zmq_utils.py`
Tests the ZMQ communication utilities in experimance_common.

```bash
# Run from the experimance root directory
uv run pytest -v utils/tests/test_zmq_utils.py
```

This test:
- Verifies the Publisher-Subscriber pattern (synchronous and asynchronous)
- Verifies the Push-Pull pattern (synchronous and asynchronous)
- Checks timeout handling and proper socket cleanup

For more detailed information about the ZMQ tests, see [README_ZMQ_TESTS.md](./README_ZMQ_TESTS.md).

## Example Code

The `utils/examples` directory contains example implementations that demonstrate proper usage patterns:

- **`zmq_example_service.py`**: Demonstrates ZMQ communication patterns with a controller-worker architecture
  ```bash
  # Run as controller
  uv run -m utils.examples.zmq_example_service --controller --name controller-1
  
  # In another terminal, run as worker
  uv run -m utils.examples.zmq_example_service --name worker-1
  ```

  This example demonstrates:
  - Proper ZeroMQ socket initialization and cleanup
  - Using both PUB/SUB and PUSH/PULL patterns
  - Handling timeouts and errors gracefully
  - Proper asyncio task management
  - Graceful shutdown with signal handling
  - Statistics tracking and reporting

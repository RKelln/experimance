# ZeroMQ Utilities, Services, and Tests

This document provides an overview of testing for ZeroMQ functionalities within the Experimance project, covering both low-level utilities and the higher-level service base classes.

> **Related Documentation**:
> - For comprehensive service testing best practices, see [README_SERVICE_TESTING.md](README_SERVICE_TESTING.md)
> - For general service implementation guidance, see [README_SERVICE.md](../../libs/common/README_SERVICE.md)

## Testing ZMQ Components

There are multiple test files focusing on different aspects of ZMQ communication:

- **`test_zmq_service.py`**: Tests the high-level ZMQ service base classes found in `experimance_common.service`. This is the primary test suite for ensuring reliable service behavior.
  - Covers `ZmqPublisherService`, `ZmqSubscriberService`, `ZmqPushService`, `ZmqPullService`, and the combined services like `ZmqPublisherSubscriberService` and `ZmqControllerService`.
  - Verifies message/task publishing and reception, handler invocation, lifecycle management, and error handling within these service abstractions.

- **`test_zmq_utils.py`**: Focuses on basic, lower-level ZMQ utilities and wrappers in `experimance_common.zmq_utils`.
  - Tests fundamental PUB/SUB and PUSH/PULL patterns using the utility functions directly.
  - Ensures correct socket setup, message serialization, and basic communication.

### Running the Tests

Run the tests using `uv` from the project root:

```bash
# Run all tests
uv run -m pytest -v

# Run tests for the ZMQ service base classes
uv run -m pytest -v utils/tests/test_zmq_service.py

# Run tests for low-level ZMQ utilities
uv run -m pytest -v utils/tests/test_zmq_utils.py

# Run all tests in the directory
uv run -m pytest -v utils/tests/

# Run tests with coverage for the common library
uv run -m pytest --cov=experimance_common utils/tests/
```

## Key Testing Considerations for ZMQ Services (`test_zmq_service.py`)

- **Asynchronous Nature**: Tests use `pytest-asyncio` to handle `async/await` code in the services.
- **Lifecycle**: Each service's `start()`, `stop()`, and `run()` methods are tested to ensure proper initialization, task execution, and cleanup.
- **Message/Task Flow**:
    - For publishers/pushers: Verify that messages/tasks are sent correctly.
    - For subscribers/pullers: Verify that messages/tasks are received and that the appropriate handlers are called with the correct data.
- **Error Handling**: Services should gracefully handle common ZMQ errors (e.g., connection issues, timeouts) although comprehensive error simulation is complex.
- **Resource Management**: Ensure ZMQ sockets and contexts are properly closed, and asyncio tasks are cancelled on service stop.
- **Configuration**: Test with various valid and potentially invalid configurations (e.g., different port numbers, topics).

## Original Issues and Solutions (Primarily for Low-Level ZMQ Utility Tests)

Many of these points were critical when developing robust low-level ZMQ tests and have informed the design of the higher-level service classes:

1.  **Hanging Tests**: Addressed by:
    *   Setting timeouts for socket operations (e.g., `socket.RCVTIMEO`, `socket.SNDTIMEO`).
    *   Using `asyncio.wait_for()` for asynchronous operations in tests.
    *   Ensuring robust test fixtures (`setup_method`, `teardown_method` or pytest fixtures) for guaranteed cleanup.
    *   Properly handling connection failures and not blocking indefinitely.

2.  **Improved Test Structure**:
    *   Pytest fixtures for managing ZMQ contexts, sockets, and service instances.
    *   Clear separation of test cases for different functionalities.

3.  **Task Management in Services & Tests**:
    *   Services manage their own asyncio tasks for operations like listening or pulling.
    *   Tests ensure these tasks are started and stopped correctly with the service lifecycle.
    *   Use of `asyncio.Event` or similar synchronization primitives for coordinating asynchronous test steps.

## Dependencies

These tests require:
- `pyzmq` (installed as part of `experimance_common` dependencies).
- `pytest` and `pytest-asyncio` (install with `uv sync --dev`).

## Troubleshooting

If you encounter issues:
1.  **Module Imports**: Ensure `uv sync` has been run and packages are installed correctly. Run `pytest` from the project root.
2.  **Port Conflicts**: Ensure no other processes are using the test ports. Test ports are typically defined within the test files or fixtures.
3.  **Permissions**: Verify necessary permissions for socket operations.
4.  **Debugging**: Use `pytest --log-cli-level=DEBUG -s` to see detailed log output and print statements. The `-s` flag captures stdout.
5.  **Timing Issues**: In distributed systems, timing can sometimes cause flaky tests. If suspected, try adding small `await asyncio.sleep()` calls at critical points in the test logic, but this should be a last resort. The service classes are designed to be robust against typical timing variations.
6.  **Hostname Resolution**: Ensure `localhost` and general network configuration are correct.
7.  **`pytest.ini` Configuration**: The `asyncio_default_fixture_loop_scope = function` setting in `utils/tests/pytest.ini` is important for `pytest-asyncio` to behave correctly with function-scoped fixtures.

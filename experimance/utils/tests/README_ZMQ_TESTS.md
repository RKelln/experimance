# ZeroMQ Utilities and Tests

This directory contains tests for the ZeroMQ utilities used throughout the Experimance project.

## Testing the ZMQ Utilities

There are multiple test files:
- `test_zmq_utils.py`: Basic tests for ZMQ utilities
- `test_connection_retry.py`: Tests for the connection retry mechanism with low-level ZMQ sockets
- `test_zmq_with_retry.py`: Tests for the connection retry mechanism with high-level ZMQ utility classes

### Running the Tests

Run the tests using `uv` from the project root:

```bash
# Run specific test files
uv run pytest -v utils/tests/test_zmq_utils.py
uv run pytest -v utils/tests/test_connection_retry.py
uv run pytest -v utils/tests/test_zmq_with_retry.py

# Run all connection retry tests
uv run pytest -v utils/tests/test_connection_retry.py utils/tests/test_zmq_with_retry.py

# Run tests with coverage
uv run pytest --cov=experimance_common utils/tests/test_zmq_utils.py
```

## Connection Retry Tests

The connection retry tests verify that our ZMQ components can robustly handle connection issues, delayed service startup, and network failures. The tests:

1. **Simulated Failures**: Explicitly simulate connection failures to test retry mechanism
2. **Delayed Services**: Test late-binding of publishers and push sockets
3. **Timeout Handling**: Verify proper timeout behavior and error propagation

There are two levels of connection retry tests:
- `test_connection_retry.py`: Tests the retry mechanism using low-level ZMQ sockets
- `test_zmq_with_retry.py`: Tests the retry mechanism with the higher-level ZMQ wrapper classes

## Known Issues and Solutions

The original ZMQ tests had several issues:

1. **Hanging Tests**: The original tests could hang indefinitely because:
   - Receive operations had no timeout
   - Cleanup wasn't guaranteed in case of test failures
   - Connection failures weren't properly handled

2. **Fixed Approach**:
   - Added proper test fixtures for setup and teardown
   - Set timeouts for all socket operations
   - Used `asyncio.wait_for()` for async operations
   - Added socket polling to avoid blocking
   - Improved error handling and recovery
   - Added explicit connection failure simulation

3. **Task Management**:
   - Created separate tasks for publisher/subscriber operations
   - Implemented proper task cancellation and cleanup
   - Used events for coordination between tasks
   - Added better logging for diagnostics

## Dependencies

These tests require:
- PyZMQ (installed as part of the project dependencies)
- Pytest and pytest-asyncio (install with `uv sync --dev`)

## Troubleshooting

If you encounter issues with importing modules, check that:
1. The library packages are properly installed with `uv sync`
2. You're running pytest from the project root
3. All required dependencies are installed

For ZeroMQ-specific issues:
1. Check that no other process is using the test ports (test ports are defined in each test file)
2. Ensure you have the necessary permissions for socket operations
3. Look at the detailed log output with `pytest --log-cli-level=DEBUG`
4. Try increasing the sleep times between operations if timing issues occur
5. Make sure your machine's hostname resolution is working correctly

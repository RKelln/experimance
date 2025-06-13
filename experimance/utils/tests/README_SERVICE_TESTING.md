# Service Testing Best Practices

This document outlines best practices for testing services in the Experimance project, particularly focusing on service lifecycle management during tests.

> **Related Documentation**:
> - For testing ZeroMQ-specific components, see [README_ZMQ_TESTS.md](README_ZMQ_TESTS.md)
> - For general service implementation guidance, see [README_SERVICE.md](../libs/common/README_SERVICE.md)

**Note**: The test utilities are now part of the `experimance_common` package and can be imported as:
```python
from experimance_common.test_utils import active_service, wait_for_service_state
```

## Using the `active_service()` Context Manager

The `active_service()` context manager is now available from the common library and is the recommended way to test services. It handles:

- Service startup
- Task creation
- Waiting for the service to reach the proper state
- Proper shutdown and cleanup, even if tests fail

### Basic Example

```python
import pytest
from experimance_common.test_utils import active_service

@pytest.mark.asyncio
async def test_my_service():
    # Create your service
    service = MyService(config=test_config)
    
    # Use active_service() to handle lifecycle
    async with active_service(service) as active:
        # The service is now running and ready for testing
        
        # Test service functionality
        result = await active.some_method()
        assert result is True
        
        # Send test messages
        await active.handle_message({"key": "value"})
        
        # Verify service state
        assert active.some_property == expected_value
    
    # No need to manually stop - active_service handles cleanup
    # The service is now stopped
```

### Advanced Usage

The `active_service()` context manager accepts several optional parameters:

```python
@pytest.mark.asyncio
async def test_advanced_service():
    service = MyService(config=test_config)
    
    # Custom setup function (runs before starting the service)
    def setup_service(svc):
        svc.custom_property = "test_value"
        # Return value is ignored
    
    # Specify custom settings
    async with active_service(
        service,
        run_task_name="my-test-task",  # Custom name for the run task (for debugging)
        target_state=ServiceState.RUNNING,  # State to wait for before yielding
        setup_func=setup_service  # Optional setup function
    ) as active:
        # Service is now running with custom setup applied
        assert active.custom_property == "test_value"
        
        # Test service functionality...
```

## Common Testing Patterns

### Test Initialization

```python
@pytest.mark.asyncio
async def test_service_initialization(test_config):
    """Test that the service initializes correctly."""
    service = MyService(config=test_config)
    
    async with active_service(service) as active:
        # Verify initialization was successful
        assert active.component is not None
        assert active.other_property == expected_value
```

### Test Message Handling

```python
@pytest.mark.asyncio
async def test_message_handling(test_config):
    """Test service message handling."""
    service = MyService(config=test_config)
    
    # Mock component before starting
    with patch('module.Component') as mock_component:
        async with active_service(service) as active:
            # Send test message
            test_message = {"key": "value", "data": [1, 2, 3]}
            await active.handle_message(test_message)
            
            # Verify handler was called correctly
            mock_component.process.assert_called_once_with(test_message["data"])
```

### Test Error Handling

```python
@pytest.mark.asyncio
async def test_error_handling(test_config):
    """Test service error handling."""
    service = MyService(config=test_config)
    
    # Set up to trigger an error
    with patch('module.Component.process', side_effect=ValueError("Test error")):
        async with active_service(service) as active:
            # Trigger error
            await active.handle_message({"key": "value"})
            
            # Verify error was recorded but service remained running
            assert active.status == ServiceStatus.DEGRADED
            assert active.running is True
            
            # Verify error was recorded
            assert len(active.errors) == 1
            assert "Test error" in str(active.errors[0])
```

## Avoiding Deadlocks with Self-Stopping Services

### The Problem: Deadlocks in `run()` Method Testing

When testing services that may call `stop()` or `request_stop()` from within their own tasks, you can encounter deadlocks if you directly await the `run()` method. This happens because:

1. Your test calls `await service.run()` which blocks waiting for the service to complete
2. The service's task calls `service.stop()` which tries to stop the service  
3. But `run()` can't complete because it's waiting for its tasks to finish
4. The tasks can't finish cleanly because `run()` is blocked

### Problematic Pattern (DO NOT USE)

```python
# ❌ DEADLOCK RISK - Don't do this for self-stopping services
@pytest.mark.asyncio
async def test_self_stopping_service():
    service = SelfStoppingService()
    await service.start()
    
    # This will deadlock if the service calls stop() from within a task
    await service.run()  # BLOCKS FOREVER
    
    assert service.state == ServiceState.STOPPED
```

### Correct Pattern: Background Task Approach

```python
# ✅ SAFE - Use this pattern for self-stopping services
@pytest.mark.asyncio
async def test_self_stopping_service():
    service = SelfStoppingService()
    await service.start()
    
    # Start service running in background task
    run_task = asyncio.create_task(service.run())
    
    try:
        # Wait for service to reach RUNNING state
        await service.wait_for_state(ServiceState.RUNNING, timeout=2.0)
        
        # Wait for the service to stop itself
        await service.wait_for_state(ServiceState.STOPPED, timeout=5.0)
        
        # Verify test expectations
        assert service.iterations >= service.expected_iterations
        
        # Wait for run task to complete cleanly
        await run_task
        
    finally:
        # Cleanup in case of test failure
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
        
        # Only stop if not already stopped
        if service.state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
            await service.stop()
```

### When to Use Background Task Pattern

Use the background task pattern when testing services that:

- Call `request_stop()` or `stop()` from within their own tasks
- Have tasks that may complete and cause the service to shut down automatically
- Need to be tested for automatic error-triggered shutdowns
- Implement timeout-based shutdown logic

### When `active_service()` Context Manager Works

The `active_service()` context manager works well for most testing scenarios, but may not be suitable for services that need to stop themselves during the test. If you need to test self-stopping behavior, use the background task pattern instead.

## Benefits of Using `active_service()`

1. **Consistent Lifecycle Management**: Standardizes how services are started, managed, and stopped in tests.

2. **Proper Error Handling**: Ensures services are always stopped, even if tests fail.

3. **State Verification**: Waits for the service to reach the desired state before testing.

4. **Resource Cleanup**: Properly handles task cancellation and service shutdown.

5. **Simplified Test Code**: Eliminates boilerplate start/stop/cleanup code in each test.

## Common Issues to Avoid

1. **Manual Service Lifecycle Management**: Don't manually call `start()`, `run()`, and `stop()` in tests.

2. **Direct Access to Internal State**: Use public properties and methods instead of accessing internal attributes like `_running`.

3. **String State Comparisons**: Use the proper enum values (e.g., `ServiceState.STOPPED`) rather than strings.

4. **Blocking Assertions**: Be careful with synchronous assertions that could block the service's asyncio loop.

5. **Missing Error Validation**: Use `wait_for_service_status()` when testing error conditions.

## Debugging Service Tests

If a test is hanging or failing in unexpected ways:

1. Use `debug_service_tasks(service)` to print information about the service's tasks.

2. Use `wait_for_service_state_and_status()` with appropriate timeouts.

3. Check for proper cleanup of resources in custom test fixtures.

4. Run tests with increased logging: `pytest --log-cli-level=DEBUG -s`

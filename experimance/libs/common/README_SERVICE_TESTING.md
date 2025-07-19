# Service Testing Best Practices

This document outlines comprehensive best practices for testing services in the Experimance project, covering modern configuration patterns, ZMQ testing, lifecycle management, and common testing utilities.

> **Related Documentation**:
> - For general service implementation guidance, see [README_SERVICE.md](README_SERVICE.md)
> - For ZMQ communication patterns and testing guidance, see [README_ZMQ.md](README_ZMQ.md)


## Quick Start: Essential Test Template

```python
# tests/test_my_service.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from experimance_common.service_state import ServiceState
from experimance_common.test_utils import active_service, wait_for_service_state
from my_service.my_service import MyService
from my_service.config import MyServiceConfig

class TestMyService:
    """Test cases for MyService using modern patterns."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a test configuration using modern config patterns."""
        return MyServiceConfig.from_overrides(
            default_config={
                "service_name": "test-my-service",
                "work_interval": 0.1,  # Fast for tests
                "debug_mode": True
            }
        )
    
    @pytest.fixture
    async def service(self, mock_config):
        """Create a service instance for testing with proper lifecycle."""
        with patch('my_service.external_dependency.Component') as mock_component:
            service = MyService(config=mock_config)
            yield service
            # Cleanup handled by active_service context manager
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test that the service initializes correctly."""
        assert service.service_name == "test-my-service"
        assert service.state == ServiceState.INITIALIZED
        assert hasattr(service, 'config')
    
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, service):
        """Test the complete service start/stop lifecycle."""
        async with active_service(service) as active:
            assert active.state == ServiceState.STARTED
            # Test service functionality here
        
        # Service is automatically stopped and cleaned up
        assert service.state == ServiceState.STOPPED
```

## Modern Configuration Testing Patterns

**🎯 TL;DR**: Use `BaseServiceConfig.from_overrides()` for test configurations, not temporary config files or manual instantiation.

### Configuration Loading for Service Tests

When testing services, use the centralized configuration loading pattern instead of temporary configuration files:

```python
from experimance_core.config import CoreServiceConfig

@pytest.mark.asyncio
async def test_service_with_config():
    # ✅ Recommended: Use configuration objects with overrides
    override_config = {
        "depth_processing": {
            "frame_delay_after_hand": 10,
            "change_threshold": 0.5
        }
    }
    config = CoreServiceConfig.from_overrides(override_config=override_config)
    service = ExperimanceCoreService(config=config)
    
    async with active_service(service) as active:
        # Test with custom configuration values
        assert active.config.depth_processing.frame_delay_after_hand == 10
```

**Key Configuration Testing Principles:**
- Use `ServiceConfig.from_overrides(override_config=...)` to customize test configurations
- Pass configuration objects to services, not configuration file paths
- Override only the specific values needed for your test
- Avoid creating temporary configuration files in tests

**Benefits:**
- More reliable and faster tests (no file I/O)
- Clear test intentions (configuration values visible in test code)
- Better error handling and validation
- Consistent with the project's configuration architecture

### Modern Configuration Override Patterns

#### Basic Configuration Override
```python
@pytest.fixture
def test_config():
    """Create test configuration with common overrides."""
    return MyServiceConfig.from_overrides(
        default_config={
            "service_name": "test-my-service",
            "work_interval": 0.1,  # Fast for tests
            "debug_mode": True,
            "log_level": "DEBUG"
        }
    )
```

#### Nested Configuration Testing
```python
@pytest.fixture
def advanced_config():
    """Test configuration with nested structures."""
    return ImageServerConfig.from_overrides(
        default_config={
            "service_name": "test-image-server",
            "cache_dir": "/tmp/test_images",
            "generator": {
                "default_strategy": "mock",
                "timeout": 10
            },
            "mock": {
                "use_existing_images": False,
                "image_size": [512, 512]
            },
            "zmq": {
                "name": "test-image-server",
                "publisher": {
                    "port": 5555,
                    "default_topic": "IMAGE_READY"
                }
            }
        }
    )
```

#### Testing Configuration Validation
```python
def test_config_validation():
    """Test that configuration validation works correctly."""
    # Test valid configuration
    valid_config = MyServiceConfig.from_overrides(
        default_config={"work_interval": 1.0}
    )
    assert valid_config.work_interval == 1.0
    
    # Test invalid configuration
    with pytest.raises(ValidationError):
        MyServiceConfig.from_overrides(
            default_config={"work_interval": -1.0}  # Should fail validation
        )
```

### Configuration Testing with ZMQ Services

For services using ZMQ composition patterns:

```python
@pytest.fixture
def zmq_service_config():
    """Configuration for ZMQ-enabled service testing."""
    return MyZmqServiceConfig.from_overrides(
        default_config={
            "service_name": "test-zmq-service",
            "zmq": {
                "name": "test-service",
                "publisher": {
                    "address": "tcp://*",
                    "port": 5555,
                    "default_topic": "TEST_TOPIC"
                },
                "subscriber": {
                    "address": "tcp://localhost",
                    "port": 5556,
                    "topics": ["INPUT_TOPIC"]
                }
            }
        }
    )
```

## Mock Utilities and Test Fixtures

**🎯 Key Insight**: Create reusable mock factories to avoid repetitive mock setup across tests.

### Creating Comprehensive Mock Files

Based on successful patterns from recent service refactoring, create a dedicated `tests/mocks.py` file:

```python
# tests/mocks.py
"""
Reusable mock utilities for service testing.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Optional, Dict, Any

from my_service.config import MyServiceConfig
from my_service.my_service import MyService

def create_mock_service_config(
    service_name: str = "test-service",
    cache_dir: Optional[Path] = None,
    **overrides
) -> MyServiceConfig:
    """
    Create a mock service configuration for testing.
    
    Args:
        service_name: Name of the service instance
        cache_dir: Directory for cache (defaults to temp)
        **overrides: Additional configuration overrides
        
    Returns:
        MyServiceConfig instance suitable for testing
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp())
        
    default_config = {
        "service_name": service_name,
        "cache_dir": str(cache_dir),
        "work_interval": 0.1,  # Fast for tests
        "debug_mode": True,
        **overrides
    }
    
    return MyServiceConfig.from_overrides(default_config=default_config)

def create_mock_zmq_service() -> Mock:
    """
    Create a mock ZMQ service for testing.
    
    Returns:
        Mock ZMQ service with common methods mocked
    """
    mock_service = Mock()
    mock_service.start = AsyncMock()
    mock_service.stop = AsyncMock()
    mock_service.publish = AsyncMock()
    mock_service.send_response = AsyncMock()
    mock_service.set_work_handler = Mock()
    mock_service.add_message_handler = Mock()
    return mock_service

def create_mock_service(
    config: Optional[MyServiceConfig] = None,
    mock_zmq: bool = True,
    mock_external_deps: bool = True
) -> MyService:
    """
    Create a mock service instance for testing.
    
    Args:
        config: Optional configuration (will create default if None)
        mock_zmq: Whether to mock ZMQ service
        mock_external_deps: Whether to mock external dependencies
        
    Returns:
        MyService with mocked dependencies
    """
    if config is None:
        config = create_mock_service_config()
        
    service = MyService(config)
    
    if mock_zmq and hasattr(service, 'zmq_service'):
        service.zmq_service = create_mock_zmq_service()
        
    if mock_external_deps:
        # Mock any external dependencies
        service.external_component = Mock()
        
    return service

# Sample test messages for consistent testing
SAMPLE_TEST_MESSAGE = {
    "type": "TEST_MESSAGE",
    "request_id": "test_request_001",
    "data": {"key": "value"}
}

SAMPLE_ERROR_MESSAGE = {
    "invalid": "data",
    # Missing required fields
}

class MockServiceTestCase:
    """Base test case class with common setup and utilities."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.test_cache_dir = Path(tempfile.mkdtemp())
        self.test_cache_dir.mkdir(exist_ok=True)
        
    def teardown_method(self):
        """Cleanup method called after each test."""
        if self.test_cache_dir.exists():
            import shutil
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)
            
    def create_test_config(self, **overrides) -> MyServiceConfig:
        """Create a test configuration with optional overrides."""
        default_overrides = {
            "cache_dir": str(self.test_cache_dir),
            **overrides
        }
        return create_mock_service_config(**default_overrides)
```

### Using Mock Factories in Tests

```python
# tests/test_my_service.py
import pytest
from tests.mocks import (
    create_mock_service_config,
    create_mock_service,
    SAMPLE_TEST_MESSAGE,
    MockServiceTestCase
)

class TestMyService(MockServiceTestCase):
    """Test cases using the mock factory pattern."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return self.create_test_config(
            service_name="test-my-service",
            strategy="mock"
        )

    @pytest.fixture
    def mock_service(self, mock_config):
        """Create a mock service instance."""
        return create_mock_service(
            config=mock_config,
            mock_zmq=True,
            mock_external_deps=True
        )

    @pytest.mark.asyncio
    async def test_with_mock_utilities(self, mock_service):
        """Test using the mock utilities."""
        async with active_service(mock_service) as active:
            # Use sample messages for consistent testing
            await active.handle_message(SAMPLE_TEST_MESSAGE)
            
            # Verify mocked ZMQ calls
            active.zmq_service.publish.assert_called()
```

## Comprehensive Testing Patterns

### Essential Test Structure

**Use the service state system for reliable testing:**

```python
# tests/test_my_service.py
import pytest
import asyncio
from experimance_common.service_state import ServiceState
from experimance_common.test_utils import active_service, wait_for_service_state
from my_service.my_service import MyService

class TestMyService:
    """Comprehensive test coverage for MyService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test that the service initializes correctly."""
        assert service.service_name == "test-my-service"
        assert service.state == ServiceState.INITIALIZED
        assert hasattr(service, 'config')
        assert hasattr(service, 'zmq_service')  # If using ZMQ

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, service):
        """Test complete service start/stop lifecycle."""
        # Service starts in INITIALIZED state
        assert service.state == ServiceState.INITIALIZED
        
        async with active_service(service) as active:
            # Service should be STARTED after start()
            assert active.state == ServiceState.STARTED
            
            # Test service functionality
            await active.do_work()
            
        # Service should be STOPPED after context exit
        assert service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_message_handling(self, service):
        """Test service message handling."""
        async with active_service(service) as active:
            # Test valid message
            test_message = {"type": "TEST", "data": "value"}
            await active.handle_message(test_message)
            
            # Verify message was processed
            assert active.last_message == test_message

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test service error handling."""
        async with active_service(service) as active:
            # Mock an error condition
            with patch.object(active, 'process_data', side_effect=ValueError("Test error")):
                await active.handle_message({"type": "TEST"})
                
                # Verify error was recorded
                assert len(active.errors) > 0
                assert "Test error" in str(active.errors[0])
```

### Testing ZMQ Services with Composition Architecture

**For services using the composition-based ZMQ pattern:**

```python
class TestZmqService:
    """Test ZMQ service functionality using composition pattern."""

    @pytest.mark.asyncio
    async def test_zmq_service_initialization(self, zmq_service):
        """Test ZMQ service component initialization."""
        assert zmq_service.zmq_service is not None
        assert hasattr(zmq_service.zmq_service, 'publisher')
        assert hasattr(zmq_service.zmq_service, 'subscriber')

    @pytest.mark.asyncio  
    async def test_message_publishing(self, zmq_service):
        """Test ZMQ message publishing."""
        async with active_service(zmq_service) as active:
            # Mock the ZMQ service
            active.zmq_service = create_mock_zmq_service()
            
            # Publish a test message
            await active.publish_message("TEST_TOPIC", {"data": "test"})
            
            # Verify publish was called
            active.zmq_service.publish.assert_called_once_with(
                "TEST_TOPIC", 
                {"data": "test"}
            )

    @pytest.mark.asyncio
    async def test_handler_registration(self, zmq_service):
        """Test that ZMQ handlers are properly registered."""
        async with active_service(zmq_service) as active:
            # Verify handlers were set up
            active.zmq_service.set_work_handler.assert_called()
            active.zmq_service.add_message_handler.assert_called()

    @pytest.mark.asyncio
    async def test_zmq_message_handling(self, zmq_service):
        """Test handling of ZMQ messages."""
        async with active_service(zmq_service) as active:
            # Create test message
            zmq_message = {
                "type": "RENDER_REQUEST",
                "request_id": "test_001",
                "prompt": "test prompt"
            }
            
            # Handle the message
            await active._handle_zmq_message(zmq_message)
            
            # Verify processing occurred
            assert active.last_processed_request == "test_001"
```

### Testing Service Error Scenarios

```python
class TestServiceErrors:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_non_fatal_error_handling(self, service):
        """Test that non-fatal errors don't stop the service."""
        async with active_service(service) as active:
            # Simulate non-fatal error
            with patch.object(active, 'process_item', side_effect=ValueError("Retryable error")):
                await active.handle_request({"item": "test"})
                
                # Service should still be running
                assert active.state == ServiceState.STARTED
                
    @pytest.mark.asyncio
    async def test_fatal_error_handling(self, service):
        """Test that fatal errors properly shut down the service.""" 
        async with active_service(service) as active:
            # Record a fatal error
            active.record_error(RuntimeError("Fatal error"), is_fatal=True)
            
            # Wait for service to stop
            await wait_for_service_state(active, ServiceState.STOPPED, timeout=5.0)
            
            assert active.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_invalid_message_handling(self, service):
        """Test handling of invalid messages."""
        async with active_service(service) as active:
            # Send invalid message
            invalid_message = {"invalid": "data"}
            
            # Should not crash the service
            await active.handle_message(invalid_message)
            
            # Service should still be running
            assert active.state == ServiceState.STARTED
```

### Testing Configuration Validation

```python
class TestServiceConfiguration:
    """Test configuration validation and override patterns."""

    def test_default_configuration(self):
        """Test service with default configuration."""
        config = MyServiceConfig()
        service = MyService(config)
        
        assert service.config.service_name == "my-service"  # Default
        assert service.config.work_interval > 0

    def test_configuration_overrides(self):
        """Test configuration override patterns."""
        override_config = {
            "service_name": "custom-service",
            "work_interval": 2.0,
            "nested_config": {
                "param": "custom_value"
            }
        }
        
        config = MyServiceConfig.from_overrides(default_config=override_config)
        service = MyService(config)
        
        assert service.config.service_name == "custom-service"
        assert service.config.work_interval == 2.0
        assert service.config.nested_config.param == "custom_value"

    def test_configuration_validation_errors(self):
        """Test that invalid configurations raise appropriate errors."""
        with pytest.raises(ValidationError):
            MyServiceConfig.from_overrides(
                default_config={"work_interval": -1.0}  # Invalid value
            )
```

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

## ## Using the `active_service()` Context Manager

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

## Common Testing Anti-Patterns to Avoid

### ❌ Configuration Anti-Patterns

```python
# BAD: Creating temporary configuration files
with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
    f.write('service_name = "test"\n')
    f.flush()
    config = MyServiceConfig.from_file(f.name)

# GOOD: Use configuration overrides
config = MyServiceConfig.from_overrides(
    default_config={"service_name": "test"}
)
```

```python
# BAD: Testing implementation details
assert service._internal_counter == 5  # Fragile!

# GOOD: Test public behavior
assert service.get_counter() == 5
```

### ❌ Service Lifecycle Anti-Patterns

```python
# BAD: Manual lifecycle management
service = MyService()
await service.start()
await service.run()  # May hang or not clean up properly
await service.stop()

# GOOD: Use active_service context manager
async with active_service(service) as active:
    # Test functionality
    pass
```

### ❌ ZMQ Testing Anti-Patterns

```python
# BAD: Testing with real ZMQ sockets in unit tests
service = MyZmqService(config)  # Uses real ZMQ
await service.start()
# Flaky, slow, requires ports

# GOOD: Mock ZMQ components
with patch('my_service.ZmqService') as mock_zmq:
    service = MyZmqService(config)
    # Fast, reliable, isolated
```

### ❌ Error Testing Anti-Patterns

```python
# BAD: Not testing error scenarios
async def test_happy_path_only():
    # Only tests when everything works

# GOOD: Test error handling
async def test_error_handling():
    with patch('external.api', side_effect=ConnectionError()):
        # Test how service handles errors
```

## Testing Best Practices Summary

### ✅ Do These Things

1. **Use Modern Configuration Patterns**: Always use `BaseServiceConfig.from_overrides()` for test configurations
2. **Create Reusable Mock Factories**: Build `tests/mocks.py` with factory functions for common test objects
3. **Use `active_service()` Context Manager**: Let it handle service lifecycle and cleanup
4. **Test Both Success and Failure Cases**: Include error scenarios in your test coverage
5. **Mock External Dependencies**: Use `patch()` for external APIs, databases, file systems
6. **Use Fixtures Effectively**: Create reusable fixtures for common test setup
7. **Test Configuration Validation**: Verify that invalid configurations raise appropriate errors
8. **Use Consistent Test Data**: Create sample messages and data in your mocks module

### ❌ Avoid These Things

1. **Manual Service Lifecycle Management**: Don't manually call `start()`, `run()`, and `stop()` in tests
2. **Testing Implementation Details**: Test public APIs, not private methods or internal state
3. **String State Comparisons**: Use proper enum values (`ServiceState.STOPPED`) rather than strings
4. **Temporary Configuration Files**: Use configuration overrides instead of file I/O
5. **Real ZMQ in Unit Tests**: Mock ZMQ components for fast, reliable unit tests, see `lib/experimance_common/zmq/mocks.py`
6. **Ignoring Cleanup**: Always ensure services are properly stopped after tests
7. **Blocking Operations**: Be careful with synchronous operations that could block the asyncio loop

## Debugging Service Tests

If a test is hanging or failing in unexpected ways:

1. **Use debug utilities**: `debug_service_tasks(service)` to print information about service tasks
2. **Set appropriate timeouts**: Use `wait_for_service_state()` with reasonable timeout values
3. **Check resource cleanup**: Ensure proper cleanup in custom test fixtures
4. **Enable debug logging**: Run tests with `pytest --log-cli-level=DEBUG -s`
5. **Check service state**: Verify services reach expected states before testing functionality
6. **Inspect mock calls**: Use `mock.assert_called_with()` to verify expected interactions

## Benefits of Modern Testing Patterns

1. **Consistent Lifecycle Management**: Standardizes how services are started, managed, and stopped in tests
2. **Proper Error Handling**: Ensures services are always stopped, even if tests fail
3. **State Verification**: Waits for the service to reach the desired state before testing
4. **Resource Cleanup**: Properly handles task cancellation and service shutdown
5. **Simplified Test Code**: Eliminates boilerplate start/stop/cleanup code in each test
6. **Better Test Coverage**: Encourages testing both success and failure scenarios
7. **Maintainable Tests**: Reusable mock factories reduce code duplication
8. **Fast Test Execution**: Mocked dependencies eliminate slow I/O operations

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


## Modernizing Existing Service Tests for New ZMQ Architecture

**🎯 Key Insight**: When refactoring services to use the new ZMQ composition architecture (ControllerService, PublisherService, etc.), existing tests need to be updated to use proper mocking patterns.

### The Challenge: Legacy vs Modern ZMQ Patterns

**Legacy Pattern (OLD):**
```python
# Old pattern - services inherited from ZmqControllerMultiWorkerService
class OldService(ZmqControllerMultiWorkerService):
    def __init__(self, config):
        super().__init__(config)
        # Service had publish_message() method directly
        
# Tests used this directly
service.publish_message(message)
```

**Modern Pattern (NEW):**
```python
# New pattern - services use composition with ControllerService
class ModernService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.zmq_service = ControllerService(config.zmq)  # Composition
        
# Tests need to mock the composed service
service.zmq_service.publish(message)
```

### Successful Refactoring Pattern for Test Fixtures

When updating test fixtures for the new architecture, use this proven pattern:

```python
# Updated test fixture pattern that works
@pytest.fixture
def core_service(test_config):
    """Create core service with properly mocked ZMQ components."""
    # Add ZMQ config to test configuration
    test_config.zmq = ControllerServiceConfig(
        name="test_zmq",
        publisher=PublisherConfig(address="tcp://*", port=5555),
        subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=[]),
        workers={}
    )
    
    # Create a mock ZMQ service with proper AsyncMock methods
    mock_zmq_service = Mock()
    mock_zmq_service.start = AsyncMock()
    mock_zmq_service.stop = AsyncMock()
    mock_zmq_service.publish = AsyncMock()
    mock_zmq_service.send_work_to_worker = AsyncMock()
    mock_zmq_service.add_message_handler = Mock()
    mock_zmq_service.add_response_handler = Mock()
    
    # Patch the ControllerService class to return our mock
    with patch('my_service.my_service.ControllerService', return_value=mock_zmq_service):
        service = MyService(config=test_config)
        
        # Store reference to mock for test assertions
        service.zmq_service = mock_zmq_service
        
        yield service
```

### Converting Test Classes to Direct Patching

For test classes that don't use fixtures, use direct patching in setup_method:

```python
class TestServiceImagePublishing:
    """Test class updated for new ZMQ architecture."""
    
    def setup_method(self):
        """Set up test fixtures with proper ZMQ mocking."""
        # Create test config with ZMQ configuration
        self.mock_config = Mock(spec=ServiceConfig)
        self.mock_config.service_name = "test_service"
        # ...other config setup...
        
        # Add proper ZMQ config (not Mock objects!)
        self.mock_config.zmq = ControllerServiceConfig(
            name="test_zmq",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=[]),
            workers={}
        )
        
        # Create mock ZMQ service with AsyncMock methods
        self.mock_zmq_service = Mock()
        self.mock_zmq_service.start = AsyncMock()
        self.mock_zmq_service.stop = AsyncMock()
        self.mock_zmq_service.publish = AsyncMock()
        self.mock_zmq_service.send_work_to_worker = AsyncMock()
        self.mock_zmq_service.add_message_handler = Mock()
        self.mock_zmq_service.add_response_handler = Mock()
        
        # Start the patch
        self.controller_patcher = patch('my_service.my_service.ControllerService')
        mock_controller_class = self.controller_patcher.start()
        mock_controller_class.return_value = self.mock_zmq_service
        
        # Create service with mocked ZMQ
        self.service = MyService(config=self.mock_config)
        
        # Mock other essential methods as needed
        self.service.add_task = Mock()
        self.service.record_error = Mock()
        self.service._sleep_if_running = AsyncMock(return_value=False)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, 'controller_patcher'):
            self.controller_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_message_publishing(self):
        """Test that message publishing works with new architecture."""
        await self.service._publish_change_map(test_data, 0.5)
        
        # Assert on the mock ZMQ service, not the service itself
        self.mock_zmq_service.publish.assert_called_once()
        published_data = self.mock_zmq_service.publish.call_args[0][0]
        assert published_data["type"] == "ChangeMap"
```

### Key Refactoring Steps

1. **Update Config Creation**: Replace Mock zmq config with real ControllerServiceConfig objects
2. **Create Proper Mock ZMQ Service**: Use AsyncMock for async methods (publish, send_work_to_worker)
3. **Patch the ControllerService Class**: Use `patch('module.ControllerService')` not service attributes
4. **Update Test Assertions**: Change `service.publish_message.assert_called()` to `service.zmq_service.publish.assert_called()`
5. **Handle Method Differences**: Some methods changed (e.g., `publish_message` → `publish`, render requests use `send_work_to_worker`)

### Common Pitfalls and Solutions

#### ❌ Wrong: Mock objects in ZMQ config
```python
# This fails because Mock objects aren't iterable
self.mock_config.zmq = Mock()
self.mock_config.zmq.workers = {}  # Mock object, not real dict!
```

#### ✅ Right: Real config objects for ZMQ
```python
# This works because it creates real config objects
self.mock_config.zmq = ControllerServiceConfig(
    name="test_zmq",
    publisher=PublisherConfig(address="tcp://*", port=5555),
    workers={}  # Real dict, not Mock
)
```

#### ❌ Wrong: Patching after service creation
```python
service = MyService(config)  # Service already created ControllerService
service.zmq_service = Mock()  # Too late, real ControllerService already exists
```

#### ✅ Right: Patching before service creation
```python
with patch('my_service.ControllerService', return_value=mock_zmq_service):
    service = MyService(config)  # Now gets the mocked ControllerService
```

#### ❌ Wrong: Using wrong assertion methods
```python
# Old pattern - service had publish_message directly
service.publish_message.assert_called_once()

# New pattern - but wrong method name
service.zmq_service.publish_message.assert_called_once()  # Doesn't exist!
```

#### ✅ Right: Using correct method names
```python
# Correct for publishing
service.zmq_service.publish.assert_called_once()

# Correct for worker communication  
service.zmq_service.send_work_to_worker.assert_called_once()
```

### Migration Checklist for Existing Tests

When updating existing service tests for the new ZMQ architecture:

- [ ] **Update imports**: Add ControllerServiceConfig, PublisherConfig, SubscriberConfig
- [ ] **Fix config creation**: Replace Mock zmq config with real config objects
- [ ] **Create proper mock ZMQ service**: Use AsyncMock for async methods
- [ ] **Add proper patching**: Patch ControllerService class before service creation
- [ ] **Update test assertions**: Change from `publish_message` to appropriate ZMQ service methods
- [ ] **Handle method differences**: Check if service uses `publish` vs `send_work_to_worker`
- [ ] **Add cleanup**: Include teardown_method to stop patches
- [ ] **Test the changes**: Verify tests pass with `uv run -m pytest`

### Benefits of Updated Test Pattern

1. **Proper Isolation**: ZMQ components are fully mocked, no real network sockets
2. **Faster Tests**: No network I/O or port conflicts
3. **Reliable Assertions**: AsyncMock provides proper call tracking
4. **Future-Proof**: Works with composition-based ZMQ architecture
5. **Clear Intent**: Test code clearly shows what's being mocked and asserted

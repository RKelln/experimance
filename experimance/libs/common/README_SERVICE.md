# Experimance Common Services

This document describes the base service classes provided in `experimance_common.service` for building distributed applications with ZeroMQ.

## Quick Start Guide for New Services

**TL;DR: The fastest way to create a working service:**

1. **Choose your base class**: `BaseService` for simple services, `ZmqPublisherService`/`ZmqSubscriberService` for messaging
2. **Use the centralized config system**: Create a Pydantic config class and use `Config.from_overrides()`
3. **Follow the lifecycle pattern**: Initialize in `start()`, add tasks, call `super().start()` last
4. **Use TDD**: Write tests first using the state management system for reliable testing
5. **Handle errors properly**: Use `record_error()` with appropriate `is_fatal` flags
6. **Use TOML for config**: Human-readable, supports comments, integrates with Pydantic
7. **Use the common CLI system**: Create a `__main__.py` with `create_simple_main()` for consistent command line interfaces

### Essential Service Template

```python
# src/my_service/my_service.py
import asyncio
import logging
from experimance_common.base_service import BaseService
from experimance_common.config import Config
from .config import MyServiceConfig

logger = logging.getLogger(__name__)

class MyService(BaseService):
    def __init__(self, config_overrides: dict = None):
        super().__init__("my_service", "worker")
        
        # Load config using centralized system
        self.config: MyServiceConfig = Config.from_overrides(
            MyServiceConfig, 
            config_overrides or {}
        )
        
    async def start(self):
        """Initialize resources before starting."""
        # Initialize your resources here
        self.my_resource = await create_resource()
        
        # Register background tasks
        self.add_task(self.main_work_loop())
        self.add_task(self.health_monitor())
        
        # ALWAYS call super().start() LAST
        await super().start()
        
    async def stop(self):
        """Clean up resources after stopping."""
        # ALWAYS call super().stop() FIRST
        await super().stop()
        
        # Clean up your resources
        if hasattr(self, 'my_resource'):
            await self.my_resource.close()
            
    async def main_work_loop(self):
        """Main service logic."""
        while self.running:
            try:
                await self.do_work()
            except RetryableError as e:
                self.record_error(e, is_fatal=False)
                await self._sleep_if_running(1.0)
            except FatalError as e:
                self.record_error(e, is_fatal=True)
                break
                
            await self._sleep_if_running(0.1)  # Prevent CPU spinning
```

### Essential Config Template

```python
# src/my_service/config.py
from pydantic import BaseModel, Field
from experimance_common.config import Config

class MyServiceConfig(Config):
    """Configuration for MyService."""
    
    # Service-specific settings
    work_interval: float = Field(default=1.0, description="Interval between work cycles")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Override base config defaults if needed
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        config_file = "config.toml"  # Default config file name
```

### Essential Test Template

```python
# tests/test_my_service.py
import pytest
import asyncio
from experimance_common.service_state import ServiceState
from my_service.my_service import MyService

class TestMyService:
    async def test_service_lifecycle(self):
        """Test basic service lifecycle."""
        service = MyService()
        
        # Test startup
        await service.start()
        assert service.state == ServiceState.STARTED
        
        # Start running and wait for RUNNING state
        run_task = asyncio.create_task(service.run())
        await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        
        # Test shutdown
        await service.stop()
        assert service.state == ServiceState.STOPPED
        
        # Clean up
        if not run_task.done():
            run_task.cancel()
```

## Core Concepts

The `experimance_common.service` module provides a set of base classes designed to simplify the creation of services that communicate using ZeroMQ. These classes handle common patterns such as:

- **Asynchronous Operations**: Built on `asyncio` for non-blocking I/O.
- **Lifecycle Management**: Standard `start()`, `stop()`, and `run()` methods.
- **Graceful Shutdown**: Signal handlers for `SIGINT` and `SIGTERM` with multiple shutdown mechanisms.
- **Heartbeating**: Automatic heartbeat messages for service discovery and monitoring (for publisher services).
- **Statistics Tracking**: Basic statistics like messages sent/received and uptime.
- **Configurable Logging**: Consistent logging across services.
- **Error Handling**: Comprehensive error handling with automatic shutdown for fatal errors.
- **State Management**: Consistent service lifecycle state handling across inheritance hierarchies.
- **Centralized Configuration**: Pydantic-based config system with TOML support and validation.

## Configuration System Deep Dive

### Using the Centralized Config System

**Why use the centralized config system?**
- Single source of truth for configuration patterns
- Automatic TOML loading with fallbacks
- Pydantic validation and type safety
- Environment variable overrides
- Consistent config structure across services

**Step-by-step config implementation:**

1. **Create your config schema** (inherit from `Config`):
```python
# src/my_service/config.py
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from experimance_common.config import Config

class DatabaseConfig(BaseModel):
    """Database connection settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = Field(...)  # Required field

class MyServiceConfig(Config):
    """Configuration for MyService."""
    
    # Service-specific settings
    work_interval: float = Field(default=1.0, gt=0, description="Work loop interval")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    features: List[str] = Field(default_factory=list)
    
    # Environment-specific overrides
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    
    @validator('work_interval')
    def validate_work_interval(cls, v):
        if v < 0.1:
            raise ValueError('work_interval must be at least 0.1 seconds')
        return v
    
    class Config:
        config_file = "config.toml"  # Default filename
```

2. **Create your TOML config file**:
```toml
# config.toml
log_level = "INFO"
work_interval = 2.0
debug_mode = false
features = ["feature_a", "feature_b"]

[database]
host = "prod-db.example.com"
port = 5432
name = "my_service_db"
```

3. **Load config in your service**:
```python
# In your service __init__
class MyService(BaseService):
    def __init__(self, config_overrides: dict = None):
        super().__init__("my_service", "worker")
        
        # Load with overrides (useful for testing)
        self.config: MyServiceConfig = Config.from_overrides(
            MyServiceConfig, 
            config_overrides or {}
        )
        
        # Access config values
        logger.info(f"Database host: {self.config.database.host}")
        logger.info(f"Work interval: {self.config.work_interval}")
```

### Config Override Patterns

**For testing:**
```python
# Test with custom config
test_config = {
    "work_interval": 0.1,  # Faster for tests
    "database": {"host": "localhost", "name": "test_db"},
    "debug_mode": True
}
service = MyService(config_overrides=test_config)
```

**For different environments:**
```python
# Production overrides
prod_config = {
    "log_level": "WARNING",
    "database": {"host": "prod-db.example.com"}
}
service = MyService(config_overrides=prod_config)
```

## Testing Patterns and Best Practices

### Essential Test Structure

**Use the service state system for reliable testing:**

```python
# tests/test_my_service.py
import pytest
import asyncio
from experimance_common.service_state import ServiceState
from my_service.my_service import MyService

@pytest.fixture
async def service():
    """Create a test service with fast config."""
    test_config = {
        "work_interval": 0.01,  # Very fast for tests
        "log_level": "DEBUG"
    }
    service = MyService(config_overrides=test_config)
    yield service
    
    # Cleanup
    if service.state != ServiceState.STOPPED:
        await service.stop()

class TestMyService:
    async def test_service_startup(self, service):
        """Test service starts correctly."""
        await service.start()
        assert service.state == ServiceState.STARTED
        
        # Start running in background
        run_task = asyncio.create_task(service.run())
        
        # Wait for RUNNING state with timeout
        await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        assert service.state == ServiceState.RUNNING
        
        # Clean shutdown
        await service.stop()
        assert service.state == ServiceState.STOPPED
        
        # Clean up background task
        if not run_task.done():
            run_task.cancel()
    
    async def test_error_handling(self, service):
        """Test service handles errors correctly."""
        # Inject a failure
        service.should_fail = True
        
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        # Wait for service to handle error
        await asyncio.sleep(0.1)
        
        # Check error was recorded
        assert len(service.error_history) > 0
        
        await service.stop()
        if not run_task.done():
            run_task.cancel()
```

### Testing ZMQ Services

**For services that use ZMQ:**

```python
class TestZmqService:
    async def test_message_publishing(self):
        """Test service publishes messages correctly."""
        service = MyZmqService()
        
        # Start service
        await service.start()
        run_task = asyncio.create_task(service.run())
        await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        
        # Give time for initial messages
        await asyncio.sleep(0.1)
        
        # Check statistics
        assert service.stats.messages_sent > 0
        
        # Clean up
        await service.stop()
        if not run_task.done():
            run_task.cancel()
```

### Common Testing Anti-Patterns

**❌ Don't test internal implementation details:**
```python
# BAD: Testing private methods or internal state
assert service._internal_counter == 5  # Fragile!

# GOOD: Test public behavior
assert service.get_counter() == 5
```

**❌ Don't ignore service lifecycle:**
```python
# BAD: Not properly starting/stopping services
service = MyService()
await service.main_work_loop()  # May fail without proper setup

# GOOD: Use proper lifecycle
service = MyService()
await service.start()
run_task = asyncio.create_task(service.run())
# ... test behavior ...
await service.stop()
```

## Service Lifecycle and Task Management

### Task Management Best Practices

**Add tasks in `start()`, not `__init__()`:**

```python
class MyService(BaseService):
    async def start(self):
        """Initialize and start background tasks."""
        # Initialize resources first
        self.database = await connect_to_database()
        
        # Add background tasks
        self.add_task(self.main_work_loop())
        self.add_task(self.health_monitor())
        self.add_task(self.periodic_cleanup())
        
        # ALWAYS call super().start() LAST
        await super().start()
    
    async def main_work_loop(self):
        """Main service work loop."""
        while self.running:
            try:
                await self.process_work()
            except RetryableError as e:
                self.record_error(e, is_fatal=False)
                await self._sleep_if_running(1.0)  # Wait before retry
            except FatalError as e:
                self.record_error(e, is_fatal=True)
                break  # Fatal error stops the loop
                
            await self._sleep_if_running(0.1)  # Prevent CPU spinning
    
    async def health_monitor(self):
        """Monitor service health."""
        while self.running:
            try:
                await self.check_health()
            except Exception as e:
                self.record_error(e, is_fatal=False)
            
            await self._sleep_if_running(5.0)  # Health check every 5 seconds
```

### Using `_sleep_if_running()` vs `asyncio.sleep()`

**Always use `_sleep_if_running()` in loops:**

```python
# ✅ GOOD: Respects service shutdown
while self.running:
    await self.do_work()
    await self._sleep_if_running(1.0)  # Interrupts on shutdown

# ❌ BAD: Ignores service shutdown
while self.running:
    await self.do_work()
    await asyncio.sleep(1.0)  # Doesn't interrupt on shutdown
```

### Error Recovery Patterns

**Distinguish between retryable and fatal errors:**

```python
async def process_item(self, item):
    """Process an item with proper error handling."""
    try:
        result = await self.risky_operation(item)
        return result
    except ConnectionError as e:
        # Retryable: network issues
        self.record_error(e, is_fatal=False)
        raise  # Let caller decide retry logic
    except ValidationError as e:
        # Not retryable: bad data
        self.record_error(e, is_fatal=False, 
                         custom_message=f"Invalid item {item.id}: {e}")
        return None  # Skip this item
    except SystemExit as e:
        # Fatal: system shutdown
        self.record_error(e, is_fatal=True)
        raise
```

## Command Line Interface (CLI) System

**TL;DR: Use the common CLI utilities for consistent command line interfaces across all services.**

The `experimance_common.cli` module provides standardized command line argument parsing, logging setup, and service execution that all Experimance services should use.

### Essential CLI Template

```python
# src/my_service/__main__.py
"""
Command line entry point for My Service.
"""
from experimance_common.cli import create_simple_main
from .my_service import run_my_service
from .config import DEFAULT_CONFIG_PATH

# Create the main function using the common CLI utility
main = create_simple_main(
    service_name="My Service",
    description="Brief description of what your service does",
    service_runner=run_my_service,
    default_config_path=DEFAULT_CONFIG_PATH
)

if __name__ == "__main__":
    main()
```

### CLI with Custom Arguments

For services that need additional command line arguments:

```python
# src/my_service/__main__.py
from experimance_common.cli import create_simple_main
from .my_service import run_my_service
from .config import DEFAULT_CONFIG_PATH

# Extra arguments specific to your service
extra_args = {
    '--model': {
        'choices': ['gpt-4', 'gpt-3.5-turbo', 'claude-3'],
        'default': 'gpt-4',
        'help': 'AI model to use for generation'
    },
    '--max-tokens': {
        'type': int,
        'default': 1000,
        'help': 'Maximum tokens for responses'
    },
    '--debug-mode': {
        'action': 'store_true',
        'help': 'Enable debug mode with extra logging'
    }
}

# Create the main function using the common CLI utility
main = create_simple_main(
    service_name="My Service",
    description="Service with custom command line arguments",
    service_runner=run_my_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    extra_args=extra_args
)

if __name__ == "__main__":
    main()
```

### Standard CLI Features

All services automatically get these standard command line arguments:

- `--log-level, -l`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--config, -c`: Path to configuration file (if default_config_path is provided)
- `--help, -h`: Show help message with all available arguments

### Usage Examples

```bash
# Basic usage with default settings
uv run -m my_service

# With custom log level
uv run -m my_service --log-level DEBUG

# With custom config file
uv run -m my_service --config /path/to/custom/config.toml

# With service-specific arguments
uv run -m my_service --log-level DEBUG --model gpt-4 --max-tokens 2000

# Show help for any service
uv run -m my_service --help
```

### ⚠️ Important: Logging Configuration

**DO NOT configure logging in library modules!** The CLI system handles logging configuration.

```python
# ❌ DON'T do this in service modules
logging.basicConfig(level=logging.INFO, format='...')

# ✅ DO this instead - just get loggers
logger = logging.getLogger(__name__)
```

Only the main entry point (CLI) should configure logging. This ensures that command line arguments like `--log-level DEBUG` work correctly.


## Common Patterns and Anti-Patterns

### ✅ Do This

1. **Use the state system for lifecycle management**
2. **Always call `super().start()` last and `super().stop()` first**
3. **Use `_sleep_if_running()` in loops for proper shutdown**
4. **Record errors with context using `record_error()`**
5. **Use TOML config with Pydantic validation**
6. **Test with fast config overrides**
7. **Use proper task cleanup in tests**

### ❌ Don't Do This

1. **Don't set service state directly (`self.state = ...`)**
2. **Don't use `asyncio.sleep()` in service loops**
3. **Don't initialize heavy resources in `__init__()`**
4. **Don't ignore service lifecycle in tests**
5. **Don't use raw dict access for config**
6. **Don't duplicate error logging**
7. **Don't forget to await `service.stop()` in tests**

## Shutdown and Error Handling Best Practices

### Graceful Shutdown Options

Services provide multiple ways to initiate graceful shutdown:

#### 1. `await service.stop()` - Immediate Shutdown
Use when you need to stop the service immediately and can wait for completion:
```python
# ✅ RECOMMENDED: For immediate, blocking shutdown
await service.stop()
```

#### 2. `service.request_stop()` - Non-blocking Shutdown Request  
Use when you want to schedule a shutdown but continue current operations:
```python
# ✅ RECOMMENDED: For non-blocking shutdown requests
service.request_stop()
# Continue with cleanup or current operations...
```

#### 3. **❌ DON'T set state directly**
Setting `self.state = ServiceState.STOPPING` only changes the state but doesn't perform cleanup:
```python
# ❌ BAD: This doesn't actually stop the service!
self.state = ServiceState.STOPPING  # Only changes state, no cleanup
```

### Error Handling Best Practices

#### Always Record Errors with Context
```python
# ✅ GOOD: Record all errors for monitoring with custom context
try:
    result = await risky_operation(request_id)
except Exception as e:
    # Use custom_message for context-specific error details
    self.record_error(e, is_fatal=False, 
                     custom_message=f"Error processing request {request_id}: {e}")
```

#### Avoid Duplicate Logging
```python
# ❌ BAD: Duplicate logging creates noise
try:
    await operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)  # First log
    self.record_error(e, is_fatal=False)  # Second log (duplicate!)

# ✅ GOOD: Single, contextual error recording
try:
    await operation()
except Exception as e:
    self.record_error(e, is_fatal=False, 
                     custom_message=f"Operation failed: {e}")  # Single log with context
```

#### record_error() Method Signature
```python
def record_error(self, error: Exception, is_fatal: bool = False, custom_message: Optional[str] = None):
    """Record an error and update service status.
    
    Args:
        error: The exception that occurred
        is_fatal: Whether this error should mark the service as fatally errored
        custom_message: Optional custom message to log instead of default format
    """
```

#### Fatal Errors Automatically Stop Services
```python
# ✅ GOOD: Fatal errors automatically trigger shutdown
try:
    critical_operation()
except CriticalError as e:
    self.record_error(e, is_fatal=True, 
                     custom_message=f"Critical system failure in {operation_name}: {e}")
    # No need to manually call stop() - it's automatic!
```

#### Error Status Management
```python
# Reset error status after recovery
try:
    recovery_operation()
    self.reset_error_status()  # Mark service as healthy again
except Exception as e:
    self.record_error(e, custom_message=f"Recovery failed: {e}")
```

### Common Shutdown Patterns

#### From Service Tasks
```python
async def main_work_loop(self):
    while self.running:
        try:
            result = await do_work()
            if should_shutdown(result):
                # ✅ Non-blocking shutdown from within a task
                self.request_stop()
                break
        except FatalError as e:
            # ✅ Fatal error auto-stops, just record it
            self.record_error(e, is_fatal=True)
            break
        except RecoverableError as e:
            # ✅ Non-fatal error, service continues
            self.record_error(e, is_fatal=False)
```

#### External Shutdown
```python
# ✅ From outside the service (tests, main, etc.)
await service.stop()

# ✅ Or request shutdown and let it complete naturally
service.request_stop()
```

#### Signal Handling
Signal handlers are automatically set up and will call `stop()` gracefully:
- `SIGINT` (Ctrl+C): Graceful shutdown
- `SIGTERM`: Graceful shutdown  
- `SIGTSTP`: Graceful shutdown

### Task Naming and Debugging

The service framework automatically creates descriptive task names for shutdown operations:
- `{service_name}-requested-stop`: Manual shutdown requests
- `{service_name}-fatal-error-stop`: Fatal error triggered shutdown
- `{service_name}-task-error-stop`: Task error triggered shutdown

These names help with debugging and monitoring.

## Comprehensive Best Practices for Service Development

### 1. Service Initialization and Setup

#### Always Call Parent Constructors
```python
class MyService(BaseService):
    def __init__(self, custom_param: str):
        # ✅ GOOD: Always call parent constructor first
        super().__init__("my_service", "worker")
        self.custom_param = custom_param
```

#### Initialize Resources in start() Method
```python
async def start(self):
    # ✅ GOOD: Initialize resources before calling super().start()
    self.database = await connect_to_database()
    self.cache = create_cache()
    
    # Register tasks for background operations
    self.add_task(self.background_worker())
    self.add_task(self.health_monitor())
    
    # Always call parent start() last
    await super().start()
```

### 2. Task Management Best Practices

#### Use add_task() for Background Operations
```python
async def start(self):
    # ✅ GOOD: Register all background tasks
    self.add_task(self.process_queue())
    self.add_task(self.monitor_health())
    self.add_task(self.periodic_cleanup())
    await super().start()
```

#### Proper Task Loop Patterns
```python
# ✅ GOOD: Simple continuous work
async def background_worker(self):
    while self.running:
        await do_work()
        await asyncio.sleep(1.0)  # Prevent CPU spinning

# ✅ GOOD: Work with delays and state checking
async def complex_worker(self):
    while self.running:
        await do_first_work()
        
        # Use _sleep_if_running for state-aware sleeping
        if not await self._sleep_if_running(5.0):
            break  # Service stopped during sleep
            
        await do_second_work()
```

#### Task Error Handling
```python
async def background_task(self):
    while self.running:
        try:
            await risky_operation()
        except RetryableError as e:
            # ✅ GOOD: Log and continue for retryable errors
            self.record_error(e, is_fatal=False)
            await asyncio.sleep(1.0)  # Brief backoff
        except FatalError as e:
            # ✅ GOOD: Fatal errors auto-stop the service
            self.record_error(e, is_fatal=True)
            break  # Exit the task
```

### 3. Resource Management and Cleanup

#### Implement Proper Cleanup in stop()
```python
async def stop(self):
    # ✅ GOOD: Call parent stop() first to handle framework cleanup
    await super().stop()
    
    # Then clean up your specific resources
    if hasattr(self, 'database'):
        await self.database.close()
    if hasattr(self, 'cache'):
        self.cache.clear()
```

#### Use Context Managers for Resources
```python
async def handle_request(self, request):
    try:
        async with self.get_database_connection() as conn:
            result = await process_with_db(conn, request)
            return result
    except Exception as e:
        self.record_error(e, is_fatal=False, 
                         custom_message=f"Database operation failed: {e}")
        raise
```

### 4. Error Handling Patterns

#### Categorize Errors Appropriately
```python
async def process_data(self, data):
    try:
        result = await complex_operation(data)
        return result
    except NetworkTimeout as e:
        # ✅ Transient error - retry possible
        self.record_error(e, is_fatal=False, 
                         custom_message=f"Network timeout during data processing (retry possible): {e}")
        raise
    except InvalidConfiguration as e:
        # ✅ Fatal error - service cannot continue
        self.record_error(e, is_fatal=True,
                         custom_message=f"Invalid configuration detected, service cannot continue: {e}")
        raise
    except DataValidationError as e:
        # ✅ Non-fatal - log and return error response
        self.record_error(e, is_fatal=False,
                         custom_message=f"Data validation failed for input: {e}")
        return {"error": "Invalid data"}
```

#### Error Recovery Patterns
```python
async def resilient_operation(self):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries and self.running:
        try:
            return await potentially_failing_operation()
        except RetryableError as e:
            retry_count += 1
            self.record_error(e, is_fatal=False,
                             custom_message=f"Retryable error (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count >= max_retries:
                self.record_error(e, is_fatal=True,
                                 custom_message=f"Max retries ({max_retries}) exceeded, operation failed: {e}")
                raise
                
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
```

### 5. State and Lifecycle Management

#### Respect Service States in Custom Methods
```python
async def process_request(self, request):
    # ✅ GOOD: Check service state before processing
    if not self.running:
        raise ServiceNotRunningError("Service is not running")
        
    return await handle_request(request)
```

#### Use State Callbacks for Custom Logic
```python
def __init__(self, name: str):
    super().__init__(name)
    
    # ✅ GOOD: Register callbacks for state transitions
    self.register_state_callback(ServiceState.RUNNING, self._on_running)
    self.register_state_callback(ServiceState.STOPPED, self._on_stopped)

def _on_running(self):
    logger.info("Service is now fully operational")
    
def _on_stopped(self):
    logger.info("Service has been stopped")
```

### 6. Logging and Monitoring Best Practices

#### Use Consistent Logging Patterns
```python
async def important_operation(self):
    logger.info(f"Starting important operation for {self.service_name}")
    
    try:
        result = await do_operation()
        logger.info(f"Operation completed successfully: {result}")
        return result
    except Exception as e:
        # ✅ GOOD: Single contextual error recording (no duplicate logging)
        self.record_error(e, is_fatal=False, 
                         custom_message=f"Operation failed: {e}")
        raise
```

#### Track Custom Metrics
```python
def __init__(self, name: str):
    super().__init__(name)
    self.requests_processed = 0
    self.custom_metric = 0

async def process_request(self, request):
    self.requests_processed += 1
    # Process request...
    self.messages_sent += 1  # Update base class counters
```

### 7. Testing Service Implementation

> **Extended Testing Resources:**
> - For comprehensive service testing best practices, see [README_SERVICE_TESTING.md](../../utils/tests/README_SERVICE_TESTING.md)
> - For ZeroMQ-specific testing guidance, see [README_ZMQ_TESTS.md](../../utils/tests/README_ZMQ_TESTS.md)
> - Consider using the `active_service()` context manager from `experimance_common.test_utils` for simplified testing

#### Test Lifecycle Transitions
```python
async def test_service_startup():
    service = MyService("test")
    
    # Test state transitions
    assert service.state == ServiceState.INITIALIZED
    
    await service.start()
    assert service.state == ServiceState.STARTED
    
    # Test running state
    run_task = asyncio.create_task(service.run())
    await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
    
    # Test shutdown
    await service.stop()
    assert service.state == ServiceState.STOPPED
```

#### Test Error Handling
```python
async def test_error_handling():
    service = MyService("test")
    await service.start()
    
    # Test non-fatal error
    service.record_error(ValueError("test"), is_fatal=False)
    assert service.status == ServiceStatus.ERROR
    assert service.state == ServiceState.RUNNING  # Still running
    
    # Test fatal error
    service.record_error(RuntimeError("fatal"), is_fatal=True)
    assert service.status == ServiceStatus.FATAL
    # Service should be stopping automatically
```

#### Simplified Testing with active_service()
```python
async def test_with_active_service():
    """Test using the active_service context manager."""
    # Create your service with test-specific config
    service = MyService(config_overrides={"work_interval": 0.1})
    
    # The context manager handles start, run, and cleanup
    async with active_service(service) as running_service:
        # Service is now running and ready for testing
        assert running_service.state == ServiceState.RUNNING
        
        # Test your service functionality
        result = await running_service.process_item(test_item)
        assert result is not None
        
    # Service is automatically stopped when exiting the context
    assert service.state == ServiceState.STOPPED
```

### 8. Common Anti-Patterns to Avoid

#### ❌ Don't Block the Event Loop
```python
# ❌ BAD: Blocking operations
def blocking_operation(self):
    time.sleep(5)  # Blocks the entire event loop
    
# ✅ GOOD: Use async operations
async def async_operation(self):
    await asyncio.sleep(5)  # Non-blocking
```

#### ❌ Don't Ignore Exceptions
```python
# ❌ BAD: Silent exception handling
try:
    await risky_operation()
except Exception:
    pass  # Silent failure

# ✅ GOOD: Proper exception handling
try:
    await risky_operation()
except Exception as e:
    self.record_error(e, is_fatal=False, 
                     custom_message=f"Risky operation failed: {e}")
```

#### ❌ Don't Manage State Manually
```python
# ❌ BAD: Manual state management
self.state = ServiceState.STOPPING  # Bypasses cleanup

# ✅ GOOD: Use proper shutdown methods
await self.stop()  # Proper cleanup
```

#### ❌ Don't Create Tasks Without Registration
```python
# ❌ BAD: Unmanaged background tasks
asyncio.create_task(self.background_work())  # Not tracked

# ✅ GOOD: Register tasks with the service
self.add_task(self.background_work())  # Properly managed
```

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
            pub_address=f"tcp://*:{DEFAULT_PORTS['events']}",
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
            pub_address=f"tcp://*:{DEFAULT_PORTS['events']}",      # For publishing on unified events channel
            sub_address=f"tcp://localhost:{DEFAULT_PORTS['events']}", # For subscribing to unified events channel
            push_address=f"tcp://*:{DEFAULT_PORTS['transitions_pull']}",        # For pushing tasks to workers
            pull_address=f"tcp://*:{DEFAULT_PORTS['loops_pull']}",      # For pulling results from workers
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
    
    # IMPORTANT: do NOT add try: and catch on KeyboardInterrupt and Exception, 
    # these are handled for you
    await service.run() # Service runs until shutdown signal

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


## Service Development Tips and Patterns

### 1. Task Loop Patterns

#### Simple Continuous Work Pattern
```python
from experimance_common.constants import TICK

async def background_worker(self):
    """Simple continuous work with consistent timing."""
    while self.running:
        await do_work()
        await asyncio.sleep(TICK)  # Small delay to prevent CPU spinning
```

#### Work-Sleep-Work Pattern
```python
async def complex_worker(self):
    """Pattern for work followed by delay followed by more work."""
    while self.running:
        await do_first_batch_of_work()
        
        # Use _sleep_if_running for state-aware sleeping
        if not await self._sleep_if_running(5.0):
            break  # Service stopped during sleep
            
        await do_second_batch_of_work()
```

#### Periodic Task Pattern
```python
async def periodic_maintenance(self):
    """Pattern for tasks that run periodically."""
    while self.running:
        try:
            await perform_maintenance()
        except Exception as e:
            self.record_error(e, is_fatal=False, 
                             custom_message=f"Maintenance task failed: {e}")
            
        # Sleep for maintenance interval
        if not await self._sleep_if_running(300):  # 5 minutes
            break
```

### 2. Performance and Resource Management

#### Batch Processing Pattern
```python
async def batch_processor(self):
    """Process items in batches for efficiency."""
    batch = []
    batch_size = 10
    
    while self.running:
        try:
            # Collect items for batch
            item = await self.get_next_item(timeout=1.0)
            if item:
                batch.append(item)
                
            # Process batch when full or on timeout
            if len(batch) >= batch_size or not item:
                if batch:
                    await self.process_batch(batch)
                    batch.clear()
                    
        except TimeoutError:
            # Process partial batch on timeout
            if batch:
                await self.process_batch(batch)
                batch.clear()
        except Exception as e:
            self.record_error(e, is_fatal=False, 
                             custom_message=f"Batch processing failed: {e}")
            batch.clear()  # Clear batch on error
```

#### Resource Pool Pattern
```python
class PooledResourceService(BaseService):
    def __init__(self, name: str, pool_size: int = 5):
        super().__init__(name)
        self.pool_size = pool_size
        self.resource_pool = []
        
    async def start(self):
        # Initialize resource pool
        for _ in range(self.pool_size):
            resource = await create_expensive_resource()
            self.resource_pool.append(resource)
            
        await super().start()
        
    async def get_resource(self):
        """Get a resource from the pool."""
        while self.running and not self.resource_pool:
            await asyncio.sleep(0.1)  # Wait for available resource
            
        if self.resource_pool:
            return self.resource_pool.pop()
        return None
        
    def return_resource(self, resource):
        """Return a resource to the pool."""
        self.resource_pool.append(resource)
```

### 3. Error Resilience Patterns

#### Circuit Breaker Pattern
```python
class CircuitBreakerService(BaseService):
    def __init__(self, name: str):
        super().__init__(name)
        self.failure_count = 0
        self.failure_threshold = 5
        self.recovery_timeout = 30
        self.last_failure_time = None
        self.circuit_open = False
        
    async def call_external_service(self):
        # Check circuit breaker state
        if self.circuit_open:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.circuit_open = False
                self.failure_count = 0
                logger.info("Circuit breaker reset")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
                
        try:
            result = await external_service_call()
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                logger.warning("Circuit breaker opened due to failures")
                
            self.record_error(e, is_fatal=False)
            raise
```

#### Retry with Backoff Pattern
```python
async def resilient_operation(self, max_retries: int = 3):
    """Perform an operation with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await potentially_failing_operation()
        except RetryableError as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                self.record_error(e, is_fatal=True)
                raise
                
            self.record_error(e, is_fatal=False)
            backoff_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {backoff_time:.1f}s")
            
            if not await self._sleep_if_running(backoff_time):
                raise ServiceStoppingError("Service stopped during retry")
```

### 4. Monitoring and Observability

#### Custom Metrics Pattern
```python
class MetricsTrackingService(BaseService):
    def __init__(self, name: str):
        super().__init__(name)
        self.custom_metrics = {
            'requests_processed': 0,
            'errors_recovered': 0,
            'last_success_time': None,
            'processing_times': []
        }
        
    async def process_request(self, request):
        start_time = time.time()
        try:
            result = await handle_request(request)
            
            # Track success metrics
            self.custom_metrics['requests_processed'] += 1
            self.custom_metrics['last_success_time'] = time.time()
            
            processing_time = time.time() - start_time
            self.custom_metrics['processing_times'].append(processing_time)
            
            # Keep only recent processing times
            if len(self.custom_metrics['processing_times']) > 100:
                self.custom_metrics['processing_times'] = \
                    self.custom_metrics['processing_times'][-50:]
                    
            return result
        except Exception as e:
            self.record_error(e, is_fatal=False)
            self.custom_metrics['errors_recovered'] += 1
            raise
            
    def get_average_processing_time(self):
        times = self.custom_metrics['processing_times']
        return sum(times) / len(times) if times else 0
```

#### Health Check Pattern
```python
async def health_monitor(self):
    """Monitor service health and report status."""
    while self.running:
        try:
            # Check various health indicators
            database_ok = await self.check_database_health()
            external_service_ok = await self.check_external_services()
            resource_usage_ok = self.check_resource_usage()
            
            if all([database_ok, external_service_ok, resource_usage_ok]):
                if self.status != ServiceStatus.HEALTHY:
                    self.reset_error_status()
            else:
                logger.warning("Health check failed")
                self.status = ServiceStatus.WARNING
                
        except Exception as e:
            self.record_error(e, is_fatal=False)
            
        await self._sleep_if_running(30)  # Check every 30 seconds
```

### 5. Configuration and Environment

#### Environment-Aware Configuration
```python
class ConfigurableService(BaseService):
    def __init__(self, name: str):
        super().__init__(name)
        
        # Load configuration based on environment
        env = os.getenv('ENVIRONMENT', 'development')
        self.config = self.load_config(env)
        
        # Validate required configuration
        self.validate_config()
        
    def load_config(self, env: str) -> dict:
        config_file = f"config/{env}.toml"
        with open(config_file, 'r') as f:
            return toml.load(f)
            
    def validate_config(self):
        required_keys = ['database_url', 'api_key', 'max_connections']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required config: {key}")
```

### 6. Testing Patterns

#### Service Testing Utilities
```python
class ServiceTestBase:
    """Base class for service testing with common utilities."""
    
    async def start_service_for_test(self, service):
        """Start a service and wait for it to be running."""
        await service.start()
        
        # Run the service in background
        self.run_task = asyncio.create_task(service.run())
        
        # Wait for it to be fully running
        await service.wait_for_state(ServiceState.RUNNING, timeout=5.0)
        
    async def stop_service_after_test(self, service):
        """Cleanly stop a service after testing."""
        if service.state != ServiceState.STOPPED:
            await service.stop()
            
        # Clean up the run task
        if hasattr(self, 'run_task') and not self.run_task.done():
            self.run_task.cancel()
            try:
                await self.run_task
            except asyncio.CancelledError:
                pass
                
    @contextmanager
    def expect_error_logged(self, error_type):
        """Context manager to verify that an error was logged."""
        initial_error_count = service.errors
        yield
        assert service.errors > initial_error_count
        assert service.status in [ServiceStatus.ERROR, ServiceStatus.FATAL]
```

### 7. Memory Management and Performance

#### Avoiding Memory Leaks
```python
async def process_large_dataset(self, dataset):
    """Process large datasets without memory accumulation."""
    # Process in chunks to avoid memory buildup
    chunk_size = 1000
    
    for i in range(0, len(dataset), chunk_size):
        if not self.running:
            break
            
        chunk = dataset[i:i + chunk_size]
        
        try:
            await self.process_chunk(chunk)
        except Exception as e:
            self.record_error(e, is_fatal=False)
            
        # Explicitly clear chunk reference
        del chunk
        
        # Yield control to prevent blocking
        await asyncio.sleep(0)
```

## Quick Reference

### Essential Service Patterns

```python
# Service structure template
class MyService(BaseService):
    def __init__(self, name: str):
        super().__init__(name, "service_type")
        # Initialize service-specific attributes
        
    async def start(self):
        # Initialize resources
        self.add_task(self.background_task())
        await super().start()
        
    async def stop(self):
        await super().stop() # do this first before stopping your own resources
        # Clean up resources
        
    async def background_task(self):
        while self.running:
            try:
                await do_work()
            except FatalError as e:
                self.record_error(e, is_fatal=True)
                break
            except RetryableError as e:
                self.record_error(e, is_fatal=False)
            
            await self._sleep_if_running(1.0)
```

### Shutdown Methods Quick Reference

| Method                                   | Use Case                      | Blocks Caller | Auto Cleanup |
| ---------------------------------------- | ----------------------------- | ------------- | ------------ |
| `await service.stop()`                   | Immediate shutdown needed     | ✅ Yes         | ✅ Yes        |
| `service.request_stop()`                 | Non-blocking shutdown request | ❌ No          | ✅ Yes        |
| `service.record_error(e, is_fatal=True)` | Error-triggered shutdown      | ❌ No          | ✅ Yes        |

### Error Handling Quick Reference

| Error Type           | `is_fatal` | Service Behavior  | Use Case                                      |
| -------------------- | ---------- | ----------------- | --------------------------------------------- |
| `RecoverableError`   | `False`    | Continues running | Network timeouts, temporary failures          |
| `ConfigurationError` | `True`     | Auto-stops        | Invalid config, missing resources             |
| `ValidationError`    | `False`    | Continues running | Bad input data, recoverable issues            |
| `SystemError`        | `True`     | Auto-stops        | System resource exhaustion, critical failures |

### State Checking Patterns

```python
# Check if service should continue
if not self.running:
    return

# State-aware sleeping
if not await self._sleep_if_running(5.0):
    break  # Service stopped during sleep

# Wait for specific state in tests
await service.wait_for_state(ServiceState.RUNNING, timeout=5.0)
```

### Common Imports

```python
import asyncio
import logging
from experimance_common.base_service import BaseService, ServiceStatus
from experimance_common.service_state import ServiceState
from experimance_common.constants import TICK
```
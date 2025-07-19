# Experimance Common Services

This document describes the base service classes and ZMQ composition patterns provided in `experimance_common` for building distributed applications with ZeroMQ.

**🏗️ Architecture Note**: This guide uses the modern **composition-based ZMQ architecture**. For ZMQ-specific patterns, see [README_ZMQ.md](README_ZMQ.md). The older inheritance-based ZMQ services have been deprecated in favor of this more flexible approach.

## Key Configuration Patterns (Updated 2024)

**📋 Modern Service Configuration Rules:**
- Extend `BaseServiceConfig` for all service configs (provides common fields like `service_name`)
- Use direct config instantiation, not factory functions
- Use shared constants from `experimance_common.constants` 
- Use shared enums like `MessageType` for ZMQ topics, not strings
- Override service name with simple field defaults: `service_name: str = "my-service"`
- Test both default and custom configurations

## Quick Start Guide for New Services

**TL;DR: The fastest way to create a working service:**

1. **Choose your base class**: `BaseService` for all services, then add ZMQ functionality using composition with `PubSubService`, `WorkerService`, or `ControllerService`
2. **Use the centralized config system**: Create a `config.py` and add Pydantic config classes that subclass `BaseServiceConfig`
3. **Follow the lifecycle pattern**: Initialize in `start()`, add tasks, call `super().start()` last
4. **Use TDD**: Write tests first using the state management system for reliable testing
5. **Handle errors properly**: Use `record_error()` with appropriate `is_fatal` flags
6. **Use TOML for config**: Human-readable, supports comments, integrates with Pydantic
7. **Use the common CLI system**: Create a `__main__.py` with `create_simple_main()` for consistent command line interfaces
8. **Add infrastructure components**: Create systemd service files and update deployment scripts

### Infrastructure Components for New Services

**🚀 Important: After creating your service, add it to the deployment infrastructure:**

1. **Create systemd service file** in `infra/systemd/`:
   ```bash
   # infra/systemd/experimance-my-service.service
   [Unit]
   Description=Experimance My Service
   After=network.target
   Wants=network.target
   
   [Service]
   Type=simple
   User=experimance
   Group=experimance
   WorkingDirectory=/home/experimance/experimance
   Environment=PATH=/home/experimance/.local/bin:/usr/local/bin:/usr/bin:/bin
   Environment=EXPERIMANCE_ENV=production
   ExecStart=/home/experimance/.local/bin/uv run -m experimance_my_service
   Restart=always
   RestartSec=10
   StandardOutput=journal
   StandardError=journal
   
   [Install]
   WantedBy=multi-user.target
   ```

2. **Update service detection** - the `get_project_services.py` script should automatically detect your service if it follows the naming conventions

3. **Test with deployment scripts**:
   ```bash
   # Test locally first
   ./scripts/dev my_service
   
   # Test with deploy script
   sudo ./infra/scripts/deploy.sh experimance status
   sudo ./infra/scripts/deploy.sh experimance start my_service
   ```

4. **Service naming conventions**:
   - **Module name**: `experimance_my_service` (matches your service directory)
   - **Systemd service**: `experimance-my-service.service` (kebab-case)
   - **Service directory**: `services/my_service/` (snake_case)
   - **Config file**: `projects/experimance/my_service.toml`

5. **Update infrastructure scripts**:
   - The `update.sh` script will automatically handle new services during updates
   - The `deploy.sh` script uses `get_project_services.py` to detect all services
   - No manual updates needed to infrastructure scripts if naming conventions are followed
   - **Note**: The `update.sh` script may need to be updated to use `uv` instead of virtual environments

6. **Production deployment checklist**:
   ```bash
   # Copy systemd service file to system
   sudo cp infra/systemd/experimance-my-service.service /etc/systemd/system/
   
   # Reload systemd and enable service
   sudo systemctl daemon-reload
   sudo systemctl enable experimance-my-service.service
   
   # Test the service
   sudo systemctl start experimance-my-service.service
   sudo systemctl status experimance-my-service.service
   ```

### Essential Service Template

```python
# src/my_service/config.py
from experimance_common.schemas import MessageType
from experimance_common.config import BaseServiceConfig
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PUBLISHER_PORT, ZMQ_BIND_ADDRESS
from pydantic import Field

class MyServiceConfig(BaseServiceConfig):
    """Configuration for MyService."""
    
    # Override service name with sensible default
    service_name: str = "my-service"
    
    # Service-specific fields
    work_interval: float = Field(default=1.0, description="Work loop interval")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # ZMQ configuration using shared patterns
    zmq: PubSubServiceConfig = Field(
        default_factory=lambda: PubSubServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT + 10,  # Offset for your service
                topics=[MessageType.STATUS, MessageType.HEARTBEAT]
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT,
                topics=[MessageType.COMMAND, MessageType.HEARTBEAT]
            )
        )
    )

# src/my_service/my_service.py
import asyncio
import logging
from experimance_common.base_service import BaseService
from experimance_common.constants import TICK
from experimance_common.zmq.services import PubSubService
from .config import MyServiceConfig

logger = logging.getLogger(__name__)

class MyService(BaseService):
    def __init__(self, config: MyServiceConfig):
        super().__init__(service_name=config.service_name)
        
        # Store immutable config
        self.config: MyServiceConfig = config
        
        # Copy values that need to be mutable during runtime
        self.retry_count = 0                   # Runtime state
        self.connection_attempts = 0           # Runtime counters
        self.last_heartbeat = None             # Runtime timestamps
        
        # Create ZMQ service with config
        self.zmq_service = PubSubService(config.zmq)
        
    async def start(self):
        """Initialize resources before starting."""
        # Set up message handlers before starting ZMQ service
        self.zmq_service.add_message_handler("heartbeat", self._handle_heartbeat)
        self.zmq_service.add_message_handler("commands", self._handle_command)
        
        # Start ZMQ service first
        await self.zmq_service.start()
        
        # Initialize your resources here
        self.my_resource = await create_resource()
        
        # Register background tasks
        self.add_task(self.main_work_loop())
        
        # ALWAYS call super().start() LAST
        await super().start()
        
    async def stop(self):
        """Clean up resources after stopping."""
        # ALWAYS call super().stop() FIRST
        await super().stop()
        
        # Stop ZMQ service
        await self.zmq_service.stop()
        
        # Clean up your resources
        if hasattr(self, 'my_resource'):
            await self.my_resource.close()
            
    async def _handle_heartbeat(self, message_data):
        """Handle heartbeat messages."""
        logger.info(f"Received heartbeat from {message_data.get('service', 'unknown')}")
        
    async def _handle_command(self, message_data):
        """Handle command messages."""
        command = message_data.get('command', 'unknown')
        logger.info(f"Received command: {command}")
        
    async def main_work_loop(self):
        """Main service logic."""
        while self.running:
            try:
                await self.do_work()
                
                # Publish status update
                status = {
                    "type": "status",
                    "service": self.service_name,
                    "state": self.state.value
                }
                await self.zmq_service.publish(status, "status")
                
            except RetryableError as e:
                self.record_error(e, is_fatal=False)
                await self._sleep_if_running(1.0)
            except FatalError as e:
                self.record_error(e, is_fatal=True)
                break
                
            await self._sleep_if_running(TICK)  # Prevent CPU spinning
```

### Essential Config Template

```python
# src/my_service/config.py
from experimance_common.schemas import MessageType
from experimance_common.config import BaseServiceConfig
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PUBLISHER_PORT, ZMQ_BIND_ADDRESS
from pydantic import Field

class MyServiceConfig(BaseServiceConfig):
    """Configuration for MyService."""
    
    # Override service name with sensible default
    service_name: str = "my-service"
    
    # Service-specific settings
    work_interval: float = Field(default=1.0, description="Interval between work cycles")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # ZMQ configuration using shared patterns
    zmq: PubSubServiceConfig = Field(
        default_factory=lambda: PubSubServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT
                topics=[MessageType.STATUS, MessageType.HEARTBEAT]
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT,
                topics=[MessageType.COMMAND, MessageType.HEARTBEAT]
            )
        )
    )
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
- **Message Handling**: Message processing patterns for ZMQ services.
- **Statistics Tracking**: Basic statistics like messages sent/received and uptime.
- **Adaptive Logging**: Automatic logging configuration with environment-aware file locations.
- **Error Handling**: Comprehensive error handling with automatic shutdown for fatal errors.
- **State Management**: Consistent service lifecycle state handling across inheritance hierarchies.
- **Centralized Configuration**: Pydantic-based config system with TOML support and validation.

## Logging System

### Adaptive Logging with BaseService

**All services automatically get properly configured logging** - no manual setup required!

The `BaseService` class automatically configures logging using the adaptive logging system:

```python
# In your service class, just get the logger:
import logging
logger = logging.getLogger(__name__)

# No setup needed - BaseService handles everything!
```

### Logging Behavior

**Development Environment:**
- Logs to `logs/service_name.log` (local directory)
- **Includes console output** for debugging
- Easy to access and version control

**Production Environment:**
- Logs to `/var/log/experimance/service_name.log`
- **File-only logging** (no console output)
- Follows Linux FHS standards
- Automatic log rotation via system logrotate
- Centralized monitoring friendly
- Systemd captures any stdout/stderr separately in journald

### Environment Detection

The system automatically detects the environment:
- **Production**: Running as root, `EXPERIMANCE_ENVIRONMENT=production`, or `/etc/experimance` exists
- **Development**: All other cases

### External Library Logging

External libraries (httpx, PIL, etc.) are automatically configured to reduce noise:
- Set to WARNING level by default
- Prevents excessive debug messages
- Keeps logs clean and focused

### Manual Logging Setup (Advanced)

If you need custom logging configuration:

```python
from experimance_common.logger import setup_logging

# Custom logging setup
logger = setup_logging(
    name=__name__,
    log_filename="custom.log",
    level=logging.DEBUG,
    include_console=False,  # File only
    external_level=logging.ERROR  # Even less external noise
)
```

### Log File Locations

Check where your logs are being written:

```python
from experimance_common.logger import get_log_file_path

log_path = get_log_file_path("my_service.log")
print(f"Logs are written to: {log_path}")
```

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
# In your service file:
class MyService(BaseService):
    def __init__(self, config: MyServiceConfig):
        super().__init__(service_name=config.service_name)
        self.config = config
        self.zmq_service = PubSubService(config.zmq)

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

### Modern Service Configuration Patterns

Based on recent refactoring work, here are the established patterns for creating maintainable, testable service configurations:

#### 1. Use BaseServiceConfig for Common Fields

All service configs should extend `BaseServiceConfig` to inherit common service fields:

```python
# src/my_service/config.py
from experimance_common.schemas import MessageType
from experimance_common.config import BaseServiceConfig
from experimance_common.zmq.config import PubSubServiceConfig
from experimance_common.constants import DEFAULT_PUBLISHER_PORT, ZMQ_BIND_ADDRESS
from pydantic import Field

class MyServiceConfig(BaseServiceConfig):
    """Configuration for MyService."""
    
    # Override the service name with a sensible default
    service_name: str = "my-service"
    
    # Service-specific fields
    work_interval: float = Field(default=1.0, description="Work loop interval")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # ZMQ configuration using shared patterns
    zmq: PubSubServiceConfig = Field(
        default_factory=lambda: PubSubServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT + 10,  # Offset for your service
                topics=[MessageType.STATUS, MessageType.HEARTBEAT]
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT,
                topics=[MessageType.COMMAND, MessageType.HEARTBEAT]
            )
        )
    )
```

#### 2. Use Shared Constants and Enums

Always use shared constants and enums instead of hardcoded strings:

```python
# ✅ Good - Uses shared constants and enums
from experimance_common.constants import ZMQ_BIND_ADDRESS, DEFAULT_PUBLISHER_PORT
from experimance_common.schemas import MessageType

zmq_config = PubSubServiceConfig(
    publisher=PublisherConfig(
        address=ZMQ_BIND_ADDRESS,
        port=DEFAULT_PUBLISHER_PORT + 10,
        topics=[MessageType.STATUS, MessageType.HEARTBEAT]
    )
)

# ❌ Bad - Hardcoded values and strings
zmq_config = PubSubServiceConfig(
    publisher=PublisherConfig(
        address="tcp://*",
        port=5555,
        topics=["status", "heartbeat"]  # String literals
    )
)
```

#### 3. Direct Config Instantiation Pattern

Use direct config instantiation, not factory functions:

```python
# ✅ Good - Direct instantiation
class MyService(BaseService):
    def __init__(self, config: MyServiceConfig):
        super().__init__(service_name=config.service_name)
        self.config = config
        self.zmq_service = PubSubService(config.zmq)

# Usage
config = MyServiceConfig(service_name="custom-service")
service = MyService(config)

# ❌ Bad - Factory pattern (deprecated)
def create_service_config(service_name: str) -> MyServiceConfig:
    # Don't do this anymore
    pass
```

#### 4. Field Override Patterns

Use simple field overrides in config classes:

```python
class AudioServiceConfig(BaseServiceConfig):
    # Simple field override with sensible default
    service_name: str = "audio-service"
    
    # Service-specific fields with validation
    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    buffer_size: int = Field(default=1024, ge=64, le=8192)

# Can still be overridden at instantiation
config = AudioServiceConfig(service_name="audio-service-custom")
```

#### 5. Dynamic Service Naming

Handle dynamic service naming at runtime, not in config:

```python
class MyService(BaseService):
    def __init__(self, config: MyServiceConfig):
        # Dynamic naming happens here, not in config
        service_name = config.service_name
        if config.instance_id:
            service_name = f"{service_name}-{config.instance_id}"
            
        super().__init__(service_name=service_name)
        self.config = config
        
        # ZMQ uses original config structure
        self.zmq_service = PubSubService(config.zmq)
```

#### 6. Testing with Config Overrides

Write tests that verify both default and custom configurations:

```python
# tests/test_my_service_config.py
import pytest
from my_service.config import MyServiceConfig
from my_service.my_service import MyService

class TestMyServiceConfig:
    def test_default_config(self):
        """Test service with default configuration."""
        config = MyServiceConfig()
        assert config.service_name == "my-service"
        assert config.work_interval == 1.0
        
        service = MyService(config)
        assert service.service_name == "my-service"
    
    def test_custom_config(self):
        """Test service with custom configuration."""
        config = MyServiceConfig(
            service_name="custom-service",
            work_interval=0.5
        )
        assert config.service_name == "custom-service"
        assert config.work_interval == 0.5
        
        service = MyService(config)
        assert service.service_name == "custom-service"
    
    def test_zmq_config_uses_enums(self):
        """Test that ZMQ config uses shared enums."""
        config = MyServiceConfig()
        
        # Check that topics use MessageType enum, not strings
        from experimance_common.schemas import MessageType
        assert MessageType.STATUS in config.zmq.publisher.topics
        assert MessageType.HEARTBEAT in config.zmq.publisher.topics
        assert MessageType.COMMAND in config.zmq.subscriber.topics
```

## Project-Aware Service Configuration

**New: Project-Specific Config Paths**

All services use a project-aware configuration system. The recommended pattern is:

```python
from experimance_common.constants import get_project_config_path, CORE_SERVICE_DIR
DEFAULT_CONFIG_PATH = get_project_config_path("core", CORE_SERVICE_DIR)
```

- Config files should live in `projects/{PROJECT_ENV}/{service}.toml` (e.g., `projects/experimance/core.toml`)
- The helper function `get_project_config_path(service_name, fallback_dir)` will:
  1. Use the project-specific config if `PROJECT_ENV` is set and the file exists
  2. Fall back to the legacy `<SERVICE>_DIR/config.toml` if the project config does not exist
  3. Always default to the project config path if `PROJECT_ENV` is set (even if the file doesn't exist yet)
  4. If `PROJECT_ENV` is not set or is empty, use the legacy path

**Example:**
```python
# In src/core/config.py
from experimance_common.constants import get_project_config_path, CORE_SERVICE_DIR
DEFAULT_CONFIG_PATH = get_project_config_path("core", CORE_SERVICE_DIR)
```

**Behavior:**
- When `PROJECT_ENV=experimance`, config path is `projects/experimance/core.toml`
- When `PROJECT_ENV=sohkepayin`, config path is `projects/sohkepayin/core.toml`
- If the project config does not exist, falls back to `services/core/config.toml`
- If `PROJECT_ENV` is not set, uses legacy path only

**Why?**
- Supports multi-project architecture
- Keeps project-specific configs isolated
- Maintains backward compatibility
- Works with the centralized config system and CLI

**Migration:**
- Move service configs to `projects/{project}/{service}.toml`
- Update service config files to use `get_project_config_path()`
- Legacy configs will still work as fallback

---

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

### Auto-Generated CLI Arguments

The CLI system can automatically generate command line arguments from your Pydantic config classes, making all config fields accessible via CLI:

```python
# src/my_service/__main__.py
"""
Command line entry point for My Service.
"""
from experimance_common.cli import create_simple_main
from .my_service import run_my_service
from .config import MyServiceConfig, DEFAULT_CONFIG_PATH

# Create the main function with auto-generated CLI args
main = create_simple_main(
    service_name="My Service",
    description="Brief description of what your service does",
    service_runner=run_my_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=MyServiceConfig  # Auto-generates CLI args from this class
)

if __name__ == "__main__":
    main()
```

**What gets auto-generated:**
- Top-level config fields: `--visualize`, `--service-name`
- Nested config fields: `--camera-fps`, `--state-machine-idle-timeout`
- Boolean fields with defaults: `--debug-mode` or `--no-debug-mode`
- Type-appropriate metavars: `N` for ints, `VALUE` for floats, `TEXT` for strings
- Help text with section context: `[Camera] Camera frames per second (default: 30)`

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

### Service Runner Pattern for Enhanced CLI

When using the auto-generated CLI arguments, your service runner should accept CLI arguments:

```python
# src/my_service/my_service.py
import argparse

async def run_my_service(config_path: str = DEFAULT_CONFIG_PATH, args:Optional[argparse.Namespace] = None):
    """
    Run My Service with CLI integration.
    
    Args:
        config_path: Path to configuration file
        args: CLI arguments from argparse (for config overrides)
    """
    # Create config with CLI overrides
    config = MyServiceConfig.from_overrides(
        config_file=config_path,
        args=args  # CLI args automatically override config values
    )
    
    service = MyService(config=config)
    await service.start()
    await service.run()
```

**Key Benefits:**
- All config fields automatically become CLI arguments
- Nested configs work: `--camera-fps 60` sets `config.camera.fps = 60`
- TOML + CLI + defaults priority system: CLI > TOML > Pydantic defaults
- Clean help text with section context and type hints
- Consistent CLI interface across all services

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

#### Quick Health Status Checks
```python
# ✅ GOOD: Use the helper method for quick health status checks
from experimance_common.health import HealthStatus

# Simple status check - returns HealthStatus enum directly
current_status = service.get_overall_health_status()
if current_status == HealthStatus.HEALTHY:
    logger.info("Service is healthy")

# Perfect for tests and assertions
assert service.get_overall_health_status() == HealthStatus.ERROR

# No need for verbose: HealthStatus(service.get_health_status()["overall_status"])
```

### Common Shutdown Patterns

#### From Service Tasks
```python
async def main_work_loop(self):
    while self.running:
        try:
            await do_work()
            # Check for shutdown condition
            if check_shutdown_condition():
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
    self.add_task(self.periodic_cleanup())
    
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
        await asyncio.sleep(TICK)  # Small delay to prevent CPU spinning

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

#### Test Error Handling
```python
async def test_error_handling():
    service = MyService("test")
    await service.start()
    
    # Test non-fatal error
    service.record_error(ValueError("test"), is_fatal=False)
    assert service.get_overall_health_status() == HealthStatus.ERROR
    assert service.state == ServiceState.RUNNING  # Still running
    
    # Test fatal error
    service.record_error(RuntimeError("fatal"), is_fatal=True)
    assert service.get_overall_health_status() == HealthStatus.FATAL
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

When building custom services by extending `BaseService` and using ZMQ composition, the state management system works automatically. The service moves through the proper state transitions during startup, execution, and shutdown without requiring any additional code.

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

### 2. ZMQ Integration (Composition-Based)

Instead of inheritance-based ZMQ services, use **composition** with `BaseService` and ZMQ components:

**For messaging services**, use `BaseService` + ZMQ service composition:

```python
from experimance_common.base_service import BaseService
from experimance_common.zmq.config import PubSubServiceConfig
from experimance_common.zmq.services import PubSubService

class MyMessagingService(BaseService):
    def __init__(self):
        super().__init__("my-service", "messaging")
        
        # Create ZMQ configuration
        self.zmq_config = PubSubServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT + 10,
                topics=[MessageType.STATUS, MessageType.HEARTBEAT]
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PUBLISHER_PORT,
                topics=[MessageType.COMMAND, MessageType.HEARTBEAT]
            )
        )
        
        # Use composition, not inheritance
        self.zmq_service = PubSubService(self.zmq_config)
    
    async def start(self):
        # Set up message handlers
        self.zmq_service.add_message_handler("heartbeat", self._handle_heartbeat)
        self.zmq_service.add_message_handler("commands", self._handle_command)
        
        # Start ZMQ service
        await self.zmq_service.start()
        
        # Register background tasks
        self.add_task(self._publishing_loop())
        
        # Call BaseService start last
        await super().start()
        
        # Record successful service start
        self.record_health_check("service_start", HealthStatus.HEALTHY, "Service started successfully")
        
    async def stop(self):
        # Call BaseService stop first
        await super().stop()
        
        # Stop ZMQ service
        await self.zmq_service.stop()
        
    async def _handle_heartbeat(self, message_data):
        """Handle heartbeat messages."""
        service = message_data.get("service", "unknown")
        logger.info(f"Received heartbeat from {service}")
        
    async def _handle_command(self, message_data):
        """Handle command messages."""
        command = message_data.get("command", "unknown")
        logger.info(f"Processing command: {command}")
        
    async def _publishing_loop(self):
        """Publish periodic status updates."""
        while self.running:
            status = {
                "type": "status",
                "service": self.service_name,
                "state": self.state.value,
                "uptime": self.uptime
            }
            await self.zmq_service.publish(status, "status")
            await self._sleep_if_running(5.0)
```

### Worker Pattern (Task Processing)
```python
from experimance_common.zmq.config import WorkerServiceConfig, PullConfig, PushConfig
from experimance_common.zmq.services import WorkerService

class ProcessingWorker(BaseService):
    def __init__(self):
        super().__init__("processing-worker", "worker")
        
        self.zmq_config = WorkerServiceConfig(
            name=self.service_name,
            pull=PullConfig(
                address="tcp://localhost",
                port=5557  # Receive work from controller
            ),
            push=PushConfig(
                address="tcp://*",
                port=5558  # Send results to controller
            )
        )
        self.zmq_service = WorkerService(self.zmq_config)
    
    async def start(self):
        # Set up task handler
        self.zmq_service.set_task_handler(self._process_task)
        
        await self.zmq_service.start()
        await super().start()
        
    async def _process_task(self, task_data):
        # Process the work
        result = await self.do_work(task_data)
        
        # Send result back
        await self.zmq_service.push_result(result)
```

### Controller Pattern (Task Distribution)
```python
from experimance_common.zmq.config import ControllerServiceConfig, WorkerConfig
from experimance_common.zmq.services import ControllerService

class TaskController(BaseService):
    def __init__(self):
        super().__init__("task-controller", "controller")
        
        self.zmq_config = ControllerServiceConfig(
            name=self.service_name,
            publisher=PublisherConfig(
                address="tcp://*",
                port=5555,
                default_topic="control"
            ),
            workers=[
                WorkerConfig(
                    name="image-worker",
                    push_port=5557,  # Send work to worker
                    pull_port=5558   # Receive results from worker
                )
            ]
        )
        self.zmq_service = ControllerService(self.zmq_config)
    
    async def start(self):
        # Set up result handler
        self.zmq_service.set_result_handler(self._handle_result)
        
        await self.zmq_service.start()
        
        # Add task distribution logic
        self.add_task(self._distribute_tasks())
        
        await super().start()
        
    async def _distribute_tasks(self):
        while self.running:
            task = await self.create_task()
            await self.zmq_service.push_task("image-worker", task)
            await self._sleep_if_running(1.0)
            
    async def _handle_result(self, worker_name: str, result_data):
        logger.info(f"Received result from {worker_name}: {result_data}")
```

## Usage

To create a new service using the composition-based approach:

1. **Inherit from `BaseService`** for lifecycle management
2. **Choose the appropriate ZMQ service type** for your communication pattern:
   - `PubSubService` for bidirectional messaging
   - `WorkerService` for task processing  
   - `ControllerService` for task distribution
3. **Create ZMQ configuration** using the config classes
4. **Implement message/task handlers** as async methods
5. **Set up ZMQ service in `start()`** method before calling `super().start()`
6. **Add background tasks** using `self.add_task()`
7. **Ensure proper cleanup** in `stop()` method

### Example: Modern Service with ZMQ

```python
# In your service module (e.g., my_service.py)
import asyncio
import logging
from experimance_common.base_service import BaseService
from experimance_common.health import HealthStatus
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.zmq.services import PubSubService
from experimance_common.constants import DEFAULT_PORTS

logger = logging.getLogger(__name__)

class MyModernService(BaseService):
    def __init__(self):
        super().__init__("my-modern-service", "messaging")
        
        # Create ZMQ configuration
        self.zmq_config = PubSubServiceConfig(
            name=self.service_name,
            publisher=PublisherConfig(
                address=ZMQ_BIND_ADDRESS,
                port=DEFAULT_PORTS["events"],
                default_topic="status"
            ),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=DEFAULT_PORTS["events"],
                topics=["heartbeat", "commands"]
            )
        )
        
        # Use composition, not inheritance
        self.zmq_service = PubSubService(self.zmq_config)
        
    async def start(self):
        # Set up message handlers before starting ZMQ
        self.zmq_service.add_message_handler("heartbeat", self._handle_heartbeat)
        self.zmq_service.add_message_handler("commands", self._handle_command)
        
        # Start ZMQ service first
        await self.zmq_service.start()
        
        # Add background tasks
        self.add_task(self._status_publisher())
        
        # Call BaseService start last
        await super().start()
        
        # Record successful service start
        self.record_health_check("service_start", HealthStatus.HEALTHY, "Service started successfully")
        
    async def stop(self):
        # Call BaseService stop first
        await super().stop()
        
        # Stop ZMQ service
        await self.zmq_service.stop()
        
    async def _handle_heartbeat(self, message_data):
        """Handle heartbeat messages."""
        service = message_data.get("service", "unknown")
        logger.info(f"Received heartbeat from {service}")
        
    async def _handle_command(self, message_data):
        """Handle command messages."""
        command = message_data.get("command", "unknown")
        logger.info(f"Processing command: {command}")
        
    async def _status_publisher(self):
        """Publish periodic status updates."""
        while self.running:
            status = {
                "type": "status",
                "service": self.service_name,
                "state": self.state.value,
                "uptime": self.uptime
            }
            await self.zmq_service.publish(status, "status")
            await self._sleep_if_running(5.0)
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


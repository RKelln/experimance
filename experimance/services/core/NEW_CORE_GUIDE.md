# New Core Service Development Guide

This guide helps you create a new core service for the Experimance project ecosystem, 
similar to the existing Experimance and Feed the Fires core services. 
Follow these steps to build a robust, well-integrated service.

## Overview

A core service is the central orchestrator for an interactive art installation. It typically:
- Receives input from sensors or from agent services (stories, user interactions, etc.)
- Processes and analyzes input using LLMs or other AI services
- Generates image prompts and coordinates with image generation services
- Manages state transitions and application logic
- Sends display commands to visualization services

## Prerequisites

- Familiarity with Python async/await patterns
- Understanding of ZeroMQ message patterns (PUB/SUB, PUSH/PULL)
- Basic knowledge of Pydantic models and configuration management
- Familiarity with the Experimance project structure

## Step-by-Step Implementation

### 1. Project Setup

#### 1.1 Create Project Directory Structure
```
projects/your_project_name/
├── .env                    # Environment variables
├── config.toml            # Project configuration  
├── core.toml              # Core service configuration
├── constants.py           # Project-specific constants
├── schemas.py             # Project-specific message schemas
└── schemas.pyi            # Type stubs for schemas
```

#### 1.2 Create Service Directory Structure
```
services/core/src/your_project_core/
├── __init__.py
├── __main__.py            # CLI entry point
├── config.py              # Service configuration models
├── your_project_core.py   # Main service implementation
├── llm.py                 # LLM integration (if needed)
├── prompt_builder.py      # Prompt generation (if needed)
└── [other_modules].py     # Additional functionality
```

### 2. Define Project-Specific Schemas

**File: `projects/your_project_name/schemas.py`**

```python
"""
Your Project specific schema extensions and overrides.
"""

from enum import Enum
from typing import Optional, List
from experimance_common.schemas_base import (
    StringComparableEnum, 
    SpaceTimeUpdate as _BaseSpaceTimeUpdate,
    RenderRequest as _BaseRenderRequest,
    ImageReady as _BaseImageReady,
    DisplayMedia as _BaseDisplayMedia,
    MessageBase,
    MessageType as _BaseMessageType,
    ContentType
)

# Define project-specific enums
class YourEnum(StringComparableEnum):
    """Project-specific enum values."""
    VALUE1 = "value1"
    VALUE2 = "value2"

# Extend MessageType with project-specific message types
class MessageType(StringComparableEnum):
    """COMPLETE redeclaration - Python enums cannot be extended."""
    # Base Experimance message types (copy from base)
    SPACE_TIME_UPDATE = "SpaceTimeUpdate"
    RENDER_REQUEST = "RenderRequest"
    # ... (copy all base types)
    
    # Your project-specific message types
    YOUR_NEW_MESSAGE = "YourNewMessage"

# Extend base message types
class RenderRequest(_BaseRenderRequest):
    """Project-specific RenderRequest extensions."""
    your_field: Optional[str] = None

# Define new message types
class YourNewMessage(MessageBase):
    """Your custom message type."""
    type: MessageType = MessageType.YOUR_NEW_MESSAGE
    content: str
    timestamp: Optional[str] = None
```

**⚠️ Important Schema Tips:**
- **Complete MessageType Redeclaration**: Python enums cannot be extended. You MUST redeclare the entire MessageType enum with all base types plus your new ones.
- **Import from schemas_base**: Always import base classes from `experimance_common.schemas_base`
- **Keep DisplayMedia Simple**: Don't add project-specific fields to DisplayMedia unless absolutely necessary

### 3. Create Service Configuration

**File: `services/core/src/your_project_core/config.py`**

```python
from pydantic import BaseModel, Field
from experimance_common.config import BaseServiceConfig
from experimance_common.constants import DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX
from experimance_common.zmq.config import (
    ControllerServiceConfig, WorkerConfig, PublisherConfig, 
    SubscriberConfig, ControllerPushConfig, ControllerPullConfig, MessageType
)

class YourProjectConfig(BaseModel):
    """Project-specific configuration section."""
    param1: str = Field(default="default_value", description="Description")
    param2: int = Field(default=42, description="Another parameter")

class YourProjectCoreConfig(BaseServiceConfig):
    """Main service configuration."""
    
    service_name: str = Field(
        default="your_project_core",
        description="Service instance name"
    )
    
    # Add your project-specific config sections
    your_project: YourProjectConfig = Field(
        default_factory=YourProjectConfig,
        description="Project-specific configuration"
    )
    
    # ZMQ configuration using ControllerService pattern
    zmq: ControllerServiceConfig = Field(
        default_factory=lambda: ControllerServiceConfig(
            name="your-project-core",
            publisher=PublisherConfig(
                address=ZMQ_TCP_BIND_PREFIX,
                port=DEFAULT_PORTS["events"],
                default_topic=MessageType.DISPLAY_MEDIA
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS["agent"],
                topics=[MessageType.YOUR_NEW_MESSAGE]  # Subscribe to relevant messages
            ),
            workers={
                "image_server": WorkerConfig(
                    name="image_server",
                    push_config=ControllerPushConfig(
                        port=DEFAULT_PORTS["image_requests"]
                    ),
                    pull_config=ControllerPullConfig(
                        port=DEFAULT_PORTS["image_results"]
                    )
                ),
            }
        ),
        description="ZMQ communication configuration"
    )
```

**⚠️ Configuration Tips:**
- **Use ControllerService**: For services that need to coordinate with workers (like image_server)
- **Proper Port Configuration**: Use DEFAULT_PORTS constants for standard services
- **Worker Names**: Use "image_server" as the worker name to match the image generation service

### 4. Implement Core Service

**File: `services/core/src/your_project_core/your_project_core.py`**

```python
import asyncio
import logging
from enum import Enum
from typing import Optional, Dict
from dataclasses import dataclass

from experimance_common.base_service import BaseService
from experimance_common.zmq.services import ControllerService  
from experimance_common.schemas import (
    YourNewMessage, ImageReady, RenderRequest, DisplayMedia, 
    ContentType, MessageType
)

from .config import YourProjectCoreConfig

logger = logging.getLogger(__name__)

class CoreState(Enum):
    """Application-level states (separate from service lifecycle)."""
    IDLE = "idle"
    PROCESSING = "processing"
    GENERATING = "generating"

class YourProjectCoreService(BaseService):
    """Core service for Your Project installation."""
    
    def __init__(self, config: YourProjectCoreConfig):
        """Initialize the service."""
        super().__init__(config)
        
        self.config = config
        self.core_state = CoreState.IDLE  # Use separate variable for app state
        
        # Initialize components
        # self.your_component = YourComponent(config.your_project)
        
        # ZMQ communication will be initialized in start()
        self.zmq_service: ControllerService = None  # type: ignore
        
        logger.info("Your Project core service initialized")
    
    async def start(self):
        """Start the service and initialize ZMQ communication."""
        logger.info("Starting Your Project core service")
        
        # Initialize ZMQ service
        self.zmq_service = ControllerService(self.config.zmq)
        
        # Set up message handlers
        self.zmq_service.add_message_handler(MessageType.YOUR_NEW_MESSAGE, self._handle_your_message)
        
        # Set up worker response handler for image results
        self.zmq_service.add_response_handler(self._handle_worker_response)
        
        # Add periodic tasks
        self.add_task(self._periodic_task())
        
        # Start ZMQ service
        await self.zmq_service.start()
        
        # Call parent start LAST
        await super().start()
        
        logger.info("Your Project core service started")
    
    async def stop(self):
        """Stop the service."""
        logger.info("Stopping Your Project core service")
        
        if self.zmq_service:
            await self.zmq_service.stop()
        
        await super().stop()
    
    # Message handlers
    async def _handle_your_message(self, topic: str, data: Dict):
        """Handle your custom message type."""
        try:
            message = YourNewMessage(**data)
            logger.info(f"Received message: {message.content}")
            
            # Process the message and transition states
            await self._transition_to_state(CoreState.PROCESSING)
            
            # Example: Generate an image
            await self._request_image_generation(message.content)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_worker_response(self, worker_name: str, response_data: Dict):
        """Handle responses from workers (e.g., image_server)."""
        try:
            if worker_name == "image_server":
                image_ready = ImageReady(**response_data)
                await self._handle_image_ready(image_ready)
        except Exception as e:
            logger.error(f"Error handling worker response: {e}")
    
    async def _handle_image_ready(self, image_ready: ImageReady):
        """Handle completed image generation."""
        logger.info(f"Image ready: {image_ready.uri}")
        
        # Send to display service
        display_message = DisplayMedia(
            content_type=ContentType.IMAGE,
            uri=image_ready.uri
        )
        
        await self.zmq_service.publish(display_message)
        
        # Transition back to idle
        await self._transition_to_state(CoreState.IDLE)
    
    # Helper methods
    async def _request_image_generation(self, prompt: str):
        """Request image generation from image_server."""
        render_request = RenderRequest(
            prompt=prompt,
            width=1920,
            height=1080
        )
        
        await self._transition_to_state(CoreState.GENERATING)
        await self.zmq_service.send_work_to_worker("image_server", render_request)
        logger.info("Requested image generation")
    
    async def _transition_to_state(self, new_state: CoreState):
        """Transition to a new application state."""
        old_state = self.core_state
        self.core_state = new_state
        logger.info(f"State transition: {old_state.value} → {new_state.value}")
    
    async def _periodic_task(self):
        """Periodic task."""
        while self.running:
            logger.debug(f"State: {self.core_state.value}")
            await self._sleep_if_running(self.config.my_interval)

# Entry point function
async def run_your_project_core_service(
    config_path: str = None, 
    args: Optional[argparse.Namespace] = None
):
    """Run the Your Project Core Service."""
    import argparse
    from pathlib import Path
    
    # Use project-aware config loading
    config = YourProjectCoreConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    service = YourProjectCoreService(config)
    
    await service.start()
    await service.run()
```

**⚠️ Service Implementation Tips:**
- **Separate Application State**: Use `self.core_state` for your application logic, not `self.state` (which is for service lifecycle)
- **Type Annotation**: Use `self.zmq_service: ControllerService = None  # type: ignore` to satisfy linters
- **Message Handler Signature**: ZMQ handlers receive `(topic: str, data: Dict)` parameters
- **Worker Communication**: Use `send_work_to_worker("image_server", request)` for image generation
- **Publishing**: Use `zmq_service.publish(message)` for display messages
- **Initialization Order**: Initialize ZMQ service in `start()`, call `super().start()` LAST

### 5. Create CLI Entry Points

**File: `services/core/src/your_project_core/__main__.py`**

```python
"""CLI entry point for Your Project Core Service."""

import asyncio
from experimance_common.cli import run_service_cli
from .your_project_core import run_your_project_core_service
from .config import YourProjectCoreConfig

if __name__ == "__main__":
    asyncio.run(run_service_cli(
        service_name="Your Project Core",
        service_description="Core orchestration service for the Your Project interactive art installation",
        config_class=YourProjectCoreConfig,
        service_runner=run_your_project_core_service
    ))
```

**File: `services/core/src/your_project_core/__init__.py`**

```python
"""Your Project Core Service package."""

from .your_project_core import YourProjectCoreService, run_your_project_core_service
from .config import YourProjectCoreConfig

__all__ = ["YourProjectCoreService", "YourProjectCoreConfig", "run_your_project_core_service"]
```

### 6. Create Configuration Files

**File: `projects/your_project_name/core.toml`**

```toml
# Your Project Core Service Configuration

service_name = "your_project_core"

[your_project]
param1 = "custom_value"
param2 = 100

# Additional configuration sections as needed
[llm]
provider = "openai"
model = "gpt-4o"
max_tokens = 500
temperature = 0.7
timeout = 30.0
```

### 7. Add pyproject.toml Scripts

Add to the main `pyproject.toml`:

```toml
[project.scripts]
your-project-core = "your_project_core.__main__:main"

[tool.uv.sources]
your-project-core = { workspace = true }
```

## Common Patterns and Best Practices

### ZMQ Communication Patterns

1. **ControllerService for Coordination**: Use when you need to coordinate with multiple workers
2. **Message Handler Pattern**: 
   ```python
   self.zmq_service.add_message_handler(MessageType.YOUR_MESSAGE, self._handle_your_message)
   ```
3. **Worker Response Pattern**:
   ```python
   self.zmq_service.add_response_handler(self._handle_worker_response)
   ```

### State Management

1. **Separate Application State**: Don't use `self.state` for application logic
2. **Clear State Transitions**: Log state changes for debugging
3. **Timeout Handling**: Consider adding timeout logic for long-running states

### Error Handling

1. **Graceful Degradation**: Handle missing services gracefully
2. **Comprehensive Logging**: Log errors with context
3. **Recovery Strategies**: Implement retry logic where appropriate

### Configuration Management

1. **Project-Aware Configs**: Use the project-based configuration system
2. **Sensible Defaults**: Provide reasonable default values
3. **Validation**: Use Pydantic validators for complex constraints

## Testing Your Service

### 1. Basic Import Test
```bash
PROJECT_ENV=your_project_name uv run python -c "from services.core.src.your_project_core.your_project_core import YourProjectCoreService; print('Import successful')"
```

### 2. CLI Help Test
```bash
PROJECT_ENV=your_project_name uv run -m your_project_core --help
```

### 3. Service Startup Test
```bash
PROJECT_ENV=your_project_name uv run -m your_project_core --log-level DEBUG
```

## Common Pitfalls and Solutions

### 1. Schema Import Errors
**Problem**: `ImportError: cannot import name 'YourMessage'`
**Solution**: Ensure proper import from `experimance_common.schemas` and that the dynamic schema system is working

### 2. State Management Confusion
**Problem**: `KeyError: <YourState.IDLE: 'idle'>`
**Solution**: Use `self.core_state` for application state, not `self.state`

### 3. ZMQ Method Errors
**Problem**: `AttributeError: 'NoneType' object has no attribute 'send_work_to_worker'`
**Solution**: Initialize `zmq_service` in `start()` method, use type annotation with `# type: ignore`

### 4. Configuration Loading Issues
**Problem**: `ImportError: cannot import name 'load_config'`
**Solution**: Use `YourConfigClass.from_overrides()` instead of non-existent `load_config()`

### 5. MessageType Extension Problems
**Problem**: Cannot add new message types to enum
**Solution**: Completely redeclare MessageType enum in project schemas with all base types plus new ones

## Resources

- **Existing Examples**: Study `experimance_core.py` and `fire_core.py` for patterns
- **Common Library**: Reference `experimance_common` for available utilities
- **ZMQ Documentation**: See `libs/common/README_ZMQ.md`
- **Service Documentation**: See `libs/common/README_SERVICE.md`
- **Testing Guide**: See `libs/common/README_SERVICE_TESTING.md`

## Summary Checklist

- [ ] Created project directory structure
- [ ] Defined project-specific schemas with complete MessageType redeclaration
- [ ] Implemented service configuration with ControllerService
- [ ] Created main service class inheriting from BaseService
- [ ] Set up proper ZMQ message handlers
- [ ] Implemented state management with separate application state
- [ ] Added CLI entry points
- [ ] Created configuration files
- [ ] Tested service import and startup
- [ ] Added error handling and logging
- [ ] Documented service-specific functionality

Following this guide should help you create a robust, well-integrated core service that follows the established patterns in the Experimance ecosystem.

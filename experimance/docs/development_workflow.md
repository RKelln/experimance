# Development Workflow Guide

This guide covers best practices for developing, testing, and contributing to the Experimance project. Follow these guidelines to maintain code quality and ensure smooth collaboration.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Code Standards and Style](#code-standards-and-style)
3. [Testing Workflow](#testing-workflow)
4. [Debugging Techniques](#debugging-techniques)
5. [Git Workflow](#git-workflow)
6. [Service Development](#service-development)
7. [Performance Optimization](#performance-optimization)
8. [Documentation Guidelines](#documentation-guidelines)
9. [Release Process](#release-process)

## Development Environment Setup

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd experimance

# Install development environment
./infra/scripts/deploy.sh install experimance dev

# Install development tools
uv tool install ruff      # Code formatting and linting
uv tool install pytest    # Testing framework
uv tool install mypy      # Type checking

# Install development dependencies
uv sync --dev
```

### IDE Configuration

#### VS Code Setup

Recommended extensions:
- Python
- Pylance
- Ruff
- GitLens
- TOML Language Support

VS Code settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["."],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true
    }
}
```

#### PyCharm Setup

1. Open project in PyCharm
2. Configure Python interpreter: `.venv/bin/python`
3. Enable pytest as test runner
4. Configure Ruff as external tool for formatting

### Environment Variables

Create a development `.env` file:

```bash
# Copy example environment
cp projects/experimance/.env.example projects/experimance/.env

# Edit with your development settings
nano projects/experimance/.env
```

Common development environment variables:
```env
# Development mode
DEBUG=true
LOG_LEVEL=DEBUG

# Mock hardware for development
MOCK_DEPTH_CAMERA=true
MOCK_WEBCAM=true

# API keys for testing (use test/development keys)
OPENAI_API_KEY=your_dev_key
ANTHROPIC_API_KEY=your_dev_key
```

## Code Standards and Style

### Python Code Style

We use Ruff for code formatting and linting:

```bash
# Format code
uv run ruff format .

# Check for linting issues
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Type Hints

All public APIs must include type hints:

```python
# Good
def process_depth_data(
    depth_frame: np.ndarray,
    threshold: float = 0.5
) -> tuple[bool, list[tuple[int, int]]]:
    """Process depth data and return hand detection results."""
    # Implementation here
    return detected, hand_positions

# Bad
def process_depth_data(depth_frame, threshold=0.5):
    # No type hints
    return detected, hand_positions
```

### Error Handling

Use comprehensive error handling with recovery strategies:

```python
# Good
async def generate_image(request: RenderRequest) -> ImageReady:
    try:
        result = await self.generator.generate(request)
        return ImageReady(image_path=result.path, metadata=result.metadata)
    except GenerationError as e:
        logger.error(f"Image generation failed: {e}")
        # Try fallback generator
        try:
            result = await self.fallback_generator.generate(request)
            return ImageReady(image_path=result.path, metadata=result.metadata)
        except Exception as fallback_error:
            logger.error(f"Fallback generation also failed: {fallback_error}")
            raise GenerationError("All generation methods failed") from e
    except Exception as e:
        logger.error(f"Unexpected error in image generation: {e}")
        raise
```

### Logging

Use structured logging with appropriate levels:

```python
import logging
from experimance_common.logger import get_logger

logger = get_logger(__name__)

# Good logging practices
logger.info("Service starting", extra={"service": "core", "version": "1.0"})
logger.debug("Processing depth frame", extra={"frame_id": frame_id, "timestamp": timestamp})
logger.warning("High CPU usage detected", extra={"cpu_percent": 85.2})
logger.error("Failed to connect to camera", extra={"camera_id": camera_id}, exc_info=True)
```

### Configuration

Use Pydantic models for all configuration:

```python
from pydantic import BaseModel, Field
from typing import Optional

class CameraConfig(BaseModel):
    """Configuration for depth camera."""
    
    device_id: int = Field(default=0, description="Camera device ID")
    resolution: tuple[int, int] = Field(default=(640, 480), description="Camera resolution")
    fps: int = Field(default=30, ge=1, le=60, description="Frames per second")
    enable_color: bool = Field(default=False, description="Enable color stream")
    
    class Config:
        extra = "forbid"  # Prevent typos in config files
```

## Testing Workflow

### Test Structure

Organize tests by service and functionality:

```
tests/
├── unit/                    # Unit tests
│   ├── test_core/
│   ├── test_display/
│   └── test_common/
├── integration/             # Integration tests
│   ├── test_zmq_communication/
│   └── test_service_coordination/
├── e2e/                     # End-to-end tests
│   └── test_full_system/
└── fixtures/                # Test data and fixtures
    ├── depth_images/
    └── config_files/
```

### Running Tests

```bash
# Run all tests
uv run -m pytest

# Run specific test categories
uv run -m pytest tests/unit/
uv run -m pytest tests/integration/
uv run -m pytest tests/e2e/

# Run tests for specific service
uv run -m pytest services/core/tests/

# Run with coverage
uv run -m pytest --cov=experimance_common --cov-report=html

# Run specific test
uv run -m pytest tests/unit/test_core/test_state_machine.py::test_era_progression

# Run tests with verbose output
uv run -m pytest -v -s
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from experimance_core.state_machine import StateMachine
from experimance_common.schemas import Era, Biome

class TestStateMachine:
    @pytest.fixture
    def state_machine(self):
        """Create a state machine for testing."""
        return StateMachine()
    
    def test_era_progression(self, state_machine):
        """Test that era progresses correctly with interaction."""
        # Arrange
        initial_era = state_machine.current_era
        
        # Act
        state_machine.process_interaction(interaction_strength=0.8)
        
        # Assert
        assert state_machine.current_era != initial_era
        assert state_machine.interaction_count > 0
    
    @patch('experimance_core.camera.RealSenseCamera')
    def test_mock_camera_integration(self, mock_camera, state_machine):
        """Test state machine with mocked camera."""
        # Arrange
        mock_camera.return_value.get_depth_frame.return_value = Mock()
        
        # Act & Assert
        # Test implementation here
```

#### Integration Tests

```python
import pytest
import asyncio
from experimance_common.zmq.mocks import create_mock_zmq_service

@pytest.mark.asyncio
async def test_service_communication():
    """Test communication between services."""
    # Arrange
    core_service = create_mock_core_service()
    display_service = create_mock_display_service()
    
    # Act
    await core_service.publish_event(SpaceTimeUpdate(era=Era.MODERN, biome=Biome.URBAN))
    await asyncio.sleep(0.1)  # Allow message processing
    
    # Assert
    assert display_service.received_events
    assert display_service.received_events[-1].era == Era.MODERN
```

### Test Data Management

Use fixtures for consistent test data:

```python
@pytest.fixture
def sample_depth_frame():
    """Provide sample depth frame for testing."""
    return np.random.randint(0, 1000, (480, 640), dtype=np.uint16)

@pytest.fixture
def sample_render_request():
    """Provide sample render request for testing."""
    return RenderRequest(
        era=Era.MODERN,
        biome=Biome.URBAN,
        prompt="A modern city at sunset"
    )
```

## Debugging Techniques

### Service Debugging

#### Enable Verbose Logging

```bash
# Enable debug logging for all services
export LOG_LEVEL=DEBUG

# Enable verbose mode for specific service
uv run -m experimance_core --verbose

# Enable performance monitoring
uv run -m experimance_core --performance
```

#### Visual Debugging

```bash
# Enable camera visualization
uv run -m experimance_core --visualize

# Enable depth camera debug output
uv run -m experimance_core --camera-debug-depth
```

#### Mock Mode for Isolation

```bash
# Run with mock hardware
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present

# Disable external dependencies
uv run -m experimance_agent \
  --no-vision-webcam_enabled \
  --no-vision-audience_detection_enabled
```

### ZMQ Communication Debugging

```bash
# Test ZMQ communication
uv run python utils/tests/test_zmq_utils.py

# Monitor ZMQ traffic
uv run python utils/examples/zmq_monitor.py

# Check port usage
netstat -ln | grep 555
```

### Performance Debugging

```bash
# Profile CPU usage
uv run python -m cProfile -o profile.stats -m experimance_core

# Analyze profile results
uv run python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Monitor memory usage
uv run python -m memory_profiler -m experimance_core

# Monitor GPU usage (if applicable)
watch -n 1 nvidia-smi
```

### Interactive Debugging

Use Python debugger for step-through debugging:

```python
import pdb

def problematic_function():
    # Set breakpoint
    pdb.set_trace()
    
    # Your code here
    result = some_complex_operation()
    
    return result
```

Or use IPython for enhanced debugging:

```python
from IPython import embed

def debug_point():
    # Drop into IPython shell
    embed()
```

## Git Workflow

### Branch Strategy

We use a feature branch workflow:

```bash
# Create feature branch
git checkout -b feature/new-interaction-detection

# Work on feature
git add .
git commit -m "Add hand gesture recognition"

# Push feature branch
git push origin feature/new-interaction-detection

# Create pull request
# (Use GitHub/GitLab interface)
```

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(core): add hand gesture recognition
fix(display): resolve shader compilation error
docs(readme): update installation instructions
test(agent): add vision processing tests
refactor(zmq): simplify message handling
```

### Pre-commit Hooks

Set up pre-commit hooks for code quality:

```bash
# Install pre-commit
uv add --dev pre-commit

# Set up hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Example `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Service Development

### Creating a New Service

1. **Create service structure**:
```bash
mkdir -p services/new_service/src/new_service
mkdir -p services/new_service/tests
```

2. **Create service class**:
```python
# services/new_service/src/new_service/service.py
from experimance_common.services import BaseService
from experimance_common.config import BaseServiceConfig

class NewServiceConfig(BaseServiceConfig):
    """Configuration for new service."""
    custom_setting: str = "default_value"

class NewService(BaseService[NewServiceConfig]):
    """New service implementation."""
    
    def __init__(self, config: NewServiceConfig):
        super().__init__(config)
        # Service-specific initialization
    
    async def start(self):
        """Start the service."""
        await super().start()
        # Service-specific startup logic
    
    async def stop(self):
        """Stop the service."""
        # Service-specific cleanup
        await super().stop()
```

3. **Add service configuration**:
```python
# services/new_service/pyproject.toml
[project]
name = "experimance-new-service"
dependencies = [
    "experimance-common",
    # Other dependencies
]

[project.scripts]
experimance-new-service = "new_service.__main__:main"
```

4. **Create main entry point**:
```python
# services/new_service/src/new_service/__main__.py
import asyncio
from .service import NewService, NewServiceConfig

async def main():
    config = NewServiceConfig.load()
    service = NewService(config)
    
    try:
        await service.start()
        await service.run()
    except KeyboardInterrupt:
        pass
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Service Integration

1. **Add ZMQ communication**:
```python
from experimance_common.zmq.services import PubSubService

class NewService(BaseService[NewServiceConfig]):
    def __init__(self, config: NewServiceConfig):
        super().__init__(config)
        self.zmq_service = PubSubService(config.zmq)
    
    async def handle_event(self, event):
        """Handle incoming ZMQ events."""
        if isinstance(event, SpaceTimeUpdate):
            await self.process_space_time_update(event)
```

2. **Add to systemd services**:
```ini
# infra/systemd/experimance-new-service@.service
[Unit]
Description=Experimance New Service (%i)
After=network.target

[Service]
Type=simple
User=experimance
WorkingDirectory=/home/experimance/experimance
ExecStart=/home/experimance/experimance/.venv/bin/python -m new_service
Environment=PROJECT_ENV=%i
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Performance Optimization

### Profiling

```bash
# Profile CPU usage
uv run python -m cProfile -o profile.stats -m experimance_core

# Profile memory usage
uv run python -m memory_profiler -m experimance_core

# Profile async code
uv run python -m asyncio_profiler -m experimance_core
```

### Optimization Strategies

1. **Async/Await Best Practices**:
```python
# Good: Use async/await for I/O operations
async def process_image(image_path: str) -> ProcessedImage:
    async with aiofiles.open(image_path, 'rb') as f:
        data = await f.read()
    
    # CPU-intensive work in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, cpu_intensive_processing, data)
    
    return result

# Bad: Blocking I/O in async function
async def process_image_bad(image_path: str) -> ProcessedImage:
    with open(image_path, 'rb') as f:  # Blocks event loop
        data = f.read()
    
    result = cpu_intensive_processing(data)  # Blocks event loop
    return result
```

2. **Memory Management**:
```python
# Use generators for large datasets
def process_large_dataset():
    for item in large_dataset:
        yield process_item(item)

# Clear references explicitly
def cleanup_resources():
    self.large_cache.clear()
    gc.collect()
```

3. **Caching Strategies**:
```python
from functools import lru_cache
import asyncio

# Cache expensive computations
@lru_cache(maxsize=128)
def expensive_computation(input_data: str) -> str:
    # Expensive operation
    return result

# Async caching
class AsyncCache:
    def __init__(self):
        self._cache = {}
        self._locks = {}
    
    async def get_or_compute(self, key: str, compute_func):
        if key in self._cache:
            return self._cache[key]
        
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            if key not in self._cache:
                self._cache[key] = await compute_func()
            return self._cache[key]
```

## Documentation Guidelines

### Code Documentation

Use clear docstrings with type information:

```python
def process_depth_frame(
    depth_frame: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100
) -> tuple[bool, list[tuple[int, int]]]:
    """
    Process depth frame to detect hand interactions.
    
    Args:
        depth_frame: Raw depth data from camera (height, width) in millimeters
        threshold: Depth threshold for interaction detection (0.0-1.0)
        min_area: Minimum area in pixels for valid hand detection
    
    Returns:
        Tuple of (interaction_detected, hand_positions)
        - interaction_detected: True if hands detected above threshold
        - hand_positions: List of (x, y) coordinates for detected hands
    
    Raises:
        ValueError: If threshold is outside valid range
        ProcessingError: If depth frame processing fails
    
    Example:
        >>> depth_frame = camera.get_depth_frame()
        >>> detected, positions = process_depth_frame(depth_frame, threshold=0.3)
        >>> if detected:
        ...     print(f"Found {len(positions)} hands")
    """
```

### README Updates

Keep service READMEs current:

1. **Update when adding features**
2. **Include configuration examples**
3. **Document troubleshooting steps**
4. **Add performance characteristics**

### API Documentation

Document ZMQ message schemas:

```python
class SpaceTimeUpdate(BaseModel):
    """
    Event published when era or biome changes.
    
    Published by: Core Service
    Subscribed by: Display, Audio, Agent, Image Server
    
    Example:
        {
            "era": "modern",
            "biome": "urban",
            "timestamp": "2024-01-01T12:00:00Z",
            "source": "core_service",
            "interaction_strength": 0.75
        }
    """
    era: Era
    biome: Biome
    timestamp: datetime
    source: str
    interaction_strength: float = Field(ge=0.0, le=1.0)
```

## Release Process

### Version Management

Use semantic versioning (MAJOR.MINOR.PATCH):

```bash
# Update version in pyproject.toml
# Create git tag
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0
```

### Release Checklist

1. **Pre-release**:
   - [ ] All tests passing
   - [ ] Documentation updated
   - [ ] Performance benchmarks run
   - [ ] Security review completed

2. **Release**:
   - [ ] Version bumped
   - [ ] Changelog updated
   - [ ] Git tag created
   - [ ] Release notes written

3. **Post-release**:
   - [ ] Deployment tested
   - [ ] Monitoring verified
   - [ ] User feedback collected

### Deployment

```bash
# Production deployment
sudo ./infra/scripts/deploy.sh experimance update

# Rollback if needed
sudo ./infra/scripts/deploy.sh experimance rollback
```

This development workflow ensures code quality, maintainability, and smooth collaboration across the Experimance project.
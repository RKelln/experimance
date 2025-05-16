# Experimance Project Instructions

This project is a Python-based distributed system for an interactive art installation called Experimance. The system uses ZeroMQ for inter-service communication and handles real-time sensor data, image processing, and audio-visual outputs.

## Project Architecture

- **Main Meta-Package**: `experimance-project` - The primary project that includes dependencies on all services
- **Common Library**: `experimance-common` - A shared library used by all services with communication utilities, constants, and schemas
- **Services**: Individual components like core, display, audio, and agent services that communicate via ZMQ
  - Each service is a standalone Python package with its own dependencies and src directory
- **Infrastructure**: Configuration for deployment, monitoring, and management

## Package Structure

```
experimance/
├── pyproject.toml       # Main project meta-package
├── setup_new.sh         # Installation script
├── libs/
│   └── common/          # Shared common libraries
│       ├── pyproject.toml
│       └── src/
│           └── experimance_common/
│               ├── __init__.py
│               ├── constants.py
│               ├── schemas.py
│               └── zmq_utils.py
├── services/
│   ├── core/            # Core service managing state machine
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_core/
│   │
│   ├── display/         # Display service for sand table visualization
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_display/
│   │
│   ├── audio/           # Audio service for sound generation
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_audio/
│   │
│   ├── agent/           # Agent service for AI interaction
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_agent/
│   │
│   ├── image_server/    # Image generation service
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_image_server/
│   │
│   └── transition/      # Image transition service
│       ├── pyproject.toml
│       └── src/
│           └── experimance_transition/
├── utils/               # Utility modules for testing and other purposes
│   └── tests/           # Testing utilities
├── scripts/             # Utility scripts for setup and management
└── infra/
    ├── ansible/
    ├── docker/
    └── grafana/
```

## Coding Standards

- Use Python 3.11+ compatible code
- Follow PEP 8 style guidelines with 4-space indentation
- Use type hints for function parameters and return values
- Document all modules, classes, and functions with docstrings
- Organize imports in the following order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Use async/await for asynchronous code
- Implement proper error handling with try/except blocks
- Use logging instead of print statements
- Update files directly, don't use temp files or scripts to write and then update files

## ZeroMQ Communication

- Services communicate via ZMQ PUB/SUB and PUSH/PULL patterns
- Always use the `experimance_common.zmq_utils` classes for ZMQ communication
- Port configurations are defined in `experimance_common.constants.DEFAULT_PORTS`
- Message types are defined in `experimance_common.zmq_utils.MessageType` enum
- ZMQ addresses should use proper formatting:
  - For binding: `tcp://*:{port}`
  - For connecting: `tcp://localhost:{port}`

## Common Patterns

- Use `asyncio` for concurrent operations
- Use `pydantic` for data validation and serialization
- Always handle cleanup in services with proper socket closing and context termination
- Implement graceful shutdown with signal handlers
- Use proper logging with configurable log levels
- Services should implement a standard interface with `start()`, `stop()`, and `run()` methods

## Dependency Management

- Always use `uv` for package management and running scripts
- Define dependencies in each service's pyproject.toml file
- Use explicit version requirements for dependencies
- Use optional-dependencies for features that aren't required
- Target Python 3.11+ compatibility
- Each service should specify its own dependencies with appropriate version constraints
- Handle SDL2 dependencies with pysdl2 and pysdl2-dll instead of sdl2
- Use src layout for all packages to ensure clean development installs

## Testing

- Write unit tests with pytest
- Use pytest-asyncio for testing async code
- Mock external dependencies
- Test failure cases as well as success
- Use fixtures for common setup

## Commenting Conventions

- Use inline comments sparingly and only when needed for complex logic
- Document public APIs with complete docstrings
- Include examples in docstrings for complex functionality
- Use TODO comments for future work with a brief explanation

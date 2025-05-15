# Experimance Project Instructions

This project is a Python-based distributed system for an interactive art installation called Experimance. The system uses ZeroMQ for inter-service communication and handles real-time sensor data, image processing, and audio-visual outputs.

## Project Architecture

- **Main Package**: `experimance` - The primary package that includes common utilities, services, and infrastructure
- **Common Library**: `experimance_common` - A shared library used by all services with communication utilities, constants, and schemas
- **Services**: Individual components like display, audio, and agent services that communicate via ZMQ
- **Infrastructure**: Configuration for deployment, monitoring, and management

## Package Structure

```
experimance/
├── __init__.py            # Main package init
├── libs/
│   ├── __init__.py
│   └── common/            # Shared common libraries
│       └── experimance_common/
│           ├── __init__.py
│           ├── constants.py
│           ├── schemas.py
│           └── zmq_utils.py
├── services/
│   ├── __init__.py
|   ├── experimance/       # Core service managing state machine
│   ├── display/
│   ├── audio/
│   └── agent/
└── infra/
    ├── __init__.py
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

- Always use `uv` for package mangagement and running scripts
- Define dependencies in pyproject.toml and setup.py
- Use explicit version requirements for dependencies
- Use extras_require for optional dependencies
- Target Python 3.11+ compatibility
- Handle SDL2 dependencies with pysdl2 and pysdl2-dll instead of sdl2

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

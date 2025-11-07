---
applyTo: '**'
---
# Experimance Project Instructions

This project is a Python-based distributed system for interactive art installations, the first project implemented is Experimance. 
The system uses ZeroMQ for inter-service communication and handles real-time sensor data, image processing, and audio-visual outputs.
The code supports additional projects and services, allowing for modular development and deployment.

## Project Architecture

- **Main Meta-Package**: `experimance-project` - The primary project that includes dependencies on all services
- **Common Library**: `experimance-common` - A shared library used by all services with communication utilities, constants, and schemas
- **Services**: Individual components like core, display, audio, and agent services that communicate via ZMQ
  - Each service is a standalone Python package with its own dependencies and src directory
- **Infrastructure**: Configuration for deployment, monitoring, and management
- **Utilities**: Scripts and tools for testing, examples, and development support
- **Multi-Project Support**: Dynamic loading system allows multiple art projects to share the same codebase while having project-specific schemas, constants, and configurations

## Package Structure

```
experimance/
├── libs/
│   └── common/          # Shared common libraries
│       └── src/
│           └── experimance_common/
│               ├── zmq/
│               │   ├── config.py           # ZMQ configuration pydantic models
│               │   ├── components.py       # ZMQ communication components
│               │   ├── services.py         # ZMQ services composed of components
│               │   └── mocks.py            # Mock ZMQ components for testing
│               ├── config.py               # Service configuration management using pydantic
│               ├── constants.py            # Constants used across the project
│               ├── logger.py               # Logging utilities
│               ├── schemas.py              # Pydantic schemas for data validation across services
│               ├── schemas.pyi             # Type stubs for schemas, supporting multiple projects
│               └── image_utils.py          # Image processing utilities
├── services/
│   ├── core/            # Core service managing state machine
│   │   └── src/
│   │       └── experimance_core/
│   │
│   ├── display/         # Display service for sand table visualization
│   │   └── src/
│   │       └── experimance_display/
│   │
│   ├── audio/           # Audio service for sound generation
│   │   └── src/
│   │       └── experimance_audio/
│   │
│   ├── agent/           # Agent service for AI interaction
│   │   └── src/
│   │       └── experimance_agent/
│   │
│   ├── image_server/    # Image generation service
│       └── src/
│           └── image_server/
│
├── utils/               # Utility modules for testing and other purposes
│   ├── examples/        # Examples of usage
│   └── tests/           # Testing utilities for cross service testing
├── scripts/             # Utility scripts for setup and management
├─ infra/
│
├── pyproject.toml       # Main project configuration file
│
└── projects/                 # **ONLY project-specific artifacts live here**
│   ├─ experimance/
│   │   ├─ .env             # Experimance project environment variables
│   │   ├─ config.toml      # Experimance project configuration
│   │   ├─ constants.py     # constants specific to Experimance project
│   │   ├─ schemas.py       # schemas specific to Experimance project
│   │   └─ schemas.pyi      # type stubs for schemas
│   └─ fire/                # overrides for Feed the Fires project
│       └─ ...              # configuration files for Feed the Fires project
```

# Primary Development Guidelines
- If you see something wrong that isn't related to the current request, please note it in the feedback to the user, but do not fix it without asking first
- Update files directly, don't use temp files or scripts to write and then update files
- Use best practices and always to to keep complexity low and clarity high
- Use the existing code as a guide for new implementations but if its a mess suggest alternatives
- To support multiple projects:
  - Use the `projects` directory for project-specific configurations, schemas, and constants
  - IMPORTANT: Ignore linter and pydantic errors related to dynamic imports (Era, Biome, etc.)

## Coding Standards

- Use Python 3.11 compatible code
- Follow PEP 8 style guidelines with 4-space indentation
- Use type hints for function parameters and return values
- Document all modules, classes, and functions with docstrings
- Organize imports in the following order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Use async/await for asynchronous code, aiohttp for HTTP requests
- Pydantic for data validation, use dataclasses for simple data structures
- Implement proper error handling with try/except blocks
- Use logging instead of print statements

## Project managed by `uv`

- **Use `uv run`** for running services and scripts
- Each service should have a `pyproject.toml` file with dependencies and scripts defined
- Use `uv run -m` to start services, e.g., `uv run -m experimance_core`
- Use `uv run -m pytest` for running tests, e.g., `uv run -m pytest utils/tests/`
- **Development**: Use `scripts/dev <service>` or `scripts/dev all` for coordinated development (auto-cleanup, logging). For isolated testing use `uv run -m` directly.
- **Environment Variables**: Override any config setting with `EXPERIMANCE_<SECTION>_<KEY>=value` (e.g., `EXPERIMANCE_ZMQ_BASE_PORT=6000`)

## Other tools (MCP, etc)

- use context7 to look up API and other library docs

## ZeroMQ Communication

- Services communicate via ZMQ PUB/SUB and PUSH/PULL patterns
- Always use the `experimance_common.services` base classes for services that need ZMQ communication
- Port configurations are defined in `experimance_common.constants.DEFAULT_PORTS`
- Message types are defined in `experimance_common.zmq_utils.MessageType` enum
- ZMQ addresses should use proper formatting:
  - For binding: `tcp://*:{port}`
  - For connecting: `tcp://localhost:{port}`

## Common Patterns

- Use `asyncio` for concurrent operations
- Use `pydantic` for data validation and serialization
- Always handle cleanup in services (note base classes handle most cleanup)
- Use proper logging with configurable log levels
- Services should implement a standard interface with `start()`, `stop()`, and `run()` methods
- When doing file locations, use pathlib and look into `constants_base.py` for path constants relevant to your service/media type
  - Use `resolve_path()` to convert from file locations specified in config to absolute paths
- Use `logger = logging.getLogger(__name__)` for module-level logging

## Dependency Management

- Always use `uv` for package management and running scripts
- Define dependencies in each service's pyproject.toml file
- Use explicit version requirements for dependencies
- Use optional-dependencies for features that aren't required
- Target Python 3.11+ compatibility
- Each service should specify its own dependencies with appropriate version constraints
- Use src layout for all packages to ensure clean development installs

## Testing

- Read README_SERVICE_TESTING.md for service-specific testing instructions
- Write unit tests with pytest, using pytest fixtures for setup
- Use pytest-asyncio for testing async code
- Mock external dependencies as needed, check for existing mocks and reuse
- Test failure cases as well as success
- Use fixtures for common setup
- Place inter-service tests in `utils/tests/` directory or in `service/NAME/tests/` for service-specific tests
- Put usage examples that can guide others  in `utils/examples/`

## Commenting Conventions

- Use inline comments sparingly and only when needed for complex logic
- Document public APIs with complete docstrings
- Include examples in docstrings for complex functionality
- Use TODO comments for future work with a brief explanation

## Documention

- There is extensive documentation:
  - [Experimance Project Overview](README.md)
  - [Technical Design](technical_design.md)
  - [Writing Services](libs/common/README_SERVICE.md)
  - [Using ZMQ](libs/common/README_ZMQ.md)
  - [Testing Services](libs/common/README_SERVICE_TESTING.md)
- Before major changes, read the documentation to understand existing patterns
- Use Markdown for documentation files
- Keep documentation up-to-date with code changes

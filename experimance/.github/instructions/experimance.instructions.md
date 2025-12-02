---
applyTo: '**'
---
# Experimance Project

Python-based distributed system for interactive art installations using ZeroMQ for inter-service communication. Handles real-time sensor data, image processing, and audio-visual outputs.

## Architecture

- **Common Library**: `experimance-common` - shared communication, constants, schemas
- **Services**: core, display, audio, agent, image_server - each a standalone package
- **Multi-Project**: `projects/` directory contains project-specific schemas, constants, configs

## Key Directories

```
libs/common/src/experimance_common/   # Shared library
services/{name}/src/{name}/           # Service packages  
projects/{project}/                   # Project-specific config
utils/tests/                          # Inter-service tests
utils/examples/                       # Usage examples
```

## Running Services

Managed by `uv`:
- Start service: `uv run -m experimance_core`
- Run tests: `uv run -m pytest utils/tests/`
- Development: `scripts/dev <service>` or `scripts/dev all`
- Environment override: `EXPERIMANCE_<SECTION>_<KEY>=value`

## ZeroMQ Patterns

- Use `experimance_common.zmq.services` base classes
- Ports: `experimance_common.constants.DEFAULT_PORTS`
- Addresses: bind `tcp://*:{port}`, connect `tcp://localhost:{port}`

## Service Patterns

- Services implement `start()`, `stop()`, `run()` interface
- Base classes handle cleanup
- Use `pathlib` and `constants_base.py` for file paths
- Use `resolve_path()` for config-specified paths

## Multi-Project Support

Ignore linter errors for dynamic imports (Era, Biome, etc.) - these are loaded at runtime from `projects/`.

## Documentation

- [Technical Design](docs/technical_design.md)
- [Writing Services](libs/common/README_SERVICE.md)
- [ZMQ Guide](libs/common/README_ZMQ.md)
- [Testing](libs/common/README_SERVICE_TESTING.md)

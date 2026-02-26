# Experimance Common Library

`experimance-common` is the shared Python library used by every Experimance service. It provides async service base classes, ZMQ messaging, Pydantic configuration, typed message schemas, health monitoring, and test utilities.

## Quick start

```bash
# Install the whole workspace (recommended)
cd <repo-root>
uv sync

# Install only this library in editable mode
cd libs/common
uv pip install -e .
```

Run the library tests:

```bash
cd libs/common
uv run pytest
```

Requires Python ≥ 3.11, Linux or macOS.

## Minimal example

```python
from experimance_common.base_service import BaseService
import asyncio

class MyService(BaseService):
    def __init__(self):
        super().__init__(service_name="my-service")

    async def start(self):
        self._register_task(self._loop())
        await super().start()   # always last

    async def _loop(self):
        while self.running:
            await self._sleep_if_running(1.0)

asyncio.run(MyService().run())
```

## Additional docs

| Doc | Description |
|-----|-------------|
| [docs/overview.md](docs/overview.md) | What this library does, module map, environment requirements |
| [docs/services.md](docs/services.md) | Building services with `BaseService`: lifecycle, logging, deployment |
| [docs/zmq.md](docs/zmq.md) | ZMQ composition architecture: components, patterns, and worked examples |
| [docs/testing.md](docs/testing.md) | Testing with `active_service`, mocks, and anti-patterns to avoid |
| [docs/configuration.md](docs/configuration.md) | `BaseConfig`/`BaseServiceConfig` reference, TOML, CLI, env vars |
| [docs/roadmap.md](docs/roadmap.md) | Near-term goals and known gaps |

### Project-level docs

| Doc | Description |
|-----|-------------|
| [../../docs/health_system.md](../../docs/health_system.md) | Health monitoring and `HealthStatus` reference |
| [../../docs/logging_system.md](../../docs/logging_system.md) | Logging paths (dev vs. production) |
| [../../docs/image_zmq_utilities.md](../../docs/image_zmq_utilities.md) | Image transport over ZMQ |

## Contributing

When extending or modifying the common library:

1. Add docstrings and type hints to all public functions and classes.
2. Add or update tests in `tests/`.
3. Keep commands in docs copy-pasteable and verify paths exist.
4. Run `uv run pytest` before committing.

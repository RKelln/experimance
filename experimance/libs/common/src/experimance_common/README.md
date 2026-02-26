# experimance_common

Shared Python library for all Experimance services.

## What this package provides

- `BaseService` — async service with lifecycle management, signal handling, and error recovery
- `PubSubService`, `WorkerService`, `ControllerService` — composition-based ZMQ messaging
- `BaseConfig` / `BaseServiceConfig` — Pydantic config with TOML loading and CLI integration
- Typed Pydantic message schemas (`MessageType`, `SpaceTimeUpdate`, `ImageReady`, …)
- `HealthStatus` / `HealthCheck` — health monitoring via file-based IPC
- `setup_logging` — adaptive logging (dev: `logs/`, production: `/var/log/experimance/`)
- `active_service`, `MockPubSubService` — test utilities

## Quick start

```bash
cd <repo-root> && uv sync    # install workspace
# or
cd libs/common && uv pip install -e .
```

```python
from experimance_common.base_service import BaseService

class MyService(BaseService):
    def __init__(self):
        super().__init__(service_name="my-service")

    async def start(self):
        self.add_task(self._loop())
        await super().start()         # always last

    async def _loop(self):
        while self.running:
            await self._sleep_if_running(1.0)
```

## Package layout

| Module | Purpose |
|--------|---------|
| `base_service.py` | `BaseService`, lifecycle, signal handling |
| `service_state.py` | `ServiceState` enum |
| `config.py` | `BaseConfig`, `BaseServiceConfig`, TOML loader |
| `constants_base.py` | `DEFAULT_PORTS`, paths, timeouts |
| `schemas_base.py` | `MessageType` and Pydantic message schemas |
| `health.py` | `HealthStatus`, `HealthCheck` |
| `logger.py` | `setup_logging`, `configure_external_loggers` |
| `cli.py` | `create_simple_main`, CLI arg generation |
| `image_utils.py` | Image encode/decode and transport helpers |
| `audio_utils.py` | Audio utilities |
| `osc_client.py` | OSC bridge to SuperCollider |
| `test_utils.py` | `active_service`, `wait_for_service_state` |
| `zmq/` | ZMQ components, services, mocks, config |

## Full documentation

See [`docs/`](../../docs/) for the complete guide:

- [overview.md](../../docs/overview.md) — what the library does, environment requirements
- [services.md](../../docs/services.md) — building and deploying services
- [zmq.md](../../docs/zmq.md) — ZMQ patterns and examples
- [testing.md](../../docs/testing.md) — testing best practices
- [configuration.md](../../docs/configuration.md) — config system reference

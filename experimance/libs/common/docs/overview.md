# experimance-common — Overview

`experimance-common` (`experimance_common`) is the shared Python library used by every Experimance service. It provides:

- **Service base classes** — `BaseService` with lifecycle, signal handling, and error recovery
- **ZMQ messaging** — composition-based Publisher, Subscriber, Push, Pull components and pre-built `PubSubService`, `WorkerService`, `ControllerService`
- **Configuration** — Pydantic-validated TOML config loading with environment and CLI overrides
- **Schemas** — typed Pydantic message models for all inter-service messages
- **Health monitoring** — `HealthStatus`, `HealthCheck`, and file-based IPC for the health service
- **Logging** — adaptive path selection (dev vs. production), external-library log control
- **Utilities** — image transport helpers, audio utilities, OSC client, test utilities

## Environment requirements

| Requirement | Value |
|-------------|-------|
| Python | ≥ 3.11 |
| OS | Linux (production); macOS supported for development |
| ZMQ daemon | pyzmq ≥ 25.1.1 (no separate broker needed) |
| Package manager | [uv](https://github.com/astral-sh/uv) |

For production deployments, services run as systemd units. See [services.md](services.md) for the deployment checklist.

## Quick start

```bash
# Install the whole workspace (recommended)
cd <repo-root>
uv sync

# Or install only this library in editable mode
cd libs/common
uv pip install -e .
```

Run the library tests:

```bash
cd libs/common
uv run pytest
```

## Minimal service example

```python
# my_service.py
from experimance_common.base_service import BaseService
import asyncio

class MyService(BaseService):
    def __init__(self):
        super().__init__(service_name="my-service")

    async def start(self):
        self._register_task(self._work_loop())
        await super().start()   # always last

    async def stop(self):
        await super().stop()    # always first

    async def _work_loop(self):
        while self.running:
            # do work …
            await self._sleep_if_running(1.0)

async def main():
    service = MyService()
    await service.start()
    await service.run()         # blocks until SIGINT/SIGTERM

if __name__ == "__main__":
    asyncio.run(main())
```

## Minimal ZMQ service example

```python
from experimance_common.base_service import BaseService
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PORTS

class MyZmqService(BaseService):
    def __init__(self):
        super().__init__(service_name="my-zmq-service")
        self.zmq = PubSubService(PubSubServiceConfig(
            name="my-zmq-service",
            publisher=PublisherConfig(address="tcp://*", port=DEFAULT_PORTS["events"]),
            subscriber=SubscriberConfig(address="tcp://localhost", port=DEFAULT_PORTS["events"],
                                        topics=["heartbeat"]),
        ))

    async def start(self):
        self.zmq.add_message_handler("heartbeat", self._on_heartbeat)
        await self.zmq.start()
        self._register_task(self._publish_loop())
        await super().start()

    async def stop(self):
        await self.zmq.stop()
        await super().stop()

    async def _on_heartbeat(self, message):
        pass

    async def _publish_loop(self):
        while self.running:
            await self.zmq.publish({"status": "ok"}, "heartbeat")
            await self._sleep_if_running(5.0)
```

## Module map

| Module | Purpose |
|--------|---------|
| `base_service.py` | `BaseService` — lifecycle, signal handling, task management |
| `service_state.py` | `ServiceState` enum and `StateManager` |
| `service_decorators.py` | Lifecycle-validation decorators (`@requires_state`) |
| `health.py` | `HealthStatus`, `HealthCheck`, `ServiceHealth` dataclasses |
| `config.py` | `BaseConfig`, `BaseServiceConfig`, `load_config_with_overrides` |
| `constants_base.py` | `DEFAULT_PORTS`, paths, timeouts, image transport constants |
| `constants.py` | Project-specific constants (extends `constants_base`) |
| `schemas_base.py` | Base Pydantic message schemas and `MessageType` enum |
| `schemas.py` | Project-specific message schemas |
| `logger.py` | `setup_logging`, `configure_external_loggers` |
| `cli.py` | `create_simple_main`, standard CLI argument parsing |
| `image_utils.py` | Image encode/decode, transport mode selection, temp-file cleanup |
| `audio_utils.py` | Audio processing helpers |
| `osc_client.py` / `osc_config.py` | OSC bridge to SuperCollider |
| `notifications.py` | Lightweight notification system |
| `test_utils.py` | `active_service`, `wait_for_service_state`, test fixtures |
| `zmq/config.py` | `ZmqSocketConfig`, `PublisherConfig`, `PubSubServiceConfig`, … |
| `zmq/components.py` | `PublisherComponent`, `SubscriberComponent`, `PushComponent`, `PullComponent` |
| `zmq/services.py` | `PubSubService`, `WorkerService`, `ControllerService` |
| `zmq/mocks.py` | `MockPubSubService`, `MockWorkerService` for testing |
| `zmq/zmq_utils.py` | Low-level ZMQ helpers |

## Further reading

- [services.md](services.md) — full service development guide
- [zmq.md](zmq.md) — ZMQ architecture and patterns
- [testing.md](testing.md) — testing best practices
- [configuration.md](configuration.md) — config system reference

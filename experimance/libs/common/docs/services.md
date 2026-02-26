# Service Development Guide

This guide covers building Experimance services with `BaseService`. For ZMQ-specific patterns see [zmq.md](zmq.md). For testing guidance see [testing.md](testing.md). For the config system see [configuration.md](configuration.md).

## Overview → Setup → Configuration → Usage → Testing → Troubleshooting → Deployment

---

## Overview

Every Experimance process is a **service** — a long-running `asyncio` coroutine with a standard lifecycle. `BaseService` provides:

- `start()` / `stop()` / `run()` lifecycle hooks
- `ServiceState` tracking (seven states, see below)
- Background task management via `add_task()` / `_register_task()`
- Signal handling (`SIGINT`, `SIGTERM`) → graceful shutdown
- Error recording with `record_error(exc, is_fatal=False)`
- `_sleep_if_running(seconds)` — sleep that respects `self.running`
- Adaptive logging (dev → `logs/`, production → `/var/log/experimance/`)

### Service state machine

```
INITIALIZING → INITIALIZED → STARTING → STARTED → RUNNING → STOPPING → STOPPED
```

Do **not** set `self.state` directly — let `BaseService` handle transitions.

---

## Setup

```bash
cd <repo-root>
uv sync                     # install all workspace deps
# or, inside libs/common only:
cd libs/common
uv pip install -e .
```

Requires Python ≥ 3.11, Linux or macOS.

---

## Configuration

All service configs extend `BaseServiceConfig` (itself a `BaseConfig`):

```python
# src/my_service/config.py
from experimance_common.config import BaseServiceConfig
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PORTS
from pydantic import Field

class MyServiceConfig(BaseServiceConfig):
    service_name: str = "my-service"          # override default
    work_interval: float = Field(default=1.0)
    max_retries: int = Field(default=3)

    zmq: PubSubServiceConfig = Field(
        default_factory=lambda: PubSubServiceConfig(
            publisher=PublisherConfig(address="tcp://*",
                                     port=DEFAULT_PORTS["events"]),
            subscriber=SubscriberConfig(address="tcp://localhost",
                                        port=DEFAULT_PORTS["events"],
                                        topics=["heartbeat"]),
        )
    )
```

Load config from a TOML file with optional overrides:

```python
config = MyServiceConfig.from_overrides(
    config_file="config.toml",          # optional; falls back to defaults
    override_config={"work_interval": 0.5}
)
```

Config file resolution priority:

1. `projects/<PROJECT_ENV>/<service_name>.toml` (if `PROJECT_ENV` is set)
2. `services/<service_name>/config.toml`
3. Built-in defaults

---

## Usage

### Minimal service

```python
# src/my_service/my_service.py
import logging
from experimance_common.base_service import BaseService
from experimance_common.constants import TICK
from .config import MyServiceConfig

logger = logging.getLogger(__name__)

class MyService(BaseService):
    def __init__(self, config: MyServiceConfig):
        super().__init__(service_name=config.service_name)
        self.config = config

    async def start(self):
        # Set up resources, register tasks, then call super LAST
        self.add_task(self._work_loop())
        await super().start()

    async def stop(self):
        # Call super FIRST, then tear down resources
        await super().stop()

    async def _work_loop(self):
        while self.running:
            try:
                # … do work …
                pass
            except Exception as e:
                self.record_error(e, is_fatal=False)
            await self._sleep_if_running(self.config.work_interval)
```

### Entry point

```python
# src/my_service/__main__.py
from experimance_common.cli import create_simple_main
from .my_service import MyService
from .config import MyServiceConfig

if __name__ == "__main__":
    create_simple_main(MyService, MyServiceConfig)()
```

Run with:

```bash
uv run -m my_service                          # defaults
uv run -m my_service --config config.toml    # explicit config file
uv run -m my_service --work-interval 0.5     # CLI override
```

### Lifecycle rules

| Method | Call `super()` | Notes |
|--------|---------------|-------|
| `start()` | **LAST** | Resources initialized before handing control to base |
| `stop()` | **FIRST** | Base tears down state/tasks before you clean up |
| `run()` | Usually not overridden | Blocks until stopped |

### Error handling

```python
try:
    await risky_operation()
except RecoverableError as e:
    self.record_error(e, is_fatal=False)   # service continues
except CriticalError as e:
    self.record_error(e, is_fatal=True)    # triggers automatic shutdown
```

### Logging

```python
import logging
logger = logging.getLogger(__name__)   # that's it; BaseService configures the root
```

Log destinations:

| Environment | Path |
|-------------|------|
| Development | `logs/<service_name>.log` (+ console) |
| Production | `/var/log/experimance/<service_name>.log` (file only) |

Detection: running as root, `EXPERIMANCE_ENV=production`, or `/etc/experimance` exists → production.

---

## Testing

See [testing.md](testing.md) for full guidance. Quick pattern:

```python
from experimance_common.test_utils import active_service
from experimance_common.service_state import ServiceState

@pytest.mark.asyncio
async def test_my_service():
    service = MyService(MyServiceConfig.from_overrides({"work_interval": 0.01}))
    async with active_service(service) as svc:
        assert svc.state == ServiceState.RUNNING
    assert service.state == ServiceState.STOPPED
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Service stuck in `STARTING` | Exception in `start()` before `super().start()` | Check logs for the error |
| `record_error` not stopping service | `is_fatal=False` | Pass `is_fatal=True` for unrecoverable errors |
| Tasks never cancelled | Task ignores `self.running` | Use `_sleep_if_running()` and check `self.running` in loops |
| Tests hang | Missing `active_service` cleanup | Use `async with active_service(service)` |
| Double-logged errors | Raising after `record_error` | Don't re-raise unless the caller needs to handle it |

---

## Deployment

### File layout for a new service

```
services/my_service/
├── src/my_service/
│   ├── __init__.py
│   ├── __main__.py       # entry point
│   ├── config.py         # MyServiceConfig
│   └── my_service.py     # MyService
├── tests/
│   ├── __init__.py
│   └── test_my_service.py
├── README.md             # brief — link to this guide
└── config.toml           # default configuration
```

### Naming conventions

| Artifact | Convention | Example |
|----------|-----------|---------|
| Service directory | `snake_case` | `services/my_service/` |
| Python package | `snake_case` | `my_service` |
| systemd unit | `kebab-case` | `experimance-my-service.service` |
| Config file | `<service_name>.toml` | `projects/experimance/my_service.toml` |

### systemd unit template

```ini
# infra/systemd/experimance-my-service.service
[Unit]
Description=Experimance My Service
After=network.target

[Service]
Type=simple
User=experimance
WorkingDirectory=/home/experimance/experimance
Environment=PATH=/home/experimance/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=EXPERIMANCE_ENV=production
ExecStart=/home/experimance/.local/bin/uv run -m my_service
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Production checklist

```bash
# Copy and enable the unit
sudo cp infra/systemd/experimance-my-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now experimance-my-service.service
sudo systemctl status experimance-my-service.service

# Dev smoke-test
./scripts/dev my_service
```

### Files touched when creating a service

- `services/<name>/src/<name>/` — service source
- `infra/systemd/experimance-<name>.service` — systemd unit
- `projects/<project>/<name>.toml` — project-specific config (if needed)

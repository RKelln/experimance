# Service Testing Guide

This guide covers testing Experimance services. For service implementation see [services.md](services.md). For ZMQ architecture see [zmq.md](zmq.md).

## Overview → Setup → Patterns → Anti-patterns → Debugging

---

## Overview

Service tests are `pytest-asyncio` coroutines. The key utilities live in `experimance_common.test_utils`:

| Utility | Purpose |
|---------|---------|
| `active_service(service)` | Async context manager: starts, yields, stops, and asserts clean shutdown |
| `wait_for_service_state(svc, state, timeout)` | Poll until service reaches the given `ServiceState` |

Mock ZMQ services are in `experimance_common.zmq.mocks`:

| Mock | Replaces |
|------|---------|
| `MockPubSubService` | `PubSubService` |
| `MockWorkerService` | `WorkerService` |

---

## Setup

```bash
cd libs/common
uv run pytest                          # run all tests
uv run pytest tests/test_base_service.py -v  # single file
uv run pytest -x --tb=short           # stop on first failure
```

---

## Patterns

### Basic lifecycle test

```python
import pytest
from experimance_common.test_utils import active_service
from experimance_common.service_state import ServiceState
from my_service.my_service import MyService
from my_service.config import MyServiceConfig

@pytest.mark.asyncio
async def test_lifecycle():
    config = MyServiceConfig.from_overrides({"work_interval": 0.01})
    service = MyService(config)

    async with active_service(service) as svc:
        assert svc.state == ServiceState.RUNNING

    assert service.state == ServiceState.STOPPED
```

### Configuration testing

Use `from_overrides()` — never write temporary config files in tests.

```python
@pytest.fixture
def config():
    return MyServiceConfig.from_overrides({
        "service_name": "test-service",
        "work_interval": 0.01,   # fast for tests
        "debug_mode": True,
    })
```

### Testing with mock ZMQ

```python
from experimance_common.zmq.mocks import MockPubSubService
from unittest.mock import patch

@pytest.fixture
async def service(config):
    with patch.object(MyService, "_create_zmq_service",
                      return_value=MockPubSubService(config.zmq)):
        svc = MyService(config)
        yield svc
        # active_service handles cleanup
```

Or simply replace the ZMQ attribute after construction:

```python
service = MyService(config)
service.zmq = MockPubSubService(config.zmq)
async with active_service(service) as svc:
    ...
```

**Mock handler signatures** — note the deliberate difference from production:

| Mock method | Handler signature |
|-------------|------------------|
| `mock.add_message_handler(topic, fn)` | `async def fn(message)` — same as `PubSubService` |
| `mock.set_message_handler(fn)` | `async def fn(topic, message)` — note: renamed vs. `set_default_handler` in production |

> **Known inconsistency**: `MockPubSubService.set_message_handler` does not match the production `PubSubService.set_default_handler` name. Prefer `add_message_handler` for per-topic handlers to stay consistent. See [roadmap.md](roadmap.md).

### Testing message handling

```python
@pytest.mark.asyncio
async def test_handles_heartbeat(service):
    mock_zmq = MockPubSubService(service.config.zmq)
    service.zmq = mock_zmq

    async with active_service(service):
        # Simulate an incoming message
        await mock_zmq.simulate_receive("heartbeat", {"service": "test", "ts": 1.0})
        await asyncio.sleep(0.05)   # let handler run
        assert service.heartbeats_received == 1
```

### Waiting for state

```python
from experimance_common.test_utils import wait_for_service_state
from experimance_common.service_state import ServiceState

await wait_for_service_state(service, ServiceState.RUNNING, timeout=5.0)
```

### Testing error recording

```python
@pytest.mark.asyncio
async def test_non_fatal_error_continues(service):
    async with active_service(service) as svc:
        svc.record_error(ValueError("oops"), is_fatal=False)
        assert svc.state == ServiceState.RUNNING
        assert svc.error_count > 0

@pytest.mark.asyncio
async def test_fatal_error_stops(service):
    async with active_service(service) as svc:
        svc.record_error(RuntimeError("fatal"), is_fatal=True)
        await wait_for_service_state(svc, ServiceState.STOPPED, timeout=2.0)
```

### Pytest fixture pattern (recommended)

```python
import pytest
import asyncio
from experimance_common.test_utils import active_service, wait_for_service_state
from experimance_common.service_state import ServiceState

@pytest.fixture
def config():
    return MyServiceConfig.from_overrides({"work_interval": 0.01})

@pytest.fixture
async def running_service(config):
    svc = MyService(config)
    async with active_service(svc) as s:
        yield s

@pytest.mark.asyncio
async def test_something(running_service):
    assert running_service.state == ServiceState.RUNNING
```

---

## Anti-patterns to avoid

### Configuration

```python
# BAD: temporary file
tmp = tmp_path / "config.toml"
tmp.write_text('[service]\nwork_interval = 0.01')
config = MyServiceConfig.from_file(tmp)

# GOOD: from_overrides
config = MyServiceConfig.from_overrides({"work_interval": 0.01})
```

### Lifecycle

```python
# BAD: manual start/stop without cleanup
await service.start()
# ... test ...
await service.stop()  # won't run if test throws

# GOOD: context manager guarantees cleanup
async with active_service(service) as svc:
    # ... test ...
```

### ZMQ

```python
# BAD: real ZMQ sockets in unit tests (slow, port collisions)
service = MyService(RealZmqConfig())

# GOOD: mock
service.zmq = MockPubSubService(config.zmq)
```

### Waiting for state

```python
# BAD: fixed sleep
await asyncio.sleep(1.0)

# GOOD: poll with timeout
await wait_for_service_state(service, ServiceState.RUNNING, timeout=2.0)
```

### Handlers

```python
# BAD: sync handler in async service
def on_message(message):
    do_something()
service.zmq.add_message_handler("topic", on_message)

# GOOD: async handler
async def on_message(message):
    await do_something()
```

---

## Debugging hanging tests

1. Add `-s` and `--log-cli-level=DEBUG` to pytest to see all output.
2. A test that hangs usually means the service did not stop. Check:
   - Is there an infinite loop that never sees `self.running == False`?
   - Did `start()` raise before `super().start()`, leaving state stuck at `STARTING`?
3. Use `wait_for_service_state` with an explicit `timeout` instead of `asyncio.sleep`.
4. Make sure tasks use `_sleep_if_running()` not `asyncio.sleep()`.

### Avoiding deadlocks

Services that call `await self.run()` from within a background task will deadlock. The correct pattern:

```python
# main entry point only
async def main():
    svc = MyService(config)
    await svc.start()
    await svc.run()   # blocks until stopped

# in tests, use active_service — never call run() in test code
```

---

## Files touched

| File | Role |
|------|------|
| `test_utils.py` | `active_service`, `wait_for_service_state` |
| `zmq/mocks.py` | `MockPubSubService`, `MockWorkerService` |
| `tests/test_base_service.py` | BaseService unit tests |
| `tests/test_service_lifecycle.py` | Lifecycle state tests |
| `tests/zmq/test_mocks.py` | Mock ZMQ service tests |
| `tests/zmq/test_integration.py` | Integration tests |

# ZMQ Architecture Guide

This guide covers the composition-based ZMQ architecture used by all Experimance services. For service lifecycle see [services.md](services.md). For ZMQ testing see [testing.md](testing.md).

## Overview

Experimance uses **ZeroMQ** (via `pyzmq`) for inter-service messaging. The library follows a **composition over inheritance** model: you combine a `BaseService` with one or more `ZMQ service` objects rather than inheriting from a ZMQ-specific base class.

```
BaseService  +  PubSubService       =  event-driven service
BaseService  +  WorkerService       =  task-processing worker
BaseService  +  ControllerService   =  work dispatcher
```

### When to use ZMQ

Use ZMQ when your service needs to:
- Broadcast state changes to many subscribers
- Distribute compute-heavy tasks across worker processes
- Receive real-time streams (depth camera, audio events, generated images)

When **not** to use ZMQ:
- Simple in-process communication (use plain Python calls)
- One-off scripts or tooling

---

## Core components

### Architecture layers

```
┌──────────────────────────────────────────────┐
│              Your Service                    │
│  BaseService + ZMQ Service (composed)        │
├──────────────────────────────────────────────┤
│  zmq/services.py  — PubSubService            │
│                     WorkerService            │
│                     ControllerService        │
├──────────────────────────────────────────────┤
│  zmq/components.py — PublisherComponent      │
│                      SubscriberComponent     │
│                      PushComponent           │
│                      PullComponent           │
├──────────────────────────────────────────────┤
│  zmq/config.py    — Pydantic config schemas  │
└──────────────────────────────────────────────┘
```

### Configuration (`zmq/config.py`)

All ZMQ config is Pydantic-validated and immutable (`frozen=True`):

| Class | Purpose |
|-------|---------|
| `ZmqSocketConfig` | Single socket (address + port + bind/connect) |
| `PublisherConfig` | Publisher socket |
| `SubscriberConfig` | Subscriber socket with topic list |
| `PushConfig` | Push socket |
| `PullConfig` | Pull socket |
| `PubSubServiceConfig` | Combines publisher + subscriber |
| `WorkerServiceConfig` | Combines pull + push |
| `ControllerServiceConfig` | Combines publisher + per-worker push/pull |

### Message schemas (`zmq/config.py`, `schemas_base.py`)

```python
from experimance_common.schemas import MessageType

# Available types
MessageType.SPACE_TIME_UPDATE
MessageType.RENDER_REQUEST
MessageType.PRESENCE_STATUS
MessageType.IMAGE_READY
MessageType.TRANSITION_READY
MessageType.LOOP_READY
MessageType.AUDIENCE_PRESENT
MessageType.SPEECH_DETECTED
MessageType.DISPLAY_MEDIA
```

Use `MessageType` values as ZMQ topics for type-safe subscriptions:

```python
SubscriberConfig(topics=[MessageType.IMAGE_READY, MessageType.LOOP_READY])
```

### Default ports

```python
from experimance_common.constants import DEFAULT_PORTS

DEFAULT_PORTS = {
    "events":               5555,  # unified pubsub channel
    "updates":              5556,  # service status updates
    "agent":                5557,  # agent service publisher
    "transition_requests":  5560,
    "transition_results":   5561,
    "video_requests":       5562,
    "video_results":        5563,
    "image_requests":       5564,
    "image_results":        5565,
    "depth":                5566,
    "audio_osc_send_port":  5570,
    "audio_osc_recv_port":  5571,
}
```

---

## Communication patterns

### Pattern 1 — PubSub (broadcast)

One publisher, many subscribers. Fire-and-forget. Topics are prefix-matched.

```
Publisher (bind :5555) ─── topic "heartbeat" ──→ Subscriber A
                       ─── topic "status"    ──→ Subscriber B
                                             ──→ Subscriber C
```

```python
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PORTS

zmq = PubSubService(PubSubServiceConfig(
    name="my-service",
    publisher=PublisherConfig(
        address="tcp://*",
        port=DEFAULT_PORTS["events"],
    ),
    subscriber=SubscriberConfig(
        address="tcp://localhost",
        port=DEFAULT_PORTS["events"],
        topics=["heartbeat", "status"],
    ),
))

# Register handlers BEFORE start()
zmq.add_message_handler("heartbeat", my_heartbeat_handler)  # handler(message)
zmq.set_default_handler(my_catch_all_handler)               # handler(topic, message)

await zmq.start()

# Publish
await zmq.publish({"status": "ok"}, "status")

# Topic "" subscribes to everything (catch-all)
SubscriberConfig(topics=[""])
```

**Handler signatures:**

| Registration method | Signature |
|--------------------|-----------| 
| `add_message_handler(topic, fn)` | `async def fn(message)` |
| `set_default_handler(fn)` | `async def fn(topic, message)` |

### Pattern 2 — Worker (task queue)

Controller pushes tasks; workers pull and push results back. Load-balanced round-robin.

```
Controller (push :5564) ──→ Worker A (pull)
                        ──→ Worker B (pull)
Worker A (push :5565) ──→ Controller (pull)
```

```python
# Worker
from experimance_common.zmq.services import WorkerService
from experimance_common.zmq.config import WorkerServiceConfig, PullConfig, PushConfig

zmq = WorkerService(WorkerServiceConfig(
    name="image-worker",
    pull=PullConfig(address="tcp://localhost", port=DEFAULT_PORTS["image_requests"]),
    push=PushConfig(address="tcp://*",        port=DEFAULT_PORTS["image_results"]),
))
zmq.set_task_handler(my_task_handler)    # async def fn(task) -> result_dict | None
await zmq.start()
```

### Pattern 3 — Controller (work dispatch)

Central service that distributes work to named worker pools and aggregates results.

```python
from experimance_common.zmq.services import ControllerService
from experimance_common.zmq.config import ControllerServiceConfig, PublisherConfig, WorkerConfig

zmq = ControllerService(ControllerServiceConfig(
    name="core",
    publisher=PublisherConfig(address="tcp://*", port=DEFAULT_PORTS["events"]),
    workers=[
        WorkerConfig(name="image",  push_port=DEFAULT_PORTS["image_requests"],
                                    pull_port=DEFAULT_PORTS["image_results"]),
        WorkerConfig(name="video",  push_port=DEFAULT_PORTS["video_requests"],
                                    pull_port=DEFAULT_PORTS["video_results"]),
    ],
))
await zmq.dispatch("image", task_data)
```

---

## Full ZMQ service example

```python
import logging
from experimance_common.base_service import BaseService
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PORTS

logger = logging.getLogger(__name__)

class MyZmqService(BaseService):
    def __init__(self):
        super().__init__(service_name="my-zmq-service")
        self.zmq = PubSubService(PubSubServiceConfig(
            name="my-zmq-service",
            publisher=PublisherConfig(address="tcp://*",
                                      port=DEFAULT_PORTS["events"]),
            subscriber=SubscriberConfig(address="tcp://localhost",
                                        port=DEFAULT_PORTS["events"],
                                        topics=["heartbeat"]),
        ))

    async def start(self):
        # 1. Register handlers
        self.zmq.add_message_handler("heartbeat", self._on_heartbeat)
        self.zmq.set_default_handler(self._on_any)
        # 2. Start ZMQ
        await self.zmq.start()
        # 3. Register background tasks
        self.add_task(self._publish_loop())
        # 4. Call super LAST
        await super().start()

    async def stop(self):
        # 1. Call super FIRST (cancels tasks)
        await super().stop()
        # 2. Tear down ZMQ
        await self.zmq.stop()

    async def _on_heartbeat(self, message):
        logger.debug("heartbeat: %s", message)

    async def _on_any(self, topic: str, message):
        logger.debug("msg on %s: %s", topic, message)

    async def _publish_loop(self):
        import time
        while self.running:
            await self.zmq.publish(
                {"service": self.service_name, "ts": time.time()},
                "heartbeat",
            )
            await self._sleep_if_running(5.0)
```

---

## Error handling

```python
from experimance_common.zmq.config import ZmqException, ZmqTimeoutError

try:
    await zmq_component.receive()
except ZmqTimeoutError:
    pass   # normal; no message in window
except ZmqException as e:
    self.record_error(e, is_fatal=False)
```

---

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Messages never received | Handlers registered after `start()` | Register before `await zmq.start()` |
| Topic not matching | String vs. `MessageType` mismatch | Use `MessageType` values consistently |
| Subscription receives everything | Topic `""` subscribes to all | Use explicit topics unless catch-all is intended |
| Port already in use | Another instance running | Check with `ss -tlnp \| grep <port>` |
| Worker tasks lost | Controller not running yet | Start controller before workers |
| ZMQ hangs on shutdown | Component not stopped | Always `await zmq.stop()` in `stop()` |

---

## Files touched

| File | Role |
|------|------|
| `zmq/config.py` | Pydantic config models and ZMQ exceptions |
| `zmq/components.py` | Individual socket wrappers |
| `zmq/services.py` | Composed high-level services |
| `zmq/mocks.py` | Mock services for testing |
| `zmq/zmq_utils.py` | Low-level helpers |
| `schemas_base.py` | `MessageType` enum and message Pydantic models |
| `constants_base.py` | `DEFAULT_PORTS` and ZMQ address patterns |

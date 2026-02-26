# experimance_common.zmq

ZeroMQ communication layer for Experimance services. Uses a **composition-based** architecture — combine `BaseService` with a `PubSubService`, `WorkerService`, or `ControllerService` object rather than inheriting from a ZMQ-specific class.

## Quick start

```python
from experimance_common.base_service import BaseService
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.constants import DEFAULT_PORTS

class MyService(BaseService):
    def __init__(self):
        super().__init__(service_name="my-service")
        self.zmq = PubSubService(PubSubServiceConfig(
            name="my-service",
            publisher=PublisherConfig(address="tcp://*",
                                      port=DEFAULT_PORTS["events"]),
            subscriber=SubscriberConfig(address="tcp://localhost",
                                        port=DEFAULT_PORTS["events"],
                                        topics=["heartbeat"]),
        ))

    async def start(self):
        self.zmq.add_message_handler("heartbeat", self._on_heartbeat)
        await self.zmq.start()
        await super().start()       # always last

    async def stop(self):
        await super().stop()        # always first
        await self.zmq.stop()

    async def _on_heartbeat(self, message):
        pass
```

## Module contents

| File | Purpose |
|------|---------|
| `config.py` | `ZmqSocketConfig`, `PublisherConfig`, `SubscriberConfig`, `PushConfig`, `PullConfig`, `PubSubServiceConfig`, `WorkerServiceConfig`, `ControllerServiceConfig`; also `ZmqException`, `ZmqTimeoutError` |
| `components.py` | Low-level single-socket wrappers: `PublisherComponent`, `SubscriberComponent`, `PushComponent`, `PullComponent` |
| `services.py` | Composed high-level services: `PubSubService`, `WorkerService`, `ControllerService` |
| `mocks.py` | `MockPubSubService`, `MockWorkerService` for testing |
| `zmq_utils.py` | Low-level ZMQ helpers |

## Handler signatures

| Method | Handler signature |
|--------|-----------------|
| `service.add_message_handler(topic, fn)` | `async def fn(message)` |
| `service.set_default_handler(fn)` | `async def fn(topic, message)` |

## Full documentation

[docs/zmq.md](../../../docs/zmq.md)

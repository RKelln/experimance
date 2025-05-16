## Service Base Classes

The `experimance_common.service` module provides a hierarchy of service classes that standardize service behavior across the Experimance system. These classes handle common concerns like lifecycle management, graceful shutdown, error handling, and statistics tracking.

### Service Class Hierarchy

The service classes are organized as follows:

1. **BaseService**: Core functionality for all services (with or without ZMQ)
   - **BaseZmqService**: Adds ZeroMQ-specific functionality
     - **ZmqPublisherService**: For services that broadcast messages
     - **ZmqSubscriberService**: For services that listen to broadcasts
     - **ZmqPushService**: For services that distribute tasks
     - **ZmqPullService**: For services that consume tasks
     - **ZmqPublisherSubscriberService**: For services that both publish and subscribe
     - **ZmqControllerService**: For controller services (publish + listen + push + pull)
     - **ZmqWorkerService**: For worker services (subscribe + pull + push responses)

### Common Functionality

All service classes provide:

- Standard lifecycle methods (`start()`, `stop()`, `run()`)
- Signal handling for graceful termination
- Statistics tracking and reporting
- Error handling and recovery
- Task management for concurrent operations

**ZMQ services additionally provide**:
- Socket initialization and proper cleanup
- Standard communication patterns
- Message handling

### Basic Usage

#### Creating a Service without ZeroMQ

For services that don't need ZeroMQ communication, you can use the BaseService class directly:

```python
from experimance_common.service import BaseService
import asyncio
import time

class LoggingService(BaseService):
    def __init__(self, name):
        super().__init__(
            service_name=name,
            service_type="logging-service"
        )
        self.log_count = 0
        
    async def start(self):
        await super().start()
        # Register custom tasks
        self._register_task(self.log_periodic_messages())
        
    async def log_periodic_messages(self):
        while self.running:
            self.log_count += 1
            print(f"Service {self.service_name} is running - log #{self.log_count}")
            await asyncio.sleep(2)

async def main():
    service = LoggingService("simple-logger")
    try:
        await service.start()
        await service.run()
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Creating a Simple Publisher Service

```python
from experimance_common.service import ZmqPublisherService
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq_utils import MessageType
import asyncio
import time

class MyPublisherService(ZmqPublisherService):
    def __init__(self, name):
        super().__init__(
            service_name=name,
            pub_address=f"tcp://*:{DEFAULT_PORTS['example_pub']}",
            heartbeat_topic="my-service.heartbeat"
        )
        
    async def start(self):
        await super().start()
        # Register custom tasks
        self._register_task(self.send_periodic_updates())
        
    async def send_periodic_updates(self):
        while self.running:
            message = {
                "type": MessageType.IDLE_STATUS,
                "timestamp": time.time(),
                "status": "active"
            }
            await self.publish_message(message, topic="my-service.status")
            await asyncio.sleep(5)

async def main():
    service = MyPublisherService("my-publisher")
    try:
        await service.start()
        await service.run()
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Creating a Controller-Worker System

See the complete example in `utils/examples/zmq_service_example.py` which demonstrates:

1. A controller that distributes tasks and broadcasts messages
2. Workers that process tasks and return results
3. Proper message handlers and task processing
4. Graceful shutdown and error handling

### Best Practices

1. **Use the Right Base Class**: Choose the most appropriate base class based on your service's communication needs
2. **Keep Services Focused**: Each service should have a clear, well-defined responsibility
3. **Register Message Handlers**: Use the handler registration methods to keep code organized
4. **Implement Graceful Shutdown**: Always respond to shutdown signals by cleaning up resources
5. **Handle Errors Properly**: Catch and handle exceptions in message and task handlers
6. **Use Lifecycle Methods**: Implement `start()`, `stop()`, and `run()` methods for consistent behavior
7. **Monitor Service Health**: Use the built-in statistics tracking to monitor service performance

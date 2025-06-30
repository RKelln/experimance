import asyncio
from contextlib import suppress, asynccontextmanager
import logging
from experimance_common.constants import TICK
import time
from typing import Optional, Any, Callable, Union, TypeVar, List, Dict, AsyncIterator

from experimance_common.base_service import BaseService, ServiceState, ServiceStatus
from experimance_common.zmq.config import ZmqTimeoutError
from experimance_common.schemas import MessageType

logger = logging.getLogger(__name__)

# Type variable for services
T = TypeVar('T', bound=BaseService)

SIMULATE_NETWORK_DELAY = 0.1

# ============================================================================
# ZMQ Mock Classes - Reusable across all service tests
# ============================================================================

class MockZmqSocketBase:
    """Base class for mock ZMQ sockets with common setup logic."""
    
    def __init__(self, address, topic=None, topics=None, **kwargs):
        self.closed = False
        self.messages = []
        self.address = address
        self.topic = str(topic) if topic else None
        self.topics = topics or ([topic] if topic else [])
        self.topics = [str(t) for t in self.topics]  # Ensure topics are strings
        self.use_asyncio = kwargs.get('use_asyncio', True)
        
        # Store any additional kwargs for subclass use
        self.kwargs = kwargs
    
    def close(self):
        """Close the socket."""
        self.closed = True
        
    def __repr__(self):
        return f"{self.__class__.__name__}(address='{self.address}', topic='{self.topic}')"


class MockZmqSocketTimeout(MockZmqSocketBase):
    """Mock ZMQ socket that simulates timeouts for all operations."""
    
    async def publish_async(self, message):
        """Mock publishing a message with timeout."""
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        raise ZmqTimeoutError("Mock publishing timeout")
    
    def publish(self, message):
        """Sync version of publish with timeout."""
        raise ZmqTimeoutError("Mock publishing timeout")
    
    async def receive_async(self):
        """Mock receiving a message with timeout."""
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        raise ZmqTimeoutError("Mock receiving timeout")
    
    def receive(self):
        """Sync version of receive with timeout."""
        raise ZmqTimeoutError("Mock receiving timeout")
    
    async def push_async(self, message):
        """Mock pushing a message with timeout."""
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        raise ZmqTimeoutError("Mock pushing timeout")
    
    def push(self, message):
        """Sync version of push with timeout."""
        raise ZmqTimeoutError("Mock pushing timeout")
    
    async def pull_async(self):
        """Mock pulling a message with timeout."""
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        raise ZmqTimeoutError("Mock pulling timeout")
    
    def pull(self):
        """Sync version of pull with timeout."""
        raise ZmqTimeoutError("Mock pulling timeout")


class MockZmqSocketWorking(MockZmqSocketBase):
    """Mock ZMQ socket that works for all operations."""
    
    async def publish_async(self, message):
        """Mock publishing a message successfully."""
        self.messages.append(message)
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        return True
    
    def publish(self, message):
        """Sync version of publish."""
        self.messages.append(message)
        return True
    
    async def receive_async(self):
        """Mock receiving a message successfully."""
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        
        if not self.messages:
            # Return a default test message
            return self.topic or "test-topic", {"type": "test", "timestamp": time.time()}
        return self.topic or "test-topic", self.messages.pop(0)
    
    def receive(self):
        """Sync version of receive."""
        if not self.messages:
            return self.topic or "test-topic", {"type": "test", "timestamp": time.time()}
        return self.topic or "test-topic", self.messages.pop(0)
    
    async def push_async(self, message):
        """Mock pushing a message successfully."""
        self.messages.append(message)
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        return True
    
    def push(self, message):
        """Sync version of push."""
        self.messages.append(message)
        return True
    
    async def pull_async(self):
        """Mock pulling a message successfully."""
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)  # Simulate network delay
        
        if not self.messages:
            return {"id": "test-id", "timestamp": time.time()}
        return self.messages.pop(0)
    
    def pull(self):
        """Sync version of pull."""
        if not self.messages:
            return {"id": "test-id", "timestamp": time.time()}
        return self.messages.pop(0)


class MockZmqPublisher(MockZmqSocketWorking):
    """Mock publisher for service tests."""
    
    def __init__(self, address, topic=None, **kwargs):
        super().__init__(address=address, topic=topic, **kwargs)
        self.published_count = 0
    
    async def publish_async(self, message):
        """Track published message count."""
        result = await super().publish_async(message)
        self.published_count += 1
        return result
    
    def publish(self, message):
        """Track published message count."""
        result = super().publish(message)
        self.published_count += 1
        return result


class MockZmqSubscriber(MockZmqSocketWorking):
    """Mock subscriber for service tests."""
    
    def __init__(self, address, topics=None, **kwargs):
        super().__init__(address=address, topics=topics, **kwargs)
        self.subscription_count = 0
        
        # Add default test messages for common message types
        self.add_test_message(MessageType.HEARTBEAT, {"timestamp": time.time()})
    
    def add_test_message(self, message_type, content=None):
        """Add a test message to be received."""
        content = content or {}
        message = {
            "type": message_type,
            "timestamp": time.time(),
            **content
        }
        # Find appropriate topic for this message type
        topic = self.topic or "test-topic"
        if self.topics and len(self.topics) > 0:
            # Use first topic that matches or first topic
            topic = self.topics[0]
        
        self.messages.append((topic, message))


class MockZmqPushSocket(MockZmqSocketWorking):
    """Mock push socket for service tests."""
    
    def __init__(self, address, **kwargs):
        super().__init__(address=address, **kwargs)


class MockZmqPullSocket(MockZmqSocketWorking):
    """Mock pull socket for service tests."""
    
    def __init__(self, address, **kwargs):
        super().__init__(address=address, **kwargs)


class MockZmqBindingPullSocket(MockZmqPullSocket):
    """Mock binding pull socket for controller/worker patterns."""
    pass


class MockZmqConnectingPushSocket(MockZmqPushSocket):
    """Mock connecting push socket for controller/worker patterns."""
    pass


# ============================================================================
# Service Test Utilities
# ============================================================================

async def wait_for_service_state(
    service: BaseService, 
    target_state: ServiceState = ServiceState.STOPPED, 
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to reach a particular state.
    
    This function is particularly useful when waiting for a service to reach 
    the STOPPED state after initiating a shutdown, especially in scenarios 
    where the service's run task might not complete normally (such as when a
    service stops itself from within the run method).
    
    Args:
        service: The service to monitor
        target_state: The state to wait for, defaults to STOPPED
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service state
        
    Raises:
        asyncio.TimeoutError: If service doesn't reach target state in time
    """
    logger.info(f"Waiting for service {service.service_name} to reach {target_state} state, current state: {service.state}")
    
    if service.state == target_state:
        logger.info(f"Service {service.service_name} is already in {target_state} state")
        return
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if service.state == target_state:
            logger.info(f"Service {service.service_name} reached {target_state} state after {time.monotonic() - start_time:.2f}s")
            return
        await asyncio.sleep(check_interval)
        
    logger.error(f"Timeout waiting for {service.service_name} to reach {target_state} state. Current state: {service.state}")
    assert False, f"Service {service.service_name} did not reach {target_state} state in {timeout}s (current state: {service.state})"

async def wait_for_service_status(
    service: BaseService, 
    target_status: ServiceStatus = ServiceStatus.HEALTHY, 
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to reach a particular health status.
    
    This function is particularly useful when waiting for a service's error status
    to change, such as when expecting a service to encounter an error during execution.
    
    Args:
        service: The service to monitor
        target_status: The status to wait for, defaults to HEALTHY
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service status
        
    Raises:
        asyncio.TimeoutError: If service doesn't reach target status in time
    """
    logger.info(f"Waiting for service {service.service_name} to reach {target_status} status, current status: {service.status}")
    
    if service.status == target_status:
        logger.info(f"Service {service.service_name} is already in {target_status} status")
        return
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if service.status == target_status:
            logger.info(f"Service {service.service_name} reached {target_status} status after {time.monotonic() - start_time:.2f}s")
            return
        await asyncio.sleep(check_interval)
        
    logger.error(f"Timeout waiting for {service.service_name} to reach {target_status} status. Current status: {service.status}")
    assert False, f"Service {service.service_name} did not reach {target_status} status in {timeout}s (current status: {service.status})"

async def wait_for_service_state_and_status(
    service: BaseService, 
    target_state: Optional[ServiceState] = None,
    target_status: Optional[ServiceStatus] = None,
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to reach a particular state and/or status.
    
    This function allows waiting for either a specific state, specific status,
    or both simultaneously. At least one of target_state or target_status must be provided.
    
    Args:
        service: The service to monitor
        target_state: The state to wait for (optional)
        target_status: The status to wait for (optional)
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service state/status
        
    Raises:
        ValueError: If neither target_state nor target_status is provided
        asyncio.TimeoutError: If service doesn't reach target state/status in time
    """
    if target_state is None and target_status is None:
        raise ValueError("At least one of target_state or target_status must be provided")
        
    # Prepare condition descriptions for logging
    conditions = []
    if target_state is not None:
        conditions.append(f"state={target_state}")
    if target_status is not None:
        conditions.append(f"status={target_status}")
    condition_desc = " and ".join(conditions)
    
    logger.info(f"Waiting for service {service.service_name} to reach {condition_desc}, "
                f"current state={service.state}, status={service.status}")
    
    # Check if already in target state/status
    if ((target_state is None or service.state == target_state) and 
        (target_status is None or service.status == target_status)):
        logger.info(f"Service {service.service_name} is already in {condition_desc}")
        return
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        # Check if conditions are met
        state_condition_met = target_state is None or service.state == target_state
        status_condition_met = target_status is None or service.status == target_status
        
        if state_condition_met and status_condition_met:
            elapsed = time.monotonic() - start_time
            logger.info(f"Service {service.service_name} reached {condition_desc} after {elapsed:.2f}s")
            return
            
        await asyncio.sleep(check_interval)
    
    # If we get here, the timeout was reached
    logger.error(f"Timeout waiting for {service.service_name} to reach {condition_desc}. "
                f"Current state={service.state}, status={service.status}")
    assert False, (f"Service {service.service_name} did not reach {condition_desc} in {timeout}s "
                  f"(current state={service.state}, status={service.status})")

async def wait_for_service_shutdown(
    run_task: asyncio.Task, 
    service: BaseService, 
    timeout: float = 5.0,
    check_interval: float = 0.1
):
    """
    Wait for a service to shut down completely, monitoring both the run task and service state.
    
    This helper ensures that both the run task completes and the service reaches the STOPPED state.
    It's particularly useful when testing signal handling and shutdown procedures.
    
    Args:
        run_task: The asyncio task running the service.run() method
        service: The service instance being monitored
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the service state
        
    Raises:
        asyncio.TimeoutError: If service doesn't reach STOPPED state in time
        AssertionError: If service doesn't reach proper shutdown state
    """
    logger.info(f"Waiting for {service.service_name} to shut down completely")
    
    start_time = time.monotonic()
    
    # First wait for the service to reach STOPPED state
    await wait_for_service_state(service, ServiceState.STOPPED, timeout, check_interval)
    
    # Now wait for the run task to complete (it should be done or nearly done)
    remaining_timeout = max(0.1, timeout - (time.monotonic() - start_time))
    try:
        logger.info(f"Waiting for {service.service_name} run task to complete, timeout: {remaining_timeout:.2f}s")
        await asyncio.wait_for(run_task, timeout=remaining_timeout)
        logger.info(f"Run task for {service.service_name} completed successfully")
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for {service.service_name} run task to complete")
        # Cancel it if it's still running
        if not run_task.done():
            logger.warning(f"Cancelling {service.service_name} run task that didn't complete")
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
        assert False, f"Run task for {service.service_name} did not complete in {remaining_timeout:.2f}s"
    except asyncio.CancelledError:
        logger.info(f"Run task for {service.service_name} was cancelled")
    
    # Extra safety check to ensure proper shutdown
    assert service.state == ServiceState.STOPPED, f"Service {service.service_name} should be in STOPPED state"
    
    logger.info(f"Service {service.service_name} shutdown confirmed after {time.monotonic() - start_time:.2f}s")

@asynccontextmanager
async def active_service(service: T, 
                        run_task_name: Optional[str] = None,
                        target_state: ServiceState = ServiceState.RUNNING,
                        setup_func: Optional[Callable[[T], Any]] = None) -> AsyncIterator[T]:
    """
    A context manager that handles service lifecycle for tests.
    
    This utility makes test code cleaner by handling the standard pattern of:
    1. Starting a service
    2. Creating a run task
    3. Waiting for it to reach a target state
    4. Running the test
    5. Stopping the service and cleaning up
    
    Args:
        service: The service to start and manage
        run_task_name: Optional name for the run task (for debugging)
        target_state: The service state to wait for before yielding
        setup_func: Optional function to run on the service before starting it
        
    Yields:
        The started service ready for testing
        
    Example:
        ```python
        async def test_example():
            service = MyService(...)
            
            async with active_service(service) as active:
                # Test the running service here
                result = await active.some_method()
                assert result is True
            
            # Service is now stopped and cleaned up
        ```
    """
    run_task = None
    task_name = run_task_name or f"{service.service_name}-run-task"
    
    try:
        # Run any setup function if provided
        if setup_func:
            setup_func(service)
            
        # Start the service
        await service.start()
        
        # Create a task to run the service
        run_task = asyncio.create_task(service.run(), name=task_name)
        
        # Wait for service to reach the target state
        await wait_for_service_state(service, target_state)
        
        # Yield the service to the context block
        yield service
        
    finally:
        # Clean up regardless of whether the context block completed normally or raised an exception
        
        # Only attempt to stop if the service is not already stopping or stopped
        if service.state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
            await service.stop()
        
        # Clean up the run task if it exists
        if run_task:
            if not run_task.done():
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task
                    
        # Wait for service to fully shut down
        await wait_for_service_state(service, ServiceState.STOPPED)


def debug_service_tasks(service: BaseService):
    """
    Print debug information about the service's tasks.

    This function is useful for debugging and understanding the state of tasks
    within a service, especially during testing.

    Args:
        service: The service whose tasks to debug
    """
    logger.debug(f"Tasks for service {service.service_name}:")
    
    # Get task names using the helper method
    task_names = service.get_task_names()
    
    # Log more detailed information about each task
    for i, task_coro in enumerate(service.tasks):
        name = task_names[i] if i < len(task_names) else str(task_coro)
        
        if hasattr(task_coro, 'get_name'):
            # It's a Task object
            logger.debug(f"  - Task Name: {name}, Type: {type(task_coro)}")
        elif hasattr(task_coro, '__name__'):
            # It's a coroutine function
            qualified_name = getattr(task_coro, '__qualname__', 'N/A')
            logger.debug(f"  - Task Name: {name}, Qualified Name: {qualified_name}, Type: {type(task_coro)}")
        else:
            # It's something else
            logger.debug(f"  - Task: {name}, Type: {type(task_coro)}")

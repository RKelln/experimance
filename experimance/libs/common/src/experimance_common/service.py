"""
Base service classes for Experimance services with and without ZeroMQ communication.

This module provides a hierarchy of service classes that standardize service 
behavior across the Experimance system. It includes:

1. BaseService: Core functionality for all services with lifecycle management
2. BaseZmqService: ZeroMQ-specific functionality extending BaseService
3. Role-specific ZMQ services (Publisher, Subscriber, Push, Pull)
4. Combined services for common patterns (PublisherSubscriber, Controller, Worker)

All services share common functionality like:
- Graceful shutdown handling with signal trapping
- Standard lifecycle methods (start, stop, run)
- Error handling with proper recovery
- Statistics tracking

ZeroMQ services additionally include:
- Proper initialization and cleanup of ZMQ sockets
- Standard communication patterns
"""

import asyncio
import logging
import signal
import time
import traceback
import inspect
import logging
import signal
import traceback  # Explicitly import traceback
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Coroutine

import zmq
import zmq.asyncio

from experimance_common.constants import (
    DEFAULT_PORTS,
    DEFAULT_TIMEOUT,
    HEARTBEAT_INTERVAL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RECV_TIMEOUT,
    HEARTBEAT_TOPIC
)

from experimance_common.zmq_utils import (
    ZmqPublisher,
    ZmqSubscriber,
    ZmqPushSocket,
    ZmqPullSocket,
    MessageType,
    ZmqTimeoutError,
    ZmqException
)

from experimance_common.service_state import ServiceState, StateManager
from experimance_common.service_decorators import lifecycle_service

# Configure logging
logger = logging.getLogger(__name__)

class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"    # Service is operating normally
    WARNING = "warning"    # Service encountered non-critical issues
    ERROR = "error"        # Service encountered serious but recoverable errors
    FATAL = "fatal"        # Service encountered unrecoverable errors


@lifecycle_service
class BaseService:
    """Base class for all services in the Experimance system.
    
    This class provides common functionality for all services:
    - Service lifecycle management (start, stop, run)
    - Signal handling for graceful shutdown
    - Statistics tracking for monitoring
    - Error handling and recovery
    
    Subclasses should implement their specific functionality
    by extending this class and implementing the necessary methods.
    """
    
    def __init__(self, service_name: str, service_type: str = "generic"):
        """Initialize the base service.
        
        Args:
            service_name: Unique name for this service instance
            service_type: Type of service (for logging and monitoring)
        """
        self.service_name = service_name
        self.service_type = service_type
        
        # Initialize state management
        self._state_manager = StateManager(service_name, ServiceState.INITIALIZING)
        self.status = ServiceStatus.HEALTHY  # Initial status
        self.tasks = []
        
        # Statistics
        self.start_time = time.monotonic()
        self.messages_sent = 0
        self.messages_received = 0
        self.errors = 0
        self.last_stats_time = self.start_time
        
        # State control
        self._stop_lock = asyncio.Lock() # Lock to serialize stop() execution
        self._run_task_handle: Optional[asyncio.Task] = None # Handle to the main run() task
        
        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
            signal.signal(sig, self._signal_handler)
            
        # Set state to INITIALIZED now that initialization is complete
        self._state_manager.state = ServiceState.INITIALIZED
    
    @property
    def state(self) -> ServiceState:
        """Get the current service state."""
        return self._state_manager.state
    
    @state.setter
    def state(self, new_state: ServiceState):
        """Set the service state."""
        self._state_manager.state = new_state
    
    def register_state_callback(self, state: ServiceState, callback: Callable[[], None]):
        """Register a callback for state transitions."""
        self._state_manager.register_state_callback(state, callback)
    
    async def wait_for_state(self, state: ServiceState, timeout: Optional[float] = None) -> bool:
        """Wait for a specific state."""
        return await self._state_manager.wait_for_state(state, timeout)
    
    @asynccontextmanager
    async def observe_state_change(self, expected_state: ServiceState, timeout: float = 5.0):
        """Observe a state change."""
        async with self._state_manager.observe_state_change(expected_state, timeout):
            yield
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals in non-asyncio contexts.
        
        This ensures proper cleanup on service termination.
        """
        # Only process the signal if we're not already stopping or stopped
        if self.state in [ServiceState.STOPPING, ServiceState.STOPPED]:
            logger.debug(f"Signal handler called while in {self.state} state, ignoring")
            return
            
        signal_name = signal.Signals(signum).name
            
        logger.info(f"Received signal {signal_name} ({signum}), shutting down gracefully...")
        
        # Update state to stopping
        self.state = ServiceState.STOPPING
    
    def add_task(self, task_or_coroutine: Union[asyncio.Task, Coroutine]):
        """Register a task to be executed in the service's run loop.
        
        Args:
            task_or_coroutine: Coroutine or asyncio.Task to execute
                               Tasks will be scheduled immediately. Coroutines will be
                               stored and converted to tasks when run() is called.
        """
        self.tasks.append(task_or_coroutine)
    
    async def _sleep_if_running(self, duration: float) -> bool:
        """Sleep for duration and return whether service is still running.
        
        Use this when you need to sleep in the middle of a loop and want to
        check if the service should continue running afterward.
        
        Args:
            duration: Time to sleep in seconds
            
        Returns:
            True if service is still running after sleep, False otherwise
            
        Example:
            # In the middle of a loop where you need to continue processing after sleep
            if not await self._sleep_if_running(1.0):
                break
            # Continue with more processing...
        """
        if self.state != ServiceState.RUNNING:
            return False
        await asyncio.sleep(duration)
        return self.state == ServiceState.RUNNING

    async def display_stats(self):
        """Periodically display service statistics."""
        while self.state == ServiceState.RUNNING:
            if not await self._sleep_if_running(10):
                break
            
            now = time.monotonic()
            elapsed = now - self.start_time
            elapsed_since_last = now - self.last_stats_time
            
            # Calculate message rates
            sent_rate = self.messages_sent / elapsed if elapsed > 0 else 0
            received_rate = self.messages_received / elapsed if elapsed > 0 else 0
            
            # Format uptime as hours:minutes:seconds
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            
            # Prepare and log statistics
            stats = {
                "service": self.service_name,
                "type": self.service_type,
                "state": self.state,
                "status": self.status,
                "uptime": uptime_str,
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "errors": self.errors,
                "msg_send_rate": f"{sent_rate:.2f}/s",
                "msg_recv_rate": f"{received_rate:.2f}/s"
            }
            
            logger.info(f"Stats for {self.service_name}: {stats}")
            self.last_stats_time = now
    
    
    async def start(self):
        """Start the service.
        
        This method should be extended by subclasses to initialize 
        their specific components before calling super().start().
        """
        logger.info(f"Starting {self.service_type} service: {self.service_name}")
        self.start_time = time.monotonic()
        
        # Always include the stats display task
        # This registers the coroutine to be executed when run() is called
        self.add_task(self.display_stats())
    
    
    async def stop(self):
        """Stop the service and clean up resources.
        
        This method ensures all tasks are properly cancelled
        and resources are cleaned up.
        """
        async with self._stop_lock:
            if self.state == ServiceState.STOPPED:
                logger.debug(f"Service {self.service_name} is already STOPPED. Ignoring stop call.")
                return

            logger.info(f"Stopping {self.service_name} (lock acquired)...")
            self.state = ServiceState.STOPPING
            
            # Cancel the main run task if it exists and is not this task
            current_task = asyncio.current_task()
            if self._run_task_handle and not self._run_task_handle.done():
                if self._run_task_handle is current_task:
                    logger.info(f"Main run task for {self.service_name} is the current task (stop() called from run() or its signal handler). Requesting cancellation, but cannot await self here.")
                    self._run_task_handle.cancel()
                    # We cannot await self._run_task_handle here as it would deadlock.
                    # The task will be cancelled, and its own exception handling (e.g., CancelledError in run())
                    # will proceed. The finally block in run() should not call stop() again due to state check.
                else:
                    logger.info(f"Cancelling main run task for {self.service_name}.")
                    self._run_task_handle.cancel()
                    try:
                        logger.debug(f"Waiting for main run task of {self.service_name} to complete cancellation.")
                        await self._run_task_handle
                        logger.debug(f"Main run task of {self.service_name} completed after cancellation.")
                    except asyncio.CancelledError:
                        logger.debug(f"Main run task of {self.service_name} was cancelled as expected.")
                    except Exception as e:
                        logger.warning(f"Main run task of {self.service_name} raised an exception during/after cancellation: {e!r}", exc_info=True)
            elif self._run_task_handle and self._run_task_handle.done():
                logger.debug(f"Main run task for {self.service_name} was already done.")
            elif not self._run_task_handle:
                logger.debug(f"No main run task handle found for {self.service_name} to cancel.")

            await self._clear_tasks()  # Clear any registered tasks
            
            self.state = ServiceState.STOPPED
            logger.info(f"Service {self.service_name} stopped")
    
    async def _clear_tasks(self):
        """Clear all registered tasks.
        
        This is used to reset the task list after stopping the service.
        """
        # Clean up all registered tasks
        if self.tasks:
            logger.debug(f"Cleaning up {len(self.tasks)} registered tasks for {self.service_name}.")
            for task in self.tasks:
                try:
                    if isinstance(task, asyncio.Task) and not task.done():
                        # Cancel any non-completed Task
                        logger.debug(f"Cancelling uncompleted task in {self.service_name}: {task.get_name() if hasattr(task, 'get_name') else task}")
                        task.cancel()
                    elif inspect.iscoroutine(task):
                        # Ensure coroutine is awaited if needed
                        try:
                            await asyncio.wait_for(task, timeout=3.0)
                        except asyncio.TimeoutError:
                            logger.warning(f"Task {task} did not complete within timeout during cleanup.")
                        # Close stored coroutines to prevent resource leaks
                        logger.debug(f"Closing coroutine in {self.service_name}: {task.__name__ if hasattr(task, '__name__') else task}")
                        task.close()
                except Exception as e:
                    logger.debug(f"Error clearing task {task} in {self.service_name}: {e}", exc_info=False)
            self.tasks = [] # Clear the list after attempting to close

    async def run(self):
        """Run the service until stopped.
        
        This method executes all registered tasks concurrently and
        handles proper cleanup on termination.
        """
        if not self.tasks:
            raise RuntimeError("No tasks registered for service")
        
        logger.info(f"Service {self.service_name} running")
        
        try:
            self._run_task_handle = asyncio.current_task() # Store handle to this task
            # Register asyncio-specific signal handlers
            loop = asyncio.get_running_loop()
            
            # Set up a custom exception handler to handle CancelledError in futures
            # This prevents the ZeroMQ _chain callback CancelledError from propagating
            def custom_exception_handler(loop, context):
                exception = context.get('exception')
                if isinstance(exception, asyncio.CancelledError):
                    # Just log and ignore CancelledError - it's expected during shutdown
                    logger.debug("Suppressed asyncio.CancelledError in event loop exception handler")
                    return
                # For other exceptions, use the default handler
                loop.default_exception_handler(context)
            
            # Save the original handler and set our custom one
            original_exception_handler = loop.get_exception_handler()
            loop.set_exception_handler(custom_exception_handler)
            
            # Register signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
                loop.add_signal_handler(
                    sig,
                    # Use a lambda to ensure the current self is captured.
                    # Assign to a variable to help with clarity if needed.
                    lambda s=sig: asyncio.create_task(self._handle_signal_async(s))
                )
            
            # Convert any coroutines to tasks and prepare for execution
            task_objects = []
            for task in self.tasks:
                if isinstance(task, asyncio.Task):
                    # If it's already a Task, just use it directly
                    task_objects.append(task)
                elif inspect.iscoroutine(task):
                    # Convert coroutines to tasks here when run() is called
                    # Check for a stored task name from _register_task
                    task_name = getattr(task, "_task_name", None)
                    if not task_name:
                        task_name = f"{self.service_name}-{task.__name__ if hasattr(task, '__name__') else 'task'}"
                    logger.debug(f"Converting coroutine to task with name: {task_name}")
                    task_obj = asyncio.create_task(task, name=task_name)
                    task_objects.append(task_obj)
                else:
                    logger.warning(f"Unsupported task type found in tasks list: {type(task)}: {task}")
                    
            # Run all tasks concurrently
            # Set up done callbacks to detect errors as they happen
            for task in task_objects:
                task.add_done_callback(self._task_done_callback)
                
            # Use asyncio.gather with return_exceptions=True to allow proper cleanup
            try:
                results = await asyncio.gather(*task_objects, return_exceptions=True)
                
                task_errors_found = False
                for result in results:
                    if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                        logger.error(f"Task error in service {self.service_name} during gather: {result}")
                        # Assuming traceback is imported at the top of the file
                        tb_str = ''.join(traceback.format_exception(type(result), result, result.__traceback__))
                        logger.debug(f"Traceback for task error:\\n{tb_str}")
                        self.record_error(result)
                        task_errors_found = True
                
                if task_errors_found:
                    self.status = ServiceStatus.ERROR
                    logger.info(f"Service {self.service_name} encountered task errors. Status set to ERROR. Stop will be handled by finally block.")
            except asyncio.CancelledError:
                # If the gather itself is cancelled, we still want to check for task exceptions
                logger.info(f"Gather was cancelled for {self.service_name}, checking individual tasks for errors")
                for task in task_objects:
                    if task.done() and not task.cancelled():
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(f"Task error in {task.get_name()} after cancellation: {exc}")
                                self.record_error(exc)
                                self.status = ServiceStatus.ERROR
                        except (asyncio.CancelledError, asyncio.InvalidStateError):
                            # Task was cancelled or not done, just skip
                            pass
                raise  # Re-raise the CancelledError
        # This CancelledError is for the run() task itself being cancelled.
        except asyncio.CancelledError:
            logger.info(f"Service {self.service_name} run task itself was cancelled, initiating stop.")
            # The finally block will handle calling stop().
        
        # This Exception is for other unexpected errors directly within the run() method's logic,
        # not from the tasks managed by gather.
        except Exception as e:
            logger.error(f"Unexpected error in service {self.service_name} run execution: {e}")
            # Assuming traceback is imported at the top of the file
            logger.debug(traceback.format_exc()) 
            self.record_error(e, is_fatal=True)
            # Re-raise to indicate a more fundamental issue with the service's run logic itself.
            raise
        finally:
            logger.debug(f"Service {self.service_name} run() method's finally block reached. Current state: {self.state}")
            # Only call stop() from here if a stop operation is NOT already in progress
            # AND the service is not already stopped.
            # This handles cases where run() exits due to an error or normal completion of its tasks,
            # without an external stop signal.
            if self.state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
                logger.info(f"Service {self.service_name} run() method ending (state: {self.state}). "
                            f"Stop not yet initiated by other means. Calling self.stop().")
                try:
                    await self.stop()
                except Exception as e:
                    logger.error(f"Error during self.stop() called from run() finally for {self.service_name}: {e}", exc_info=True)
                    self.record_error(e)
            elif self.state == ServiceState.STOPPING:
                logger.info(f"Service {self.service_name} run() method's finally block: A stop operation is already in progress "
                            f"(state: {self.state}). Letting it complete.")
            elif self.state == ServiceState.STOPPED:
                logger.info(f"Service {self.service_name} run() method's finally block: Service already STOPPED.")
            
            logger.info(f"Service {self.service_name} run() method completed. Final state: {self.state}")

    async def _handle_signal_async(self, sig):
        """Handle signals in the asyncio event loop by calling stop() or handling SIGTSTP."""
        signal_name = signal.Signals(sig).name
        logger.info(f"Received signal {signal_name} in asyncio event loop for {self.service_name}")

        # Prevent re-entrant calls or acting on signals if already fully stopped.
        # If stop() is already in progress, stop() itself is idempotent.
        if self.state == ServiceState.STOPPED:
            logger.debug(f"Service {self.service_name} is already STOPPED. Ignoring signal {signal_name}.")
            return

        if self.state == ServiceState.STOPPING:
            logger.debug(f"Service {self.service_name} is already STOPPING. Signal {signal_name} received again.")
            # Potentially, here you could implement a force stop mechanism if a second signal is received
            # For now, we let the current stop() proceed.
            return

        logger.info(f"Service {self.service_name} initiating shutdown via stop() due to signal {signal_name}")
        try:
            # Call stop() directly. stop() is responsible for setting the state
            await self.stop()
        except Exception as e:
            logger.error(f"Error during signal-initiated stop for {self.service_name}: {e}", exc_info=True)
            # Ensure the service is in a non-operational state even if stop fails.
            self.state = ServiceState.STOPPED
            self.status = ServiceStatus.ERROR

    def _task_done_callback(self, task):
        """Callback for completed tasks to immediately detect errors.
        
        This callback is called when a task completes, is cancelled, or raises an exception.
        It's used to detect errors as they happen rather than waiting for all tasks to complete.
        """
        try:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc:
                    task_name = task.get_name()
                    logger.error(f"Error detected in task {task_name} via callback: {exc}")
                    self.record_error(exc)
                    
                    # If the service is still running, schedule a stop
                    if self.state == ServiceState.RUNNING:
                        logger.warning(f"Scheduling service {self.service_name} to stop due to task error")
                        # We can't call stop() directly from this callback as it could deadlock,
                        # so we schedule it to run soon in the event loop
                        asyncio.create_task(self.stop(), name=f"{self.service_name}-error-stop")
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            # Task was cancelled or not done, just skip
            pass
            
    def record_error(self, error: Exception, is_fatal: bool = False):
        """Record an error and update service status.
        
        Args:
            error: The exception that occurred
            is_fatal: Whether this error should mark the service as fatally errored
        """
        self.errors += 1
        if is_fatal:
            self.status = ServiceStatus.FATAL
            logger.error(f"Fatal error in service {self.service_name}: {error!r}", exc_info=True)
        else:
            self.status = ServiceStatus.ERROR
            logger.error(f"Error in service {self.service_name}: {error!r}", exc_info=True)
            
    def reset_error_status(self):
        """Reset the service's error status to HEALTHY.
        
        Call this after recovering from an error condition.
        """
        self.status = ServiceStatus.HEALTHY
        logger.info(f"Service {self.service_name} error status reset to HEALTHY")


class BaseZmqService(BaseService):
    """Base class for ZeroMQ-based services in the Experimance system.
    
    This class extends BaseService with ZeroMQ-specific functionality:
    - ZMQ socket registration and cleanup
    - Common ZMQ communication patterns
    
    Subclasses should implement their specific communication patterns
    by extending this class and implementing the necessary methods.
    """
    
    
    def __init__(self, service_name: str, service_type: str = "zmq-service"):
        """Initialize the base ZMQ service.
        
        Args:
            service_name: Unique name for this service instance
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        
        # ZMQ sockets - to be initialized by subclasses
        self._sockets = []
        self._zmq_sockets_closed = False # Initialize flag
    
    def register_socket(self, socket):
        """Register a ZMQ socket for automatic cleanup.
        
        Args:
            socket: ZMQ socket wrapper to register
        """
        self._sockets.append(socket)
    
    
    async def stop(self):
        """Stop the service and clean up ZMQ resources.
        
        This method ensures all ZMQ sockets are properly closed
        in addition to the standard service cleanup.
        """
        logger.debug(f"Entering BaseZmqService.stop() for {self.service_name}. Current state: {self.state}")

        # Close ZMQ sockets first. This should happen before tasks that might use them 
        # are cancelled by super().stop(). This needs to be idempotent.
        if not self._zmq_sockets_closed:
            logger.info(f"Closing ZMQ sockets for {self.service_name}...")
            
            # Give pending operations a chance to complete or be cancelled
            # This helps prevent CancelledError exceptions in callbacks
            try:
                # Short sleep to allow any pending async operations to complete
                # or at least reach a cancellable state
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.debug(f"Sleep interrupted during ZMQ socket cleanup for {self.service_name}")
            
            socket_errors = 0
            # Iterate over a copy if closing modifies the list, or ensure socket.close() is safe
            for socket_obj in reversed(list(self._sockets)): # Iterate over a copy
                if socket_obj: # Check if socket_obj is not None
                    try:
                        logger.debug(f"Closing socket: {type(socket_obj).__name__}")
                        # Make sure linger is set to 0 for immediate close
                        if hasattr(socket_obj, 'socket') and hasattr(socket_obj.socket, 'set'):
                            try:
                                socket_obj.socket.set(zmq.LINGER, 0)
                            except Exception as e:
                                logger.debug(f"Could not set LINGER on socket: {e}")
                        
                        socket_obj.close() # Assuming this is synchronous and idempotent
                    except asyncio.CancelledError:
                        # Ignore CancelledError, as this might happen if the socket is already closed
                        logger.debug(f"CancelledError while closing socket {type(socket_obj).__name__} in {self.service_name}")
                    except Exception as e:
                        logger.warning(f"Error closing ZMQ socket {type(socket_obj).__name__} in {self.service_name}: {e}")
                        socket_errors += 1
            
            if socket_errors > 0:
                logger.warning(f"Encountered {socket_errors} errors while closing ZMQ sockets for {self.service_name}")
            
            self._zmq_sockets_closed = True
            # Clear the original list after closing all sockets from the copy
            self._sockets.clear()
        else:
            logger.debug(f"ZMQ sockets for {self.service_name} already marked as closed.")

        # Delegate to the base class stop method for general task cancellation and state management.
        logger.debug(f"Calling super().stop() from BaseZmqService.stop() for {self.service_name}")
        await super().stop()
        
        logger.debug(f"Exiting BaseZmqService.stop() for {self.service_name}. Final state from super: {self.state}")


class ZmqPublisherService(BaseZmqService):
    """Service that publishes messages on specific topics.
    
    This service type establishes a ZeroMQ PUB socket to broadcast
    messages to subscribing services.
    """
    
    
    def __init__(self, service_name: str, 
                 pub_address: str, 
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "publisher"):
        """Initialize a publisher service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.pub_address = pub_address
        self.heartbeat_topic = heartbeat_topic
        self.publisher:Optional[ZmqPublisher] = None
    
    
    async def start(self):
        """Start the publisher service."""
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Register heartbeat task - _register_task will automatically create a Task
        self.add_task(self.send_heartbeat())
        
        await super().start()
    
    async def send_heartbeat(self, interval: float = HEARTBEAT_INTERVAL):
        """Send periodic heartbeat messages.
        
        Args:
            interval: Time between heartbeats in seconds
        """
        while self.state == ServiceState.RUNNING:
            try:
                heartbeat = {
                    "type": MessageType.HEARTBEAT,
                    "timestamp": time.time(),
                    "service": self.service_name,
                    "state": self.state
                }
                
                if self.publisher:
                    success = await self.publisher.publish_async(heartbeat)
                    if success:
                        logger.debug(f"Sent heartbeat: {self.service_name}")
                        self.messages_sent += 1
                    else:
                        logger.warning("Failed to send heartbeat")
                        self.errors += 1
                else:
                    logger.warning("Cannot send heartbeat: publisher not initialized")
                    self.errors += 1
            
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                self.errors += 1
            
            await asyncio.sleep(interval)
    
    async def publish_message(self, message: Dict[str, Any], topic: Optional[str] = None) -> bool:
        """Publish a message to subscribers.
        
        Args:
            message: Message to publish
            topic: Topic to publish on (if None, uses the default heartbeat topic)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.publisher:
            logger.error("Cannot publish message: publisher not initialized")
            self.errors += 1
            return False
        
        # If topic provided, create a new publisher or use existing one with that topic
        publisher = self.publisher
        if topic is not None and topic != self.heartbeat_topic:
            publisher = ZmqPublisher(self.pub_address, topic)
            try:
                success = await publisher.publish_async(message)
                if success:
                    self.messages_sent += 1
                else:
                    self.errors += 1
                return success
            finally:
                publisher.close()
        else:
            # Use the default publisher
            try:
                success = await self.publisher.publish_async(message)
                if success:
                    self.messages_sent += 1
                else:
                    self.errors += 1
                return success
            except Exception as e:
                logger.error(f"Error publishing message: {e}")
                self.errors += 1
                return False


class ZmqSubscriberService(BaseZmqService):
    """Service that subscribes to messages on specific topics.
    
    This service type establishes a ZeroMQ SUB socket to receive
    broadcasts from publishing services.
    """
    
    
    def __init__(self, service_name: str, 
                 sub_address: str, 
                 topics: List[str],
                 service_type: str = "subscriber"):
        """Initialize a subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.sub_address = sub_address
        self.topics = topics
        self.subscriber = None
        self.message_handlers = {}
    
    
    async def start(self):
        """Start the subscriber service."""
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Register message listening task - _register_task will automatically create a Task
        self.add_task(self.listen_for_messages())
        
        await super().start()
    
    def register_handler(self, topic: str, 
                        handler: Union[Callable[[Dict[str, Any]], None], 
                                      Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]]):
        """Register a handler for a specific topic.
        
        Accepts both synchronous and asynchronous handler functions.
        
        Args:
            topic: Topic to handle messages for
            handler: Function to call with message data (can be sync or async)
        """
        if topic not in self.topics:
            logger.warning(f"Registering handler for topic {topic} which is not in subscription list")
        
        self.message_handlers[topic] = handler
    
    async def listen_for_messages(self):
        """Listen for incoming messages on subscribed topics."""
        if not self.subscriber:
            logger.error("Cannot listen for messages: subscriber not initialized")
            return
            
        while self.state == ServiceState.RUNNING:
            try:
                topic, message = await self.subscriber.receive_async()
                logger.debug(f"Received message on {topic}: {message}")
                self.messages_received += 1
                
                # Process message with registered handler if any
                if topic in self.message_handlers:
                    try:
                        handler = self.message_handlers[topic]
                        # Check if the handler is a coroutine function and await if it is
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            # Call synchronous handler directly
                            handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler for topic {topic}: {e}")
                        self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                self.errors += 1
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning


class ZmqPushService(BaseZmqService):
    """Service that pushes tasks to workers.
    
    This service type establishes a ZeroMQ PUSH socket to distribute
    tasks to pulling workers.
    """
    
    
    def __init__(self, service_name: str, 
                 push_address: str,
                 service_type: str = "push"):
        """Initialize a push service.
        
        Args:
            service_name: Unique name for this service instance
            push_address: ZeroMQ address to bind push socket to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.push_address = push_address
        self.push_socket = None
    
    
    async def start(self):
        """Start the push service."""
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        await super().start()
    
    async def push_task(self, task: Dict[str, Any]) -> bool:
        """Push a task to workers.
        
        Args:
            task: Task data to send
            
        Returns:
            True if task was sent successfully, False otherwise
        """
        if not self.push_socket:
            logger.error("Cannot push task: push socket not initialized")
            self.errors += 1
            return False
        
        try:
            success = await self.push_socket.push_async(task)
            if success:
                logger.debug(f"Pushed task: {task.get('id', 'unknown')}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to push task: {task.get('id', 'unknown')}")
                self.errors += 1
            return success
        except Exception as e:
            logger.error(f"Error pushing task: {e}")
            self.errors += 1
            return False


class ZmqPullService(BaseZmqService):
    """Service that pulls tasks from pushers.
    
    This service type establishes a ZeroMQ PULL socket to receive
    tasks from pushing services.
    """
    
    
    def __init__(self, service_name: str, 
                 pull_address: str,
                 service_type: str = "pull"):
        """Initialize a pull service.
        
        Args:
            service_name: Unique name for this service instance
            pull_address: ZeroMQ address to connect pull socket to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.pull_address = pull_address
        self.pull_socket = None
        self.task_handler = None
    
    
    async def start(self):
        """Start the pull service."""
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register message listening task
        self.add_task(self.pull_tasks())
        
        await super().start()
    
    def register_task_handler(self, handler: Callable[[Dict[str, Any]], Coroutine]):
        """Register a handler for incoming tasks.
        
        Args:
            handler: Async function to call with task data
        """
        self.task_handler = handler
    
    async def pull_tasks(self):
        """Pull and process tasks."""
        if not self.pull_socket:
            logger.error("Cannot pull tasks: pull socket not initialized")
            return
            
        while self.state == ServiceState.RUNNING:
            try:
                task = await self.pull_socket.pull_async()
                if task:
                    logger.debug(f"Received task: {task.get('id', 'unknown')}")
                    self.messages_received += 1
                    
                    # Process task with registered handler if any
                    if self.task_handler:
                        try:
                            await self.task_handler(task)
                        except Exception as e:
                            logger.error(f"Error in task handler: {e}")
                            self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error pulling task: {e}")
                self.errors += 1
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning


class ZmqPublisherSubscriberService(ZmqPublisherService, ZmqSubscriberService):
    """Service that both publishes and subscribes to messages.
    
    This combined service type is suitable for services that need to
    both broadcast their state and listen for events from other services.
    """
    
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 topics: List[str],
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "pubsub"):
        """Initialize a publisher-subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type of service (for logging and monitoring)
        """
        BaseZmqService.__init__(self, service_name, service_type)
        self.pub_address = pub_address
        self.sub_address = sub_address
        self.topics = topics
        self.heartbeat_topic = heartbeat_topic
        self.publisher = None
        self.subscriber = None
        self.message_handlers = {}
    
    
    async def start(self):
        """Start the publisher-subscriber service."""
        # Initialize publisher
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        
        await BaseZmqService.start(self)


class ZmqControllerService(ZmqPublisherSubscriberService, ZmqPushService, ZmqPullService):  # Added ZmqPullService
    """Controller service that publishes events, listens for responses, and pushes tasks.
    
    This combined service is suitable for central coordinator services that
    need to broadcast messages, listen for responses, and distribute tasks.
    It also pulls responses from workers.
    """
    
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 push_address: str,
                 pull_address: str,
                 topics: List[str],
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "controller"):
        ZmqPublisherSubscriberService.__init__(
            self,
            service_name=service_name,
            pub_address=pub_address,
            sub_address=sub_address,
            topics=topics,
            heartbeat_topic=heartbeat_topic,
            service_type=service_type
        )
        ZmqPushService.__init__(self, service_name=service_name, push_address=push_address, service_type=service_type)
        ZmqPullService.__init__(self, service_name=service_name, pull_address=pull_address, service_type=service_type)

        # Register the handler for messages from the PULL socket
        #self.register_task_handler(self._handle_worker_response)
    
    
    async def start(self):
        """Start the controller service."""
        # Initialize publisher for broadcasting
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber for receiving responses
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize push socket for distributing tasks
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        # Initialize pull socket for receiving worker responses
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)


class ZmqWorkerService(ZmqSubscriberService, ZmqPullService, ZmqPushService):
    """Worker service that subscribes to events and pulls tasks.
    
    This combined service is suitable for worker services that
    need to listen for control messages and receive tasks to process.
    """
    
    
    def __init__(self, service_name: str,
                 sub_address: str,
                 pull_address: str,
                 push_address: Optional[str] = None,
                 topics: List[str] = [],
                 service_type: str = "worker"):
        """Initialize a worker service.
        
        Args:
            service_name: Unique name for this service instance
            sub_address: ZeroMQ address to connect subscriber to
            pull_address: ZeroMQ address to connect pull socket to
            push_address: Optional ZeroMQ address to bind push socket for responses
            topics: List of topics to subscribe to
            service_type: Type of service (for logging and monitoring)
        """
        ZmqSubscriberService.__init__(self, service_name, sub_address, topics, service_type)
        ZmqPullService.__init__(self, service_name, pull_address, service_type)

        self.push_address:Optional[str] = push_address
        if self.push_address is not None:
            ZmqPushService.__init__(self, service_name, self.push_address, service_type)
    
    
    async def start(self):
        """Start the worker service."""
        # Initialize subscriber for receiving control messages
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize pull socket for receiving tasks
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Initialize push socket for sending responses back (if address provided)
        if self.push_address:
            logger.info(f"Initializing push socket on {self.push_address}")
            self.push_socket = ZmqPushSocket(self.push_address)
            self.register_socket(self.push_socket)
        
        # Register tasks
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)
    
    async def send_response(self, response: Dict[str, Any]) -> bool:
        """Send a response back to the controller.
        
        Args:
            response: Response data to send
            
        Returns:
            True if response was sent successfully, False otherwise
        """
        if not self.push_socket:
            logger.error("Cannot send response: push socket not initialized")
            self.errors += 1
            return False
        
        try:
            success = await self.push_socket.push_async(response)
            if success:
                logger.debug(f"Sent response: {response.get('type', 'unknown')}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to send response: {response.get('type', 'unknown')}")
                self.errors += 1
            return success
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.errors += 1
            return False

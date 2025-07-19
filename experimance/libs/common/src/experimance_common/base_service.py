"""
Base service classes for Experimance services with and without ZeroMQ communication.

This module provides a hierarchy of service classes that standardize service 
behavior across the Experimance system. It includes:

1. BaseService: Core functionality for all services with lifecycle management
2. BaseZmqService: ZeroMQ-specific functionality extending BaseService

All services share common functionality like:
- Graceful shutdown handling with signal trapping
- Standard lifecycle methods (start, stop, run)
- Error handling with proper recovery
- Statistics tracking

ZeroMQ services additionally include:
- Proper initialization and cleanup of ZMQ sockets
- Standard communication patterns

For ZeroMQ-specific service implementations, see the experimance_common.zmq submodule.
"""

import asyncio
import inspect
import logging
import signal
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Coroutine

from experimance_common.logger import setup_logging
from experimance_common.service_state import ServiceState, StateManager
from experimance_common.service_decorators import lifecycle_service
from experimance_common.health import HealthStatus, HealthReporter, create_health_reporter

# Configure logging with adaptive file location and console handling
logger = setup_logging(__name__)  # Auto-detects: console in dev, file-only in production


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
        
        # Initialize health reporting (replaces heartbeat system)
        self._health_reporter = create_health_reporter(self.service_name, self.service_type)
        
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
        self._stop_requested = False # Flag to prevent multiple stop requests
        
        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
            signal.signal(sig, self._signal_handler)
            
        # Record initial health check
        self._health_reporter.record_health_check(
            "service_initialization",
            HealthStatus.HEALTHY,
            "Service initialized successfully"
        )
        
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
    
    @property
    def status(self) -> HealthStatus:
        """Get the overall health status of the service."""
        return self.get_overall_health_status()

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
    
    @property
    def running(self) -> bool:
        """Check if the service is currently running."""
        return self.state == ServiceState.RUNNING

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
                               If a coroutine is provided and the service is running,
                               it will be automatically converted to a task and scheduled.
                               Otherwise coroutines are stored and converted when run() is called.
        """
        # Make sure we don't add None or invalid objects
        if task_or_coroutine is None:
            logger.warning(f"{self.service_name}: Attempted to add None as a task - ignoring")
            return
            
        # Make sure we don't accidentally add the same task/coroutine twice
        if task_or_coroutine in self.tasks:
            logger.warning(f"{self.service_name}: Task {task_or_coroutine} already registered - ignoring duplicate")
            return
        
        # If it's a coroutine and we're already running, create and schedule the task immediately
        if asyncio.iscoroutine(task_or_coroutine) and self.running:
            task = asyncio.create_task(task_or_coroutine)
            self.tasks.append(task)
            logger.debug(f"{self.service_name}: Created and scheduled task immediately")
        else:
            # Otherwise, just store it (will be converted to task in run() if needed)
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

    async def start(self):
        """Start the service.
        
        This method should be extended by subclasses to initialize 
        their specific components before calling super().start().
        """
        logger.debug(f"Starting {self.service_type} service: {self.service_name}")
        self.start_time = time.monotonic()
        
        # Add health monitoring task (replaces heartbeat system)
        self.add_task(self._health_monitoring_loop())
        
        # Record service start in health system
        self._health_reporter.record_health_check(
            "service_start",
            HealthStatus.HEALTHY,
            "Service started successfully"
        )
        
    async def _health_monitoring_loop(self):
        """Health monitoring loop that replaces the heartbeat system."""
        while self.state == ServiceState.RUNNING:
            try:
                # Update service statistics in health reporter
                self._health_reporter.update_service_stats(
                    messages_sent=self.messages_sent,
                    messages_received=self.messages_received,
                    errors=self.errors
                )
                
                # Record periodic health check
                self._health_reporter.record_health_check(
                    "periodic_health_check",
                    HealthStatus.HEALTHY,
                    f"Service {self.service_name} is responsive",
                    metadata={
                        "uptime": time.monotonic() - self.start_time,
                        "state": self.state.value,
                        "tasks_count": len(self.tasks)
                    }
                )
                
                # Sleep for health check interval
                if not await self._sleep_if_running(30):  # 30 second interval
                    break
                    
            except Exception as e:
                logger.error(f"Error in health monitoring loop for {self.service_name}: {e}")
                self._health_reporter.record_error(e, is_fatal=False)
                if not await self._sleep_if_running(30):
                    break
    
    
    async def stop(self):
        """Stop the service and clean up resources.
        
        This method ensures all tasks are properly cancelled
        and resources are cleaned up.
        """
        async with self._stop_lock:
            if self.state == ServiceState.STOPPED:
                logger.debug(f"Service {self.service_name} is already STOPPED. Ignoring stop call.")
                return

            # Record service stop in health system
            self._health_reporter.record_health_check(
                "service_stop",
                HealthStatus.HEALTHY,
                "Service stopping gracefully"
            )

            logger.debug(f"Stopping {self.service_name} (lock acquired)...")
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
                    logger.debug(f"Cancelling main run task for {self.service_name}.")
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
            
            # Record final health check before stopping - flush immediately for shutdown
            self._health_reporter.record_health_check(
                "service_stopped",
                HealthStatus.WARNING,  # Use WARNING to indicate service is no longer active
                "Service stopped gracefully",
                flush=True  # Flush immediately on shutdown
            )
            
            self.state = ServiceState.STOPPED
            logger.debug(f"Service {self.service_name} stopped")
    
    def _request_stop(self, suffix: str):
        """Internal method to request a graceful shutdown with a specific task name suffix.
        
        This method schedules a stop operation without blocking the caller and creates
        a task with a descriptive name for debugging purposes.
        
        Args:
            suffix: Suffix to append to create the task name: "{service_name}-{suffix}-stop"
        """
        if self.state in [ServiceState.STOPPING, ServiceState.STOPPED]:
            logger.debug(f"Stop already requested/completed for {self.service_name}")
            return
    
        # Check if stop has already been requested to prevent multiple stop tasks
        if self._stop_requested:
            logger.debug(f"Stop already requested for {self.service_name} ({suffix})")
            return
            
        # Set the flag to prevent additional stop requests
        self._stop_requested = True
        logger.info(f"Shutdown requested for {self.service_name} ({suffix})")
        # Schedule the stop operation to run soon
        asyncio.create_task(self.stop(), name=f"{self.service_name}-{suffix}-stop")
    
    def request_stop(self):
        """Request a graceful shutdown of the service.
        
        This method schedules a stop operation without blocking the caller.
        It's useful when you want to initiate shutdown from within a service task
        or callback without blocking the current operation.
        
        Returns immediately after scheduling the stop operation.
        """
        self._request_stop("requested")
    
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
                        task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
                        logger.debug(f"Cancelling uncompleted task in {self.service_name}: {task_name}")
                        task.cancel()
                    elif inspect.iscoroutine(task):
                        # For coroutines, we need to be careful as they might have already been converted to tasks and awaited
                        # Just close them to prevent resource leaks - don't try to await them
                        try:
                            # Safely get coroutine name if available
                            coroutine_name = task.__name__ if hasattr(task, '__name__') else str(task)
                            logger.debug(f"Closing coroutine in {self.service_name}: {coroutine_name}")
                            task.close()
                        except (RuntimeError, AttributeError) as e:
                            # This can happen if the coroutine was already awaited or closed
                            logger.debug(f"Could not close coroutine {task} - it may have already been awaited: {e}")
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
        
        logger.debug(f"Service {self.service_name} running")
        
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
            new_task_list = []  # We'll replace self.tasks with this to remove raw coroutines
            
            for task in self.tasks:
                if isinstance(task, asyncio.Task):
                    # If it's already a Task, just use it directly
                    task_objects.append(task)
                    new_task_list.append(task)  # Keep tasks in the list
                elif inspect.iscoroutine(task):
                    # Convert coroutines to tasks here when run() is called
                    # Check for a stored task name from _register_task
                    task_name = getattr(task, "_task_name", None)
                    if not task_name:
                        # Safely get coroutine name or use a generic name
                        if hasattr(task, '__name__'):
                            coro_name = task.__name__
                        elif hasattr(task, '__qualname__'):
                            coro_name = task.__qualname__
                        else:
                            coro_name = 'task'
                        task_name = f"{self.service_name}-{coro_name}"
                    
                    logger.debug(f"Converting coroutine to task with name: {task_name}")
                    try:
                        task_obj = asyncio.create_task(task, name=task_name)
                        task_objects.append(task_obj)
                        new_task_list.append(task_obj)  # Replace coroutine with task in list
                    except RuntimeError as e:
                        logger.warning(f"Could not convert coroutine to task: {e}. This might be an already used coroutine.")
                else:
                    logger.warning(f"Unsupported task type found in tasks list: {type(task)}: {task}")
            
            # Replace the tasks list with one that contains only Task objects
            self.tasks = new_task_list
                    
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
                    self.record_error(
                        Exception(f"Service {self.service_name} encountered task errors during run() execution."),
                        is_fatal=False
                    )
            except asyncio.CancelledError:
                # If the gather itself is cancelled, we still want to check for task exceptions
                logger.debug(f"Gather was cancelled for {self.service_name}, checking individual tasks for errors")
                for task in task_objects:
                    if task.done() and not task.cancelled():
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(f"Task error in {task.get_name()} after cancellation: {exc}")
                                self.record_error(exc)
                        except (asyncio.CancelledError, asyncio.InvalidStateError):
                            # Task was cancelled or not done, just skip
                            pass
                raise  # Re-raise the CancelledError
        # This CancelledError is for the run() task itself being cancelled.
        except asyncio.CancelledError:
            logger.debug(f"Service {self.service_name} run task itself was cancelled, initiating stop.")
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
                logger.debug(f"Service {self.service_name} run() method ending (state: {self.state}). "
                            f"Stop not yet initiated by other means. Calling self.stop().")
                try:
                    await self.stop()
                except Exception as e:
                    logger.error(f"Error during self.stop() called from run() finally for {self.service_name}: {e}", exc_info=True)
                    self.record_error(e)
            elif self.state == ServiceState.STOPPING:
                logger.debug(f"Service {self.service_name} run() method's finally block: A stop operation is already in progress "
                            f"(state: {self.state}). Letting it complete.")
            elif self.state == ServiceState.STOPPED:
                logger.debug(f"Service {self.service_name} run() method's finally block: Service already STOPPED.")
            
            logger.debug(f"Service {self.service_name} run() method completed. Final state: {self.state}")

    async def _handle_signal_async(self, sig):
        """Handle signals in the asyncio event loop by calling stop() or handling SIGTSTP."""
        signal_name = signal.Signals(sig).name
        logger.debug(f"Received signal {signal_name} in asyncio event loop for {self.service_name}")

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

        logger.debug(f"Service {self.service_name} initiating shutdown via stop() due to signal {signal_name}")
        try:
            # Call stop() directly. stop() is responsible for setting the state
            await self.stop()
        except Exception as e:
            logger.error(f"Error during signal-initiated stop for {self.service_name}: {e}", exc_info=True)
            # Ensure the service is in a non-operational state even if stop fails.
            self.record_error(e, is_fatal=False)
            self.state = ServiceState.STOPPED

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
                        self._request_stop("task-error")
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            # Task was cancelled or not done, just skip
            pass
            
    def record_error(self, error: Exception, is_fatal: bool = False, custom_message: Optional[str] = None):
        """Record an error and update service status.
        
        Args:
            error: The exception that occurred
            is_fatal: Whether this error should mark the service as fatally errored
            custom_message: Optional custom message to log instead of default format
        """
        self.errors += 1
        
        # Use custom message if provided, otherwise use default format
        if custom_message:
            log_message = custom_message
        else:
            log_message = f"{'Fatal error' if is_fatal else 'Error'} in service {self.service_name}: {error!r}"
        
        # Record in unified health system
        self._health_reporter.record_error(error, is_fatal)
        
        # Log the error
        if is_fatal:
            logger.error(log_message, exc_info=True)
            
            # Automatically initiate shutdown for fatal errors
            self._request_stop("fatal-error")
        else:
            logger.error(log_message, exc_info=True)
            
    def reset_error_status(self):
        """Reset the service's error status to HEALTHY.
        
        Call this after recovering from an error condition.
        """
        # Record recovery in health system
        self._health_reporter.record_health_check(
            "error_recovery",
            HealthStatus.HEALTHY,
            "Service recovered from error condition"
        )
        logger.info(f"Service {self.service_name} error status reset to HEALTHY")
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status information.
        
        Returns:
            Dictionary containing health status, checks, and service metrics
        """
        return self._health_reporter.get_health_summary()
    
    def get_overall_health_status(self) -> HealthStatus:
        """Get the overall health status as a HealthStatus enum.
        
        This is a convenience method for quick health checks, especially useful
        in tests and simple health monitoring scenarios.
        
        Returns:
            Current overall health status as HealthStatus enum
        """
        health_data = self.get_health_status()
        return HealthStatus(health_data["overall_status"])
        
    def record_health_check(self, name: str, status: HealthStatus, 
                          message: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Record a custom health check.
        
        Args:
            name: Name of the health check
            status: Health status result
            message: Optional message describing the check
            metadata: Optional additional data
        """
        self._health_reporter.record_health_check(name, status, message, metadata)
        
    def get_task_names(self) -> List[str]:
        """Get a list of task names from the current tasks.
        
        This method extracts names from both coroutines and Task objects in the tasks list.
        Task objects will have their name extracted via get_name(), and coroutines via __name__.
        
        Returns:
            List of task name strings
        """
        names = []
        for task in self.tasks:
            if hasattr(task, 'get_name'):
                # It's a Task object
                name = task.get_name()
                # If the task was named with the service name as prefix (service_name-function_name pattern),
                # strip it for cleaner output
                if '-' in name and name.startswith(self.service_name):
                    names.append(name.split('-', 1)[1])
                else:
                    names.append(name)
            elif hasattr(task, '__name__'):
                # It's a coroutine function or method
                names.append(task.__name__)
            else:
                # Can't get a name, use string representation
                names.append(str(task))
        return names

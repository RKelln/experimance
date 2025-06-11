#!/usr/bin/env python3
"""
Tests for the BaseService class in experimance_common.service module.

This test suite validates:
1. Service lifecycle (init, start, run, stop)
2. Signal handling
3. Task management
4. Graceful shutdown
5. Error handling

Run with:
    uv run -m pytest utils/tests/test_base_service.py -v
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from unittest.mock import AsyncMock, MagicMock, patch

from experimance_common.constants import TICK
from experimance_common.test_utils import wait_for_service_state
import pytest

from experimance_common.base_service import BaseService, ServiceState, ServiceStatus
from experimance_common.test_utils import wait_for_service_shutdown


# Configure test logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def run_service_concurrently(service: BaseService):
    """
    An async context manager to run a service's run() method in a background task
    and ensure proper cleanup.
    The service will be started (state -> STARTING) if it's in the INITIALIZED state.
    The service's run() method is expected to set the state to RUNNING.
    """
    if service.state == ServiceState.INITIALIZED:
        logger.info(f"Service {service.service_name} is INITIALIZED. Starting it (state -> STARTING) before running concurrently.")
        await service.start() # This sets state to STARTING, then STARTED
    
    # Ensure service is STARTED before attempting to run
    if service.state not in [ServiceState.STARTED, ServiceState.RUNNING]:
        raise RuntimeError(
            f"Service {service.service_name} must be in STARTED or RUNNING state to use run_service_concurrently. "
            f"Current state: {service.state}"
        )

    run_task = asyncio.create_task(service.run(), name=f"{service.service_name}-run-ctx-mgr")
    try:
        yield run_task
    finally:
        logger.debug(f"Context manager for {service.service_name} cleaning up...")
        current_service_state = service.state # Capture state before potential stop
        
        # First, cancel the run task directly to break any potential deadlocks
        if not run_task.done():
            logger.debug(f"Context manager cancelling run_task for {service.service_name}")
            run_task.cancel()
        
        # Now attempt to stop the service if it's not already stopping or stopped
        # This ensures proper cleanup even if service.run() might have exited due to an error
        if current_service_state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
            logger.debug(f"Context manager stopping service {service.service_name} (current state: {current_service_state})")
            await service.stop() # stop() should clean up any remaining resources
            
        # Wait for the run task with a short timeout to avoid hanging
        try:
            # Use a short timeout to prevent indefinite hanging
            await asyncio.wait_for(asyncio.shield(run_task), timeout=0.5)
        except asyncio.TimeoutError:
            logger.debug(f"Timed out waiting for run_task to complete for {service.service_name}")
        except asyncio.CancelledError:
            logger.debug(f"run_task for {service.service_name} was cancelled.")
        except Exception as e:
            logger.error(f"run_task for {service.service_name} (service.run()) raised an exception: {e!r}")
            # Don't re-raise exceptions from the run task, the test should continue
        
        final_state = service.state
        logger.debug(f"Context manager for {service.service_name} finished cleanup. Service state: {final_state}")
        assert final_state == ServiceState.STOPPED, \
            f"Service {service.service_name} should be STOPPED after context manager cleanup, but is {final_state}"


class TestBaseService:
    """Tests for the BaseService class."""
    
    @pytest.fixture
    async def base_service(self):
        """Create a BaseService instance for testing."""
        service = BaseService(service_name="test-service", service_type="test")
        yield service
        # Clean up after test
        if service.state != ServiceState.STOPPED:
            logger.debug(f"base_service fixture ensuring service {service.service_name} is stopped. Current state: {service.state}")
            await service.stop()
    
    @pytest.fixture
    async def started_service(self, base_service: BaseService):
        """
        A BaseService instance on which start() has been called.
        The service will be STARTING during start() and in the 
        STARTED state after start().
        """
        await base_service.start() # This sets state to STARTED
        assert base_service.state == ServiceState.STARTED
        yield base_service
        # Cleanup (stop) is implicitly handled by the base_service fixture's yield

    @pytest.mark.asyncio
    async def test_initialization(self, base_service):
        """Test that BaseService initializes with correct default values."""
        assert base_service.service_name == "test-service"
        assert base_service.service_type == "test"
        assert base_service.state == ServiceState.INITIALIZED
        assert len(base_service.tasks) == 0
        
        # Statistics should be initialized
        assert base_service.messages_sent == 0
        assert base_service.messages_received == 0
        assert base_service.errors == 0
    
    @pytest.mark.asyncio
    async def test_start(self, base_service: BaseService):
        """Test the start method sets the correct state and registers stats task."""
        assert base_service.state == ServiceState.INITIALIZED # Pre-condition

        await base_service.start() # Action being tested
        
        # Post-conditions
        assert base_service.state == ServiceState.STARTED
        # Check that the display_stats coroutine is registered
        assert any(
            coro.__name__ == "display_stats"  # Check coroutine name
            for coro in base_service.tasks
        ), "display_stats coroutine was not registered by start()"
        assert len(base_service.tasks) >= 1 # Should have at least display_stats

    @pytest.mark.asyncio
    async def test_stop(self, started_service: BaseService):
        """Test the stop method cleans up properly from STARTING state."""
        # started_service is in STARTED state from the fixture
        assert started_service.state == ServiceState.STARTED

        await started_service.stop() # Call stop on the service
        
        assert started_service.state == ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_duplicate_stop_calls(self, started_service: BaseService):
        """Test that multiple stop() calls don't cause issues when started from STARTING state."""
        # started_service is in STARTING state
        assert started_service.state == ServiceState.STARTED
        
        await started_service.stop()
        assert started_service.state == ServiceState.STOPPED
        
        # Second call should be handled gracefully
        await started_service.stop()
        assert started_service.state == ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_task_registration_and_execution(self):
        """Test registering and running tasks."""
        service = BaseService(service_name="task-test-service")
        
        task_mock = MagicMock()
        task_completed_event = asyncio.Event()
        
        async def custom_test_task():
            task_mock()
            task_completed_event.set()
            task_mock.complete()

        service.tasks = [] 
        service.add_task(custom_test_task())
        
        try:
            # run_service_concurrently will call service.start() (state -> STARTING),
            # then service.run() (state -> RUNNING), which executes the tasks.
            async with run_service_concurrently(service):
                await asyncio.wait_for(task_completed_event.wait(), timeout=1.0)
            
            task_mock.assert_called_once()
            task_mock.complete.assert_called_once()
            assert service.state == ServiceState.STOPPED
        except asyncio.TimeoutError:
            pytest.fail("custom_test_task didn't complete within timeout")
    
    @pytest.mark.asyncio
    async def test_error_during_task_execution(self):
        """Test that an error in a registered task is handled properly."""
        service = BaseService(service_name="error-task-service")

        # Set up a flag to track if the task raised an exception 
        task_raised_exception = False
        
        async def error_task():
            nonlocal task_raised_exception
            logger.info("Error task starting, will raise an exception.")
            await asyncio.sleep(TICK)  # Give a moment for the task to start
            task_raised_exception = True
            raise ValueError("Simulated task error")

        await service.start()
        service.add_task(error_task())
        
        # Create the run task but don't await it directly - we'll check service state instead
        run_task = asyncio.create_task(service.run(), name=f"{service.service_name}-run-error-test")
        
        try:
            # Wait for the service to transition to STOPPED state and ERROR status
            # This approach avoids the deadlock from waiting for the run_task directly
            # when service.stop() is called from within run()
            from utils.tests.test_utils import wait_for_service_state_and_status
            await wait_for_service_state_and_status(
                service, 
                target_state=ServiceState.STOPPED,
                target_status=ServiceStatus.ERROR, 
                timeout=3.0
            )
            
            # Verify the task actually raised the exception
            assert task_raised_exception, "The error_task did not execute or didn't raise an exception"
            
            # Verify the service has ERROR status
            assert service.status == ServiceStatus.ERROR, f"Service should have ERROR status after task error, but has {service.status}"
            
            # Check errors
            assert service.errors >= 1, "Service should have recorded at least one error"

            logger.info("test_error_during_task_execution completed successfully!")
        finally:
            # Ensure the run_task is cleaned up if it's still running
            if 'run_task' in locals() and not run_task.done():
                logger.info(f"Cancelling run_task for {service.service_name}")
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task

    @pytest.mark.asyncio
    async def test_error_during_task_execution_with_diagnostics(self):
        """Test task error handling, with enhanced diagnostics."""
        service = BaseService(service_name="error-diagnostic-service")
        
        # Set up a flag to track if the task raised an exception
        task_raised_exception = False
        
        async def error_task():
            nonlocal task_raised_exception
            logger.info("Error task starting, will raise an exception.")
            await asyncio.sleep(TICK)  # Give a moment for the task to start
            task_raised_exception = True
            raise ValueError("Simulated task error for diagnostics")

        await service.start()
        service.add_task(error_task())
        
        # Create the run task
        run_task = asyncio.create_task(service.run(), name=f"{service.service_name}-run-error-test")
        
        # Track state and status changes
        state_history = []
        status_history = []
        
        async def monitor_service_state():
            last_state = None
            last_status = None
            
            while True:
                # Check for state changes
                if service.state != last_state:
                    state_history.append((time.monotonic(), service.state))
                    logger.info(f"Service state changed: {last_state} -> {service.state}")
                    last_state = service.state
                    
                # Check for status changes
                if service.status != last_status:
                    status_history.append((time.monotonic(), service.status))
                    logger.info(f"Service status changed: {last_status} -> {service.status}")
                    last_status = service.status
                    
                await asyncio.sleep(TICK)  # Short interval to catch all changes
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor_service_state())
        
        try:
            # Wait for the service to transition to error status
            from utils.tests.test_utils import wait_for_service_status
            await wait_for_service_status(service, ServiceStatus.ERROR, timeout=2.0)
            
            # Wait a bit to collect state/status changes
            await asyncio.sleep(1.0)
            
            # Print diagnostics
            logger.info(f"Diagnostics for service {service.service_name}:")
            logger.info(f"  Current state: {service.state}")
            logger.info(f"  Current status: {service.status}")
            logger.info(f"  Exception raised: {task_raised_exception}")
            logger.info(f"  Error count: {service.errors}")
            logger.info(f"  Run task done: {run_task.done()}")
            
            logger.info("State history:")
            for timestamp, state in state_history:
                logger.info(f"  {timestamp:.3f}: {state}")
            
            logger.info("Status history:")
            for timestamp, status in status_history:
                logger.info(f"  {timestamp:.3f}: {status}")
            
            # Verify expectations
            assert task_raised_exception, "The error_task did not execute or didn't raise an exception"
            assert service.errors >= 1, "Service should have recorded at least one error"
            assert service.status == ServiceStatus.ERROR, f"Service should have ERROR status, has {service.status}"
            assert service.state == ServiceState.STOPPED, f"Service should be STOPPED, is {service.state}"
            
            logger.info("Diagnostic test completed successfully!")
        finally:
            # Clean up
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task
                
            if not run_task.done():
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        service = BaseService(service_name="stats-test")
        
        async def update_stats_task():
            for i in range(5):
                if service.state != ServiceState.RUNNING: # Check service state
                    logger.debug(f"update_stats_task: service not running (state: {service.state}), breaking loop at iteration {i}")
                    break
                service.messages_sent += 1
                service.messages_received += 2
                logger.debug(f"update_stats_task: iter {i}, sent={service.messages_sent}, recv={service.messages_received}")
                await asyncio.sleep(0.05) 
            logger.debug("update_stats_task task finished")

        service.add_task(update_stats_task())
        
        stop_event = asyncio.Event()
        async def trigger_stop_task():
            # Wait for update_stats_task to likely complete its iterations
            await asyncio.sleep(0.3) # update_stats_task runs for ~0.25s
            logger.debug("trigger_stop_task: setting stop_event")
            stop_event.set()

        service.add_task(trigger_stop_task())

        # run_service_concurrently calls start() (-> STARTING), then run() (-> RUNNING)
        async with run_service_concurrently(service):
            try:
                logger.debug("test_statistics_tracking: waiting for stop_event")
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                logger.debug("test_statistics_tracking: stop_event received. Exiting 'with' block, context manager will stop service.")
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for stop_event in test_statistics_tracking")
                pytest.fail("stop_event was not set in time")
        
        assert service.messages_sent == 5, f"Expected 5 messages sent, got {service.messages_sent}"
        assert service.messages_received == 10, f"Expected 10 messages received, got {service.messages_received}"
        assert service.state == ServiceState.STOPPED, f"Service should be STOPPED, but is {service.state}"
    
    @pytest.mark.asyncio
    async def test_cancellation_during_stop(self):
        """Test that cancellation during stop is handled correctly."""
        logger.info("Starting test_cancellation_during_stop")
        service = BaseService(service_name="cancel-test")
        
        task_started_event = asyncio.Event()
        task_cancelled_event = asyncio.Event()

        async def long_running_cancellable_task():
            logger.info("long_running_cancellable_task starting")
            task_started_event.set()
            try:
                while True:
                    if service.state != ServiceState.RUNNING:
                        logger.debug(f"long_running_cancellable_task detected service.state={service.state}, breaking")
                        break
                    await asyncio.sleep(TICK) 
            except asyncio.CancelledError:
                logger.info("long_running_cancellable_task was cancelled")
                task_cancelled_event.set()
                raise 
            logger.debug("long_running_cancellable_task exiting normally (should not happen)")

        service.add_task(long_running_cancellable_task())

        logger.info("Using run_service_concurrently context manager")
        async with run_service_concurrently(service): 
            try:
                logger.info("Waiting for task_started_event")
                await asyncio.wait_for(task_started_event.wait(), timeout=1.0)
                logger.info("task_started_event set, long_running_cancellable_task has started")
                # Context manager will call service.stop() upon exiting this block.
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for task_started_event")
                pytest.fail("long_running_cancellable_task did not start within timeout")
            
        logger.info("Context manager exited, checking state and event")
        assert service.state == ServiceState.STOPPED, f"Expected state STOPPED, got {service.state}"
        
        # Add a small timeout to wait for the cancellation event if needed
        if not task_cancelled_event.is_set():
            logger.info("task_cancelled_event not set yet, waiting briefly...")
            try:
                await asyncio.wait_for(task_cancelled_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for task_cancelled_event")
                
        assert task_cancelled_event.is_set(), \
            "long_running_cancellable_task should have been cancelled and set its event"
        logger.info("test_cancellation_during_stop completed successfully")

    @pytest.mark.asyncio
    async def test_run_with_no_tasks(self):
        """Test that run raises an error when no tasks are registered if start was not called."""
        service = BaseService(service_name="no-tasks-service")
        await service.start()

        # remove any default tatsks that might have been registered by start()
        await service._clear_tasks()  # Clear tasks using the provided method

        with pytest.raises(RuntimeError, match="No tasks registered"):
            await service.run()

    @pytest.mark.asyncio
    async def test_run_after_start_with_only_default_tasks(self, base_service: BaseService):
        """Test that run executes with default tasks if only start() was called."""
        # base_service is INITIALIZED
        await base_service.start() # Adds display_stats
        assert base_service.state == ServiceState.STARTED
        # Check that the display_stats coroutine is registered
        assert any(
            coro.__name__ == "display_stats" # Check coroutine name
            for coro in base_service.tasks
        ), "display_stats coroutine was not registered by start()"

        # run_service_concurrently will take the STARTED service and run it.
        # It should run the display_stats task and then stop gracefully.
        async with run_service_concurrently(base_service):
            # Let it run for a very short period to ensure run() starts and display_stats runs once
            await asyncio.sleep(0.1) 
            # Context manager will stop it.
        
        assert base_service.state == ServiceState.STOPPED
        # Further assertions could be made if display_stats had observable side effects,
        # e.g., checking logs if it logs something specific. For now, just ensuring it runs and stops.


class TestServiceErrorHandling:
    """Tests for service error handling and automatic shutdown functionality."""
    
    @pytest.fixture
    async def error_service(self):
        """Create a service that can simulate different types of errors."""
        
        class ErrorTestService(BaseService):
            def __init__(self):
                super().__init__("error-test-service", "test")
                self.task_error_count = 0
                self.fatal_error_triggered = False
                
            async def error_prone_task(self):
                """A task that can raise different types of errors."""
                while self.running:
                    await asyncio.sleep(0.1)
                    if self.task_error_count > 0:
                        if self.task_error_count == 1:
                            self.task_error_count += 1
                            raise ValueError("Non-fatal test error")
                        elif self.task_error_count == 2:
                            self.task_error_count += 1
                            raise RuntimeError("Fatal test error")
                    
            async def start(self):
                self.add_task(self.error_prone_task())
                await super().start()
        
        service = ErrorTestService()
        yield service
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_record_error_non_fatal(self, error_service):
        """Test recording non-fatal errors."""
        await error_service.start()
        
        # Record a non-fatal error
        test_error = ValueError("Test non-fatal error")
        error_service.record_error(test_error, is_fatal=False)
        
        # Service should continue running
        assert error_service.status == ServiceStatus.ERROR
        assert error_service.errors == 1
        assert error_service.state == ServiceState.STARTED  # Still operational
    
    @pytest.mark.asyncio
    async def test_record_error_fatal_auto_shutdown(self, error_service):
        """Test that fatal errors automatically trigger shutdown."""
        await error_service.start()
        
        # Start the service running
        run_task = asyncio.create_task(error_service.run())
        
        try:
            # Wait for service to be running
            await error_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Record a fatal error
            test_error = RuntimeError("Test fatal error")
            error_service.record_error(test_error, is_fatal=True)
            
            # Service should automatically stop
            assert error_service.status == ServiceStatus.FATAL
            assert error_service.errors == 1
            
            # Wait for automatic shutdown
            await error_service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
            assert error_service.state == ServiceState.STOPPED
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_reset_error_status(self, error_service):
        """Test resetting error status after recovery."""
        # Record an error
        test_error = ValueError("Recoverable error")
        error_service.record_error(test_error, is_fatal=False)
        
        assert error_service.status == ServiceStatus.ERROR
        assert error_service.errors == 1
        
        # Reset error status
        error_service.reset_error_status()
        
        assert error_service.status == ServiceStatus.HEALTHY
        assert error_service.errors == 1  # Error count remains
    
    @pytest.mark.asyncio
    async def test_task_error_auto_shutdown(self, error_service):
        """Test that task errors automatically trigger shutdown."""
        await error_service.start()
        
        # Start the service running
        run_task = asyncio.create_task(error_service.run())
        
        try:
            # Wait for service to be running
            await error_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Trigger an error in the task
            error_service.task_error_count = 1
            
            # Wait for the service to detect the error and stop
            await error_service.wait_for_state(ServiceState.STOPPED, timeout=3.0)
            assert error_service.state == ServiceState.STOPPED
            assert error_service.errors >= 1
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass


class TestServiceShutdownMethods:
    """Tests for different service shutdown methods."""
    
    @pytest.fixture
    async def shutdown_service(self):
        """Create a service for testing shutdown methods."""
        
        class ShutdownTestService(BaseService):
            def __init__(self):
                super().__init__("shutdown-test-service", "test")
                self.stop_reason = None
                self.custom_task_running = False
                
            async def custom_background_task(self):
                """A background task for testing shutdown."""
                self.custom_task_running = True
                try:
                    while self.running:
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    self.custom_task_running = False
                    raise
                finally:
                    self.custom_task_running = False
                    
            async def start(self):
                self.add_task(self.custom_background_task())
                await super().start()
        
        service = ShutdownTestService()
        yield service
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_direct_stop_method(self, shutdown_service):
        """Test direct call to stop() method."""
        await shutdown_service.start()
        
        # Start the service running
        run_task = asyncio.create_task(shutdown_service.run())
        
        try:
            # Wait for service to be running
            await shutdown_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            assert shutdown_service.custom_task_running
            
            # Call stop() directly
            await shutdown_service.stop()
            
            # Service should be stopped
            assert shutdown_service.state == ServiceState.STOPPED
            assert not shutdown_service.custom_task_running
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_request_stop_method(self, shutdown_service):
        """Test request_stop() method for non-blocking shutdown."""
        await shutdown_service.start()
        
        # Start the service running
        run_task = asyncio.create_task(shutdown_service.run())
        
        try:
            # Wait for service to be running
            await shutdown_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            assert shutdown_service.custom_task_running
            
            # Call request_stop() - should be non-blocking
            shutdown_service.request_stop()
            
            # Wait for the service to stop automatically
            await shutdown_service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
            assert shutdown_service.state == ServiceState.STOPPED
            assert not shutdown_service.custom_task_running
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_request_stop_idempotent(self, shutdown_service):
        """Test that request_stop() is idempotent."""
        await shutdown_service.start()
        
        # Start the service running
        run_task = asyncio.create_task(shutdown_service.run())
        
        try:
            # Wait for service to be running
            await shutdown_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Call request_stop() multiple times
            shutdown_service.request_stop()
            shutdown_service.request_stop()
            shutdown_service.request_stop()
            
            # Should still stop cleanly
            await shutdown_service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
            assert shutdown_service.state == ServiceState.STOPPED
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_stop_from_task(self, shutdown_service):
        """Test calling stop() from within a service task."""
        
        class SelfStoppingService(BaseService):
            def __init__(self):
                super().__init__("self-stopping-service", "test")
                self.stop_after_iterations = 3
                self.iterations = 0
                
            async def self_stopping_task(self):
                """A task that stops the service after a few iterations."""
                while self.running:
                    self.iterations += 1
                    if self.iterations >= self.stop_after_iterations:
                        # Stop the service from within the task
                        self.request_stop()  # Now non-async, no await needed
                        break
                    await asyncio.sleep(0.1)
                    
            async def start(self):
                self.add_task(self.self_stopping_task())
                await super().start()
        
        service = SelfStoppingService()
        
        try:
            await service.start()
            
            # Start the service running in a background task
            run_task = asyncio.create_task(service.run())
            
            # Wait for service to be running
            await service.wait_for_state(ServiceState.RUNNING, timeout=2.0)
            
            # Wait for the service to stop itself
            await service.wait_for_state(ServiceState.STOPPED, timeout=5.0)
            
            # Verify that the task ran long enough
            assert service.iterations >= service.stop_after_iterations
            
            # Wait for run task to complete
            await run_task
            
        finally:
            # Clean up
            if not run_task.done():
                run_task.cancel()
            
            # Only call stop() if the service isn't already stopping or stopped
            if service.state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
                await service.stop()


class TestServiceShutdownTaskNaming:
    """Tests for service shutdown task naming and debugging."""
    
    @pytest.mark.asyncio
    async def test_shutdown_task_names(self):
        """Test that shutdown tasks have descriptive names."""
        service = BaseService("task-name-test-service", "test")
        await service.start()
        
        # Mock asyncio.create_task to capture task names
        original_create_task = asyncio.create_task
        created_tasks = []
        
        def mock_create_task(coro, **kwargs):
            task = original_create_task(coro, **kwargs)
            created_tasks.append((task, kwargs.get('name', 'no-name')))
            return task
        
        try:
            with patch('asyncio.create_task', side_effect=mock_create_task):
                # Test different shutdown methods
                service.request_stop()
                
                # Record a fatal error
                service.record_error(RuntimeError("test fatal"), is_fatal=True)
                
                # Brief delay to let tasks be created
                await asyncio.sleep(0.1)
            
            # Check that tasks were created with expected names
            task_names = [name for _, name in created_tasks]
            
            expected_patterns = [
                "task-name-test-service-requested-stop",
                "task-name-test-service-fatal-error-stop"
            ]
            
            for pattern in expected_patterns:
                assert any(pattern in name for name in task_names), \
                    f"Expected task name pattern '{pattern}' not found in {task_names}"
            
        finally:
            # Restore original function
            if service.state != ServiceState.STOPPED:
                await service.stop()

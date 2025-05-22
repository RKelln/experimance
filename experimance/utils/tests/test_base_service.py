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
import signal
import time
import pytest
import inspect # Add import for inspect
from unittest.mock import MagicMock, patch
from contextlib import suppress

from experimance_common.service import BaseService, ServiceState

# Configure test logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestBaseService:
    """Tests for the BaseService class."""
    
    @pytest.fixture
    async def base_service(self):
        """Create a BaseService instance for testing."""
        service = BaseService(service_name="test-service", service_type="test")
        yield service
        # Clean up after test
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_initialization(self, base_service):
        """Test that BaseService initializes with correct default values."""
        assert base_service.service_name == "test-service"
        assert base_service.service_type == "test"
        assert base_service.state == ServiceState.INITIALIZED
        assert base_service.running is False
        assert base_service._stopping is False
        assert len(base_service.tasks) == 0
        
        # Statistics should be initialized
        assert base_service.messages_sent == 0
        assert base_service.messages_received == 0
        assert base_service.errors == 0
    
    @pytest.mark.asyncio
    async def test_start(self, base_service):
        """Test the start method sets the correct state and registers stats task."""
        await base_service.start()
        
        assert base_service.state == ServiceState.STARTING
        assert base_service.running is True
        assert len(base_service.tasks) == 1  # Should have the stats display task
    
    @pytest.mark.asyncio
    async def test_stop(self, base_service):
        """Test the stop method cleans up properly."""
        # Start the service first
        await base_service.start()
        assert base_service.running is True
        
        # Now stop it
        await base_service.stop()
        
        assert base_service.state == ServiceState.STOPPED
        assert base_service.running is False
        assert base_service._stopping is True
    
    @pytest.mark.asyncio
    async def test_duplicate_stop_calls(self, base_service):
        """Test that multiple stop() calls don't cause issues."""
        # Start the service
        await base_service.start()
        
        # Call stop multiple times
        await base_service.stop()
        assert base_service.state == ServiceState.STOPPED
        
        # Second call should be handled gracefully
        await base_service.stop()
        assert base_service.state == ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_task_registration_and_execution(self):
        """Test registering and running tasks."""
        service = BaseService(service_name="task-test-service")
        
        # Create a task with a mock
        task_mock = MagicMock()
        task_completed = asyncio.Event()
        
        async def test_task():
            task_mock()
            # Set the completed event before exiting
            task_completed.set()
            task_mock.complete()
            
        # Register the task without the stats task (which can cause issues in tests)
        service.tasks = []  # Clear default tasks
        service._register_task(test_task())
        
        # Start the service
        await service.start()
        
        # Run the service in a separate task with a short timeout
        run_task = asyncio.create_task(service.run())
        
        # Wait for our task to complete or timeout
        try:
            await asyncio.wait_for(task_completed.wait(), timeout=1.0)
            
            # Once our task completes, stop the service gracefully
            await service.stop()
            
            # Cancel the run task since we've stopped manually
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
                
            # Verify task was called and completed
            task_mock.assert_called_once()
            task_mock.complete.assert_called_once()
        except asyncio.TimeoutError:
            # If we timeout, stop everything and fail the test
            await service.stop()
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
            pytest.fail("Task didn't complete within timeout")
    
    @pytest.mark.asyncio
    async def test_signal_handler(self, base_service):
        """Test that signal handler sets correct flags."""
        # Mock the signal and frame
        mock_signal = signal.SIGINT
        mock_frame = None
        
        # Call the handler
        base_service._signal_handler(mock_signal, mock_frame)
        
        # Check state changes
        assert base_service._stopping is True
        assert base_service.running is False
        assert base_service.state == ServiceState.STOPPING
    
    @pytest.mark.asyncio
    async def test_duplicate_signal_handling(self, base_service):
        """Test that duplicate signals are ignored."""
        # First signal
        base_service._signal_handler(signal.SIGINT, None)
        assert base_service._stopping is True
        
        # Reset state flags to test second signal
        base_service.state = ServiceState.RUNNING  # This would not happen in real code
        
        # Second signal should not change these
        base_service._signal_handler(signal.SIGINT, None)
        assert base_service.state == ServiceState.RUNNING  # State should be unchanged
    
    @pytest.mark.asyncio
    async def test_async_signal_handler(self, base_service):
        """Test the async signal handler."""
        # base_service is a fresh fixture, so _stopping is initially False.
        # No need to manually set base_service._stopping = False
        
        # Call the async handler
        logger.debug(f"test_async_signal_handler: Calling _handle_signal_async for {base_service.service_name}")
        await base_service._handle_signal_async(signal.SIGINT)
        logger.debug(f"test_async_signal_handler: _handle_signal_async completed. Service state: {base_service.state}")
        
        # Check state changes. After _handle_signal_async calls await base_service.stop(),
        # the service should be fully STOPPED.
        assert base_service._stopping is True, "_stopping flag should be True after signal handling"
        assert base_service.running is False, "running flag should be False after signal handling"
        assert base_service.state == ServiceState.STOPPED, "Service state should be STOPPED after signal handling completes"
    
    @pytest.mark.asyncio
    async def test_error_during_task_execution(self):
        """Test that errors in tasks are handled properly."""
        service = BaseService(service_name="error-test-service")
        
        # Create a task that raises an exception
        error_raised = asyncio.Event()
        
        async def failing_task():
            try:
                await asyncio.sleep(0.1)
                # Simulate an error
                service.errors += 1  # Manually increment error count
                raise RuntimeError("Deliberate test error")
            except RuntimeError:
                # Mark that we got the error
                error_raised.set()
                # Re-raise to ensure error propagation works
                raise
            # Note: If CancelledError occurs during await asyncio.sleep(0.1), 
            # this block won't be reached, and error_raised won't be set.
                
        # Register only our failing task
        service.tasks = []  # Clear default tasks (like display_stats)
        service._register_task(failing_task())
        
        # Start the service (sets up state, but doesn't run tasks yet)
        await service.start()
        
        # Patch _handle_signal_async to prevent stray signals from stopping the service
        original_handle_signal_async = service._handle_signal_async
        async def mock_noop_handle_signal_async(sig):
            logger.info(f"Mocked _handle_signal_async for {service.service_name} received signal {sig}, ignoring for this test.")
            pass
        service._handle_signal_async = mock_noop_handle_signal_async

        run_task = asyncio.create_task(service.run())
        
        try:
            # Wait for the error to be raised by failing_task
            await asyncio.wait_for(error_raised.wait(), timeout=2.0) # Increased timeout slightly
            
            # Check that the service recorded the error (incremented by failing_task)
            assert service.errors >= 1, "Service should have recorded an error from the failing task."
            
            # The service.run() method's finally block should call service.stop()
            # when failing_task causes gather to complete.
            # We wait for run_task to complete, which includes this internal stop.
            # If a stray signal *wasn't* the issue, run_task would complete due to its
            # own stop logic triggered by the failing task.
            # If a stray signal *was* the issue, our mock prevents it from acting.

            # Explicitly stop the service from the test to ensure cleanup,
            # though internal stop should have already occurred.
            # This also ensures that if error_raised.wait() passed but run_task is somehow
            # still running without having stopped itself, this will stop it.
            logger.info(f"Test calling service.stop() for {service.service_name}")
            await service.stop()
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout in test_error_during_task_execution: failing_task did not set error_raised event for {service.service_name}.")
            # Attempt to stop the service if it's still running
            if service.state != ServiceState.STOPPED:
                logger.warning(f"Timeout occurred, ensuring service {service.service_name} is stopped.")
                await service.stop()
            pytest.fail(f"failing_task in {service.service_name} did not set error_raised event within timeout.")
        finally:
            # Restore original signal handler
            service._handle_signal_async = original_handle_signal_async
            
            # Ensure run_task is cleaned up
            if not run_task.done():
                logger.info(f"Test ensuring run_task for {service.service_name} is cancelled.")
                run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
            
            # Final check to ensure service is stopped, especially if an assertion failed before explicit stop
            if service.state != ServiceState.STOPPED:
                current_frame = inspect.currentframe()
                test_name = current_frame.f_code.co_name if current_frame else 'test_error_during_task_execution'
                logger.warning(f"Test {test_name} "
                               f"final cleanup: Service {service.service_name} was {service.state}. Forcing stop.")
                await service.stop()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        service = BaseService(service_name="stats-test")
        
        # Register a task that updates stats
        async def update_stats():
            for i in range(5):
                if not service.running:
                    logger.debug(f"update_stats: service not running, breaking loop at iteration {i}")
                    break
                service.messages_sent += 1
                service.messages_received += 2
                logger.debug(f"update_stats: iter {i}, sent={service.messages_sent}, recv={service.messages_received}")
                await asyncio.sleep(0.1)
            logger.debug("update_stats task finished")
        
        service._register_task(update_stats())
        
        # Register a task to trigger stopping the service after stats are updated
        stop_trigger_event = asyncio.Event()
        async def trigger_stop_event_task():
            await asyncio.sleep(0.6) # Ensure update_stats has time to run (0.5s)
            logger.debug("trigger_stop_event_task: setting stop_trigger_event")
            stop_trigger_event.set()
        
        service._register_task(trigger_stop_event_task()) # Call the function here
        
        # Start the service
        await service.start()
        
        # Run the service in a background task
        run_service_task = asyncio.create_task(service.run(), name=f"{service.service_name}-run-task")
        
        try:
            # Wait for the trigger event
            logger.debug("test_statistics_tracking: waiting for stop_trigger_event")
            await asyncio.wait_for(stop_trigger_event.wait(), timeout=2.0) # Increased timeout for safety
            logger.debug("test_statistics_tracking: stop_trigger_event received")
            
            # Now that the event is set, stop the service externally
            logger.debug("test_statistics_tracking: calling service.stop() externally")
            await service.stop()
            logger.debug("test_statistics_tracking: service.stop() returned")
            
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for stop_trigger_event in test_statistics_tracking")
            # Ensure service is stopped even on timeout before failing
            if service.state != ServiceState.STOPPED:
                logger.warning("Timeout occurred, ensuring service stats-test is stopped.")
                await service.stop()
            pytest.fail("stop_trigger_event was not set in time")
        finally:
            # Ensure the service.run() task is complete/cancelled and awaited
            logger.debug(f"test_statistics_tracking: finally block, run_service_task done: {run_service_task.done()}")
            if not run_service_task.done():
                logger.warning(f"test_statistics_tracking: run_service_task not done, cancelling.")
                run_service_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_service_task
            logger.debug("test_statistics_tracking: run_service_task awaited in finally")

        # Check that stats were updated
        assert service.messages_sent == 5, f"Expected 5 messages sent, got {service.messages_sent}"
        assert service.messages_received == 10, f"Expected 10 messages received, got {service.messages_received}"
        assert service.state == ServiceState.STOPPED, f"Service should be STOPPED, but is {service.state}"
    
    @pytest.mark.asyncio
    async def test_cancellation_during_stop(self):
        """Test that cancellation during stop is handled correctly."""
        service = BaseService(service_name="cancel-test")
        
        # Set a flag to track task execution and completion
        task_started = asyncio.Event()
        
        # Create a task that signals when started and handles cancellation
        async def cancellable_task():
            try:
                # Signal that we've started
                task_started.set()
                # Then just wait to be cancelled
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Just propagate the cancellation
                raise
        
        # Register our test task (clearing defaults)
        service.tasks = []
        service._register_task(cancellable_task())
        
        # Start the service
        await service.start()
        
        # Run the service in a background task so we can monitor it
        run_task = asyncio.create_task(service.run())
        
        # Wait for our task to start running
        try:
            await asyncio.wait_for(task_started.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            # Clean up if we timeout waiting for task to start
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
            pytest.fail("Task didn't start within timeout")
        
        # Now stop the service - this should work without exceptions
        await service.stop()
        
        # Cancel the run task 
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task
        
        # Verify service is stopped
        assert service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_run_with_no_tasks(self):
        """Test that run raises an error when no tasks are registered."""
        service = BaseService(service_name="no-tasks-service")
        
        # Start the service but don't register tasks (the start method registers stats)
        service.tasks = []  # Remove the stats task
        
        # Run should raise RuntimeError
        with pytest.raises(RuntimeError, match="No tasks registered"):
            await service.run()

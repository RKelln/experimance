#!/usr/bin/env python3
"""
Integration test for signal handling in service classes.

This test specifically validates:
1. Proper handling of Ctrl+C (SIGINT)
2. Graceful shutdown on SIGTERM
3. Multiple signal handling
4. ZMQ resource cleanup

Run with:
    uv run -m utils.tests.test_service_signals -v
"""

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import suppress

from experimance_common.service import (
    BaseService, BaseZmqService, ZmqPublisherService, ServiceState
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestSimpleService(BaseService):
    """Simple service for testing signal handling."""
    
    def __init__(self, name="signal-test-service"):
        super().__init__(service_name=name, service_type="test")
        self.iterations = 0
    
    async def start(self):
        """Start the service."""
        await super().start()
        
        # Register a simple task that counts
        self._register_task(self.count_task())
        logger.info("TestSimpleService started")
    
    async def count_task(self):
        """Simple task that counts iterations."""
        while self.running:
            self.iterations += 1
            logger.info(f"Iteration #{self.iterations}")
            await asyncio.sleep(1.0)


class TestZmqService(ZmqPublisherService):
    """ZMQ service for testing signal handling with ZMQ resources."""
    
    def __init__(self, name="zmq-signal-test-service"):
        super().__init__(
            service_name=name,
            pub_address="tcp://*:15555",
            heartbeat_topic="test.signal"
        )
        self.iterations = 0
    
    async def start(self):
        """Start the ZMQ service."""
        await super().start()
        
        # Register a simple task that counts and publishes
        self._register_task(self.count_and_publish_task())
        logger.info("TestZmqService started")
    
    async def count_and_publish_task(self):
        """Task that counts and publishes the count."""
        while self.running:
            self.iterations += 1
            
            # Publish the count
            message = {
                "count": self.iterations,
                "timestamp": time.time()
            }
            
            await self.publish_message(message)
            logger.info(f"Published count #{self.iterations}")
            
            await asyncio.sleep(1.0)


async def test_basic_service_signal():
    """Test signal handling in BasicService."""
    service = TestSimpleService(name="basic-signal-test")
    
    logger.info("Starting basic service signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service
    run_task = asyncio.create_task(service.run())
    
    # Let it run for a bit
    await asyncio.sleep(3.0)
    
    # Send a SIGINT to the process (simulating Ctrl+C)
    logger.info("Sending SIGINT to process")
    os.kill(os.getpid(), signal.SIGINT)
    
    await wait_for_service_shutdown(run_task, service)

    logger.info(f"Service state after SIGINT: {service.state}")
    assert service.state == ServiceState.STOPPED
    assert not service.running
    assert service._stopping  # Should have been set by stop()
    logger.info("Basic service signal test completed successfully")


async def wait_for_service_shutdown(service_run_task: asyncio.Task, service: BaseService, timeout: float = 5.0):
    """
    Waits for the service.run() task to complete and the service to reach STOPPED state.
    """
    logger.info(f"Waiting for service {service.service_name} to shut down (run task: {service_run_task.get_name()})...")
    try:
        # Wait for the service's main run task to complete.
        # This task should finish as a result of service.stop() being called (e.g., by a signal).
        await asyncio.wait_for(service_run_task, timeout=timeout)
        logger.info(f"Service {service.service_name} run task completed. Current service state: {service.state}")
    except asyncio.CancelledError:
        logger.info(f"Service {service.service_name} run task was cancelled, as expected during shutdown. Current service state: {service.state}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for service {service.service_name} run task to complete (timeout: {timeout}s). Current state: {service.state}")
        # Log details if timeout occurs
        all_tasks = asyncio.all_tasks()
        logger.error(f"Dumping all {len(all_tasks)} asyncio tasks at timeout:")
        for i, task in enumerate(all_tasks):
            logger.error(f"  Task {i}: {task.get_name()}, done: {task.done()}, cancelled: {task.cancelled()}")
            if not task.done():
                task.print_stack(file=sys.stderr) # Print stack to stderr
        if not service_run_task.done():
            logger.error(f"Service {service.service_name} run task is still not done. Attempting to cancel it now.")
            service_run_task.cancel()
            with suppress(asyncio.CancelledError): # Suppress if already cancelled
                await service_run_task
        assert False, f"Service {service.service_name} run task did not complete in time. State: {service.state}"
    except Exception as e:
        logger.error(f"Unexpected error waiting for {service.service_name} run task: {e!r}", exc_info=True)
        assert False, f"Unexpected error waiting for {service.service_name} shutdown: {e!r}"

    # After the run_task has finished, service.stop() should have set the state to STOPPED.
    # A very short poll can confirm this, mainly for robustness against tiny timing windows.
    if service.state != ServiceState.STOPPED:
        logger.debug(f"Service {service.service_name} state is {service.state}, polling briefly for STOPPED state...")
        await asyncio.sleep(0.1) # Brief pause for final state transition if needed

    assert service.state == ServiceState.STOPPED, f"Service {service.service_name} should be STOPPED, but is {service.state}"
    logger.info(f"Service {service.service_name} successfully shut down and confirmed STOPPED.")


async def test_zmq_service_signal():
    """Test signal handling in ZmqService."""
    service = TestZmqService(name="zmq-signal-test")
    
    logger.info("Starting ZMQ service signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service
    run_task = asyncio.create_task(service.run())
    
    # Let it run for a bit
    await asyncio.sleep(3.0)
    
    # Send a SIGTERM to the process
    logger.info("Sending SIGTERM to process")
    os.kill(os.getpid(), signal.SIGTERM)
    
    await wait_for_service_shutdown(run_task, service)

    logger.info(f"Service state after SIGTERM: {service.state}")
    assert service.state == ServiceState.STOPPED
    assert not service.running
    assert service._stopping

    # Ensure ZMQ resources are cleaned up
    # Depending on mock setup, 'closed' might be an attribute or a method call check
    assert service.publisher is not None, "Publisher should exist"
    assert service.publisher.closed, "Publisher socket should be closed"

    logger.info("ZMQ service signal test completed successfully")


async def test_multiple_signals():
    """Test handling multiple signals in succession."""
    service = TestSimpleService(name="multi-signal-test")
    
    logger.info("Starting multiple signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service
    run_task = asyncio.create_task(service.run(), name=f"{service.service_name}-run")
    
    # Let it run for a bit to ensure it's fully up
    await asyncio.sleep(1.0) # Reduced from 2.0 to speed up, ensure tasks are running
    assert service.running, "Service should be running before sending signals"
    
    # Send multiple signals
    # The first signal should initiate shutdown. Subsequent signals should be ignored by the handler.
    logger.info(f"Sending first SIGINT to {service.service_name}")
    os.kill(os.getpid(), signal.SIGINT)
    
    # Give a very short moment for the first signal to be processed by the event loop
    # and for the service to enter the stopping state.
    await asyncio.sleep(0.2) 
    
    logger.info(f"Sending second SIGINT to {service.service_name} (should be ignored if already stopping)")
    os.kill(os.getpid(), signal.SIGINT)
    await asyncio.sleep(0.1)

    logger.info(f"Sending third SIGINT to {service.service_name} (should be ignored if already stopping)")
    os.kill(os.getpid(), signal.SIGINT)
    
    # Now wait for the service to fully shut down based on the first signal.
    # The timeout here is critical. If stop() hangs, this will catch it.
    await wait_for_service_shutdown(run_task, service, timeout=5.0) 
    
    # Check that service is in a stopped state
    logger.info(f"Service {service.service_name} state after multiple signals: {service.state}")
    assert service.state == ServiceState.STOPPED, f"Service {service.service_name} expected STOPPED, got {service.state}"
    assert not service.running, f"Service {service.service_name} should not be running"
    assert service._stopping, f"Service {service.service_name} _stopping flag should be True"
    logger.info("Multiple signal test completed")


async def main():
    """Run all signal tests."""
    logger.info("Starting service signal tests")
    
    try:
        # Run basic service test
        await test_basic_service_signal()
        
        await asyncio.sleep(1.0)  # Brief pause between tests
        
        # Run ZMQ service test
        await test_zmq_service_signal()
        
        await asyncio.sleep(1.0)  # Brief pause between tests
        
        # Run multiple signal test
        await test_multiple_signals()
        
        logger.info("All signal tests passed!")
        
    except Exception as e:
        logger.error(f"Signal test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by keyboard")
        sys.exit(1)
    except Exception as e:
        print(f"Unhandled error: {e}")
        sys.exit(1)

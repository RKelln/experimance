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

from experimance_common.base_service import BaseService, ServiceState
from experimance_common.zmq.publisher import ZmqPublisherService

from utils.tests.test_utils import wait_for_service_shutdown, wait_for_service_state

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestSimpleService(BaseService):
    """Simple service for testing signal handling."""
    __test__ = False  # Prevent pytest from treating this as a test case
    
    def __init__(self, name="signal-test-service"):
        super().__init__(service_name=name, service_type="test")
        self.iterations = 0
    
    async def start(self):
        """Start the service."""
        await super().start()
        
        # Register a simple task that counts
        self.add_task(self.count_task())
        logger.info("TestSimpleService started")
    
    async def count_task(self):
        """Simple task that counts iterations."""
        while self.state == ServiceState.RUNNING:
            self.iterations += 1
            logger.info(f"Iteration #{self.iterations}")
            await asyncio.sleep(1.0)


class TestZmqService(ZmqPublisherService):
    """ZMQ service for testing signal handling with ZMQ resources."""
    __test__ = False  # Prevent pytest from treating this as a test case
    
    def __init__(self, name="zmq-signal-test"):
        super().__init__(
            service_name=name,
            pub_address="tcp://*:15555",
            topic="test.signal"
        )
        self.iterations = 0
    
    async def start(self):
        """Start the ZMQ service."""
        await super().start()
        
        # Register a simple task that counts and publishes
        self.add_task(self.count_and_publish_task())
        logger.info("TestZmqService started")
    
    async def count_and_publish_task(self):
        """Task that counts and publishes the count."""
        while self.state == ServiceState.RUNNING:
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
    
    # Create a task to run the service - this will move it to RUNNING state
    run_task = asyncio.create_task(service.run())
    
    # Wait for the service to be fully running
    await wait_for_service_state(service, ServiceState.RUNNING)
    
    # Let it run for a bit to ensure tasks are established
    await asyncio.sleep(1.0)
    
    # Send a SIGINT to the process (simulating Ctrl+C)
    logger.info("Sending SIGINT to process")
    os.kill(os.getpid(), signal.SIGINT)
    
    await wait_for_service_shutdown(run_task, service)

    logger.info(f"Service state after SIGINT: {service.state}")
    assert service.state == ServiceState.STOPPED
    assert service.state in [ServiceState.STOPPING, ServiceState.STOPPED]  # Should be in stopping/stopped state
    logger.info("Basic service signal test completed successfully")


async def test_zmq_service_signal():
    """Test signal handling in ZmqService."""
    service = TestZmqService(name="zmq-signal-test")
    
    logger.info("Starting ZMQ service signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service - this will move it to RUNNING state
    run_task = asyncio.create_task(service.run())
    
    # Wait for the service to be fully running
    await wait_for_service_state(service, ServiceState.RUNNING)
    
    # Let it run for a bit to ensure tasks are established
    await asyncio.sleep(1.0)
    
    # Send a SIGTERM to the process
    logger.info("Sending SIGTERM to process")
    os.kill(os.getpid(), signal.SIGTERM)
    
    await wait_for_service_shutdown(run_task, service)

    logger.info(f"Service state after SIGTERM: {service.state}")
    assert service.state == ServiceState.STOPPED
    assert service.state in [ServiceState.STOPPING, ServiceState.STOPPED]  # Should be in stopping/stopped state

    # Ensure ZMQ resources are cleaned up
    # Depending on mock setup, 'closed' might be an attribute or a method call check
    assert service.publisher is not None, "Publisher should exist"
    assert service.publisher.closed, "Publisher socket should be closed"

    logger.info("ZMQ service signal test completed successfully")


async def test_zmq_service_sigint_signal():
    """Test signal handling in ZmqService."""
    service = TestZmqService(name="zmq-sigint-test")
    
    logger.info("Starting ZMQ service signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service - this will move it to RUNNING state
    run_task = asyncio.create_task(service.run())
    
    # Wait for the service to be fully running
    await wait_for_service_state(service, ServiceState.RUNNING)
    
    # Let it run for a bit to ensure tasks are established
    await asyncio.sleep(1.0)
    
    # Send a SIGTERM to the process
    logger.info("Sending SIGINT to process")
    os.kill(os.getpid(), signal.SIGINT)
    
    await wait_for_service_shutdown(run_task, service)

    logger.info(f"Service state after SIGINT: {service.state}")
    assert service.state == ServiceState.STOPPED
    assert service.state in [ServiceState.STOPPING, ServiceState.STOPPED]  # Should be in stopping/stopped state

    # Ensure ZMQ resources are cleaned up
    # Depending on mock setup, 'closed' might be an attribute or a method call check
    assert service.publisher is not None, "Publisher should exist"
    assert service.publisher.closed, "Publisher socket should be closed"

    logger.info("ZMQ service signal test completed successfully")


async def test_zmq_service_sigtstp_signal():
    """Test SIGTSTP (Ctrl+Z) handling in ZmqService."""
    service = TestZmqService(name="zmq-sigtstp-test")
    
    logger.info("Starting ZMQ service SIGTSTP test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service - this will move it to RUNNING state
    run_task = asyncio.create_task(service.run())
    
    # Wait for the service to be fully running
    await wait_for_service_state(service, ServiceState.RUNNING)
    
    # Let it run for a bit to ensure tasks are established
    await asyncio.sleep(1.0)
    
    # Send a SIGTSTP to the process
    logger.info("Sending SIGTSTP to process")
    os.kill(os.getpid(), signal.SIGTSTP)
    
    await wait_for_service_shutdown(run_task, service)

    logger.info(f"Service state after SIGTSTP: {service.state}")
    assert service.state == ServiceState.STOPPED

    # Ensure ZMQ resources are cleaned up
    assert service.publisher is not None, "Publisher should exist"
    assert service.publisher.closed, "Publisher socket should be closed"

    logger.info("ZMQ service SIGTSTP test completed successfully")


async def test_multiple_signals():
    """Test handling multiple signals in succession."""
    service = TestSimpleService(name="multi-signal-test")
    
    logger.info("Starting multiple signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service - this will move it to RUNNING state
    run_task = asyncio.create_task(service.run(), name=f"{service.service_name}-run")
    
    # Wait for the service to be fully running
    await wait_for_service_state(service, ServiceState.RUNNING)
    assert service.state == ServiceState.RUNNING, "Service should be running before sending signals"
    
    # Send multiple signals
    # The first signal should initiate shutdown. Subsequent signals should be ignored by the handler.
    logger.info(f"Sending first SIGINT to {service.service_name}")
    os.kill(os.getpid(), signal.SIGINT)
    
    # Wait for the service to finish shutdown
    await wait_for_service_state(service, ServiceState.STOPPED, timeout=2.0)
    
    logger.info(f"Sending second SIGINT to {service.service_name} (should be ignored if already stopping)")
    os.kill(os.getpid(), signal.SIGINT)
    
    # Brief pause to ensure signal is processed
    await asyncio.sleep(0.1)

    logger.info(f"Sending third SIGINT to {service.service_name} (should be ignored if already stopping)")
    os.kill(os.getpid(), signal.SIGINT)
    
    # Now wait for the service to fully shut down based on the first signal.
    # The timeout here is critical. If stop() hangs, this will catch it.
    await wait_for_service_shutdown(run_task, service, timeout=5.0) 
    
    # Check that service is in a stopped state
    logger.info(f"Service {service.service_name} state after multiple signals: {service.state}")
    assert service.state == ServiceState.STOPPED, f"Service {service.service_name} expected STOPPED, got {service.state}"
    logger.info("Multiple signal test completed")


async def main():
    """Run all signal tests."""
    logger.info("Starting service signal tests")
    
    try:
        # Run basic service test
        await test_basic_service_signal()
        
        # Brief pause to reset signal handlers and ensure clean state between tests
        await asyncio.sleep(0.5)  
        
        # Run ZMQ service test
        await test_zmq_service_signal()
        
        # Brief pause to reset signal handlers and ensure clean state between tests
        await asyncio.sleep(0.5)  
        
        # Run multiple signal test
        await test_multiple_signals()
        
        # Brief pause to reset signal handlers and ensure clean state between tests
        await asyncio.sleep(0.5)
        
        # Run SIGTSTP test
        await test_zmq_service_sigtstp_signal()
        
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

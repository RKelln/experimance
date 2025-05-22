#!/usr/bin/env python3
"""
Integration test for signal handling in service classes.

This test specifically validates:
1. Proper handling of Ctrl+C (SIGINT)
2. Graceful shutdown on SIGTERM
3. Multiple signal handling
4. ZMQ resource cleanup

Run with:
    uv run -m utils.tests.test_service_signals
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
    
    # Wait for the service to stop
    with suppress(asyncio.CancelledError):
        await run_task
    
    # Check that service stopped properly
    logger.info(f"Service state after SIGINT: {service.state}")
    assert service.state == ServiceState.STOPPED
    
    logger.info("Basic service signal test completed successfully")


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
    
    # Wait for the service to stop
    with suppress(asyncio.CancelledError):
        await run_task
    
    # Check that service stopped properly
    logger.info(f"Service state after SIGTERM: {service.state}")
    assert service.state == ServiceState.STOPPED
    
    # Ensure publisher was closed
    assert service.publisher is not None
    assert getattr(service.publisher, "closed", False) is True
    
    logger.info("ZMQ service signal test completed successfully")


async def test_multiple_signals():
    """Test handling multiple signals in succession."""
    service = TestSimpleService(name="multi-signal-test")
    
    logger.info("Starting multiple signal test")
    
    # Start the service
    await service.start()
    
    # Create a task to run the service
    run_task = asyncio.create_task(service.run())
    
    # Let it run for a bit
    await asyncio.sleep(2.0)
    
    # Send multiple signals in rapid succession
    for _ in range(3):
        logger.info("Sending SIGINT")
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.1)
    
    # Wait for the service to stop
    with suppress(asyncio.CancelledError):
        await run_task
    
    # Check that service stopped properly
    logger.info(f"Service state after multiple signals: {service.state}")
    assert service.state == ServiceState.STOPPED
    
    logger.info("Multiple signal test completed successfully")


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

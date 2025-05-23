#!/usr/bin/env python3
"""
Example implementation using the basic BaseService class from experimance_common.service.

This example demonstrates how to:
1. Create services using the BaseService class without ZeroMQ
2. Implement periodic tasks
3. Use standard service lifecycle methods (start, stop, run)
4. Implement proper error handling and shutdown

Usage:
    # Run the example
    uv run -m utils.examples.basic_service_example
"""

import argparse
import asyncio
import logging
import random
import signal
import sys
import time
from typing import Dict, Any, List, Optional

from experimance_common.service import (
    BaseService,
    ServiceState
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExampleBasicService(BaseService):
    """Example service using the base BaseService class.
    
    This service demonstrates:
    1. Implementing a service without ZeroMQ dependencies
    2. Setting up periodic tasks
    3. Proper lifecycle management
    4. Error handling
    """
    
    def __init__(self, name: str = "example-basic"):
        """Initialize the basic service.
        
        Args:
            name: Name of this service instance
        """
        super().__init__(
            service_name=name,
            service_type="example-basic"
        )
        self.operations_performed = 0
        self.last_operation_time = time.time()
        
    async def start(self):
        """Start the service with additional tasks."""
        await super().start()
        
        # Add custom tasks
        self._register_task(self.perform_periodic_operation())
        self._register_task(self.simulate_random_errors())
        
        logger.info(f"Service {self.service_name} started with all tasks registered")
    
    async def perform_periodic_operation(self):
        """Perform a periodic operation."""
        while self.running:
            try:
                # Simulate doing some work
                duration = random.uniform(0.1, 0.5)
                await asyncio.sleep(duration)
                
                self.operations_performed += 1
                self.last_operation_time = time.time()
                
                logger.info(f"Operation #{self.operations_performed} completed (took {duration:.2f}s)")
                
                # Count this as a "message" for statistics
                self.messages_sent += 1
                
            except Exception as e:
                logger.error(f"Error in periodic operation: {e}")
                self.errors += 1
            
            # Wait between operations
            await asyncio.sleep(random.uniform(1.0, 3.0))
    
    async def simulate_random_errors(self):
        """Occasionally simulate errors to demonstrate error handling."""
        while self.running:
            await asyncio.sleep(2.0 * random.random() + 2.0)  # every few seconds
            
            try:
                # Simulate an error
                logger.warning("Simulating a recoverable error...")
                raise RuntimeError("Simulated error")
            except Exception as e:
                logger.error(f"Caught error: {e}")
                self.errors += 1


async def main():
    """Main entry point for the example service."""
    parser = argparse.ArgumentParser(description="Basic Service Example")
    parser.add_argument("--name", type=str, help="Service instance name")
    args = parser.parse_args()
    
    name = args.name or "basic-service-1"
    service = ExampleBasicService(name=name)

    await service.start()
    await service.run()

if __name__ == "__main__":
    asyncio.run(main())


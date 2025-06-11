"""
Example demonstrating recommended shutdown and error handling patterns for BaseService.

This example shows the correct ways to handle graceful shutdown and error conditions
in services that inherit from BaseService.
"""

import asyncio
import logging
from experimance_common.base_service import BaseService, ServiceStatus
from experimance_common.service_state import ServiceState

# Configure logging for the example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleService(BaseService):
    """Example service demonstrating proper shutdown and error handling patterns."""
    
    def __init__(self):
        super().__init__("example_service", "demo")
        self.work_counter = 0
    
    async def start(self):
        """Start the service and add tasks."""
        await super().start()
        
        # Add some example tasks
        self.add_task(self.main_work_loop())
        self.add_task(self.monitoring_task())
    
    async def main_work_loop(self):
        """Main work loop that demonstrates different shutdown scenarios."""
        logger.info("Starting main work loop")
        
        while self.running:
            try:
                # Simulate some work
                await asyncio.sleep(1)
                self.work_counter += 1
                
                # Demonstrate different shutdown scenarios
                if self.work_counter == 5:
                    # Scenario 1: Graceful self-shutdown
                    logger.info("Work complete, initiating graceful shutdown")
                    await self.stop()  # ✅ RECOMMENDED: Use stop() for immediate shutdown
                    break
                
                elif self.work_counter == 10:
                    # Scenario 2: Request shutdown without blocking
                    logger.info("Requesting shutdown from work loop")
                    self.request_stop()  # ✅ RECOMMENDED: Use request_stop() for non-blocking
                    # Continue working until shutdown completes
                    
                elif self.work_counter == 15:
                    # Scenario 3: Fatal error should auto-shutdown
                    logger.info("Simulating fatal error")
                    try:
                        raise RuntimeError("Simulated fatal error")
                    except Exception as e:
                        self.record_error(e, is_fatal=True)  # ✅ RECOMMENDED: Auto-stops on fatal
                        break
                
                elif self.work_counter == 20:
                    # Scenario 4: Non-fatal error (service continues)
                    logger.info("Simulating recoverable error")
                    try:
                        raise ValueError("Simulated recoverable error")
                    except Exception as e:
                        self.record_error(e, is_fatal=False)  # ✅ RECOMMENDED: Service continues
                
                logger.info(f"Work iteration {self.work_counter}")
                
            except Exception as e:
                logger.error(f"Unexpected error in work loop: {e}")
                self.record_error(e)
                break
    
    async def monitoring_task(self):
        """Monitoring task that demonstrates proper error handling."""
        logger.info("Starting monitoring task")
        
        while self.running:
            try:
                await asyncio.sleep(2)
                logger.info(f"Service status: {self.status}, Work counter: {self.work_counter}")
                
                # Example of checking for error conditions
                if self.status == ServiceStatus.ERROR and self.errors > 3:
                    logger.warning("Too many errors detected, requesting shutdown")
                    self.request_stop()
                    break
                    
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
                self.record_error(e)
                break


async def demonstrate_patterns():
    """Demonstrate the different shutdown and error patterns."""
    service = ExampleService()
    
    logger.info("=" * 50)
    logger.info("DEMONSTRATING RECOMMENDED PATTERNS")
    logger.info("=" * 50)
    
    try:
        await service.start()
        await service.run()
    except Exception as e:
        logger.error(f"Service failed: {e}")
    finally:
        logger.info(f"Service final state: {service.state}")
        logger.info(f"Service final status: {service.status}")
        logger.info(f"Total errors recorded: {service.errors}")


async def demonstrate_bad_patterns():
    """Demonstrate what NOT to do."""
    service = ExampleService()
    
    logger.info("=" * 50)
    logger.info("DEMONSTRATING BAD PATTERNS (DON'T DO THIS)")
    logger.info("=" * 50)
    
    await service.start()
    
    # ❌ BAD: Setting state directly bypasses cleanup
    logger.warning("BAD PATTERN: Setting state directly")
    service.state = ServiceState.STOPPING  # This doesn't actually stop the service!
    
    # ❌ BAD: Not calling record_error for exceptions
    try:
        raise RuntimeError("Unhandled error")
    except Exception as e:
        logger.error(f"BAD PATTERN: Error not recorded: {e}")
        # Should call: service.record_error(e, is_fatal=True)
    
    # Clean up properly
    await service.stop()


if __name__ == "__main__":
    async def main():
        await demonstrate_patterns()
        await asyncio.sleep(1)  # Brief pause between demos
        await demonstrate_bad_patterns()
    
    asyncio.run(main())

#!/usr/bin/env python3
"""
Example service using the experimance-common library.
This is a simple ZeroMQ publisher/subscriber example.
"""

import asyncio
import logging
from typing import Dict, Any

from experimance_common import (
    ZmqPublisher, 
    ZmqSubscriber, 
    MessageType,
    DEFAULT_PORTS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("example_service")

class ExampleService:
    """Example service that publishes and subscribes to messages."""
    
    def __init__(self):
        """Initialize the example service."""
        # Create ZMQ addresses from port numbers
        example_pub_address = f"tcp://*:{DEFAULT_PORTS['example_pub']}"
        coordinator_pub_address = f"tcp://localhost:{DEFAULT_PORTS['coordinator_pub']}"
        
        self.publisher = ZmqPublisher(example_pub_address)
        self.subscriber = ZmqSubscriber(
            coordinator_pub_address,
            topics=[MessageType.HEARTBEAT, MessageType.ERA_CHANGED]
        )
        self.running = False
        
    async def start(self):
        """Start the service."""
        logger.info("Starting example service")
        self.running = True
        # ZmqPublisher and ZmqSubscriber already connect in their __init__ methods
        
        try:
            # Start message processing
            await asyncio.gather(
                self.publish_loop(),
                self.message_loop()
            )
        except asyncio.CancelledError:
            logger.info("Service shutdown requested")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown the service."""
        logger.info("Shutting down example service")
        self.running = False
        await self.publisher.close()
        await self.subscriber.close()
            
    async def publish_loop(self):
        """Publish status messages periodically."""
        while self.running:
            status_message = {
                "type": MessageType.IDLE_STATUS,
                "service": "example",
                "status": "idle"
            }
            await self.publisher.publish(MessageType.IDLE_STATUS, status_message)
            await asyncio.sleep(5)
            
    async def message_loop(self):
        """Process incoming messages."""
        logger.info("Waiting for messages")
        while self.running:
            try:
                message_type, message_data = await self.subscriber.receive()
                logger.info(f"Received: {message_type} {message_data}")
                
                if message_type == MessageType.ERA_CHANGED:
                    # Handle era change
                    logger.info(f"Era changed to: {message_data.get('era')}")
            except Exception as e:
                logger.error(f"Error in message loop: {str(e)}")
                await asyncio.sleep(1)
                
async def main():
    """Run the example service."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Example Experimance service")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for messages, just test setup and exit")
    args = parser.parse_args()
    
    service = ExampleService()
    
    if args.no_wait:
        logger.info("Testing setup only (--no-wait option provided)")
        # Just initialize and exit successfully
        logger.info("Setup test successful! All components initialized.")
    else:
        # Normal operation
        await service.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service terminated by user")

#!/usr/bin/env python3
"""
Test script based on the README_ZMQ.md examples.

This demonstrates the simple publisher/subscriber pattern from the README,
using proper BaseConfig integration instead of factory functions.
Run with --publisher or --subscriber to test the examples.
"""

import argparse
import asyncio
import logging
import time

from experimance_common.base_service import BaseService, ServiceStatus
from experimance_common.service_state import ServiceState
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.zmq.services import PubSubService
from experimance_common.constants import DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePublisher(BaseService):
    """Simple publisher from README example - using proper BaseConfig integration."""
    
    def __init__(self, name: str = "simple-publisher"):
        super().__init__(service_name=name, service_type="publisher")
        
        # Use proper BaseConfig integration instead of factory function
        default_config = {
            "name": self.service_name,
            "publisher": {
                "address": ZMQ_TCP_BIND_PREFIX,
                "port": 5555,
                "default_topic": "general"
            },
            "subscriber": None  # Publisher only
        }
        
        self.zmq_config = PubSubServiceConfig.from_overrides(
            default_config=default_config
        )
        self.zmq_service = PubSubService(self.zmq_config)
        self.counter = 0
    
    async def start(self):
        logger.info(f"Starting {self.service_name}")
        
        # Start ZMQ service first
        await self.zmq_service.start()
        
        # Add publishing task to BaseService
        self.add_task(self._publish_loop())
        
        # Call BaseService start (this handles state transitions)
        await super().start()
        
        self.status = ServiceStatus.HEALTHY
        logger.info(f"✅ {self.service_name} started successfully")
    
    async def stop(self):
        logger.info(f"Stopping {self.service_name}")
        
        # Stop ZMQ service first
        await self.zmq_service.stop()
        
        # Call BaseService stop (this handles state transitions and task cleanup)
        await super().stop()
        
        logger.info(f"✅ {self.service_name} stopped successfully")
    
    async def _publish_loop(self):
        logger.info("Starting publish loop")
        while self.running:
            self.counter += 1
            
            message = {
                "type": "heartbeat",
                "service": self.service_name,
                "sequence": self.counter,
                "timestamp": time.time()
            }
            
            await self.zmq_service.publish(message, "heartbeat")
            self.messages_sent += 1
            logger.info(f"📤 Published heartbeat #{self.counter}")
            
            await self._sleep_if_running(2.0)
        logger.info("Publish loop ended")


class SimpleSubscriber(BaseService):
    """Simple subscriber from README example - using proper BaseConfig integration."""
    
    def __init__(self, name: str = "simple-subscriber"):
        super().__init__(service_name=name, service_type="subscriber")
        
        # Use proper BaseConfig integration instead of factory function
        default_config = {
            "name": self.service_name,
            "publisher": None,  # Subscriber only
            "subscriber": {
                "address": ZMQ_TCP_CONNECT_PREFIX,
                "port": 5555,
                "topics": ["heartbeat", "status"]
            }
        }
        
        self.zmq_config = PubSubServiceConfig.from_overrides(
            default_config=default_config
        )
        self.zmq_service = PubSubService(self.zmq_config)
    
    async def start(self):
        logger.info(f"Starting {self.service_name}")
        
        # Set up handlers before starting
        self.zmq_service.add_message_handler("heartbeat", self._handle_heartbeat)
        self.zmq_service.add_message_handler("status", self._handle_status)
        self.zmq_service.set_default_handler(self._handle_general)
        
        # Start ZMQ service first
        await self.zmq_service.start()
        
        # Call BaseService start (this handles state transitions)
        await super().start()
        
        self.status = ServiceStatus.HEALTHY
        logger.info(f"✅ {self.service_name} started successfully")
    
    async def stop(self):
        logger.info(f"Stopping {self.service_name}")
        
        # Stop ZMQ service first
        await self.zmq_service.stop()
        
        # Call BaseService stop (this handles state transitions)
        await super().stop()
        
        logger.info(f"✅ {self.service_name} stopped successfully")
    
    async def _handle_heartbeat(self, message_data):
        service = message_data.get("service", "unknown")
        sequence = message_data.get("sequence", 0)
        self.messages_received += 1
        logger.info(f"❤️ Heartbeat #{sequence} from {service}")
    
    async def _handle_status(self, message_data):
        service = message_data.get("service", "unknown")
        state = message_data.get("state", "unknown")
        self.messages_received += 1
        logger.info(f"📊 Status from {service}: {state}")
    
    async def _handle_general(self, topic: str, message_data):
        msg_type = message_data.get("type", "unknown")
        self.messages_received += 1
        logger.info(f"📝 Message on '{topic}': {msg_type}")


async def run_publisher():
    """Run the publisher service."""
    service = SimplePublisher()
    try:
        await service.start()
        await service.run()  # This handles signals and will call stop() automatically
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Publisher error: {e}")
    # No finally block - BaseService handles cleanup automatically


async def run_subscriber():
    """Run the subscriber service."""
    service = SimpleSubscriber()
    try:
        await service.start()
        await service.run()  # This handles signals and will call stop() automatically
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Subscriber error: {e}")
    # No finally block - BaseService handles cleanup automatically


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the README ZMQ examples"
    )
    parser.add_argument(
        "--publisher", 
        action="store_true", 
        help="Run as publisher service"
    )
    parser.add_argument(
        "--subscriber", 
        action="store_true", 
        help="Run as subscriber service"
    )
    
    args = parser.parse_args()
    
    if args.publisher:
        logger.info("🚀 Starting README Publisher Example")
        asyncio.run(run_publisher())
    elif args.subscriber:
        logger.info("🚀 Starting README Subscriber Example")
        asyncio.run(run_subscriber())
    else:
        print("Please specify --publisher or --subscriber")
        parser.print_help()


if __name__ == "__main__":
    main()

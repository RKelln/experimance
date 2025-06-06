#!/usr/bin/env python3
"""
Integration test for running multiple services together.

This test specifically validates:
1. Running multiple services in parallel
2. Proper coordination during shutdown
3. Resource cleanup across service types
4. Preventing deadlocks during shutdown

Run with:
    uv run -m utils.tests.test_service_integration -v
"""

import asyncio
import logging
import random
import signal
import sys
import time
from contextlib import suppress

from experimance_common.service import BaseService, BaseZmqService, ServiceState
from experimance_common.zmq.publisher import ZmqPublisherService
from experimance_common.zmq.subscriber import ZmqSubscriberService
from experimance_common.zmq_utils import MessageType
from utils.tests.test_utils import wait_for_service_state, wait_for_service_shutdown, wait_for_service_state_and_status

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsService(BaseService):
    """Simple metrics service that tracks system stats."""
    
    def __init__(self, name="metrics-service"):
        super().__init__(service_name=name, service_type="metrics")
        self.metrics = {
            "cpu": 0.0,
            "memory": 0.0,
            "disk": 0.0,
            "timestamp": 0.0
        }
    
    async def start(self):
        """Start the metrics service."""
        await super().start()
        
        # Register metrics collection task
        self.add_task(self.collect_metrics())
        logger.info("Metrics service started")
    
    async def collect_metrics(self):
        """Collect system metrics periodically."""
        while self.running:
            # Simulate collecting metrics
            self.metrics = {
                "cpu": random.uniform(0.0, 100.0),
                "memory": random.uniform(0.0, 100.0),
                "disk": random.uniform(50.0, 95.0),
                "timestamp": time.time()
            }
            
            logger.debug(f"Collected metrics: {self.metrics}")
            self.messages_sent += 1
            
            # In a real service, we'd use asyncio.sleep for longer periods,
            # but for tests we want faster interaction cycles
            await asyncio.sleep(0.2)


class EventPublisher(ZmqPublisherService):
    """Service that publishes events based on metrics."""
    
    def __init__(self, name="event-publisher", metrics_service=None):
        super().__init__(
            service_name=name,
            pub_address="tcp://*:16555",
            heartbeat_topic="events.heartbeat"
        )
        self.metrics_service = metrics_service
        self.event_count = 0
    
    async def start(self):
        """Start the event publisher service."""
        await super().start()
        
        # Register event publishing task
        if self.metrics_service:
            self.add_task(self.publish_events())
        
        logger.info("Event publisher service started")
    
    async def publish_events(self):
        """Publish events based on metrics."""
        while self.running:
            # Get the latest metrics
            if self.metrics_service:
                metrics = self.metrics_service.metrics
                
                # Generate an event if CPU or memory is high
                if metrics["cpu"] > 80.0 or metrics["memory"] > 80.0:
                    self.event_count += 1
                    event = {
                        "type": MessageType.ALERT,
                        "id": f"event-{self.event_count}",
                        "timestamp": time.time(),
                        "level": "warning",
                        "message": f"High resource usage: CPU={metrics['cpu']:.1f}%, Memory={metrics['memory']:.1f}%"
                    }
                    
                    await self.publish_message(event, topic="events.alert")
                    logger.info(f"Published alert event #{self.event_count}")
            
            # For testing, use shorter sleep periods
            await asyncio.sleep(0.5)


class EventSubscriber(ZmqSubscriberService):
    """Service that subscribes to events."""
    
    def __init__(self, name="event-subscriber"):
        super().__init__(
            service_name=name,
            sub_address="tcp://localhost:16555",
            topics=["events.heartbeat", "events.alert"]
        )
        self.alerts_received = 0
        self.heartbeats_received = 0
    
    async def start(self):
        """Start the event subscriber service."""
        # Register message handlers
        self.register_handler("events.heartbeat", self.handle_heartbeat)
        self.register_handler("events.alert", self.handle_alert)
        
        await super().start()
        logger.info("Event subscriber service started")
    
    def handle_heartbeat(self, message):
        """Handle heartbeat messages."""
        self.heartbeats_received += 1
        logger.debug(f"Received heartbeat #{self.heartbeats_received}")
    
    def handle_alert(self, message):
        """Handle alert messages."""
        self.alerts_received += 1
        level = message.get("level", "unknown")
        msg = message.get("message", "No message")
        
        logger.info(f"Received {level} alert #{self.alerts_received}: {msg}")


async def run_services_integration_test():
    """Run an integration test with multiple services."""
    logger.info("Starting service integration test")
    
    # Create the services
    metrics = MetricsService()
    publisher = EventPublisher(metrics_service=metrics)
    subscriber = EventSubscriber()
    
    # Keep track of running tasks
    service_tasks = []
    
    try:
        # Start each service
        for service in (metrics, publisher, subscriber):
            await service.start()
            service_tasks.append(asyncio.create_task(service.run()))
        
        # Let services run and interact
        logger.info("All services running, waiting to observe interactions...")
        
        # Wait for sufficient interaction between services 
        # (Wait until we've received at least 3 heartbeats)
        start_time = time.monotonic()
        max_wait_time = 10.0  # Maximum time to wait
        
        while (subscriber.heartbeats_received < 3 and 
               time.monotonic() - start_time < max_wait_time):
            await asyncio.sleep(0.1)  # Small sleep to avoid busy waiting
            
        # If we didn't receive any heartbeats in the max wait time, that's a failure
        if subscriber.heartbeats_received == 0:
            assert False, "No heartbeats received within timeout period"
            
        # Verify the services interacted correctly
        logger.info(f"Metrics collected: {metrics.messages_sent}")
        logger.info(f"Events published: {publisher.event_count}")
        logger.info(f"Alerts received: {subscriber.alerts_received}")
        logger.info(f"Heartbeats received: {subscriber.heartbeats_received}")
        
        # Initiate graceful shutdown of all services
        logger.info("Initiating graceful shutdown")
        
        # Stop in reverse order to test proper shutdown sequence
        for service in reversed([metrics, publisher, subscriber]):
            await service.stop()
            # Wait for each service to fully stop before stopping the next one
            await wait_for_service_state(service, ServiceState.STOPPED)
        
        # Cancel all service tasks
        for task in service_tasks:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        
        # Verify all services stopped correctly - this should be redundant after wait_for_service_state above,
        # but we'll keep it as an extra verification
        for i, service in enumerate([metrics, publisher, subscriber]):
            logger.info(f"Service {i+1} state: {service.state}")
            assert service.state == ServiceState.STOPPED
            assert not service.running
            
        logger.info("All services stopped successfully")
        logger.info("Service integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        # Ensure services are stopped even on error
        for service in (metrics, publisher, subscriber):
            if service.state != ServiceState.STOPPED:
                try:
                    await service.stop()
                except Exception as stop_error:
                    logger.error(f"Error stopping service: {stop_error}")
        
        raise


async def main():
    """Run the integration test."""
    try:
        await run_services_integration_test()
    except KeyboardInterrupt:
        logger.info("Test interrupted by keyboard")
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by keyboard")
    except Exception as e:
        print(f"Unhandled error: {e}")
        sys.exit(1)

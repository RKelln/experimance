#!/usr/bin/env python3
"""
ZMQ Messaging Test for Image Server Service.

This script tests the ZeroMQ communication between a client and the Image Server Service
by sending RenderRequest messages and validating responses. It creates a standalone test
that can be run independently to verify ZMQ message flow is working properly.

$ uv run -m services.image_server.tests.test_zmq_messaging
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import zmq
import zmq.asyncio

from experimance_common.zmq.publisher import ZmqPublisher
from experimance_common.zmq.subscriber import ZmqSubscriber
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.schemas import Era, Biome

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("zmq_message_test")


class ZmqMessageTest:
    """Test client for verifying ZMQ communication with the Image Server."""
    
    def __init__(
        self,
        events_pub_address: str,
        images_sub_address: str,
        timeout: float = 30.0,
        debug: bool = False
    ):
        """Initialize the ZMQ Message Test client.
        
        Args:
            events_pub_address: ZMQ address for publishing events
            images_sub_address: ZMQ address for subscribing to images
            timeout: Timeout in seconds for waiting for responses
            debug: Enable debug logging
        """
        self.events_pub_address = events_pub_address
        self.images_sub_address = images_sub_address
        self.timeout = timeout
        self.publisher = None
        self.subscriber = None
        
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def setup(self):
        """Set up ZMQ connections."""
        logger.info("Setting up ZMQ connections...")
        
        # Create publisher for sending RenderRequest messages
        self.publisher = ZmqPublisher(self.events_pub_address, MessageType.RENDER_REQUEST)
        
        # Create subscriber for receiving ImageReady messages
        # Subscribe to an empty string to receive all messages
        self.subscriber = ZmqSubscriber(
            self.images_sub_address, 
            [""]  # An empty string means subscribe to all topics
        )
        
        logger.info(f"Connected to events channel: {self.events_pub_address}")
        logger.info(f"Listening for images on: {self.images_sub_address}")
        
        # Add a small delay to ensure connections are established
        # This helps with the ZMQ slow joiner syndrome
        logger.info("Waiting for ZMQ connections to establish...")
        await asyncio.sleep(1.0)
    
    async def teardown(self):
        """Clean up ZMQ resources."""
        logger.info("Cleaning up ZMQ resources...")
        if self.publisher:
            self.publisher.close()
        if self.subscriber:
            self.subscriber.close()
    
    async def send_render_request(self, prompt: str, era: str, biome: str) -> str:
        """Send a RenderRequest message.
        
        Args:
            prompt: Text prompt for image generation
            era: Era context for the image
            biome: Biome context for the image
            
        Returns:
            The request ID for tracking the request
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Create the message
        message = {
            "type": MessageType.RENDER_REQUEST,
            "request_id": request_id,
            "era": era,
            "biome": biome,
            "prompt": prompt
        }
        
        # Send the message
        logger.info(f"Sending RenderRequest with ID: {request_id}")
        assert self.publisher is not None, "Publisher not initialized"
        assert self.publisher.topic == MessageType.RENDER_REQUEST, "Publisher topic mismatch"
        success = await self.publisher.publish_async(message)
        
        if success:
            logger.info(f"Successfully sent RenderRequest with ID: {request_id}")
        else:
            logger.error("Failed to send RenderRequest")
        
        return request_id
    
    async def listen_for_messages(self, request_id: str, max_duration: float = 10.0) -> List[Dict[str, Any]]:
        """Listen for messages for a specified duration.
        
        Args:
            request_id: The request ID to filter for
            max_duration: Maximum duration to listen for in seconds
            
        Returns:
            List of received messages
        """
        messages = []
        start_time = time.time()
        
        logger.info(f"Listening for messages for {max_duration} seconds...")
        
        while time.time() - start_time < max_duration:
            try:
                # Try to receive a message with timeout
                assert self.subscriber is not None, "Subscriber not initialized"
                topic, message = await self.subscriber.receive_async()
                
                if not message:
                    continue
                
                messages.append(message)
                
                logger.info(f"Received message type: {message.get('type')}")
                
                # If we receive an ImageReady message with our request ID, we're done
                if (message.get("type") == MessageType.IMAGE_READY and 
                    message.get("request_id") == request_id):
                    logger.info(f"Received target message: ImageReady for request {request_id}")
                    break
                
            except Exception as e:
                if "timed out" in str(e).lower():
                    # This is normal, just continue
                    pass
                else:
                    logger.error(f"Error receiving message: {e}")
            
            # Brief pause to avoid tight polling
            await asyncio.sleep(0.1)
        
        duration = time.time() - start_time
        logger.info(f"Listened for {duration:.2f} seconds, received {len(messages)} messages")
        
        return messages


async def run_test():
    """Run the ZMQ messaging test."""
    # Use the unified events channel for all communication
    events_pub_address = f"tcp://*:{DEFAULT_PORTS['events']}"
    images_sub_address = f"tcp://localhost:{DEFAULT_PORTS['events']}"
    
    logger.info("===== Starting ZMQ Messaging Test =====")
    logger.info(f"Events publish address: {events_pub_address}")
    logger.info(f"Images subscribe address: {images_sub_address}")
    
    # Test parameters
    test_prompt = "A forest scene with tall pine trees and a mountain in the background"
    test_era = Era.WILDERNESS.value
    test_biome = Biome.BOREAL_FOREST.value
    
    # Create the test client
    client = ZmqMessageTest(
        events_pub_address=events_pub_address,
        images_sub_address=images_sub_address,
        debug=True
    )
    
    try:
        # Setup ZMQ connections
        await client.setup()
        
        # Send a render request
        request_id = await client.send_render_request(
            prompt=test_prompt,
            era=test_era,
            biome=test_biome
        )
        
        # Listen for messages
        messages = await client.listen_for_messages(request_id)
        
        # Analyze results
        if not messages:
            logger.error("❌ TEST FAILED: No messages received")
            return False
        
        # Check if we received a response for our request
        response_received = any(
            msg.get("type") == MessageType.IMAGE_READY and 
            msg.get("request_id") == request_id 
            for msg in messages
        )
        
        if response_received:
            logger.info("✅ TEST PASSED: Received ImageReady response for our request")
            return True
        else:
            logger.info("❌ TEST FAILED: Did not receive ImageReady for our request")
            return False
        
    finally:
        # Clean up
        await client.teardown()


def main():
    """Main entry point for the script."""
    try:
        result = asyncio.run(run_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # uv run -m services.image_server.tests.test_zmq_messaging
    main()

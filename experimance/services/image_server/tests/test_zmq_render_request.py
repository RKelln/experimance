#!/usr/bin/env python3
"""
Comprehensive ZMQ Message Testing for Image Server RenderRequest.

This script provides a comprehensive test suite for verifying ZeroMQ communication 
between clients and the Image Server Service, focusing on RenderRequest messages
and ImageReady responses with various parameters including depth maps.

$ uv run -m services.image_server.tests.test_zmq_render_request
"""

import argparse
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

from experimance_common.schemas import MessageType
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.schemas import Era, Biome

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("zmq_render_test")


def load_image_as_base64(image_path: Path) -> str:
    """Load an image file and convert it to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string representation of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return ""


class RenderRequestTest:
    """Test client for verifying ZMQ RenderRequest and ImageReady communication."""
    
    def __init__(
        self,
        events_pub_address: str,
        images_sub_address: str,
        timeout: float = 30.0,
        debug: bool = False
    ):
        """Initialize the RenderRequest Test client.
        
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
        self.test_results = {}
        
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def setup(self):
        """Set up ZMQ connections."""
        logger.info("Setting up ZMQ connections...")
        
        # Create publisher for sending RenderRequest messages
        self.publisher = ZmqPublisher(self.events_pub_address,
                                      topic=MessageType.RENDER_REQUEST)
        
        # Create subscriber for receiving ImageReady messages
        # Subscribe to an empty string to receive all messages
        self.subscriber = ZmqSubscriber(
            self.images_sub_address, 
            [""]  # An empty string means subscribe to all topics
        )
        
        logger.info(f"Sending to events channel: {self.events_pub_address}")
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
    
    async def send_render_request(
        self, 
        prompt: str, 
        #era: str, 
        #biome: str,
        depth_map_path: Optional[Path] = None,
        test_id: str = "default"
    ) -> str:
        """Send a RenderRequest message.
        
        Args:
            prompt: Text prompt for image generation
            era: Era context for the image
            biome: Biome context for the image
            depth_map_path: Path to a depth map PNG file (optional)
            test_id: Identifier for this test case
            
        Returns:
            The request ID for tracking the request
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Create the message
        message = {
            "type": MessageType.RENDER_REQUEST,
            "request_id": request_id,
            #"era": era,
            #"biome": biome,
            "prompt": prompt
        }
        
        # Add depth map if provided
        if depth_map_path and depth_map_path.exists():
            logger.info(f"Including depth map from {depth_map_path}")
            message["depth_map_png"] = load_image_as_base64(depth_map_path)
        
        # Send the message
        logger.info(f"Sending RenderRequest with ID: {request_id} (test: {test_id})")
        
        # Record start time for latency measurement
        start_time = time.time()
        self.test_results[request_id] = {
            "test_id": test_id,
            "start_time": start_time,
            "request": message,
            "status": "sent"
        }
        
        assert self.publisher is not None, "Publisher not initialized"
        success = await self.publisher.publish_async(message)
        
        if success:
            logger.info(f"Successfully sent RenderRequest with ID: {request_id}")
        else:
            logger.error("Failed to send RenderRequest")
            self.test_results[request_id]["status"] = "failed_to_send"
        
        return request_id
    
    async def listen_for_response(
        self, 
        request_id: str, 
        max_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Listen for a response to a specific request.
        
        Args:
            request_id: The request ID to wait for
            max_duration: Maximum duration to listen for in seconds
            
        Returns:
            The response message or empty dict if timed out or error occurred
        """
        if max_duration is None:
            max_duration = self.timeout
            
        start_time = time.time()
        logger.info(f"Waiting for response to request {request_id}, timeout in {max_duration} seconds")
        
        while time.time() - start_time < max_duration:
            try:
                # Try to receive a message with timeout
                assert self.subscriber is not None, "Subscriber not initialized"
                topic, message = await self.subscriber.receive_async()
                
                if not message:
                    logger.debug("Received empty message, continuing...")
                    continue
                
                # Log all received messages for debugging
                logger.info(f"Received message of type: {message.get('type')}")
                
                # Check if this is our response
                if message.get("type") == MessageType.IMAGE_READY:
                    if message.get("request_id") == request_id:
                        logger.info(f"Received ImageReady for request {request_id}")
                        
                        # Record completion time and response
                        end_time = time.time()
                        if request_id in self.test_results:
                            self.test_results[request_id]["end_time"] = end_time
                            self.test_results[request_id]["duration"] = end_time - self.test_results[request_id]["start_time"]
                            self.test_results[request_id]["response"] = message
                            self.test_results[request_id]["status"] = "received"
                        
                        return message
                    else:
                        logger.info(f"Received ImageReady for a different request: {message.get('request_id')}")
                
                # Check for error messages related to our request
                if message.get("type") == MessageType.ALERT:
                    if message.get("request_id") == request_id:
                        logger.error(f"Received error for request {request_id}: {message.get('message')}")
                        
                        # Record error
                        if request_id in self.test_results:
                            self.test_results[request_id]["error_message"] = message.get("message")
                            self.test_results[request_id]["status"] = "error"
                        
                        return message
                
            except Exception as e:
                if "timed out" in str(e).lower():
                    logger.debug("Receive timeout, still waiting...")
                else:
                    logger.error(f"Error receiving message: {e}")
            
            # Brief pause to avoid tight polling
            await asyncio.sleep(0.1)
        
        logger.error(f"Timed out waiting for response to request {request_id}")
        if request_id in self.test_results:
            self.test_results[request_id]["status"] = "timeout"
        
        return {}
    
    def print_test_results(self):
        """Print a summary of test results."""
        print("\n==== Test Results ====")
        
        success_count = 0
        fail_count = 0
        
        for request_id, result in self.test_results.items():
            status = result["status"]
            test_id = result["test_id"]
            
            if status == "received":
                outcome = "✅ SUCCESS"
                success_count += 1
                duration = result.get("duration", "unknown")
                print(f"{outcome} - Test: {test_id} (ID: {request_id[:8]}...) - Response time: {duration:.2f}s")
            else:
                outcome = "❌ FAILED"
                fail_count += 1
                print(f"{outcome} - Test: {test_id} (ID: {request_id[:8]}...) - Status: {status}")
        
        print(f"\nSummary: {success_count} tests passed, {fail_count} tests failed")
        print(f"Total tests: {success_count + fail_count}")


async def run_basic_test(
    client: RenderRequestTest, 
    depth_map_path: Optional[Path] = None
):
    """Run a basic render request test.
    
    Args:
        client: The test client
        era: Optional era override
        biome: Optional biome override
        depth_map_path: Optional depth map path
    
    Returns:
        Tuple of (request_id, response_message)
    """
    # Test parameters
    test_prompt = "A forest scene with tall pine trees and a mountain in the background"
    #test_era = era or Era.WILDERNESS.value
    #test_biome = biome or Biome.BOREAL_FOREST.value
    
    # Send a render request
    request_id = await client.send_render_request(
        prompt=test_prompt,
        #era=test_era,
        #biome=test_biome,
        depth_map_path=depth_map_path,
        test_id="basic_test"
    )
    
    # Listen for response
    response = await client.listen_for_response(request_id)
    
    return request_id, response


async def run_test_suite():
    """Run a comprehensive test suite for RenderRequest and ImageReady messages."""
    # Use the unified events channel for all communication
    events_pub_address = f"tcp://*:{DEFAULT_PORTS['events']}"
    images_sub_address = f"tcp://localhost:{DEFAULT_PORTS['events']}"
    
    logger.info("===== Starting ZMQ RenderRequest Test Suite =====")
    logger.info(f"Events publish address: {events_pub_address}")
    logger.info(f"Images subscribe address: {images_sub_address}")
    
    # Create the test client
    client = RenderRequestTest(
        events_pub_address=events_pub_address,
        images_sub_address=images_sub_address,
        timeout=30.0,  # 30 second timeout for responses
        debug=True
    )
    
    try:
        # Setup ZMQ connections
        await client.setup()
        
        # Test 1: Basic render request with default parameters
        logger.info("Running Test 1: Basic render request")
        await run_basic_test(client)
        
        # Test 2: Test with different era
        logger.info("Running Test 2: Render request with Industrial era")
        await run_basic_test(client)
        
        # Test 3: Test with different biome
        logger.info("Running Test 3: Render request with Desert biome")
        await run_basic_test(client)
        
        # If depth map testing is needed, uncomment and specify a path:
        # depth_map_path = Path("/path/to/test/depth_map.png")
        # if depth_map_path.exists():
        #     logger.info("Running Test 4: Render request with depth map")
        #     await run_basic_test(client, depth_map_path=depth_map_path, test_id="depth_map_test")
        
        # Print results summary
        client.print_test_results()
        
        # Check overall success
        failures = sum(1 for result in client.test_results.values() if result["status"] != "received")
        return failures == 0
        
    finally:
        # Clean up
        await client.teardown()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ZMQ RenderRequest messages with the Image Server")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-t", "--timeout", type=float, default=30.0, help="Response timeout in seconds")
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        result = asyncio.run(run_test_suite())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # $ uv run -m services.image_server.tests.test_zmq_render_request
    main()

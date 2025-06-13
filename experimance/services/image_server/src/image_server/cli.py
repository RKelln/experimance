#!/usr/bin/env python3
"""
CLI utility for testing the Image Server Service by sending RenderRequest messages.

This utility allows users to send test image generation requests to the Image Server Service
using ZeroMQ. It supports selecting predefined prompts or entering custom prompts,
specifying era and biome parameters, and optionally including a depth map image.

$ uv run -m image_server.cli
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
from typing import Dict, Any, Optional, List

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
logger = logging.getLogger("image_server_cli")

# Set debug level for more verbose output if needed
# logging.getLogger().setLevel(logging.DEBUG)


# Define some sample prompts
SAMPLE_PROMPTS = {
    "default": "colorful satellite image in the style of experimance, (dense urban:1.2) dramatic landscape, buildings, farmland, (industrial:1.1), (rivers, lakes:1.1), busy highways, hills, vibrant hyper detailed photorealistic maximum detail, 32k, high resolution ultra HD",
    "forest_scene": "Dense forest with ancient trees, sunlight filtering through the canopy",
    "mountain_view": "Majestic snow-capped mountains with a crystal clear lake in the foreground",
    "desert_landscape": "Vast desert landscape with striking sand dunes and a small oasis",
    "industrial_area": "Industrial zone with factories, smokestacks and railway systems",
    "futuristic_metropolis": "Futuristic metropolis with flying vehicles and holographic billboards",
    "ancient_ruins": "Ancient stone ruins covered in vines and vegetation, partially submerged in water",
    "underwater_scene": "Vibrant coral reef with colorful fish and underwater vegetation",
}


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


class ImageServerClient:
    """Client for interacting with the Image Server Service."""
    
    def __init__(
        self,
        events_pub_address: str,
        images_sub_address: str,
        timeout: int = 300,
        debug: bool = False
    ):
        """Initialize the Image Server Client.
        
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
            logging.getLogger("image_server_cli").setLevel(logging.DEBUG)
        
    async def start(self):
        """Start the ZMQ connections."""
        # Create publisher for sending RenderRequest messages
        self.publisher = ZmqPublisher(self.events_pub_address, MessageType.RENDER_REQUEST)
        
        # Create subscriber for receiving ImageReady messages
        # Subscribe to an empty string to receive all messages
        self.subscriber = ZmqSubscriber(
            self.images_sub_address, 
            [MessageType.IMAGE_READY] # An empty string means subscribe to all topics
        )
        
        logger.info(f"Connected to events channel: {self.events_pub_address}")
        logger.info(f"Listening for images on: {self.images_sub_address}")
        
        # Add a small delay to ensure connections are established
        # This helps with the ZMQ slow joiner syndrome
        logger.info("Waiting for ZMQ connections to establish...")
        await asyncio.sleep(1.0)
        
    async def stop(self):
        """Stop ZMQ connections and clean up resources."""
        if self.publisher:
            self.publisher.close()
        if self.subscriber:
            self.subscriber.close()
        logger.info("Client stopped")
    
    async def send_render_request(
        self,
        prompt: str,
        # era: str,
        # biome: str,
        depth_map_path: Optional[Path] = None
    ) -> str:
        """Send a RenderRequest message to the Image Server Service.
        
        Args:
            prompt: Text prompt for image generation
            era: Era context for the image
            biome: Biome context for the image
            depth_map_path: Path to optional depth map image file
            
        Returns:
            The request ID for tracking the request
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Create the message
        message = {
            "type": MessageType.RENDER_REQUEST,
            "request_id": request_id,
            # "era": era,
            # "biome": biome,
            "prompt": prompt
        }
        
        # Add depth map if provided
        if depth_map_path and depth_map_path.exists():
            logger.info(f"Including depth map from {depth_map_path}")
            message["depth_map_png"] = load_image_as_base64(depth_map_path)
        
        # Log the full message for debugging
        logger.debug(f"Sending message: {json.dumps(message)[:200]}...")
        
        # Send the message
        assert self.publisher is not None, "Publisher not initialized"
        success = await self.publisher.publish_async(message)
        
        if success:
            logger.info(f"Sent RenderRequest with ID: {request_id}")
        else:
            logger.error("Failed to send RenderRequest")
        
        return request_id
    
    async def wait_for_response(self, request_id: str) -> Dict[str, Any]:
        """Wait for a response to a specific request.
        
        Args:
            request_id: The request ID to wait for
            
        Returns:
            The response message or None if timed out or error occurred
        """
        start_time = time.time()
        logger.info(f"Waiting for response to request {request_id}, timeout in {self.timeout} seconds")
        
        while time.time() - start_time < self.timeout:
            try:
                # Try to receive a message with timeout
                assert self.subscriber is not None, "Subscriber not initialized"    
                topic, message = await self.subscriber.receive_async()
                
                if not message:
                    logger.debug("Received empty message, continuing...")
                    continue
                
                # Log all received messages for debugging
                logger.info(f"Received message: {json.dumps(message, indent=2)[:200]}...")
                
                # Check if this is our response - noting that the request_id might be missing in some messages
                if message.get("type") == MessageType.IMAGE_READY:
                    if message.get("request_id") == request_id:
                        logger.info(f"Received ImageReady for request {request_id}")
                        return message
                    else:
                        logger.info(f"Received ImageReady for a different request: {message.get('request_id')}")
                
                # Check for error messages related to our request
                if message.get("type") == MessageType.ALERT:
                    if message.get("request_id") == request_id:
                        logger.error(f"Received error for request {request_id}: {message.get('message')}")
                        return message
                    else:
                        logger.info(f"Received Alert for a different request: {message.get('request_id')}")
                
                # For any other message type
                logger.info(f"Received message of type: {message.get('type')}")
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                if "timed out" in str(e).lower():
                    logger.info("Still waiting for response... (timeout on receive is normal)")
                
            # Brief pause to avoid tight polling
            await asyncio.sleep(0.1)
        
        logger.error(f"Timed out waiting for response to request {request_id}")
        return {}


async def interactive_mode(debug: bool = False):
    """Run the client in interactive mode with a menu-based interface."""
    # Default settings - connecting to server's binding addresses
    # For events, we connect to the unified events channel
    events_pub_address = f"tcp://localhost:{DEFAULT_PORTS['events']}"
    # For images, we connect to the unified events channel
    images_sub_address = f"tcp://localhost:{DEFAULT_PORTS['events']}"
    
    print("\n=== Image Server Test Client ===\n")
    print("ZMQ Configuration:")
    print(f"  Events publish address: {events_pub_address}")
    print(f"  Images subscribe address: {images_sub_address}")
    
    # Allow custom ZMQ addresses
    custom_addresses = input("\nUse custom ZMQ addresses? (y/N): ").lower() == 'y'
    if custom_addresses:
        events_pub_address = input(f"Events publish address [{events_pub_address}]: ") or events_pub_address
        images_sub_address = input(f"Images subscribe address [{images_sub_address}]: ") or images_sub_address
    
    # Create and start client
    client = ImageServerClient(events_pub_address, images_sub_address)
    await client.start()
    
    try:
        while True:
            print("\n=== Image Generation Menu ===")
            
            # Prompt selection
            print("\nSelect a prompt:")
            print("  0. Enter custom prompt")
            prompt_options = list(SAMPLE_PROMPTS.keys())
            for i, name in enumerate(prompt_options):
                print(f"  {i+1}. {name}: {SAMPLE_PROMPTS[name][:50]}...")
            
            prompt_choice = int(input(f"Choose prompt (0-{len(prompt_options)}, default=0): ") or "0")
            if prompt_choice == 0:
                selected_prompt = input("Enter your custom prompt: ")
            else:
                if 1 <= prompt_choice <= len(prompt_options):
                    selected_prompt = SAMPLE_PROMPTS[prompt_options[prompt_choice - 1]]
                else:
                    selected_prompt = "A beautiful landscape"
            
            # Depth map selection
            use_depth_map = input("\nUse a depth map? (y/N): ").lower() == 'y'
            depth_map_path = None
            if use_depth_map:
                default_depth_map = "services/imagge_server/images/mocks/depth_map.png"
                depth_map_path_str = input(f"Enter path to depth map PNG file: ({default_depth_map})")
                depth_map_path = Path(depth_map_path_str).expanduser().absolute()
                if not depth_map_path.exists() or not depth_map_path.is_file():
                    print(f"Warning: Depth map file not found at {depth_map_path}")
                    depth_map_path = None
            
            # Show summary and confirm
            print("\n=== Request Summary ===")
            # print(f"Era: {selected_era}")
            # print(f"Biome: {selected_biome}")
            print(f"Prompt: {selected_prompt}")
            print(f"Depth Map: {'Yes - ' + str(depth_map_path) if depth_map_path else 'No'}")
            
            confirm = input("\nSend this request? (Y/n): ").lower() != 'n'
            if not confirm:
                print("Request canceled")
                continue
            
            # Send request and wait for response
            request_id = await client.send_render_request(
                prompt=selected_prompt,
                # era=selected_era,
                # biome=selected_biome,
                depth_map_path=depth_map_path
            )
            
            print(f"\nRequest {request_id} sent. Waiting for response...")
            response = await client.wait_for_response(request_id)
            
            if response:
                if response.get("type") == MessageType.IMAGE_READY:
                    print("\n=== Image Ready ===")
                    print(f"Image ID: {response.get('image_id')}")
                    print(f"Image URI: {response.get('uri')}")
                    
                    # If it's a file URI, check if the file exists
                    uri = response.get('uri', '')
                    if uri.startswith('file://'):
                        file_path = Path(uri.replace('file://', ''))
                        if file_path.exists():
                            print(f"File exists: Yes ({file_path.stat().st_size} bytes)")
                        else:
                            print("File exists: No")
                else:
                    print("\n=== Response ===")
                    print(json.dumps(response, indent=2))
            else:
                print("\nNo response received within the timeout period")
            
            # Ask to continue
            if input("\nGenerate another image? (Y/n): ").lower() == 'n':
                break
    
    finally:
        await client.stop()


async def command_line_mode(args):
    """Run the client in command line mode with arguments.
    
    Args:
        args: Command line arguments
    """
    client = ImageServerClient(
        args.events_address,
        args.images_address,
        timeout=args.timeout,
        debug=args.debug
    )
    
    await client.start()
    
    try:
        # Send the render request
        depth_map_path = Path(args.depth_map) if args.depth_map else None
        
        request_id = await client.send_render_request(
            prompt=args.prompt,
            # era=args.era,
            # biome=args.biome,
            depth_map_path=depth_map_path
        )
        
        if args.no_wait:
            print(f"Request {request_id} sent. Not waiting for response.")
            return
        
        # Wait for the response
        print(f"Request {request_id} sent. Waiting for response...")
        response = await client.wait_for_response(request_id)
        
        if response:
            # Pretty print the response
            print("\nResponse received:")
            print(json.dumps(response, indent=2))
            
            # If it's an image response, show file info
            if response.get("type") == MessageType.IMAGE_READY:
                uri = response.get('uri', '')
                if uri.startswith('file://'):
                    file_path = Path(uri.replace('file://', ''))
                    if file_path.exists():
                        print(f"\nImage saved to: {file_path}")
                        print(f"File size: {file_path.stat().st_size} bytes")
        else:
            print("No response received within the timeout period")
    
    finally:
        await client.stop()


def main():
    """Main entry point for the CLI utility."""
    parser = argparse.ArgumentParser(description="Test client for the Image Server Service")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with a menu interface"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--era", "-e",
        type=str,
        choices=[e.value for e in Era],
        default="wilderness",
        help="Era context for the image"
    )
    parser.add_argument(
        "--biome", "-b",
        type=str,
        choices=[b.value for b in Biome],
        default="forest",
        help="Biome context for the image"
    )
    parser.add_argument(
        "--depth-map", "--depth_map", "-d",
        type=str,
        help="Path to depth map PNG file"
    )
    parser.add_argument(
        "--events-address", "--events_address",
        type=str,
        default=f"tcp://localhost:{DEFAULT_PORTS['events']}",
        help=f"ZMQ address for publishing events (default: tcp://localhost:{DEFAULT_PORTS['events']})"
    )
    parser.add_argument(
        "--images-address", "--images_address",
        type=str,
        default=f"tcp://localhost:{DEFAULT_PORTS['events']}",
        help=f"ZMQ address for subscribing to images (default: tcp://localhost:{DEFAULT_PORTS['events']})"
    )
    parser.add_argument(
        "--debug", "-D",
        action="store_true",
        help="Enable debug logging for more detailed output"
    )
    parser.add_argument(
        "--no-wait", "--no_wait",
        action="store_true",
        help="Don't wait for a response after sending the request"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=300,
        help="Timeout in seconds for waiting for a response (default: 300)"
    )
    parser.add_argument(
        "--list-prompts", "--list_prompts", "-l",
        action="store_true",
        help="List available sample prompts and exit"
    )
    parser.add_argument(
        "--sample-prompt", "--sample_prompt", "-s",
        type=str,
        help="Use a sample prompt by name (use --list-prompts to see available options)"
    )
    
    args = parser.parse_args()
    
    # List sample prompts if requested
    if args.list_prompts:
        print("Available sample prompts:")
        for name, prompt in SAMPLE_PROMPTS.items():
            print(f"  {name}: {prompt}")
        return
    
    # Use a sample prompt if specified
    if args.sample_prompt:
        if args.sample_prompt in SAMPLE_PROMPTS:
            args.prompt = SAMPLE_PROMPTS[args.sample_prompt]
            print(f"Using sample prompt: {args.prompt}")
        else:
            print(f"Error: Sample prompt '{args.sample_prompt}' not found.")
            print("Use --list-prompts to see available options.")
            return 1

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("image_server_cli").setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Check for interactive mode or required parameters
    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.prompt or args.sample_prompt:
        asyncio.run(command_line_mode(args))
    else:
        print("Error: Either --interactive mode or --prompt must be specified.")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    # $ uv run -m image_server.cli -i
    sys.exit(main())

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

from experimance_common.schemas import MessageType
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.schemas import Era, Biome
from experimance_common.zmq.components import PushComponent, PullComponent
from experimance_common.zmq.config import PushConfig, PullConfig

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
    "default": "top down satellite aerial photography, mountain, pre-industrial landscape, (wilderness), hills, valleys, highly detailed, Rocky Mountains",
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
    """Client for interacting with the Image Server Service using Push/Pull."""

    def __init__(
        self,
        push_address: str,
        pull_address: str,
        timeout: int = 300,
        debug: bool = False
    ):
        self.push_address = push_address
        self.pull_address = pull_address
        self.timeout = timeout
        self.debug = debug
        self.push_component: Optional[PushComponent] = None
        self.pull_component: Optional[PullComponent] = None
        self._response_queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        """Start the Push/Pull components for ZMQ communication."""
        # Extract port from address string (format: tcp://host:port)
        def extract_port(addr: str) -> int:
            try:
                return int(addr.split(":")[-1])
            except Exception:
                raise ValueError(f"Could not extract port from address: {addr}")

        push_port = extract_port(self.push_address)
        pull_port = extract_port(self.pull_address)
        push_config = PushConfig(address=self.push_address.rsplit(":", 1)[0], port=push_port)
        pull_config = PullConfig(address=self.pull_address.rsplit(":", 1)[0], port=pull_port)
        self.push_component = PushComponent(push_config)
        self.pull_component = PullComponent(pull_config)
        self.pull_component.set_work_handler(self._on_image_ready)
        await self.push_component.start()
        await self.pull_component.start()
        logger.info(f"Connected to image_server push address: {self.push_address}")
        logger.info(f"Listening for image ready on: {self.pull_address}")
        await asyncio.sleep(1.0)

    async def stop(self):
        if self.push_component:
            await self.push_component.stop()
        if self.pull_component:
            await self.pull_component.stop()
        logger.info("Client stopped")

    async def send_render_request(
        self,
        prompt: str,
        depth_map_path: Optional[Path] = None
    ) -> str:
        request_id = str(uuid.uuid4())
        message = {
            "type": MessageType.RENDER_REQUEST,
            "request_id": request_id,
            "prompt": prompt
        }
        if depth_map_path and depth_map_path.exists():
            logger.info(f"Including depth map from {depth_map_path}")
            message["depth_map_png"] = load_image_as_base64(depth_map_path)
        logger.debug(f"Sending message: {json.dumps(message)[:200]}...")
        assert self.push_component is not None, "PushComponent not initialized"
        await self.push_component.push(message)
        logger.info(f"Sent RenderRequest with ID: {request_id}")
        return request_id

    async def wait_for_response(self, request_id: str) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Waiting for response to request {request_id}, timeout in {self.timeout} seconds")
        while time.time() - start_time < self.timeout:
            try:
                message = await asyncio.wait_for(self._response_queue.get(), timeout=1.0)
                if message.get("request_id") == request_id:
                    logger.info(f"Received response for request {request_id}")
                    return message
            except asyncio.TimeoutError:
                continue
        logger.error(f"Timed out waiting for response to request {request_id}")
        return {}

    async def _on_image_ready(self, message):
        logger.info(f"Received IMAGE_READY: {json.dumps(message)[:200]}...")
        await self._response_queue.put(message)


async def interactive_mode(debug: bool = False):
    """Run the client in interactive mode with a menu-based interface."""
    # Default settings - use DEFAULT_PORTS for unified events channel
    push_address = f"tcp://localhost:{DEFAULT_PORTS['image_requests']}"
    pull_address = f"tcp://*:{DEFAULT_PORTS['image_results']}"

    print("\n=== Image Server Test Client ===\n")
    print("ZMQ Configuration:")
    print(f"  Push address (to send requests): {push_address}")
    print(f"  Pull address (to receive images): {pull_address}")

    # Allow custom ZMQ addresses
    custom_addresses = input("\nUse custom ZMQ addresses? (y/N): ").lower() == 'y'
    if custom_addresses:
        push_address = input(f"Push address [{push_address}]: ") or push_address
        pull_address = input(f"Pull address [{pull_address}]: ") or pull_address

    # Create and start client
    client = ImageServerClient(push_address, pull_address)
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
                default_depth_map = "services/image_server/images/mocks/depth_map.png"
                depth_map_path_str = input(f"Enter path to depth map PNG file: ({default_depth_map})") or default_depth_map
                depth_map_path = Path(depth_map_path_str).expanduser().absolute()
                if not depth_map_path.exists() or not depth_map_path.is_file():
                    print(f"Warning: Depth map file not found at {depth_map_path}")
                    depth_map_path = None

            # Show summary and confirm
            print("\n=== Request Summary ===")
            print(f"Prompt: {selected_prompt}")
            print(f"Depth Map: {'Yes - ' + str(depth_map_path) if depth_map_path else 'No'}")

            confirm = input("\nSend this request? (Y/n): ").lower() != 'n'
            if not confirm:
                print("Request canceled")
                continue

            # Send request and wait for response
            request_id = await client.send_render_request(
                prompt=selected_prompt,
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
    """Run the client in command line mode with arguments using Push/Pull."""
    # Use push/pull addresses for image server
    push_address = args.events_address if hasattr(args, 'events_address') else f"tcp://localhost:{DEFAULT_PORTS['image_server_push']}"
    pull_address = args.images_address if hasattr(args, 'images_address') else f"tcp://localhost:{DEFAULT_PORTS['image_server_pull']}"
    client = ImageServerClient(
        push_address,
        pull_address,
        timeout=args.timeout,
        debug=args.debug
    )

    await client.start()

    try:
        # Send the render request
        depth_map_path = Path(args.depth_map) if args.depth_map else None

        request_id = await client.send_render_request(
            prompt=args.prompt,
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
        "--request-address", "--requests_address",
        type=str,
        default=f"tcp://localhost:{DEFAULT_PORTS['image_requests']}",
        help=f"ZMQ address for publishing events (default: tcp://localhost:{DEFAULT_PORTS['image_requests']})"
    )
    parser.add_argument(
        "--result-address", "--results_address",
        type=str,
        default=f"tcp://*:{DEFAULT_PORTS['image_results']}",
        help=f"ZMQ address for subscribing to images (default: tcp://localhost:{DEFAULT_PORTS['image_results']})"
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

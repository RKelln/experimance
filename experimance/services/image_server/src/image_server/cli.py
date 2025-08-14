#!/usr/bin/env python3
"""
CLI utility for testing the Image Server Service by sending RenderRequest messages.

This utility allows users to send test        # Create proper RenderRequest object like core service does
        request = RenderRequest(
            request_id=request_id,
            era=era,  # Pass string directly, let RenderRequest handle conversion
            biome=biome,  # Pass string directly, let RenderRequest handle conversion
            prompt=prompt,
            depth_map=depth_map_source
        )eneration requests to the Image Server Service
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

from experimance_common.constants import MOCK_IMAGES_DIR
from experimance_common.schemas import MessageType, ImageSource
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq.components import PushComponent, PullComponent
from experimance_common.zmq.config import ControllerPushConfig, ControllerPullConfig
from experimance_common.zmq.zmq_utils import prepare_image_source, IMAGE_TRANSPORT_MODES

PROJECT_ENV = os.getenv("PROJECT_ENV", "experimance").lower()
# Import project-specific schemas (Era, Biome, RenderRequest)
if PROJECT_ENV == "experimance":
    from experimance_common.schemas import Era, Biome
from experimance_common.schemas import RenderRequest

# Try to import prompt generator for enhanced prompt generation
try:
    from experimance_core.prompt_generator import PromptGenerator, RandomStrategy
    PROMPT_GENERATOR_AVAILABLE = True
except ImportError:
    PROMPT_GENERATOR_AVAILABLE = False

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
        push_config = ControllerPushConfig(port=push_port)
        pull_config = ControllerPullConfig(port=pull_port)
        self.push_component = PushComponent(push_config)
        self.pull_component = PullComponent(pull_config)
        self.pull_component.set_work_handler(self._on_image_ready)
        await self.push_component.start()
        await self.pull_component.start()
        logger.info(f"CLI controller bound to push requests on: {self.push_address}")
        logger.info(f"CLI controller bound to pull results on: {self.pull_address}")
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
        depth_map_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        era: Optional[str] = None,
        biome: Optional[str] = None
    ) -> str:
        request_id = str(uuid.uuid4())
        
        # Handle depth map similar to core service using prepare_image_source
        depth_map_source = None
        if depth_map_path and depth_map_path.exists():
            logger.info(f"Including depth map from {depth_map_path}")
            # Use prepare_image_source like the core service does
            depth_map_source = prepare_image_source(
                image_data=depth_map_path,  # Can pass Path directly
                transport_mode=IMAGE_TRANSPORT_MODES['BASE64'],
                request_id=request_id,
            )
        
        # Handle source image for image-to-image generation
        image_source = None
        if image_path and image_path.exists():
            logger.info(f"Including source image for image-to-image from {image_path}")
            image_source = prepare_image_source(
                image_data=image_path,  # Can pass Path directly
                transport_mode=IMAGE_TRANSPORT_MODES['BASE64'],
                request_id=request_id,
            )

        # Create proper RenderRequest object like core service does
        request = RenderRequest(
            request_id=request_id,
            prompt=prompt,
            depth_map=depth_map_source,
            image=image_source
        )
        if PROJECT_ENV == "experimance":
            request.era = Era(era) if era else None
            request.biome = Biome(biome) if biome else None

        logger.debug(f"Sending RenderRequest: {request.request_id}")
        assert self.push_component is not None, "PushComponent not initialized"
        await self.push_component.push(request)
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
        # Convert message to dict if it's a Pydantic object
        if hasattr(message, 'model_dump'):
            message_dict = message.model_dump()
        elif hasattr(message, 'dict'):
            message_dict = message.dict()
        else:
            message_dict = message
        
        logger.info(f"Received IMAGE_READY: {json.dumps(message_dict)[:200]}...")
        await self._response_queue.put(message_dict)


async def interactive_mode(debug: bool = False):
    """Run the client in interactive mode with a menu-based interface."""
    # Default settings - CLI acts as controller, so it binds to ports
    push_address = f"tcp://*:{DEFAULT_PORTS['image_requests']}"
    pull_address = f"tcp://*:{DEFAULT_PORTS['image_results']}"

    print("\n=== Image Server Test Client ===\n")
    print("ZMQ Configuration (CLI acts as Controller):")
    print(f"  Push address (bind to send requests): {push_address}")
    print(f"  Pull address (bind to receive results): {pull_address}")

    # Allow custom ZMQ addresses
    custom_addresses = input("\nUse custom ZMQ addresses? (y/N): ").lower() == 'y'
    if custom_addresses:
        push_address = input(f"Push address [{push_address}]: ") or push_address
        pull_address = input(f"Pull address [{pull_address}]: ") or pull_address

    # Initialize prompt generator if available
    prompt_gen = None
    available_eras = []
    available_biomes = []
    
    if PROMPT_GENERATOR_AVAILABLE:
        try:
            # Look for data directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent.parent.parent  # Navigate up to project root
            data_path = project_root / "data"
            
            if data_path.exists() and (data_path / "locations.json").exists() and (data_path / "anthropocene.json").exists():
                prompt_gen = PromptGenerator(
                    data_path=data_path,
                    strategy=RandomStrategy.SHUFFLE
                )
                available_eras = prompt_gen.get_available_eras()
                available_biomes = prompt_gen.get_available_biomes()
                print(f"✅ Prompt generator loaded with {len(available_eras)} eras and {len(available_biomes)} biomes")
            else:
                print(f"⚠️  Data files not found at {data_path}, using fallback prompts")
        except Exception as e:
            print(f"⚠️  Could not initialize prompt generator: {e}, using fallback prompts")
    else:
        print("⚠️  Prompt generator not available, using fallback prompts")

    # Create and start client
    client = ImageServerClient(push_address, pull_address)
    await client.start()

    try:
        while True:
            print("\n=== Image Generation Menu ===")

            # Prompt selection
            print("\nSelect a prompt source:")
            print("  1. Select from predefined prompts")
            print("  2. Enter custom prompt")
            if PROJECT_ENV == "experimance":
                print("  3. Generate from era/biome (recommended)" if prompt_gen else "  3. Generate from era/biome (unavailable)")

            if prompt_gen:
                prompt_choice = input("Choose option (1-3, default=3): ") or "3"
            else:
                prompt_choice = input("Choose option (1-2, default=1): ") or "1"

            selected_prompt = ""
            selected_era = "wilderness"
            selected_biome = "temperate_forest"

            if prompt_choice == "3" and prompt_gen:
                # Era/biome based prompt generation
                print(f"\nEra selection (available: {len(available_eras)}):")
                for i, era in enumerate(available_eras):
                    print(f"  {i+1}. {era.value}")
                
                era_choice = input(f"Choose era (1-{len(available_eras)}, default=1): ") or "1"
                try:
                    selected_era_enum = available_eras[int(era_choice) - 1]
                    selected_era = selected_era_enum.value
                except (ValueError, IndexError):
                    selected_era_enum = available_eras[0]
                    selected_era = selected_era_enum.value

                print(f"\nBiome selection (available: {len(available_biomes)}):")
                for i, biome in enumerate(available_biomes):
                    print(f"  {i+1}. {biome.value}")
                
                biome_choice = input(f"Choose biome (1-{len(available_biomes)}, default=1): ") or "1"
                try:
                    selected_biome_enum = available_biomes[int(biome_choice) - 1]
                    selected_biome = selected_biome_enum.value
                except (ValueError, IndexError):
                    selected_biome_enum = available_biomes[0]
                    selected_biome = selected_biome_enum.value

                # Generate prompt
                try:
                    positive_prompt, negative_prompt = prompt_gen.generate_prompt(selected_era_enum, selected_biome_enum)
                    print(f"\n🎨 Generated prompt for {selected_era} + {selected_biome}:")
                    print(f"Positive: {positive_prompt}")
                    if negative_prompt:
                        print(f"Negative: {negative_prompt}")
                    
                    # Allow editing
                    edit_prompt = input("\nEdit this prompt? (y/N): ").lower() == 'y'
                    if edit_prompt:
                        print("\nEnter your modified prompt (press Enter to keep current):")
                        custom_positive = input(f"Positive [{positive_prompt[:50]}...]: ")
                        if custom_positive.strip():
                            positive_prompt = custom_positive.strip()
                        
                        if negative_prompt:
                            custom_negative = input(f"Negative [{negative_prompt[:50]}...]: ")
                            if custom_negative.strip():
                                negative_prompt = custom_negative.strip()
                    
                    selected_prompt = positive_prompt
                    
                except Exception as e:
                    print(f"Error generating prompt: {e}")
                    selected_prompt = "A beautiful landscape"

            elif prompt_choice == "1":
                # Predefined prompts
                print("\nSelect a predefined prompt:")
                print("  0. Enter custom prompt")
                prompt_options = list(SAMPLE_PROMPTS.keys())
                for i, name in enumerate(prompt_options):
                    print(f"  {i+1}. {name}: {SAMPLE_PROMPTS[name][:50]}...")

                predefined_choice = int(input(f"Choose prompt (0-{len(prompt_options)}, default=0): ") or "0")
                if predefined_choice == 0:
                    selected_prompt = input("Enter your custom prompt: ")
                else:
                    if 1 <= predefined_choice <= len(prompt_options):
                        selected_prompt = SAMPLE_PROMPTS[prompt_options[predefined_choice - 1]]
                    else:
                        selected_prompt = "A beautiful landscape"

            else:  # prompt_choice == "3" or fallback
                # Custom prompt
                selected_prompt = input("Enter your custom prompt: ")

            if PROJECT_ENV == "experimance":
                # Era and biome selection (if not already set from prompt generation)
                if prompt_choice != "3" or not prompt_gen:
                    print(f"\nEra context (default: wilderness):")
                    era_options = [e.value for e in Era]
                    for i, era in enumerate(era_options):
                        print(f"  {i+1}. {era}")
                    era_choice = input(f"Choose era (1-{len(era_options)}, default=1): ") or "1"
                    try:
                        selected_era = era_options[int(era_choice) - 1]
                    except (ValueError, IndexError):
                        selected_era = "wilderness"

                    print(f"\nBiome context (default: temperate_forest):")
                    biome_options = [b.value for b in Biome]
                    for i, biome in enumerate(biome_options):
                        print(f"  {i+1}. {biome}")
                    biome_choice = input(f"Choose biome (1-{len(biome_options)}, default=2): ") or "2"
                    try:
                        selected_biome = biome_options[int(biome_choice) - 1]
                    except (ValueError, IndexError):
                        selected_biome = "temperate_forest"

            # Source image selection (for image-to-image generation)
            use_source_image = input("\nUse a source image for image-to-image generation? (y/N): ").lower() == 'y'
            source_image_path = None
            if use_source_image:
                print("\nSource image options:")
                print("  1. Browse media/images/ directory")
                print("  2. Enter custom path")
                
                image_choice = input("Choose option (1-2, default=1): ") or "1"
                
                if image_choice == "1":
                    # Browse media/images directory
                    script_dir = Path(__file__).parent
                    project_root = script_dir.parent.parent.parent.parent  # Navigate up to project root
                    media_images_dir = project_root / "media" / "images"
                    
                    if media_images_dir.exists():
                        # Find image files in media/images and subdirectories
                        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
                        image_files = []
                        
                        for subdir in media_images_dir.iterdir():
                            if subdir.is_dir():
                                for img_file in subdir.rglob('*'):
                                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                                        # Store relative path from media/images for display
                                        rel_path = img_file.relative_to(media_images_dir)
                                        image_files.append((str(rel_path), img_file))
                        
                        if image_files:
                            print(f"\nFound {len(image_files)} images in media/images/:")
                            for i, (rel_path, full_path) in enumerate(image_files[:20]):  # Show first 20
                                size_kb = full_path.stat().st_size // 1024
                                print(f"  {i+1}. {rel_path} ({size_kb}KB)")
                            
                            if len(image_files) > 20:
                                print(f"  ... and {len(image_files) - 20} more")
                            
                            print("  0. Enter custom path")
                            
                            img_choice = input(f"Choose image (0-{min(len(image_files), 20)}, default=1): ") or "1"
                            try:
                                img_idx = int(img_choice)
                                if img_idx == 0:
                                    source_image_path = Path(input("Enter path to source image: ")).expanduser().absolute()
                                elif 1 <= img_idx <= len(image_files):
                                    source_image_path = image_files[img_idx - 1][1]
                                    print(f"Selected: {image_files[img_idx - 1][0]}")
                                else:
                                    source_image_path = image_files[0][1] if image_files else None
                            except (ValueError, IndexError):
                                source_image_path = image_files[0][1] if image_files else None
                        else:
                            print(f"No images found in {media_images_dir}")
                            source_image_path = Path(input("Enter path to source image: ")).expanduser().absolute()
                    else:
                        print(f"Media directory not found at {media_images_dir}")
                        source_image_path = Path(input("Enter path to source image: ")).expanduser().absolute()
                else:
                    # Custom path
                    source_image_path = Path(input("Enter path to source image: ")).expanduser().absolute()
                
                if source_image_path and not source_image_path.exists():
                    print(f"Warning: Source image file not found at {source_image_path}")
                    source_image_path = None

            # Depth map selection
            use_depth_map = input("\nUse a depth map? (y/N): ").lower() == 'y'
            depth_map_path = None
            if use_depth_map:
                default_depth_map = MOCK_IMAGES_DIR / "depth" / "mock_depth_map.png"
                depth_map_path_str = input(f"Enter path to depth map PNG file: ({default_depth_map})") or default_depth_map
                depth_map_path = Path(depth_map_path_str).expanduser().absolute()
                if not depth_map_path.exists() or not depth_map_path.is_file():
                    print(f"Warning: Depth map file not found at {depth_map_path}")
                    depth_map_path = None

            # Show summary and confirm
            print("\n=== Request Summary ===")
            print(f"Prompt: {selected_prompt}")
            if PROJECT_ENV == "experimance":
                print(f"Era: {selected_era}")
                print(f"Biome: {selected_biome}")
            print(f"Source Image: {'Yes - ' + str(source_image_path) if source_image_path else 'No'}")
            print(f"Depth Map: {'Yes - ' + str(depth_map_path) if depth_map_path else 'No'}")
            
            # Show generation mode
            if source_image_path and depth_map_path:
                print("Generation Mode: Image-to-image with ControlNet depth guidance")
            elif source_image_path:
                print("Generation Mode: Image-to-image")
            elif depth_map_path:
                print("Generation Mode: Text-to-image with ControlNet depth guidance")
            else:
                print("Generation Mode: Text-to-image")

            confirm = input("\nSend this request? (Y/n): ").lower() != 'n'
            if not confirm:
                print("Request canceled")
                continue

            # Send request and wait for response
            request = {
                "prompt": selected_prompt,
                "depth_map_path": depth_map_path,
                "image_path": source_image_path,
            }
            if PROJECT_ENV == "experimance":
                request["era"] = selected_era
                request["biome"] = selected_biome

            request_id = await client.send_render_request(**request)

            print(f"\nRequest {request_id} sent. Waiting for response...")
            start_time = time.monotonic()   
            response = await client.wait_for_response(request_id)
            duration = time.monotonic() - start_time
            print(f"Response received in {duration:.1f} seconds")

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
    # CLI acts as controller, so it binds to ports
    push_address = args.request_address if hasattr(args, 'request_address') else f"tcp://*:{DEFAULT_PORTS['image_requests']}"
    pull_address = args.result_address if hasattr(args, 'result_address') else f"tcp://*:{DEFAULT_PORTS['image_results']}"
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
        source_image_path = Path(args.source_image) if args.source_image else None

        # Send request and wait for response
        request = {
            "prompt": selected_prompt,
            "depth_map_path": depth_map_path,
            "image_path": source_image_path,
        }
        if PROJECT_ENV == "experimance":
            request["era"] = selected_era
            request["biome"] = selected_biome

        request_id = await client.send_render_request(**request)

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
    if PROJECT_ENV == "experimance":
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
            default="temperate_forest",
            help="Biome context for the image"
        )
    parser.add_argument(
        "--depth-map", "--depth_map", "-d",
        type=str,
        help="Path to depth map PNG file for ControlNet guidance"
    )
    parser.add_argument(
        "--source-image", "--source_image",
        type=str,
        help="Path to source image for image-to-image generation"
    )
    parser.add_argument(
        "--request-address", "--requests_address",
        type=str,
        default=f"tcp://*:{DEFAULT_PORTS['image_requests']}",
        help=f"ZMQ address for sending requests (default: tcp://*:{DEFAULT_PORTS['image_requests']})"
    )
    parser.add_argument(
        "--result-address", "--results_address",
        type=str,
        default=f"tcp://*:{DEFAULT_PORTS['image_results']}",
        help=f"ZMQ address for receiving results (default: tcp://*:{DEFAULT_PORTS['image_results']})"
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

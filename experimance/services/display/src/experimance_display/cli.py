#!/usr/bin/env python3
"""
Display Service CLI Tool for Manual Testing

This CLI tool allows you to send various ZMQ messages to the display service
for manual testing and development. It can send images, text overlays, video masks,
and other display control messages.

Usage:
    python cli.py image <image_path>                    # Display an image
    python cli.py text <text> [options]                 # Display text overlay
    python cli.py remove-text <text_id>                 # Remove text overlay
    python cli.py video-mask <mask_path>                # Update video mask
    python cli.py cycle-images [directory]              # Cycle through images
    python cli.py demo                                   # Run interactive demo
    python cli.py era-change <era> <biome>              # Send era change event
"""

import argparse
import asyncio
import logging
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import zmq
import zmq.asyncio

from experimance_common.schemas import ContentType, DisplayMedia, MessageType

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "common" / "src"))


from experimance_common.constants import (
    DEFAULT_PORTS, 
    GENERATED_IMAGES_DIR_ABS, 
    MOCK_IMAGES_DIR_ABS, 
    VIDEOS_DIR_ABS,
    ZMQ_TCP_BIND_PREFIX
)

# Import PubSubService and config
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class DisplayCLI:
    """CLI tool for testing the display service using PubSubService abstraction."""

    def __init__(self):
        self.pubsub_service: PubSubService = None  # type: ignore
        self._running = False

    async def setup_publishers(self):
        """Setup PubSubService for publishing events."""
        config = PubSubServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_TCP_BIND_PREFIX,
                port=DEFAULT_PORTS['events'],
            ),
            subscriber=None  # Not used for CLI
        )
        self.pubsub_service = PubSubService(config, name="DisplayCLI")
        await self.pubsub_service.start()
        self._running = True
        await asyncio.sleep(0.5)  # Allow time for service to start
        logger.info("PubSubService publisher initialized")

    async def cleanup(self):
        """Cleanup PubSubService resources."""
        if self.pubsub_service and self._running:
            await self.pubsub_service.stop()
            self._running = False

    
    async def send_display_media(self, image_path: str, image_type: str = "satellite_landscape"):
        """Send an ImageReady message."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return

        # Convert to absolute path and URI
        abs_path = os.path.abspath(image_path)
        uri = f"file://{abs_path}"

        message = DisplayMedia(
            request_id=str(uuid.uuid4()),
            content_type=ContentType.IMAGE,
            uri=uri,
        )

        logger.debug(f"Sending ImageReady: {message}")

        await self.pubsub_service.publish(message, topic=MessageType.DISPLAY_MEDIA)
        logger.info(f"Sent ImageReady: {os.path.basename(image_path)}")

        await asyncio.sleep(0.2)  # Allow time for processing before cleanup
    
    async def send_text_overlay(self, text_id: str, content: str, speaker: str = "system", 
                               duration: Optional[float] = None, position: str = "bottom_center"):
        """Send a TextOverlay message."""
        message = {
            "type": MessageType.DISPLAY_TEXT,
            "text_id": text_id,
            "content": content,
            "speaker": speaker,
            "duration": duration,
            "style": {
                "position": position
            }
        }
        #print(message)

        await self.pubsub_service.publish(message, topic=MessageType.DISPLAY_TEXT)
        logger.info(f"Sent TextOverlay: {text_id} - '{content[:50]}...'")
    
    async def send_remove_text(self, text_id: str):
        """Send a RemoveText message."""
        message = {
            "type": MessageType.REMOVE_TEXT,
            "text_id": text_id
        }

        await self.pubsub_service.publish(message, topic=MessageType.REMOVE_TEXT)
        logger.info(f"Sent RemoveText: {text_id}")
    
    async def send_video_mask(self, mask_path: str, fade_in_duration: float = 0.2, 
                             fade_out_duration: float = 1.0):
        """Send a VideoMask message."""
        if not os.path.exists(mask_path):
            logger.error(f"Mask file not found: {mask_path}")
            return

        # Convert to absolute path and URI
        abs_path = os.path.abspath(mask_path)
        uri = f"file://{abs_path}"

        message = {
            "type": MessageType.CHANGE_MAP,
            "mask_id": str(uuid.uuid4()),
            "uri": uri,
            "fade_in_duration": fade_in_duration,
            "fade_out_duration": fade_out_duration
        }

        await self.pubsub_service.publish(message, topic=MessageType.CHANGE_MAP)
        logger.info(f"Sent VideoMask: {os.path.basename(mask_path)}")
    
    async def send_era_changed(self, era: str, biome: str):
        """Send an EraChanged message."""
        message = {
            "type": MessageType.SPACE_TIME_UPDATE,
            "era": era,
            "biome": biome
        }
        await self.pubsub_service.publish(message, topic=MessageType.SPACE_TIME_UPDATE)
        logger.info(f"Sent EraChanged: {era}/{biome}")
    
    async def send_transition_ready(self, transition_path: str, from_image: str, to_image: str):
        """Send a TransitionReady message."""
        if not os.path.exists(transition_path):
            logger.error(f"Transition file not found: {transition_path}")
            return
        
        # Convert to absolute path and URI
        abs_path = os.path.abspath(transition_path)
        uri = f"file://{abs_path}"
        
        message = {
            "type": MessageType.TRANSITION_READY,
            "request_id": str(uuid.uuid4()),
            "transition_id": str(uuid.uuid4()),
            "uri": uri,
            "from_image": from_image,
            "to_image": to_image
        }
        
        await self.pubsub_service.publish(message, topic=MessageType.TRANSITION_READY)
        logger.info(f"Sent TransitionReady: {os.path.basename(transition_path)}")
    
    async def send_loop_ready(self, loop_path: str, still_uri: str, loop_type: str = "idle_animation"):
        """Send a LoopReady message."""
        if not os.path.exists(loop_path):
            logger.error(f"Loop file not found: {loop_path}")
            return
        
        # Convert to absolute path and URI
        abs_path = os.path.abspath(loop_path)
        uri = f"file://{abs_path}"
        
        message = {
            "type": MessageType.LOOP_READY,
            "request_id": str(uuid.uuid4()),
            "loop_id": str(uuid.uuid4()),
            "uri": uri,
            "still_uri": still_uri,
            "metadata": {
                "loop_type": loop_type
            }
        }
        
        await self.pubsub_service.publish(message, topic=MessageType.LOOP_READY)
        logger.info(f"Sent LoopReady: {os.path.basename(loop_path)}")


def get_available_images() -> List[str]:
    """Get list of available images for testing."""
    images = []
    
    # Generated images
    if GENERATED_IMAGES_DIR_ABS.exists():
        images.extend([str(p) for p in GENERATED_IMAGES_DIR_ABS.glob("*.webp")])
        images.extend([str(p) for p in GENERATED_IMAGES_DIR_ABS.glob("*.png")])
        images.extend([str(p) for p in GENERATED_IMAGES_DIR_ABS.glob("*.jpg")])
    
    # Mock images
    if MOCK_IMAGES_DIR_ABS.exists():
        images.extend([str(p) for p in MOCK_IMAGES_DIR_ABS.glob("*.png")])
        images.extend([str(p) for p in MOCK_IMAGES_DIR_ABS.glob("*.jpg")])
    
    return sorted(images)


def get_random_image() -> Optional[str]:
    """Get a random image from available images."""
    images = get_available_images()
    return random.choice(images) if images else None



def get_available_masks() -> List[str]:
    """Get list of available mask images for testing (from mock/mask/ directory)."""
    mask_dir = MOCK_IMAGES_DIR_ABS / "mask"
    masks = []
    if mask_dir.exists():
        masks.extend([str(p) for p in mask_dir.glob("*.png")])
        masks.extend([str(p) for p in mask_dir.glob("*.jpg")])
        masks.extend([str(p) for p in mask_dir.glob("*.webp")])
    return sorted(masks)

def get_random_mask() -> Optional[str]:
    """Get a random mask image from available masks."""
    masks = get_available_masks()
    return random.choice(masks) if masks else None

def get_default_video_mask() -> Optional[str]:
    """Get the default video mask file (first available mask, or None)."""
    masks = get_available_masks()
    return masks[0] if masks else None


def get_default_video() -> Optional[str]:
    """Get the default video file for transitions."""
    video_path = VIDEOS_DIR_ABS / "video_overlay.mp4"
    return str(video_path) if video_path.exists() else None


# Default text content options
DEFAULT_TEXTS = [
    "Welcome to Experimance",
    "System status: Online",
    "Exploring the intersection of nature and technology",
    "Interactive art installation active",
    "The landscape evolves with your presence",
    "Data flows like water through digital veins",
    "Where past and future converge",
    "Monitoring environmental changes...",
    "Sensors detecting movement",
    "Transitioning between eras",
    "The earth remembers everything",
    "Digital shadows of ancient forests",
]

# Default era/biome combinations
DEFAULT_ERAS_BIOMES = [
    ("wilderness", "forest"),
    ("wilderness", "grassland"),
    ("wilderness", "wetland"),
    ("anthropocene", "urban"),
    ("anthropocene", "suburban"),
    ("anthropocene", "industrial"),
    ("rewilded", "forest"),
    ("rewilded", "grassland"),
    ("rewilded", "wetland"),
]


async def cycle_images_command(cli: DisplayCLI, directory: Optional[str] = None, interval: float = 3.0):
    """Cycle through available images."""
    if directory:
        image_dir = Path(directory)
        if not image_dir.exists():
            logger.error(f"Directory not found: {directory}")
            return
        images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.webp"))
        images = [str(p) for p in images]
    else:
        images = get_available_images()
    
    if not images:
        logger.error("No images found to cycle through")
        return
    
    logger.info(f"Cycling through {len(images)} images every {interval}s")
    logger.info("Press Ctrl+C to stop")
    
    try:
        for i, image_path in enumerate(images):
            logger.info(f"[{i+1}/{len(images)}] Displaying: {os.path.basename(image_path)}")
            await cli.send_display_media(image_path)
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Image cycling stopped")


async def demo_command(cli: DisplayCLI):
    """Run an interactive demo showing various display features."""
    logger.info("Starting Display Service Demo")
    logger.info("=" * 50)
    
    # Demo text overlays
    logger.info("1. Text Overlay Demo")
    await cli.send_text_overlay("welcome", "Welcome to Experimance", "agent", 5.0, "bottom_center")
    await asyncio.sleep(2)
    
    await cli.send_text_overlay("system_info", "System Status: Running", "system", None, "top_right")
    await asyncio.sleep(2)
    
    await cli.send_text_overlay("debug_info", "FPS: 60.0 | GPU: 45%", "debug", None, "top_left")
    await asyncio.sleep(3)
    
    # Demo era change
    logger.info("2. Era Change Demo")
    await cli.send_era_changed("wilderness", "forest")
    await asyncio.sleep(2)
    
    # Demo images
    logger.info("3. Image Display Demo")
    images = get_available_images()[:5]  # Show first 5 images
    for i, image_path in enumerate(images):
        logger.info(f"Showing image {i+1}/{len(images)}: {os.path.basename(image_path)}")
        await cli.send_display_media(image_path)
        await asyncio.sleep(3)
    
    # Demo video mask
    logger.info("4. Video Mask Demo")
    mock_mask = MOCK_IMAGES_DIR_ABS / "mock_video_mask.png"
    if mock_mask.exists():
        await cli.send_video_mask(str(mock_mask))
        await asyncio.sleep(3)
    
    # Remove text overlays
    logger.info("5. Text Removal Demo")
    await cli.send_remove_text("system_info")
    await asyncio.sleep(1)
    await cli.send_remove_text("debug_info")
    await asyncio.sleep(1)
    
    # Demo transition (using video if available)
    logger.info("6. Transition Demo")
    video_path = VIDEOS_DIR_ABS / "video_overlay.mp4"
    if video_path.exists():
        await cli.send_transition_ready(str(video_path), "prev_image", "next_image")
        await asyncio.sleep(3)
    
    logger.info("Demo completed!")


def main():
    parser = argparse.ArgumentParser(description="Display Service CLI Tool for Manual Testing")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Image command
    img_parser = subparsers.add_parser("image", help="Send an image to display")
    img_parser.add_argument("path", nargs="?", help="Path to image file (random if not provided)")
    img_parser.add_argument("--type", default="satellite_landscape", help="Image type")
    
    # Text overlay command
    text_parser = subparsers.add_parser("text", help="Send text overlay")
    text_parser.add_argument("content", nargs="?", help="Text content to display (random if not provided)")
    text_parser.add_argument("--id", default=None, help="Text ID (auto-generated if not provided)")
    text_parser.add_argument("--speaker", default="system", choices=["agent", "system", "debug"])
    text_parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    text_parser.add_argument("--position", default="bottom_center", 
                           choices=["top_left", "top_center", "top_right", 
                                   "center_left", "center", "center_right",
                                   "bottom_left", "bottom_center", "bottom_right"])
    
    # Remove text command
    remove_parser = subparsers.add_parser("remove-text", help="Remove text overlay")
    remove_parser.add_argument("text_id", help="ID of text to remove")
    
    # Video mask command
    mask_parser = subparsers.add_parser("video-mask", help="Send video mask")
    mask_parser.add_argument("path", nargs="?", help="Path to mask image file (default mock if not provided)")
    mask_parser.add_argument("--fade-in", type=float, default=0.2, help="Fade in duration")
    mask_parser.add_argument("--fade-out", type=float, default=1.0, help="Fade out duration")
    
    # Era change command
    era_parser = subparsers.add_parser("era-change", help="Send era change event")
    era_parser.add_argument("era", nargs="?", help="Era name (random if not provided)")
    era_parser.add_argument("biome", nargs="?", help="Biome name (random if not provided)")
    
    # Transition command
    trans_parser = subparsers.add_parser("transition", help="Send transition ready")
    trans_parser.add_argument("path", nargs="?", help="Path to transition video/image (default if not provided)")
    trans_parser.add_argument("--from-image", default="prev", help="From image ID")
    trans_parser.add_argument("--to-image", default="next", help="To image ID")
    
    # Loop command
    loop_parser = subparsers.add_parser("loop", help="Send loop ready")
    loop_parser.add_argument("path", nargs="?", help="Path to loop video (default if not provided)")
    loop_parser.add_argument("still_uri", nargs="?", help="URI of still image this loop animates (random if not provided)")
    loop_parser.add_argument("--type", default="idle_animation", help="Loop type")
    
    # Cycle images command
    cycle_parser = subparsers.add_parser("cycle-images", help="Cycle through images")
    cycle_parser.add_argument("directory", nargs="?", help="Directory with images (uses default if not provided)")
    cycle_parser.add_argument("--interval", type=float, default=3.0, help="Interval between images in seconds")
    
    # Demo command
    subparsers.add_parser("demo", help="Run interactive demo")

    # List command
    subparsers.add_parser("list", help="List available test resources")

    # Stress test command
    stress_parser = subparsers.add_parser("stress", help="Run stress test (randomly send text, image, or change map)")
    stress_parser.add_argument("interval", type=float, help="Interval in seconds between sends (float)")

    parser.add_argument("-vv", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except errors")

    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Set logging level based on flags
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    async def run_command():
        cli = DisplayCLI()
        try:
            await cli.setup_publishers()

            if args.command == "image":
                image_path = args.path or get_random_image()
                if not image_path:
                    logger.error("No image provided and no default images available")
                    return
                await cli.send_display_media(image_path, args.type)

            elif args.command == "text":
                content = args.content or random.choice(DEFAULT_TEXTS)
                text_id = args.id or str(uuid.uuid4())[:8]
                await cli.send_text_overlay(text_id, content, args.speaker, args.duration, args.position)

            elif args.command == "remove-text":
                await cli.send_remove_text(args.text_id)

            elif args.command == "video-mask":
                mask_path = args.path or get_random_mask()
                if not mask_path:
                    logger.error("No mask provided and no available mask found in mock/mask/")
                    return
                await cli.send_video_mask(mask_path, args.fade_in, args.fade_out)

            elif args.command == "era-change":
                if args.era and args.biome:
                    era, biome = args.era, args.biome
                elif args.era and not args.biome:
                    era = args.era
                    # Pick a random biome
                    matching_biomes = [b for e, b in DEFAULT_ERAS_BIOMES if e == era]
                    biome = random.choice(matching_biomes) if matching_biomes else random.choice([b for _, b in DEFAULT_ERAS_BIOMES])
                else:
                    # Pick random era/biome combination
                    era, biome = random.choice(DEFAULT_ERAS_BIOMES)
                await cli.send_era_changed(era, biome)

            elif args.command == "transition":
                transition_path = args.path or get_default_video()
                if not transition_path:
                    logger.error("No transition provided and no default video available")
                    return
                await cli.send_transition_ready(transition_path, args.from_image, args.to_image)

            elif args.command == "loop":
                loop_path = args.path or get_default_video()
                still_uri = args.still_uri or f"file://{get_random_image()}" if get_random_image() else "file:///tmp/placeholder.png"
                if not loop_path:
                    logger.error("No loop provided and no default video available")
                    return
                await cli.send_loop_ready(loop_path, still_uri, args.type)

            elif args.command == "cycle-images":
                await cycle_images_command(cli, args.directory, args.interval)

            elif args.command == "demo":
                await demo_command(cli)

            elif args.command == "list":
                print("Available test resources:")
                print("\nGenerated Images:")
                images = get_available_images()
                for img in images[:10]:  # Show first 10
                    print(f"  {img}")
                if len(images) > 10:
                    print(f"  ... and {len(images) - 10} more")

                print("\nMock Files:")
                for mock_file in [MOCK_IMAGES_DIR_ABS / "mock_depth_map.png", 
                                MOCK_IMAGES_DIR_ABS / "mock_video_mask.png"]:
                    if mock_file.exists():
                        print(f"  {mock_file}")

                print("\nVideo Files:")
                video_file = VIDEOS_DIR_ABS / "video_overlay.mp4"
                if video_file.exists():
                    print(f"  {video_file}")

                return  # Don't wait for cleanup

            elif args.command == "stress":
                logger.info(f"Starting stress test with interval {args.interval}s")
                actions = ["text", "image", "change_map"]
                active_text_ids = set()  # Track active text IDs to avoid duplicates
                progressive_texts = {}  # Track progressive texts
                while True:
                    action = random.choice(actions)
                    if action == "text":
                        # chance to do progressive text, else normal
                        if random.random() < 0.8:
                            # Pick an active text_id to update, or create a new one if none
                            if len(progressive_texts) == 0:
                                # Start a new progressive text
                                content = random.choice(DEFAULT_TEXTS)
                                text_id = str(uuid.uuid4())[:8]
                                active_text_ids.add(text_id)  # Track this text ID
                                words = content.split()
                                progressive_texts[text_id] = {"words": words, "idx": 1, 
                                    "speaker": random.choice(["agent", "system", "debug"]), 
                                    "position": random.choice([
                                        "top_left", "top_center", "top_right",
                                        "center_left", "center", "center_right",
                                        "bottom_left", "bottom_center", "bottom_right"]), 
                                    "duration": random.choice([None, 2.0, 5.0, 10.0])}
                            else:
                                # Pick a random progressive text to update
                                text_id = random.choice(list(progressive_texts.keys()))
                            prog = progressive_texts[text_id]
                            idx = prog["idx"]
                            words = prog["words"]
                            if idx < len(words):
                                prog["idx"] += 1
                            content = " ".join(words[:prog["idx"]])
                            await cli.send_text_overlay(text_id, content, prog["speaker"], prog["duration"], prog["position"])
                            logger.debug(f"Stress: Progressive text overlay {text_id}: '{content}'")
                            # If finished, remove from progressive_texts
                            if prog["idx"] >= len(words):
                                del progressive_texts[text_id]
                        else:
                            if len(active_text_ids) >= 10:
                                # Remove a random text overlay if we have too many active
                                text_id = random.choice(list(active_text_ids))
                                active_text_ids.remove(text_id)
                                await cli.send_remove_text(text_id)
                                logger.debug(f"Stress: Removed text overlay {text_id}, remaining: {len(active_text_ids)}")
                            content = random.choice(DEFAULT_TEXTS)
                            text_id = str(uuid.uuid4())[:8]
                            active_text_ids.add(text_id)  # Track this text ID
                            speaker = random.choice(["agent", "system", "debug"])
                            position = random.choice([
                                "top_left", "top_center", "top_right",
                                "center_left", "center", "center_right",
                                "bottom_left", "bottom_center", "bottom_right"
                            ])
                            duration = random.choice([None, 2.0, 5.0, 10.0])
                            await cli.send_text_overlay(text_id, content, speaker, duration, position)
                            logger.debug(f"Stress: Sent text overlay {text_id}")
                            # For progressive text, also add to progressive_texts dict
                            if not hasattr(run_command, "progressive_texts"):
                                run_command.progressive_texts = {}
                            run_command.progressive_texts[text_id] = {"words": content.split(), "idx": 1, "speaker": speaker, "position": position, "duration": duration}
                    elif action == "image":
                        image_path = get_random_image()
                        if image_path:
                            await cli.send_display_media(image_path)
                            logger.debug(f"Stress: Sent image {image_path}")
                        else:
                            logger.warning("Stress: No image available to send")
                    elif action == "change_map":
                        mask_path = get_random_mask()
                        if mask_path:
                            fade_in = random.choice([0.1, 0.2, 0.5])
                            fade_out = random.choice([0.5, 1.0, 2.0])
                            await cli.send_video_mask(mask_path, fade_in, fade_out)
                            logger.debug(f"Stress: Sent change map {mask_path}")
                        else:
                            logger.warning("Stress: No mask available to send")
                    await asyncio.sleep(args.interval)

            # Give time for message to be sent
            await asyncio.sleep(0.1)

        finally:
            await cli.cleanup()

    # Run the async command
    asyncio.run(run_command())


if __name__ == "__main__":
    main()

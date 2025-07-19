#!/usr/bin/env python3
"""
Example showing how to use image transport utilities for ZMQ communication.

This demonstrates the different transport modes and how publishers can easily
choose between file URI and base64 encoded images based on configuration.
"""

import asyncio
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq.zmq_utils import (
    IMAGE_TRANSPORT_MODES,
    prepare_image_message, 
    choose_image_transport_mode,
    MessageType
)
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService

logger = logging.getLogger(__name__)


class ImageTransportExample(ZmqPublisherSubscriberService):
    """Example service demonstrating image transport patterns."""
    
    def __init__(self, transport_mode: str = IMAGE_TRANSPORT_MODES["AUTO"]):
        super().__init__(
            service_name="image_transport_example",
            pub_address=f"tcp://*:{DEFAULT_PORTS['events']}",
            sub_address=f"tcp://localhost:{DEFAULT_PORTS['events']}",
            subscribe_topics=["test"],
            publish_topic="image_transport"
        )
        
        self.transport_mode = transport_mode
        self.target_addresses = [
            "tcp://localhost:5555",  # Local display service
            "tcp://192.168.1.100:5555",  # Remote display service
        ]
        
    async def run(self):
        """Run the example."""
        logger.info(f"Starting image transport example with mode: {self.transport_mode}")
        
        # Create a test image
        test_image = self._create_test_image()
        test_image_path = Path("test_image.png")
        test_image.save(test_image_path)
        
        # Create a test numpy array
        test_array = self._create_test_array()
        
        try:
            # Example 1: Send image to local target
            await self._send_video_mask_local(test_image_path)
            
            # Example 2: Send image to remote target
            await self._send_video_mask_remote(test_image_path)
            
            # Example 3: Send PIL image directly
            await self._send_pil_image(test_image)
            
            # Example 4: Send numpy array directly
            await self._send_numpy_array(test_array)
            
            # Example 5: Show different transport modes
            await self._demonstrate_transport_modes(test_image_path)
            
        finally:
            # Cleanup
            if test_image_path.exists():
                test_image_path.unlink()

    def _create_test_image(self) -> Image.Image:
        """Create a test image for demonstration."""
        img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple mask pattern
        draw.ellipse([100, 100, 400, 400], fill=(255, 255, 255, 255))
        draw.ellipse([150, 150, 350, 350], fill=(0, 0, 0, 0))
        
        return img
    
    def _create_test_array(self) -> np.ndarray:
        """Create a test numpy array for demonstration."""
        # Create a simple pattern
        height, width = 256, 256
        array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a checkerboard pattern
        square_size = 32
        for y in range(height):
            for x in range(width):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    array[y, x] = [255, 255, 255]  # White squares
                else:
                    array[y, x] = [255, 0, 0]      # Red squares
        
        return array

    async def _send_video_mask_local(self, image_path: Path):
        """Example: Send VideoMask to local target."""
        logger.info("Sending VideoMask to local target...")
        
        # Prepare message for local target
        message = prepare_image_message(
            image_data=image_path,
            target_address="tcp://localhost:5555",
            transport_mode=self.transport_mode,
            
            # VideoMask message fields
            mask_id="example_mask_local",
            fade_in_duration=2.0,
            fade_out_duration=1.5
        )
        
        logger.info(f"Local message has URI: {'uri' in message}, image_data: {'image_data' in message}")
        
        # Publish the message
        await self.publish_message(message, topic=MessageType.CHANGE_MAP)
    
    async def _send_video_mask_remote(self, image_path: Path):
        """Example: Send VideoMask to remote target."""
        logger.info("Sending VideoMask to remote target...")
        
        # Prepare message for remote target
        message = prepare_image_message(
            image_data=image_path,
            target_address="tcp://192.168.1.100:5555",
            transport_mode=self.transport_mode,
            
            # VideoMask message fields
            mask_id="example_mask_remote",
            fade_in_duration=2.0,
            fade_out_duration=1.5
        )
        
        logger.info(f"Remote message has URI: {'uri' in message}, image_data: {'image_data' in message}")
        
        # Publish the message
        await self.publish_message(message, topic=MessageType.CHANGE_MAP)
    
    async def _send_pil_image(self, image: Image.Image):
        """Example: Send PIL Image directly."""
        logger.info("Sending PIL Image directly...")
        
        # Prepare message with PIL Image
        message = prepare_image_message(
            image_data=image,
            target_address="tcp://localhost:5555",
            transport_mode=self.transport_mode,
            
            # VideoMask message fields
            mask_id="example_mask_pil",
            fade_in_duration=1.0,
            fade_out_duration=1.0
        )
        
        logger.info(f"PIL message has URI: {'uri' in message}, image_data: {'image_data' in message}")
        
        # Publish the message
        await self.publish_message(message, topic=MessageType.CHANGE_MAP)
    
    async def _send_numpy_array(self, array: np.ndarray):
        """Example: Send numpy array directly."""
        logger.info("Sending numpy array directly...")
        
        # Prepare message with numpy array
        message = prepare_image_message(
            image_data=array,
            target_address="tcp://localhost:5555",
            transport_mode=self.transport_mode,
            
            # VideoMask message fields
            mask_id="example_mask_numpy",
            fade_in_duration=1.5,
            fade_out_duration=2.0
        )
        
        logger.info(f"Numpy message has URI: {'uri' in message}, image_data: {'image_data' in message}")
        
        # Publish the message
        await self.publish_message(message, topic=MessageType.CHANGE_MAP)
    
    async def _demonstrate_transport_modes(self, image_path: Path):
        """Demonstrate different transport modes."""
        logger.info("Demonstrating different transport modes...")
        
        modes = [
            IMAGE_TRANSPORT_MODES["FILE_URI"],
            IMAGE_TRANSPORT_MODES["BASE64"],
            IMAGE_TRANSPORT_MODES["HYBRID"],
            IMAGE_TRANSPORT_MODES["AUTO"]
        ]
        
        for mode in modes:
            logger.info(f"\n--- Testing mode: {mode} ---")
            
            # Test with local target
            chosen_local = choose_image_transport_mode(
                file_path=image_path,
                target_address="tcp://localhost:5555",
                transport_mode=mode
            )
            logger.info(f"Local target -> chosen mode: {chosen_local}")
            
            # Test with remote target
            chosen_remote = choose_image_transport_mode(
                file_path=image_path,
                target_address="tcp://192.168.1.100:5555",
                transport_mode=mode
            )
            logger.info(f"Remote target -> chosen mode: {chosen_remote}")
            
            # Create message for local target
            message_local = prepare_image_message(
                image_data=image_path,
                target_address="tcp://localhost:5555",
                transport_mode=mode,
                mask_id=f"test_{mode}_local"
            )
            
            # Create message for remote target
            message_remote = prepare_image_message(
                image_data=image_path,
                target_address="tcp://192.168.1.100:5555",
                transport_mode=mode,
                mask_id=f"test_{mode}_remote"
            )
            
            logger.info(f"Local message: URI={bool(message_local.get('uri'))}, "
                       f"image_data={bool(message_local.get('image_data'))}")
            logger.info(f"Remote message: URI={bool(message_remote.get('uri'))}, "
                       f"image_data={bool(message_remote.get('image_data'))}")


async def main():
    """Run the image transport example."""
    logging.basicConfig(level=logging.INFO)
    
    # Test different transport modes
    modes = [
        IMAGE_TRANSPORT_MODES["AUTO"],
        IMAGE_TRANSPORT_MODES["FILE_URI"],
        IMAGE_TRANSPORT_MODES["BASE64"],
        IMAGE_TRANSPORT_MODES["HYBRID"]
    ]
    
    for mode in modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing transport mode: {mode}")
        logger.info(f"{'='*50}")
        
        service = ImageTransportExample(transport_mode=mode)
        await service.start()
        
        try:
            await service.run()
            await asyncio.sleep(1)  # Brief pause between modes
        finally:
            await service.stop()


if __name__ == "__main__":
    asyncio.run(main())

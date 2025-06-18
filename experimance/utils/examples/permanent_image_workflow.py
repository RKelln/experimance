#!/usr/bin/env python3
"""
Example showing the recommended workflow for services that generate images.

This demonstrates:
1. Generating images with numpy/PIL
2. Saving them permanently to GENERATED_IMAGES_DIR  
3. Using file paths with prepare_image_message() for efficient transport
4. The difference between temporary files vs permanent files
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime

from experimance_common.constants import GENERATED_IMAGES_DIR_ABS
from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES, MessageType
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService

logger = logging.getLogger(__name__)


class ImageGeneratorService:
    """Example service that generates and saves images permanently."""
    
    def __init__(self):
        # Ensure generated images directory exists
        GENERATED_IMAGES_DIR_ABS.mkdir(parents=True, exist_ok=True)
        
    def generate_and_save_image(self, image_id: str, era: str = "default") -> Path:
        """Generate an image and save it permanently.
        
        Args:
            image_id: Unique identifier for the image
            era: Era/category for organizing images
            
        Returns:
            Path to the saved image file
        """
        # Create era subdirectory
        era_dir = GENERATED_IMAGES_DIR_ABS / era
        era_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_id}_{timestamp}.png"
        file_path = era_dir / filename
        
        # Generate image content (example: gradient pattern)
        image = self._create_gradient_image(image_id)
        
        # Save permanently
        image.save(file_path)
        logger.info(f"Saved generated image: {file_path}")
        
        return file_path
    
    def generate_numpy_and_save(self, image_id: str, era: str = "numpy") -> Path:
        """Generate a numpy array image and save it permanently.
        
        Args:
            image_id: Unique identifier for the image
            era: Era/category for organizing images
            
        Returns:
            Path to the saved image file
        """
        from experimance_common.image_utils import ndarray_to_base64url
        
        # Create era subdirectory
        era_dir = GENERATED_IMAGES_DIR_ABS / era
        era_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_id}_{timestamp}.png"
        file_path = era_dir / filename
        
        # Generate numpy image content
        array = self._create_numpy_pattern(image_id)
        
        # Convert to PIL and save permanently
        if len(array.shape) == 2:
            pil_image = Image.fromarray(array, mode='L')
        else:
            pil_image = Image.fromarray(array, mode='RGB')
        
        pil_image.save(file_path)
        logger.info(f"Saved numpy-generated image: {file_path}")
        
        return file_path
    
    def _create_gradient_image(self, image_id: str) -> Image.Image:
        """Create a gradient image based on image_id."""
        width, height = 512, 512
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Create different patterns based on image_id
        hash_val = hash(image_id) % 256
        
        # Gradient background
        for y in range(height):
            color_value = int((y / height) * 255)
            color = (hash_val, color_value, 255 - color_value)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add some geometric shapes
        draw.ellipse([100, 100, 400, 400], outline=(255, 255, 255), width=3)
        draw.text((200, 240), f"ID: {image_id}", fill=(255, 255, 255))
        
        return image
    
    def _create_numpy_pattern(self, image_id: str) -> np.ndarray:
        """Create a numpy pattern based on image_id."""
        width, height = 256, 256
        array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create pattern based on hash of image_id
        hash_val = hash(image_id)
        
        # Fill with pattern
        for y in range(height):
            for x in range(width):
                r = (x + hash_val) % 256
                g = (y + hash_val) % 256
                b = (x * y + hash_val) % 256
                array[y, x] = [r, g, b]
        
        return array


class ImagePublisherService(ZmqPublisherSubscriberService):
    """Service that publishes generated images using file paths."""
    
    def __init__(self):
        super().__init__(
            service_name="image_publisher_example",
            pub_address="tcp://*:5555",
            sub_address="tcp://localhost:5555",
            subscribe_topics=["heartbeat"],
            publish_topic="generated_images"
        )
        self.generator = ImageGeneratorService()
    
    async def publish_generated_image(self, image_id: str, era: str = "default"):
        """Generate an image, save it permanently, and publish via ZMQ."""
        
        # Step 1: Generate and save image permanently
        image_path = self.generator.generate_and_save_image(image_id, era)
        
        # Step 2: Send via ZMQ using file path (most efficient)
        message = prepare_image_message(
            image_data=image_path,  # Pass the saved file path
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"],  # Will prefer FILE_URI for local
            
            # Message metadata
            image_id=image_id,
            era=era,
            file_path=str(image_path),  # Include path for receiver reference
            generated_at=datetime.now().isoformat()
        )
        
        logger.info(f"Publishing image: {image_id}")
        logger.info(f"  File path: {image_path}")
        logger.info(f"  Transport: {'URI' if 'uri' in message else 'base64'}")
        
        await self.publish_message(message, topic=MessageType.IMAGE_READY)
        
        return image_path
    
    async def publish_numpy_image(self, image_id: str):
        """Generate numpy image, save permanently, and publish."""
        
        # Step 1: Generate and save numpy image permanently
        image_path = self.generator.generate_numpy_and_save(image_id, "numpy")
        
        # Step 2: Send via ZMQ using file path
        message = prepare_image_message(
            image_data=image_path,  # Pass the saved file path
            target_address="tcp://192.168.1.100:5555",  # Remote target
            transport_mode=IMAGE_TRANSPORT_MODES["HYBRID"],  # Send both URI and base64
            
            # Message metadata
            image_id=image_id,
            era="numpy",
            file_path=str(image_path),
            generated_at=datetime.now().isoformat()
        )
        
        logger.info(f"Publishing numpy image: {image_id}")
        logger.info(f"  File path: {image_path}")
        logger.info(f"  URI: {'uri' in message}")
        logger.info(f"  Base64: {'image_data' in message}")
        
        await self.publish_message(message, topic=MessageType.IMAGE_READY)
        
        return image_path


async def demonstrate_permanent_vs_temporary():
    """Demonstrate the difference between permanent and temporary image workflows."""
    
    logger.info("=== PERMANENT IMAGE WORKFLOW ===")
    
    # This is the RECOMMENDED workflow for generated images
    publisher = ImagePublisherService()
    await publisher.start()
    
    try:
        # Generate and publish images that are saved permanently
        perm_path1 = await publisher.publish_generated_image("landscape_001", "landscapes")
        perm_path2 = await publisher.publish_generated_image("abstract_001", "abstract")
        perm_path3 = await publisher.publish_numpy_image("numpy_pattern_001")
        
        logger.info(f"Permanent images saved:")
        logger.info(f"  {perm_path1}")
        logger.info(f"  {perm_path2}")
        logger.info(f"  {perm_path3}")
        
        # These files will persist for future use, analysis, etc.
        
    finally:
        await publisher.stop()
    
    logger.info("\n=== TEMPORARY IMAGE WORKFLOW ===")
    
    # This workflow uses temporary files (for one-time use)
    temp_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    
    temp_message = prepare_image_message(
        image_data=temp_array,  # Numpy array -> creates temp file
        target_address="tcp://localhost:5555",
        transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],
        image_id="temp_image_001"
    )
    
    logger.info("Temporary image message:")
    logger.info(f"  URI: {temp_message.get('uri')}")
    logger.info(f"  Temp file: {temp_message.get('_temp_file')}")
    
    # Cleanup temporary file after use
    if '_temp_file' in temp_message:
        temp_file = temp_message['_temp_file']
        if Path(temp_file).exists():
            Path(temp_file).unlink()
            logger.info(f"  Cleaned up: {temp_file}")


async def main():
    """Run the permanent image workflow example."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    logger.info("Image Generation and Publishing Example")
    logger.info(f"Generated images will be saved to: {GENERATED_IMAGES_DIR_ABS}")
    
    await demonstrate_permanent_vs_temporary()
    
    logger.info("\n✅ Example completed!")
    logger.info(f"Check {GENERATED_IMAGES_DIR_ABS} for saved images")


if __name__ == "__main__":
    asyncio.run(main())

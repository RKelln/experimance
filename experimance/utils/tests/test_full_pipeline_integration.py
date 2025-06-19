#!/usr/bin/env python3
"""
Full pipeline integration test: Core → Image Server → Core → Display

Tests the complete message flow:
1. Core service publishes RenderRequest 
2. Image Server receives and generates mock image
3. Image Server publishes ImageReady back to Core
4. Core evaluates transition logic and publishes DisplayMedia
5. Display service receives and processes DisplayMedia

This test uses real ZMQ communication but mocked image generation for reliability.
"""

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

import numpy as np
import pytest
import zmq
import zmq.asyncio
from PIL import Image

from experimance_common.constants import (
    DEFAULT_PORTS, 
    IMAGE_TRANSPORT_MODES
)
from experimance_common.image_utils import ImageLoadFormat
from experimance_common.zmq.zmq_utils import (
    MessageType, 
    ZmqPublisher, 
    ZmqSubscriber,
    ZmqPushSocket,
    ZmqPullSocket,
    prepare_image_message
)
from experimance_common.schemas import ContentType, Era, Biome
from experimance_common.image_utils import load_image_from_message

# Set up logging for better debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockImageServer:
    """Mock image server that generates test images and publishes ImageReady messages."""
    
    def __init__(self, events_address: str = "tcp://localhost:5570"):
        self.events_address = events_address
        self.subscriber = None      # SUB socket for RenderRequest
        self.publisher = None       # PUB socket for ImageReady  
        self.running = False
        self.generated_images = {}  # Store generated images for verification
        
    async def start(self):
        """Start the mock image server."""
        # SUB socket for RenderRequest (connects to Core's PUB)
        self.subscriber = ZmqSubscriber(self.events_address, [MessageType.RENDER_REQUEST], use_asyncio=True)
        
        # PUB socket for ImageReady (Core will SUB from this)
        pub_address = self.events_address.replace("5570", "5580")  # Use different port for publishing
        self.publisher = ZmqPublisher(pub_address.replace("localhost", "*"), use_asyncio=True)
        
        self.running = True
        logger.info("Mock Image Server started")
        
    async def stop(self):
        """Stop the mock image server."""
        self.running = False
        if self.subscriber:
            self.subscriber.close()
        if self.publisher:
            self.publisher.close()
        logger.info("Mock Image Server stopped")
        
    async def run(self):
        """Main server loop - listens for RenderRequest and responds with ImageReady."""
        while self.running:
            try:
                # Listen for RenderRequest messages using SUB socket
                topic, message = await asyncio.wait_for(
                    self.subscriber.receive_async(), 
                    timeout=0.1
                )
                
                if message.get("type") == MessageType.RENDER_REQUEST.value:
                    await self._handle_render_request(message)
                    
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue listening
            except Exception as e:
                logger.error(f"Error in mock image server: {e}")
                
    async def _handle_render_request(self, request: Dict[str, Any]):
        """Handle a RenderRequest by generating a mock image and publishing ImageReady."""
        request_id = request.get("request_id")
        era = request.get("era", "wilderness")
        biome = request.get("biome", "temperate_forest")
        
        logger.info(f"Mock Image Server: Generating image for {era}/{biome}")
        
        # Generate a mock image based on era/biome
        mock_image = self._generate_mock_image(era, biome)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        mock_image.save(temp_file.name)
        temp_file.close()
        
        # Store for verification
        image_id = str(uuid.uuid4())
        self.generated_images[image_id] = {
            "era": era,
            "biome": biome,
            "file_path": temp_file.name,
            "request_id": request_id
        }
        
        # Create ImageReady message
        image_ready = {
            "type": MessageType.IMAGE_READY.value,
            "request_id": request_id,
            "image_id": image_id,
            "uri": f"file://{temp_file.name}"
        }
        
        # Publish ImageReady using PUB socket
        await self.publisher.publish_async(image_ready, MessageType.IMAGE_READY)
        logger.info(f"Mock Image Server: Published ImageReady for {image_id}")
        
    def _generate_mock_image(self, era: str, biome: str) -> Image.Image:
        """Generate a mock satellite image based on era and biome."""
        # Create a 512x512 image with era/biome-specific colors
        width, height = 512, 512
        
        # Era-based base colors (RGB)
        era_colors = {
            "wilderness": (34, 139, 34),      # Forest green
            "pre_industrial": (139, 69, 19),  # Saddle brown
            "modern": (128, 128, 128),        # Gray (urban)
            "ai_future": (0, 191, 255),       # Deep sky blue
            "dystopia": (139, 0, 0),          # Dark red
        }
        
        # Biome-based modifiers
        biome_modifiers = {
            "temperate_forest": (1.0, 1.2, 1.0),
            "desert": (1.3, 1.1, 0.7),
            "coastal": (0.8, 0.9, 1.4),
            "urban": (0.9, 0.9, 0.9),
            "mountain": (0.8, 0.8, 1.2),
        }
        
        base_color = era_colors.get(era, (100, 100, 100))
        modifier = biome_modifiers.get(biome, (1.0, 1.0, 1.0))
        
        # Apply biome modifier to era color
        final_color = tuple(
            int(min(255, max(0, base_color[i] * modifier[i]))) 
            for i in range(3)
        )
        
        # Create image with some variation
        image_array = np.full((height, width, 3), final_color, dtype=np.uint8)
        
        # Add some random variation to make it look more realistic
        noise = np.random.randint(-20, 21, (height, width, 3))
        image_array = np.clip(image_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(image_array)


class MockCoreService:
    """Mock core service that publishes RenderRequest and handles ImageReady/DisplayMedia."""
    
    def __init__(self, events_address: str = "tcp://localhost:5561"):  # Use unique port
        self.events_address = events_address
        self.publisher = None
        self.subscriber = None
        self.running = False
        self.published_display_media = []  # Store for verification
        self.received_image_ready = []     # Store for verification
        
    async def start(self):
        """Start the mock core service."""
        # Publisher for RenderRequest and DisplayMedia messages (bind to the event bus)
        self.publisher = ZmqPublisher(self.events_address.replace("localhost", "*"), use_asyncio=True)
        
        # Subscriber for ImageReady messages (connect to image server)
        self.subscriber = ZmqSubscriber("tcp://localhost:5571", [MessageType.IMAGE_READY], use_asyncio=True)
        
        self.running = True
        logger.info("Mock Core Service started")
        
    async def stop(self):
        """Stop the mock core service."""
        self.running = False
        if self.publisher:
            self.publisher.close()
        if self.subscriber:
            self.subscriber.close()
        logger.info("Mock Core Service stopped")
            
    async def publish_render_request(self, era: str = "modern", biome: str = "urban") -> str:
        """Publish a RenderRequest and return the request_id."""
        request_id = str(uuid.uuid4())
        
        render_request = {
            "type": MessageType.RENDER_REQUEST.value,
            "request_id": request_id,
            "era": era,
            "biome": biome,
            "prompt": f"A satellite view of {biome} landscape in the {era} era",
            "depth_map_png": None  # No depth map for this test
        }
        
        await self.publisher.publish_async(render_request, MessageType.RENDER_REQUEST)
        logger.info(f"Mock Core: Published RenderRequest {request_id} for {era}/{biome}")
        return request_id
        
    async def listen_for_image_ready(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Listen for ImageReady messages and process them."""
        try:
            topic, message = await asyncio.wait_for(
                self.subscriber.receive_async(),
                timeout=timeout
            )
            
            if message.get("type") == MessageType.IMAGE_READY.value:
                self.received_image_ready.append(message)
                await self._handle_image_ready(message)
                return message
                
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for ImageReady message")
            return None
        except Exception as e:
            logger.error(f"Error listening for ImageReady: {e}")
            return None
            
    async def _handle_image_ready(self, image_message: Dict[str, Any]):
        """Handle ImageReady by creating and publishing DisplayMedia."""
        # Simulate core service logic for deciding transitions
        should_transition = True  # For test, always transition
        transition_type = "fade"
        
        # Load the image to include in DisplayMedia (simulate enum-based transport)
        image_uri = image_message.get("uri")
        if not image_uri:
            logger.error("ImageReady message missing URI")
            return
            
        # Create DisplayMedia message with intelligent transport
        display_media = {
            "type": MessageType.DISPLAY_MEDIA.value,
            "content_type": ContentType.IMAGE.value,
            "uri": image_uri,  # Use file URI for efficiency
            "fade_in": 0.5 if should_transition else 0.0,
            "fade_out": 0.3,
            "era": image_message.get("era", "unknown"),
            "biome": image_message.get("biome", "unknown"),
            "source_request_id": image_message.get("request_id")
        }
        
        # Publish DisplayMedia
        await self.publisher.publish_async(display_media, MessageType.DISPLAY_MEDIA)
        self.published_display_media.append(display_media)
        logger.info(f"Mock Core: Published DisplayMedia with {transition_type} transition")


class MockDisplayService:
    """Mock display service that receives and processes DisplayMedia messages."""
    
    def __init__(self, events_address: str = "tcp://localhost:5562"):  # Use unique port
        self.events_address = events_address
        self.subscriber = None
        self.running = False
        self.received_display_media = []  # Store for verification
        self.loaded_images = []           # Store loaded images for verification
        
    async def start(self):
        """Start the mock display service."""
        # Subscriber for DisplayMedia messages
        self.subscriber = ZmqSubscriber(self.events_address, [MessageType.DISPLAY_MEDIA], use_asyncio=True)
        
        self.running = True
        logger.info("Mock Display Service started")
        
    async def stop(self):
        """Stop the mock display service."""
        self.running = False
        if self.subscriber:
            self.subscriber.close()
        logger.info("Mock Display Service stopped")
        
    async def listen_for_display_media(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Listen for DisplayMedia messages and process them."""
        try:
            topic, message = await asyncio.wait_for(
                self.subscriber.receive_async(),
                timeout=timeout
            )
            
            if message.get("type") == MessageType.DISPLAY_MEDIA.value:
                self.received_display_media.append(message)
                await self._handle_display_media(message)
                return message
                
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for DisplayMedia message")
            return None
        except Exception as e:
            logger.error(f"Error listening for DisplayMedia: {e}")
            return None
            
    async def _handle_display_media(self, message: Dict[str, Any]):
        """Handle DisplayMedia by loading and processing the image."""
        content_type = message.get("content_type")
        
        if content_type == ContentType.IMAGE.value:
            # Load the image using modern enum-based utilities
            try:
                # Simulate pyglet image loading
                image_uri = message.get("uri")
                if image_uri and image_uri.startswith("file://"):
                    file_path = image_uri[7:]  # Remove file:// prefix
                    
                    # Load as PIL Image to verify content
                    pil_image = Image.open(file_path)
                    self.loaded_images.append({
                        "image": pil_image,
                        "era": message.get("era"),
                        "biome": message.get("biome"),
                        "fade_in": message.get("fade_in"),
                        "fade_out": message.get("fade_out")
                    })
                    
                    logger.info(f"Mock Display: Loaded image {pil_image.size} for {message.get('era')}/{message.get('biome')}")
                    
            except Exception as e:
                logger.error(f"Error loading DisplayMedia image: {e}")


@pytest.mark.asyncio
class TestFullPipelineIntegration:
    """Integration tests for the full Core → Image Server → Core → Display pipeline."""
    
    async def test_complete_render_flow_single_image(self):
        """Test complete flow: RenderRequest → ImageReady → DisplayMedia for single image."""
        # Create services with different ports to avoid conflicts
        image_server = MockImageServer("tcp://localhost:5570")
        core_service = MockCoreService("tcp://localhost:5571") 
        display_service = MockDisplayService("tcp://localhost:5572")
        
        try:
            # Start all services
            await image_server.start()
            await core_service.start()  
            await display_service.start()
            
            # Start image server listening loop
            server_task = asyncio.create_task(image_server.run())
            
            # Give services time to connect
            await asyncio.sleep(0.2)
            
            # 1. Core publishes RenderRequest
            request_id = await core_service.publish_render_request(era="modern", biome="urban")
            
            # 2. Wait for Image Server to generate and publish ImageReady
            await asyncio.sleep(0.5)  # Give image server time to process
            
            # 3. Core listens for ImageReady and publishes DisplayMedia
            image_ready = await core_service.listen_for_image_ready(timeout=2.0)
            assert image_ready is not None, "Should receive ImageReady message"
            assert image_ready["request_id"] == request_id
            
            # 4. Display listens for and processes DisplayMedia
            display_media = await display_service.listen_for_display_media(timeout=2.0)
            assert display_media is not None, "Should receive DisplayMedia message"
            
            # Verify the complete flow
            assert len(image_server.generated_images) == 1, "Image server should generate 1 image"
            assert len(core_service.received_image_ready) == 1, "Core should receive 1 ImageReady"
            assert len(core_service.published_display_media) == 1, "Core should publish 1 DisplayMedia"
            assert len(display_service.received_display_media) == 1, "Display should receive 1 DisplayMedia"
            assert len(display_service.loaded_images) == 1, "Display should load 1 image"
            
            # Verify message content consistency
            generated_image = list(image_server.generated_images.values())[0]
            received_display_media = core_service.published_display_media[0]
            loaded_image_info = display_service.loaded_images[0]
            
            assert generated_image["era"] == "modern"
            assert generated_image["biome"] == "urban"
            assert received_display_media["era"] == "modern"
            assert received_display_media["biome"] == "urban"
            assert loaded_image_info["era"] == "modern" 
            assert loaded_image_info["biome"] == "urban"
            
            # Verify transition properties
            assert received_display_media["content_type"] == ContentType.IMAGE.value
            assert received_display_media["fade_in"] == 0.5
            assert received_display_media["fade_out"] == 0.3
            assert loaded_image_info["fade_in"] == 0.5
            
            # Stop server task
            server_task.cancel()
            
        finally:
            # Clean up
            await image_server.stop()
            await core_service.stop()
            await display_service.stop()
            
            # Clean up generated files
            for image_info in image_server.generated_images.values():
                try:
                    Path(image_info["file_path"]).unlink()
                except:
                    pass
                    
    async def test_multiple_era_transitions(self):
        """Test multiple era transitions with different images."""
        image_server = MockImageServer()
        core_service = MockCoreService()
        display_service = MockDisplayService()
        
        eras_to_test = [
            ("wilderness", "temperate_forest"),
            ("modern", "urban"),
            ("ai_future", "coastal")
        ]
        
        try:
            await image_server.start()
            await core_service.start()
            await display_service.start()
            
            server_task = asyncio.create_task(image_server.run())
            await asyncio.sleep(0.2)
            
            request_ids = []
            
            for era, biome in eras_to_test:
                # Publish RenderRequest
                request_id = await core_service.publish_render_request(era=era, biome=biome)
                request_ids.append(request_id)
                
                # Wait for processing
                await asyncio.sleep(0.3)
                
                # Listen for responses
                image_ready = await core_service.listen_for_image_ready(timeout=2.0)
                assert image_ready is not None, f"Should receive ImageReady for {era}/{biome}"
                
                display_media = await display_service.listen_for_display_media(timeout=2.0)
                assert display_media is not None, f"Should receive DisplayMedia for {era}/{biome}"
                
            # Verify all transitions processed
            assert len(image_server.generated_images) == 3
            assert len(core_service.received_image_ready) == 3
            assert len(core_service.published_display_media) == 3
            assert len(display_service.received_display_media) == 3
            assert len(display_service.loaded_images) == 3
            
            # Verify era/biome variety in generated images
            generated_eras = [img["era"] for img in image_server.generated_images.values()]
            generated_biomes = [img["biome"] for img in image_server.generated_images.values()]
            
            assert "wilderness" in generated_eras
            assert "modern" in generated_eras
            assert "ai_future" in generated_eras
            assert "temperate_forest" in generated_biomes
            assert "urban" in generated_biomes
            assert "coastal" in generated_biomes
            
            server_task.cancel()
            
        finally:
            await image_server.stop()
            await core_service.stop()
            await display_service.stop()
            
            # Clean up generated files
            for image_info in image_server.generated_images.values():
                try:
                    Path(image_info["file_path"]).unlink()
                except:
                    pass
                    
    async def test_error_handling_missing_image_server(self):
        """Test that core service handles missing image server gracefully."""
        core_service = MockCoreService()
        display_service = MockDisplayService()
        
        try:
            await core_service.start()
            await display_service.start()
            await asyncio.sleep(0.2)
            
            # Publish RenderRequest but no image server to respond
            request_id = await core_service.publish_render_request(era="modern", biome="urban")
            
            # Should timeout waiting for ImageReady
            image_ready = await core_service.listen_for_image_ready(timeout=1.0)
            assert image_ready is None, "Should timeout when no image server responds"
            
            # No DisplayMedia should be published
            assert len(core_service.published_display_media) == 0
            
        finally:
            await core_service.stop()
            await display_service.stop()


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])

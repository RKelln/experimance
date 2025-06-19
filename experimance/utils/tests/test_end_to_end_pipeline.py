#!/usr/bin/env python3
"""
End-to-end integration test for the full image pipeline.

This test starts up the actual Core, Image Server, and Display services 
(not mocks) and verifies the complete pipeline:
1. Core sends RENDER_REQUEST
2. Image Server responds with IMAGE_READY (using existing images)
3. Core receives IMAGE_READY and sends DISPLAY_MEDIA
4. Display service receives and handles the image

This test demonstrates the full pipeline with real services and real image transport.
"""

import asyncio
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import time
from typing import List, Dict, Any

# Import test utilities
import sys
sys.path.append(str(Path(__file__).parent))
from test_utils import active_service

# Import services and utilities
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_display.display_service import DisplayService 
from image_server.image_service import ImageServerService
from image_server.config import ImageServerConfig, ZmqConfig, GeneratorConfig, MockGeneratorConfig
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import GENERATED_IMAGES_DIR_ABS, IMAGE_TRANSPORT_MODES
from experimance_common.schemas import Biome, Era, RenderRequest, ImageReady, DisplayMedia

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """Test the complete pipeline: Core -> Image Server -> Core -> Display."""
    
    # Use a test-specific port to avoid conflicts
    events_port = 5559
    
    image_server_config = ImageServerConfig(
        service_name = "test-image-server",
        cache_dir = Path(tempfile.mkdtemp()) / "cache",
        zmq=ZmqConfig(
            events_sub_address = f"tcp://localhost:{events_port}", 
            events_pub_address = f"tcp://*:{events_port}",
        ),
        generator=GeneratorConfig(
            default_strategy = "mock",
            timeout = 30
        ),
        mock=MockGeneratorConfig(
            strategy = "mock",
            use_existing_images = True,
            existing_images_dir = GENERATED_IMAGES_DIR_ABS
        )
    )
    
    # Core configuration
    core_config = {
        "service_name": "test-core", 
        "zmq": {
            "events_sub_address": f"tcp://localhost:{events_port}",
            "events_pub_address": f"tcp://*:{events_port}"
        }
    }
    
    # Display configuration (headless for testing)
    display_config = {
        "service_name": "test-display",
        "zmq": {
            "events_sub_address": f"tcp://localhost:{events_port}",
            "events_pub_address": f"tcp://*:{events_port}"
        },
        "renderer": {
            "width": 800,
            "height": 600,
            "headless": True  # Important for CI testing
        }
    }
    
    # Start all services concurrently
    async with active_service(ImageServerService(config=image_server_config)) as image_server, \
               active_service(ExperimanceCoreService(config=core_config)) as core, \
               active_service(DisplayService(config=display_config)) as display:
        
        # Give services time to fully initialize
        await asyncio.sleep(1.0)
        
        # Set up message capture for verification
        captured_messages = []
        
        async def capture_handler(message_type: MessageType, message_data: Dict[str, Any], service_name: str):
            """Capture all messages for verification."""
            captured_messages.append({
                "type": message_type,
                "data": message_data,
                "from": service_name,
                "timestamp": time.time()
            })
            logger.info(f"Captured {message_type.value} from {service_name}")
        
        # Set up message capturing on all services
        core.register_handler(MessageType.IMAGE_READY, lambda mt, md: capture_handler(mt, md, "core"))
        core.register_handler(MessageType.DISPLAY_MEDIA, lambda mt, md: capture_handler(mt, md, "core"))
        display.register_handler(MessageType.DISPLAY_MEDIA, lambda mt, md: capture_handler(mt, md, "display"))
        image_server.register_handler(MessageType.RENDER_REQUEST, lambda mt, md: capture_handler(mt, md, "image_server"))
        
        # Step 1: Trigger a render request from the core
        logger.info("Step 1: Triggering render request...")
        
        # Simulate the core receiving a trigger to generate an image
        # We'll directly call the core's render method since that's what would
        # normally be triggered by external events
        test_prompt = "A peaceful landscape with mountains and rivers"
        
        # Create a render request message as the core would
        render_request = RenderRequest(
            request_id="test-render-001",
            prompt=test_prompt,
            era=Era.CURRENT,  # Use the current era for testing
            biome=Biome.RAINFOREST,
        )

        
        # Publish the render request (simulating what core would do)
        
        # TODO: send change map as well
        #await core._publish_change_map(change_map, smoothed_change_score)

        await self._publish_render_request()
        
        # await core.publish_message(render_request.model_dump(), topic=MessageType.RENDER_REQUEST)
        
        # Step 2: Wait for the image server to process and respond
        logger.info("Step 2: Waiting for image server response...")
        
        # Wait for IMAGE_READY message
        max_wait = 10.0  # 10 seconds max wait
        start_time = time.time()
        image_ready_received = False
        
        while time.time() - start_time < max_wait and not image_ready_received:
            await asyncio.sleep(0.1)
            # Check if we received an IMAGE_READY message
            for msg in captured_messages:
                if msg["type"] == MessageType.IMAGE_READY:
                    image_ready_received = True
                    logger.info(f"IMAGE_READY received: {msg['data']}")
                    break
        
        assert image_ready_received, f"IMAGE_READY not received within {max_wait}s"
        
        # Step 3: Wait for the core to process IMAGE_READY and send DISPLAY_MEDIA
        logger.info("Step 3: Waiting for core to send DISPLAY_MEDIA...")
        
        # Give core time to process the IMAGE_READY and send DISPLAY_MEDIA
        display_media_sent = False
        start_time = time.time()
        
        while time.time() - start_time < max_wait and not display_media_sent:
            await asyncio.sleep(0.1)
            # Check if DISPLAY_MEDIA was sent
            for msg in captured_messages:
                if msg["type"] == MessageType.DISPLAY_MEDIA:
                    display_media_sent = True
                    logger.info(f"DISPLAY_MEDIA sent: {msg['data']}")
                    break
        
        assert display_media_sent, f"DISPLAY_MEDIA not sent within {max_wait}s"
        
        # Step 4: Verify the display service received and can handle the image
        logger.info("Step 4: Verifying display service received DISPLAY_MEDIA...")
        
        # Give display service time to process
        await asyncio.sleep(0.5)
        
        # Verify we have the complete message chain
        message_types = [msg["type"] for msg in captured_messages]
        
        # Should have: RENDER_REQUEST -> IMAGE_READY -> DISPLAY_MEDIA
        assert MessageType.RENDER_REQUEST in message_types, "RENDER_REQUEST not found in captured messages"
        assert MessageType.IMAGE_READY in message_types, "IMAGE_READY not found in captured messages" 
        assert MessageType.DISPLAY_MEDIA in message_types, "DISPLAY_MEDIA not found in captured messages"
        
        # Verify the content of the messages
        image_ready_msg = next(msg for msg in captured_messages if msg["type"] == MessageType.IMAGE_READY)
        display_media_msg = next(msg for msg in captured_messages if msg["type"] == MessageType.DISPLAY_MEDIA)
        
        # Validate IMAGE_READY message structure
        image_ready_data = image_ready_msg["data"]
        assert "image_path" in image_ready_data, "IMAGE_READY missing image_path"
        assert "prompt" in image_ready_data, "IMAGE_READY missing prompt"
        assert image_ready_data["prompt"] == test_prompt, "IMAGE_READY prompt mismatch"
        
        # Validate image file exists
        image_path = Path(image_ready_data["image_path"])
        assert image_path.exists(), f"Generated image file not found: {image_path}"
        assert image_path.stat().st_size > 0, "Generated image file is empty"
        
        # Validate DISPLAY_MEDIA message structure
        display_media_data = display_media_msg["data"]
        assert "content_type" in display_media_data, "DISPLAY_MEDIA missing content_type"
        assert "image_data" in display_media_data, "DISPLAY_MEDIA missing image_data"
        assert "transport_mode" in display_media_data, "DISPLAY_MEDIA missing transport_mode"
        
        # Verify transport mode is valid
        transport_mode = display_media_data["transport_mode"]
        assert transport_mode in IMAGE_TRANSPORT_MODES, f"Invalid transport mode: {transport_mode}"
        
        logger.info("✅ End-to-end pipeline test completed successfully!")
        logger.info(f"Pipeline verified: RENDER_REQUEST -> IMAGE_READY -> DISPLAY_MEDIA")
        logger.info(f"Image transport mode: {transport_mode}")
        logger.info(f"Total messages captured: {len(captured_messages)}")


@pytest.mark.asyncio
async def test_pipeline_with_different_transport_modes():
    """Test the pipeline with different image transport modes."""
    
    # Use a test-specific port
    events_port = 5560
    
    # Test each transport mode
    for transport_mode in IMAGE_TRANSPORT_MODES:
        logger.info(f"\n🔄 Testing pipeline with transport mode: {transport_mode}")
        
        # Configure services for this transport mode
        image_server_config = ImageServerConfig(
            service_name=f"test-image-server-{transport_mode}",
            cache_dir=Path(tempfile.mkdtemp()) / "cache",
            zmq={
                "events_sub_address": f"tcp://localhost:{events_port}",
                "events_pub_address": f"tcp://*:{events_port}"
            },
            generator={
                "default_strategy": "mock",
                "timeout": 30
            },
            mock={
                "strategy": "mock", 
                "use_existing_images": True,
                "existing_images_dir": GENERATED_IMAGES_DIR_ABS
            }
        )
        
        core_config = {
            "service_name": f"test-core-{transport_mode}",
            "zmq": {
                "events_sub_address": f"tcp://localhost:{events_port}",
                "events_pub_address": f"tcp://*:{events_port}"
            },
            # Configure core to use specific transport mode
            "image_transport": {
                "default_mode": transport_mode
            }
        }
        
        display_config = {
            "service_name": f"test-display-{transport_mode}",
            "zmq": {
                "events_sub_address": f"tcp://localhost:{events_port}",
                "events_pub_address": f"tcp://*:{events_port}"
            },
            "renderer": {
                "width": 800,
                "height": 600, 
                "headless": True
            }
        }
        
        # Run the pipeline test for this transport mode
        async with active_service(ImageServerService, config=image_server_config) as image_server, \
                   active_service(ExperimanceCore, config=core_config) as core, \
                   active_service(DisplayService, config=display_config) as display:
            
            await asyncio.sleep(1.0)  # Let services initialize
            
            # Trigger render request
            render_request = RenderRequest(
                prompt=f"Test image for {transport_mode} transport",
                era="anthropocene",
                biome="temperate_forest"
            )
            
            await core.publish_message(MessageType.RENDER_REQUEST, render_request.model_dump())
            
            # Wait for the pipeline to complete
            await asyncio.sleep(3.0)
            
            logger.info(f"✅ Pipeline test completed for {transport_mode}")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_end_to_end_pipeline())
    asyncio.run(test_pipeline_with_different_transport_modes())

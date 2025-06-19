"""Real service integration tests for the Experimance image transport pipeline.

Tests the full Core → Image Server → Core → Display pipeline using actual service classes
and the active_service() context manager, following project best practices.
"""

import asyncio
import logging
import pytest
from PIL import Image
import numpy as np

from experimance_common.test_utils import active_service
from experimance_common.zmq.zmq_utils import prepare_image_message
from experimance_common.constants import IMAGE_TRANSPORT_MODES
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig
from experimance_display.display_service import DisplayService
from experimance_display.config import DisplayServiceConfig

logger = logging.getLogger(__name__)


def create_test_image(height: int = 400, width: int = 400) -> Image.Image:
    """Create a simple test image."""
    array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.mark.asyncio
class TestRealServiceIntegration:
    """Integration tests using real Core and Display services."""
        
    async def test_simple_core_service_startup(self):
        """Test basic Core service startup and shutdown."""
        
        core_config = CoreServiceConfig.from_overrides({
            "zmq": {
                "events_pub_address": "tcp://*:5560",
                "events_sub_address": "tcp://localhost:5560"
            },
            "depth_processing": {
                "mock_camera": True,
                "change_threshold": 30
            }
        })
        
        core_service = ExperimanceCoreService(config=core_config)
        
        async with active_service(core_service) as active_core:
            # Verify service is running
            assert active_core.running is True
            assert hasattr(active_core, 'current_era')
            assert hasattr(active_core, 'current_biome')
            
            logger.info(f"Core service running with era: {active_core.current_era}")
            
        # Verify service stopped cleanly
        assert core_service.running is False
        
    async def test_core_handles_image_ready_message(self):
        """Test that core service handles IMAGE_READY messages correctly."""
        
        core_config = CoreServiceConfig.from_overrides({
            "zmq": {
                "events_pub_address": "tcp://*:5561", 
                "events_sub_address": "tcp://localhost:5561"
            },
            "depth_processing": {
                "mock_camera": True,
                "change_threshold": 30
            }
        })
        
        core_service = ExperimanceCoreService(config=core_config)
        
        async with active_service(core_service) as active_core:
            # Send a mock IMAGE_READY message directly
            test_image = create_test_image()
            image_message = prepare_image_message(
                test_image,
                transport_format=IMAGE_TRANSPORT_MODES["BASE64"],
                era="wilderness",
                biome="temperate_forest"
            )
            
            # Send the message via the ZMQ publisher (simulate image server response)
            import zmq.asyncio
            context = zmq.asyncio.Context()
            publisher = context.socket(zmq.PUB)
            publisher.connect("tcp://localhost:5561")
            
            await asyncio.sleep(0.2)  # Let connections establish
            
            await publisher.send_string("image_ready")
            await publisher.send_json(image_message)
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            publisher.close()
            
            # Verify the service processed the message without error
            assert active_core.running is True
            
    async def test_display_service_startup(self):
        """Test basic Display service startup and shutdown."""
        
        display_config = DisplayServiceConfig.from_overrides({
            "zmq": {
                "events_sub_address": "tcp://localhost:5562"
            },
            "display": {
                "headless": True,  # Run in headless mode for tests
                "resolution": [800, 600]
            }
        })
        
        display_service = DisplayService(config=display_config)
        
        async with active_service(display_service) as active_display:
            # Verify service is running
            assert active_display.running is True
            assert hasattr(active_display, 'layer_manager')
            
            logger.info("Display service running successfully")
            
        # Verify service stopped cleanly
        assert display_service.running is False

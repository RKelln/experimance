#!/usr/bin/env python3
"""
Test script for the Display Service.

This script tests the display service by creating mock messages
and testing both ZMQ and direct interface functionality.

NOTE: This test is skipped by default when running through pytest.
To run it manually, use: pytest -xvs tests/test_display_service.py::test_display_service
"""

import asyncio
import logging
import time
import pytest
from pathlib import Path
import sys

# Add the display service to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_common.test_utils import active_service
from experimance_display import DisplayService, DisplayServiceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Test requires a display. Run manually with: pytest -xvs tests/test_display_service.py::test_display_service")
async def test_display_service():
    """Test the display service with mock data."""
    
    # Create test configuration
    config = DisplayServiceConfig()
    config.display.fullscreen = False
    config.display.resolution = (800, 600)
    config.display.debug_overlay = True
    
    logger.info("Creating DisplayService...")
    service = DisplayService(config=config, service_name="test-display")
    
    async with active_service(service) as service:
        # Test text overlay
        logger.info("Testing text overlay...")
        text_message = {
            "type": "TextOverlay",
            "text_id": "test_text_1",
            "speaker": "agent",
            "content": "Hello! Welcome to Experimance. This is a test message.",
            "duration": 5.0
        }
        service.trigger_display_update("text_overlay", text_message)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test text replacement (streaming)
        logger.info("Testing text replacement...")
        text_update = {
            "type": "TextOverlay",
            "text_id": "test_text_1",  # Same ID to replace
            "speaker": "agent",
            "content": "Updated message: The system is working correctly!",
            "duration": 3.0
        }
        service.trigger_display_update("text_overlay", text_update)
        
        # Wait a bit more
        await asyncio.sleep(3)
        
        # Test system text
        logger.info("Testing system text...")
        system_message = {
            "type": "TextOverlay",
            "text_id": "system_status",
            "speaker": "system",
            "content": "System Status: Running normally",
            "duration": None  # Infinite duration
        }
        service.trigger_display_update("text_overlay", system_message)
        
        # Wait and then remove system text
        await asyncio.sleep(2)
        logger.info("Removing system text...")
        remove_message = {
            "type": "RemoveText",
            "text_id": "system_status"
        }
        service.trigger_display_update("remove_text", remove_message)
        
        # Test image if we have one
        test_image_path = Path("services/image_server/images/generated")
        if test_image_path.exists():
            image_files = list(test_image_path.glob("*.png")) + list(test_image_path.glob("*.jpg"))
            if image_files:
                logger.info(f"Testing image display with: {image_files[0]}")
                image_message = {
                    "type": "ImageReady",
                    "image_id": "test_image_1",
                    "uri": f"file://{image_files[0].absolute()}",
                    "request_id": "test_request_1"
                }
                service.trigger_display_update("image_ready", image_message)
        
        # Run for a while to see the display in action
        logger.info("Display service running... Press Ctrl+C to stop")
        await asyncio.sleep(10)
    

if __name__ == "__main__":
    # When running directly, ignore the pytest skip decorator
    pytest.skip = lambda *args, **kwargs: None  # No-op for direct execution
    
    try:
        asyncio.run(test_display_service())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

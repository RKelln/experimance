#!/usr/bin/env python3
"""
Test script for the Display Service with a real window.

This script demonstrates the direct interface for testing the display service
without requiring ZMQ infrastructure, but with a real window for visual inspection.

NOTE: This test is skipped by default when running through pytest.
To run it manually, use: pytest -xvs tests/test_display.py::test_display_service
"""

import asyncio
import logging
import time
import pytest
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pyglet after the try/except check in main to prevent errors
# when this module is imported by pytest without --no-headless flag
pyglet = None

from experimance_common.test_utils import active_service
from experimance_display.display_service import DisplayService
from experimance_display.config import DisplayServiceConfig, DisplayConfig


@pytest.mark.skip(reason="Test requires a display. Run manually with: pytest -xvs tests/test_display.py::test_display_service")
async def test_display_service():
    """Test the display service with direct interface and a real window."""
    global pyglet
    
    if pyglet is None:
        import pyglet
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Display Service test...")
    
    # Create test configuration
    config = DisplayServiceConfig(
        service_name="test-display",
        display=DisplayConfig(
            fullscreen=False,
            resolution=(800, 600),
            debug_overlay=True,
            vsync=False
        )
    )
    
    # Create service
    service = DisplayService(config=config)
    
    async with active_service(service):
        # Schedule some test updates
        pyglet.clock.schedule_once(schedule_test_text, 2.0, service)
        pyglet.clock.schedule_once(schedule_test_image, 4.0, service)
        pyglet.clock.schedule_once(schedule_more_text, 6.0, service)
        pyglet.clock.schedule_once(schedule_cleanup, 10.0, service)
        
        # Run pyglet main loop
        pyglet.app.run()


def schedule_test_text(dt, service):
    """Schedule test text overlay."""
    service.trigger_display_update("text_overlay", {
        "text_id": "test_text_1",
        "content": "Hello, Experimance!",
        "speaker": "agent",
        "duration": 5.0
    })
    print("Scheduled test text")


def schedule_test_image(dt, service):
    """Schedule test image (if available)."""
    # Try to find a test image
    test_image_paths = [
        "services/image_server/images/generated/test.png",
        "services/image_server/images/generated/test.jpg",
        "test_image.png",
        "test_image.jpg"
    ]
    
    for image_path in test_image_paths:
        if Path(image_path).exists():
            service.trigger_display_update("image_ready", {
                "image_id": "test_image_1",
                "uri": f"file://{Path(image_path).absolute()}"
            })
            print(f"Scheduled test image: {image_path}")
            return
    
    print("No test image found, skipping image test")


def schedule_more_text(dt, service):
    """Schedule more test text."""
    service.trigger_display_update("text_overlay", {
        "text_id": "test_text_2", 
        "content": "This is a system message",
        "speaker": "system",
        "duration": 3.0
    })
    print("Scheduled system text")


def schedule_cleanup(dt, service):
    """Schedule service cleanup."""
    # Instead of manually stopping, just exit the app
    # The active_service context manager will handle cleanup
    import pyglet
    pyglet.app.exit()
    print("Scheduled cleanup")


if __name__ == "__main__":
    # Check if we can import required modules
    try:
        import pyglet
        print(f"Pyglet version: {pyglet.version}")
    except ImportError as e:
        print(f"Cannot import pyglet: {e}")
        exit(1)
    
    # Run the test without pytest
    pytest.skip = lambda *args, **kwargs: None  # No-op for direct execution
    asyncio.run(test_display_service())

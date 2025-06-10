#!/usr/bin/env python3
"""
Integration tests for the Display Service with real window rendering.

These tests create actual windows and verify visual functionality.
Run these manually to see the display service in action.

NOTE: All tests are skipped by default when running through pytest.
To run them manually, use: pytest -xvs tests/test_integration.py::TestIntegration::test_full_sequence
"""

import asyncio
import logging
import time
import base64
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import shutil

# Add the display service to the path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_display.display_service import DisplayService
from experimance_display.config import DisplayServiceConfig, DisplayConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Integration test runner for visual testing."""
    temp_dir: Path

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_images = {}
        self.service = None
        
    def setup_test_resources(self):
        """Create temporary test resources."""
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temp directory: {self.temp_dir}")
        
        # Create test images
        self.create_test_images()
        
        # Create test video mask
        self.create_test_mask()
    
    def create_test_images(self):
        """Create test images for transitions."""
        # Red image
        red_img = Image.new('RGB', (800, 600), color='red')
        red_path = self.temp_dir / "red_landscape.png"
        red_img.save(red_path)
        self.test_images['red'] = red_path
        
        # Blue image
        blue_img = Image.new('RGB', (800, 600), color='blue')
        blue_path = self.temp_dir / "blue_landscape.png"
        blue_img.save(blue_path)
        self.test_images['blue'] = blue_path
        
        # Green image with pattern
        green_img = Image.new('RGB', (800, 600), color='green')
        # Add some pattern
        import numpy as np
        arr = np.array(green_img)
        arr[100:500, 100:700] = [0, 255, 0]  # Brighter green rectangle
        arr[200:400, 200:600] = [255, 255, 0]  # Yellow rectangle
        green_img = Image.fromarray(arr)
        green_path = self.temp_dir / "green_landscape.png"
        green_img.save(green_path)
        self.test_images['green'] = green_path
        
        logger.info(f"Created test images: {list(self.test_images.keys())}")
    
    def create_test_mask(self):
        """Create a test video mask."""
        # Create circular mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        center = (100, 100)
        radius = 80
        
        y, x = np.ogrid[:200, :200]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_circle] = 255
        
        # Convert to base64
        mask_image = Image.fromarray(mask, mode='L')
        import io
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        self.test_mask = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info("Created test video mask (circular)")
    
    def cleanup(self):
        """Clean up temporary resources."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temp directory")
    
    async def run_text_overlay_test(self):
        """Test text overlays with different speakers and positions."""
        logger.info("=== Testing Text Overlays ===")
        
        # Agent text (bottom center, with background)
        agent_message = {
            "text_id": "agent_welcome",
            "content": "Welcome to Experimance! I am your AI guide for this installation.",
            "speaker": "agent",
            "duration": 4.0
        }
        assert self.service is not None
        self.service.trigger_display_update("text_overlay", agent_message)
        await asyncio.sleep(2)
        
        # System text (top right, no background)
        system_message = {
            "text_id": "system_status",
            "content": "System Status: All services running",
            "speaker": "system",
            "duration": None  # Persistent
        }
        self.service.trigger_display_update("text_overlay", system_message)
        await asyncio.sleep(1)
        
        # Debug text (top left, yellow)
        debug_message = {
            "text_id": "debug_info",
            "content": "DEBUG: FPS=30.0, Memory=245MB",
            "speaker": "debug",
            "duration": None
        }
        self.service.trigger_display_update("text_overlay", debug_message)
        await asyncio.sleep(2)
        
        # Test text replacement (streaming text)
        logger.info("Testing text replacement...")
        for i in range(3):
            update_message = {
                "text_id": "agent_welcome",  # Same ID
                "content": f"Streaming update #{i+1}: The installation is responding to your presence...",
                "speaker": "agent",
                "duration": 2.0,
                "replace": True
            }
            self.service.trigger_display_update("text_overlay", update_message)
            await asyncio.sleep(1.5)
        
        # Remove system text
        logger.info("Removing system text...")
        remove_message = {"text_id": "system_status"}
        self.service.trigger_display_update("remove_text", remove_message)
        await asyncio.sleep(1)
        
        # Remove debug text
        remove_debug = {"text_id": "debug_info"}
        self.service.trigger_display_update("remove_text", remove_debug)
        await asyncio.sleep(1)
    
    async def run_image_transition_test(self):
        """Test image loading and crossfade transitions."""
        logger.info("=== Testing Image Transitions ===")
        
        # Load first image (red)
        logger.info("Loading red landscape...")
        red_message = {
            "image_id": "landscape_1",
            "uri": f"file://{self.test_images['red']}",
            "image_type": "satellite_landscape"
        }
        assert self.service is not None
        self.service.trigger_display_update("image_ready", red_message)
        await asyncio.sleep(2)
        
        # Add text overlay on the image
        overlay_text = {
            "text_id": "image_description",
            "content": "Red Landscape - Testing image display and text overlay",
            "speaker": "system",
            "duration": None
        }
        self.service.trigger_display_update("text_overlay", overlay_text)
        await asyncio.sleep(2)
        
        # Transition to blue image
        logger.info("Transitioning to blue landscape...")
        blue_message = {
            "image_id": "landscape_2",
            "uri": f"file://{self.test_images['blue']}",
            "image_type": "satellite_landscape",
            "transition_duration": 2.0  # 2 second crossfade
        }
        self.service.trigger_display_update("image_ready", blue_message)
        
        # Update text during transition
        update_text = {
            "text_id": "image_description",
            "content": "Blue Landscape - Crossfade transition in progress...",
            "speaker": "system",
            "duration": None,
            "replace": True
        }
        self.service.trigger_display_update("text_overlay", update_text)
        await asyncio.sleep(3)  # Wait for transition to complete
        
        # Final transition to green image
        logger.info("Transitioning to green landscape...")
        green_message = {
            "image_id": "landscape_3",
            "uri": f"file://{self.test_images['green']}",
            "image_type": "satellite_landscape",
            "transition_duration": 1.5
        }
        self.service.trigger_display_update("image_ready", green_message)
        
        final_text = {
            "text_id": "image_description",
            "content": "Green Landscape - Final image with pattern overlay",
            "speaker": "system",
            "duration": None,
            "replace": True
        }
        self.service.trigger_display_update("text_overlay", final_text)
        await asyncio.sleep(2)
        
        # Remove text overlay
        self.service.trigger_display_update("remove_text", {"text_id": "image_description"})
        await asyncio.sleep(1)
    
    async def run_video_overlay_test(self):
        """Test video overlay with dynamic masking."""
        logger.info("=== Testing Video Overlay ===")
        
        # Add explanatory text
        explanation = {
            "text_id": "video_explanation", 
            "content": "Testing video overlay with dynamic mask - simulating sand interaction",
            "speaker": "agent",
            "duration": 3.0
        }
        assert self.service is not None
        self.service.trigger_display_update("text_overlay", explanation)
        await asyncio.sleep(1)
        
        # Apply video mask (fade in)
        logger.info("Applying video mask (fade in)...")
        mask_message = {
            "mask_data": self.test_mask,
            "fade_in_duration": 1.0,
            "fade_out_duration": 2.0
        }
        self.service.trigger_display_update("video_mask", mask_message)
        await asyncio.sleep(2)
        
        # Update mask intensity (simulate changing sand interaction)
        logger.info("Updating mask (simulating sand movement)...")
        for i in range(3):
            # Create variations of the mask
            mask_variation = self.create_mask_variation(i)
            mask_update = {
                "mask_data": mask_variation,
                "fade_in_duration": 0.3,
                "fade_out_duration": 0.3
            }
            self.service.trigger_display_update("video_mask", mask_update)
            await asyncio.sleep(1.5)
        
        # Fade out video overlay
        logger.info("Fading out video overlay...")
        fadeout_message = {
            "mask_data": "",  # Empty mask = fade out
            "fade_in_duration": 0.1,
            "fade_out_duration": 1.5
        }
        self.service.trigger_display_update("video_mask", fadeout_message)
        await asyncio.sleep(2)
    
    def create_mask_variation(self, variation: int):
        """Create a variation of the test mask."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        
        if variation == 0:
            # Oval mask
            center = (100, 100)
            y, x = np.ogrid[:200, :200]
            mask_oval = ((x - center[0])/60)**2 + ((y - center[1])/80)**2 <= 1
            mask[mask_oval] = 255
        elif variation == 1:
            # Rectangle mask
            mask[50:150, 60:140] = 255
        else:
            # Multiple circles
            centers = [(70, 70), (130, 130), (70, 130), (130, 70)]
            for center in centers:
                y, x = np.ogrid[:200, :200]
                mask_circle = (x - center[0])**2 + (y - center[1])**2 <= 30**2
                mask[mask_circle] = 255
        
        # Convert to base64
        mask_image = Image.fromarray(mask, mode='L')
        import io
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def run_comprehensive_test(self):
        """Run all tests in sequence."""
        logger.info("Starting comprehensive integration test...")
        
        try:
            # Setup
            self.setup_test_resources()
            
            # Create service configuration
            config = DisplayServiceConfig(
                service_name="integration-test",
                display=DisplayConfig(
                    fullscreen=False,
                    resolution=(1000, 800),
                    debug_overlay=True,
                    vsync=True
                )
            )
            
            # Create and start service
            self.service = DisplayService(config=config)
            await self.service.start()
            logger.info("Display service started successfully")
            
            # Show welcome message
            welcome = {
                "text_id": "welcome",
                "content": "Integration Test Started - Testing all display features",
                "speaker": "system",
                "duration": 3.0
            }
            self.service.trigger_display_update("text_overlay", welcome)
            await asyncio.sleep(3)
            
            # Run test sequences
            await self.run_text_overlay_test()
            await asyncio.sleep(1)
            
            await self.run_image_transition_test()
            await asyncio.sleep(1)
            
            await self.run_video_overlay_test()
            await asyncio.sleep(1)
            
            # Final message
            final_msg = {
                "text_id": "complete",
                "content": "Integration Test Complete! Press ESC or Q to exit",
                "speaker": "agent",
                "duration": None
            }
            self.service.trigger_display_update("text_overlay", final_msg)
            
            # Keep running until user exits
            logger.info("Integration test complete. Window will remain open.")
            logger.info("Press ESC or Q in the window to exit, or Ctrl+C in terminal.")
            
            # Run until shutdown
            await self.service.run()
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
        finally:
            if self.service:
                await self.service.stop()
            self.cleanup()


async def run_quick_test():
    """Run a quick test with minimal setup."""
    logger.info("Running quick integration test...")
    
    config = DisplayServiceConfig(
        service_name="quick-test",
        display=DisplayConfig(
            fullscreen=False,
            resolution=(800, 600),
            debug_overlay=True,
        )
    )
    
    service = DisplayService(config=config)
    
    try:
        await service.start()
        
        # Quick text test
        test_msg = {
            "text_id": "quick_test",
            "content": "Quick Test - Display service is working!",
            "speaker": "agent", 
            "duration": 5.0
        }
        service.trigger_display_update("text_overlay", test_msg)
        
        logger.info("Quick test running for 5 seconds...")
        await asyncio.sleep(5)
        
        logger.info("Quick test completed successfully!")
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
    finally:
        await service.stop()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display Service Integration Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if args.quick:
        await run_quick_test()
    else:
        runner = IntegrationTestRunner()
        await runner.run_comprehensive_test()


# Convert functions to pytest compatible test classes
@pytest.mark.skip(reason="Test requires a display. Run manually with: pytest -xvs tests/test_integration.py::TestIntegration")
class TestIntegration:
    """Integration tests for display service with real windows."""
    
    @pytest.fixture(autouse=True)
    def setup_test_runner(self):
        """Set up test runner."""
        self.runner = IntegrationTestRunner()
        self.runner.setup_test_resources()
        yield
        self.runner.cleanup()
    
    @pytest.mark.skip(reason="Test requires a display")
    @pytest.mark.asyncio
    async def test_text_overlay(self):
        """Test text overlay rendering."""
        await self.runner.run_text_overlay_test()
    
    @pytest.mark.skip(reason="Test requires a display")
    @pytest.mark.asyncio
    async def test_image_transition(self):
        """Test image transition rendering."""
        await self.runner.run_image_transition_test()
    
    @pytest.mark.skip(reason="Test requires a display") 
    @pytest.mark.asyncio
    async def test_video_overlay(self):
        """Test video overlay with mask."""
        await self.runner.run_video_overlay_test()
    
    @pytest.mark.skip(reason="Test requires a display")
    @pytest.mark.asyncio 
    async def test_full_sequence(self):
        """Run the full comprehensive test."""
        await self.runner.run_comprehensive_test()


# For direct script execution
if __name__ == "__main__":
    # Allow direct execution without pytest
    asyncio.run(main())

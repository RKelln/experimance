#!/usr/bin/env python3
"""
Integration test for Feed the Fires project services.

This script tests the full pipeline of core, image_server, and display services
by sending StoryHeard messages and monitoring the complete flow.

Usage:
    # Set environment for fire project
    export PROJECT_ENV=fire
    
    # Run the test (make sure services are running first)
    uv run python utils/tests/test_fire_integration.py
    
Services to start before running this test:
    uv run -m fire_core
    uv run -m image_server  
    uv run -m experimance_display
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project schemas and common utilities
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig

# Import fire-specific schemas
try:
    from experimance_common.schemas import StoryHeard, MessageType, ImageReady, DisplayMedia
except ImportError:
    print("Error: Could not import schemas. Make sure PROJECT_ENV=fire is set.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FireTestRunner:
    """Test runner for fire integration tests."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.publisher_service = None
        self.image_subscriber_service = None
        self.display_subscriber_service = None
        self.test_data = []
        self.results = []
        self.running = True
        
        # Track received messages for validation
        self.received_images = {}
        self.received_displays = {}
        
    async def setup(self):
        """Set up ZMQ connections and load test data."""
        logger.info("Setting up test runner...")
        
        # Create publisher service to send StoryHeard messages to core's subscriber
        # Core subscribes to agent port for StoryHeard messages
        publisher_config = PubSubServiceConfig(
            publisher=PublisherConfig(
                address="tcp://localhost",
                port=DEFAULT_PORTS["agent"],
                default_topic=MessageType.STORY_HEARD
            )
        )
        self.publisher_service = PubSubService(publisher_config, "TestPublisher")
        await self.publisher_service.start()
        logger.info(f"Publisher connected to port {DEFAULT_PORTS['agent']} (core subscriber)")
        
        # Create subscriber service to monitor image generation
        image_subscriber_config = PubSubServiceConfig(
            subscriber=SubscriberConfig(
                address="tcp://localhost", 
                port=DEFAULT_PORTS["image_results"],
                topics=[MessageType.IMAGE_READY]
            )
        )
        self.image_subscriber_service = PubSubService(image_subscriber_config, "ImageSubscriber")
        self.image_subscriber_service.add_message_handler(MessageType.IMAGE_READY, self.handle_image_response)
        await self.image_subscriber_service.start()
        logger.info(f"Image subscriber connected to port {DEFAULT_PORTS['image_results']}")
        
        # Create subscriber service to monitor display messages
        display_subscriber_config = PubSubServiceConfig(
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=DEFAULT_PORTS["events"], 
                topics=[MessageType.DISPLAY_MEDIA]
            )
        )
        self.display_subscriber_service = PubSubService(display_subscriber_config, "DisplaySubscriber")
        self.display_subscriber_service.add_message_handler(MessageType.DISPLAY_MEDIA, self.handle_display_response)
        await self.display_subscriber_service.start()
        logger.info(f"Display subscriber connected to port {DEFAULT_PORTS['events']}")
        
        # Load test data
        await self.load_test_data()
        
        logger.info("Test runner setup complete")

        await asyncio.sleep(1)  # Ensure all services are ready
        
    async def load_test_data(self):
        """Load test stories from JSON file."""
        test_data_path = project_root / "services" / "core" / "tests" / "test_stories.json"
        
        if not test_data_path.exists():
            logger.error(f"Test data file not found: {test_data_path}")
            return
            
        try:
            with open(test_data_path, 'r') as f:
                data = json.load(f)
                
            # Filter out invalid entries
            for story in data:
                if story.get("prompt") != "<invalid>":
                    self.test_data.append(story)
                    
            logger.info(f"Loaded {len(self.test_data)} valid test stories")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            
    async def send_story_heard(self, story_data: Dict) -> str:
        """Send a StoryHeard message based on test data.
        
        Args:
            story_data: Dictionary containing story context and expected prompt
            
        Returns:
            The request ID for tracking this story
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Extract the user story from the context
        # Parse the context to find the user's story
        content = story_data.get("context", "")
        #content = self.extract_user_story(context)
        
        message = StoryHeard(
            content=content,
            speaker_id=f"test_speaker_{request_id[:8]}",
            confidence=0.95,
            timestamp=timestamp
        )
        
        logger.info(f"Sending StoryHeard message with ID {request_id}")
        logger.debug(f"Story content: {content}")
        
        # Send the message
        if self.publisher_service:
            await self.publisher_service.publish(message)
        else:
            raise RuntimeError("Publisher service not initialized")
        
        # Store for tracking
        self.results.append({
            "request_id": request_id,
            "timestamp": timestamp,
            "content": content,
            "expected_prompt": story_data.get("prompt", ""),
            "expected_negative": story_data.get("negative_prompt", ""),
            "image_received": False,
            "display_received": False
        })
        
        return request_id
    
    def extract_user_story(self, context: str) -> str:
        """Extract the user's story from the conversation context.
        
        Args:
            context: Full conversation context
            
        Returns:
            The user's story content
        """
        # Parse the context to find user statements
        lines = context.split('\n')
        user_parts = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('User: "') and line.endswith('"'):
                # Extract the quoted content
                user_content = line[7:-1]  # Remove 'User: "' and '"'
                user_parts.append(user_content)
                
        if user_parts:
            return " ".join(user_parts)
        else:
            # Fallback: return the entire context if we can't parse it
            return context
    
    async def handle_image_response(self, message):
        """Handle ImageReady messages."""
        if isinstance(message, dict):
            # Convert dict to ImageReady object for validation
            try:
                image_msg = ImageReady(**message)
            except Exception as e:
                logger.error(f"Invalid ImageReady message: {e}")
                return
        else:
            image_msg = message
            
        request_id = getattr(image_msg, 'request_id', None)
        image_uri = getattr(image_msg, 'uri', None)
        prompt = getattr(image_msg, 'prompt', None)
        
        logger.info(f"Received ImageReady: request_id={request_id}, uri={image_uri}")
        
        if request_id:
            self.received_images[request_id] = {
                "timestamp": datetime.now().isoformat(),
                "uri": image_uri,
                "prompt": prompt
            }
            
            # Update results
            for result in self.results:
                if result["request_id"] == request_id:
                    result["image_received"] = True
                    result["actual_prompt"] = prompt
                    result["image_uri"] = image_uri
                    break
    
    async def handle_display_response(self, message):
        """Handle DisplayMedia messages."""
        if isinstance(message, dict):
            # Convert dict to DisplayMedia object for validation
            try:
                display_msg = DisplayMedia(**message)
            except Exception as e:
                logger.error(f"Invalid DisplayMedia message: {e}")
                return
        else:
            display_msg = message
            
        content_type = getattr(display_msg, 'content_type', None)
        uri = getattr(display_msg, 'uri', None)
        
        logger.info(f"Received DisplayMedia: type={content_type}, uri={uri}")
        
        # Try to match with our tracked requests by URI
        for result in self.results:
            if result.get("image_uri") == uri:
                result["display_received"] = True
                result["display_timestamp"] = datetime.now().isoformat()
                break
    
    async def run_test_sequence(self, delay_between_stories: float = 10.0):
        """Run the complete test sequence.
        
        Args:
            delay_between_stories: Delay between sending each story in seconds
        """
        logger.info(f"Starting test sequence with {len(self.test_data)} stories")
        
        try:
            # Send all stories with delays
            for i, story in enumerate(self.test_data):
                logger.info(f"Sending story {i+1}/{len(self.test_data)}")
                await self.send_story_heard(story)
                
                if i < len(self.test_data) - 1:  # Don't delay after the last story
                    logger.info(f"Waiting {delay_between_stories}s before next story...")
                    await asyncio.sleep(delay_between_stories)
            
            # Wait for all responses (give extra time for processing)
            logger.info("All stories sent. Waiting for responses...")
            total_wait_time = len(self.test_data) * 60  # 60 seconds per story max
            await asyncio.sleep(total_wait_time)
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            self.running = False
    
    def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        total_stories = len(self.results)
        images_received = sum(1 for r in self.results if r["image_received"])
        displays_received = sum(1 for r in self.results if r["display_received"])
        
        logger.info(f"Total stories sent: {total_stories}")
        logger.info(f"Images received: {images_received}/{total_stories}")
        logger.info(f"Display messages received: {displays_received}/{total_stories}")
        logger.info(f"Complete pipeline success: {displays_received}/{total_stories}")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 60)
        
        for i, result in enumerate(self.results, 1):
            story_id = result["request_id"][:8]
            image_status = "✓" if result["image_received"] else "✗"
            display_status = "✓" if result["display_received"] else "✗"
            
            logger.info(f"Story {i} ({story_id}): Image {image_status} | Display {display_status}")
            logger.info(f"  Content: {result['content'][:60]}...")
            
            if result["image_received"]:
                logger.info(f"  Image URI: {result.get('image_uri', 'N/A')}")
                
            logger.info("")
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.publisher_service:
            await self.publisher_service.stop()
        if self.image_subscriber_service:
            await self.image_subscriber_service.stop()
        if self.display_subscriber_service:
            await self.display_subscriber_service.stop()
            
        logger.info("Cleanup complete")


async def main():
    """Main test function."""
    # Check environment
    if os.getenv("PROJECT_ENV") != "fire":
        logger.error("PROJECT_ENV must be set to 'fire'")
        logger.error("Run: export PROJECT_ENV=fire")
        return 1
    
    logger.info("Starting fire integration test")
    logger.info("Make sure these services are running:")
    logger.info("  - uv run -m fire_core")
    logger.info("  - uv run -m image_server")
    logger.info("  - uv run -m experimance_display")
    logger.info("")
    
    input("Press Enter to continue or Ctrl+C to cancel...\n")
    
    test_runner = FireTestRunner()
    
    try:
        await test_runner.setup()
        await test_runner.run_test_sequence(delay_between_stories=15.0)
        test_runner.print_results()
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    finally:
        await test_runner.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        sys.exit(0)

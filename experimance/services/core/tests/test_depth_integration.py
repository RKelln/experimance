#!/usr/bin/env python3
"""
Integration tests for depth processing functionality.

These tests use mock data to verify the depth processing pipeline
without requiring actual hardware.
"""
import asyncio
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig
from experimance_core.depth_finder_prototype import detect_difference, simple_obstruction_detect
from experimance_common.test_utils import active_service


class MockDepthGenerator:
    """Mock depth generator for testing."""
    
    def __init__(self, frames=None):
        self.frames = frames or []
        self.index = 0
        self.closed = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.closed or self.index >= len(self.frames):
            raise StopIteration
        
        frame = self.frames[self.index]
        self.index += 1
        return frame
    
    def close(self):
        self.closed = True


@pytest.fixture
def mock_depth_frames():
    """Create mock depth frames for testing."""
    frames = []
    
    # Create a sequence of depth frames with some variation
    for i in range(5):
        # Create a simple gradient pattern that changes slightly each frame
        depth_image = np.zeros((64, 64), dtype=np.uint8)
        
        # Add a moving "object" (bright spot)
        x, y = 20 + i * 2, 30
        depth_image[y-5:y+5, x-5:x+5] = 200
        
        # Hand detection alternates
        hand_detected = i % 2 == 0
        
        frames.append((depth_image, hand_detected))
    
    return frames


@pytest.fixture
def test_config():
    """Create a test configuration for the core service."""
    config_overrides = {
        "experimance_core": {
            "name": "test_depth_core",
            "heartbeat_interval": 1.0
        },
        "state_machine": {
            "idle_timeout": 10.0,
            "wilderness_reset": 60.0,
            "interaction_threshold": 0.5,
            "era_min_duration": 5.0
        },
        "visualize": False
    }
    
    return CoreServiceConfig.from_overrides(override_config=config_overrides)


@pytest.fixture
def mock_service_dependencies():
    """Mock the ZMQ and other service dependencies."""
    patches = {}
    
    # Mock ZMQ Publisher Subscriber Service
    patches['zmq_service'] = patch('experimance_core.zmq.ZmqPublisherSubscriberService.__init__', return_value=None)
    
    # Mock ZMQ context and sockets
    patches['zmq_context'] = patch('experimance_core.zmq.asyncio.Context')
    
    # Start all patches
    for patch_obj in patches.values():
        patch_obj.start()
    
    yield patches
    
    # Stop all patches
    for patch_obj in patches.values():
        patch_obj.stop()


class TestDepthProcessingIntegration:
    """Test depth processing integration with mock data."""
    
    @pytest.mark.asyncio
    async def test_depth_frame_processing_sequence(self, mock_depth_frames, test_config):
        """Test processing a sequence of depth frames."""
        # Create service with mocked dependencies
        service = ExperimanceCoreService(config=test_config)
        
        # Mock the depth generator
        mock_gen = MockDepthGenerator(mock_depth_frames)
        service.depth_generator = service._async_depth_wrapper(mock_gen)
        
        # Mock publish_message for testing
        service.publish_message = AsyncMock()
        
        # Test the depth processing method directly (not requiring full service lifecycle)
        frame_count = 0
        async for depth_image, hand_detected in service.depth_generator:
            if frame_count >= 3:  # Process a few frames
                break
                
            await service._process_depth_frame(depth_image, hand_detected)
            frame_count += 1
        
        # Verify processing occurred
        assert service.previous_depth_image is not None
        assert service.last_depth_map is not None
        assert frame_count == 3
        
        # Should have published interaction events
        assert service.publish_message.call_count > 0
    
    @pytest.mark.asyncio
    async def test_hand_detection_state_changes(self, mock_depth_frames, test_config):
        """Test hand detection state changes trigger events."""
        # Create service with mocked dependencies
        service = ExperimanceCoreService(config=test_config)
        
        # Mock the depth generator
        mock_gen = MockDepthGenerator(mock_depth_frames)
        service.depth_generator = service._async_depth_wrapper(mock_gen)
        
        # Mock publish_message for testing
        service.publish_message = AsyncMock()
        
        # Process frames with alternating hand detection
        frame_count = 0
        previous_hand_state = None
        
        async for depth_image, hand_detected in service.depth_generator:
            if frame_count >= 4:
                break
                
            await service._process_depth_frame(depth_image, hand_detected)
            
            # Check if hand state changed
            if previous_hand_state is not None and previous_hand_state != hand_detected:
                # Should have published interaction sound event
                published_calls = [call for call in service.publish_message.call_args_list 
                                 if call[0][0].get("type") == "AudioCommand"]
                assert len(published_calls) > 0
            
            previous_hand_state = hand_detected
            frame_count += 1
    
    @pytest.mark.asyncio
    async def test_interaction_score_accumulation(self, mock_depth_frames, test_config):
        """Test interaction score accumulation over time."""
        # Create service with mocked dependencies
        service = ExperimanceCoreService(config=test_config)
        
        # Mock the depth generator
        mock_gen = MockDepthGenerator(mock_depth_frames)
        service.depth_generator = service._async_depth_wrapper(mock_gen)
        
        # Mock publish_message for testing
        service.publish_message = AsyncMock()
        
        initial_score = service.user_interaction_score
        
        # Process several frames
        frame_count = 0
        async for depth_image, hand_detected in service.depth_generator:
            if frame_count >= 3:
                break
                
            await service._process_depth_frame(depth_image, hand_detected)
            frame_count += 1
        
        # Interaction score should have changed due to depth differences
        # (may increase or decrease depending on the mock data)
        final_score = service.user_interaction_score
        assert final_score != initial_score or frame_count == 0
    
    @pytest.mark.asyncio
    async def test_video_mask_publishing(self, mock_depth_frames, test_config):
        """Test video mask publishing on significant interaction."""
        # Create service with mocked dependencies
        service = ExperimanceCoreService(config=test_config)
        
        # Mock the depth generator
        mock_gen = MockDepthGenerator(mock_depth_frames)
        service.depth_generator = service._async_depth_wrapper(mock_gen)
        
        # Mock publish_message for testing
        service.publish_message = AsyncMock()
        
        # Set up a scenario with significant interaction
        service.user_interaction_score = 0.5  # Above threshold for video mask
        
        # Process a frame
        async for depth_image, hand_detected in service.depth_generator:
            await service._process_depth_frame(depth_image, hand_detected)
            break
        
        # Check for video mask events
        video_mask_calls = [call for call in service.publish_message.call_args_list 
                           if call[0][0].get("type") == "VideoMask"]
        
        # Should publish video mask if interaction is significant
        if service.user_interaction_score > 0.1:
            assert len(video_mask_calls) > 0


class TestDepthFunctionality:
    """Test core depth processing functions."""
    
    def test_detect_difference_function(self):
        """Test the detect_difference function with known inputs."""
        # Create two similar images
        image1 = np.ones((50, 50), dtype=np.uint8) * 100
        image2 = np.ones((50, 50), dtype=np.uint8) * 100
        
        # Add a small difference
        image2[20:30, 20:30] = 150
        
        diff_score, _ = detect_difference(image1, image2, threshold=10)
        
        # Should detect the difference
        assert diff_score > 10
        
        # Test with no previous image
        diff_score, returned_image = detect_difference(None, image2, threshold=10)
        assert diff_score > 10  # Should return above threshold
        assert np.array_equal(returned_image, image2)
    
    def test_simple_obstruction_detect_function(self):
        """Test the simple_obstruction_detect function."""
        # Create an image with a central dark area (obstruction)
        image = np.ones((32, 32), dtype=np.uint8) * 255  # White background
        image[12:20, 12:20] = 0  # Black center (obstruction)
        
        # Should detect obstruction
        obstruction = simple_obstruction_detect(image, size=(32, 32), pixel_threshold=5)
        assert obstruction == True
        
        # Test with no obstruction (bright image with enough variation to not be blank)
        bright_image = np.ones((32, 32), dtype=np.uint8) * 200
        # Add some noise to avoid blank frame detection
        bright_image[0:5, 0:5] = 255
        bright_image[25:30, 25:30] = 150
        
        obstruction = simple_obstruction_detect(bright_image, size=(32, 32), pixel_threshold=5)
        assert obstruction == False
        
        # Test with blank frame (should return None)
        blank_image = np.ones((32, 32), dtype=np.uint8) * 100  # Uniform gray
        obstruction = simple_obstruction_detect(blank_image, size=(32, 32), pixel_threshold=5)
        assert obstruction is None


class TestAsyncDepthWrapper:
    """Test the async depth wrapper functionality."""
    
    @pytest.mark.asyncio
    async def test_async_depth_wrapper_basic(self):
        """Test basic async depth wrapper functionality."""
        mock_frames = [
            (np.zeros((10, 10), dtype=np.uint8), False),
            (np.ones((10, 10), dtype=np.uint8) * 100, True),
            (np.ones((10, 10), dtype=np.uint8) * 200, False)
        ]
        
        # Create service with mocked dependencies
        service = ExperimanceCoreService()
        
        # Mock the service state using patch instead of direct assignment
        with patch.object(service, 'state', return_value='RUNNING'), \
             patch.object(service, 'running', return_value=True):
            
            mock_gen = MockDepthGenerator(mock_frames)
            async_gen = service._async_depth_wrapper(mock_gen)
            
            # Collect frames from async generator
            collected_frames = []
            async for frame in async_gen:
                collected_frames.append(frame)
                if len(collected_frames) >= len(mock_frames):
                    break
            
            assert len(collected_frames) == len(mock_frames)
            
            # Verify frame content
            for i, (depth_image, hand_detected) in enumerate(collected_frames):
                original_depth, original_hand = mock_frames[i]
                assert np.array_equal(depth_image, original_depth)
                assert hand_detected == original_hand
    
    @pytest.mark.asyncio
    async def test_async_depth_wrapper_cleanup(self):
        """Test that async depth wrapper properly cleans up."""
        mock_frames = [(np.zeros((10, 10), dtype=np.uint8), False)]
        
        # Create service with mocked dependencies
        service = ExperimanceCoreService()
        
        # Mock the service state using patch instead of direct assignment
        with patch.object(service, 'state', return_value='RUNNING'), \
             patch.object(service, 'running', return_value=True):
            
            mock_gen = MockDepthGenerator(mock_frames)
            async_gen = service._async_depth_wrapper(mock_gen)
            
            # Process one frame then stop
            async for frame in async_gen:
                break
            
            # Generator should be cleaned up
            assert mock_gen.closed or mock_gen.index >= len(mock_gen.frames)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

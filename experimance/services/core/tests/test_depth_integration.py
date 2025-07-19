#!/usr/bin/env python3
"""
Integration tests for depth processing functionality.

These tests use mock data to verify the depth processing pipeline
without requiring actual hardware.
"""
import asyncio
import cv2
import numpy as np
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig, CameraConfig, DepthFrame
from experimance_core.depth_processor import DepthProcessor
from experimance_core.mock_depth_processor import MockDepthProcessor
from experimance_core.depth_utils import detect_difference, simple_obstruction_detect
from experimance_common.test_utils import active_service
from experimance_common.zmq.config import ControllerServiceConfig, PublisherConfig, SubscriberConfig
from unittest.mock import AsyncMock, MagicMock, patch, Mock


@pytest.fixture
def camera_config():
    """Create a test camera configuration."""
    return CameraConfig(
        resolution=(640, 480),
        fps=30,
        output_resolution=(256, 256),  # Smaller for faster testing
        change_threshold=50,
        detect_hands=True,
        crop_to_content=False,  # Disable for simpler testing
        lightweight_mode=True,  # Faster processing
        verbose_performance=False,
        debug_mode=True  # Enable debug images for verification
    )


@pytest.fixture
def test_config(camera_config):
    """Create a test configuration for the core service."""
    config_overrides = {
        "experimance_core": {
            "name": "test_depth_core"
        },
        "state_machine": {
            "idle_timeout": 10.0,
            "wilderness_reset": 60.0,
            "interaction_threshold": 0.5,
            "era_min_duration": 5.0
        },
        "depth_processing": {
            "change_threshold": 50,
            "camera_config_path": "",
            "resolution": [640, 480],
            "fps": 30,
            "output_size": [256, 256]
        },
        "visualize": False
    }
    
    return CoreServiceConfig.from_overrides(override_config=config_overrides)


@pytest.fixture
def mock_depth_processor(camera_config):
    """Create a mock depth processor for testing."""
    return MockDepthProcessor(camera_config)


class TestDepthProcessorUnit:
    """Test depth processor unit functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_depth_processor_initialization(self, camera_config):
        """Test mock depth processor initializes correctly."""
        processor = MockDepthProcessor(camera_config)
        
        # Should not be initialized yet
        assert not processor.is_initialized
        
        # Initialize
        success = await processor.initialize()
        assert success
        assert processor.is_initialized
        
        # Clean up
        processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_depth_processor_frame_generation(self, camera_config):
        """Test mock depth processor generates valid frames."""
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        try:
            # Get a single frame
            frame = await processor.get_processed_frame()
            
            assert frame is not None
            assert isinstance(frame, DepthFrame)
            assert frame.depth_image is not None
            assert frame.depth_image.shape == camera_config.output_resolution
            assert frame.hand_detected is not None
            assert isinstance(frame.hand_detected, bool)
            assert frame.frame_number > 0
            assert frame.timestamp > 0
            
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_depth_processor_frame_streaming(self, camera_config):
        """Test mock depth processor streams frames correctly."""
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        frames_received = []
        
        try:
            # Stream a few frames
            async for frame in processor.stream_frames():
                frames_received.append(frame)
                if len(frames_received) >= 3:
                    break
            
            # Verify we got the expected number of frames
            assert len(frames_received) == 3
            
            # Verify frame numbers increment
            frame_numbers = [frame.frame_number for frame in frames_received]
            assert frame_numbers == [1, 2, 3]
            
            # Verify frames are distinct
            for i in range(1, len(frames_received)):
                # Images should be different (random generation)
                assert not np.array_equal(
                    frames_received[i-1].depth_image, 
                    frames_received[i].depth_image
                )
                
        finally:
            processor.stop()


class TestDepthProcessingIntegration:
    """Test depth processing integration with core service."""
    
    def create_mocked_service(self, test_config):
        """Create a properly mocked test service for depth integration tests."""
        # Create mock ZMQ service
        mock_zmq_service = Mock()
        mock_zmq_service.start = AsyncMock()
        mock_zmq_service.stop = AsyncMock()
        mock_zmq_service.publish = AsyncMock()
        mock_zmq_service.send_work_to_worker = AsyncMock()
        mock_zmq_service.add_message_handler = Mock()
        mock_zmq_service.add_response_handler = Mock()
        
        # Patch and create service
        patcher = patch('experimance_core.experimance_core.ControllerService')
        mock_controller_class = patcher.start()
        mock_controller_class.return_value = mock_zmq_service
        
        service = ExperimanceCoreService(config=test_config)
        
        # Mock essential methods
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        
        # Store patcher and mock for test access
        service._mock_patcher = patcher
        service._mock_zmq_service = mock_zmq_service
        
        return service
    
    @pytest.mark.asyncio
    async def test_depth_frame_processing_with_mock_processor(self, test_config, camera_config):
        """Test depth frame processing using mock processor directly."""
        # Create a mock processor
        mock_processor = MockDepthProcessor(camera_config)
        await mock_processor.initialize()
        
        try:
            # Get a few frames and process them
            frames = []
            async for frame in mock_processor.stream_frames():
                frames.append(frame)
                if len(frames) >= 3:
                    break
            
            # Verify we got frames
            assert len(frames) == 3
            
            # Verify frames have expected properties
            for frame in frames:
                assert isinstance(frame, DepthFrame)
                assert frame.depth_image is not None
                assert frame.depth_image.shape == camera_config.output_resolution
                assert frame.hand_detected is not None
                assert frame.timestamp > 0
                
            # Frames should be different due to random generation
            assert not np.array_equal(frames[0].depth_image, frames[1].depth_image)
            
        finally:
            mock_processor.stop()
    
    @pytest.mark.asyncio 
    async def test_core_service_depth_frame_processing(self, test_config, camera_config):
        """Test core service processes depth frames correctly."""
        # Mock the depth processor creation in core service
        with patch('experimance_core.experimance_core.create_depth_processor') as mock_factory:
            mock_processor = MockDepthProcessor(camera_config)
            mock_factory.return_value = mock_processor
            
            service = self.create_mocked_service(test_config)
            
            try:
                # Test direct frame processing
                test_frame = DepthFrame(
                    depth_image=np.ones((256, 256), dtype=np.uint8) * 128,
                    hand_detected=False,
                    change_score=0.5,
                    frame_number=1,
                    timestamp=time.time()
                )
                
                # Process the frame
                await service._process_depth_frame(test_frame)
                
                # Verify frame was processed (should update previous_depth_image)
                assert service.previous_depth_image is not None
                assert np.array_equal(service.previous_depth_image, test_frame.depth_image)
            finally:
                if hasattr(service, '_mock_patcher'):
                    service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_hand_detection_triggers_interaction_sound(self, test_config, camera_config):
        """Test that hand detection changes trigger interaction sound events."""
        with patch('experimance_core.experimance_core.create_depth_processor') as mock_factory:
            mock_processor = MockDepthProcessor(camera_config)
            mock_factory.return_value = mock_processor
            
            service = self.create_mocked_service(test_config)
            
            try:
                # Test hand detection state change
                frame_with_hands = DepthFrame(
                    depth_image=np.ones((256, 256), dtype=np.uint8) * 100,
                    hand_detected=True,
                    change_score=0.3,
                    frame_number=1,
                    timestamp=time.time()
                )
                
                frame_without_hands = DepthFrame(
                    depth_image=np.ones((256, 256), dtype=np.uint8) * 120,
                    hand_detected=False,
                    change_score=0.1,
                    frame_number=2,
                    timestamp=time.time()
                )
                
                # Process frames to trigger hand state changes
                await service._process_depth_frame(frame_with_hands)
                initial_calls = service._mock_zmq_service.publish.call_count
                
                await service._process_depth_frame(frame_without_hands)
                final_calls = service._mock_zmq_service.publish.call_count
                
                # Should have published at least one additional message for the state change
                assert final_calls > initial_calls
            finally:
                if hasattr(service, '_mock_patcher'):
                    service._mock_patcher.stop()


class TestDepthUtilityFunctions:
    """Test core depth processing utility functions."""
    
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


class TestDepthFrameDataStructure:
    """Test DepthFrame data structure."""
    
    def test_depth_frame_creation(self):
        """Test DepthFrame creation and properties."""
        test_image = np.ones((128, 128), dtype=np.uint8) * 100
        
        frame = DepthFrame(
            depth_image=test_image,
            hand_detected=True,
            change_score=0.8,
            frame_number=42,
            timestamp=1234567890.0
        )
        
        assert np.array_equal(frame.depth_image, test_image)
        assert frame.hand_detected == True
        assert frame.change_score == 0.8
        assert frame.frame_number == 42
        assert frame.timestamp == 1234567890.0
    
    def test_depth_frame_interaction_property(self):
        """Test DepthFrame has_interaction property."""
        test_image = np.zeros((64, 64), dtype=np.uint8)
        
        # Test with hand detected
        frame_with_hand = DepthFrame(
            depth_image=test_image,
            hand_detected=True,
            change_score=0.05
        )
        assert frame_with_hand.has_interaction == True
        
        # Test with high change score
        frame_with_change = DepthFrame(
            depth_image=test_image,
            hand_detected=False,
            change_score=0.5
        )
        assert frame_with_change.has_interaction == True
        
        # Test with no interaction
        frame_no_interaction = DepthFrame(
            depth_image=test_image,
            hand_detected=False,
            change_score=0.05
        )
        assert frame_no_interaction.has_interaction == False
    
    def test_depth_frame_debug_images(self):
        """Test DepthFrame debug image functionality."""
        test_image = np.ones((64, 64), dtype=np.uint8) * 128
        raw_image = np.ones((64, 64), dtype=np.uint8) * 200
        
        # Frame without debug images
        frame_no_debug = DepthFrame(depth_image=test_image)
        assert frame_no_debug.has_debug_images == False
        
        # Frame with debug images
        frame_with_debug = DepthFrame(
            depth_image=test_image,
            raw_depth_image=raw_image
        )
        assert frame_with_debug.has_debug_images == True

class TestMockDepthProcessorAdvanced:
    """Test advanced MockDepthProcessor functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_processor_frame_rate_control(self, camera_config):
        """Test mock processor respects frame rate settings."""
        # Set specific FPS for testing
        camera_config.fps = 10  # 10 FPS = 100ms per frame
        
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        frame_times = []
        
        try:
            start_time = time.time()
            frame_count = 0
            
            async for frame in processor.stream_frames():
                frame_times.append(time.time())
                frame_count += 1
                
                if frame_count >= 3:
                    break
            
            # Check timing between frames (should be approximately 1/FPS seconds)
            if len(frame_times) >= 2:
                intervals = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
                expected_interval = 1.0 / camera_config.fps
                
                # Allow some tolerance (±50% due to async timing variations)
                for interval in intervals:
                    assert 0.05 <= interval <= 0.2  # Should be roughly 0.1s ±50%
                    
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_processor_with_debug_mode(self, camera_config):
        """Test mock processor generates debug images when enabled."""
        camera_config.debug_mode = True
        
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        try:
            frame = await processor.get_processed_frame()
            
            assert frame is not None
            assert frame.has_debug_images == True
            assert frame.raw_depth_image is not None
            assert frame.importance_mask is not None
            assert frame.masked_image is not None
            assert frame.hand_detection_image is not None
            
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_processor_realistic_hand_detection_patterns(self, camera_config):
        """Test mock processor generates deterministic hand detection patterns with 0% and 100% rates."""
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        try:
            # Test with 0% hand detection rate - should never detect hands
            processor.set_hand_detection_rate(0.0)
            
            hand_detections_zero = []
            frame_count = 0
            
            async for frame in processor.stream_frames():
                hand_detections_zero.append(frame.hand_detected)
                frame_count += 1
                
                if frame_count >= 10:
                    break
            
            # With 0% rate, should have no hand detections
            assert not any(hand_detections_zero), f"Expected no hands with 0% rate, got {hand_detections_zero}"
            
            # Test with 100% hand detection rate - should always detect hands
            processor.set_hand_detection_rate(1.0)
            
            hand_detections_hundred = []
            frame_count = 0
            
            async for frame in processor.stream_frames():
                hand_detections_hundred.append(frame.hand_detected)
                frame_count += 1
                
                if frame_count >= 10:
                    break
            
            # With 100% rate, should have all hand detections
            assert all(hand_detections_hundred), f"Expected all hands with 100% rate, got {hand_detections_hundred}"
            
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_processor_configurable_hand_detection_rate(self, camera_config):
        """Test mock processor respects configurable hand detection rate."""
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        try:
            # Test with very low hand detection rate
            processor.set_hand_detection_rate(0.0)
            
            hand_detections = []
            frame_count = 0
            
            async for frame in processor.stream_frames():
                hand_detections.append(frame.hand_detected)
                frame_count += 1
                
                if frame_count >= 10:
                    break
            
            # With 0% rate, should have no hand detections
            assert not any(hand_detections), f"Expected no hands with 0% rate, got {hand_detections}"
            
            # Test with high hand detection rate
            processor.set_hand_detection_rate(1.0)
            hand_detections_high = []
            frame_count = 0
            
            async for frame in processor.stream_frames():
                hand_detections_high.append(frame.hand_detected)
                frame_count += 1
                
                if frame_count >= 10:
                    break
            
            # With 100% rate, should have all hand detections
            assert all(hand_detections_high), f"Expected all hands with 100% rate, got {hand_detections_high}"
                
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_processor_deterministic_frame_sequence(self, camera_config):
        """Test mock processor can use deterministic frame sequences."""
        processor = MockDepthProcessor(camera_config)
        
        # Create a deterministic sequence
        test_frames = [
            (np.ones((64, 64), dtype=np.uint8) * 100, False),
            (np.ones((64, 64), dtype=np.uint8) * 150, True),
            (np.ones((64, 64), dtype=np.uint8) * 200, False),
        ]
        
        processor.set_frame_sequence(test_frames)
        await processor.initialize()
        
        try:
            collected_frames = []
            frame_count = 0
            
            async for frame in processor.stream_frames():
                collected_frames.append((frame.depth_image.copy(), frame.hand_detected))
                frame_count += 1
                
                if frame_count >= 6:  # Test sequence repetition
                    break
            
            # Verify we got the expected number of frames
            assert len(collected_frames) == 6
            
            # Verify hand detection pattern matches our deterministic sequence (repeated)
            expected_hand_pattern = [False, True, False, False, True, False]
            actual_hand_pattern = [hand for _, hand in collected_frames]
            assert actual_hand_pattern == expected_hand_pattern
            
            # Verify that frames follow the repeating pattern by checking some basic properties
            # (We can't check exact image equality due to processing pipeline, but we can check patterns)
            for i in range(3):  # Check first 3 frames
                frame1 = collected_frames[i]
                frame2 = collected_frames[i + 3]  # Should be the same in the repeated sequence
                
                # Hand detection should match
                assert frame1[1] == frame2[1], f"Hand detection mismatch at positions {i} and {i+3}"
                
                # Images should have the same mean (approximately, accounting for processing variations)
                mean1 = np.mean(frame1[0])
                mean2 = np.mean(frame2[0])
                assert abs(mean1 - mean2) < 5, f"Image means too different: {mean1} vs {mean2}"
                    
        finally:
            processor.stop()
    
    @pytest.mark.asyncio
    async def test_mock_processor_statistics(self, camera_config):
        """Test mock processor provides useful statistics."""
        processor = MockDepthProcessor(camera_config)
        await processor.initialize()
        
        try:
            # Get initial statistics
            initial_stats = processor.get_frame_statistics()
            assert initial_stats["total_frames"] == 0
            assert initial_stats["has_generator"] == True
            assert initial_stats["is_initialized"] == True
            
            # Generate some frames
            frame_count = 0
            async for frame in processor.stream_frames():
                frame_count += 1
                if frame_count >= 3:
                    break
            
            # Check updated statistics
            final_stats = processor.get_frame_statistics()
            assert final_stats["total_frames"] == 3
            
        finally:
            processor.stop()
    


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

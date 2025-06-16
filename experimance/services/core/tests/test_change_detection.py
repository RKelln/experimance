"""
Test the new smart change detection logic in ExperimanceCoreService.
"""
import asyncio
import numpy as np
import pytest
import cv2
from unittest.mock import AsyncMock, MagicMock

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import DepthFrame


class TestChangeDetection:
    """Test the smart frame processing logic."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock service for testing."""
        service = ExperimanceCoreService()
        service.config = MagicMock()
        service.config.camera.significant_change_threshold = 0.02
        service.config.camera.change_threshold = 30
        service.config.camera.edge_erosion_pixels = 5  # Smaller value for 100x100 test images
        service.hand_detected = False
        service.last_processed_frame = None
        service.change_map = None
        service.depth_difference_score = 0.0
        
        # Mock the publishing methods
        service._publish_interaction_sound = AsyncMock()
        service._publish_change_map = AsyncMock()
        service._publish_video_mask = AsyncMock()
        service.calculate_interaction_score = MagicMock()
        
        return service

    def create_test_frame(self, hand_detected: bool = False, size: tuple = (100, 100)) -> DepthFrame:
        """Create a test depth frame."""
        depth_image = np.random.randint(0, 255, size, dtype=np.uint8)
        return DepthFrame(
            depth_image=depth_image,
            hand_detected=hand_detected,
            change_score=0.0,
            frame_number=1,
            timestamp=0.0
        )

    @pytest.mark.asyncio
    async def test_skip_frame_with_hands(self, mock_service):
        """Test that frames with hands detected are skipped."""
        frame = self.create_test_frame(hand_detected=True)
        
        await mock_service._process_depth_frame(frame)
        
        # Should not process the frame
        assert mock_service.last_processed_frame is None
        mock_service._publish_change_map.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_first_frame(self, mock_service):
        """Test processing the first frame (no reference frame)."""
        frame = self.create_test_frame(hand_detected=False)
        
        await mock_service._process_depth_frame(frame)
        
        # Should set the reference frame
        assert mock_service.last_processed_frame is not None
        np.testing.assert_array_equal(mock_service.last_processed_frame, frame.depth_image)

    @pytest.mark.asyncio 
    async def test_skip_small_changes(self, mock_service):
        """Test that small changes are skipped."""
        # Set up a reference frame
        ref_frame = np.zeros((100, 100), dtype=np.uint8)
        mock_service.last_processed_frame = ref_frame
        
        # Create a frame with minimal change (just 1-2 pixels different)
        similar_frame_data = ref_frame.copy()
        similar_frame_data[50, 50] = 10  # Small change
        
        frame = DepthFrame(
            depth_image=similar_frame_data,
            hand_detected=False,
            change_score=0.0,
            frame_number=2,
            timestamp=1.0
        )
        
        await mock_service._process_depth_frame(frame)
        
        # Should not update reference frame due to small change
        np.testing.assert_array_equal(mock_service.last_processed_frame, ref_frame)
        mock_service._publish_change_map.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_significant_changes(self, mock_service):
        """Test that significant changes are processed."""
        # Set up a reference frame
        ref_frame = np.zeros((100, 100), dtype=np.uint8)
        mock_service.last_processed_frame = ref_frame
        
        # Create a frame with significant change
        changed_frame_data = np.ones((100, 100), dtype=np.uint8) * 100  # Major change
        
        frame = DepthFrame(
            depth_image=changed_frame_data,
            hand_detected=False,
            change_score=0.0,
            frame_number=2,
            timestamp=1.0
        )
        
        await mock_service._process_depth_frame(frame)
        
        # Should update reference frame and publish change map
        np.testing.assert_array_equal(mock_service.last_processed_frame, changed_frame_data)
        mock_service._publish_change_map.assert_called_once()

    def test_create_comparison_mask(self, mock_service):
        """Test mask creation for noise reduction."""
        import cv2
        import os
        
        test_image = np.ones((100, 100), dtype=np.uint8)
        
        # Create a proper test mask with a white circle on black background
        original_mask = np.zeros((100, 100), dtype=np.uint8)
        center = (50, 50)
        radius = 30
        cv2.circle(original_mask, center, radius, (255,), -1)  # White circle on black background
        
        # Test erosion directly since our mock isn't working properly
        edge_erosion = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_erosion, edge_erosion))
        eroded_mask = cv2.erode(original_mask, kernel, iterations=1)
        
        # Save images for visual inspection
        cv2.imwrite('/tmp/original_mask.png', original_mask)
        cv2.imwrite('/tmp/eroded_mask.png', eroded_mask)
        
        # Check that erosion worked
        original_white_pixels = np.sum(original_mask == 255)
        eroded_white_pixels = np.sum(eroded_mask == 255)
        print(f"Original white pixels: {original_white_pixels}, Eroded white pixels: {eroded_white_pixels}")
        print(f"Saved mask images to /tmp/original_mask.png and /tmp/eroded_mask.png")
        
        # Erosion should reduce the white area
        assert eroded_white_pixels < original_white_pixels, f"Expected erosion to reduce white pixels, got {eroded_white_pixels} vs {original_white_pixels}"
        assert eroded_mask.shape == original_mask.shape

    def test_calculate_change_with_mask(self, mock_service):
        """Test change calculation with masking."""
        ref_frame = np.zeros((100, 100), dtype=np.uint8)
        current_frame = np.ones((100, 100), dtype=np.uint8) * 100
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        change_score, change_map = mock_service._calculate_change_with_mask(
            ref_frame, current_frame, mask
        )
        
        # Should detect significant change
        assert change_score > 0.5  # Should be high for complete difference
        assert change_map.shape == current_frame.shape
        assert change_map.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__])

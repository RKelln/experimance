"""
Test suite for image utility functions.

This module tests the crop_to_content function and other image utilities
to ensure they handle various edge cases correctly.
"""
import pytest
import numpy as np
import cv2
from experimance_common.image_utils import crop_to_content


class TestCropToContent:
    """Test cases for the crop_to_content function."""
    
    def create_test_image(self, height, width, content_rect=None):
        """
        Create a test image with specified dimensions and optional content area.
        
        Args:
            height: Image height
            width: Image width
            content_rect: Tuple (x, y, w, h) for content area, or None for centered circle
        
        Returns:
            numpy array representing the image
        """
        image = np.zeros((height, width), dtype=np.uint8)
        
        if content_rect is None:
            # Create a centered circular content area
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(image, (center_x, center_y), radius, (255,), -1)
        else:
            # Create rectangular content area
            x, y, w, h = content_rect
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, -1)
        
        return image
    
    def test_square_image_to_square_target(self):
        """Test cropping a square image to a square target."""
        # 400x400 image with centered content → 256x256
        image = self.create_test_image(400, 400)
        result, bounds = crop_to_content(image, (256, 256))
        
        assert result.shape == (256, 256), f"Expected (256, 256), got {result.shape}"
        assert bounds is not None, "Bounds should not be None"
        
        # Bounds should be roughly square and centered
        x, y, w, h = bounds
        assert w == h, f"Crop should be square, got w={w}, h={h}"
    
    def test_wide_image_to_square_target(self):
        """Test cropping a wide image to a square target (like camera 1280x720 → 1024x1024)."""
        # Create 1280x720 image with content across most of the width
        image = self.create_test_image(720, 1280, content_rect=(100, 100, 1080, 520))
        result, bounds = crop_to_content(image, (1024, 1024))
        
        assert result.shape == (1024, 1024), f"Expected (1024, 1024), got {result.shape}"
        
        # Bounds should be square
        x, y, w, h = bounds
        assert w == h, f"Crop should be square, got w={w}, h={h}"
        
        # Should crop from the width (reduce width to match height)
        assert w <= 720, f"Crop width should be <= 720, got {w}"
    
    def test_tall_image_to_square_target(self):
        """Test cropping a tall image to a square target."""
        # Create 720x1280 image with content across most of the height
        image = self.create_test_image(1280, 720, content_rect=(100, 100, 520, 1080))
        result, bounds = crop_to_content(image, (512, 512))
        
        assert result.shape == (512, 512), f"Expected (512, 512), got {result.shape}"
        
        # Bounds should be square
        x, y, w, h = bounds
        assert w == h, f"Crop should be square, got w={w}, h={h}"
        
        # Should crop from the height (reduce height to match width)
        assert h <= 720, f"Crop height should be <= 720, got {h}"
    
    def test_square_image_to_wide_target(self):
        """Test cropping a square image to a wide target (16:9)."""
        # 800x800 image → 1920x1080 (16:9)
        image = self.create_test_image(800, 800)
        result, bounds = crop_to_content(image, (1920, 1080))
        
        assert result.shape == (1080, 1920), f"Expected (1080, 1920), got {result.shape}"
        
        # Bounds should have 16:9 aspect ratio
        x, y, w, h = bounds
        aspect_ratio = w / h
        target_aspect = 1920 / 1080
        assert abs(aspect_ratio - target_aspect) < 0.01, f"Aspect ratio should be ~{target_aspect:.3f}, got {aspect_ratio:.3f}"
    
    def test_square_image_to_tall_target(self):
        """Test cropping a square image to a tall target (9:16)."""
        # 800x800 image → 1080x1920 (9:16)
        image = self.create_test_image(800, 800)
        result, bounds = crop_to_content(image, (1080, 1920))
        
        assert result.shape == (1920, 1080), f"Expected (1920, 1080), got {result.shape}"
        
        # Bounds should have 9:16 aspect ratio
        x, y, w, h = bounds
        aspect_ratio = w / h
        target_aspect = 1080 / 1920
        assert abs(aspect_ratio - target_aspect) < 0.01, f"Aspect ratio should be ~{target_aspect:.3f}, got {aspect_ratio:.3f}"
    
    def test_with_existing_bounds(self):
        """Test using existing bounds (simulating locked mask scenario)."""
        image = self.create_test_image(720, 1280)
        
        # First call without bounds
        result1, bounds1 = crop_to_content(image, (1024, 1024))
        
        # Second call with the same bounds
        result2, bounds2 = crop_to_content(image, (1024, 1024), bounds=bounds1)
        
        assert result1.shape == result2.shape, "Results should have same shape"
        assert bounds1 == bounds2, "Bounds should be identical when reused"
        
        # Results should be very similar (allowing for minor floating point differences)
        diff = cv2.absdiff(result1, result2)
        max_diff = np.max(diff)
        assert max_diff <= 1, f"Results should be nearly identical, max diff was {max_diff}"
    
    def test_small_content_area(self):
        """Test with a very small content area."""
        image = self.create_test_image(720, 1280, content_rect=(600, 300, 50, 50))
        result, bounds = crop_to_content(image, (256, 256))
        
        assert result.shape == (256, 256), f"Expected (256, 256), got {result.shape}"
        
        # Should still create a valid square crop
        x, y, w, h = bounds
        assert w == h, f"Crop should be square, got w={w}, h={h}"
        assert w > 0 and h > 0, "Crop dimensions should be positive"
    
    def test_edge_content(self):
        """Test with content at the edge of the image."""
        # Content at the top-left corner
        image = self.create_test_image(720, 1280, content_rect=(0, 0, 200, 200))
        result, bounds = crop_to_content(image, (512, 512))
        
        assert result.shape == (512, 512), f"Expected (512, 512), got {result.shape}"
        
        x, y, w, h = bounds
        assert x >= 0 and y >= 0, "Crop coordinates should be non-negative"
        assert x + w <= 1280 and y + h <= 720, "Crop should not exceed image bounds"
    
    def test_full_image_content(self):
        """Test with content filling the entire image."""
        # Create image filled with content
        image = np.full((720, 1280), 128, dtype=np.uint8)
        result, bounds = crop_to_content(image, (1024, 1024))
        
        assert result.shape == (1024, 1024), f"Expected (1024, 1024), got {result.shape}"
        
        # Should crop to a square from the center
        x, y, w, h = bounds
        assert w == h, f"Crop should be square, got w={w}, h={h}"
        assert w == 720, f"Should crop to image height (720), got {w}"  # Limited by height
    
    def test_no_content(self):
        """Test with an empty image (no content)."""
        image = np.zeros((720, 1280), dtype=np.uint8)
        result, bounds = crop_to_content(image, (256, 256))
        
        # Should return a resized version of the original image
        assert result.shape == (256, 256), f"Expected (256, 256), got {result.shape}"
        
        # All pixels should be black
        assert np.max(result) == 0, "Result should be all black for empty image"
    
    def test_aspect_ratio_preservation(self):
        """Test that the crop maintains the target aspect ratio."""
        test_cases = [
            ((720, 1280), (1024, 1024)),  # Wide to square
            ((1280, 720), (512, 512)),    # Tall to square  
            ((800, 800), (1920, 1080)),   # Square to wide
            ((800, 800), (1080, 1920)),   # Square to tall
            ((1920, 1080), (256, 256)),   # Wide to square (downscale)
        ]
        
        for img_size, target_size in test_cases:
            image = self.create_test_image(img_size[0], img_size[1])
            result, bounds = crop_to_content(image, target_size)
            
            # Check output size
            assert result.shape == (target_size[1], target_size[0]), \
                f"For {img_size}→{target_size}: Expected {(target_size[1], target_size[0])}, got {result.shape}"
            
            # Check crop aspect ratio matches target
            x, y, w, h = bounds
            crop_aspect = w / h
            target_aspect = target_size[0] / target_size[1]
            
            assert abs(crop_aspect - target_aspect) < 0.01, \
                f"For {img_size}→{target_size}: Crop aspect {crop_aspect:.3f} should match target aspect {target_aspect:.3f}"
    
    def test_content_centering(self):
        """Test that cropped content is properly centered in the output."""
        # Create a 200x200 image with content at (20, 20, 60, 60) - center at (50, 50)
        image = np.zeros((200, 200), dtype=np.uint8)
        content_x, content_y = 20, 20
        content_w, content_h = 60, 60
        cv2.rectangle(image, (content_x, content_y), (content_x + content_w, content_y + content_h), 255, -1)
        
        # Crop to square
        result, bounds = crop_to_content(image, (100, 100))
        
        # Calculate centers
        original_content_center_x = content_x + content_w // 2  # Should be 50
        original_content_center_y = content_y + content_h // 2  # Should be 50
        
        crop_x, crop_y, crop_w, crop_h = bounds
        crop_center_x = crop_x + crop_w // 2
        crop_center_y = crop_y + crop_h // 2
        
        image_center_x = image.shape[1] // 2  # 100
        image_center_y = image.shape[0] // 2  # 100
        
        print(f"Original content center: ({original_content_center_x}, {original_content_center_y})")
        print(f"Crop bounds: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        print(f"Crop center: ({crop_center_x}, {crop_center_y})")
        print(f"Image center: ({image_center_x}, {image_center_y})")
        
        # The crop should be centered around the content center, not necessarily the image center
        # But if content is already centered, the crop should also be centered
        expected_crop_size = min(crop_w, crop_h)
        
        # Check if crop is roughly centered around the content
        content_to_crop_offset_x = abs(crop_center_x - original_content_center_x)
        content_to_crop_offset_y = abs(crop_center_y - original_content_center_y)
        
        # Allow some tolerance for rounding
        assert content_to_crop_offset_x <= 2, f"Crop should be centered on content X, offset: {content_to_crop_offset_x}"
        assert content_to_crop_offset_y <= 2, f"Crop should be centered on content Y, offset: {content_to_crop_offset_y}"
        
        assert result.shape == (100, 100), f"Expected (100, 100), got {result.shape}"
        
        # Content should be visible in the center of the result
        center_region = result[40:60, 40:60]  # Center 20x20 area
        assert np.max(center_region) > 0, "Content should be visible in center of result"
    
    def test_bounds_within_image(self):
        """Test that bounds are always within the image boundaries."""
        test_cases = [
            (720, 1280, (1024, 1024)),
            (480, 640, (512, 512)),
            (1080, 1920, (256, 256)),
        ]
        
        for height, width, target_size in test_cases:
            image = self.create_test_image(height, width)
            result, bounds = crop_to_content(image, target_size)
            
            x, y, w, h = bounds
            
            # Check bounds are within image
            assert x >= 0, f"For {(height, width)}→{target_size}: x should be >= 0, got {x}"
            assert y >= 0, f"For {(height, width)}→{target_size}: y should be >= 0, got {y}"
            assert x + w <= width, f"For {(height, width)}→{target_size}: x + w ({x + w}) should be <= width ({width})"
            assert y + h <= height, f"For {(height, width)}→{target_size}: y + h ({y + h}) should be <= height ({height})"
            
            # Check dimensions are positive
            assert w > 0, f"For {(height, width)}→{target_size}: width should be > 0, got {w}"
            assert h > 0, f"For {(height, width)}→{target_size}: height should be > 0, got {h}"
    
    def test_wide_content_should_have_letterboxing(self):
        """
        Test that wide rectangular content cropped to square should have black bars (letterboxing).
        
        This test specifically checks for the bug where wide content becomes entirely white
        instead of having proper letterboxing with black bars on top and bottom.
        """
        # Create a wide image with wide rectangular content (like a sand table)
        image = np.zeros((720, 1280), dtype=np.uint8)
        
        # Create wide rectangular content area (16:10 aspect ratio)
        content_x, content_y = 240, 160  # Centered horizontally, some padding vertically
        content_w, content_h = 800, 400  # Wide rectangle
        cv2.rectangle(image, (content_x, content_y), (content_x + content_w, content_y + content_h), 255, -1)
        
        # Crop to square output
        result, bounds = crop_to_content(image, (1024, 1024))
        
        # Basic checks
        assert result.shape == (1024, 1024), f"Expected (1024, 1024), got {result.shape}"
        
        # Critical check: The result should NOT be entirely white
        # If working correctly, there should be black pixels (letterboxing)
        white_pixels = cv2.countNonZero(result)
        total_pixels = result.shape[0] * result.shape[1]
        white_percentage = white_pixels / total_pixels
        
        print(f"White pixels: {white_pixels}/{total_pixels} ({white_percentage*100:.1f}%)")
        
        # The result should have less than 95% white pixels (allowing some tolerance)
        # If it's 100% white, then the function is broken
        assert white_percentage < 0.95, f"Result should have letterboxing, but {white_percentage*100:.1f}% is white (too much)"
        
        # Additionally, check that we have content in the middle and black bars on top/bottom
        middle_row = result[result.shape[0] // 2, :]
        top_row = result[50, :]  # Near top
        bottom_row = result[-50, :]  # Near bottom
        
        # Middle should have some white content
        middle_white = cv2.countNonZero(middle_row)
        assert middle_white > 0, "Middle row should have white content"
        
        # Top and bottom should have some black pixels (letterboxing)
        top_black = len(top_row) - cv2.countNonZero(top_row)
        bottom_black = len(bottom_row) - cv2.countNonZero(bottom_row)
        
        assert top_black > 0 or bottom_black > 0, "Should have black letterboxing on top or bottom"

    
    def test_tall_content_should_have_pillarboxing(self):
        """
        Test that tall rectangular content cropped to square should have black bars (pillarboxing).
        
        This test specifically checks for the bug where tall content becomes entirely white
        instead of having proper pillarboxing with black bars on left and right.
        """
        # Create a tall image with tall rectangular content (like a portrait sand table)
        image = np.zeros((1280, 720), dtype=np.uint8)
        
        # Create tall rectangular content area (10:16 aspect ratio)
        content_x, content_y = 160, 240  # Centered vertically, some padding horizontally
        content_w, content_h = 400, 800  # Tall rectangle
        cv2.rectangle(image, (content_x, content_y), (content_x + content_w, content_y + content_h), 255, -1)
        
        # Crop to square output
        result, bounds = crop_to_content(image, (1024, 1024))
        
        # Basic checks
        assert result.shape == (1024, 1024), f"Expected (1024, 1024), got {result.shape}"
        
        # Critical check: The result should NOT be entirely white
        # If working correctly, there should be black pixels (pillarboxing)
        white_pixels = cv2.countNonZero(result)
        total_pixels = result.shape[0] * result.shape[1]
        white_percentage = white_pixels / total_pixels
        
        print(f"White pixels: {white_pixels}/{total_pixels} ({white_percentage*100:.1f}%)")
        
        # The result should have less than 95% white pixels (allowing some tolerance)
        # If it's 100% white, then the function is broken
        assert white_percentage < 0.95, f"Result should have pillarboxing, but {white_percentage*100:.1f}% is white (too much)"
        
        # Additionally, check that we have content in the middle and black bars on left/right
        middle_col = result[:, result.shape[1] // 2]
        left_col = result[:, 50]  # Near left
        right_col = result[:, -50]  # Near right
        
        # Middle should have some white content
        middle_white = cv2.countNonZero(middle_col)
        assert middle_white > 0, "Middle column should have white content"
        
        # Left and right should have some black pixels (pillarboxing)
        left_black = len(left_col) - cv2.countNonZero(left_col)
        right_black = len(right_col) - cv2.countNonZero(right_col)
        
        assert left_black > 0 or right_black > 0, "Should have black pillarboxing on left or right"
        
    # ...existing code...


def test_debug_output():
    """Test the debug output functionality."""
    # This test verifies that debug prints don't crash the function
    image = np.zeros((720, 1280), dtype=np.uint8)
    cv2.circle(image, (640, 360), 100, (255,), -1)
    
    # Should not raise any exceptions
    result, bounds = crop_to_content(image, (512, 512))
    
    assert result.shape == (512, 512)
    assert bounds is not None


if __name__ == "__main__":
    # Run tests when executed directly
    import sys
    sys.path.append("../..")  # Add project root to path
    
    test_instance = TestCropToContent()
    
    print("Running crop_to_content tests...")
    
    # Run a few key tests manually
    try:
        test_instance.test_wide_image_to_square_target()
        print("✅ Wide to square test passed")
        
        test_instance.test_aspect_ratio_preservation()
        print("✅ Aspect ratio preservation test passed")
        
        test_instance.test_bounds_within_image()
        print("✅ Bounds validation test passed")
        
        print("\n🎉 All manual tests passed! Run with pytest for full suite.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

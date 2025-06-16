"""
Mock depth processor for testing without RealSense hardware.
"""

import asyncio
import cv2
import logging
import numpy as np
import time
from random import randint
from typing import Optional, Tuple, Generator, AsyncGenerator, List
from experimance_core.config import CameraConfig
from experimance_core.depth_processor import DepthProcessor, DepthFrame, calculate_change_score, mask_bright_area
from experimance_common.image_utils import get_mock_images

logger = logging.getLogger(__name__)


class MockDepthProcessor(DepthProcessor):
    """Mock depth processor for testing."""
    
    def __init__(self, config: CameraConfig, mock_images_path: Optional[str] = None):
        # Initialize with camera config but don't create real camera
        self.config = config
        self.frame_number = 0
        self.previous_frame: Optional[np.ndarray] = None
        self.crop_bounds: Optional[Tuple[int, int, int, int]] = None
        self.is_warmed_up = False
        self.mock_images_path = mock_images_path
        self.mock_generator = None
        
        # Mask stability tracking (inherited behavior)
        self.stable_mask: Optional[np.ndarray] = None
        self.mask_locked = False
        self.mask_history: List[np.ndarray] = []
        self.frames_since_mask_update = 0
        self.previous_hand_detected = False
        
        # Don't create a real camera for mock
        
    async def initialize(self) -> bool:
        """Initialize mock processor."""
        if self.mock_images_path:
            mock_images = get_mock_images(self.mock_images_path)
            self.mock_generator = self._create_mock_generator(mock_images)
        else:
            # Generate random mock frames
            self.mock_generator = self._random_mock_generator()
        
        logger.info("Mock depth processor initialized")
        return True
    
    @property
    def is_initialized(self) -> bool:
        """Check if the mock processor is initialized and ready."""
        return self.mock_generator is not None
    
    def _create_mock_generator(self, mock_images: list) -> Generator:
        """Create mock generator from image list."""
        while True:
            for img_path in mock_images:
                depth_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if depth_image is not None:
                    depth_image = cv2.resize(depth_image, self.config.output_resolution)
                    hand_detected = randint(0, 100) < 10  # 10% chance of hands
                    yield depth_image, hand_detected
                else:
                    yield None, False
    
    def _random_mock_generator(self) -> Generator:
        """Generate realistic mock depth frames using Perlin-like noise."""
        frame_count = 0
        
        def generate_perlin_like_noise(width: int, height: int, scale: float = 100.0, octaves: int = 4) -> np.ndarray:
            """Generate Perlin-like noise using OpenCV and numpy."""
            # Create base noise
            noise = np.zeros((height, width), dtype=np.float32)
            
            for octave in range(octaves):
                # Create random noise at different scales
                # Adjust scale calculation for larger blobs
                octave_scale = scale / (2 ** octave)  # Changed to division for larger features
                
                # Generate smooth noise by interpolating random values
                # Ensure minimum resolution for very large scales
                low_res_width = max(2, int(width / octave_scale))
                low_res_height = max(2, int(height / octave_scale))
                
                # Generate random values
                random_values = np.random.random((low_res_height, low_res_width)).astype(np.float32)
                
                # Interpolate to full resolution with cubic interpolation for smoothness
                interpolated = cv2.resize(random_values, (width, height), interpolation=cv2.INTER_CUBIC)
                
                # Add to noise with decreasing amplitude
                amplitude = 1.0 / (2 ** octave)
                noise += interpolated * amplitude
            
            # Normalize to 0-1
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            return noise
        
        while True:
            frame_count += 1
            height, width = self.config.output_resolution
            
            # Generate Perlin-like noise for base depth with larger blobs and slower change
            # Much slower time progression for less frequent changes
            slow_time = frame_count * 0.01  # Reduced from 0.05 for slower change
            noise_scale = 3.0 + np.sin(slow_time) * 1.0  # Larger scale = bigger blobs (reduced from 8.0)
            base_noise = generate_perlin_like_noise(width, height, scale=noise_scale, octaves=3)  # Fewer octaves for smoother blobs
            
            # Create circular gradient (simulating sand table depth)
            center_x, center_y = width // 2, height // 2
            y_indices, x_indices = np.ogrid[:height, :width]
            distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            max_distance = min(width, height) // 2
            
            # Circular gradient (bright center, darker edges)
            circular_gradient = np.clip(1.0 - (distances / max_distance), 0.2, 1.0)
            
            # Combine noise with circular gradient
            depth_base = base_noise * circular_gradient
            
            # Add very gentle motion/waves with larger scale and slower movement
            time_offset = frame_count * 0.02  # Much slower wave motion (reduced from 0.08)
            wave_noise = generate_perlin_like_noise(width, height, scale=5.0, octaves=2)  # Larger wave blobs (reduced from 15.0)
            wave_offset = np.sin(wave_noise * 3.14 + time_offset) * 0.05  # Smaller amplitude (reduced from 0.1)
            depth_combined = np.clip(depth_base + wave_offset, 0, 1)
            
            # Convert to uint8
            depth_image = (depth_combined * 255).astype(np.uint8)
            
            # Apply Gaussian blur for smooth depth feel
            depth_image = cv2.GaussianBlur(depth_image, (9, 9), 3.0)  # Larger blur for smoother appearance
            
            # Less frequent "hand" disturbance for more stable visualization
            hand_detected = False
            if np.random.random() < 0.05:  # Reduced from 12% to 5% chance
                hand_detected = True
                # Add circular disturbance to simulate hand
                hand_x = center_x + np.random.randint(-width//4, width//4)
                hand_y = center_y + np.random.randint(-height//4, height//4)
                hand_distances = np.sqrt((x_indices - hand_x)**2 + (y_indices - hand_y)**2)
                hand_radius = 30 + np.random.randint(0, 30)
                hand_mask = hand_distances < hand_radius
                
                # Create hand disturbance (darker area)
                hand_disturbance = np.exp(-(hand_distances / hand_radius)**2) * 80
                depth_image = depth_image.astype(np.float32)
                depth_image[hand_mask] = np.maximum(depth_image[hand_mask] - hand_disturbance[hand_mask], 30)
                depth_image = depth_image.astype(np.uint8)
            
            yield depth_image, hand_detected
    
    async def get_processed_frame(self) -> Optional[DepthFrame]:
        """Get a mock processed frame."""
        if self.mock_generator is None:
            return None
            
        try:
            depth_image, hand_detected = next(self.mock_generator)
            
            if depth_image is None:
                return None
            
            self.frame_number += 1
            
            # Calculate mock change score
            change_score = np.random.random() * 0.5 if hand_detected else np.random.random() * 0.1
            
            # Create mock frame
            frame = DepthFrame(
                depth_image=depth_image,
                color_image=None,
                hand_detected=hand_detected,
                change_score=change_score,
                frame_number=self.frame_number,
                timestamp=time.time()
            )
            
            # Add debug images if debug mode is enabled
            if self.config.debug_mode:
                # Create mock intermediate images for visualization
                frame.raw_depth_image = depth_image.copy()
                
                # Mock importance mask (circular pattern)
                height, width = depth_image.shape
                center = (width // 2, height // 2)
                radius = min(width, height) // 3
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(mask, center, radius, (255,), -1)
                frame.importance_mask = mask
                
                # Mock masked image (apply the mask)
                frame.masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
                
                # Mock cropped before resize (add border)
                frame.cropped_before_resize = cv2.copyMakeBorder(
                    depth_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(128,)
                )
                
                # Mock change diff (random noise pattern)
                change_diff = np.random.randint(0, 255, depth_image.shape, dtype=np.uint8)
                _, frame.change_diff_image = cv2.threshold(change_diff, 200, 255, cv2.THRESH_BINARY)
                
                # Mock hand detection visualization
                debug_img = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                color = (0, 255, 0) if hand_detected else (0, 0, 255)
                cv2.circle(debug_img, center, 30, color, 2)
                cv2.putText(debug_img, f"Mock Hand: {hand_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                frame.hand_detection_image = debug_img
            
            return frame
            
        except StopIteration:
            return None
    
    def stop(self):
        """Stop the mock depth processor."""
        self.mock_generator = None
        logger.info("Mock depth processor stopped")
    
    async def stream_frames(self) -> AsyncGenerator[DepthFrame, None]:
        """
        Stream mock processed depth frames continuously.
        
        Yields:
            DepthFrame objects with mock data
        """
        # Only initialize if not already initialized
        if not self.is_initialized:
            if not await self.initialize():
                logger.error("Failed to initialize mock depth processor")
                return
        
        logger.info("Starting mock depth frame stream")
        
        try:
            target_frame_time = 1.0 / self.config.fps
            
            while True:
                frame_start = time.time()
                frame = await self.get_processed_frame()
                
                if frame is not None:
                    yield frame
                
                # Adaptive frame rate control
                frame_duration = time.time() - frame_start
                sleep_time = max(0, target_frame_time - frame_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Mock depth frame stream cancelled")
        except Exception as e:
            logger.error(f"Error in mock depth frame stream: {e}")
        finally:
            self.stop()
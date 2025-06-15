"""
Robust RealSense Camera Module for Experimance Core Service.

This module provides a modern, clean, and robust interface to Intel RealSense cameras with:
- Automatic retry and reset on failures
- Clean async/await integration
- Comprehensive error handling
- Type hints and dataclasses
- Easy testing and mocking support

This is a complete replacement for depth_finder.py with modern design patterns.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, AsyncGenerator, Generator, Any
from random import randint

import cv2
import numpy as np
import pyrealsense2 as rs
from blessed import Terminal

from experimance_common.image_utils import get_mock_images, crop_to_content

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DEPTH_RESOLUTION = (640, 480)
DEFAULT_FPS = 30
DEFAULT_OUTPUT_RESOLUTION = (1024, 1024)

# Global variables for image processing
change_threshold_resolution = (128, 128)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
term = Terminal()


# ==================== UTILITY FUNCTIONS ====================

def print_status(message: str, style: str = 'info'):
    """Print colored status messages to terminal."""
    colors = {
        'info': term.lightskyblue,
        'warning': term.orange,
        'error': term.red3
    }
    color = colors.get(style, term.snow)
    
    parts = message.split(":", maxsplit=1)
    if len(parts) == 2:
        message = f"{term.normal}{parts[0]}:{term.bold}{parts[1]}{term.normal}"
    
    message = f"{message}{term.clear_eol}"
    print(color(message), sep='', end='\r', flush=True)


def mask_bright_area(image: np.ndarray) -> np.ndarray:
    """Create a mask for the bright area in the center of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    center = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)
    
    _, thresholded = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    
    # Use floodFill to find contiguous bright area from center
    flood_mask = np.zeros((gray_image.shape[0] + 2, gray_image.shape[1] + 2), np.uint8)
    cv2.floodFill(thresholded, flood_mask, center, (255,), loDiff=(20,), upDiff=(20,), flags=cv2.FLOODFILL_MASK_ONLY)
    
    # Crop the flood fill mask
    cropped_mask = flood_mask[1:-1, 1:-1]
    
    # Find and fill contours
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cropped_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    return cropped_mask


def simple_obstruction_detect(image: np.ndarray, size: Tuple[int, int] = (32, 32), pixel_threshold: int = 0) -> Optional[bool]:
    """
    Detect obstruction (hands) in the depth image.
    
    Returns:
        True if obstruction detected, False if not, None if test fails
    """
    resized = cv2.resize(image, size)
    
    if is_blank_frame(resized):
        return None
    
    thickness_multiplier = 0.3
    circle_radius = int(size[0] * (1.0 + thickness_multiplier)) // 2
    circle_center = (size[0] // 2, size[1] // 2)
    
    # Mask everything outside the circle as white
    cv2.circle(resized, circle_center, circle_radius, (255, 255, 255), thickness=int(size[0] * thickness_multiplier))
    
    # Count black pixels inside the circle
    not_black_pixels = cv2.countNonZero(resized)
    black_pixels = size[0] * size[1] - not_black_pixels
    
    return black_pixels > pixel_threshold


def is_blank_frame(image: np.ndarray, threshold: float = 1.0) -> bool:
    """Detect blank/empty frames."""
    if np.std(image) < threshold:
        print_status("blank frame detected", style='warning')
        return True
    return False


def detect_difference(image1: Optional[np.ndarray], image2: np.ndarray, threshold: int = 60) -> Tuple[float, np.ndarray]:
    """
    Calculate the amount of difference between two images.
    
    Returns:
        Tuple of (difference_score, frame_to_use_for_next_comparison)
    """
    if image1 is None:
        return threshold + 1, image2
    
    if is_blank_frame(image2):
        return 0, image1
    
    # Calculate absolute difference
    diff = cv2.absdiff(np.asanyarray(image1), np.asanyarray(image2))
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to remove noise
    cleaned_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
    cleaned_diff = cv2.morphologyEx(cleaned_diff, cv2.MORPH_OPEN, kernel)
    
    # Count non-zero pixels
    difference_score = cv2.countNonZero(cleaned_diff)
    
    return difference_score, image2


def reset_realsense_camera() -> bool:
    """
    Reset the RealSense camera hardware.
    
    Returns:
        True if reset was successful, False otherwise
    """
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            logger.warning('No RealSense devices found for reset')
            return False
        
        dev = devices[0]
        device_name = dev.get_info(rs.camera_info.name)
        logger.info(f'Resetting device: {device_name}')
        
        dev.hardware_reset()
        logger.info('Hardware reset successful')
        
        time.sleep(2)  # Wait for reinitialization
        return True
        
    except Exception as e:
        logger.error(f'Camera reset failed: {e}')
        return False


# ==================== MODERN CAMERA INTERFACE ====================

class CameraState(Enum):
    """Camera operational states."""
    DISCONNECTED = "disconnected"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    RESETTING = "resetting"


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    resolution: Tuple[int, int] = DEFAULT_DEPTH_RESOLUTION
    fps: int = DEFAULT_FPS
    align_frames: bool = True
    min_depth: float = 0.0
    max_depth: float = 10.0
    colorizer_scheme: int = 2  # WhiteToBlack
    json_config_path: Optional[str] = None
    
    # Processing parameters
    output_resolution: Tuple[int, int] = DEFAULT_OUTPUT_RESOLUTION
    change_threshold: int = 60
    detect_hands: bool = True
    crop_to_content: bool = True
    warm_up_frames: int = 10
    lightweight_mode: bool = False  # Skip some processing for higher FPS
    verbose_performance: bool = False  # Show detailed performance timing
    
    # Retry parameters
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 30.0


@dataclass
class DepthFrame:
    """Depth frame data with metadata."""
    depth_image: np.ndarray
    color_image: Optional[np.ndarray] = None
    hand_detected: Optional[bool] = None
    change_score: float = 0.0
    frame_number: int = 0
    timestamp: float = 0.0
    
    @property
    def has_interaction(self) -> bool:
        """Check if frame shows user interaction."""
        return self.hand_detected or self.change_score > 0.1



class RealSenseCamera:
    """
    Robust RealSense camera interface with automatic error recovery.
    
    This class handles all low-level camera operations with retry logic.
    """
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.state = CameraState.DISCONNECTED
        self.pipeline = None
        self.profile = None
        self.colorizer = None
        self.align = None
        self.retry_count = 0
        self.current_retry_delay = config.retry_delay
        
    async def initialize(self) -> bool:
        """Initialize the camera with retry logic."""
        return await self._execute_with_retry("camera initialization", self._init_camera)
    
    async def get_frame(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Capture a single frame with retry logic."""
        if self.state != CameraState.READY:
            logger.warning(f"Camera not ready (state: {self.state})")
            return None
            
        return await self._execute_with_retry("frame capture", self._capture_frame)
    
    def stop(self):
        """Stop the camera pipeline."""
        self.state = CameraState.DISCONNECTED
        if self.pipeline:
            try:
                self.pipeline.stop()
                logger.info("Camera pipeline stopped")
            except Exception as e:
                logger.warning(f"Error stopping pipeline: {e}")
            finally:
                self.pipeline = None
                self.profile = None
    
    async def _execute_with_retry(self, operation_name: str, operation) -> Any:
        """Execute an operation with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
                # Reset retry state on success
                self.retry_count = 0
                self.current_retry_delay = self.config.retry_delay
                return result
                
            except Exception as e:
                last_exception = e
                self.retry_count += 1
                
                logger.warning(f"{operation_name} failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Attempt camera reset
                    await self._reset_camera()
                    
                    # Exponential backoff
                    delay = min(self.current_retry_delay * (2 ** attempt), self.config.max_retry_delay)
                    logger.info(f"Retrying {operation_name} in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {operation_name}")
                    self.state = CameraState.ERROR
        
        return None
    
    async def _reset_camera(self):
        """Reset the camera hardware."""
        self.state = CameraState.RESETTING
        logger.info("Resetting camera hardware...")
        
        # Stop current pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.profile = None
        
        # Hardware reset
        try:
            success = reset_realsense_camera()
            if success:
                logger.info("Camera hardware reset successful")
                await asyncio.sleep(2)  # Wait for device reinitialization
            else:
                logger.warning("Camera hardware reset failed")
        except Exception as e:
            logger.error(f"Error during camera reset: {e}")
    
    def _init_camera(self) -> bool:
        """Initialize the camera pipeline (synchronous)."""
        self.state = CameraState.INITIALIZING
        
        # Configure streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(
            rs.stream.depth, 
            self.config.resolution[0], 
            self.config.resolution[1], 
            rs.format.z16, 
            self.config.fps
        )
        
        if self.config.align_frames:
            config.enable_stream(
                rs.stream.color,
                self.config.resolution[0],
                self.config.resolution[1],
                rs.format.bgr8,
                self.config.fps
            )
        
        # Start pipeline
        self.profile = self.pipeline.start(config)
        
        # Load advanced configuration if specified
        if self.config.json_config_path:
            self._load_advanced_config()
        
        # Configure sensor settings
        self._configure_sensor()
        
        # Setup colorizer
        self._setup_colorizer()
        
        # Setup frame alignment
        if self.config.align_frames:
            self.align = rs.align(rs.stream.color)
        
        self.state = CameraState.READY
        logger.info(f"Camera initialized: {self.config.resolution}@{self.config.fps}fps")
        return True
    
    def _load_advanced_config(self):
        """Load advanced camera configuration from JSON file."""
        if not self.config.json_config_path:
            return
            
        config_path = Path(self.config.json_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            json_obj = json.load(f)
        
        json_string = str(json_obj).replace("'", '"')
        
        device = self.profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(device)
        advanced_mode.load_json(json_string)
        
        logger.info(f"Loaded advanced configuration from {config_path}")
    
    def _configure_sensor(self):
        """Configure depth sensor settings."""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        
        # Set to High Accuracy preset if available
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
            if preset_name == "High Accuracy":
                depth_sensor.set_option(rs.option.visual_preset, i)
                logger.info("Set visual preset to High Accuracy")
                break
    
    def _setup_colorizer(self):
        """Setup depth colorizer."""
        self.colorizer = rs.colorizer(self.config.colorizer_scheme)
        self.colorizer.set_option(rs.option.visual_preset, 1)  # Fixed range
        self.colorizer.set_option(rs.option.min_distance, self.config.min_depth)
        self.colorizer.set_option(rs.option.max_distance, self.config.max_depth)
        self.colorizer.set_option(rs.option.color_scheme, self.config.colorizer_scheme)
    
    def _capture_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Capture a single frame (synchronous)."""
        frames = self.pipeline.wait_for_frames()
        
        color_image = None
        
        if self.config.align_frames and self.align:
            # Align frames
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
        else:
            depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            raise RuntimeError("No depth frame received from camera")
        
        # Colorize depth frame
        depth_color_frame = self.colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())
        
        # Convert to grayscale if needed
        if len(depth_colormap.shape) == 3 and depth_colormap.shape[2] == 3:
            depth_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        else:
            depth_image = depth_colormap
        
        return depth_image, color_image


class DepthProcessor:
    """
    High-level depth processing pipeline with interaction detection.
    
    This class wraps the camera interface and provides processed depth frames
    with interaction detection, change analysis, and cropping.
    """
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = RealSenseCamera(config)
        self.frame_number = 0
        self.previous_frame: Optional[np.ndarray] = None
        self.crop_bounds: Optional[Tuple[int, int, int, int]] = None
        self.is_warmed_up = False
        
    async def initialize(self) -> bool:
        """Initialize the depth processor."""
        if self.camera.state == CameraState.READY:
            logger.debug("Camera already initialized")
            return True
        return await self.camera.initialize()
    
    @property
    def is_initialized(self) -> bool:
        """Check if the depth processor is initialized and ready."""
        return self.camera.state == CameraState.READY
    
    async def get_processed_frame(self) -> Optional[DepthFrame]:
        """
        Get a processed depth frame with interaction detection.
        
        Returns:
            DepthFrame with processed data or None on failure
        """
        frame_start = time.time()
        
        # Get raw frame from camera
        frame_data = await self.camera.get_frame()
        if frame_data is None:
            return None
        
        depth_image, color_image = frame_data
        if depth_image is None:
            return None

        capture_time = time.time() - frame_start
        
        self.frame_number += 1
        
        # Apply importance mask (skip in lightweight mode)
        mask_start = time.time()
        if self.config.lightweight_mode:
            masked_image = depth_image
            mask_time = 0
        else:
            importance_mask = mask_bright_area(depth_image)
            masked_image = cv2.bitwise_and(depth_image, depth_image, mask=importance_mask)
            mask_time = time.time() - mask_start
        
        # Crop and resize if enabled
        crop_start = time.time()
        if self.config.crop_to_content and not self.config.lightweight_mode:
            output, self.crop_bounds = crop_to_content(
                masked_image, 
                size=self.config.output_resolution,
                bounds=self.crop_bounds if self.is_warmed_up else None
            )
        else:
            output = cv2.resize(masked_image, self.config.output_resolution)
        crop_time = time.time() - crop_start
        
        # Detect hands/obstruction
        hand_start = time.time()
        hand_detected = None
        if self.config.detect_hands:
            hand_detected = simple_obstruction_detect(output, pixel_threshold=1)
        hand_time = time.time() - hand_start
        
        # Calculate change score
        change_start = time.time()
        change_score = 0.0
        if self.previous_frame is not None:
            small_current = cv2.resize(output, change_threshold_resolution)
            small_previous = cv2.resize(self.previous_frame, change_threshold_resolution)
            
            diff, _ = detect_difference(
                small_previous.astype(np.uint8),
                small_current.astype(np.uint8),
                threshold=self.config.change_threshold
            )
            
            # Normalize to [0, 1]
            max_diff = change_threshold_resolution[0] * change_threshold_resolution[1]
            change_score = min(1.0, diff / max_diff) if diff > 0 else 0.0
        change_time = time.time() - change_start
        
        # Update state
        self.previous_frame = output.copy()
        
        if self.frame_number >= self.config.warm_up_frames:
            self.is_warmed_up = True
        
        total_time = time.time() - frame_start
        
        # Log performance based on configuration
        if self.config.verbose_performance:
            # Show performance every 30 frames when verbose
            if self.frame_number % 30 == 0:
                logger.info(f"🔧 Frame {self.frame_number} performance: "
                           f"capture={capture_time*1000:.1f}ms, "
                           f"mask={mask_time*1000:.1f}ms, "
                           f"crop={crop_time*1000:.1f}ms, "
                           f"hand={hand_time*1000:.1f}ms, "
                           f"change={change_time*1000:.1f}ms, "
                           f"total={total_time*1000:.1f}ms")
        else:
            # Less frequent logging for normal operation
            if self.frame_number % 60 == 0:
                logger.debug(f"Frame {self.frame_number} timing: "
                            f"capture={capture_time*1000:.1f}ms, "
                            f"mask={mask_time*1000:.1f}ms, "
                            f"crop={crop_time*1000:.1f}ms, "
                            f"hand={hand_time*1000:.1f}ms, "
                            f"change={change_time*1000:.1f}ms, "
                            f"total={total_time*1000:.1f}ms")
        
        return DepthFrame(
            depth_image=output,
            color_image=color_image,
            hand_detected=hand_detected,
            change_score=change_score,
            frame_number=self.frame_number,
            timestamp=time.time()
        )
    
    async def stream_frames(self) -> AsyncGenerator[DepthFrame, None]:
        """
        Stream processed depth frames continuously.
        
        Yields:
            DepthFrame objects with processed data
        """
        # Only initialize if not already initialized
        if self.camera.state != CameraState.READY:
            if not await self.initialize():
                logger.error("Failed to initialize depth processor")
                return
        
        logger.info("Starting depth frame stream")
        
        try:
            target_frame_time = 1.0 / self.config.fps
            last_frame_time = time.time()
            
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
            logger.info("Depth frame stream cancelled")
        except Exception as e:
            logger.error(f"Error in depth frame stream: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the depth processor."""
        self.camera.stop()
        logger.info("Depth processor stopped")


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
        """Generate random mock depth frames."""
        while True:
            # Create random depth image
            depth_image = np.random.randint(0, 255, self.config.output_resolution, dtype=np.uint8)
            
            # Occasionally simulate hand detection
            hand_detected = np.random.random() < 0.1
            
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
            
            return DepthFrame(
                depth_image=depth_image,
                color_image=None,
                hand_detected=hand_detected,
                change_score=change_score,
                frame_number=self.frame_number,
                timestamp=time.time()
            )
            
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


def create_depth_processor(config: CameraConfig, mock_path: Optional[str] = None) -> DepthProcessor:
    """
    Factory function to create a depth processor.
    
    Args:
        config: Camera configuration
        mock_path: Path to mock images (if None, uses real camera)
        
    Returns:
        DepthProcessor instance (real or mock)
    """
    if mock_path:
        return MockDepthProcessor(config, mock_path)
    else:
        return DepthProcessor(config)



"""
Robust Depth Processor for Experimance Core Service.

This module provides a modern, clean, and robust interface to Intel RealSense cameras with:
- Automatic retry and reset on failures
- Clean async/await integration
- Comprehensive error handling
- Type hints and dataclasses
- Easy testing and mocking support

"""
import asyncio
import logging
import time
from typing import Optional, Tuple, AsyncGenerator, List
from random import randint

import cv2
import numpy as np

from experimance_common.image_utils import crop_to_content
from experimance_core.config import CameraConfig, DepthFrame, CameraState
from experimance_core.realsense_camera import RealSenseCamera

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DEPTH_RESOLUTION = (640, 480)
DEFAULT_FPS = 30
DEFAULT_OUTPUT_RESOLUTION = (1024, 1024)

# Global variables for image processing
change_threshold_resolution = (128, 128)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ==================== UTILITY FUNCTIONS ====================

def mask_bright_area(image: np.ndarray) -> np.ndarray:
    """Create a mask for the bright area in the center of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray_image.shape
    center = (width // 2, height // 2)
    
    # For tight depth ranges, we need adaptive thresholding
    # Calculate image statistics to determine appropriate threshold
    non_zero_pixels = gray_image[gray_image > 0]
    if len(non_zero_pixels) == 0:
        # No depth data, return circular fallback
        logger.debug("No depth data found, using circular fallback mask")
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        radius = int(min(width, height) * 0.35)
        cv2.circle(fallback_mask, center, radius, (255,), -1)
        return fallback_mask

    # Use adaptive threshold based on image statistics
    non_zero_pixels = non_zero_pixels.astype(np.float32)  # Ensure correct dtype for mean/std
    mean_val = np.mean(non_zero_pixels)
    std_val = np.std(non_zero_pixels)
    
    # For tight depth ranges, use a higher threshold relative to the data range
    # If std is very low (tight range), use mean - 0.5*std as threshold
    # If std is higher (wide range), use fixed threshold
    if std_val < 20:  # Tight depth range
        threshold = max(10, int(mean_val - 0.5 * std_val))
        logger.debug(f"Tight depth range detected: mean={mean_val:.1f}, std={std_val:.1f}, threshold={threshold:.1f}")
    else:  # Normal depth range
        threshold = 30
        logger.debug(f"Normal depth range: using threshold={threshold}")
    
    _, thresholded = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    # Use floodFill to find contiguous bright area from center
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(thresholded, flood_mask, center, (255,), loDiff=(20,), upDiff=(20,), flags=cv2.FLOODFILL_MASK_ONLY)
    
    # Crop the flood fill mask
    cropped_mask = flood_mask[1:-1, 1:-1]
    
    # Find and fill contours
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cropped_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Check if flood fill found a reasonable area (for sand bowl detection)
    mask_area = cv2.countNonZero(cropped_mask)
    total_area = height * width
    mask_ratio = mask_area / total_area
    
    logger.debug(f"Mask area ratio: {mask_ratio:.3f} ({mask_area}/{total_area} pixels)")
    
    # If flood fill found too much (>80%) or too little (<5%), create a fallback circular mask
    if mask_ratio < 0.05 or mask_ratio > 0.8:
        logger.debug(f"Flood fill area ratio {mask_ratio:.3f} out of range, using fallback circular mask")
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        # Create circular mask sized for typical sand bowl (about 70% of smaller dimension)
        radius = int(min(width, height) * 0.35)
        cv2.circle(fallback_mask, center, radius, (255,), -1)
        return fallback_mask
    
    return cropped_mask


def simple_obstruction_detect(image: np.ndarray, size: Tuple[int, int] = (32, 32), pixel_threshold: int = 1, debug_save: bool = False, debug_prefix: str = "hand_detect") -> Optional[bool]:
    """
    Detect obstruction (hands) in the depth image.

    This function checks if the center of a downscaled copy of the image has a significant number of black pixels
    within a circular area, indicating an obstruction like a hand.
    Args:
        image: Input depth image as a numpy array.
        size: Size to resize the image for processing (default is 32x32).
        pixel_threshold: Minimum number of black pixels required to consider it an obstruction.
        debug_save: If True, save debug images to disk
        debug_prefix: Prefix for debug image filenames
    If the image is blank or has no significant content, returns None.
    
    Returns:
        True if obstruction detected, False if not, None if test fails
    """
    resized = cv2.resize(image, size)
    
    # Save debug image before processing if requested
    if debug_save:
        import os
        debug_dir = "debug_hand_detection"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        #cv2.imwrite(f"{debug_dir}/{debug_prefix}_input_{timestamp}.png", resized)
    
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
    
    result = black_pixels > pixel_threshold
    
    # Add debug logging
    if debug_save:
        print(f"🔍 Hand detection: {black_pixels} black pixels (threshold: {pixel_threshold}) → {result}")
        print(f"    Image stats: min={resized.min()}, max={resized.max()}, mean={resized.mean():.1f}")
    
    # Save debug image after processing if requested
    if debug_save:
        # Create a visualization showing the circle and detection result
        debug_vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR) if len(resized.shape) == 2 else resized.copy()
        color = (0, 255, 0) if result else (0, 0, 255)  # Green if hand detected, red if not
        cv2.circle(debug_vis, circle_center, circle_radius, color, 2)
        cv2.putText(debug_vis, f"Hand: {result} ({black_pixels}px)", (2, size[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.imwrite(f"{debug_dir}/{debug_prefix}_processed_{timestamp}.png", resized)
    
    return result


def is_blank_frame(image: np.ndarray, threshold: float = 1.0) -> bool:
    """Detect blank/empty frames."""
    if np.std(image) < threshold:
        logger.debug("Detected blank frame (std dev below threshold)")
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


def calculate_change_score(current_frame: np.ndarray, previous_frame: np.ndarray, threshold: int) -> float:
    """
    Calculate change score between two frames.
    
    Returns:
        Change score as a float between 0.0 and 1.0
    """
    try:
        # Calculate absolute difference
        diff = cv2.absdiff(current_frame, previous_frame)
        
        # Count pixels above threshold
        changed_pixels = np.sum(diff > threshold)
        total_pixels = diff.size
        
        if total_pixels == 0:
            return 0.0
        
        return float(changed_pixels / total_pixels)
        
    except Exception as e:
        logger.warning(f"Change score calculation failed: {e}")
        return 0.0


class DepthProcessor:
    """
    High-level depth processing pipeline with interaction detection.
    
    This class wraps the camera interface and provides processed depth frames
    with interaction detection, change analysis, and cropping.
    """
    
    def __init__(self, config:CameraConfig):
        """Initialize the depth processor with configuration from config.py."""
        self.config = config
        self.camera = RealSenseCamera(config)
        self.frame_number = 0
        self.previous_frame: Optional[np.ndarray] = None
        self.crop_bounds: Optional[Tuple[int, int, int, int]] = None
        
        # Mask stability tracking
        self.stable_mask: Optional[np.ndarray] = None
        self.mask_locked = False
        self.mask_history: List[np.ndarray] = []  # Store recent masks for stability analysis
        self.frames_since_mask_update = 0
        self.previous_hand_detected = False  # Use previous frame's hand detection
        
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
        frame_start = time.perf_counter()
        
        # Get raw frame from camera
        frame_data = await self.camera.get_frame()
        if frame_data is None:
            return None
        
        depth_image, color_image = frame_data
        if depth_image is None:
            return None

        capture_time = time.perf_counter() - frame_start
        
        self.frame_number += 1
        
        # Initialize debug image containers
        debug_importance_mask = None
        debug_cropped_before_resize = None
        debug_change_diff = None
        debug_hand_detection = None
        
        # Apply importance mask with stability logic (skip in lightweight mode)
        # Use previous frame's hand detection to avoid interference
        mask_start = time.perf_counter()
        if self.config.lightweight_mode:
            masked_image = depth_image
            mask_time = 0
        else:
            # Get mask (optimized for locked state)
            mask_compute_start = time.perf_counter()
            importance_mask = self._get_importance_mask(depth_image, self.previous_hand_detected)
            mask_compute_time = time.perf_counter() - mask_compute_start
            
            # Apply mask efficiently using numpy operations (faster than cv2.bitwise_and)
            mask_apply_start = time.perf_counter()
            # Convert binary mask (0/255) to boolean for efficient multiplication
            binary_mask = importance_mask > 0
            masked_image = depth_image * binary_mask.astype(depth_image.dtype)
            mask_apply_time = time.perf_counter() - mask_apply_start
            
            mask_time = time.perf_counter() - mask_start
            
            # Store debug image if enabled
            if self.config.debug_mode:
                debug_importance_mask = importance_mask.copy()
                
            # Log detailed mask timing if verbose and periodic
            if (self.config.verbose_performance and 
                self.frame_number % 30 == 0 and 
                mask_compute_time > 0.001):  # Only log if compute time > 1ms
                logger.debug(f"🔧 Mask detail: compute={mask_compute_time*1000:.1f}ms, "
                           f"apply={mask_apply_time*1000:.1f}ms, "
                           f"locked={self.mask_locked}")
        
        # Crop and resize if enabled
        crop_start = time.perf_counter()
        if self.config.crop_to_content and not self.config.lightweight_mode:
            # Only use saved bounds if mask is locked (for consistency)
            bounds_to_use = self.crop_bounds if self.mask_locked else None
            
            cropped_image, new_bounds = crop_to_content(
                masked_image, 
                size=self.config.output_resolution,
                bounds=bounds_to_use
            )
            
            # Save bounds once mask is locked
            if self.mask_locked:
                self.crop_bounds = new_bounds
            
            # Store debug image if enabled (before resize)
            if self.config.debug_mode:
                debug_cropped_before_resize = cropped_image.copy()
        else:
            cropped_image = cv2.resize(masked_image, self.config.output_resolution)
        crop_time = time.perf_counter() - crop_start
        
        # Detect hands/obstruction on the processed image
        hand_start = time.perf_counter()
        hand_detected = None
        if self.config.detect_hands:
            hand_detected = simple_obstruction_detect(cropped_image, pixel_threshold=1, debug_save=False, debug_prefix="core_service")
            
            # Store debug image if enabled
            if self.config.debug_mode:
                debug_hand_detection = cropped_image.copy()
                # Add hand detection visualization
                if hand_detected is not None:
                    # Draw a circle or text to show hand detection area
                    debug_img = cv2.cvtColor(debug_hand_detection, cv2.COLOR_GRAY2BGR) if len(debug_hand_detection.shape) == 2 else debug_hand_detection
                    color = (0, 255, 0) if hand_detected else (0, 0, 255)  # Green if hand detected, red if not
                    cv2.circle(debug_img, (debug_img.shape[1]//2, debug_img.shape[0]//2), 30, color, 2)
                    cv2.putText(debug_img, f"Hand: {hand_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    debug_hand_detection = debug_img
                    
        hand_time = time.perf_counter() - hand_start
        
        # Store debug change image if enabled (for debugging, but no change score calculation)
        change_start = time.perf_counter()
        change_score = 0.0  # Placeholder - actual change detection happens in core service
        if self.config.debug_mode and self.previous_frame is not None:
            # Create difference visualization for debugging only
            small_current = cv2.resize(cropped_image, change_threshold_resolution)
            small_previous = cv2.resize(self.previous_frame, change_threshold_resolution)
            diff_img = cv2.absdiff(small_previous.astype(np.uint8), small_current.astype(np.uint8))
            _, binary_diff = cv2.threshold(diff_img, self.config.change_threshold, 255, cv2.THRESH_BINARY)
            # Resize back to output resolution for consistency
            debug_change_diff = cv2.resize(binary_diff, self.config.output_resolution)
        change_time = time.perf_counter() - change_start
        
        # Update state
        self.previous_frame = cropped_image.copy()
        
        total_time = time.perf_counter() - frame_start
        
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
        
        # Create DepthFrame with optional debug images
        frame = DepthFrame(
            depth_image=cropped_image,
            color_image=color_image,
            hand_detected=hand_detected,
            change_score=change_score,
            frame_number=self.frame_number,
            timestamp=time.time()
        )
        
        # Update previous hand detection for next frame's mask stability
        if hand_detected is not None:
            self.previous_hand_detected = hand_detected
        
        # Add debug images if debug mode is enabled
        if self.config.debug_mode:
            frame.raw_depth_image = depth_image.copy()
            frame.importance_mask = debug_importance_mask
            frame.masked_image = masked_image.copy() if not self.config.lightweight_mode else depth_image.copy()
            frame.cropped_before_resize = debug_cropped_before_resize
            frame.change_diff_image = debug_change_diff
            frame.hand_detection_image = debug_hand_detection
        
        return frame
    
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
            
            while True:
                frame_start = time.perf_counter()
                frame = await self.get_processed_frame()
                
                if frame is not None:
                    yield frame
                
                # Adaptive frame rate control
                frame_duration = time.perf_counter() - frame_start
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
    
    def _calculate_mask_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate similarity between two masks (0.0 to 1.0)."""
        if mask1.shape != mask2.shape:
            return 0.0
        
        # Calculate intersection over union (IoU)
        intersection = np.logical_and(mask1 > 0, mask2 > 0).astype(np.uint8)
        union = np.logical_or(mask1 > 0, mask2 > 0).astype(np.uint8)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area == 0:
            return 1.0  # Both masks are empty
        
        return float(intersection_area / union_area)
    
    def _is_mask_stable(self) -> bool:
        """Check if the recent masks are stable enough to lock."""
        if len(self.mask_history) < self.config.mask_stability_frames:
            return False
        
        # Take the last N masks for analysis
        recent_masks = self.mask_history[-self.config.mask_stability_frames:]
        
        # Calculate average similarity between consecutive masks
        similarities = []
        for i in range(1, len(recent_masks)):
            similarity = self._calculate_mask_similarity(recent_masks[i-1], recent_masks[i])
            similarities.append(similarity)
        
        if not similarities:
            return False
        
        average_similarity = np.mean(similarities)
        return bool(average_similarity >= self.config.mask_stability_threshold)
    
    def _should_update_mask(self, hand_detected: bool) -> bool:
        """Determine if we should update the stable mask."""
        # Never update if hands are detected
        if hand_detected:
            return False
        
        # Always update if mask is not locked yet
        if not self.mask_locked:
            return True
        
        # Don't update if mask is locked and updates are disabled
        if self.mask_locked:
            if self.config.mask_allow_updates and self.frames_since_mask_update > 60:
                # Allow updates if enough frames have passed since last update
                return True

        return False
    
    def _get_importance_mask(self, depth_image: np.ndarray, hand_detected: bool) -> np.ndarray:
        """Get importance mask with stability and locking logic."""
        # If mask is locked and updates are not allowed, return immediately
        if self.mask_locked and not self.config.mask_allow_updates:
            # Return stable mask or create fallback if None
            if self.stable_mask is not None:
                return self.stable_mask
            else: # reset
                print("reset mask lock")
                self.mask_locked = False  # Reset lock if no stable mask available
                self.frames_since_mask_update = 0
        
        # track how long since we checked for mask updates
        if self.config.mask_allow_updates:
            self.frames_since_mask_update += 1

        # Determine if we should update our mask
        should_update = self._should_update_mask(hand_detected)


        if should_update:
            # Generate current mask
            current_mask = mask_bright_area(depth_image)
  
            # Check if sand has moved significantly (only if updates are allowed)
            if self.stable_mask is not None and self.config.mask_allow_updates and self.frames_since_mask_update > 60:
                self.frames_since_mask_update = 0  # Reset frame count for next check
                similarity = self._calculate_mask_similarity(self.stable_mask, current_mask)
                if similarity < self.config.mask_update_threshold:
                    logger.info(f"Container movement detected (similarity: {similarity:.3f}), updating mask")
                    self.stable_mask = None
                    self.mask_locked = False
        
        if should_update:
            # Add to mask history for stability analysis
            self.mask_history.append(current_mask.copy())
            
            # Keep only recent masks in history
            if len(self.mask_history) > self.config.mask_stability_frames * 2:
                self.mask_history = self.mask_history[-self.config.mask_stability_frames:]
            
            # Check if we should lock the mask
            if not self.mask_locked and self.config.mask_lock_after_stable:
                if self._is_mask_stable():
                    self.stable_mask = current_mask.copy()
                    self.mask_locked = True
                    logger.info(f"🔒 Mask locked after {len(self.mask_history)} stable frames")
                    return self.stable_mask
            
            # Update stable mask if not locked or if significant change detected
            if not self.mask_locked or self.config.mask_allow_updates:
                if self.mask_locked:
                    logger.info(f"🔄 Updating stable mask (frame {self.frame_number})")
                self.stable_mask = current_mask.copy()
                return self.stable_mask
        
        # Use stable mask if available, otherwise use current mask
        if self.stable_mask is not None:
            return self.stable_mask
        else:
            return current_mask







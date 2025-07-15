"""
Webcam management for the Experimance Agent Service.

Handles webcam capture, frame processing, and basic image preprocessing
for vision analysis and audience detection.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

from ..config import VisionConfig

logger = logging.getLogger(__name__)


class WebcamManager:
    """
    Manages webcam capture and basic image processing.
    
    Provides asynchronous frame capture, preprocessing, and basic
    computer vision utilities for audience detection and VLM analysis.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        
        # Frame processing settings
        self.frame_width = config.webcam_width
        self.frame_height = config.webcam_height
        self.fps = config.webcam_fps
        
    async def start(self):
        """Initialize and start webcam capture."""
        if not self.config.webcam_enabled:
            logger.info("Webcam capture disabled in configuration")
            return
            
        device_id = self._determine_device_id()
        
        try:
            # Initialize OpenCV video capture
            self.cap = cv2.VideoCapture(device_id)
            
            if not self.cap.isOpened():
                if self.config.webcam_auto_detect and device_id == self.config.webcam_device_id:
                    logger.warning(f"Primary camera {device_id} failed, trying auto-detection...")
                    alternative_id = self._auto_detect_camera()
                    if alternative_id is not None:
                        self.cap = cv2.VideoCapture(alternative_id)
                        device_id = alternative_id
                        logger.info(f"Using auto-detected camera {device_id}")
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open webcam device {device_id}")
            
            # Configure capture settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings were applied
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture initial frame from webcam")
            
            self.last_frame = frame
            self.last_frame_time = time.time()
            self.is_active = True
            
            logger.info(f"Webcam initialized on device {device_id}")
            logger.info(f"Resolution: {actual_width}x{actual_height} (requested: {self.frame_width}x{self.frame_height})")
            logger.info(f"FPS: {actual_fps} (requested: {self.fps})")
            
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            await self.stop()
            raise
    
    def _determine_device_id(self) -> int:
        """Determine which device ID to use based on configuration."""
        # If device name is specified, try to find it
        if self.config.webcam_device_name:
            device_id = self._find_device_by_name(self.config.webcam_device_name)
            if device_id is not None:
                logger.info(f"Found camera '{self.config.webcam_device_name}' at device {device_id}")
                return device_id
            else:
                logger.warning(f"Camera '{self.config.webcam_device_name}' not found, using device_id")
        
        return self.config.webcam_device_id
    
    def _find_device_by_name(self, name_pattern: str) -> Optional[int]:
        """Find a camera device by partial name match."""
        # This is a simplified implementation - in practice, you might want to use
        # platform-specific APIs to get actual device names
        logger.info(f"Searching for camera matching pattern: '{name_pattern}'")
        
        # For now, just try the first few device IDs
        for device_id in range(5):
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    # In a more complete implementation, you'd check actual device names here
                    # For now, just return the first working camera if pattern matches a common name
                    if any(keyword in name_pattern.lower() for keyword in ['usb', 'webcam', 'camera']):
                        logger.info(f"Found potential match at device {device_id}")
                        return device_id
        
        return None
    
    def _auto_detect_camera(self) -> Optional[int]:
        """Auto-detect the first working camera."""
        logger.info("Auto-detecting available cameras...")
        
        for device_id in range(10):  # Check devices 0-9
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        logger.info(f"Found working camera at device {device_id}")
                        return device_id
            except Exception:
                continue
        
        logger.warning("No working cameras found during auto-detection")
        return None
    
    async def stop(self):
        """Stop webcam capture and clean up resources."""
        self.is_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.last_frame = None
        logger.info("Webcam capture stopped")
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the webcam.
        
        Returns:
            numpy.ndarray: BGR image frame, or None if capture failed
        """
        if not self.is_active or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.last_frame_time = time.time()
                return frame
            else:
                logger.warning("Failed to capture frame from webcam")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing webcam frame: {e}")
            return None
    
    async def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recently captured frame without triggering a new capture.
        
        Returns:
            numpy.ndarray: BGR image frame, or None if no frame available
        """
        return self.last_frame.copy() if self.last_frame is not None else None
    
    def get_frame_age(self) -> float:
        """
        Get the age of the last captured frame in seconds.
        
        Returns:
            float: Age of last frame in seconds, or float('inf') if no frame
        """
        if self.last_frame_time == 0.0:
            return float('inf')
        return time.time() - self.last_frame_time
    
    def preprocess_for_vlm(self, frame: np.ndarray, max_size: Optional[int] = None) -> np.ndarray:
        """
        Preprocess frame for VLM analysis.
        
        Args:
            frame: BGR image frame
            max_size: Maximum dimension size (will resize maintaining aspect ratio)
            
        Returns:
            numpy.ndarray: RGB image frame preprocessed for VLM
        """
        if max_size is None:
            max_size = self.config.vlm_max_image_size
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if max_size > 0:
            h, w = rgb_frame.shape[:2]
            if max(h, w) > max_size:
                if h > w:
                    new_h, new_w = max_size, int(w * max_size / h)
                else:
                    new_h, new_w = int(h * max_size / w), max_size
                    
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return rgb_frame
    
    def preprocess_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for audience detection.
        
        Args:
            frame: BGR image frame
            
        Returns:
            numpy.ndarray: Preprocessed frame suitable for detection algorithms
        """
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        return blurred
    
    def get_capture_info(self) -> dict:
        """
        Get information about the current capture configuration.
        
        Returns:
            dict: Capture configuration and status information
        """
        info = {
            "enabled": self.config.webcam_enabled,
            "active": self.is_active,
            "device_id": self.config.webcam_device_id,
            "device_name": self.config.webcam_device_name,
            "auto_detect": self.config.webcam_auto_detect,
            "requested_resolution": f"{self.frame_width}x{self.frame_height}",
            "requested_fps": self.fps,
            "has_frame": self.last_frame is not None,
            "frame_age": self.get_frame_age()
        }
        
        if self.cap and self.cap.isOpened():
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            info.update({
                "actual_resolution": f"{actual_width}x{actual_height}",
                "actual_fps": actual_fps,
                "backend": self.cap.getBackendName(),
                "resolution_match": (actual_width == self.frame_width and actual_height == self.frame_height)
            })
            
        return info
    
    async def save_frame(self, frame: np.ndarray, filename: str, directory: Optional[Path] = None) -> bool:
        """
        Save a frame to disk for debugging or archival.
        
        Args:
            frame: BGR image frame to save
            filename: Filename for the saved image
            directory: Directory to save to (defaults to logs/webcam/)
            
        Returns:
            bool: True if save was successful
        """
        try:
            if directory is None:
                directory = Path("logs/webcam")
            
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            success = cv2.imwrite(str(filepath), frame)
            if success:
                logger.debug(f"Saved frame to {filepath}")
            else:
                logger.error(f"Failed to save frame to {filepath}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False

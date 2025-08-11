"""
Webcam management for the Experimance Agent Service.

Handles webcam capture, frame processing, and basic image preprocessing
for vision analysis and audience detection.
"""

import asyncio
import logging
import time
import subprocess
from typing import Optional, Tuple, Dict, Any
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
        self.device_id: Optional[int] = None
        
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
            
            # Store the device ID for camera settings
            self.device_id = device_id
            
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
    
    def apply_camera_profile(self, camera_profile) -> bool:
        """
        Apply camera settings from a camera profile.
        
        Args:
            camera_profile: CameraProfile instance with settings to apply
            
        Returns:
            bool: True if all critical settings were applied successfully
        """
        if not camera_profile or not camera_profile.auto_apply:
            return True  # Nothing to apply
            
        logger.info(f"Applying camera profile: {camera_profile.name}")
        
        # Convert settings to dictionary, filtering out None values
        settings_dict = {}
        for field_name, field_value in camera_profile.settings.dict().items():
            if field_value is not None:
                # Handle format settings separately
                if field_name.startswith('preferred_'):
                    continue
                settings_dict[field_name] = field_value
        
        # Apply v4l2 control settings
        results = {}
        if settings_dict:
            results = self.apply_camera_settings(settings_dict)
        
        # Apply format settings if specified
        settings = camera_profile.settings
        if (settings.preferred_format and 
            settings.preferred_width and 
            settings.preferred_height):
            
            format_success = self.set_camera_format(
                settings.preferred_width,
                settings.preferred_height,
                settings.preferred_format
            )
            results['format'] = format_success
        
        # Consider application successful if most settings worked
        success_count = sum(results.values())
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 1.0
        
        if success_rate >= 0.7:  # 70% success rate threshold
            logger.info(f"Camera profile '{camera_profile.name}' applied successfully ({success_count}/{total_count})")
            return True
        else:
            logger.warning(f"Camera profile '{camera_profile.name}' had limited success ({success_count}/{total_count})")
            return False
    
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
        logger.info(f"Searching for camera matching pattern: '{name_pattern}'")
        
        # Try to use v4l2-ctl for accurate device name detection
        try:
            import subprocess
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return self._parse_v4l2_devices(result.stdout, name_pattern)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("v4l2-ctl not available or failed, falling back to basic detection")
        
        # Fallback: basic pattern matching with device testing
        logger.info("Using fallback camera detection method")
        name_lower = name_pattern.lower()
        
        # Check devices 0-9 for working cameras
        for device_id in range(10):
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        # Check if pattern matches common camera name patterns
                        if any(keyword in name_lower for keyword in [
                            'emeet', 'usb', 'webcam', 'camera', 'hd', 'c960', 'c920', 'c930'
                        ]):
                            logger.info(f"Found potential camera match at device {device_id}")
                            return device_id
            except Exception:
                continue
        
        logger.warning(f"No camera found matching pattern '{name_pattern}'")
        return None
    
    def _parse_v4l2_devices(self, v4l2_output: str, name_pattern: str) -> Optional[int]:
        """Parse v4l2-ctl --list-devices output to find matching camera."""
        lines = v4l2_output.strip().split('\n')
        current_device_name = None
        name_lower = name_pattern.lower()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Device name lines don't start with /dev/ or whitespace
            if not line.startswith('/dev/') and not line.startswith('\t') and not line.startswith(' '):
                current_device_name = line.lower()
                logger.debug(f"Found device: {current_device_name}")
            
            # Video device lines start with /dev/video
            elif line.startswith('/dev/video') and current_device_name:
                # Check if current device name matches our pattern
                if name_lower in current_device_name:
                    # Extract device number from /dev/videoN
                    try:
                        device_id = int(line.split('/dev/video')[1])
                        logger.info(f"Found matching camera '{current_device_name}' at device {device_id}")
                        
                        # Verify the device actually works
                        cap = cv2.VideoCapture(device_id)
                        if cap.isOpened():
                            ret, _ = cap.read()
                            cap.release()
                            if ret:
                                return device_id
                        logger.warning(f"Device {device_id} found but not working")
                    except (ValueError, IndexError):
                        continue
        
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
    
    def apply_camera_settings(self, camera_settings: Dict[str, Any]) -> Dict[str, bool]:
        """
        Apply camera hardware settings using v4l2-ctl.
        
        Args:
            camera_settings: Dictionary of v4l2 control settings
            
        Returns:
            Dict[str, bool]: Results of each setting application
        """
        if self.device_id is None:
            logger.error("Cannot apply camera settings: no device ID available")
            return {}
        
        device_path = f"/dev/video{self.device_id}"
        results = {}
        
        logger.info(f"Applying camera settings to {device_path}")
        
        for setting, value in camera_settings.items():
            if value is None:
                continue  # Skip None values
                
            try:
                cmd = ['v4l2-ctl', f'--device={device_path}', f'--set-ctrl={setting}={value}']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    logger.debug(f"Applied {setting}={value}")
                    results[setting] = True
                else:
                    logger.warning(f"Failed to apply {setting}={value}: {result.stderr.strip()}")
                    results[setting] = False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout applying {setting}={value}")
                results[setting] = False
            except Exception as e:
                logger.error(f"Error applying {setting}={value}: {e}")
                results[setting] = False
        
        # Log summary
        success_count = sum(results.values())
        total_count = len(results)
        logger.info(f"Applied {success_count}/{total_count} camera settings successfully")
        
        return results
    
    def set_camera_format(self, width: int, height: int, format_name: str = "MJPG") -> bool:
        """
        Set camera format and resolution using v4l2-ctl.
        
        Args:
            width: Frame width
            height: Frame height  
            format_name: Pixel format (YUYV, MJPG, etc.)
            
        Returns:
            bool: True if format was set successfully
        """
        if self.device_id is None:
            logger.error("Cannot set camera format: no device ID available")
            return False
        
        device_path = f"/dev/video{self.device_id}"
        
        try:
            cmd = ['v4l2-ctl', f'--device={device_path}', 
                   f'--set-fmt-video=width={width},height={height},pixelformat={format_name}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                logger.info(f"Set camera format to {format_name} {width}x{height}")
                
                # Update OpenCV capture settings to match
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                return True
            else:
                logger.warning(f"Failed to set camera format: {result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout setting camera format")
            return False
        except Exception as e:
            logger.error(f"Error setting camera format: {e}")
            return False
    
    def get_camera_controls(self) -> Dict[str, Any]:
        """
        Get current camera control values using v4l2-ctl.
        
        Returns:
            Dict[str, Any]: Current camera control values
        """
        if self.device_id is None:
            logger.error("Cannot get camera controls: no device ID available")
            return {}
        
        device_path = f"/dev/video{self.device_id}"
        controls = {}
        
        try:
            # Get list of available controls
            cmd = ['v4l2-ctl', f'--device={device_path}', '--list-ctrls']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse control names and current values
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if ':' in line and '=' in line:
                        # Example: "brightness 0x00980900 (int) : min=-64 max=64 step=1 default=0 value=0"
                        parts = line.split(':')
                        if len(parts) >= 2:
                            control_info = parts[0].strip()
                            value_info = parts[1].strip()
                            
                            # Extract control name
                            control_name = control_info.split()[0]
                            
                            # Extract current value
                            if 'value=' in value_info:
                                value_str = value_info.split('value=')[1].split()[0]
                                try:
                                    controls[control_name] = int(value_str)
                                except ValueError:
                                    controls[control_name] = value_str
            
            logger.debug(f"Retrieved {len(controls)} camera controls")
            return controls
            
        except subprocess.TimeoutExpired:
            logger.error("Timeout getting camera controls")
            return {}
        except Exception as e:
            logger.error(f"Error getting camera controls: {e}")
            return {}

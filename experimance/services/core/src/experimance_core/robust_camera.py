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
from typing import Optional, Tuple, AsyncGenerator, Generator, Any, List, Dict
from random import randint
import subprocess
import psutil
import os
import signal

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


# ==================== MODERN CAMERA INTERFACE ====================

class CameraState(Enum):
    """Camera operational states."""
    DISCONNECTED = "disconnected"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    RESETTING = "resetting"

class ColorizerScheme(Enum):
    """Colorizer schemes for depth visualization."""
    # from: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.colorizer.html
    JET = 0
    CLASSIC = 1
    WHITE_TO_BLACK = 2
    BLACK_TO_WHITE = 3
    BIO = 4
    COLD = 5
    WARM = 6
    QUANTIZED = 7
    PATTERN = 8

@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    resolution: Tuple[int, int] = DEFAULT_DEPTH_RESOLUTION
    fps: int = DEFAULT_FPS
    align_frames: bool = True
    min_depth: float = 0.0
    max_depth: float = 10.0
    colorizer_scheme: ColorizerScheme = ColorizerScheme.CLASSIC  # WhiteToBlack
    json_config_path: Optional[str] = None
    
    # Processing parameters
    output_resolution: Tuple[int, int] = DEFAULT_OUTPUT_RESOLUTION
    change_threshold: int = 60
    detect_hands: bool = True
    crop_to_content: bool = True
    warm_up_frames: int = 10
    lightweight_mode: bool = False  # Skip some processing for higher FPS
    verbose_performance: bool = False  # Show detailed performance timing
    debug_mode: bool = False  # Include intermediate images for visualization
    
    # Retry parameters
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 30.0
    aggressive_reset: bool = False  # Use more aggressive reset strategies
    skip_advanced_config: bool = False  # Skip advanced JSON config loading
    
    # RealSense filters (alternative to JSON config)
    enable_filters: bool = True
    spatial_filter: bool = True
    temporal_filter: bool = True
    decimation_filter: bool = False
    hole_filling_filter: bool = True
    threshold_filter: bool = False
    
    # Spatial filter settings
    spatial_filter_magnitude: float = 2.0
    spatial_filter_alpha: float = 0.5
    spatial_filter_delta: float = 20.0
    spatial_filter_hole_fill: int = 1
    
    # Temporal filter settings
    temporal_filter_alpha: float = 0.4
    temporal_filter_delta: float = 20.0
    temporal_filter_persistence: int = 3
    
    # Decimation filter settings
    decimation_filter_magnitude: int = 2
    
    # Hole filling filter settings
    hole_filling_mode: int = 1  # 0=disabled, 1=fill_from_left, 2=farest_from_around
    
    # Threshold filter settings
    threshold_filter_min: float = 0.15
    threshold_filter_max: float = 4.0


@dataclass
class DepthFrame:
    """Depth frame data with metadata."""
    depth_image: np.ndarray
    color_image: Optional[np.ndarray] = None
    hand_detected: Optional[bool] = None
    change_score: float = 0.0
    frame_number: int = 0
    timestamp: float = 0.0
    
    # Debug/visualization intermediate images (only populated when debug_mode=True)
    raw_depth_image: Optional[np.ndarray] = None
    masked_image: Optional[np.ndarray] = None
    importance_mask: Optional[np.ndarray] = None
    cropped_before_resize: Optional[np.ndarray] = None
    change_diff_image: Optional[np.ndarray] = None
    hand_detection_image: Optional[np.ndarray] = None
    
    @property
    def has_interaction(self) -> bool:
        """Check if frame shows user interaction."""
        return self.hand_detected or self.change_score > 0.1
    
    @property
    def has_debug_images(self) -> bool:
        """Check if frame contains debug/intermediate images."""
        return self.raw_depth_image is not None



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
    height, width = gray_image.shape
    center = (width // 2, height // 2)
    
    _, thresholded = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    
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
    
    # If flood fill didn't find much (less than 5% of image), create a fallback circular mask
    # This handles cases like pointing at ceiling or low contrast scenes
    if mask_ratio < 0.05:
        logger.debug(f"Flood fill found minimal area ({mask_ratio:.3f}), using fallback circular mask")
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        # Create circular mask sized for typical sand bowl (about 70% of smaller dimension)
        radius = int(min(width, height) * 0.35)
        cv2.circle(fallback_mask, center, radius, 255, -1)
        return fallback_mask
    
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


def get_camera_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive camera diagnostics.
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'devices': [],
        'processes': [],
        'usb_devices': [],
        'realsense_info': {}
    }
    
    try:
        # RealSense device enumeration
        ctx = rs.context()
        devices = ctx.query_devices()
        
        for i, dev in enumerate(devices):
            device_info = {
                'index': i,
                'name': dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else 'Unknown',
                'serial': dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else 'Unknown',
                'firmware': dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else 'Unknown',
                'product_id': dev.get_info(rs.camera_info.product_id) if dev.supports(rs.camera_info.product_id) else 'Unknown',
                'usb_type': dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else 'Unknown',
                'sensors': []
            }
            
            # Get sensor information
            for sensor in dev.query_sensors():
                sensor_info = {
                    'name': sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else 'Unknown',
                    'profiles': []
                }
                
                try:
                    profiles = sensor.get_stream_profiles()
                    for profile in profiles[:5]:  # Limit to first 5 profiles
                        if profile.is_video_stream_profile():
                            vp = profile.as_video_stream_profile()
                            sensor_info['profiles'].append({
                                'stream': str(vp.stream_type()),
                                'format': str(vp.format()),
                                'width': vp.width(),
                                'height': vp.height(),
                                'fps': vp.fps()
                            })
                except Exception as e:
                    sensor_info['error'] = str(e)
                
                device_info['sensors'].append(sensor_info)
                
            diagnostics['devices'].append(device_info)
            
        diagnostics['realsense_info'] = {
            'device_count': len(devices),
            'context_created': True
        }
        
    except Exception as e:
        diagnostics['realsense_info'] = {
            'error': str(e),
            'context_created': False
        }
    
    # Check for processes using camera
    try:
        camera_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process might be using camera
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                name = proc.info['name'].lower()
                
                if any(keyword in name or keyword in cmdline.lower() for keyword in 
                       ['realsense', 'camera', 'opencv', 'gstreamer', 'v4l2', 'experimance']):
                    camera_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        diagnostics['processes'] = camera_processes
        
    except Exception as e:
        diagnostics['processes'] = [{'error': str(e)}]
    
    # Check USB devices
    try:
        usb_devices = []
        lsusb_output = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if lsusb_output.returncode == 0:
            for line in lsusb_output.stdout.split('\n'):
                if 'intel' in line.lower() or '8086' in line or 'realsense' in line.lower():
                    usb_devices.append(line.strip())
        diagnostics['usb_devices'] = usb_devices
        
    except Exception as e:
        diagnostics['usb_devices'] = [f'Error: {e}']
    
    return diagnostics


def kill_camera_processes() -> bool:
    """
    Kill processes that might be holding camera resources.
    
    Returns:
        True if any processes were killed, False otherwise
    """
    killed = False
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                name = proc.info['name'].lower()
                
                # Be conservative - only kill obvious camera processes
                if any(keyword in name for keyword in ['realsense-viewer', 'intel-realsense']):
                    logger.info(f"Killing camera process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.terminate()
                    killed = True
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        if killed:
            time.sleep(2)  # Wait for processes to terminate
            
    except Exception as e:
        logger.error(f"Error killing camera processes: {e}")
        
    return killed


def usb_reset_device(vendor_id: str = "8086", product_id: str = None) -> bool:
    """
    Attempt to reset USB device by vendor/product ID.
    
    Args:
        vendor_id: USB vendor ID (default: Intel)
        product_id: USB product ID (optional)
        
    Returns:
        True if reset was attempted, False otherwise
    """
    try:
        # Find USB device
        lsusb_output = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if lsusb_output.returncode != 0:
            logger.warning("lsusb command failed")
            return False
            
        usb_device = None
        for line in lsusb_output.stdout.split('\n'):
            if vendor_id in line:
                if product_id is None or product_id in line:
                    # Extract bus and device numbers
                    parts = line.split()
                    if len(parts) >= 4:
                        bus = parts[1]
                        device = parts[3].rstrip(':')
                        usb_device = f"/dev/bus/usb/{bus}/{device}"
                        break
        
        if not usb_device:
            logger.warning(f"USB device with vendor ID {vendor_id} not found")
            return False
            
        # Attempt USB reset using python usb library if available
        try:
            import usb.core
            import usb.util
            
            devices = usb.core.find(find_all=True, idVendor=int(vendor_id, 16))
            for device in devices:
                if product_id is None or device.idProduct == int(product_id, 16):
                    logger.info(f"Resetting USB device: vendor={vendor_id}, product={device.idProduct:04x}")
                    device.reset()
                    return True
                    
        except ImportError:
            logger.warning("pyusb not available for USB reset")
        except Exception as e:
            logger.warning(f"USB reset via pyusb failed: {e}")
            
        # Fallback: try to unbind/rebind driver
        try:
            # This is more complex and requires root privileges
            logger.info("USB reset attempted but requires additional privileges")
            return False
            
        except Exception as e:
            logger.warning(f"USB driver reset failed: {e}")
            
    except Exception as e:
        logger.error(f"USB reset error: {e}")
        
    return False


def reset_realsense_camera(aggressive: bool = False) -> bool:
    """
    Reset the RealSense camera hardware with multiple strategies.
    
    Args:
        aggressive: If True, use more aggressive reset strategies
    
    Returns:
        True if reset was successful, False otherwise
    """
    logger.info(f"Starting camera reset (aggressive={aggressive})")
    
    # Step 1: Get diagnostics before reset
    diagnostics = get_camera_diagnostics()
    logger.info(f"Found {len(diagnostics['devices'])} RealSense devices")
    
    if len(diagnostics['devices']) == 0:
        logger.warning('No RealSense devices found for reset')
        return False
    
    # Step 2: Kill potentially interfering processes
    if aggressive:
        logger.info("Killing potentially interfering processes...")
        kill_camera_processes()
    
    # Step 3: Hardware reset
    success = False
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        for i, dev in enumerate(devices):
            device_name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else f'Device {i}'
            logger.info(f'Resetting device: {device_name}')
            
            try:
                dev.hardware_reset()
                logger.info(f'Hardware reset successful for {device_name}')
                success = True
                
                # Wait longer for device to reinitialize
                time.sleep(3 if not aggressive else 5)
                
            except Exception as e:
                logger.warning(f'Hardware reset failed for {device_name}: {e}')
                
        if not success:
            logger.error('All hardware resets failed')
            
    except Exception as e:
        logger.error(f'Camera enumeration failed during reset: {e}')
        
    # Step 4: USB reset (if aggressive and hardware reset failed)
    if aggressive and not success:
        logger.info("Attempting USB reset...")
        usb_success = usb_reset_device()
        if usb_success:
            logger.info("USB reset completed")
            time.sleep(5)  # Wait longer after USB reset
            success = True
        else:
            logger.warning("USB reset failed or not available")
    
    # Step 5: Verify reset by checking device availability
    if success:
        logger.info("Verifying reset by checking device availability...")
        time.sleep(2)  # Additional wait
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                logger.info(f"Reset verification successful: {len(devices)} devices available")
                return True
            else:
                logger.warning("Reset verification failed: no devices found")
                return False
                
        except Exception as e:
            logger.warning(f"Reset verification failed: {e}")
            return False
    
    logger.error("Camera reset failed")
    return False


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
        self.filters = []  # Post-processing filters
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
        """Reset the camera hardware and reinitialize."""
        self.state = CameraState.RESETTING
        logger.info("Resetting camera hardware...")
        
        # Stop current pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
                logger.debug("Pipeline stopped")
            except Exception as e:
                logger.debug(f"Error stopping pipeline: {e}")
            self.pipeline = None
            self.profile = None
        
        # Hardware reset
        try:
            success = reset_realsense_camera(aggressive=self.config.aggressive_reset)
            if success:
                logger.info("Camera hardware reset successful")
                await asyncio.sleep(3)  # Wait for device reinitialization
                
                # Reinitialize the camera pipeline after reset
                logger.info("Reinitializing camera pipeline after reset...")
                try:
                    init_success = self._init_camera()
                    if init_success:
                        logger.info("Camera pipeline reinitialized successfully")
                        self.state = CameraState.READY
                    else:
                        logger.error("Failed to reinitialize camera pipeline after reset")
                        self.state = CameraState.ERROR
                except Exception as e:
                    logger.error(f"Error reinitializing camera after reset: {e}")
                    
                    # Try again without advanced config if it was enabled
                    if self.config.json_config_path and not self.config.skip_advanced_config:
                        logger.info("Retrying initialization without advanced config...")
                        self.config.skip_advanced_config = True
                        try:
                            init_success = self._init_camera()
                            if init_success:
                                logger.info("Camera pipeline reinitialized successfully (without advanced config)")
                                self.state = CameraState.READY
                            else:
                                logger.error("Failed to reinitialize even without advanced config")
                                self.state = CameraState.ERROR
                        except Exception as e2:
                            logger.error(f"Error reinitializing without advanced config: {e2}")
                            self.state = CameraState.ERROR
                    else:
                        self.state = CameraState.ERROR
            else:
                logger.warning("Camera hardware reset failed")
                self.state = CameraState.ERROR
        except Exception as e:
            logger.error(f"Error during camera reset: {e}")
            self.state = CameraState.ERROR
    
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
        
        # Load advanced configuration if specified and not skipped
        if self.config.json_config_path and not self.config.skip_advanced_config:
            self._load_advanced_config()
        
        # Configure sensor settings
        self._configure_sensor()
        
        # Setup colorizer
        self._setup_colorizer()
        
        # Setup frame alignment
        if self.config.align_frames:
            self.align = rs.align(rs.stream.color)
        
        # Setup post-processing filters
        self._setup_post_processing()
        
        self.state = CameraState.READY
        logger.info(f"Camera initialized: {self.config.resolution}@{self.config.fps}fps min_depth={self.config.min_depth}m max_depth={self.config.max_depth}m colorizer={self.config.colorizer_scheme.name}")
        return True
    
    def _load_advanced_config(self):
        """Load advanced camera configuration from JSON file with error handling."""
        if not self.config.json_config_path:
            return
            
        config_path = Path(self.config.json_config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path) as f:
                json_obj = json.load(f)
            
            json_string = str(json_obj).replace("'", '"')
            
            device = self.profile.get_device()
            advanced_mode = rs.rs400_advanced_mode(device)
            
            # Check if device supports advanced mode
            if not advanced_mode.is_enabled():
                logger.warning("Advanced mode not enabled on device, skipping advanced config")
                return
                
            advanced_mode.load_json(json_string)
            logger.info(f"Loaded advanced configuration from {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load advanced configuration: {e}")
            logger.info("Continuing with default camera settings")
            # Don't raise the exception - continue with basic configuration
    
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
        self.colorizer = rs.colorizer(self.config.colorizer_scheme.value)
        self.colorizer.set_option(rs.option.visual_preset, 1)  # Fixed range
        self.colorizer.set_option(rs.option.min_distance, self.config.min_depth)
        self.colorizer.set_option(rs.option.max_distance, self.config.max_depth)
        self.colorizer.set_option(rs.option.color_scheme, self.config.colorizer_scheme.value)
    
    def _setup_post_processing(self):
        """Setup RealSense post-processing filters."""
        if not self.config.enable_filters:
            logger.info("Post-processing filters disabled")
            return
        
        self.filters = []
        
        # Decimation filter (reduces resolution)
        if self.config.decimation_filter:
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, self.config.decimation_filter_magnitude)
            self.filters.append(("decimation", decimation))
            logger.info(f"Enabled decimation filter (magnitude: {self.config.decimation_filter_magnitude})")
        
        # Threshold filter (depth range)
        if self.config.threshold_filter:
            threshold = rs.threshold_filter()
            threshold.set_option(rs.option.min_distance, self.config.threshold_filter_min)
            threshold.set_option(rs.option.max_distance, self.config.threshold_filter_max)
            self.filters.append(("threshold", threshold))
            logger.info(f"Enabled threshold filter (range: {self.config.threshold_filter_min}-{self.config.threshold_filter_max}m)")
        
        # Spatial filter (edge-preserving)
        if self.config.spatial_filter:
            # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.spatial_filter.html
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, self.config.spatial_filter_magnitude)
            spatial.set_option(rs.option.filter_smooth_alpha, self.config.spatial_filter_alpha)
            spatial.set_option(rs.option.filter_smooth_delta, self.config.spatial_filter_delta)
            spatial.set_option(rs.option.holes_fill, self.config.spatial_filter_hole_fill)
            self.filters.append(("spatial", spatial))
            logger.info(f"Enabled spatial filter (mag: {self.config.spatial_filter_magnitude}, "
                       f"alpha: {self.config.spatial_filter_alpha}, delta: {self.config.spatial_filter_delta})")
        
        # Temporal filter (reduces temporal noise)
        if self.config.temporal_filter:
            # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.temporal_filter.html
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, self.config.temporal_filter_alpha)
            temporal.set_option(rs.option.filter_smooth_delta, self.config.temporal_filter_delta)
            temporal.set_option(rs.option.holes_fill, self.config.temporal_filter_persistence)
            self.filters.append(("temporal", temporal))
            logger.info(f"Enabled temporal filter (alpha: {self.config.temporal_filter_alpha}, "
                       f"delta: {self.config.temporal_filter_delta}, persistence: {self.config.temporal_filter_persistence})")
        
        # Hole filling filter
        if self.config.hole_filling_filter:
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill, self.config.hole_filling_mode)
            self.filters.append(("hole_filling", hole_filling))
            logger.info(f"Enabled hole filling filter (mode: {self.config.hole_filling_mode})")
        
        if self.filters:
            logger.info(f"Initialized {len(self.filters)} post-processing filters")
        else:
            logger.info("No post-processing filters enabled")
    
    def _apply_filters(self, frame):
        """Apply post-processing filters to a RealSense frame.
        Recommended to be called before alignment.
        """
        if not hasattr(self, 'filters') or not self.filters:
            return frame
        
        filtered_frame = frame
        for filter_name, filter_obj in self.filters:
            try:
                filtered_frame = filter_obj.process(filtered_frame)
            except Exception as e:
                logger.warning(f"Filter {filter_name} failed: {e}")
                # Continue with the last successful filtered frame
                break
                
        return filtered_frame

    def _capture_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Capture a single frame (synchronous)."""
        frames = self.pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            raise RuntimeError("No depth frame received from camera")
        
        # Handle alignment if enabled
        color_image = None
        if self.config.align_frames and self.align:
            # Align the original composite frames (before filtering)
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # Apply filters to the aligned depth frame
            filtered_depth_frame = self._apply_filters(aligned_depth_frame)
            
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
        else:
            # Apply post-processing filters to depth frame
            filtered_depth_frame = self._apply_filters(depth_frame)

        # Colorize depth frame
        depth_color_frame = self.colorizer.colorize(filtered_depth_frame)
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
        
        # Initialize debug image containers
        debug_importance_mask = None
        debug_cropped_before_resize = None
        debug_change_diff = None
        debug_hand_detection = None
        
        # Apply importance mask (skip in lightweight mode)
        mask_start = time.time()
        if self.config.lightweight_mode:
            masked_image = depth_image
            mask_time = 0
        else:
            importance_mask = mask_bright_area(depth_image)
            masked_image = cv2.bitwise_and(depth_image, depth_image, mask=importance_mask)
            mask_time = time.time() - mask_start
            
            # Store debug image if enabled
            if self.config.debug_mode:
                debug_importance_mask = importance_mask.copy()
        
        # Crop and resize if enabled
        crop_start = time.time()
        if self.config.crop_to_content and not self.config.lightweight_mode:
            output, self.crop_bounds = crop_to_content(
                masked_image, 
                size=self.config.output_resolution,
                bounds=self.crop_bounds if self.is_warmed_up else None
            )
            
            # Store debug image if enabled (before resize)
            if self.config.debug_mode:
                debug_cropped_before_resize = masked_image.copy()
        else:
            output = cv2.resize(masked_image, self.config.output_resolution)
        crop_time = time.time() - crop_start
        
        # Detect hands/obstruction
        hand_start = time.time()
        hand_detected = None
        if self.config.detect_hands:
            hand_detected = simple_obstruction_detect(output, pixel_threshold=1)
            
            # Store debug image if enabled
            if self.config.debug_mode:
                debug_hand_detection = output.copy()
                # Add hand detection visualization
                if hand_detected is not None:
                    # Draw a circle or text to show hand detection area
                    debug_img = cv2.cvtColor(debug_hand_detection, cv2.COLOR_GRAY2BGR) if len(debug_hand_detection.shape) == 2 else debug_hand_detection
                    color = (0, 255, 0) if hand_detected else (0, 0, 255)  # Green if hand detected, red if not
                    cv2.circle(debug_img, (debug_img.shape[1]//2, debug_img.shape[0]//2), 30, color, 2)
                    cv2.putText(debug_img, f"Hand: {hand_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    debug_hand_detection = debug_img
                    
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
            
            # Store debug image if enabled
            if self.config.debug_mode:
                # Create difference visualization
                diff_img = cv2.absdiff(small_previous.astype(np.uint8), small_current.astype(np.uint8))
                _, binary_diff = cv2.threshold(diff_img, self.config.change_threshold, 255, cv2.THRESH_BINARY)
                # Resize back to output resolution for consistency
                debug_change_diff = cv2.resize(binary_diff, self.config.output_resolution)
            
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
        
        # Create DepthFrame with optional debug images
        frame = DepthFrame(
            depth_image=output,
            color_image=color_image,
            hand_detected=hand_detected,
            change_score=change_score,
            frame_number=self.frame_number,
            timestamp=time.time()
        )
        
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
                cv2.circle(mask, center, radius, 255, -1)
                frame.importance_mask = mask
                
                # Mock masked image (apply the mask)
                frame.masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
                
                # Mock cropped before resize (add border)
                frame.cropped_before_resize = cv2.copyMakeBorder(depth_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=128)
                
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



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
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, List

import cv2
from experimance_core.depth_utils import debug_raw_depth_values
import numpy as np
import pyrealsense2 as rs  # type: ignore

from experimance_core.config import CameraState, CAMERA_RESET_TIMEOUT
from experimance_core.camera_utils import reset_realsense_camera_async

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """
    Robust RealSense camera interface with automatic error recovery.
    
    This class handles all low-level camera operations with retry logic.
    """
    
    def __init__(self, config):
        """Initialize the camera with configuration from config.py."""
        self.config = config
        self.state = CameraState.DISCONNECTED
        self.pipeline: Any = None
        self.profile: Any = None
        self.colorizer: Any = None
        self.align: Any = None
        self.filters: List[Tuple[str, Any]] = []  # Post-processing filters
        self.retry_count = 0
        self.current_retry_delay = config.retry_delay
        self._store_raw_depth = False  # Flag to store raw depth data for debugging
        
    async def initialize(self) -> bool:
        """Initialize the camera with retry logic."""
        result = await self._execute_with_retry("camera initialization", self._init_camera)
        return result is not None and result
    
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
                
            except asyncio.CancelledError:
                logger.info(f"Operation {operation_name} was cancelled")
                raise
                
            except Exception as e:
                last_exception = e
                self.retry_count += 1
                
                logger.warning(f"{operation_name} failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Attempt camera reset (now async and cancellable)
                    try:
                        reset_success = await asyncio.wait_for(
                            self._reset_camera(), 
                            timeout=CAMERA_RESET_TIMEOUT  # Overall timeout for reset operation
                        )
                        
                        if reset_success:
                            # Reset was successful, try the operation immediately
                            logger.info(f"Reset successful, retrying {operation_name} immediately...")
                            try:
                                result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
                                # Reset retry state on success
                                self.retry_count = 0
                                self.current_retry_delay = self.config.retry_delay
                                logger.info(f"Operation {operation_name} succeeded after reset!")
                                return result
                            except Exception as retry_e:
                                logger.warning(f"Operation {operation_name} failed even after successful reset: {retry_e}")
                                # Continue with normal retry delay logic
                        else:
                            logger.warning(f"Reset failed for {operation_name}, continuing with retry delay")
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Reset operation timed out for {operation_name}")
                    except asyncio.CancelledError:
                        logger.info(f"Reset operation was cancelled for {operation_name}")
                        raise
                    
                    # Exponential backoff (whether reset failed or post-reset operation failed)
                    delay = min(self.current_retry_delay * (2 ** attempt), self.config.max_retry_delay)
                    logger.info(f"Retrying {operation_name} in {delay:.1f}s...")
                    
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        logger.info(f"Retry delay was cancelled for {operation_name}")
                        raise
                else:
                    logger.error(f"All retry attempts failed for {operation_name}")
                    self.state = CameraState.ERROR
        
        return None
    
    async def _reset_camera(self) -> bool:
        """Reset the camera hardware and reinitialize (fully async and cancellable).
        
        Returns:
            True if reset and reinitialization were successful, False otherwise
        """
        self.state = CameraState.RESETTING
        logger.info("Resetting camera hardware...")
        
        try:
            # Stop current pipeline
            if self.pipeline:
                try:
                    self.pipeline.stop()
                    logger.debug("Pipeline stopped")
                except Exception as e:
                    logger.debug(f"Error stopping pipeline: {e}")
                self.pipeline = None
                self.profile = None
            
            # Hardware reset (now fully async and cancellable)
            success = await reset_realsense_camera_async(aggressive=self.config.aggressive_reset)
            
            if success:
                logger.info("Camera hardware reset successful")
                
                # Wait for device reinitialization (cancellable)
                await asyncio.sleep(3)
                
                # Reinitialize the camera pipeline after reset
                logger.info("Reinitializing camera pipeline after reset...")
                try:
                    init_success = self._init_camera()
                    if init_success:
                        logger.info("Camera pipeline reinitialized successfully")
                        self.state = CameraState.READY
                        return True
                    else:
                        logger.error("Failed to reinitialize camera pipeline after reset")
                        self.state = CameraState.ERROR
                        return False
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
                                return True
                            else:
                                logger.error("Failed to reinitialize even without advanced config")
                                self.state = CameraState.ERROR
                                return False
                        except Exception as e2:
                            logger.error(f"Error reinitializing without advanced config: {e2}")
                            self.state = CameraState.ERROR
                            return False
                    else:
                        self.state = CameraState.ERROR
                        return False
            else:
                logger.warning("Camera hardware reset failed")
                self.state = CameraState.ERROR
                return False
                
        except asyncio.CancelledError:
            logger.info("Camera reset was cancelled")
            self.state = CameraState.ERROR
            raise
        except Exception as e:
            logger.error(f"Error during camera reset: {e}")
            self.state = CameraState.ERROR
            return False
    
    def _init_camera(self) -> bool:
        """Initialize the camera pipeline (synchronous)."""
        self.state = CameraState.INITIALIZING
        
        # Configure streams
        self.pipeline = rs.pipeline() # type: ignore
        config = rs.config() # type: ignore
        
        config.enable_stream(
            rs.stream.depth,  # type: ignore
            self.config.resolution[0], 
            self.config.resolution[1], 
            rs.format.z16,  # type: ignore
            self.config.fps
        )
        
        if self.config.align_frames:
            config.enable_stream(
                rs.stream.color, # type: ignore
                self.config.resolution[0],
                self.config.resolution[1],
                rs.format.bgr8, # type: ignore
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
            self.align = rs.align(rs.stream.color) # type: ignore
        
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
            
            if self.profile is not None:
                device = self.profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device) # type: ignore
                
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
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset) # type: ignore
        for i in range(int(preset_range.max)):
            preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, i) # type: ignore
            if preset_name == "High Accuracy":
                depth_sensor.set_option(rs.option.visual_preset, i) # type: ignore
                logger.info("Set visual preset to High Accuracy")
                break
    
    def _setup_colorizer(self):
        """Setup depth colorizer."""
        self.colorizer = rs.colorizer(self.config.colorizer_scheme.value) # type: ignore
        self.colorizer.set_option(rs.option.visual_preset, 1)  # Fixed range # type: ignore
        self.colorizer.set_option(rs.option.min_distance, self.config.min_depth) # type: ignore
        self.colorizer.set_option(rs.option.max_distance, self.config.max_depth) # type: ignore
        self.colorizer.set_option(rs.option.color_scheme, self.config.colorizer_scheme.value) # type: ignore
    
    def _setup_post_processing(self):
        """Setup RealSense post-processing filters."""
        if not self.config.enable_filters:
            logger.info("Post-processing filters disabled")
            return
        
        self.filters = []
        
        # Decimation filter (reduces resolution)
        if self.config.decimation_filter:
            decimation = rs.decimation_filter() # type: ignore
            decimation.set_option(rs.option.filter_magnitude, self.config.decimation_filter_magnitude) # type: ignore
            self.filters.append(("decimation", decimation))
            logger.info(f"Enabled decimation filter (magnitude: {self.config.decimation_filter_magnitude})")
        
        # Threshold filter (depth range)
        if self.config.threshold_filter:
            threshold = rs.threshold_filter() # type: ignore
            threshold.set_option(rs.option.min_distance, self.config.threshold_filter_min) # type: ignore
            threshold.set_option(rs.option.max_distance, self.config.threshold_filter_max) # type: ignore
            self.filters.append(("threshold", threshold))
            logger.info(f"Enabled threshold filter (range: {self.config.threshold_filter_min}-{self.config.threshold_filter_max}m)")
        
        # Spatial filter (edge-preserving)
        if self.config.spatial_filter:
            # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.spatial_filter.html
            spatial = rs.spatial_filter() # type: ignore
            spatial.set_option(rs.option.filter_magnitude, self.config.spatial_filter_magnitude) # type: ignore
            spatial.set_option(rs.option.filter_smooth_alpha, self.config.spatial_filter_alpha) # type: ignore
            spatial.set_option(rs.option.filter_smooth_delta, self.config.spatial_filter_delta) # type: ignore
            spatial.set_option(rs.option.holes_fill, self.config.spatial_filter_hole_fill) # type: ignore
            self.filters.append(("spatial", spatial))
            logger.info(f"Enabled spatial filter (mag: {self.config.spatial_filter_magnitude}, "
                       f"alpha: {self.config.spatial_filter_alpha}, delta: {self.config.spatial_filter_delta})")
        
        # Temporal filter (reduces temporal noise)
        if self.config.temporal_filter:
            # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.temporal_filter.html
            temporal = rs.temporal_filter() # type: ignore
            temporal.set_option(rs.option.filter_smooth_alpha, self.config.temporal_filter_alpha) # type: ignore
            temporal.set_option(rs.option.filter_smooth_delta, self.config.temporal_filter_delta) # type: ignore
            temporal.set_option(rs.option.holes_fill, self.config.temporal_filter_persistence) # type: ignore
            self.filters.append(("temporal", temporal))
            logger.info(f"Enabled temporal filter (alpha: {self.config.temporal_filter_alpha}, "
                       f"delta: {self.config.temporal_filter_delta}, persistence: {self.config.temporal_filter_persistence})")
        
        # Hole filling filter
        if self.config.hole_filling_filter:
            hole_filling = rs.hole_filling_filter() # type: ignore
            hole_filling.set_option(rs.option.holes_fill, self.config.hole_filling_mode) # type: ignore
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

        # Get raw depth data first (in millimeters)
        # if self._store_raw_depth:
        #     raw_depth_data = np.asanyarray(depth_frame.get_data())
        #     # Store raw depth data for debugging (optional)
        #     self._last_raw_depth = raw_depth_data
        #raw_depth_data = np.asanyarray(depth_frame.get_data())
        #debug_raw_depth_values(raw_depth_data)
        
        # Colorize depth frame for visualization
        depth_color_frame = self.colorizer.colorize(filtered_depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())
        
        # Convert to grayscale if needed
        if len(depth_colormap.shape) == 3 and depth_colormap.shape[2] == 3:
            depth_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        else:
            depth_image = depth_colormap
        
        return depth_image, color_image
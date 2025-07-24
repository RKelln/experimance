#!/usr/bin/env python3
"""
Interactive detector parameter tuning tool.

Provides real-time visualization of detection results with adjustable parameters
via trackbars. Allows saving tuned parameters to detector profiles.

Usage:
    uv run python scripts/tune_detector.py [options]

Options:
    --profile PROFILE     Detector profile to use (default: indoor_office)
    --camera CAMERA_ID    Camera device ID (default: 0)  
    --camera-name NAME    Camera device name pattern (e.g. "EMEET", "HD Webcam")
    --list-cameras        List available cameras and exit
    --camera-info         Show detailed info about selected camera and exit
    --list-profiles       List available profiles and exit
    --verbose             Enable verbose logging

Examples:
    # Use specific camera device ID
    uv run python scripts/tune_detector.py --camera 6
    
    # Use camera by name pattern  
    uv run python scripts/tune_detector.py --camera-name "EMEET"
    uv run python scripts/tune_detector.py --camera-name "HD Webcam"
    
    # Show detailed camera information
    uv run python scripts/tune_detector.py --camera-name "EMEET" --camera-info
    uv run python scripts/tune_detector.py --camera 6 --camera-info
    
    # List available cameras and profiles
    uv run python scripts/tune_detector.py --list-cameras
    uv run python scripts/tune_detector.py --list-profiles
"""

import cv2
import numpy as np
import argparse
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from .cpu_audience_detector import CPUAudienceDetector
from .detector_profile import (
    DetectorProfile, load_profile, create_default_profiles,
    get_profile_directory, list_available_profiles
)
from .webcam import WebcamManager
from ..config import VisionConfig

logger = logging.getLogger(__name__)


class InteractiveDetectorTuner:
    """Interactive detector parameter tuning with live visualization."""
    
    def __init__(self, camera_id: int = 0, camera_name: Optional[str] = None, profile_name: str = "indoor_office"):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.profile_name = profile_name
        self.cap = None
        self.webcam_manager = None
        self.detector = None
        self.profile = None
        
        # UI state
        self.window_name = "Detector Tuning"
        self.controls_window = "Controls"
        self.running = True
        self.show_hog = True
        self.show_faces = True
        self.show_motion = True
        self.show_info = True
        self.detector_needs_reload = False
        self.setup_complete = False  # Prevent callback during setup
        self._debug_motion = False  # Debug motion detection
        
        # Current parameters (will be set from profile)
        self.params = {
            # Detection params - default values
            'scale_factor': 50,  # 0.5 * 100
            'min_height': 80,
            'motion_threshold': 1000,
            'motion_intensity': 10,  # 0.01 * 1000
            'stability': 50,  # 0.5 * 100
            
            # HOG params - default values
            'hog_threshold': 100,  # 0.0 * 100 + 100 
            'hog_scale': 105,  # 1.05 * 100
            'win_stride': 8,
            
            # MOG2 params - default values
            'mog2_var_threshold': 25,
            'mog2_history': 200,
            
            # Confidence params - default values
            'person_base_conf': 30,  # 0.3 * 100
            'motion_weight': 50,  # 0.5 * 100
        }
        
        # Create a single event loop for async operations
        self.loop = None
        
    def setup_camera(self) -> bool:
        """Initialize camera capture using WebcamManager."""
        try:
            # Create a VisionConfig for the webcam manager
            webcam_config = VisionConfig(
                webcam_enabled=True,
                webcam_device_id=self.camera_id,
                webcam_device_name=self.camera_name,
                webcam_auto_detect=True,  # Enable auto-detection as fallback
                webcam_width=640,
                webcam_height=480,
                webcam_fps=30
            )
            
            # Create and start webcam manager
            self.webcam_manager = WebcamManager(webcam_config)
            
            # Use asyncio to start the webcam manager
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            
            self.loop.run_until_complete(self.webcam_manager.start())
            
            # Apply camera settings from profile if available
            if hasattr(self, 'profile') and self.profile and self.profile.camera:
                logger.info(f"Applying camera profile: {self.profile.camera.name}")
                success = self.webcam_manager.apply_camera_profile(self.profile.camera)
                if success:
                    print(f"Applied camera profile: {self.profile.camera.name}")
                else:
                    print(f"Warning: Camera profile '{self.profile.camera.name}' had limited success")
            
            # Test capturing a frame
            test_frame = self.loop.run_until_complete(self.webcam_manager.capture_frame())
            if test_frame is None:
                raise RuntimeError("Failed to capture test frame")
            
            # Get info about the webcam setup
            info = self.webcam_manager.get_capture_info()
            actual_device_id = info.get('device_id', 'unknown')
            resolution = info.get('actual_resolution', 'unknown')
            
            print(f"Successfully opened camera {actual_device_id} ({resolution})")
            if self.camera_name:
                print(f"Camera name pattern: '{self.camera_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            print(f"Error: Could not open camera")
            if self.camera_name:
                print(f"Camera name pattern: '{self.camera_name}'")
            print(f"Camera ID: {self.camera_id}")
            print("Make sure:")
            print("  - Camera is connected and not used by another application")
            print("  - Try a different camera ID (0, 1, 2, etc.) or camera name")
            print("  - Check camera permissions")
            print("  - Use 'v4l2-ctl --list-devices' to see available cameras")
            return False
    
    def setup_detector(self) -> bool:
        """Initialize detector with profile."""
        try:
            # Profile should already be loaded by run()
            if not self.profile:
                raise RuntimeError("Profile not loaded - setup_detector() called before run()")
            
            logger.info(f"Using profile: {self.profile.name}")
            
            # Create a minimal VisionConfig for the detector
            vision_config = VisionConfig(
                webcam_enabled=True,
                audience_detection_enabled=True,  # Explicitly enable detection
                detector_profile=self.profile_name,
                detector_profile_dir=get_profile_directory()
            )
            
            # Create detector with config
            self.detector = CPUAudienceDetector(vision_config)
            
            # Set up event loop for async operations
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            
            # Initialize/start the detector
            self.loop.run_until_complete(self.detector.start())
            print("Detector started successfully")
            
            # Extract current parameters for UI
            self._update_params_from_profile()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup detector: {e}")
            print(f"Error setting up detector: {e}")
            return False
    
    def _update_params_from_profile(self):
        """Update UI parameters from current profile."""
        if self.profile is None:
            print("Warning: No profile loaded, using default parameter values")
            return
        
        self.params = {
            # Detection params
            'scale_factor': int(self.profile.detection.detection_scale_factor * 100),
            'min_height': self.profile.detection.min_person_height,
            'motion_threshold': self.profile.detection.motion_threshold,
            'motion_intensity': int(self.profile.detection.motion_intensity_threshold * 1000),
            'stability': int(self.profile.detection.stability_threshold * 100),
            
            # HOG params
            'hog_threshold': int((self.profile.hog.hit_threshold + 1.0) * 100),  # Scale to 0-200
            'hog_scale': int(self.profile.hog.scale * 100),
            'win_stride': self.profile.hog.win_stride_x,
            
            # Face detection params
            'face_score_threshold': int(self.profile.face.score_threshold * 100),
            'face_nms_threshold': int(self.profile.face.nms_threshold * 100),
            'face_min_size': self.profile.face.min_face_size,
            'face_top_k': min(100, self.profile.face.top_k),  # Limit for trackbar
            
            # MOG2 params
            'mog2_var_threshold': int(self.profile.mog2.var_threshold),
            'mog2_history': self.profile.mog2.history,
            
            # Confidence params
            'person_base_conf': int(self.profile.confidence.person_base_confidence * 100),
            'motion_weight': int(self.profile.confidence.motion_confidence_weight * 100),
        }
    
    
    def setup_ui(self):
        """Setup OpenCV windows and trackbars."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.controls_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.controls_window, 400, 800)
        
        # Detection parameters
        cv2.createTrackbar('Scale Factor %', self.controls_window, 
                          self.params['scale_factor'], 100, self.on_param_change)
        cv2.createTrackbar('Min Height', self.controls_window,
                          self.params['min_height'], 200, self.on_param_change)
        cv2.createTrackbar('Motion Threshold', self.controls_window,
                          self.params['motion_threshold'], 5000, self.on_param_change)
        cv2.createTrackbar('Motion Intensity x1000', self.controls_window,
                          self.params['motion_intensity'], 100, self.on_param_change)
        cv2.createTrackbar('Stability %', self.controls_window,
                          self.params['stability'], 100, self.on_param_change)
        
        # HOG parameters
        cv2.createTrackbar('HOG Threshold x100', self.controls_window,
                          self.params['hog_threshold'], 200, self.on_param_change)
        cv2.createTrackbar('HOG Scale x100', self.controls_window,
                          self.params['hog_scale'], 150, self.on_param_change)
        cv2.createTrackbar('Win Stride', self.controls_window,
                          self.params['win_stride'], 20, self.on_param_change)
        
        # Face detection parameters
        cv2.createTrackbar('Face Score Thresh %', self.controls_window,
                          self.params['face_score_threshold'], 100, self.on_param_change)
        cv2.createTrackbar('Face NMS Thresh %', self.controls_window,
                          self.params['face_nms_threshold'], 100, self.on_param_change)
        cv2.createTrackbar('Face Min Size', self.controls_window,
                          self.params['face_min_size'], 100, self.on_param_change)
        cv2.createTrackbar('Face Top K', self.controls_window,
                          self.params['face_top_k'], 100, self.on_param_change)
        
        # MOG2 parameters  
        cv2.createTrackbar('MOG2 Var Threshold', self.controls_window,
                          self.params['mog2_var_threshold'], 100, self.on_param_change)
        cv2.createTrackbar('MOG2 History', self.controls_window,
                          self.params['mog2_history'], 1000, self.on_param_change)
        
        # Confidence parameters
        cv2.createTrackbar('Person Base Conf %', self.controls_window,
                          self.params['person_base_conf'], 100, self.on_param_change)
        cv2.createTrackbar('Motion Weight %', self.controls_window,
                          self.params['motion_weight'], 100, self.on_param_change)
        
        # Display toggles
        cv2.createTrackbar('Show HOG', self.controls_window, 1, 1, self.on_toggle_change)
        cv2.createTrackbar('Show Faces', self.controls_window, 1, 1, self.on_toggle_change)
        cv2.createTrackbar('Show Motion', self.controls_window, 1, 1, self.on_toggle_change)
        cv2.createTrackbar('Show Info', self.controls_window, 1, 1, self.on_toggle_change)
        
        # Add instructions to the controls window title
        cv2.setWindowTitle(self.controls_window, "Controls - SPACEBAR to reload, R to reset")
        
        # Mark setup as complete to enable callbacks
        self.setup_complete = True
        profile_name = self.profile.name if self.profile else "Unknown"
        print(f"Initialized trackbars with profile '{profile_name}' values")
    
    
    
    def on_param_change(self, val):
        """Handle parameter changes from trackbars."""
        # Skip callbacks during initial setup
        if not self.setup_complete:
            return
            
        # Read all current values
        self.params['scale_factor'] = cv2.getTrackbarPos('Scale Factor %', self.controls_window)
        self.params['min_height'] = cv2.getTrackbarPos('Min Height', self.controls_window)
        self.params['motion_threshold'] = cv2.getTrackbarPos('Motion Threshold', self.controls_window)
        self.params['motion_intensity'] = cv2.getTrackbarPos('Motion Intensity x1000', self.controls_window)
        self.params['stability'] = cv2.getTrackbarPos('Stability %', self.controls_window)
        self.params['hog_threshold'] = cv2.getTrackbarPos('HOG Threshold x100', self.controls_window)
        self.params['hog_scale'] = cv2.getTrackbarPos('HOG Scale x100', self.controls_window)
        self.params['win_stride'] = cv2.getTrackbarPos('Win Stride', self.controls_window)
        self.params['face_score_threshold'] = cv2.getTrackbarPos('Face Score Thresh %', self.controls_window)
        self.params['face_nms_threshold'] = cv2.getTrackbarPos('Face NMS Thresh %', self.controls_window)
        self.params['face_min_size'] = cv2.getTrackbarPos('Face Min Size', self.controls_window)
        self.params['face_top_k'] = cv2.getTrackbarPos('Face Top K', self.controls_window)
        self.params['mog2_var_threshold'] = cv2.getTrackbarPos('MOG2 Var Threshold', self.controls_window)
        self.params['mog2_history'] = cv2.getTrackbarPos('MOG2 History', self.controls_window)
        self.params['person_base_conf'] = cv2.getTrackbarPos('Person Base Conf %', self.controls_window)
        self.params['motion_weight'] = cv2.getTrackbarPos('Motion Weight %', self.controls_window)
        
        # Mark that detector needs reload
        self.detector_needs_reload = True
    
    def on_toggle_change(self, val):
        """Handle display toggle changes."""
        self.show_hog = cv2.getTrackbarPos('Show HOG', self.controls_window) == 1
        self.show_faces = cv2.getTrackbarPos('Show Faces', self.controls_window) == 1
        self.show_motion = cv2.getTrackbarPos('Show Motion', self.controls_window) == 1
        self.show_info = cv2.getTrackbarPos('Show Info', self.controls_window) == 1
    
    def reload_detector(self):
        """Reload detector with current parameter values."""
        if self.profile is None:
            return
            
        try:
            print("Reloading detector with new parameters...")
            
            # Update profile with current parameters
            self.profile.detection.detection_scale_factor = max(0.1, self.params['scale_factor'] / 100.0)
            self.profile.detection.min_person_height = max(20, self.params['min_height'])
            self.profile.detection.motion_threshold = max(100, self.params['motion_threshold'])
            self.profile.detection.motion_intensity_threshold = max(0.001, self.params['motion_intensity'] / 1000.0)
            self.profile.detection.stability_threshold = max(0.1, self.params['stability'] / 100.0)
            
            self.profile.hog.hit_threshold = (self.params['hog_threshold'] / 100.0) - 1.0
            self.profile.hog.scale = max(1.01, self.params['hog_scale'] / 100.0)
            self.profile.hog.win_stride_x = max(1, self.params['win_stride'])
            self.profile.hog.win_stride_y = max(1, self.params['win_stride'])
            
            self.profile.face.score_threshold = max(0.1, self.params['face_score_threshold'] / 100.0)
            self.profile.face.nms_threshold = max(0.1, self.params['face_nms_threshold'] / 100.0)
            self.profile.face.min_face_size = max(10, self.params['face_min_size'])
            self.profile.face.top_k = max(1, self.params['face_top_k'] * 50)  # Scale back up
            
            self.profile.mog2.var_threshold = max(10, float(self.params['mog2_var_threshold']))
            self.profile.mog2.history = max(50, self.params['mog2_history'])
            
            self.profile.confidence.person_base_confidence = max(0.0, self.params['person_base_conf'] / 100.0)
            self.profile.confidence.motion_confidence_weight = max(0.0, self.params['motion_weight'] / 100.0)
            
            # Print current parameter values for debugging
            print(f"  Scale factor: {self.profile.detection.detection_scale_factor}")
            print(f"  Min person height: {self.profile.detection.min_person_height}")
            print(f"  Motion threshold: {self.profile.detection.motion_threshold}")
            print(f"  HOG threshold: {self.profile.hog.hit_threshold}")
            print(f"  MOG2 var threshold: {self.profile.mog2.var_threshold}")
            
            # Save updated profile temporarily
            temp_profile_path = get_profile_directory() / f"temp_tuning_{self.profile_name}.toml"
            self.profile.save_to_file(temp_profile_path)
            
            # Create new VisionConfig with temp profile
            vision_config = VisionConfig(
                webcam_enabled=True,
                audience_detection_enabled=True,  # Explicitly enable detection
                detector_profile=f"temp_tuning_{self.profile_name}",
                detector_profile_dir=get_profile_directory()
            )
            
            # Recreate detector
            self.detector = CPUAudienceDetector(vision_config)
            
            # Start the new detector
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.detector.start())
            
            self.detector_needs_reload = False
            
            print("Detector reloaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to reload detector: {e}")
            print(f"Error reloading detector: {e}")
    
    def draw_detections(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw detection visualizations on frame."""
        vis_frame = frame.copy()
        
        # Draw HOG detections
        if self.show_hog and 'person_detection' in result:
            person_det = result['person_detection']
            if 'detections' in person_det:
                for (x, y, w, h) in person_det['detections']:
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(vis_frame, 'HOG', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw face detections
        if self.show_faces and 'face_detection' in result:
            face_det = result['face_detection']
            if 'detections' in face_det:
                for (x, y, w, h) in face_det['detections']:
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(vis_frame, 'FACE', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw motion areas
        if self.show_motion and 'motion_detection' in result:
            motion_det = result['motion_detection']
            if 'motion_mask' in motion_det and motion_det['motion_mask'] is not None:
                motion_mask = motion_det['motion_mask']
                try:
                    # Ensure motion_mask is uint8 and single channel
                    if len(motion_mask.shape) == 3:
                        motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_BGR2GRAY)
                    if motion_mask.dtype != np.uint8:
                        motion_mask = motion_mask.astype(np.uint8)
                    
                    # Resize motion mask to match frame size if needed
                    frame_height, frame_width = vis_frame.shape[:2]
                    mask_height, mask_width = motion_mask.shape
                    if (mask_height != frame_height) or (mask_width != frame_width):
                        motion_mask = cv2.resize(motion_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                        
                        # Debug resize operation
                        if hasattr(self, '_debug_motion') and getattr(self, '_frame_count', 0) % 60 == 0:
                            print(f"Resized motion mask: {mask_width}x{mask_height} → {frame_width}x{frame_height}")
                    
                    # Find contours in motion mask
                    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) > 0:
                        # Draw contour outlines in red
                        cv2.drawContours(vis_frame, contours, -1, (0, 0, 255), 2)
                        
                        # Draw filled contours with transparency for better visibility
                        mask_overlay = np.zeros_like(vis_frame)
                        cv2.drawContours(mask_overlay, contours, -1, (0, 0, 255), -1)
                        vis_frame = cv2.addWeighted(vis_frame, 0.8, mask_overlay, 0.2, 0)
                        
                except Exception as e:
                    if hasattr(self, '_debug_motion'):
                        print(f"Error drawing motion: {e}")
            elif hasattr(self, '_debug_motion') and getattr(self, '_frame_count', 0) % 60 == 0:
                # Debug: Show why motion isn't being drawn
                if 'motion_detection' not in result:
                    print("No 'motion_detection' in result")
                elif 'motion_mask' not in motion_det:
                    print("No 'motion_mask' in motion_detection")
                elif motion_det['motion_mask'] is None:
                    print("motion_mask is None")
        
        
        # Draw overall detection status
        presence = result.get('audience_detected', False)
        confidence = result.get('confidence', 0.0)
        
        status_color = (0, 255, 0) if presence else (0, 0, 255)
        status_text = f"PRESENCE: {presence} ({confidence:.2f})"
        cv2.putText(vis_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw error message if present
        if 'error' in result:
            error_text = f"ERROR: {result['error'][:50]}..."  # Truncate long errors
            cv2.putText(vis_frame, error_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw reload status
        if self.detector_needs_reload:
            cv2.putText(vis_frame, "RELOAD NEEDED - Press SPACEBAR to apply changes", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw parameter info
        if self.show_info:
            info_lines = [
                f"Scale: {self.params['scale_factor']}%",
                f"MinH: {self.params['min_height']}",
                f"MotThresh: {self.params['motion_threshold']}",
                f"HOG: {self.params['hog_threshold']/100-1:.2f}",
                f"MOG2Var: {self.params['mog2_var_threshold']}",
                f"HOG Dets: {len(result.get('person_detection', {}).get('detections', []))}",
                f"Face Dets: {len(result.get('face_detection', {}).get('detections', []))}",
                f"Motion: {result.get('motion_detection', {}).get('motion_intensity', 0.0):.3f}",
                f"Person Count: {result.get('person_detection', {}).get('person_count', 0)}"
            ]
            
            start_y = 90 if self.detector_needs_reload else 70
            for i, line in enumerate(info_lines):
                cv2.putText(vis_frame, line, (10, start_y + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_frame
    
    def save_profile(self, profile_name: Optional[str] = None):
        """Save current parameters to a profile."""
        if self.profile is None:
            logger.error("No profile loaded")
            return False
            
        if profile_name is None:
            profile_name = f"{self.profile_name}_tuned"
        
        try:
            # Update profile name and description
            self.profile.name = f"{self.profile.name} (Tuned)"
            self.profile.description = f"{self.profile.description} - Interactively tuned"
            
            # Save to file
            profile_dir = get_profile_directory()
            profile_path = profile_dir / f"{profile_name}.toml"
            self.profile.save_to_file(profile_path)
            
            print(f"\nSaved tuned profile to: {profile_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False
    
    def run(self):
        """Main tuning loop."""
        # Load profile first to get camera settings
        try:
            self.profile = load_profile(self.profile_name)
            logger.info(f"Loaded profile: {self.profile.name}")
        except Exception as e:
            logger.error(f"Failed to load profile '{self.profile_name}': {e}")
            print(f"Error: Failed to load detector profile '{self.profile_name}': {e}")
            return False
        
        if not self.setup_camera():
            return False
        
        if not self.setup_detector():
            return False
        
        self.setup_ui()
        
        print(f"\nInteractive Detector Tuning")
        if self.profile:
            print(f"Profile: {self.profile.name}")
        print(f"Camera: {self.camera_id}")
        print(f"\nControls:")
        print(f"  q/ESC: Quit")
        print(f"  s: Save current settings to '{self.profile_name}_tuned.toml'")
        print(f"  r: Reset to original profile")
        print(f"  SPACEBAR: Reload detector with current parameters")
        print(f"  h: Toggle HOG detection display")
        print(f"  f: Toggle face detection display")
        print(f"  m: Toggle motion detection display")
        print(f"  i: Toggle info display")
        print(f"  d: Toggle motion debug output")
        print(f"\nAdjust parameters using the trackbars in the Controls window")
        print(f"Press SPACEBAR to apply changes to detector")
        print(f"\nDetection Tips:")
        print(f"  - Try lowering HOG Threshold (more sensitive detection)")
        print(f"  - Increase Scale Factor for better quality (slower)")
        print(f"  - Lower Motion Threshold for more motion sensitivity")
        print(f"  - Move around to trigger motion detection")
        print(f"  - Check that HOG detections and motion are showing on screen")
        
        try:
            while self.running:
                if self.webcam_manager is None or self.detector is None:
                    break
                
                # Ensure loop is available
                if self.loop is None:
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                    
                # Capture frame using WebcamManager
                frame = self.loop.run_until_complete(self.webcam_manager.capture_frame())
                if frame is None:
                    break
                
                # Run detection (handle async)
                try:
                    if self.loop is None:
                        self.loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(self.loop)
                    
                    # Add timing debug
                    import time
                    start_time = time.perf_counter()
                    result = self.loop.run_until_complete(
                        self.detector.detect_audience(frame, include_motion_mask=True)
                    )
                    detection_time = (time.perf_counter() - start_time) * 1000.0  # Convert to milliseconds
                    
                    # Debug output every 30 frames (roughly once per second)
                    frame_count = getattr(self, '_frame_count', 0)
                    self._frame_count = frame_count + 1
                    if frame_count % 30 == 0:
                        print(f"Detection: {detection_time:.1f}ms | "
                              f"HOG: {len(result.get('person_detection', {}).get('detections', []))} | "
                              f"Faces: {len(result.get('face_detection', {}).get('detections', []))} | "
                              f"Motion: {result.get('motion_detection', {}).get('motion_intensity', 0.0):.3f} | "
                              f"Presence: {result.get('audience_detected', False)}")
                        if 'error' in result:
                            print(f"  ERROR: {result['error']}")
                    
                except Exception as e:
                    logger.warning(f"Detection failed: {e}")
                    # Create dummy result for visualization
                    result = {
                        'audience_detected': False,
                        'confidence': 0.0,
                        'person_detection': {'detections': [], 'count': 0},
                        'motion_detection': {'motion_intensity': 0.0},
                        'error': str(e)
                    }
                
                # Draw visualizations
                vis_frame = self.draw_detections(frame, result)
                
                # Show frame
                cv2.imshow(self.window_name, vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('s'):
                    self.save_profile()
                elif key == ord('r'):
                    self.reset_to_original()
                elif key == ord(' '):  # SPACEBAR for reload
                    self.reload_detector()
                elif key == ord('h'):
                    self.show_hog = not self.show_hog
                    cv2.setTrackbarPos('Show HOG', self.controls_window, int(self.show_hog))
                elif key == ord('f'):
                    self.show_faces = not self.show_faces
                    cv2.setTrackbarPos('Show Faces', self.controls_window, int(self.show_faces))
                elif key == ord('m'):
                    self.show_motion = not self.show_motion
                    cv2.setTrackbarPos('Show Motion', self.controls_window, int(self.show_motion))
                elif key == ord('i'):
                    self.show_info = not self.show_info
                    cv2.setTrackbarPos('Show Info', self.controls_window, int(self.show_info))
                elif key == ord('d'):
                    self._debug_motion = not self._debug_motion
                    print(f"Motion debug: {'ON' if self._debug_motion else 'OFF'}")
        
        finally:
            self.cleanup()
        
        return True
    
    def reset_to_original(self):
        """Reset parameters to original profile values."""
        try:
            # Reload original profile
            self.profile = load_profile(self.profile_name)
            self._update_params_from_profile()
            
            # Update ALL trackbar positions with original values
            cv2.setTrackbarPos('Scale Factor %', self.controls_window, self.params['scale_factor'])
            cv2.setTrackbarPos('Min Height', self.controls_window, self.params['min_height'])
            cv2.setTrackbarPos('Motion Threshold', self.controls_window, self.params['motion_threshold'])
            cv2.setTrackbarPos('Motion Intensity x1000', self.controls_window, self.params['motion_intensity'])
            cv2.setTrackbarPos('Stability %', self.controls_window, self.params['stability'])
            cv2.setTrackbarPos('HOG Threshold x100', self.controls_window, self.params['hog_threshold'])
            cv2.setTrackbarPos('HOG Scale x100', self.controls_window, self.params['hog_scale'])
            cv2.setTrackbarPos('Win Stride', self.controls_window, self.params['win_stride'])
            cv2.setTrackbarPos('Face Score Thresh %', self.controls_window, self.params['face_score_threshold'])
            cv2.setTrackbarPos('Face NMS Thresh %', self.controls_window, self.params['face_nms_threshold'])
            cv2.setTrackbarPos('Face Min Size', self.controls_window, self.params['face_min_size'])
            cv2.setTrackbarPos('Face Top K', self.controls_window, self.params['face_top_k'])
            cv2.setTrackbarPos('MOG2 Var Threshold', self.controls_window, self.params['mog2_var_threshold'])
            cv2.setTrackbarPos('MOG2 History', self.controls_window, self.params['mog2_history'])
            cv2.setTrackbarPos('Person Base Conf %', self.controls_window, self.params['person_base_conf'])
            cv2.setTrackbarPos('Motion Weight %', self.controls_window, self.params['motion_weight'])
            
            # Reload detector with original profile
            self.setup_detector()
            self.detector_needs_reload = False
            
            print("Reset to original profile values and reloaded detector")
            
        except Exception as e:
            logger.error(f"Failed to reset profile: {e}")
            print(f"Error resetting profile: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.webcam_manager:
            if self.loop and not self.loop.is_closed():
                try:
                    self.loop.run_until_complete(self.webcam_manager.stop())
                except:
                    pass
            self.webcam_manager = None
        cv2.destroyAllWindows()
        
        # Clean up event loop
        if self.loop and not self.loop.is_closed():
            try:
                self.loop.close()
            except:
                pass
        
        # Clean up temporary profile file
        try:
            temp_profile_path = get_profile_directory() / f"temp_tuning_{self.profile_name}.toml"
            if temp_profile_path.exists():
                temp_profile_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temporary profile: {e}")


def get_camera_detailed_info(camera_id: Optional[int] = None, camera_name: Optional[str] = None) -> None:
    """Display detailed information about a camera."""
    import subprocess
    from .webcam import WebcamManager
    from ..config import VisionConfig
    
    print("=" * 60)
    print("CAMERA DETAILED INFORMATION")
    print("=" * 60)
    
    # Determine device ID
    device_id = camera_id
    if camera_name:
        # Use WebcamManager to find device by name
        config = VisionConfig(webcam_device_name=camera_name, webcam_device_id=camera_id or 0)
        manager = WebcamManager(config)
        device_id = manager._find_device_by_name(camera_name)
        if device_id is None:
            print(f"Camera '{camera_name}' not found")
            return
        print(f"Camera name pattern: '{camera_name}' → Device ID: {device_id}")
    else:
        device_id = camera_id or 0
        print(f"Using device ID: {device_id}")
    
    device_path = f"/dev/video{device_id}"
    
    try:
        # Check if device exists
        import os
        if not os.path.exists(device_path):
            print(f"Error: Device {device_path} does not exist")
            return
        
        print(f"Device: {device_path}")
        print()
        
        # Get device name from v4l2-ctl --list-devices
        print("1. DEVICE IDENTIFICATION")
        print("-" * 30)
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                current_device_name = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith('/dev/') and not line.startswith('\t') and not line.startswith(' '):
                        current_device_name = line
                    elif line.startswith(f'/dev/video{device_id}') and current_device_name:
                        print(f"Name: {current_device_name}")
                        break
        except:
            print("Could not get device name")
        
        # Get supported formats and resolutions
        print("\n2. SUPPORTED FORMATS & RESOLUTIONS")
        print("-" * 40)
        try:
            result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--list-formats-ext'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Could not get format information")
        except Exception as e:
            print(f"Error getting formats: {e}")
        
        # Get current camera settings
        print("\n3. CURRENT CAMERA SETTINGS")
        print("-" * 30)
        try:
            result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--get-fmt-video'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Could not get current settings")
        except Exception as e:
            print(f"Error getting current settings: {e}")
        
        # Get camera controls
        print("\n4. AVAILABLE CAMERA CONTROLS")
        print("-" * 35)
        try:
            result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--list-ctrls'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Could not get camera controls")
        except Exception as e:
            print(f"Error getting camera controls: {e}")
        
        # Test with OpenCV
        print("\n5. OPENCV COMPATIBILITY TEST")
        print("-" * 35)
        try:
            import cv2
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                # Get OpenCV reported properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
                
                print(f"OpenCV can open device: YES")
                print(f"Current resolution: {width}x{height}")
                print(f"Current FPS: {fps}")
                print(f"Current format: {fourcc_str}")
                
                # Test frame capture
                ret, frame = cap.read()
                if ret:
                    print(f"Frame capture: SUCCESS")
                    print(f"Actual frame shape: {frame.shape}")
                else:
                    print(f"Frame capture: FAILED")
                
                cap.release()
            else:
                print(f"OpenCV can open device: NO")
        except Exception as e:
            print(f"OpenCV test failed: {e}")
        
        # WebcamManager compatibility test
        print("\n6. WEBCAM MANAGER TEST")
        print("-" * 28)
        try:
            import asyncio
            
            async def test_webcam_manager():
                config = VisionConfig(
                    webcam_enabled=True,
                    webcam_device_id=device_id,
                    webcam_device_name=camera_name,
                    webcam_width=640,
                    webcam_height=480,
                    webcam_fps=30
                )
                
                manager = WebcamManager(config)
                await manager.start()
                
                info = manager.get_capture_info()
                print("WebcamManager compatibility: SUCCESS")
                print(f"Configuration:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                
                # Test frame capture
                frame = await manager.capture_frame()
                if frame is not None:
                    print(f"Frame capture via WebcamManager: SUCCESS")
                    print(f"Frame shape: {frame.shape}")
                else:
                    print(f"Frame capture via WebcamManager: FAILED")
                
                await manager.stop()
            
            asyncio.run(test_webcam_manager())
            
        except Exception as e:
            print(f"WebcamManager test failed: {e}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error getting camera information: {e}")


def setup_camera_for_gallery(camera_id: Optional[int] = None, camera_name: Optional[str] = None) -> None:
    """Configure camera settings optimized for gallery/dim lighting environment."""
    import subprocess
    from .webcam import WebcamManager
    from ..config import VisionConfig
    
    print("=" * 60)
    print("CAMERA SETUP FOR GALLERY ENVIRONMENT")
    print("=" * 60)
    
    # Determine device ID
    device_id = camera_id
    if camera_name:
        # Use WebcamManager to find device by name
        config = VisionConfig(webcam_device_name=camera_name, webcam_device_id=camera_id or 0)
        manager = WebcamManager(config)
        device_id = manager._find_device_by_name(camera_name)
        if device_id is None:
            print(f"Camera '{camera_name}' not found")
            return
        print(f"Camera name pattern: '{camera_name}' → Device ID: {device_id}")
    else:
        device_id = camera_id or 0
        print(f"Using device ID: {device_id}")
    
    device_path = f"/dev/video{device_id}"
    
    try:
        # Check if device exists
        import os
        if not os.path.exists(device_path):
            print(f"Error: Device {device_path} does not exist")
            return
        
        print(f"Device: {device_path}")
        print()
        
        # Gallery-optimized camera settings
        gallery_settings = {
            # Exposure settings for low light
            'auto_exposure': 3,  # Aperture Priority Mode (best for low light)
            'exposure_dynamic_framerate': 1,  # Allow frame rate to drop for longer exposure
            
            # Lighting compensation
            'backlight_compensation': 60,  # Higher value for uneven gallery lighting
            'gain': 20,  # Moderate gain boost for sensitivity
            
            # White balance for gallery lighting
            'white_balance_automatic': 1,  # Auto white balance
            'white_balance_temperature': 4000,  # Slightly warmer for gallery lights
            
            # Image quality for detection
            'brightness': 10,  # Slightly brighter for detection
            'contrast': 40,  # Higher contrast for better edges
            'saturation': 50,  # Reduce saturation to avoid color bleeding
            'sharpness': 4,  # Higher sharpness for detection
            'gamma': 120,  # Adjust gamma for better dynamic range
            
            # Power line frequency (adjust for your region)
            'power_line_frequency': 2,  # 60 Hz (use 1 for 50 Hz in Europe)
        }
        
        print("1. CURRENT CAMERA SETTINGS")
        print("-" * 30)
        
        # Get current settings
        try:
            result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--get-ctrl=auto_exposure,exposure_dynamic_framerate,backlight_compensation,gain,brightness,contrast'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print("Current settings:")
                print(result.stdout)
            else:
                print("Could not get current settings")
        except Exception as e:
            print(f"Error getting current settings: {e}")
        
        print("\n2. APPLYING GALLERY-OPTIMIZED SETTINGS")
        print("-" * 45)
        
        success_count = 0
        total_count = len(gallery_settings)
        
        for setting, value in gallery_settings.items():
            try:
                print(f"Setting {setting} = {value}... ", end="")
                result = subprocess.run(['v4l2-ctl', f'--device={device_path}', f'--set-ctrl={setting}={value}'], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ SUCCESS")
                    success_count += 1
                else:
                    print(f"✗ FAILED: {result.stderr.strip()}")
            except Exception as e:
                print(f"✗ ERROR: {e}")
        
        print(f"\nApplied {success_count}/{total_count} settings successfully")
        
        print("\n3. RECOMMENDED FORMAT SETTINGS")
        print("-" * 35)
        print("For gallery environments, consider using:")
        print("  Format: MJPG (better compression, higher resolutions)")
        print("  Resolution options:")
        print("    - 1280x720 (balanced performance/quality)")
        print("    - 1920x1080 (best quality for face detection)")
        print("    - 640x480 (fastest performance)")
        print()
        print("To set format and resolution:")
        print(f"  v4l2-ctl --device={device_path} --set-fmt-video=width=1280,height=720,pixelformat=MJPG")
        
        # Apply recommended format
        print("\n4. APPLYING RECOMMENDED FORMAT")
        print("-" * 35)
        try:
            print("Setting format to MJPG 1280x720... ", end="")
            result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--set-fmt-video=width=1280,height=720,pixelformat=MJPG'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ SUCCESS")
                
                # Verify the setting
                result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--get-fmt-video'], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    print("New format:")
                    print(result.stdout)
            else:
                print(f"✗ FAILED: {result.stderr.strip()}")
                print("Keeping current format")
        except Exception as e:
            print(f"✗ ERROR: {e}")
        
        print("\n5. VERIFICATION")
        print("-" * 15)
        
        # Test with OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                # Get properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"OpenCV reports: {width}x{height} @ {fps} FPS")
                
                # Test frame capture
                ret, frame = cap.read()
                if ret:
                    print(f"Frame capture: ✓ SUCCESS")
                    print(f"Frame shape: {frame.shape}")
                    
                    # Check brightness/quality
                    mean_brightness = frame.mean()
                    print(f"Mean brightness: {mean_brightness:.1f} (good range: 80-150)")
                else:
                    print(f"Frame capture: ✗ FAILED")
                
                cap.release()
            else:
                print(f"✗ OpenCV cannot open device")
        except Exception as e:
            print(f"Verification failed: {e}")
        
        print("\n6. USAGE RECOMMENDATIONS")
        print("-" * 28)
        print("For gallery use:")
        print("  • Use face_detection profile for seated audiences")
        print("  • Monitor detection performance in actual lighting")
        print("  • Adjust exposure settings if needed:")
        print(f"    v4l2-ctl --device={device_path} --set-ctrl=backlight_compensation=80")
        print(f"    v4l2-ctl --device={device_path} --set-ctrl=gain=30")
        print("  • Test with: uv run python scripts/tune_detector.py --profile face_detection --camera-name 'EMEET'")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error setting up camera: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive detector parameter tuning")
    parser.add_argument('--profile', '-p', default='face_detection',
                       help='Detector profile to start with')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--camera-name', '-n', type=str, default=None,
                       help='Camera device name pattern (e.g. "EMEET", "webcam", "USB")')
    parser.add_argument('--list-profiles', action='store_true',
                       help='List available profiles and exit')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available cameras and exit')
    parser.add_argument('--camera-info', action='store_true',
                       help='Show detailed info about selected camera and exit')
    parser.add_argument('--setup-camera', action='store_true',
                       help='Configure camera settings for gallery environment and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # List cameras if requested
    if args.list_cameras:
        try:
            print("Listing available cameras using v4l2-ctl...")
            import subprocess
            result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("v4l2-ctl not available. Try:")
                print("  sudo apt install v4l-utils")
                print("Then use: v4l2-ctl --list-devices")
        except Exception as e:
            print(f"Error listing cameras: {e}")
        return
    
    # Show camera info if requested
    if args.camera_info:
        get_camera_detailed_info(args.camera, args.camera_name)
        return
    
    # List profiles if requested
    if args.list_profiles:
        try:
            profiles = list_available_profiles()
            print("Available detector profiles:")
            for profile in profiles:
                print(f"  {profile}")
        except Exception as e:
            print(f"Error listing profiles: {e}")
        return
    
    # Validate profile exists
    try:
        profiles = list_available_profiles()
        if args.profile not in profiles:
            print(f"Profile '{args.profile}' not found.")
            print(f"Available profiles: {', '.join(profiles)}")
            return
    except Exception as e:
        print(f"Error loading profiles: {e}")
        return
    
    # Show camera setup info
    if args.camera_name:
        print(f"Using camera name pattern: '{args.camera_name}' (fallback device ID: {args.camera})")
    else:
        print(f"Using camera device ID: {args.camera}")
    
    # Run tuner
    tuner = InteractiveDetectorTuner(
        camera_id=args.camera, 
        camera_name=args.camera_name,
        profile_name=args.profile
    )
    success = tuner.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

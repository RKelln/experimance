#!/usr/bin/env python3
"""
Interactive detector parameter tuning tool.

Provides real-time visualization of detection results with adjustable parameters
via trackbars. Allows saving tuned parameters to detector profiles.

Usage:
    python interactive_detector_tuning.py [--profile PROFILE_NAME] [--camera CAMERA_ID]
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
from ..config import VisionConfig

logger = logging.getLogger(__name__)


class InteractiveDetectorTuner:
    """Interactive detector parameter tuning with live visualization."""
    
    def __init__(self, camera_id: int = 0, profile_name: str = "indoor_office"):
        self.camera_id = camera_id
        self.profile_name = profile_name
        self.cap = None
        self.detector = None
        self.profile = None
        
        # UI state
        self.window_name = "Detector Tuning"
        self.controls_window = "Controls"
        self.running = True
        self.show_hog = True
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
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            print(f"Error: Could not open camera {self.camera_id}")
            print("Make sure:")
            print("  - Camera is connected and not used by another application")
            print("  - Try a different camera ID (0, 1, 2, etc.)")
            print("  - Check camera permissions")
            return False
        
        # Set reasonable resolution for tuning
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Test reading a frame
        ret, test_frame = self.cap.read()
        if not ret:
            logger.error(f"Camera {self.camera_id} opened but cannot read frames")
            print(f"Error: Camera {self.camera_id} opened but cannot read frames")
            return False
        
        print(f"Successfully opened camera {self.camera_id} ({test_frame.shape[1]}x{test_frame.shape[0]})")
        return True
    
    def setup_detector(self) -> bool:
        """Initialize detector with profile."""
        try:
            # Load profile
            self.profile = load_profile(self.profile_name)
            logger.info(f"Loaded profile: {self.profile.name}")
            
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
        self.params['mog2_var_threshold'] = cv2.getTrackbarPos('MOG2 Var Threshold', self.controls_window)
        self.params['mog2_history'] = cv2.getTrackbarPos('MOG2 History', self.controls_window)
        self.params['person_base_conf'] = cv2.getTrackbarPos('Person Base Conf %', self.controls_window)
        self.params['motion_weight'] = cv2.getTrackbarPos('Motion Weight %', self.controls_window)
        
        # Mark that detector needs reload
        self.detector_needs_reload = True
    
    def on_toggle_change(self, val):
        """Handle display toggle changes."""
        self.show_hog = cv2.getTrackbarPos('Show HOG', self.controls_window) == 1
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
                f"Motion: {result.get('motion_detection', {}).get('motion_intensity', 0.0):.3f}",
                f"Person Count: {result.get('person_detection', {}).get('count', 0)}"
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
                if self.cap is None or self.detector is None:
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Run detection (handle async)
                try:
                    if self.loop is None:
                        self.loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(self.loop)
                    
                    # Add timing debug
                    import time
                    start_time = time.time()
                    result = self.loop.run_until_complete(
                        self.detector.detect_audience(frame, include_motion_mask=True)
                    )
                    detection_time = time.time() - start_time
                    
                    # Debug output every 30 frames (roughly once per second)
                    frame_count = getattr(self, '_frame_count', 0)
                    self._frame_count = frame_count + 1
                    if frame_count % 30 == 0:
                        print(f"Detection: {detection_time:.3f}s | "
                              f"HOG: {len(result.get('person_detection', {}).get('detections', []))} | "
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
        if self.cap:
            self.cap.release()
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


def main():
    parser = argparse.ArgumentParser(description="Interactive detector parameter tuning")
    parser.add_argument('--profile', '-p', default='indoor_office',
                       help='Detector profile to start with')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--list-profiles', action='store_true',
                       help='List available profiles and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
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
    
    # Run tuner
    tuner = InteractiveDetectorTuner(camera_id=args.camera, profile_name=args.profile)
    success = tuner.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

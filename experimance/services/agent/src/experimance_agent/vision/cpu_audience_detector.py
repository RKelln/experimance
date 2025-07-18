"""
CPU-optimized audience detection for the Experimance Agent Service.

Uses OpenCV's HOG (Histogram of Oriented Gradients) person detector
and other computer vision techniques for fast, real-time audience detection
without requiring GPU acceleration.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from ..config import VisionConfig
from .detector_profile import DetectorProfile, load_profile

logger = logging.getLogger(__name__)


class CPUAudienceDetector:
    """
    CPU-optimized audience detection using OpenCV computer vision techniques.
    
    This detector combines multiple fast CV techniques:
    - HOG person detection
    - Motion detection with background subtraction
    - Contour analysis
    - Temporal smoothing for stability
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        
        # Load detector profile
        try:
            self.profile = load_profile(config.detector_profile, config.detector_profile_dir)
            logger.info(f"Loaded detector profile: {self.profile.name} ({self.profile.description})")
        except Exception as e:
            logger.warning(f"Failed to load detector profile '{config.detector_profile}': {e}")
            logger.info("Using default profile parameters")
            from .detector_profile import create_default_profiles
            self.profile = create_default_profiles()["indoor_office"]
        
        # Detection state
        self.is_active = False
        self.audience_present = False
        self.last_detection_time: float = 0.0
        self.confidence_score: float = 0.0
        
        # Precompute scaling factors to avoid division in hot path
        self.scale_factor = self.profile.detection.detection_scale_factor
        self.scale_inverse = 1.0 / self.scale_factor if self.scale_factor > 0 else 1.0
        
        # HOG person detector
        self.hog = None
        if self.profile.detection.enable_hog_detection:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Face detector (YuNet)
        self.face_detector = None
        if self.profile.detection.enable_face_detection:
            try:
                face_params = self.profile.face
                self.face_detector = cv2.FaceDetectorYN.create(
                    face_params.model_path,
                    "",
                    face_params.input_size,
                    face_params.score_threshold,
                    face_params.nms_threshold,
                    face_params.top_k
                )
                logger.info(f"Initialized YuNet face detector with model: {face_params.model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize face detector: {e}")
                self.profile.detection.enable_face_detection = False
        
        # Background subtractor for motion detection
        self.background_subtractor = None
        
        # Apply performance mode if specified (this will override profile settings)
        if hasattr(config, 'cpu_performance_mode'):
            self.profile.update_from_performance_mode(config.cpu_performance_mode)
            logger.info(f"Applied performance mode: {config.cpu_performance_mode}")
        
        # Temporal smoothing
        self.detection_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_detections = 0
        self.detection_times: List[float] = []
        self.false_positive_count = 0
        
    async def start(self):
        """Initialize CPU audience detection components."""
        if not self.config.audience_detection_enabled:
            logger.info("CPU audience detection disabled in configuration")
            return
            
        try:
            # Initialize background subtractor using profile parameters
            if self.profile.detection.enable_motion_detection:
                mog2_params = self.profile.mog2
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=mog2_params.detect_shadows,
                    varThreshold=mog2_params.var_threshold,
                    history=mog2_params.history
                )
                logger.debug(f"MOG2 params - shadows: {mog2_params.detect_shadows}, "
                            f"varThreshold: {mog2_params.var_threshold}, history: {mog2_params.history}")
            
            enabled_detectors = []
            if self.profile.detection.enable_hog_detection:
                enabled_detectors.append("HOG")
            if self.profile.detection.enable_face_detection:
                enabled_detectors.append("Face")
            if self.profile.detection.enable_motion_detection:
                enabled_detectors.append("Motion")
            
            self.is_active = True
            logger.info(f"CPU audience detection initialized with profile: {self.profile.name}")
            logger.info(f"Enabled detectors: {', '.join(enabled_detectors) if enabled_detectors else 'None'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CPU audience detection: {e}")
            raise
    
    async def stop(self):
        """Stop audience detection and clean up resources."""
        self.is_active = False
        self.background_subtractor = None
        logger.info("CPU audience detection stopped")
    
    async def detect_audience(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform CPU-optimized audience detection on a frame.
        
        Args:
            frame: BGR video frame from webcam
            **kwargs: Additional arguments including:
                     - include_motion_mask: bool - Whether to include motion mask in results (default: False)
                     - webcam_manager, vlm_processor, etc. - Accepted for compatibility but not used
            
        Returns:
            dict: Detection results with confidence scores and metadata
        """
        if not self.is_active:
            return {"error": "CPU audience detection not active"}
        
        try:
            start_time = time.time()
            
            # Extract options from kwargs
            include_motion_mask = kwargs.get('include_motion_mask', False)
            
            # Scale down frame for faster processing
            small_frame = self._scale_frame(frame)
            
            # Perform person detection using HOG (if enabled)
            person_result = await self._detect_persons_hog(small_frame, self.scale_inverse)
            
            # Perform face detection (if enabled)
            face_result = await self._detect_faces(small_frame, self.scale_inverse)
            
            # Perform motion detection (if enabled)
            motion_result = await self._detect_motion(small_frame, include_mask=include_motion_mask)
            
            # Combine results with face detection
            detection_result = self._combine_cpu_results(person_result, face_result, motion_result)
            
            # Apply temporal smoothing
            smoothed_result = self._apply_temporal_smoothing(detection_result)
            
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            # Keep only recent performance data
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-50:]
            
            # Build result
            result = {
                "audience_detected": smoothed_result["detected"],
                "confidence": smoothed_result["confidence"],
                "detection_time": detection_time,
                "timestamp": time.time(),
                "person_detection": person_result,
                "face_detection": face_result,
                "motion_detection": motion_result,
                "method_used": "cpu_multimodal",
                "performance": {
                    "avg_detection_time": np.mean(self.detection_times),
                    "frame_scale": self.profile.detection.detection_scale_factor,
                    "profile_name": self.profile.name,
                    "profile_environment": self.profile.environment
                },
                "success": True
            }
            
            # Update internal state
            self.audience_present = result["audience_detected"]
            self.confidence_score = result["confidence"]
            self.last_detection_time = result["timestamp"]
            self.total_detections += 1
            
            return result
            
        except Exception as e:
            logger.error(f"CPU audience detection failed: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }
    
    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Scale frame down for faster processing using profile parameters."""
        if self.scale_factor == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    async def _detect_persons_hog(self, frame: np.ndarray, scale_inverse: float = 1.0) -> Dict[str, Any]:
        """
        Detect persons using HOG descriptor.
        
        Args:
            frame: BGR image frame (preferably scaled down)
            scale_inverse: Inverse of scale factor to scale detections back to original size
            
        Returns:
            dict: Person detection results
        """
        if not self.profile.detection.enable_hog_detection or self.hog is None:
            return {
                "persons_detected": False,
                "person_count": 0,
                "confidence": 0.0,
                "detections": [],
                "detection_weights": [],
                "method": "hog_disabled"
            }
        
        try:
            # Convert to grayscale for HOG detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect people using HOG with profile parameters
            hog_params = self.profile.hog
            detections, weights = self.hog.detectMultiScale(
                gray,
                winStride=(hog_params.win_stride_x, hog_params.win_stride_y),
                padding=(hog_params.padding_x, hog_params.padding_y),
                scale=hog_params.scale,
                hitThreshold=hog_params.hit_threshold,
                groupThreshold=hog_params.group_threshold
            )
            
            # Filter detections by size using profile parameters and scale back to original size
            min_height = self.profile.detection.min_person_height
            valid_detections = []
            for (x, y, w, h) in detections:
                if h >= min_height:  # Minimum height filter from profile
                    # Scale detection back to original frame size
                    scaled_x = int(x * scale_inverse)
                    scaled_y = int(y * scale_inverse)
                    scaled_w = int(w * scale_inverse)
                    scaled_h = int(h * scale_inverse)
                    valid_detections.append((scaled_x, scaled_y, scaled_w, scaled_h))
            
            person_count = len(valid_detections)
            
            # Calculate confidence based on detection count and weights using profile parameters
            conf_params = self.profile.confidence
            if person_count > 0:
                avg_weight = np.mean(weights[:len(valid_detections)]) if len(weights) > 0 else 0.5
                confidence = min(0.9, 
                               conf_params.person_base_confidence + 
                               (person_count * conf_params.person_count_weight) + 
                               (avg_weight * conf_params.person_weight_factor))
            else:
                confidence = 0.0
            
            return {
                "persons_detected": person_count > 0,
                "person_count": person_count,
                "confidence": confidence,
                "detections": valid_detections,
                "detection_weights": weights[:len(valid_detections)] if len(weights) > 0 else [],
                "method": "hog"
            }
            
        except Exception as e:
            logger.error(f"HOG person detection failed: {e}")
            return {
                "persons_detected": False,
                "person_count": 0,
                "confidence": 0.0,
                "detections": [],
                "detection_weights": [],
                "error": str(e),
                "method": "hog_error"
            }
    
    async def _detect_faces(self, frame: np.ndarray, scale_inverse: float = 1.0) -> Dict[str, Any]:
        """
        Detect faces using YuNet face detector.
        
        Args:
            frame: BGR image frame
            scale_inverse: Inverse of scale factor to scale detections back to original size
            
        Returns:
            dict: Face detection results
        """
        if not self.profile.detection.enable_face_detection or self.face_detector is None:
            return {
                "faces_detected": False,
                "face_count": 0,
                "confidence": 0.0,
                "detections": [],
                "method": "face_disabled"
            }
        
        try:
            # Set input size for current frame
            height, width = frame.shape[:2]
            self.face_detector.setInputSize((width, height))
            
            # Detect faces
            _, faces = self.face_detector.detect(frame)
            
            if faces is not None:
                # Filter faces by minimum size
                min_face_size = self.profile.face.min_face_size
                valid_faces = []
                confidences = []
                
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    confidence = face[14]  # Face score
                    
                    # Filter by minimum face size (before scaling)
                    if w >= min_face_size and h >= min_face_size:
                        # Scale detection back to original frame size
                        scaled_x = int(x * scale_inverse)
                        scaled_y = int(y * scale_inverse)
                        scaled_w = int(w * scale_inverse)
                        scaled_h = int(h * scale_inverse)
                        valid_faces.append((scaled_x, scaled_y, scaled_w, scaled_h))
                        confidences.append(confidence)
                
                face_count = len(valid_faces)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                return {
                    "faces_detected": face_count > 0,
                    "face_count": face_count,
                    "confidence": min(0.9, avg_confidence * 0.8),  # Scale confidence
                    "detections": valid_faces,
                    "method": "yunet"
                }
            else:
                return {
                    "faces_detected": False,
                    "face_count": 0,
                    "confidence": 0.0,
                    "detections": [],
                    "method": "yunet"
                }
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {
                "faces_detected": False,
                "face_count": 0,
                "confidence": 0.0,
                "detections": [],
                "error": str(e),
                "method": "face_error"
            }
    
    async def _detect_motion(self, frame: np.ndarray, include_mask: bool = False) -> Dict[str, Any]:
        """
        Detect motion using background subtraction.
        
        Args:
            frame: BGR image frame
            include_mask: Whether to include the motion mask in results (for visualization)
            
        Returns:
            dict: Motion detection results
        """
        if not self.profile.detection.enable_motion_detection or self.background_subtractor is None:
            result = {
                "motion_detected": False,
                "motion_intensity": 0.0,
                "contour_count": 0,
                "total_motion_area": 0,
                "confidence": 0.0,
                "method": "motion_disabled"
            }
            if include_mask:
                result["motion_mask"] = None
            return result
        
        try:
            if not self.background_subtractor:
                return {"motion_detected": False, "error": "No background subtractor"}
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size using profile parameters
            motion_threshold = self.profile.detection.motion_threshold
            significant_contours = [c for c in contours if cv2.contourArea(c) > motion_threshold]
            
            # Calculate motion metrics
            total_motion_area = sum(cv2.contourArea(c) for c in significant_contours)
            frame_area = frame.shape[0] * frame.shape[1]
            motion_intensity = total_motion_area / frame_area
            
            # Determine if significant motion detected using profile parameters
            motion_intensity_threshold = self.profile.detection.motion_intensity_threshold
            motion_detected = len(significant_contours) > 0 and motion_intensity > motion_intensity_threshold
            
            result = {
                "motion_detected": motion_detected,
                "motion_intensity": motion_intensity,
                "contour_count": len(significant_contours),
                "total_motion_area": total_motion_area,
                "confidence": min(0.8, motion_intensity * 10) if motion_detected else 0.0
            }
            
            # Only include motion mask if requested (for visualization during tuning)
            if include_mask:
                result["motion_mask"] = fg_mask
            
            return result
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return {"motion_detected": False, "error": str(e)}
    
    def _combine_cpu_results(self, person_result: Dict[str, Any], 
                           face_result: Dict[str, Any],
                           motion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine HOG person detection, face detection, and motion detection results using profile parameters.
        
        Args:
            person_result: Results from HOG person detection
            face_result: Results from face detection
            motion_result: Results from motion detection
            
        Returns:
            dict: Combined detection result
        """
        persons_detected = person_result.get("persons_detected", False)
        faces_detected = face_result.get("faces_detected", False)
        motion_detected = motion_result.get("motion_detected", False)
        
        person_confidence = person_result.get("confidence", 0.0)
        face_confidence = face_result.get("confidence", 0.0)
        motion_confidence = motion_result.get("confidence", 0.0)
        
        conf_params = self.profile.confidence
        
        # Prioritize face detection as it's most reliable for seated people
        if faces_detected:
            if motion_detected:
                # Face + motion = very high confidence
                return {
                    "detected": True,
                    "confidence": min(conf_params.max_combined_confidence, 
                                    face_confidence + motion_confidence * conf_params.motion_confidence_weight),
                    "primary_method": "face+motion"
                }
            else:
                # Face only = high confidence (good for seated people)
                return {
                    "detected": True,
                    "confidence": face_confidence,
                    "primary_method": "face_only"
                }
        elif persons_detected:
            if motion_detected:
                # Person + motion = high confidence
                return {
                    "detected": True,
                    "confidence": min(conf_params.max_combined_confidence, 
                                    person_confidence + motion_confidence * conf_params.motion_confidence_weight),
                    "primary_method": "person+motion"
                }
            else:
                # Person only = medium confidence
                return {
                    "detected": True,
                    "confidence": person_confidence,
                    "primary_method": "person_only"
                }
        elif motion_detected:
            # Motion only = lower confidence
            return {
                "detected": motion_confidence > conf_params.motion_only_threshold,
                "confidence": motion_confidence * conf_params.motion_only_confidence_factor,
                "primary_method": "motion_only"
            }
        else:
            # No detection
            return {
                "detected": False,
                "confidence": conf_params.absence_confidence,
                "primary_method": "none"
            }
    
    def _apply_temporal_smoothing(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply temporal smoothing to reduce false positives and improve stability using profile parameters.
        
        Args:
            detection_result: Current detection result
            
        Returns:
            dict: Smoothed detection result
        """
        # Add current result to history
        self.detection_history.append(detection_result)
        history_size = self.profile.detection.detection_history_size
        if len(self.detection_history) > history_size:
            self.detection_history.pop(0)
        
        if len(self.detection_history) < 2:
            return detection_result  # Not enough history
        
        # Calculate consensus from recent detections
        recent_detections = [r["detected"] for r in self.detection_history]
        positive_count = sum(recent_detections)
        total_count = len(recent_detections)
        
        # Calculate average confidence
        confidences = [r["confidence"] for r in self.detection_history]
        avg_confidence = np.mean(confidences)
        
        # Determine stable result using majority vote with confidence weighting using profile parameters
        stability_threshold = self.profile.detection.stability_threshold
        stable_detected = (positive_count / total_count) >= stability_threshold
        
        # Adjust confidence based on stability
        stability_score = 1.0 - abs((positive_count / total_count) - 0.5) * 2
        final_confidence = avg_confidence * stability_score
        
        return {
            "detected": stable_detected,
            "confidence": final_confidence,
            "stability": stability_score
        }
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        return {
            "total_detections": self.total_detections,
            "current_audience_present": self.audience_present,
            "current_confidence": self.confidence_score,
            "avg_detection_time": np.mean(self.detection_times) if self.detection_times else 0.0,
            "max_detection_time": np.max(self.detection_times) if self.detection_times else 0.0,
            "detection_history_size": len(self.detection_history),
            "performance_optimized": True,
            "detection_scale_factor": self.profile.detection.detection_scale_factor,
            "profile_name": self.profile.name,
            "profile_description": self.profile.description,
            "profile_environment": self.profile.environment,
            "profile_lighting": self.profile.lighting,
            "last_detection_age": time.time() - self.last_detection_time if self.last_detection_time > 0 else float('inf')
        }
    
    def reset_detection_history(self):
        """Reset detection history for clean state."""
        self.detection_history.clear()
        self.audience_present = False
        self.confidence_score = 0.0
        logger.info("CPU detection history reset")
    
    def set_performance_mode(self, mode: str):
        """
        Set performance optimization mode by updating profile parameters.
        
        Args:
            mode: 'fast', 'balanced', or 'accurate'
        """
        self.profile.update_from_performance_mode(mode)
        
        # Recalculate scaling factors
        self.scale_factor = self.profile.detection.detection_scale_factor
        self.scale_inverse = 1.0 / self.scale_factor if self.scale_factor > 0 else 1.0
        
        logger.info(f"CPU detection performance mode set to: {mode}")
        logger.debug(f"Updated parameters - scale: {self.scale_factor}, "
                    f"min_height: {self.profile.detection.min_person_height}, "
                    f"motion_threshold: {self.profile.detection.motion_threshold}")
    
    def update_profile(self, new_profile: DetectorProfile):
        """
        Update the detector profile during runtime.
        
        Args:
            new_profile: New detector profile to use
        """
        self.profile = new_profile
        
        # Recalculate scaling factors
        self.scale_factor = self.profile.detection.detection_scale_factor
        self.scale_inverse = 1.0 / self.scale_factor if self.scale_factor > 0 else 1.0
        
        logger.info(f"Updated detector profile to: {new_profile.name}")
        
        # Reinitialize background subtractor with new parameters if active
        if self.is_active and self.background_subtractor is not None:
            mog2_params = self.profile.mog2
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=mog2_params.detect_shadows,
                varThreshold=mog2_params.var_threshold,
                history=mog2_params.history
            )
            logger.info("Reinitialized background subtractor with new profile parameters")

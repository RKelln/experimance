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
        
        # Detection state
        self.is_active = False
        self.audience_present = False
        self.last_detection_time: float = 0.0
        self.confidence_score: float = 0.0
        
        # HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Background subtractor for motion detection
        self.background_subtractor = None
        
        # Detection parameters (optimized for performance)
        self.detection_scale_factor = 0.5  # Scale down for faster processing
        self.min_person_height = 60  # Minimum height for person detection (in scaled image)
        self.motion_threshold = 1000  # Minimum contour area for significant motion
        
        # Temporal smoothing
        self.detection_history: List[Dict[str, Any]] = []
        self.detection_history_size = 5
        
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
            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=50,  # More sensitive for better detection
                history=300
            )
            
            self.is_active = True
            logger.info("CPU audience detection initialized")
            
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
            **kwargs: Additional arguments (webcam_manager, vlm_processor, etc.)
                     Accepted for compatibility but not used in CPU detection
            
        Returns:
            dict: Detection results with confidence scores and metadata
        """
        if not self.is_active:
            return {"error": "CPU audience detection not active"}
        
        try:
            start_time = time.time()
            
            # Scale down frame for faster processing
            small_frame = self._scale_frame(frame)
            
            # Perform person detection using HOG
            person_result = await self._detect_persons_hog(small_frame)
            
            # Perform motion detection
            motion_result = await self._detect_motion(small_frame)
            
            # Combine results
            detection_result = self._combine_cpu_results(person_result, motion_result)
            
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
                "motion_detection": motion_result,
                "method_used": "cpu_hog_motion",
                "performance": {
                    "avg_detection_time": np.mean(self.detection_times),
                    "frame_scale": self.detection_scale_factor
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
        """Scale frame down for faster processing."""
        if self.detection_scale_factor == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        new_width = int(width * self.detection_scale_factor)
        new_height = int(height * self.detection_scale_factor)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    async def _detect_persons_hog(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect persons using HOG descriptor.
        
        Args:
            frame: BGR image frame (preferably scaled down)
            
        Returns:
            dict: Person detection results
        """
        try:
            # Convert to grayscale for HOG detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect people using HOG
            # detectMultiScale parameters optimized for speed vs accuracy
            detections, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),  # Larger stride for speed
                padding=(4, 4),   # Smaller padding for speed
                scale=1.05,       # Smaller scale increment for speed
                hitThreshold=0.0, # Lower threshold for better detection
                groupThreshold=2  # Group overlapping detections
            )
            
            # Filter detections by size
            valid_detections = []
            for (x, y, w, h) in detections:
                if h >= self.min_person_height:  # Minimum height filter
                    valid_detections.append((x, y, w, h))
            
            person_count = len(valid_detections)
            
            # Calculate confidence based on detection count and weights
            if person_count > 0:
                avg_weight = np.mean(weights[:len(valid_detections)]) if len(weights) > 0 else 0.5
                confidence = min(0.9, 0.3 + (person_count * 0.2) + (avg_weight * 0.4))
            else:
                confidence = 0.0
            
            return {
                "persons_detected": person_count > 0,
                "person_count": person_count,
                "confidence": confidence,
                "detections": valid_detections,
                "detection_weights": weights[:len(valid_detections)] if len(weights) > 0 else []
            }
            
        except Exception as e:
            logger.error(f"HOG person detection failed: {e}")
            return {
                "persons_detected": False,
                "person_count": 0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _detect_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect motion using background subtraction.
        
        Args:
            frame: BGR image frame
            
        Returns:
            dict: Motion detection results
        """
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
            
            # Filter contours by size
            significant_contours = [c for c in contours if cv2.contourArea(c) > self.motion_threshold]
            
            # Calculate motion metrics
            total_motion_area = sum(cv2.contourArea(c) for c in significant_contours)
            frame_area = frame.shape[0] * frame.shape[1]
            motion_intensity = total_motion_area / frame_area
            
            # Determine if significant motion detected
            motion_detected = len(significant_contours) > 0 and motion_intensity > 0.01
            
            return {
                "motion_detected": motion_detected,
                "motion_intensity": motion_intensity,
                "contour_count": len(significant_contours),
                "total_motion_area": total_motion_area,
                "confidence": min(0.8, motion_intensity * 10) if motion_detected else 0.0
            }
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return {"motion_detected": False, "error": str(e)}
    
    def _combine_cpu_results(self, person_result: Dict[str, Any], 
                           motion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine HOG person detection and motion detection results.
        
        Args:
            person_result: Results from HOG person detection
            motion_result: Results from motion detection
            
        Returns:
            dict: Combined detection result
        """
        persons_detected = person_result.get("persons_detected", False)
        motion_detected = motion_result.get("motion_detected", False)
        
        person_confidence = person_result.get("confidence", 0.0)
        motion_confidence = motion_result.get("confidence", 0.0)
        
        if persons_detected and motion_detected:
            # Both methods agree - high confidence
            return {
                "detected": True,
                "confidence": min(0.95, person_confidence + motion_confidence * 0.3),
                "primary_method": "person+motion"
            }
        elif persons_detected:
            # Person detected without motion (static person)
            return {
                "detected": True,
                "confidence": person_confidence,
                "primary_method": "person_only"
            }
        elif motion_detected:
            # Motion without person detection (possible false positive)
            # Lower confidence, but still possible
            return {
                "detected": motion_confidence > 0.4,  # Threshold for motion-only detection
                "confidence": motion_confidence * 0.6,
                "primary_method": "motion_only"
            }
        else:
            # Neither detected
            return {
                "detected": False,
                "confidence": 0.9,  # High confidence in absence
                "primary_method": "none"
            }
    
    def _apply_temporal_smoothing(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply temporal smoothing to reduce false positives and improve stability.
        
        Args:
            detection_result: Current detection result
            
        Returns:
            dict: Smoothed detection result
        """
        # Add current result to history
        self.detection_history.append(detection_result)
        if len(self.detection_history) > self.detection_history_size:
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
        
        # Determine stable result using majority vote with confidence weighting
        stability_threshold = 0.6  # Require 60% agreement for positive detection
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
            "detection_scale_factor": self.detection_scale_factor,
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
        Set performance optimization mode.
        
        Args:
            mode: 'fast', 'balanced', or 'accurate'
        """
        if mode == "fast":
            self.detection_scale_factor = 0.3
            self.min_person_height = 40
            self.motion_threshold = 2000
        elif mode == "balanced":
            self.detection_scale_factor = 0.5
            self.min_person_height = 60
            self.motion_threshold = 1000
        elif mode == "accurate":
            self.detection_scale_factor = 0.7
            self.min_person_height = 80
            self.motion_threshold = 500
        else:
            raise ValueError(f"Unknown performance mode: {mode}")
        
        logger.info(f"CPU detection performance mode set to: {mode}")

"""
Audience detection for the Experimance Agent Service.

Combines computer vision techniques and VLM analysis to detect audience presence
and engagement in the installation space.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

from ..config import VisionConfig

logger = logging.getLogger(__name__)


class AudienceDetector:
    """
    Audience detection system combining traditional computer vision and VLM analysis.
    
    Uses motion detection, background subtraction, and VLM queries to determine
    audience presence with high confidence and low false positive rates.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        
        # Detection state
        self.is_active = False
        self.audience_present = False
        self.last_detection_time: float = 0.0
        self.confidence_score: float = 0.0
        
        # Motion detection components
        self.background_subtractor = None
        self.motion_threshold = 1000  # Minimum contour area for motion detection
        self.motion_history: List[float] = []
        self.motion_history_size = 10
        
        # Detection history for stability
        self.detection_history: List[bool] = []
        self.detection_history_size = 5
        
        # Performance tracking
        self.total_detections = 0
        self.false_positive_count = 0
        self.vlm_queries = 0
    
    async def start(self):
        """Initialize audience detection components."""
        if not self.config.audience_detection_enabled:
            logger.info("Audience detection disabled in configuration")
            return
            
        try:
            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=16,
                history=500
            )
            
            self.is_active = True
            logger.info("Audience detection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audience detection: {e}")
            raise
    
    async def stop(self):
        """Stop audience detection and clean up resources."""
        self.is_active = False
        self.background_subtractor = None
        logger.info("Audience detection stopped")
    
    async def detect_audience(self, frame: np.ndarray, 
                            **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive audience detection on a frame.
        
        Args:
            frame: BGR video frame from webcam
            **kwargs: Additional arguments (webcam_manager, vlm_processor, etc.)
            
        Returns:
            dict: Detection results with confidence scores and metadata
        """
        if not self.is_active:
            return {"error": "Audience detection not active"}
        
        # Extract kwargs
        webcam_manager = kwargs.get('webcam_manager')
        vlm_processor = kwargs.get('vlm_processor')
        
        try:
            start_time = time.time()
            
            # Perform motion-based detection
            motion_result = await self._detect_motion(frame)
            
            # If motion detected or we need VLM confirmation, use VLM
            vlm_result = None
            if (motion_result["motion_detected"] or 
                self._should_use_vlm_verification() or
                vlm_processor is None):
                
                if vlm_processor and vlm_processor.is_loaded:
                    # Convert frame for VLM processing
                    if webcam_manager:
                        rgb_frame = webcam_manager.preprocess_for_vlm(frame)
                    else:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    vlm_result = await vlm_processor.detect_audience(rgb_frame)
                    self.vlm_queries += 1
            
            # Combine results to make final decision
            detection_result = self._combine_detection_results(motion_result, vlm_result)
            
            # Update detection history for stability
            self._update_detection_history(detection_result["audience_detected"])
            
            # Calculate final confidence and stable result
            stable_result = self._get_stable_detection_result()
            
            detection_time = time.time() - start_time
            
            # Build comprehensive result
            result = {
                "audience_detected": stable_result["detected"],
                "confidence": stable_result["confidence"],
                "detection_time": detection_time,
                "timestamp": time.time(),
                "motion_analysis": motion_result,
                "vlm_analysis": vlm_result,
                "stability_score": stable_result["stability"],
                "method_used": self._get_detection_method_used(motion_result, vlm_result),
                "success": True
            }
            
            # Update internal state
            self.audience_present = result["audience_detected"]
            self.confidence_score = result["confidence"]
            self.last_detection_time = result["timestamp"]
            self.total_detections += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Audience detection failed: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }
    
    async def _detect_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect motion in the frame using background subtraction.
        
        Args:
            frame: BGR video frame
            
        Returns:
            dict: Motion detection results
        """
        try:
            if not self.background_subtractor:
                return {"motion_detected": False, "error": "No background subtractor"}
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Remove noise with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            significant_contours = [c for c in contours if cv2.contourArea(c) > self.motion_threshold]
            
            # Calculate motion metrics
            total_motion_area = sum(cv2.contourArea(c) for c in significant_contours)
            motion_intensity = total_motion_area / (frame.shape[0] * frame.shape[1])
            
            # Update motion history
            self.motion_history.append(motion_intensity)
            if len(self.motion_history) > self.motion_history_size:
                self.motion_history.pop(0)
            
            # Determine if significant motion detected
            motion_detected = len(significant_contours) > 0 and motion_intensity > 0.01
            
            return {
                "motion_detected": motion_detected,
                "motion_intensity": motion_intensity,
                "contour_count": len(significant_contours),
                "total_motion_area": total_motion_area,
                "motion_history_avg": np.mean(self.motion_history) if self.motion_history else 0.0
            }
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return {"motion_detected": False, "error": str(e)}
    
    def _should_use_vlm_verification(self) -> bool:
        """
        Determine if VLM verification should be used based on recent detection history.
        
        Returns:
            bool: True if VLM verification is recommended
        """
        # Use VLM verification if:
        # 1. Detection results have been unstable
        # 2. It's been a while since last VLM query
        # 3. Motion detection confidence is low
        
        if len(self.detection_history) < 3:
            return True  # Not enough history, use VLM
        
        # Check for instability in recent detections
        recent_detections = self.detection_history[-3:]
        if len(set(recent_detections)) > 1:  # Mixed results
            return True
        
        # Use VLM verification periodically
        time_since_last_vlm = time.time() - self.last_detection_time
        if time_since_last_vlm > 30.0:  # 30 seconds since last VLM query
            return True
        
        return False
    
    def _combine_detection_results(self, motion_result: Dict[str, Any], 
                                 vlm_result: Optional[bool]) -> Dict[str, Any]:
        """
        Combine motion detection and VLM results into a unified decision.
        
        Args:
            motion_result: Results from motion detection
            vlm_result: Boolean result from VLM detection, or None if not used
            
        Returns:
            dict: Combined detection result
        """
        motion_detected = motion_result.get("motion_detected", False)
        motion_confidence = motion_result.get("motion_intensity", 0.0)
        
        if vlm_result is not None:
            # VLM result available - combine with motion
            if vlm_result and motion_detected:
                # Both agree on presence
                return {
                    "audience_detected": True,
                    "confidence": min(0.9, 0.6 + motion_confidence * 0.3),
                    "primary_method": "combined"
                }
            elif vlm_result and not motion_detected:
                # VLM detects but no motion (static audience)
                return {
                    "audience_detected": True,
                    "confidence": 0.7,
                    "primary_method": "vlm"
                }
            elif not vlm_result and motion_detected:
                # Motion but VLM doesn't detect people (false positive likely)
                return {
                    "audience_detected": False,
                    "confidence": 0.8,
                    "primary_method": "vlm_override"
                }
            else:
                # Both agree on absence
                return {
                    "audience_detected": False,
                    "confidence": 0.9,
                    "primary_method": "combined"
                }
        else:
            # Motion detection only
            return {
                "audience_detected": motion_detected,
                "confidence": min(0.6, motion_confidence * 2.0) if motion_detected else 0.8,
                "primary_method": "motion_only"
            }
    
    def _update_detection_history(self, detected: bool):
        """
        Update the detection history for stability analysis.
        
        Args:
            detected: Current detection result
        """
        self.detection_history.append(detected)
        if len(self.detection_history) > self.detection_history_size:
            self.detection_history.pop(0)
    
    def _get_stable_detection_result(self) -> Dict[str, Any]:
        """
        Get a stable detection result based on recent history.
        
        Returns:
            dict: Stable detection result with confidence and stability metrics
        """
        if not self.detection_history:
            return {"detected": False, "confidence": 0.0, "stability": 0.0}
        
        # Calculate consensus from recent detections
        recent_count = len(self.detection_history)
        positive_count = sum(self.detection_history)
        
        # Majority vote with stability weighting
        detected = positive_count > (recent_count / 2)
        
        # Calculate stability score (how consistent recent detections are)
        if recent_count > 1:
            stability = 1.0 - (len(set(self.detection_history)) - 1) / (recent_count - 1)
        else:
            stability = 1.0
        
        # Calculate confidence based on consensus strength and stability
        consensus_strength = abs(positive_count - (recent_count / 2)) / (recent_count / 2)
        confidence = min(0.95, consensus_strength * stability)
        
        return {
            "detected": detected,
            "confidence": confidence,
            "stability": stability
        }
    
    def _get_detection_method_used(self, motion_result: Dict[str, Any], 
                                 vlm_result: Optional[bool]) -> str:
        """
        Determine which detection method was primarily used.
        
        Args:
            motion_result: Motion detection results
            vlm_result: VLM detection result or None
            
        Returns:
            str: Description of primary detection method
        """
        if vlm_result is not None:
            if motion_result.get("motion_detected", False):
                return "motion+vlm"
            else:
                return "vlm_only"
        else:
            return "motion_only"
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get performance and usage statistics.
        
        Returns:
            dict: Detection statistics and performance metrics
        """
        return {
            "total_detections": self.total_detections,
            "vlm_queries": self.vlm_queries,
            "vlm_usage_ratio": self.vlm_queries / max(1, self.total_detections),
            "current_audience_present": self.audience_present,
            "current_confidence": self.confidence_score,
            "detection_history_size": len(self.detection_history),
            "motion_history_size": len(self.motion_history),
            "last_detection_age": time.time() - self.last_detection_time if self.last_detection_time > 0 else float('inf')
        }
    
    def reset_detection_history(self):
        """Reset detection history for clean state."""
        self.detection_history.clear()
        self.motion_history.clear()
        self.audience_present = False
        self.confidence_score = 0.0
        logger.info("Detection history reset")

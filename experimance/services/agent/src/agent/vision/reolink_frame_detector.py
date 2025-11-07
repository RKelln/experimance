"""
Reolink Camera Frame-based Person Detection

Combines Reolink camera snapshot capture with CPU-based HOG person detection
for more reliable audience detection than the camera's built-in AI.

Uses OpenCV HOG person detector for robust person detection at distance.
"""

import asyncio
import time
import logging
import cv2
import numpy as np
import aiohttp
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urlencode

# Import YOLO11n detector
try:
    from .yolo_person_detector import YOLO11PersonDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReolinkFrameDetector:
    """
    Person detection using Reolink camera snapshots and HOG person detection.
    
    This detector:
    1. Captures frames from Reolink camera via snapshot URL
    2. Uses OpenCV HOG person detection for reliable person detection
    3. Provides hysteresis and temporal smoothing for stable results
    """
    
    def __init__(self, host: str, user: str, password: str, https: bool = True, 
                 channel: int = 0, timeout: int = 2, vision_config=None):
        """
        Initialize Reolink frame-based person detector.
        
        Args:
            host: Camera IP address (e.g., '192.168.2.229')
            user: Camera username
            password: Camera password
            https: Use HTTPS (True) or HTTP (False)
            channel: Camera channel (usually 0)
            timeout: Request timeout in seconds
            vision_config: Vision configuration (unused, for compatibility)
        """
        self.host = host
        self.user = user
        self.password = password
        self.https = https
        self.channel = channel
        self.timeout = timeout
        
        # Build snapshot URL
        scheme = "https" if https else "http"
        self.snapshot_url = f"{scheme}://{host}/cgi-bin/api.cgi"
        
        # Initialize HOG person detector (fallback)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize YOLO11n person detector (primary method)
        self.yolo_detector = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_detector = YOLO11PersonDetector(
                    confidence_threshold=0.6,
                    edge_filter_percent=0.12,
                    min_person_size=(60, 120)
                )
                if self.yolo_detector.is_available():
                    logger.info("✅ YOLO11n person detector initialized")
                    self.detection_method = 'yolo11n'
                else:
                    self.yolo_detector = None
                    logger.warning("YOLO11n not available, using HOG fallback")
                    self.detection_method = 'hog'
            except Exception as e:
                logger.error(f"Failed to initialize YOLO11n: {e}")
                self.yolo_detector = None
                self.detection_method = 'hog'
        else:
            logger.info("YOLO11n not available, using HOG detection")
            self.detection_method = 'hog'
        
        # HTTP session for efficient connection reuse
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Detection state
        self.is_active = False
        self.last_detection_time = 0.0
        self.consecutive_present = 0
        self.consecutive_absent = 0
        self.current_state = False  # False = absent, True = present
        
        # Hysteresis settings (can be overridden)
        self.hysteresis_present = 3   # Consecutive detections needed to confirm present
        self.hysteresis_absent = 8    # Consecutive non-detections needed to confirm absent
        
        # Performance tracking
        self.frame_capture_times = []
        self.detection_times = []
        self.total_frames = 0
        self.failed_captures = 0
        
    def set_hysteresis(self, present: int, absent: int):
        """Set hysteresis thresholds for stable detection."""
        self.hysteresis_present = present
        self.hysteresis_absent = absent
        logger.info(f"Hysteresis updated: {present} present, {absent} absent")
    
    async def start(self):
        """Start the frame detector."""
        if self.is_active:
            return
        
        # Create HTTP session with SSL verification disabled (common for IP cameras)
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        self.is_active = True
        logger.info(f"Reolink frame detector started: {self.snapshot_url}")
    
    async def stop(self):
        """Stop the frame detector and clean up."""
        self.is_active = False
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Reolink frame detector stopped")
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the Reolink camera.
        
        Returns:
            BGR frame as numpy array, or None if capture failed
        """
        if not self.session:
            logger.error("Session not initialized - call start() first")
            return None
        
        capture_start = time.time()
        
        try:
            # Build snapshot request parameters
            params = {
                'cmd': 'Snap',
                'channel': self.channel,
                'rs': f'frame_{int(time.time())}',  # Cache busting
                'user': self.user,
                'password': self.password
            }
            
            # Make request
            async with self.session.get(self.snapshot_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Snapshot request failed: {response.status}")
                    self.failed_captures += 1
                    return None
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if 'image' not in content_type:
                    logger.error(f"Invalid content type: {content_type}")
                    self.failed_captures += 1
                    return None
                
                # Read image data
                image_data = await response.read()
                
                # Decode image using OpenCV
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error("Failed to decode image data")
                    self.failed_captures += 1
                    return None
                
                # Track performance
                capture_time = time.time() - capture_start
                self.frame_capture_times.append(capture_time)
                if len(self.frame_capture_times) > 100:
                    self.frame_capture_times.pop(0)
                
                self.total_frames += 1
                
                return frame
                
        except asyncio.TimeoutError:
            logger.warning(f"Frame capture timeout ({self.timeout}s)")
            self.failed_captures += 1
            return None
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            self.failed_captures += 1
            return None
    
    def detect_people_hog(self, frame: np.ndarray) -> Tuple[int, float, List[Tuple[float, Tuple[int, int, int, int]]]]:
        """
        Detect people using HOG descriptor.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (number_of_people, max_confidence, detections_list)
            detections_list contains (confidence, (x, y, w, h)) tuples
        """
        if frame is None:
            return 0, 0.0, []
            
        try:
            # Scale down for faster processing
            height, width = frame.shape[:2]
            scale_factor = 0.6  # Increased from 0.5 for better accuracy
            small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            
            # Detect people using HOG with parameters tuned for person detection
            people, weights = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),      # Larger stride for efficiency
                padding=(16, 16),      # More padding for edge detection
                scale=1.05,            # Standard scale factor
                hitThreshold=0.0       # Very low threshold, let confidence filtering handle it
            )
            
            # Scale bounding boxes back to original frame size and filter detections
            detections = []
            frame_height, frame_width = frame.shape[:2]
            
            # More conservative margins to not filter out real people
            edge_margin_x = int(frame_width * 0.05)   # 5% margin on sides (was 10%)
            edge_margin_y = int(frame_height * 0.05)  # 5% margin on top/bottom (was 10%)
            min_confidence = 0.4  # Lower confidence threshold (was 0.7)
            
            filtered_detections = []
            all_detections = []  # Keep track for debugging
            
            for i, (x, y, w, h) in enumerate(people):
                # Scale coordinates back up
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor) 
                orig_w = int(w / scale_factor)
                orig_h = int(h / scale_factor)
                
                confidence = weights[i] if i < len(weights) else 0.0
                
                # Store all detections for debugging
                all_detections.append((confidence, (orig_x, orig_y, orig_w, orig_h)))
                
                # Filter criteria
                center_x = orig_x + orig_w // 2
                center_y = orig_y + orig_h // 2
                
                is_edge = (center_x < edge_margin_x or center_x > (frame_width - edge_margin_x) or
                          center_y < edge_margin_y or center_y > (frame_height - edge_margin_y))
                is_low_conf = confidence < min_confidence
                
                if is_low_conf or is_edge:
                    logger.debug(f"Filtered detection: conf={confidence:.3f}, center=({center_x},{center_y}), "
                               f"edge={is_edge}, low_conf={is_low_conf}, "
                               f"margins=({edge_margin_x},{edge_margin_y}), frame=({frame_width},{frame_height})")
                    continue
                    
                filtered_detections.append((confidence, (orig_x, orig_y, orig_w, orig_h)))
            
            # Sort by confidence (highest first)
            filtered_detections.sort(key=lambda x: x[0], reverse=True)
            
            # Use filtered results, but fall back to all if we filtered everything
            if len(filtered_detections) == 0 and len(all_detections) > 0:
                logger.warning(f"All detections filtered out! Using top confidence from {len(all_detections)} raw detections")
                # Use the highest confidence detection even if it was filtered
                all_detections.sort(key=lambda x: x[0], reverse=True)
                detections = [all_detections[0]]  # Just use the best one
            else:
                detections = filtered_detections
            
            # Use filtered results
            num_people = len(detections)
            max_confidence = max([d[0] for d in detections]) if detections else 0.0
            
            # Enhanced debug logging
            raw_detections = len(people)
            if raw_detections != num_people or num_people > 0:
                logger.info(f"HOG detection: {raw_detections} raw → {num_people} filtered, "
                          f"max_confidence: {max_confidence:.3f}, frame: {frame_width}x{frame_height}")
                if num_people == 0 and raw_detections > 0:
                    logger.warning("No people detected after filtering! Check filter settings.")
            
            return num_people, max_confidence, detections
            
        except Exception as e:
            logger.error(f"HOG detection error: {e}")
            return 0, 0.0, []

    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on the frame.
        
        Args:
            frame: Original BGR frame
            detections: List of (confidence, (x, y, w, h)) tuples
            
        Returns:
            Annotated frame with bounding boxes
        """
        annotated_frame = frame.copy()
        
        for i, (confidence, (x, y, w, h)) in enumerate(detections):
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Person {i+1}: {confidence:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(annotated_frame, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add frame info
        info_text = f"Total: {len(detections)} people detected"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame
    
    async def detect_person(self) -> Dict[str, Any]:
        """
        Detect person presence using frame capture and HOG detection.
        
        Returns:
            Dict with detection results:
            {
                'person_detected': bool,
                'confidence': float,
                'stable_state': bool,  # True if state has passed hysteresis
                'stable_state_changed': bool,  # True if stable state just changed
                'consecutive_readings': int,
                'detection_details': dict,
                'frame_info': dict
            }
        """
        if not self.is_active:
            return {
                'person_detected': False,
                'confidence': 0.0,
                'stable_state': False,
                'stable_state_changed': False,
                'consecutive_readings': 0,
                'detection_details': {},
                'frame_info': {'error': 'Detector not active'}
            }
        
        detection_start = time.time()
        
        # Capture frame from camera
        frame = await self.capture_frame()
        if frame is None:
            return {
                'person_detected': False,
                'confidence': 0.0,
                'stable_state': self.current_state,
                'stable_state_changed': False,
                'consecutive_readings': max(self.consecutive_present, self.consecutive_absent),
                'detection_details': {},
                'frame_info': {'error': 'Frame capture failed'}
            }
        
        # Run person detection using the best available method
        try:
            if self.yolo_detector and self.yolo_detector.is_available():
                # Use YOLO11n detection (primary method)
                num_people, confidence, detections = self.yolo_detector.detect_people(frame)
                detection_details = {
                    'method': 'yolo11n',
                    'num_people': num_people,
                    'confidence': confidence,
                    'detections': detections,
                    'threshold': 0.6
                }
                logger.debug(f"YOLO11n detection: {num_people} people, conf={confidence:.3f}")
            else:
                # Fallback to HOG detection
                num_people, confidence, detections = self.detect_people_hog(frame)
                detection_details = {
                    'method': 'hog_fallback',
                    'num_people': num_people,
                    'confidence': confidence,
                    'detections': detections,
                    'threshold': 0.4
                }
                logger.debug(f"HOG fallback detection: {num_people} people, conf={confidence:.3f}")
            
            person_detected = num_people > 0
            
            # Save debug frame with annotations if detection state might change or periodically
            should_save_debug = (person_detected != self.current_state or 
                               abs(num_people - getattr(self, '_last_num_people', 0)) > 0 or
                               getattr(self, '_debug_frame_counter', 0) % 30 == 0)  # Every 30 frames
            
            if should_save_debug:
                try:
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    method_label = detection_details['method']
                    debug_filename = f"/tmp/debug_frame_{method_label}_{int(time.time())}_{num_people}people_conf{confidence:.3f}.jpg"
                    cv2.imwrite(debug_filename, annotated_frame)
                    logger.info(f"Saved {method_label} debug frame: {debug_filename}")
                    
                    # Track for next comparison
                    self._last_num_people = num_people
                    self._debug_frame_counter = getattr(self, '_debug_frame_counter', 0) + 1
                except Exception as e:
                    logger.warning(f"Failed to save debug frame: {e}")
            
        except Exception as e:
            logger.error(f"Person detection error: {e}")
            return {
                'person_detected': False,
                'confidence': 0.0,
                'stable_state': self.current_state,
                'stable_state_changed': False,
                'consecutive_readings': max(self.consecutive_present, self.consecutive_absent),
                'detection_details': {'error': str(e)},
                'frame_info': {'frame_shape': frame.shape}
            }
        
        # Apply hysteresis logic
        stable_state_changed = False
        
        if person_detected:
            self.consecutive_present += 1
            self.consecutive_absent = 0
            
            # Check if we should transition to "present"
            if not self.current_state and self.consecutive_present >= self.hysteresis_present:
                self.current_state = True
                stable_state_changed = True
                logger.info(f"Person detected (after {self.consecutive_present} consecutive readings)")
        else:
            self.consecutive_absent += 1
            self.consecutive_present = 0
            
            # Check if we should transition to "absent"
            if self.current_state and self.consecutive_absent >= self.hysteresis_absent:
                self.current_state = False
                stable_state_changed = True
                logger.info(f"Person absent (after {self.consecutive_absent} consecutive readings)")
        
        # Track performance
        detection_time = time.time() - detection_start
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
        
        self.last_detection_time = time.time()
        
        return {
            'person_detected': person_detected,
            'num_people': num_people,  # Always report actual count
            'confidence': confidence,
            'stable_state': self.current_state,
            'stable_state_changed': stable_state_changed,
            'consecutive_readings': self.consecutive_present if person_detected else self.consecutive_absent,
            'detection_details': detection_details,
            'frame_info': {
                'frame_shape': frame.shape,
                'detection_time': detection_time
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and detection statistics."""
        avg_capture_time = np.mean(self.frame_capture_times) if self.frame_capture_times else 0
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        
        return {
            'total_frames': self.total_frames,
            'failed_captures': self.failed_captures,
            'success_rate': (self.total_frames - self.failed_captures) / max(self.total_frames, 1),
            'avg_capture_time': avg_capture_time,
            'avg_detection_time': avg_detection_time,
            'current_state': self.current_state,
            'consecutive_present': self.consecutive_present,
            'consecutive_absent': self.consecutive_absent,
            'hysteresis': {
                'present_threshold': self.hysteresis_present,
                'absent_threshold': self.hysteresis_absent
            }
        }
    
    def reset_state(self):
        """Reset detection state and counters."""
        self.consecutive_present = 0
        self.consecutive_absent = 0
        self.current_state = False
        logger.info("Detection state reset")


# Convenience function for testing
async def test_reolink_frame_detector():
    """Test function for the Reolink frame detector."""
    detector = ReolinkFrameDetector(
        host='192.168.2.229',
        user='admin',
        password='feedthefiresia360',  # You should use env var in production
        https=True
    )
    
    detector.set_hysteresis(present=3, absent=8)
    
    await detector.start()
    
    try:
        print("Testing Reolink frame-based person detection...")
        print("Stand in front of the camera to test detection.")
        
        for i in range(30):
            result = await detector.detect_person()
            
            print(f"Frame {i+1:2d}: "
                  f"Raw={result['person_detected']:5} | "
                  f"People={result['num_people']} | "
                  f"Stable={result['stable_state']:5} | "
                  f"Confidence={result['confidence']:.3f} | "
                  f"Consecutive={result['consecutive_readings']:2d}")
            
            if result['stable_state_changed']:
                state = "PRESENT" if result['stable_state'] else "ABSENT"
                print(f"  >>> STATE CHANGE: {state} <<<")
            
            await asyncio.sleep(1.0)
            
    finally:
        await detector.stop()
        
        # Print final stats
        stats = detector.get_stats()
        print(f"\nFinal Stats:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Avg capture time: {stats.get('avg_capture_time', 0):.3f}s")
        print(f"  Avg detection time: {stats.get('avg_detection_time', 0):.3f}s")


if __name__ == "__main__":
    asyncio.run(test_reolink_frame_detector())

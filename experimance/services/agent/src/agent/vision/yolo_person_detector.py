"""
YOLO11-based person detection for Reolink camera frames.

Uses the lightweight YOLO11n model for fast, accurate person detection optimized for CPU inference.
Designed to work with both color and infrared/night vision camera feeds.
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field
import asyncio
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not available, YOLO person detection disabled")

from experimance_common.constants import MODELS_DIR

logger = logging.getLogger(__name__)


class YOLO11DetectionConfig(BaseModel):
    """Configuration for YOLO11 person detection."""
    confidence_threshold: float = Field(
        default=0.6,
        description="YOLO11 minimum confidence threshold (0.0-1.0)"
    )
    edge_filter_percent: float = Field(
        default=0.1,
        description="Filter detections within this percent of frame edges (0.0-0.5)"
    )
    min_person_width: int = Field(
        default=40,
        description="Minimum person detection width in pixels"
    )
    min_person_height: int = Field(
        default=80,
        description="Minimum person detection height in pixels"
    )
    max_detections: int = Field(
        default=10,
        description="Maximum number of person detections to process per frame"
    )
    input_size: int = Field(
        default=640,
        description="YOLO11 input image size (320=fast, 640=balanced, 1280=accurate)"
    )
    device: str = Field(
        default="cpu",
        description="Device for YOLO11 inference ('cpu', 'cuda', 'mps')"
    )
    model_name: str = Field(
        default="yolo11n.pt",
        description="YOLO11 model name (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)"
    )
    save_debug_frames: bool = Field(
        default=False,
        description="Save annotated debug frames for troubleshooting"
    )
    debug_frame_interval: int = Field(
        default=10,
        description="Save every Nth frame when debug frames are enabled"
    )

class YOLO11PersonDetector:
    """
    YOLO11n-based person detection optimized for CPU inference.
    
    Features:
    - Uses YOLO11n (nano) model for speed and efficiency
    - Works with color and infrared/grayscale images
    - Edge filtering to avoid false positives from walls/furniture
    - Confidence-based filtering
    - Debug frame saving capabilities
    - Configurable parameters for different deployment scenarios
    """
    
    def __init__(self, config: Optional[YOLO11DetectionConfig] = None):
        """
        Initialize YOLO11n person detector with configuration.
        
        Args:
            config: YOLO11DetectionConfig instance. If None, uses default configuration.
        """
        self.config = config or YOLO11DetectionConfig()
        
        self.model = None
        self._frame_count = 0
        self._detection_times = []
        self._total_raw_detections = 0
        self._total_filtered_detections = 0
        
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available - install with: uv add ultralytics")
            return
            
        self._load_model()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'YOLO11PersonDetector':
        """Create detector from configuration dictionary."""
        # Create Pydantic config from dictionary
        config = YOLO11DetectionConfig(**config_dict)
        return cls(config)
    
    @classmethod 
    def from_reolink_config(cls, reolink_config) -> 'YOLO11PersonDetector':
        """
        Create detector from a FireReolinkConfig.
        
        Args:
            reolink_config: FireReolinkConfig instance
            
        Returns:
            YOLO11PersonDetector instance with default YOLO config
        """
        # For now, just use default config since FireReolinkConfig doesn't have YOLO fields
        logger.info("Creating YOLO detector with default configuration from FireReolinkConfig")
        return cls()
    
    def _load_model(self):
        """Load and configure YOLO11n model for person detection."""
        logger.info(f"Loading YOLO11n model for person detection (device: {self.config.device}, input_size: {self.config.input_size})")
        try:
            # Construct model path using MODELS_DIR
            model_path = MODELS_DIR / self.config.model_name
            
            # Create models directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load YOLO11 model (will download to models dir if not present)
            logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(str(model_path))
            
            # Configure device for inference
            self.model.to(self.config.device)
            
            # Run a quick warm-up prediction to initialize everything
            dummy_image = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_image, 
                conf=self.config.confidence_threshold, 
                classes=[0], 
                verbose=False,
                imgsz=self.config.input_size
            )
            
            logger.info(f"✅ YOLO11n model loaded and ready (device: {self.config.device}) from {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO11n model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if YOLO detection is available and ready."""
        return YOLO_AVAILABLE and self.model is not None
    
    def detect_people(self, frame: np.ndarray, save_debug: bool = False) -> Tuple[int, float, List[Tuple[float, Tuple[int, int, int, int]]]]:
        """
        Detect people in the given frame using YOLO11n.
        
        Args:
            frame: Input image frame (BGR format, color or grayscale)
            save_debug: Whether to save annotated debug frames
            
        Returns:
            Tuple of (person_count, max_confidence, detections_list)
            detections_list contains (confidence, (x, y, w, h)) tuples
        """
        if not self.is_available():
            logger.error("YOLO11n detector not available")
            return 0, 0.0, []
            
        self._frame_count += 1
        start_time = time.time()
        
        try:
            # Run YOLO11n inference - class 0 is 'person' in COCO dataset
            # YOLO automatically handles scaling: input image -> resize to input_size -> detect -> scale back to original
            results = self.model.predict(
                frame, 
                conf=self.config.confidence_threshold,
                classes=[0],  # Only detect persons
                verbose=False,
                imgsz=self.config.input_size  # Configurable input size (640, 320, or 1280)
            )
            
            detections = []
            filtered_detections = []
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate edge margins for filtering false positives
            edge_margin_x = int(frame_width * self.config.edge_filter_percent)
            edge_margin_y = int(frame_height * self.config.edge_filter_percent)
            
            # Process detection results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract bounding box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Convert to x, y, width, height format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Store raw detection for statistics
                        detections.append((confidence, (x, y, w, h)))
                        
                        # Apply filtering criteria
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Filter criteria
                        too_small = w < self.config.min_person_width or h < self.config.min_person_height
                        at_edge = (center_x < edge_margin_x or center_x > frame_width - edge_margin_x or
                                 center_y < edge_margin_y or center_y > frame_height - edge_margin_y)
                        
                        if too_small or at_edge:
                            logger.debug(f"Filtered detection: conf={confidence:.3f}, size=({w},{h}), "
                                       f"center=({center_x},{center_y}), too_small={too_small}, at_edge={at_edge}")
                            continue
                            
                        filtered_detections.append((confidence, (x, y, w, h)))
                        
                        if len(filtered_detections) >= self.config.max_detections:
                            break
            
            # Sort by confidence (highest first)
            filtered_detections.sort(key=lambda x: x[0], reverse=True)
            
            # Calculate results
            person_count = len(filtered_detections)
            max_confidence = max([d[0] for d in filtered_detections], default=0.0)
            
            # Track performance statistics
            detection_time = time.time() - start_time
            self._detection_times.append(detection_time)
            if len(self._detection_times) > 100:
                self._detection_times.pop(0)
                
            self._total_raw_detections += len(detections)
            self._total_filtered_detections += person_count
            
            # Detailed logging
            if len(detections) != person_count:
                logger.debug(f"YOLO11n frame {self._frame_count}: {len(detections)} raw → {person_count} filtered "
                           f"detections, max_conf={max_confidence:.3f}, time={detection_time:.3f}s")
            elif person_count > 0:
                logger.debug(f"YOLO11n frame {self._frame_count}: {person_count} people detected, "
                           f"max_conf={max_confidence:.3f}, time={detection_time:.3f}s")
            
            # Save debug frame if requested
            if save_debug and (person_count > 0 or self._frame_count % 20 == 0):
                self._save_debug_frame(frame, filtered_detections, len(detections), person_count)
                
            return person_count, max_confidence, filtered_detections
            
        except Exception as e:
            logger.error(f"YOLO11n detection error on frame {self._frame_count}: {e}")
            return 0, 0.0, []
    
    def _save_debug_frame(self, frame: np.ndarray, detections: List[Tuple[float, Tuple[int, int, int, int]]], 
                         raw_count: int, filtered_count: int):
        """Save annotated debug frame showing detections."""
        try:
            debug_frame = frame.copy()
            
            # Draw detection boxes and labels
            for i, (confidence, (x, y, w, h)) in enumerate(detections):
                # Draw bounding box (green for valid detections)
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Draw confidence label with background
                label = f"Person {i+1}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(debug_frame, (x, y - label_size[1] - 15), 
                             (x + label_size[0] + 10, y), (0, 255, 0), -1)
                cv2.putText(debug_frame, label, (x + 5, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add frame statistics
            stats_text = [
                f"Frame {self._frame_count}: {filtered_count}/{raw_count} people (YOLO11n)",
                f"Confidence threshold: {self.config.confidence_threshold}",
                f"Frame: {frame.shape[1]}x{frame.shape[0]}"
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = 35 + i * 25
                cv2.putText(debug_frame, text, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(debug_frame, text, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Save debug frame
            timestamp = int(time.time())
            debug_filename = f"/tmp/yolo11_debug_{self._frame_count:05d}_{timestamp}_{filtered_count}people.jpg"
            cv2.imwrite(debug_filename, debug_frame)
            
            logger.info(f"💾 Saved YOLO11n debug frame: {debug_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save YOLO11n debug frame: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance and detection statistics."""
        if not self._detection_times:
            return {}
            
        avg_detection_time = sum(self._detection_times) / len(self._detection_times)
        fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0.0
        filter_efficiency = (self._total_raw_detections - self._total_filtered_detections) / max(self._total_raw_detections, 1)
        
        return {
            'frames_processed': self._frame_count,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'estimated_fps': fps,
            'total_raw_detections': self._total_raw_detections,
            'total_filtered_detections': self._total_filtered_detections,
            'filter_efficiency_percent': filter_efficiency * 100,
            'confidence_threshold': self.config.confidence_threshold,
            'model_available': self.is_available()
        }


async def test_yolo11_detector():
    """Test function for YOLO11n person detection."""
    import aiohttp
    import os
    from pathlib import Path
    
    logger.info("🧪 Testing YOLO11n person detector...")
    
    # Initialize detector with Pydantic configuration
    config = YOLO11DetectionConfig(
        confidence_threshold=0.5,
        edge_filter_percent=0.15,
        min_person_width=50,
        min_person_height=100,
        save_debug_frames=True
    )
    detector = YOLO11PersonDetector(config)
    
    if not detector.is_available():
        logger.error("❌ YOLO11n detector not available")
        return
    
    # Test with Reolink camera frames
    snapshot_url = f"https://{os.getenv('REOLINK_IP')}/cgi-bin/api.cgi"
    params = {
        'cmd': 'Snap',
        'channel': 0,
        'user': os.getenv("REOLINK_USER"),
        'password': os.getenv("REOLINK_PASSWORD")
    }
    
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False),
        timeout=aiohttp.ClientTimeout(total=5)
    ) as session:
        
        try:
            for frame_num in range(10):
                logger.info(f"\n--- Testing frame {frame_num + 1} ---")
                
                # Capture frame from camera
                async with session.get(snapshot_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Failed to capture frame: {response.status}")
                        continue
                        
                    image_data = await response.read()
                    
                # Decode image
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error("Failed to decode image")
                    continue
                
                # Run YOLO11n detection
                start_time = time.time()
                people_count, max_confidence, detections = detector.detect_people(frame, save_debug=True)
                detection_time = time.time() - start_time
                
                # Report results
                logger.info(f"🔍 Detection results:")
                logger.info(f"   👥 People detected: {people_count}")
                logger.info(f"   🎯 Max confidence: {max_confidence:.3f}")
                logger.info(f"   ⏱️  Detection time: {detection_time:.3f}s")
                
                if detections:
                    logger.info(f"   📦 Bounding boxes:")
                    for i, (conf, (x, y, w, h)) in enumerate(detections):
                        logger.info(f"      Person {i+1}: conf={conf:.3f}, bbox=({x}, {y}, {w}, {h})")
                
                # Small delay between frames
                await asyncio.sleep(2.0)
                
        except Exception as e:
            logger.error(f"Test error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final statistics
    stats = detector.get_statistics()
    logger.info(f"\n📊 YOLO11n Detection Statistics:")
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the test
    asyncio.run(test_yolo11_detector())

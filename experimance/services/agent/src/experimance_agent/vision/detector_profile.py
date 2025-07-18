"""
Detector profile configuration for CPU audience detection tuning.

Provides a clean separation between operational configuration (agent.toml)
and detector tuning parameters (detector profiles).
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import tomllib
import tomli_w
import logging

from experimance_common.constants import AGENT_SERVICE_DIR

logger = logging.getLogger(__name__)


class HOGParams(BaseModel):
    """HOG person detection parameters."""
    win_stride_x: int = Field(default=8, description="Window stride X (larger = faster, less accurate)")
    win_stride_y: int = Field(default=8, description="Window stride Y (larger = faster, less accurate)")
    padding_x: int = Field(default=4, description="Padding X around detection window")
    padding_y: int = Field(default=4, description="Padding Y around detection window")
    scale: float = Field(default=1.05, description="Pyramid scale factor (smaller = more scales, slower)")
    hit_threshold: float = Field(default=0.0, description="Detection confidence threshold (lower = more detections)")
    group_threshold: int = Field(default=2, description="Non-maximum suppression threshold")


class MOG2Params(BaseModel):
    """MOG2 background subtractor parameters."""
    detect_shadows: bool = Field(default=True, description="Enable shadow detection")
    var_threshold: float = Field(default=50.0, description="Variance threshold (lower = more sensitive)")
    history: int = Field(default=300, description="Number of frames for background model")


class DetectionParams(BaseModel):
    """General detection parameters."""
    detection_scale_factor: float = Field(default=0.5, description="Frame scaling factor (smaller = faster)")
    min_person_height: int = Field(default=60, description="Minimum person height in pixels (scaled frame)")
    motion_threshold: int = Field(default=1000, description="Minimum contour area for motion detection")
    motion_intensity_threshold: float = Field(default=0.01, description="Motion intensity threshold (0-1)")
    detection_history_size: int = Field(default=5, description="Number of frames for temporal smoothing")
    stability_threshold: float = Field(default=0.6, description="Majority vote threshold for stability")
    
    # Detector enable/disable flags
    enable_hog_detection: bool = Field(default=True, description="Enable HOG person detection")
    enable_face_detection: bool = Field(default=False, description="Enable face detection")
    enable_motion_detection: bool = Field(default=True, description="Enable motion detection")


class FaceDetectionParams(BaseModel):
    """Face detection parameters for YuNet."""
    model_path: str = Field(default="models/face_detection_yunet_2023mar.onnx", description="Path to YuNet ONNX model")
    input_size: tuple[int, int] = Field(default=(320, 320), description="Input size for face detector")
    score_threshold: float = Field(default=0.7, description="Face detection confidence threshold")
    nms_threshold: float = Field(default=0.3, description="Non-maximum suppression threshold")
    top_k: int = Field(default=5000, description="Maximum number of faces to detect")
    min_face_size: int = Field(default=30, description="Minimum face size in pixels")


class ConfidenceParams(BaseModel):
    """Confidence calculation and fusion parameters."""
    person_base_confidence: float = Field(default=0.3, description="Base confidence for person detection")
    person_count_weight: float = Field(default=0.2, description="Weight per additional person detected")
    person_weight_factor: float = Field(default=0.4, description="HOG detection weight influence")
    motion_confidence_weight: float = Field(default=0.3, description="Motion confidence weight in fusion")
    motion_only_threshold: float = Field(default=0.4, description="Threshold for motion-only detection")
    motion_only_confidence_factor: float = Field(default=0.6, description="Confidence reduction for motion-only")
    max_combined_confidence: float = Field(default=0.95, description="Maximum combined confidence")
    absence_confidence: float = Field(default=0.9, description="Confidence when no detection")


class DetectorProfile(BaseModel):
    """Complete detector profile configuration."""
    name: str = Field(description="Profile name")
    description: str = Field(description="Profile description")
    environment: str = Field(description="Environment type (indoor/outdoor/gallery/etc.)")
    lighting: str = Field(description="Lighting conditions (bright/dim/mixed/etc.)")
    
    hog: HOGParams = Field(default_factory=HOGParams)
    mog2: MOG2Params = Field(default_factory=MOG2Params)
    detection: DetectionParams = Field(default_factory=DetectionParams)
    face: FaceDetectionParams = Field(default_factory=FaceDetectionParams)
    confidence: ConfidenceParams = Field(default_factory=ConfidenceParams)
    
    @classmethod
    def load_from_file(cls, profile_path: Path) -> "DetectorProfile":
        """Load detector profile from TOML file."""
        try:
            with open(profile_path, 'rb') as f:
                data = tomllib.load(f)
            return cls(**data)
        except Exception as e:
            logger.error(f"Failed to load detector profile from {profile_path}: {e}")
            raise
    
    def save_to_file(self, profile_path: Path) -> None:
        """Save detector profile to TOML file."""
        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(profile_path, 'wb') as f:
                tomli_w.dump(self.model_dump(), f)
            logger.info(f"Saved detector profile to {profile_path}")
        except Exception as e:
            logger.error(f"Failed to save detector profile to {profile_path}: {e}")
            raise
    
    def get_performance_mode_params(self, mode: str) -> Dict[str, Any]:
        """Get parameters for a specific performance mode."""
        base_params = self.detection.model_dump()
        
        if mode == "fast":
            base_params.update({
                "detection_scale_factor": 0.3,
                "min_person_height": 40,
                "motion_threshold": 2000
            })
        elif mode == "balanced":
            base_params.update({
                "detection_scale_factor": 0.5,
                "min_person_height": 60,
                "motion_threshold": 1000
            })
        elif mode == "accurate":
            base_params.update({
                "detection_scale_factor": 0.7,
                "min_person_height": 80,
                "motion_threshold": 500
            })
        
        return base_params
    
    def update_from_performance_mode(self, mode: str) -> None:
        """Update detection parameters based on performance mode."""
        params = self.get_performance_mode_params(mode)
        for key, value in params.items():
            if hasattr(self.detection, key):
                setattr(self.detection, key, value)


def create_default_profiles() -> Dict[str, DetectorProfile]:
    """Create default detector profiles for common environments."""
    
    profiles = {}
    
    # Indoor office/studio profile
    profiles["indoor_office"] = DetectorProfile(
        name="Indoor Office",
        description="Optimized for indoor office/studio environments with stable lighting",
        environment="indoor",
        lighting="bright",
        detection=DetectionParams(
            detection_scale_factor=0.5,
            min_person_height=60,
            motion_threshold=800,
            motion_intensity_threshold=0.008,
            stability_threshold=0.7
        ),
        mog2=MOG2Params(
            var_threshold=40.0,
            history=400
        )
    )
    
    # Face detection profile for seated audiences
    profiles["face_detection"] = DetectorProfile(
        name="Face Detection",
        description="Face detection optimized for seated audiences (disable HOG)",
        environment="indoor",
        lighting="mixed",
        detection=DetectionParams(
            detection_scale_factor=0.7,
            enable_hog_detection=False,  # Disable HOG
            enable_face_detection=True,  # Enable face detection
            enable_motion_detection=True,
            motion_threshold=800,
            motion_intensity_threshold=0.01,
            stability_threshold=0.6
        ),
        face=FaceDetectionParams(
            score_threshold=0.6,  # Slightly more sensitive
            min_face_size=25,     # Allow smaller faces
            input_size=(416, 416) # Larger input for better detection
        ),
        mog2=MOG2Params(
            var_threshold=35.0,
            history=300
        ),
        confidence=ConfidenceParams(
            person_base_confidence=0.4,
            motion_confidence_weight=0.4
        )
    )
    
    # Face detection optimized profile
    profiles["face_detection"] = DetectorProfile(
        name="Face Detection",
        description="Optimized for face detection with motion, ideal for seated audiences",
        environment="indoor",
        lighting="mixed",
        detection=DetectionParams(
            detection_scale_factor=0.8,  # Higher quality for face detection
            min_person_height=50,
            motion_threshold=800,
            motion_intensity_threshold=0.008,
            stability_threshold=0.5,
            enable_hog_detection=False,      # Disable HOG
            enable_face_detection=True,      # Enable face detection
            enable_motion_detection=True     # Keep motion for confirmation
        ),
        face=FaceDetectionParams(
            score_threshold=0.5,   # More sensitive for face detection
            min_face_size=20,      # Allow smaller faces
            input_size=(320, 320), # Standard YuNet input size
            nms_threshold=0.3,
            top_k=5000
        ),
        mog2=MOG2Params(
            var_threshold=40.0,
            history=250
        ),
        confidence=ConfidenceParams(
            person_base_confidence=0.0,  # No HOG, so no person confidence
            motion_confidence_weight=0.4,
            motion_only_threshold=0.3,
            motion_only_confidence_factor=0.7
        )
    )
    
    # Gallery/museum profile
    profiles["gallery_dim"] = DetectorProfile(
        name="Gallery Dim",
        description="Optimized for gallery/museum with dim, controlled lighting",
        environment="indoor",
        lighting="dim",
        detection=DetectionParams(
            detection_scale_factor=0.6,
            min_person_height=50,
            motion_threshold=600,
            motion_intensity_threshold=0.005,
            stability_threshold=0.6
        ),
        mog2=MOG2Params(
            var_threshold=30.0,
            history=500
        ),
        hog=HOGParams(
            hit_threshold=-0.2,  # More sensitive for dim lighting
            scale=1.03  # More scales for better detection
        )
    )
    
    # Outdoor/bright profile
    profiles["outdoor_bright"] = DetectorProfile(
        name="Outdoor Bright",
        description="Optimized for outdoor or very bright environments",
        environment="outdoor", 
        lighting="bright",
        detection=DetectionParams(
            detection_scale_factor=0.4,
            min_person_height=80,
            motion_threshold=1500,
            motion_intensity_threshold=0.015,
            stability_threshold=0.8
        ),
        mog2=MOG2Params(
            var_threshold=60.0,
            history=200,
            detect_shadows=True
        ),
        hog=HOGParams(
            hit_threshold=0.2,  # Less sensitive for bright conditions
            win_stride_x=6,
            win_stride_y=6
        )
    )
    
    # Workshop/cluttered profile
    profiles["workshop_cluttered"] = DetectorProfile(
        name="Workshop Cluttered",
        description="Optimized for cluttered environments with variable lighting",
        environment="indoor",
        lighting="mixed",
        detection=DetectionParams(
            detection_scale_factor=0.7,
            min_person_height=70,
            motion_threshold=1200,
            motion_intensity_threshold=0.012,
            stability_threshold=0.8,
            detection_history_size=7  # More smoothing for cluttered environments
        ),
        mog2=MOG2Params(
            var_threshold=45.0,
            history=350
        ),
        confidence=ConfidenceParams(
            motion_only_threshold=0.5,  # Higher threshold in cluttered space
            person_base_confidence=0.4
        )
    )
    
    return profiles


def get_profile_directory() -> Path:
    """Get the default profile directory."""
    return AGENT_SERVICE_DIR / "profiles"


def load_profile(profile_name: str, profile_dir: Optional[Path] = None) -> DetectorProfile:
    """Load a detector profile by name."""
    if profile_dir is None:
        profile_dir = get_profile_directory()
    
    profile_path = profile_dir / f"{profile_name}.toml"
    
    if not profile_path.exists():
        # Try to create default profiles if they don't exist
        create_default_profile_files(profile_dir)
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Detector profile '{profile_name}' not found at {profile_path}")
    
    return DetectorProfile.load_from_file(profile_path)


def create_default_profile_files(profile_dir: Path) -> None:
    """Create default profile files in the specified directory."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    profiles = create_default_profiles()
    for name, profile in profiles.items():
        profile_path = profile_dir / f"{name}.toml"
        if not profile_path.exists():
            profile.save_to_file(profile_path)
            logger.info(f"Created default profile: {profile_path}")


def list_available_profiles(profile_dir: Optional[Path] = None) -> list[str]:
    """List all available detector profiles."""
    if profile_dir is None:
        profile_dir = get_profile_directory()
    
    if not profile_dir.exists():
        create_default_profile_files(profile_dir)
    
    return [p.stem for p in profile_dir.glob("*.toml")]

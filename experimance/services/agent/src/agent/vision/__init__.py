"""
Vision processing components for the Experimance Agent Service.

This module provides webcam capture, audience detection, and vision language model
processing capabilities for scene understanding and interaction.
"""

from .webcam import WebcamManager
from .vlm import VLMProcessor
from .audience_detector import AudienceDetector
from .cpu_audience_detector import CPUAudienceDetector
from .reolink_audience_detector import ReolinkAudienceDetector, create_reolink_detector
from .detector_profile import DetectorProfile, load_profile, list_available_profiles, create_default_profile_files

__all__ = [
    "WebcamManager", 
    "VLMProcessor", 
    "AudienceDetector", 
    "CPUAudienceDetector",
    "ReolinkAudienceDetector",
    "create_reolink_detector",
    "DetectorProfile",
    "load_profile",
    "list_available_profiles",
    "create_default_profile_files"
]

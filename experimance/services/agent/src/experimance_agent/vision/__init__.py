"""
Vision processing components for the Experimance Agent Service.

This module provides webcam capture, audience detection, and vision language model
processing capabilities for scene understanding and interaction.
"""

from .webcam import WebcamManager
from .vlm import VLMProcessor
from .audience_detector import AudienceDetector
from .cpu_audience_detector import CPUAudienceDetector

__all__ = ["WebcamManager", "VLMProcessor", "AudienceDetector", "CPUAudienceDetector"]

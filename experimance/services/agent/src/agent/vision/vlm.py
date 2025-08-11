"""
Vision Language Model (VLM) processing for the Experimance Agent Service.

Provides scene understanding and visual analysis using Moondream and other VLM models.
Handles image analysis, audience detection queries, and scene description generation.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from ..config import VisionConfig

logger = logging.getLogger(__name__)


class VLMProcessor:
    """
    Vision Language Model processor for scene understanding and analysis.
    
    Uses Moondream (or other VLM models) to analyze webcam frames and provide
    natural language descriptions of the scene, audience presence detection,
    and interaction context for the conversation AI.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.vlm_device
        self.is_loaded = False
        
        # Analysis caching
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.last_analysis_time: float = 0.0
        
        # Predefined prompts for different analysis types
        self.prompts = {
            "audience_detection": (
                "Look at this image carefully. Are there any people visible in the scene? "
                "Respond with 'YES' if you can see any people (full body, partial, faces, etc.) "
                "or 'NO' if no people are visible. Then briefly describe what you see."
            ),
            "scene_description": (
                "Describe this scene in 1-2 sentences. Focus on: "
                "1) What objects or people are visible "
                "2) The general setting/environment "
                "3) Any notable activities or interactions "
                "Keep it concise and factual."
            ),
            "interaction_context": (
                "Analyze this scene for an interactive art installation. Describe: "
                "1) How many people appear to be present "
                "2) What they seem to be doing or looking at "
                "3) Their apparent level of engagement "
                "Keep it brief and relevant for conversation context."
            ),
            "change_detection": (
                "Compare this current scene to what you might expect in an art installation space. "
                "Describe any notable changes, movements, or new elements. "
                "Focus on people, objects, or activities that suggest active engagement."
            )
        }
    
    async def start(self):
        """Initialize and load the VLM model."""
        if not self.config.vlm_enabled:
            logger.info("VLM processing disabled in configuration")
            return
            
        try:
            await self._load_model()
            self.is_loaded = True
            logger.info(f"VLM processor initialized with {self.config.vlm_model} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VLM processor: {e}")
            raise
    
    async def stop(self):
        """Clean up VLM resources."""
        if self.model:
            del self.model
            self.model = None
            
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_loaded = False
        logger.info("VLM processor stopped")
    
    async def _load_model(self):
        """Load the specified VLM model."""
        model_name = self.config.vlm_model.lower()
        
        if model_name == "moondream":
            await self._load_moondream()
        else:
            raise ValueError(f"Unsupported VLM model: {model_name}")
    
    async def _load_moondream(self):
        """Load the Moondream model."""
        try:
            # Import moondream components
            from transformers import AutoModelForCausalLM, CodeGenTokenizerFast
            
            # Load model and tokenizer
            model_id = "vikhyatk/moondream2"
            
            logger.info(f"Loading Moondream model from {model_id}...")
            
            # Load tokenizer
            self.tokenizer = CodeGenTokenizerFast.from_pretrained(model_id)
            
            # Load model with appropriate device and precision settings
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda" and torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            
            # Move to device if not using device_map
            if "device_map" not in load_kwargs:
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Moondream model loaded successfully")
            
        except ImportError as e:
            logger.error("Failed to import required libraries for Moondream. "
                        "Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}")
            raise
    
    async def analyze_scene(self, frame: np.ndarray, 
                          analysis_type: str = "scene_description") -> Dict[str, Any]:
        """
        Analyze a video frame using the VLM.
        
        Args:
            frame: RGB image frame (numpy array)
            analysis_type: Type of analysis to perform (see self.prompts keys)
            
        Returns:
            dict: Analysis results with description, confidence, and metadata
        """
        if not self.is_loaded:
            return {"error": "VLM not loaded"}
            
        try:
            # Convert numpy array to PIL Image
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame
            
            # Get appropriate prompt
            prompt = self.prompts.get(analysis_type, self.prompts["scene_description"])
            
            # Perform analysis
            start_time = time.time()
            response = await self._query_model(image, prompt)
            analysis_time = time.time() - start_time
            
            # Parse and structure the response
            result = {
                "analysis_type": analysis_type,
                "description": response,
                "timestamp": time.time(),
                "analysis_time": analysis_time,
                "model": self.config.vlm_model,
                "success": True
            }
            
            # Add audience detection specific parsing
            if analysis_type == "audience_detection":
                result["audience_detected"] = self._parse_audience_detection(response)
            
            # Cache the result
            self.last_analysis = result
            self.last_analysis_time = result["timestamp"]
            
            return result
            
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return {
                "analysis_type": analysis_type,
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }
    
    async def _query_model(self, image: Image.Image, prompt: str) -> str:
        """
        Query the VLM model with an image and prompt.
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the analysis
            
        Returns:
            str: Model response text
        """
        try:
            # This is model-specific implementation for Moondream
            if self.config.vlm_model.lower() == "moondream":
                return await self._query_moondream(image, prompt)
            else:
                raise ValueError(f"Unsupported model: {self.config.vlm_model}")
                
        except Exception as e:
            logger.error(f"Model query failed: {e}")
            raise
    
    async def _query_moondream(self, image: Image.Image, prompt: str) -> str:
        """
        Query Moondream model specifically.
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for analysis
            
        Returns:
            str: Moondream response text
        """
        try:
            # Use the model's answer method
            with torch.no_grad():
                # Run in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.model.answer_question(self.model.encode_image(image), prompt, self.tokenizer)
                )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Moondream query failed: {e}")
            raise
    
    def _parse_audience_detection(self, response: str) -> bool:
        """
        Parse audience detection response to extract boolean result.
        
        Args:
            response: VLM response text
            
        Returns:
            bool: True if audience detected, False otherwise
        """
        response_lower = response.lower()
        
        # Look for explicit YES/NO responses
        if response_lower.startswith("yes"):
            return True
        elif response_lower.startswith("no"):
            return False
        
        # Look for people-related keywords
        people_keywords = ["person", "people", "human", "man", "woman", "child", "visitor", "someone"]
        negative_keywords = ["no one", "nobody", "empty", "no people", "no person"]
        
        # Check for negative indicators first
        for keyword in negative_keywords:
            if keyword in response_lower:
                return False
        
        # Check for positive indicators
        for keyword in people_keywords:
            if keyword in response_lower:
                return True
        
        # Default to False if uncertain
        return False
    
    async def detect_audience(self, frame: np.ndarray) -> bool:
        """
        Simplified audience detection method.
        
        Args:
            frame: RGB image frame
            
        Returns:
            bool: True if audience detected, False otherwise
        """
        result = await self.analyze_scene(frame, "audience_detection")
        
        if result.get("success", False):
            return result.get("audience_detected", False)
        else:
            logger.warning("Audience detection failed, defaulting to False")
            return False
    
    async def get_scene_context(self, frame: np.ndarray) -> str:
        """
        Get conversational context about the current scene.
        
        Args:
            frame: RGB image frame
            
        Returns:
            str: Scene description suitable for conversation context
        """
        result = await self.analyze_scene(frame, "interaction_context")
        
        if result.get("success", False):
            return result.get("description", "Scene analysis unavailable")
        else:
            return "Unable to analyze current scene"
    
    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent analysis result.
        
        Returns:
            dict: Last analysis result, or None if no analysis available
        """
        return self.last_analysis.copy() if self.last_analysis else None
    
    def get_analysis_age(self) -> float:
        """
        Get the age of the last analysis in seconds.
        
        Returns:
            float: Age of last analysis in seconds, or float('inf') if no analysis
        """
        if self.last_analysis_time == 0.0:
            return float('inf')
        return time.time() - self.last_analysis_time
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get VLM processor status information.
        
        Returns:
            dict: Status information including model, device, and performance metrics
        """
        status = {
            "enabled": self.config.vlm_enabled,
            "loaded": self.is_loaded,
            "model": self.config.vlm_model,
            "device": self.device,
            "has_analysis": self.last_analysis is not None,
            "analysis_age": self.get_analysis_age()
        }
        
        if self.last_analysis:
            status.update({
                "last_analysis_type": self.last_analysis.get("analysis_type"),
                "last_analysis_time": self.last_analysis.get("analysis_time")
            })
        
        return status

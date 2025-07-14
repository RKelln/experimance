#!/usr/bin/env python3
"""
LLM integration for Sohkepayin story processing.

This module provides an abstraction layer for different LLM providers
to analyze stories and generate image prompts for the Sohkepayin project.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from experimance_common.schemas import StoryHeard, UpdateLocation, Biome, Emotion

logger = logging.getLogger(__name__)


@dataclass
class LocationInference:
    """Result of location inference from a story."""
    
    biome: Biome
    emotion: Emotion
    time_period: Optional[str] = None
    weather: Optional[str] = None
    lighting: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 0.0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def infer_location(self, story_content: str) -> LocationInference:
        """Infer location details from story content."""
        pass
    
    @abstractmethod
    async def update_location(
        self, 
        current_location: LocationInference, 
        update_content: str
    ) -> LocationInference:
        """Update location based on new information."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for story analysis."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: float = 30.0
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            api_key: API key (if None, uses OPENAI_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-2.0)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Import OpenAI here to avoid import errors if not installed
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )
        
        logger.info(f"OpenAI provider initialized: model={model}, timeout={timeout}s")
    
    async def infer_location(self, story_content: str) -> LocationInference:
        """
        Infer location details from story content using OpenAI.
        
        Args:
            story_content: The story text to analyze
            
        Returns:
            LocationInference with biome, emotion, and environmental details
        """
        prompt = self._build_location_inference_prompt(story_content)
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ),
                timeout=self.timeout
            )
            
            return self._parse_location_response(response.choices[0].message.content)
            
        except asyncio.TimeoutError:
            logger.error(f"OpenAI request timed out after {self.timeout}s")
            return self._get_fallback_location()
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return self._get_fallback_location()
    
    async def update_location(
        self, 
        current_location: LocationInference, 
        update_content: str
    ) -> LocationInference:
        """
        Update location inference based on new information.
        
        Args:
            current_location: Current location inference
            update_content: New information to incorporate
            
        Returns:
            Updated LocationInference
        """
        prompt = self._build_location_update_prompt(current_location, update_content)
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ),
                timeout=self.timeout
            )
            
            return self._parse_location_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Location update failed: {e}")
            return current_location  # Return unchanged on error
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Sohkepayin location inference."""
        return """You are an AI assistant for the Sohkepayin (Fire Spirit) art installation. 
Your role is to analyze audience stories and infer environmental settings that will be transformed 
into immersive panoramic visualizations.

Sohkepayin creates dream-like panoramas that wrap around a room. The audience tells personal stories,
and you help translate these into visual settings that capture the emotional and physical environment
of their narrative.

Analyze stories for:
1. BIOME: The natural environment (forest, grassland, wetland, mountain, river, lake, urban, indoors, desert, coast)
2. EMOTION: The emotional tone (joy, sorrow, anger, peace, longing, hope)  
3. TIME_PERIOD: Historical period or time of day if specified
4. WEATHER: Weather conditions if mentioned
5. LIGHTING: Lighting conditions (dawn, dusk, overcast, bright, etc.)
6. DESCRIPTION: Brief atmospheric description for image generation

Respond in JSON format with these fields. Focus on creating evocative, immersive environments
that honor the storyteller's experience while being suitable for panoramic visualization."""

    def _build_location_inference_prompt(self, story_content: str) -> str:
        """Build prompt for initial location inference."""
        return f"""Analyze this personal story and infer the environmental setting for visualization:

STORY:
{story_content}

Please respond with a JSON object containing:
{{
    "biome": "one of: forest, grassland, wetland, mountain, river, lake, urban, indoors, desert, coast",
    "emotion": "one of: joy, sorrow, anger, peace, longing, hope", 
    "time_period": "historical period or time of day if mentioned",
    "weather": "weather conditions if specified",
    "lighting": "lighting conditions",
    "description": "brief atmospheric description for panoramic image generation",
    "confidence": "confidence score 0.0-1.0"
}}

Focus on the environmental and emotional essence that would create a meaningful panoramic visualization."""

    def _build_location_update_prompt(
        self, 
        current_location: LocationInference, 
        update_content: str
    ) -> str:
        """Build prompt for updating location inference."""
        return f"""Update the environmental setting based on new information:

CURRENT SETTING:
- Biome: {current_location.biome}
- Emotion: {current_location.emotion}
- Time Period: {current_location.time_period}
- Weather: {current_location.weather}
- Lighting: {current_location.lighting}
- Description: {current_location.description}

NEW INFORMATION:
{update_content}

Please respond with an updated JSON object using the same format as before.
Preserve elements that are still relevant and update based on the new information."""

    def _parse_location_response(self, response_text: str) -> LocationInference:
        """Parse LLM response into LocationInference object."""
        try:
            import json
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
                data = json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
            
            # Map string values to enums
            biome = Biome(data.get('biome', 'forest'))
            emotion = Emotion(data.get('emotion', 'peace'))
            
            return LocationInference(
                biome=biome,
                emotion=emotion,
                time_period=data.get('time_period'),
                weather=data.get('weather'),
                lighting=data.get('lighting'),
                description=data.get('description'),
                confidence=float(data.get('confidence', 0.5))
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
            return self._get_fallback_location()
    
    def _get_fallback_location(self) -> LocationInference:
        """Get fallback location when LLM fails."""
        return LocationInference(
            biome=Biome.FOREST,
            emotion=Emotion.PEACE,
            time_period="present",
            weather="clear",
            lighting="soft daylight",
            description="a peaceful forest clearing with dappled sunlight",
            confidence=0.1
        )


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize mock provider.
        
        Args:
            delay: Simulated response delay in seconds
        """
        self.delay = delay
        logger.info(f"Mock LLM provider initialized with {delay}s delay")
    
    async def infer_location(self, story_content: str) -> LocationInference:
        """Mock location inference."""
        await asyncio.sleep(self.delay)
        
        # Simple keyword-based mock inference
        content_lower = story_content.lower()
        
        if any(word in content_lower for word in ['forest', 'tree', 'woods']):
            biome = Biome.FOREST
        elif any(word in content_lower for word in ['ocean', 'beach', 'coast', 'sea']):
            biome = Biome.COAST
        elif any(word in content_lower for word in ['mountain', 'hill', 'peak']):
            biome = Biome.MOUNTAIN
        elif any(word in content_lower for word in ['city', 'urban', 'street']):
            biome = Biome.URBAN
        else:
            biome = Biome.GRASSLAND
        
        if any(word in content_lower for word in ['happy', 'joy', 'celebration']):
            emotion = Emotion.JOY
        elif any(word in content_lower for word in ['sad', 'loss', 'grief']):
            emotion = Emotion.SORROW
        elif any(word in content_lower for word in ['angry', 'frustrated', 'mad']):
            emotion = Emotion.ANGER
        else:
            emotion = Emotion.PEACE
        
        return LocationInference(
            biome=biome,
            emotion=emotion,
            time_period="present",
            weather="clear",
            lighting="natural",
            description=f"a {emotion.value} {biome.value} scene",
            confidence=0.8
        )
    
    async def update_location(
        self, 
        current_location: LocationInference, 
        update_content: str
    ) -> LocationInference:
        """Mock location update."""
        await asyncio.sleep(self.delay * 0.5)  # Faster for updates
        
        # Simple mock: just update description
        updated = LocationInference(
            biome=current_location.biome,
            emotion=current_location.emotion,
            time_period=current_location.time_period,
            weather=current_location.weather,
            lighting=current_location.lighting,
            description=f"{current_location.description} (updated)",
            confidence=current_location.confidence
        )
        
        return updated


class LLMManager:
    """Manager for LLM providers with fallback support."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: float = 30.0
    ):
        """
        Initialize LLM manager.
        
        Args:
            provider: Provider name ("openai", "mock")
            model: Model name for the provider
            api_key: API key for the provider
            max_tokens: Maximum tokens in response
            temperature: Response creativity
            timeout: Request timeout
        """
        self.provider_name = provider
        
        if provider == "openai":
            self.provider = OpenAIProvider(
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
        elif provider == "mock":
            self.provider = MockLLMProvider(delay=1.0)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        logger.info(f"LLM manager initialized with {provider} provider")
    
    async def infer_location(self, story: StoryHeard) -> LocationInference:
        """
        Infer location from a StoryHeard message.
        
        Args:
            story: StoryHeard message containing story content
            
        Returns:
            LocationInference with environmental details
        """
        story_content = getattr(story, 'content', str(story))
        logger.info(f"Inferring location from story (length: {len(story_content)} chars)")
        
        result = await self.provider.infer_location(story_content)
        
        logger.info(
            f"Location inferred: {result.biome}/{result.emotion} "
            f"(confidence: {result.confidence:.2f})"
        )
        
        return result
    
    async def update_location(
        self, 
        current_location: LocationInference, 
        update: UpdateLocation
    ) -> LocationInference:
        """
        Update location based on UpdateLocation message.
        
        Args:
            current_location: Current location inference
            update: UpdateLocation message with new information
            
        Returns:
            Updated LocationInference
        """
        update_content = getattr(update, 'content', str(update))
        logger.info(f"Updating location with new info (length: {len(update_content)} chars)")
        
        result = await self.provider.update_location(current_location, update_content)
        
        logger.info(
            f"Location updated: {result.biome}/{result.emotion} "
            f"(confidence: {result.confidence:.2f})"
        )
        
        return result

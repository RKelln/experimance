"""
LLM integration for Sohkepayin story processing.

This module provides an abstraction layer for different LLM providers.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    system_prompt: str = ""
    system_prompt_file: Optional[Path] = None

    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt directly.
        
        Args:
            system_prompt: Complete system prompt with instructions and examples
        """
        self.system_prompt = system_prompt
        self.system_prompt_file = None
        logger.info("System prompt set directly for LLM prompt builder")
    
    def set_system_prompt_file(self, file_path: str | Path):
        """
        Set the system prompt from a file.
        
        Args:
            file_path: Path to file containing the system prompt
        """
        self.system_prompt_file = Path(file_path)
        try:
            self.system_prompt = self.system_prompt_file.read_text(encoding='utf-8')
            logger.info(f"System prompt loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load system prompt from {file_path}: {e}")
            raise

    @abstractmethod
    async def query(self, content: str) -> str | None:
        """
        Query the LLM with the given content.
        
        Args:
            content: prompt to pass to the LLM
            
        Returns:
            Text response from the LLM
        """
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for story analysis."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: float = 30.0,
        **kwargs: Optional[dict]
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
    
    async def query(self, content: str) -> str | None:
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user", 
                            "content": content
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ),
                timeout=self.timeout
            )
            # Extract the string content from the response
            return response.choices[0].message.content if response.choices else None
        
        except asyncio.TimeoutError:
            logger.error(f"OpenAI request timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return None
    

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, delay: float = 0.5):
        """
        Initialize mock provider.
        
        Args:
            delay: Simulated response delay in seconds
        """
        self.delay = delay
        logger.info(f"Mock LLM provider initialized with {delay}s delay")
    
    async def query(self, content: str) -> str | None:
        """Mock location inference."""
        await asyncio.sleep(self.delay)
        
        # Simple keyword-based mock inference
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['forest', 'tree', 'woods']):
            biome = "forest"
        elif any(word in content_lower for word in ['ocean', 'beach', 'coast', 'sea']):
            biome = "coast"
        elif any(word in content_lower for word in ['mountain', 'hill', 'peak']):
            biome = "mountains"
        elif any(word in content_lower for word in ['city', 'urban', 'street']):
            biome = "cityscape"
        else:
            biome = ""
        
        if any(word in content_lower for word in ['happy', 'joy', 'celebration']):
            emotion = "joyful"
        elif any(word in content_lower for word in ['sad', 'loss', 'grief']):
            emotion = "sad"
        elif any(word in content_lower for word in ['angry', 'frustrated', 'mad']):
            emotion = "anger"
        else:
            emotion = "peaceful"
        
        mock_data = {
            "prompt": f"{biome} scene with {emotion} mood",
            "negative_prompt": "blurry, distorted, low quality"
        }
        return json.dumps(mock_data)


def get_llm_provider(provider: str, **kwargs) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        provider: Provider name (openai, mock)
        **kwargs: Additional parameters for provider initialization
        
    Returns:
        An instance of the specified LLMProvider
    """
    if provider == "openai":
        return OpenAIProvider(**kwargs)
    elif provider == "mock":
        return MockLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
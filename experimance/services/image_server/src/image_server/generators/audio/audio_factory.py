"""
Audio generator factory for creating and managing audio generators.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, List, Set

from image_server.generators.audio.audio_generator import AudioGenerator
from image_server.generators.audio.audio_config import BaseAudioGeneratorConfig
from image_server.generators.audio.prompt2audio import Prompt2AudioGenerator
from image_server.generators.audio.audio_config import Prompt2AudioConfig

logger = logging.getLogger(__name__)

# Map audio generator strategies to their config and generator classes
AUDIO_GENERATORS = {
    "prompt2audio": {
        "config_class": Prompt2AudioConfig,
        "generator_class": Prompt2AudioGenerator
    },
}


def create_audio_generator_from_config(
    strategy: str,
    config_data: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60
) -> AudioGenerator:
    """Create an audio generator from configuration data.
    
    Args:
        strategy: The audio generator strategy to use
        config_data: Configuration data for the generator
        output_dir: Directory to store generated audio files (optional)
        timeout: Default timeout for generation in seconds
        
    Returns:
        Configured AudioGenerator instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    if strategy not in AUDIO_GENERATORS:
        available_strategies = list(AUDIO_GENERATORS.keys())
        raise ValueError(f"Unsupported audio generator strategy: {strategy}. "
                         f"Available strategies: {available_strategies}")
    
    # Create a base configuration with common settings
    base_config = {
        "strategy": strategy,
        "timeout": timeout
    }
    
    # Merge with strategy-specific configuration
    config_dict = {**base_config, **config_data}
    
    # Create the config object using the appropriate class
    config_class = AUDIO_GENERATORS[strategy]["config_class"]
    generator_class = AUDIO_GENERATORS[strategy]["generator_class"]
    
    logger.debug(f"Creating {strategy} audio generator with config: {config_dict}")
    
    # Create and return the generator with output_dir as constructor parameter
    config = config_class(**config_dict)
    if output_dir:
        return generator_class(config=config, output_dir=str(output_dir))
    else:
        return generator_class(config=config)


def create_audio_generator(
    config: Union[Dict[str, Any], BaseAudioGeneratorConfig],
    output_dir: Optional[Union[str, Path]] = None
) -> AudioGenerator:
    """Factory function to create audio generators using either a config dict or object.
    
    Args:
        config: Either a BaseAudioGeneratorConfig object or a dict with a 'strategy' key
        output_dir: Optional output directory to override config
        
    Returns:
        Configured AudioGenerator instance
        
    Raises:
        ValueError: If strategy is not supported or config is invalid
    """
    if isinstance(config, dict):
        if "strategy" not in config:
            raise ValueError("Strategy must be specified in config dictionary")
        
        strategy = config["strategy"]
        return create_audio_generator_from_config(strategy, config, output_dir)
    else:
        # We already have a config object, just extract the strategy and use it
        strategy = config.strategy
        config_dict = config.model_dump() if hasattr(config, "model_dump") else vars(config)
        return create_audio_generator_from_config(strategy, config_dict, output_dir)


class AudioGeneratorManager:
    """Manager for dynamic audio generator creation and caching.
    
    This class manages multiple audio generator instances, creating them on-demand
    and caching them for reuse. It supports dynamic strategy selection
    based on AudioRenderRequest.generator attribute.
    """
    
    def __init__(self, default_strategy: str, output_dir: Optional[Union[str, Path]] = None, 
                 timeout: int = 60, default_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize the audio generator manager.
        
        Args:
            default_strategy: Default audio generator strategy to use
            output_dir: Directory to store generated audio files
            timeout: Default timeout for generation in seconds
            default_configs: Default configurations for each strategy
        """
        self.default_strategy = default_strategy
        self.output_dir = output_dir
        self.timeout = timeout
        self.default_configs = default_configs or {}
        self._generators: Dict[str, AudioGenerator] = {}
        self._started_generators: Set[str] = set()

    async def get_generator(self, strategy: Optional[str] = None) -> AudioGenerator:
        """Get or create an audio generator for the specified strategy.
        
        Args:
            strategy: Generator strategy to use (uses default if None)
            
        Returns:
            AudioGenerator instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        strategy = strategy or self.default_strategy
        
        if strategy not in self._generators:
            # Get default config for this strategy
            default_config = self.default_configs.get(strategy, {})
            
            # Create generator
            self._generators[strategy] = create_audio_generator_from_config(
                strategy=strategy,
                config_data=default_config,
                output_dir=self.output_dir,
                timeout=self.timeout
            )
            
            logger.info(f"Created audio generator: {strategy}")
        
        generator = self._generators[strategy]
        
        # Start generator if not already started
        if strategy not in self._started_generators:
            await generator.start()
            self._started_generators.add(strategy)
            logger.debug(f"Started audio generator: {strategy}")
        
        return generator

    async def generate_audio(self, prompt: str, strategy: Optional[str] = None, **kwargs) -> str:
        """Generate audio using the specified strategy.
        
        Args:
            prompt: Text prompt for audio generation
            strategy: Generator strategy to use (uses default if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Path to generated audio file
        """
        generator = await self.get_generator(strategy)
        return await generator.generate_audio(prompt, **kwargs)

    async def list_capabilities(self, strategy: Optional[str] = None) -> Set[str]:
        """Get capabilities for the specified strategy.
        
        Args:
            strategy: Generator strategy to check (uses default if None)
            
        Returns:
            Set of supported capability strings
        """
        generator = await self.get_generator(strategy)
        return generator.get_supported_capabilities()

    async def supports_capability(self, capability: str, strategy: Optional[str] = None) -> bool:
        """Check if a strategy supports a specific capability.
        
        Args:
            capability: Capability to check
            strategy: Generator strategy to check (uses default if None)
            
        Returns:
            True if capability is supported
        """
        generator = await self.get_generator(strategy)
        return generator.supports_capability(capability)

    def get_available_strategies(self) -> List[str]:
        """Get list of available audio generator strategies.
        
        Returns:
            List of strategy names
        """
        return list(AUDIO_GENERATORS.keys())

    async def stop_all(self):
        """Stop all managed generators."""
        for strategy, generator in self._generators.items():
            try:
                await generator.stop()
                logger.debug(f"Stopped audio generator: {strategy}")
            except Exception as e:
                logger.error(f"Error stopping audio generator {strategy}: {e}")
        
        self._generators.clear()
        self._started_generators.clear()
        
    async def restart_generator(self, strategy: Optional[str] = None):
        """Restart a specific generator.
        
        Args:
            strategy: Generator strategy to restart (uses default if None)
        """
        strategy = strategy or self.default_strategy
        
        if strategy in self._generators:
            generator = self._generators[strategy]
            await generator.stop()
            self._started_generators.discard(strategy)
            
            # Remove from cache so it gets recreated
            del self._generators[strategy]
            
            logger.info(f"Restarted audio generator: {strategy}")

    def get_generator_info(self) -> Dict[str, Any]:
        """Get information about all managed generators.
        
        Returns:
            Dict with generator information
        """
        info = {
            "default_strategy": self.default_strategy,
            "available_strategies": self.get_available_strategies(),
            "active_generators": list(self._generators.keys()),
            "started_generators": list(self._started_generators),
            "output_dir": str(self.output_dir) if self.output_dir else None
        }
        
        # Add capability information for each strategy
        capabilities = {}
        for strategy in AUDIO_GENERATORS:
            generator_class = AUDIO_GENERATORS[strategy]["generator_class"]
            capabilities[strategy] = list(getattr(generator_class, 'supported_capabilities', set()))
        info["capabilities"] = capabilities
        
        return info

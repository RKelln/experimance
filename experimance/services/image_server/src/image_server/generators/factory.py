import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, List

from image_server.generators.generator import ImageGenerator
from image_server.generators.config import BaseGeneratorConfig
from image_server.generators.mock.mock_generator import MockImageGenerator
from image_server.generators.mock.mock_generator_config import MockGeneratorConfig
from image_server.generators.fal.fal_comfy_generator import FalComfyGenerator
from image_server.generators.fal.fal_comfy_config import FalComfyGeneratorConfig
from image_server.generators.fal.fal_lightning_i2i_generator import FalLightningI2IGenerator
from image_server.generators.fal.fal_lightning_i2i_config import FalLightningI2IConfig
from image_server.generators.vastai.vastai_generator import VastAIGenerator
from image_server.generators.vastai.vastai_config import VastAIGeneratorConfig
#from image_server.generators.openai.openai_generator import OpenAIGenerator, OpenAIGeneratorConfig
#from image_server.generators.local.sdxl_generator import LocalSDXLGenerator, SDXLGeneratorConfig

logger = logging.getLogger(__name__)

# Map strategies to their respective config and generator classes
# needs to match GENERATOR_NAMES
GENERATORS = {
    "mock": {
        "config_class": MockGeneratorConfig,
        "generator_class": MockImageGenerator
    },
    "fal_comfy": {
        "config_class": FalComfyGeneratorConfig,
        "generator_class": FalComfyGenerator
    },
    "falai_lightning_i2i": {
        "config_class": FalLightningI2IConfig,
        "generator_class": FalLightningI2IGenerator
    },
    "vastai": {
        "config_class": VastAIGeneratorConfig,
        "generator_class": VastAIGenerator
    },
    # "openai": {
    #     "config_class": OpenAIGeneratorConfig,
    #     "generator_class": OpenAIGenerator
    # },
    # "local": {
    #     "config_class": SDXLGeneratorConfig,
    #     "generator_class": LocalSDXLGenerator
    # },
}

def create_generator_from_config(
    strategy: str,
    config_data: Dict[str, Any],
    cache_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60
) -> ImageGenerator:
    """Create an image generator from configuration data.
    
    Args:
        strategy: The generator strategy to use
        config_data: Configuration data for the generator
        cache_dir: Directory to store generated images (optional)
        timeout: Default timeout for generation in seconds
        
    Returns:
        Configured ImageGenerator instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    if strategy not in GENERATORS:
        available_strategies = list(GENERATORS.keys())
        raise ValueError(f"Unsupported generator strategy: {strategy}. "
                         f"Available strategies: {available_strategies}")
    
    # Create a base configuration with common settings
    base_config = {
        "strategy": strategy,
        "timeout": timeout
    }
    
    # Merge with strategy-specific configuration
    config_dict = {**base_config, **config_data}
    
    # Create the config object using the appropriate class
    config_class = GENERATORS[strategy]["config_class"]
    generator_class = GENERATORS[strategy]["generator_class"]
    
    logger.debug(f"Creating {strategy} generator with config: {config_dict}")
    
    # Create and return the generator with output_dir as constructor parameter
    config = config_class(**config_dict)
    if cache_dir:
        return generator_class(config=config, output_dir=str(cache_dir))
    else:
        return generator_class(config=config)

def create_generator(
    config: Union[Dict[str, Any], BaseGeneratorConfig],
    cache_dir: Optional[Union[str, Path]] = None
) -> ImageGenerator:
    """Factory function to create image generators using either a config dict or object.
    
    This is a wrapper around create_generator_from_config for backward compatibility.
    
    Args:
        config: Either a BaseGeneratorConfig object or a dict with a 'strategy' key
        cache_dir: Optional cache directory to override config
        
    Returns:
        Configured ImageGenerator instance
        
    Raises:
        ValueError: If strategy is not supported or config is invalid
    """
    if isinstance(config, dict):
        if "strategy" not in config:
            raise ValueError("Strategy must be specified in config dictionary")
        
        strategy = config["strategy"]
        return create_generator_from_config(strategy, config, cache_dir)
    else:
        # We already have a config object, just extract the strategy and use it
        strategy = config.strategy
        config_dict = config.model_dump() if hasattr(config, "model_dump") else vars(config)
        return create_generator_from_config(strategy, config_dict, cache_dir)


class GeneratorManager:
    """Manager for dynamic generator creation and caching.
    
    This class manages multiple generator instances, creating them on-demand
    and caching them for reuse. It supports dynamic strategy selection
    based on RenderRequest.generator attribute.
    """
    
    def __init__(self, default_strategy: str, cache_dir: Optional[Union[str, Path]] = None, 
                 timeout: int = 60, default_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize the generator manager.
        
        Args:
            default_strategy: Default generator strategy to use
            cache_dir: Directory to store generated images
            timeout: Default timeout for generation in seconds
            default_configs: Default configurations for each strategy
        """
        self.default_strategy = default_strategy
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.default_configs = default_configs or {}
        self._generators: Dict[str, ImageGenerator] = {}
        
        logger.info(f"GeneratorManager initialized with default strategy: {default_strategy}")
        
        # Pre-warm generators that have pre_warm=True in their configuration
        self._pre_warm_generators()
    
    def get_generator(self, strategy: Optional[str] = None, 
                     config_overrides: Optional[Dict[str, Any]] = None) -> ImageGenerator:
        """Get or create a generator for the specified strategy.
        
        Args:
            strategy: Generator strategy to use (defaults to default_strategy)
            config_overrides: Optional configuration overrides for this request
            
        Returns:
            ImageGenerator instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy is None:
            strategy = self.default_strategy
        
        # Create a cache key that includes config overrides if any
        cache_key = strategy
        if config_overrides:
            # Create a deterministic key from sorted config items
            override_key = "_".join(f"{k}={v}" for k, v in sorted(config_overrides.items()))
            cache_key = f"{strategy}_{override_key}"
        
        # Return cached generator if available
        if cache_key in self._generators:
            logger.debug(f"Using cached generator for strategy: {strategy}")
            return self._generators[cache_key]
        
        # Create new generator
        logger.info(f"Creating new generator for strategy: {strategy}")
        
        # Start with default config for this strategy
        config_data = self.default_configs.get(strategy, {}).copy()
        
        # Apply any config overrides
        if config_overrides:
            config_data.update(config_overrides)
        
        # Create the generator
        generator = create_generator_from_config(
            strategy=strategy,
            config_data=config_data,
            cache_dir=self.cache_dir,
            timeout=self.timeout
        )
        
        # Cache the generator
        self._generators[cache_key] = generator
        
        logger.debug(f"Created and cached generator for strategy: {strategy}")
        return generator
    
    def is_image_to_image_generator(self, strategy: Optional[str] = None) -> bool:
        """Check if the specified strategy supports image-to-image generation.
        
        Args:
            strategy: Generator strategy to check (defaults to default_strategy)
            
        Returns:
            True if the strategy supports image-to-image generation
        """
        if strategy is None:
            strategy = self.default_strategy
        
        # Check if strategy name indicates image-to-image capability
        i2i_strategies = {"falai_lightning_i2i"}
        return strategy in i2i_strategies
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available generator strategies.
        
        Returns:
            List of available strategy names
        """
        return list(GENERATORS.keys())
    
    async def stop_all_generators(self):
        """Stop all cached generators and clear the cache."""
        logger.info("Stopping all generators...")
        
        for strategy, generator in self._generators.items():
            try:
                if hasattr(generator, 'stop'):
                    await generator.stop()
                logger.debug(f"Stopped generator for strategy: {strategy}")
            except Exception as e:
                logger.warning(f"Error stopping generator {strategy}: {e}")
        
        self._generators.clear()
        logger.info("All generators stopped and cache cleared")
    
    def _pre_warm_generators(self):
        """Pre-warm generators that have pre_warm=True in their configuration."""
        for strategy, config_data in self.default_configs.items():
            # Check if this strategy should be pre-warmed
            if config_data.get("pre_warm", False):
                logger.info(f"Pre-warming generator for strategy: {strategy}")
                try:
                    # Create the generator immediately (this will trigger pre-warming)
                    self.get_generator(strategy)
                    logger.info(f"Successfully initiated pre-warming for strategy: {strategy}")
                except Exception as e:
                    logger.warning(f"Failed to pre-warm generator for strategy {strategy}: {e}")
                    # Don't fail startup just because pre-warming failed
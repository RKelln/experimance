import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type

from image_server.generators.generator import ImageGenerator
from image_server.generators.config import BaseGeneratorConfig
from image_server.generators.mock.mock_generator import MockImageGenerator
from image_server.generators.mock.mock_generator_config import MockGeneratorConfig
from image_server.generators.fal.fal_comfy_generator import FalComfyGenerator
from image_server.generators.fal.fal_comfy_config import FalComfyGeneratorConfig
#from image_server.generators.openai.openai_generator import OpenAIGenerator, OpenAIGeneratorConfig
#from image_server.generators.local.sdxl_generator import LocalSDXLGenerator, SDXLGeneratorConfig

logger = logging.getLogger(__name__)

# Map strategies to their respective config and generator classes
GENERATORS = {
    "mock": {
        "config_class": MockGeneratorConfig,
        "generator_class": MockImageGenerator
    },
    "falai": {
        "config_class": FalComfyGeneratorConfig,
        "generator_class": FalComfyGenerator
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
    
    # Add cache directory if provided
    if cache_dir:
        base_config["output_dir"] = str(cache_dir)
    
    # Merge with strategy-specific configuration
    config_dict = {**base_config, **config_data}
    
    # Create the config object using the appropriate class
    config_class = GENERATORS[strategy]["config_class"]
    generator_class = GENERATORS[strategy]["generator_class"]
    
    logger.debug(f"Creating {strategy} generator with config: {config_dict}")
    
    # Create and return the generator
    config = config_class(**config_dict)
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
#!/usr/bin/env python3
"""
Audio generator factory that handles subprocess wrapping transparently.

This factory checks if subprocess execution is requested and wraps the generator
automatically, without changing any existing configuration structures.
"""

import logging
from typing import Any, Dict, Union, cast

from image_server.generators.audio.audio_generator import AudioGenerator
from image_server.generators.audio.audio_config import BaseAudioGeneratorConfig
from image_server.generators.subprocess_wrapper import create_subprocess_wrapper

logger = logging.getLogger(__name__)


def create_audio_generator(
    generator_class: type,
    config: BaseAudioGeneratorConfig,
    output_dir: str = "/tmp",
    **kwargs
) -> AudioGenerator:
    """Create an audio generator, optionally wrapped in subprocess for GPU isolation.
    
    This factory function checks if the config specifies subprocess execution
    and automatically wraps the generator if needed, without changing the
    existing configuration structure.
    
    Args:
        generator_class: The audio generator class to instantiate
        config: Audio generator configuration
        output_dir: Directory for output files
        **kwargs: Additional arguments for generator construction
        
    Returns:
        AudioGenerator instance (possibly wrapped in subprocess)
    """
    # Check if subprocess execution is requested
    if getattr(config, 'use_subprocess', False) and getattr(config, 'cuda_visible_devices', None):
        logger.info(f"Creating subprocess wrapper for {generator_class.__name__} with CUDA_VISIBLE_DEVICES={config.cuda_visible_devices}")
        
        # Convert config to dict for subprocess
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif hasattr(config, 'dict'):
            config_dict = config.dict()
        else:
            config_dict = vars(config)
        
        logger.debug(f"Original config_dict: {config_dict}")
        
        # Remove subprocess-specific fields from the wrapped config
        # but keep strategy as it's required by BaseAudioGeneratorConfig
        wrapped_config = config_dict.copy()
        wrapped_config.pop('use_subprocess', None)
        wrapped_config.pop('cuda_visible_devices', None)
        wrapped_config.pop('subprocess_timeout_seconds', None)
        wrapped_config.pop('subprocess_max_retries', None)
        # Keep strategy field - it's required by BaseAudioGeneratorConfig
        
        logger.debug(f"Wrapped config: {wrapped_config}")
        
        # Create subprocess wrapper - keep strategy for the subprocess config
        return cast(AudioGenerator, create_subprocess_wrapper(
            generator_class=f"{generator_class.__module__}.{generator_class.__name__}",
            generator_config=wrapped_config,
            generator_type="audio",
            cuda_visible_devices=config.cuda_visible_devices,
            timeout_seconds=getattr(config, 'subprocess_timeout_seconds', 300),
            max_retries=getattr(config, 'subprocess_max_retries', 3),
            output_dir=output_dir,
            **kwargs
        ))
    else:
        # Create normal generator
        logger.debug(f"Creating normal {generator_class.__name__} instance")
        return generator_class(config=config, output_dir=output_dir, **kwargs)


# Convenience function for the most common case
def create_prompt2audio_generator(config: BaseAudioGeneratorConfig, **kwargs) -> AudioGenerator:
    """Create a Prompt2AudioGenerator with optional subprocess wrapping."""
    from image_server.generators.audio.prompt2audio import Prompt2AudioGenerator
    return create_audio_generator(Prompt2AudioGenerator, config, **kwargs)

"""
Configuration loading and management for Experimance services.
"""

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar

import toml
from pydantic import BaseModel, ConfigDict, Field

# Note: Logging is configured by the CLI or service entry point
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with values that override base
        
    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        # If both values are dictionaries, recursively merge them
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Warn if empty dict is overriding a non-empty dict
            if not value and result[key]:
                logger.warning(f"Empty config section '[{key}]' is clearing defaults. "
                              f"Remove the section from config file to use defaults, "
                              f"or add configuration values to customize.")
            result[key] = deep_merge(result[key], value)
        # Otherwise, override with the new value
        else:
            result[key] = deepcopy(value)
    
    return result


def namespace_to_dict(namespace: argparse.Namespace) -> Dict[str, Any]:
    """Convert an argparse.Namespace to a nested dictionary.
    
    This handles nested keys specified with dots (e.g., 'zmq.port').
    Only includes values that were explicitly set (not defaults).
    
    Args:
        namespace: Argparse namespace object
        
    Returns:
        Nested dictionary representation
    """
    result = {}
    
    # Convert namespace to flat dict
    flat_dict = vars(namespace)
    
    # Check if the namespace has a tracking attribute for explicitly set arguments
    # This is set by a custom argparse action
    explicitly_set = getattr(namespace, '_explicitly_set', set())
    
    for key, value in flat_dict.items():
        # Skip None values
        if value is None:
            continue
            
        # Skip values that weren't explicitly provided on command line
        if key not in explicitly_set:
            continue
            
        # Handle nested keys with dots
        if '.' in key:
            parts = key.split('.')
            current = result
            
            # Navigate to the deepest level
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set the value at the deepest level
            current[parts[-1]] = value
        else:
            result[key] = value
    
    return result


def load_config_with_overrides(
    override_config: Optional[Dict[str, Any]] = None,
    config_file: Optional[Union[str, Path]] = None,
    default_config: Optional[Dict[str, Any]] = None,
    args: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    """Load configuration with flexible overrides and defaults.
    
    The priority order is:
    1. Command line args (from args parameter, highest priority)
    2. Provided override_config dictionary
    3. Config loaded from config_file
    4. Default config (lowest priority)
    
    Args:
        override_config: Dictionary with configuration overrides
        config_file: Path to TOML configuration file
        default_config: Default configuration dictionary
        args: Command line arguments as argparse.Namespace
            
    Returns:
        Merged configuration dictionary
    """
    # Start with an empty config or the default
    config = {} if default_config is None else deepcopy(default_config)
    
    # Add config from file if available
    if config_file is not None:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = toml.load(f)
                    config = deep_merge(config, file_config)
                    logger.info(f"Loaded configuration from {config_path.relative_to(Path.cwd())}")
                    logger.debug(config)
            except Exception as e:
                logger.warning(f"Error loading config from {config_file}: {e}")
        else:
            logger.warning(f"Config file not found: {config_file}")
    
    # Add override config if provided
    if override_config is not None:
        config = deep_merge(config, override_config)
    
    # Add command line args if provided
    if args is not None:
        args_dict = namespace_to_dict(args)
        config = deep_merge(config, args_dict)
    
    return config


# Common config base classes to reduce duplication across services

T = TypeVar('T', bound='BaseConfig')
class BaseConfig(BaseModel):
    """Base configuration model with loading methods.
    
    Extend this class with specific configuration fields for each service.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )

    @classmethod
    def from_overrides(cls: Type[T], 
                     override_config: Optional[Dict[str, Any]] = None,
                     config_file: Optional[Union[str, Path]] = None,
                     default_config: Optional[Dict[str, Any]] = None,
                     args: Optional[argparse.Namespace] = None) -> T:
        """Create a Config instance from multiple sources with flexible overrides.
        
        This combines the power of load_config_with_overrides with Pydantic validation.
        The priority order for configuration is:
        1. Command line args (from args parameter, highest priority)
        2. Provided override_config dictionary
        3. Config loaded from config_file
        4. Default config (lowest priority)
        
        Args:
            override_config: Dictionary with configuration overrides
            config_file: Path to TOML configuration file
            default_config: Default configuration dictionary
            args: Command line arguments as argparse.Namespace
            
        Returns:
            Config instance validated by Pydantic
            
        Raises:
            ValidationError: If the merged configuration doesn't match the model
        """
        # First merge all configuration sources
        merged_config = load_config_with_overrides(
            override_config=override_config,
            config_file=config_file,
            default_config=default_config,
            args=args
        )
        
        # Then validate with Pydantic and return instance
        return cls(**merged_config)

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}:\n{self.model_dump_json(indent=2)}"
# =============================================================================
# SERVICE CONFIGURATION BASE CLASSES
# =============================================================================

class BaseServiceConfig(BaseConfig):
    """Base service configuration with common fields."""
    
    service_name: str = Field(
        description="Name of this service instance"
    )

    

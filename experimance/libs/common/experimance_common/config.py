"""
Configuration loading and management for Experimance services.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import toml
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def load_config(config_path: Union[str, Path], 
                default_config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from a TOML file, with optional fallback to default config.
    
    Args:
        config_path: Path to the primary configuration file
        default_config_path: Path to a default configuration file (optional)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If neither config file could be loaded
    """
    config_path = Path(config_path)
    
    # Try to load primary config
    if config_path.exists():
        try:
            config = toml.load(config_path)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            if default_config_path is None:
                raise ConfigError(f"Failed to load config from {config_path} and no default provided") from e
    elif default_config_path is None:
        raise ConfigError(f"Configuration file {config_path} not found and no default provided")
    
    # Fall back to default config if needed
    default_config_path = Path(default_config_path)
    if default_config_path.exists():
        try:
            config = toml.load(default_config_path)
            logger.info(f"Loaded default configuration from {default_config_path}")
            return config
        except Exception as e:
            raise ConfigError(f"Failed to load default config from {default_config_path}") from e
    else:
        raise ConfigError(f"Neither config file {config_path} nor default {default_config_path} exists")


class Config(BaseModel):
    """Base configuration model with loading methods.
    
    Extend this class with specific configuration fields for each service.
    """
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path], 
                 default_config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Create a Config instance from a TOML file.
        
        Args:
            config_path: Path to the configuration file
            default_config_path: Path to the default configuration file (optional)
            
        Returns:
            Config instance
            
        Raises:
            ConfigError: If configuration couldn't be loaded
        """
        config_dict = load_config(config_path, default_config_path)
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls, env_var: str, 
                default_config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Create a Config instance from a path specified in an environment variable.
        
        Args:
            env_var: Name of the environment variable containing the config path
            default_config_path: Path to the default configuration file (optional)
            
        Returns:
            Config instance
            
        Raises:
            ConfigError: If configuration couldn't be loaded
        """
        config_path = os.environ.get(env_var)
        if not config_path:
            if default_config_path is None:
                raise ConfigError(f"Environment variable {env_var} not set and no default path provided")
            return cls.from_file(default_config_path)
        return cls.from_file(config_path, default_config_path)

"""
Configuration loading and management for Experimance services.
"""

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar

import toml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Note: Logging is configured by the CLI or service entry point
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def get_project_services(project_name: str) -> List[str]:
    """
    Get the list of services for a project.
    
    First checks for SERVICES environment variable, then auto-detects
    by scanning for *.toml files in the project directory.
    
    Args:
        project_name: Name of the project to get services for
        
    Returns:
        List of service names that should be monitored/managed for this project
        
    Examples:
        # Auto-detect services for experimance project
        services = get_project_services("experimance")
        # Returns: ['experimance-core', 'experimance-display', 'image-server', 'experimance-agent', 'experimance-audio']
        
        # With explicit SERVICES environment variable
        os.environ["SERVICES"] = "experimance-core,experimance-display"
        services = get_project_services("experimance")
        # Returns: ['experimance-core', 'experimance-display']
    """
    # Check if services are explicitly defined in environment
    services_env = os.environ.get("SERVICES", "").strip()
    if services_env:
        return [s.strip() for s in services_env.split(",") if s.strip()]
    
    # Auto-detect services by scanning project directory for *.toml files
    try:
        from experimance_common.constants import PROJECT_SPECIFIC_DIR
        
        project_dir = PROJECT_SPECIFIC_DIR / project_name
        if not project_dir.exists():
            logger.warning(f"Project directory not found: {project_dir}")
            return []
        
        service_files = list(project_dir.glob("*.toml"))
        services = []
        
        for service_file in service_files:
            service_name = service_file.stem  # filename without extension
            
            # Map service config names to actual service names
            service_mapping = {
                "core": "experimance-core",
                "display": "experimance-display",
                "audio": "experimance-audio", 
                "agent": "experimance-agent",
                "image_server": "image-server"
            }
            
            actual_service_name = service_mapping.get(service_name, service_name)
            services.append(actual_service_name)
        
        logger.info(f"Auto-detected services for project {project_name}: {services}")
        return services
        
    except Exception as e:
        logger.error(f"Error auto-detecting services for project {project_name}: {e}")
        # Fallback to default services
        return [
            "experimance-core",
            "experimance-display", 
            "image-server",
            "experimance-agent",
            "experimance-audio"
        ]


def resolve_path(
    path_or_string: Union[str, Path], 
    hint: Optional[str] = None
) -> Path:
    """Resolve a path from configuration, with support for relative paths based on service hints.
    
    This function helps resolve file paths from configuration strings, supporting:
    - Absolute paths (returned as-is)
    - Relative paths (resolved relative to appropriate service directory based on hint)
    - Project-specific paths (checks project directory first)
    
    Args:
        path_or_string: Path string or Path object to resolve
        hint: Service type hint to determine base directory. Supported values:
              - "core": Uses CORE_SERVICE_DIR
              - "agent": Uses AGENT_SERVICE_DIR  
              - "display": Uses DISPLAY_SERVICE_DIR
              - "audio": Uses AUDIO_SERVICE_DIR
              - "image_server": Uses IMAGE_SERVER_SERVICE_DIR
              - "project": Uses current project directory
              - "data": Uses DATA_DIR
              - None: Uses PROJECT_ROOT
              
    Returns:
        Resolved Path object
        
    Raises:
        ConfigError: If the resolved path doesn't exist
        
    Examples:
        # Absolute path - returned as-is
        resolve_config_path("/etc/prompts/system.txt")
        
        # Relative path with service hint
        resolve_config_path("prompts/system.txt", hint="core")
        # -> PROJECT_ROOT/services/core/prompts/system.txt
        
        # Project-specific path
        resolve_config_path("system_prompt.txt", hint="project") 
        # -> PROJECT_ROOT/projects/sohkepayin/system_prompt.txt
        
        # Data directory path
        resolve_config_path("locations.json", hint="data")
        # -> PROJECT_ROOT/data/locations.json
    """
    if not is_file(path_or_string):
        raise ConfigError(f"Invalid path: {path_or_string}. Expected a file path.")

    path = Path(path_or_string)

    # If it's already absolute, return as-is (but validate existence)
    if path.is_absolute():
        if not path.exists():
            raise ConfigError(f"Absolute path does not exist: {path}")
        return path
    
    # import here to avoid circular imports
    from experimance_common.constants import (
        PROJECT_ROOT, PROJECT_SPECIFIC_DIR,
        CORE_SERVICE_DIR, AGENT_SERVICE_DIR, DISPLAY_SERVICE_DIR,
        AUDIO_SERVICE_DIR, IMAGE_SERVER_SERVICE_DIR, DATA_DIR
    )

    # Determine base directory based on hint
    base_dirs = []
    hint = hint.lower() if hint else None

    if hint == "core":
        base_dirs = [CORE_SERVICE_DIR]
    elif hint == "agent":
        base_dirs = [AGENT_SERVICE_DIR]
    elif hint == "display":
        base_dirs = [DISPLAY_SERVICE_DIR]
    elif hint == "audio":
        base_dirs = [AUDIO_SERVICE_DIR]
    elif hint == "image_server" or hint == "image":
        base_dirs = [IMAGE_SERVER_SERVICE_DIR]
    elif hint == "project" or hint == "projects":
        project_env = os.getenv("PROJECT_ENV", "experimance")
        base_dirs = [PROJECT_SPECIFIC_DIR / project_env]
    elif hint == "data":
        base_dirs = [DATA_DIR]
    else:
        # Default: try project directory first, then PROJECT_ROOT
        project_env = os.getenv("PROJECT_ENV", "experimance")
        base_dirs = [PROJECT_SPECIFIC_DIR / project_env, PROJECT_ROOT]
    
    # Try each base directory in order
    for base_dir in base_dirs:
        resolved_path = base_dir / path
        if resolved_path.exists():
            return resolved_path
    
    # If no existing path found, return the first candidate (for creating new files)
    if base_dirs:
        candidate_path = base_dirs[0] / path
        logger.warning(f"Path does not exist: {candidate_path}. Returning candidate path.")
        return candidate_path
    
    # Fallback to PROJECT_ROOT
    fallback_path = PROJECT_ROOT / path
    logger.warning(f"No hint provided and path not found. Using PROJECT_ROOT: {fallback_path}")
    return fallback_path


def load_file_content(
    path_or_string: Union[str, Path],
    hint: Optional[str] = None,
    encoding: str = "utf-8"
) -> str:
    """Load text content from a file, with path resolution support.
    
    This is a convenience function that combines resolve_config_path() with file reading.
    
    Args:
        path_or_string: Path to the file
        hint: Service type hint for path resolution (see resolve_config_path)
        encoding: File encoding (default: utf-8)
        
    Returns:
        File content as string
        
    Raises:
        ConfigError: If file cannot be read
        
    Examples:
        # Load system prompt from core service directory
        prompt = load_file_content("prompts/system.txt", hint="core")
        
        # Load project-specific config
        config_text = load_file_content("custom_config.txt", hint="project")
    """
    try:
        resolved_path = resolve_path(path_or_string, hint)
        
        if not resolved_path.exists():
            raise ConfigError(f"File does not exist: {resolved_path}")
            
        with open(resolved_path, 'r', encoding=encoding) as f:
            content = f.read()
            
        logger.debug(f"Loaded file content from: {resolved_path}")
        return content
        
    except Exception as e:
        raise ConfigError(f"Failed to load file content from {path_or_string}: {e}") from e


def is_file(path_or_string: Union[str, Path]) -> bool:
    """Check if a Path or string is a plausible file path.
    Just a simple check designed to allow for configuration to specific text/json or files/dirs containing them.
    """
    if not path_or_string:
        return False
    s = str(path_or_string).lower()
    if s.strip() == "":
        return False
    if s.startswith(".env"):
        return True
    if s.startswith("/") or s.startswith("./") or s.startswith("../") or s.startswith("~"):
        return True
    if s.endswith("/"): # directory
        return True
    if s.endswith(".txt") or s.endswith(".md") or s.endswith(".json") or s.endswith(".toml"): # text file
        return True
    return False

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
        try: 
            config_instance = cls(**merged_config)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e.errors()}")
            raise ConfigError(f"Invalid configuration: {e}") from e
        return config_instance

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

    

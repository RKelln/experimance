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
    Get the list of service types for a project.
    
    First checks for SERVICES environment variable, then auto-detects
    by scanning for *.toml files in the project directory.
    
    Args:
        project_name: Name of the project to get service types for
        
    Returns:
        List of service types that should be monitored/managed for this project.
        Service types are standardized identifiers like "core", "agent", "audio", etc.
        
    Examples:
        # Auto-detect service types for experimance project
        service_types = get_project_services("experimance")
        # Returns: ['core', 'display', 'image_server', 'agent', 'audio']
        
        # With explicit SERVICES environment variable
        os.environ["SERVICES"] = "core,display"
        service_types = get_project_services("experimance")
        # Returns: ['core', 'display']
    """
    # Check if services are explicitly defined in environment
    services_env = os.environ.get("SERVICES", "").strip()
    if services_env:
        return [s.strip() for s in services_env.split(",") if s.strip()]
    
    # Auto-detect service types by scanning project directory for *.toml files
    try:
        from experimance_common.constants import PROJECT_SPECIFIC_DIR, SERVICE_TYPES
        
        project_dir = PROJECT_SPECIFIC_DIR / project_name
        if not project_dir.exists():
            logger.warning(f"Project directory not found: {project_dir}")
            return []
        
        service_files = list(project_dir.glob("*.toml"))
        service_types = []
        
        for service_file in service_files:
            service_type = service_file.stem  # filename without extension
            
            # Validate that this is a known service type
            if service_type in SERVICE_TYPES:
                service_types.append(service_type)
            else:
                logger.warning(f"Unknown service type '{service_type}' found in {service_file}")
        
        logger.info(f"Auto-detected service types for project {project_name}: {service_types}")
        return service_types
        
    except Exception as e:
        logger.error(f"Error auto-detecting service types for project {project_name}: {e}")
        # Fallback to default service types
        return [
            "core",
            "display", 
            "image_server",
            "agent",
            "audio"
        ]


def _normalize_to_service_type(name_or_type: str) -> Optional[str]:
    """
    Normalize a service name or type to the standard service type.
    
    Args:
        name_or_type: Service name or type (e.g., "experimance-core", "core", "agent")
        
    Returns:
        Standardized service type or None if not recognized
        
    Standard service types: agent, core, image_server, display, audio, health
    """
    name_or_type = name_or_type.lower().strip()
    
    # Direct service type matches
    standard_types = {"agent", "core", "image_server", "display", "audio", "health"}
    if name_or_type in standard_types:
        return name_or_type
    
    # Handle various service name formats
    name_mappings = {
        # Core service variations
        "experimance-core": "core",
        "experimance_core": "core",
        "core-service": "core",
        
        # Display service variations  
        "experimance-display": "display",
        "experimance_display": "display",
        "display-service": "display",
        
        # Audio service variations
        "experimance-audio": "audio", 
        "experimance_audio": "audio",
        "audio-service": "audio",
        
        # Agent service variations
        "experimance-agent": "agent",
        "experimance_agent": "agent",
        "agent-service": "agent",
        
        # Image server variations
        "image-server": "image_server",
        "image_server": "image_server",
        "imageserver": "image_server",
        
        # Health service variations
        "experimance-health": "health",
        "experimance_health": "health", 
        "health-service": "health",
    }
    
    return name_mappings.get(name_or_type)


def resolve_path(
    path_or_string: Union[str, Path], 
    hint: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve a path from configuration, with support for relative paths based on service and directory hints.
    
    This function helps resolve file paths from configuration strings, supporting:
    - Absolute paths (returned as-is)
    - Relative paths (resolved relative to appropriate directory based on hint)
    - Smart path resolution when input already contains part of the hint directory structure
    
    Args:
        path_or_string: Path string or Path object to resolve
        hint: Directory hint to determine base directory. Can be:
              - A Path object (or string representing a path): Uses that path as base directory
              - Service type strings:
                - "core": Uses CORE_SERVICE_DIR
                - "agent": Uses AGENT_SERVICE_DIR  
                - "display": Uses DISPLAY_SERVICE_DIR
                - "audio_service": Uses AUDIO_SERVICE_DIR
                - "image_server": Uses IMAGE_SERVER_SERVICE_DIR
              - Media directory constants:
                - "audio_dir", "AUDIO_DIR": Uses AUDIO_DIR_ABS
                - "images_dir", "IMAGES_DIR": Uses IMAGES_DIR_ABS
                - "videos_dir", "VIDEOS_DIR": Uses VIDEOS_DIR_ABS
                - "media_dir", "MEDIA_DIR": Uses MEDIA_DIR_ABS
              - Other directories:
                - "project": Uses current project directory
                - "data": Uses DATA_DIR
                - None: Uses PROJECT_ROOT
              
    Returns:
        Resolved Path object
        
    Raises:
        ConfigError: If the resolved path doesn't exist
        
    Examples:
        # Using AUDIO_DIR_ABS constant as hint
        from experimance_common.constants import AUDIO_DIR_ABS
        resolve_path("environment/bonfire.mp3", hint=AUDIO_DIR_ABS)
        # -> PROJECT_ROOT/media/audio/environment/bonfire.mp3
        
        # Path already includes media structure - smart deduplication
        resolve_path("media/audio/environment/bonfire.mp3", hint=AUDIO_DIR_ABS)
        # -> PROJECT_ROOT/media/audio/environment/bonfire.mp3 (avoids double nesting)
        
        # Using string version of path
        resolve_path("environment/bonfire.mp3", hint="media/audio")
        # -> PROJECT_ROOT/media/audio/environment/bonfire.mp3
        
        # Service-relative path
        resolve_path("prompts/system.txt", hint="core")
        # -> PROJECT_ROOT/services/core/prompts/system.txt
        
        # Project-specific path
        resolve_path("system_prompt.txt", hint="project") 
        # -> PROJECT_ROOT/projects/fire/system_prompt.txt
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
        AUDIO_SERVICE_DIR, IMAGE_SERVER_SERVICE_DIR, DATA_DIR,
        MEDIA_DIR_ABS, AUDIO_DIR_ABS, IMAGES_DIR_ABS, VIDEOS_DIR_ABS
    )

    # Determine base directories based on hint
    base_dirs = []
    
    # Check if hint is a Path object or a path-like string
    if isinstance(hint, (Path, str)) and (isinstance(hint, Path) or ('/' in str(hint) or '\\' in str(hint))):
        # Treat hint as a directory path
        hint_path = Path(hint)
        if hint_path.is_absolute():
            base_dirs = [hint_path]
        else:
            # Relative path hint - resolve relative to PROJECT_ROOT
            base_dirs = [PROJECT_ROOT / hint_path]
    elif hint:
        # Treat hint as a service/directory type string
        hint_normalized = str(hint).lower()

        # Service directories
        if hint_normalized == "core":
            base_dirs = [CORE_SERVICE_DIR]
        elif hint_normalized == "agent":
            base_dirs = [AGENT_SERVICE_DIR]
        elif hint_normalized == "display":
            base_dirs = [DISPLAY_SERVICE_DIR]
        elif hint_normalized == "audio_service":
            base_dirs = [AUDIO_SERVICE_DIR]
        elif hint_normalized == "image_server" or hint_normalized == "image":
            base_dirs = [IMAGE_SERVER_SERVICE_DIR]
        # Media directories
        elif hint_normalized in ("audio_dir", "audio_dir_abs"):
            base_dirs = [AUDIO_DIR_ABS]
        elif hint_normalized in ("images_dir", "images_dir_abs"):
            base_dirs = [IMAGES_DIR_ABS]
        elif hint_normalized in ("videos_dir", "videos_dir_abs"):
            base_dirs = [VIDEOS_DIR_ABS]
        elif hint_normalized in ("media_dir", "media_dir_abs"):
            base_dirs = [MEDIA_DIR_ABS]
        # Other directories
        elif hint_normalized == "project" or hint_normalized == "projects":
            project_env = os.getenv("PROJECT_ENV", "experimance")
            base_dirs = [PROJECT_SPECIFIC_DIR / project_env]
        elif hint_normalized == "data":
            base_dirs = [DATA_DIR]
        else:
            # Default: try project directory first, then PROJECT_ROOT
            project_env = os.getenv("PROJECT_ENV", "experimance")
            base_dirs = [PROJECT_SPECIFIC_DIR / project_env, PROJECT_ROOT]
    else:
        # No hint provided: try project directory first, then PROJECT_ROOT
        project_env = os.getenv("PROJECT_ENV", "experimance")
        base_dirs = [PROJECT_SPECIFIC_DIR / project_env, PROJECT_ROOT]
    
    # Smart path resolution: handle cases where input path already contains
    # part of the hint directory structure
    candidates = []
    for base_dir in base_dirs:
        # Try direct concatenation first
        direct_path = base_dir / path
        candidates.append(direct_path)
        
        # Try smart deduplication for media directories
        if base_dir in (MEDIA_DIR_ABS, AUDIO_DIR_ABS, IMAGES_DIR_ABS, VIDEOS_DIR_ABS):
            relative_base = base_dir.relative_to(PROJECT_ROOT)
            path_parts = Path(path).parts
            
            # Check if path already starts with the relative base structure
            if len(path_parts) >= len(relative_base.parts):
                # See if the beginning of the path matches the relative base
                if path_parts[:len(relative_base.parts)] == relative_base.parts:
                    # Path already includes the base structure, resolve from PROJECT_ROOT
                    deduplicated_path = PROJECT_ROOT / path
                    candidates.append(deduplicated_path)
                    
                # Also try partial matches (e.g., path starts with "audio" when base is "media/audio")
                elif len(relative_base.parts) > 1:
                    for i in range(1, len(relative_base.parts)):
                        partial_base = relative_base.parts[i:]
                        if len(path_parts) >= len(partial_base) and path_parts[:len(partial_base)] == partial_base:
                            # Path starts with partial base, prepend the missing parts
                            missing_parts = relative_base.parts[:i]
                            reconstructed_path = PROJECT_ROOT / Path(*missing_parts) / path
                            candidates.append(reconstructed_path)
    
    # Try each candidate path in order
    for candidate_path in candidates:
        if candidate_path.exists():
            return candidate_path
    
    # If no existing path found, return the best candidate (prefer direct paths)
    if candidates:
        # Prefer the first direct path from base_dirs
        for base_dir in base_dirs:
            direct_candidate = base_dir / path
            if direct_candidate in candidates:
                logger.warning(f"Path does not exist: {direct_candidate}. Returning candidate path.")
                return direct_candidate
        
        # Fallback to first candidate
        candidate_path = candidates[0]
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
    # Text and config files
    if s.endswith(".txt") or s.endswith(".md") or s.endswith(".json") or s.endswith(".toml"): 
        return True
    # Audio files
    if s.endswith(".mp3") or s.endswith(".wav") or s.endswith(".flac") or s.endswith(".ogg") or s.endswith(".m4a"):
        return True
    # Image files
    if s.endswith(".png") or s.endswith(".jpg") or s.endswith(".jpeg") or s.endswith(".gif") or s.endswith(".bmp"):
        return True
    # Video files
    if s.endswith(".mp4") or s.endswith(".avi") or s.endswith(".mkv") or s.endswith(".mov"):
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
                    
                    # Safe relative path calculation
                    try:
                        display_path = config_path.relative_to(Path.cwd())
                    except ValueError:
                        # If not a subpath of cwd, just use the absolute path
                        display_path = config_path
                    
                    logger.info(f"Loaded configuration from {display_path}")
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
    def _extract_env_overrides(cls, env_prefix: Optional[str] = None) -> Dict[str, Any]:
        """Extract configuration overrides from environment variables.
        
        Environment variables are mapped to config fields by matching against the actual
        Pydantic model structure. This ensures proper field mapping.
        
        Args:
            env_prefix: Prefix to filter environment variables (e.g., "EXPERIMANCE")
                       If None, uses the service name from class annotations
                       
        Returns:
            Dictionary of configuration overrides from environment variables
            
        Examples:
            EXPERIMANCE_CORE_NAME="custom-core" → {"experimance_core": {"name": "custom-core"}}
            ZMQ_PUBLISHER_PORT="5556" → {"zmq": {"publisher": {"port": 5556}}}
            CAMERA_FPS="15" → {"camera": {"fps": 15}}
        """
        overrides = {}
        
        # If no prefix specified, try to determine from service_name field default
        if env_prefix is None:
            # Look for service_name field in model fields
            fields = getattr(cls, 'model_fields', {})
            if 'service_name' in fields:
                field = fields['service_name']
                default_value = getattr(field, 'default', None)
                if default_value and isinstance(default_value, str):
                    # Convert service name to env prefix (experimance_core → EXPERIMANCE)
                    env_prefix = default_value.split('_')[0].upper()
        
        # If still no prefix, skip environment parsing
        if not env_prefix:
            return overrides
        
        # Get model fields to understand the config structure
        model_fields = getattr(cls, 'model_fields', {})
        
        # Scan environment variables
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix + '_'):
                continue
                
            # Remove prefix and convert to lowercase
            config_key = env_key[len(env_prefix) + 1:].lower()
            
            # Skip environment variables that don't look like config overrides
            # (e.g., secrets, API keys that just happen to start with the prefix)
            if not cls._looks_like_config_override(config_key, model_fields):
                continue
            
            # Convert value to appropriate type
            converted_value = cls._convert_env_value(env_value)
            
            # Try to map the environment variable to the config structure
            mapped_override = cls._map_env_key_to_config(config_key, converted_value, model_fields)
            if mapped_override:
                # Merge this override into our overrides dict
                overrides = deep_merge(overrides, mapped_override)
                
        return overrides
    
    @classmethod 
    def _looks_like_config_override(cls, env_key: str, model_fields: Dict) -> bool:
        """Check if an environment variable looks like it's intended as a config override.
        
        This helps filter out secrets, API keys, and other environment variables
        that happen to start with our prefix but aren't meant to be config overrides.
        
        Args:
            env_key: Environment variable key (without prefix, lowercase)
            model_fields: Model fields to check against
            
        Returns:
            True if this looks like a config override, False otherwise
        """
        # Import here to avoid circular imports
        from experimance_common.cli import CLI_ONLY_ARGS
        
        # Skip CLI-only arguments that aren't part of the config model
        if env_key in CLI_ONLY_ARGS:
            return False
            
        # Check if it directly matches a top-level field
        if env_key in model_fields:
            return True
            
        # Check if it matches a nested field pattern (section_key)
        key_parts = env_key.split('_')
        if len(key_parts) >= 2:
            # Check if first part matches a section name in model_fields
            section_name = key_parts[0]
            for field_name, field_info in model_fields.items():
                field_type = getattr(field_info, 'annotation', None)
                
                # Check direct section match
                if field_name == section_name:
                    return True
                    
                # Check if field_name ends with this section (e.g., "experimance_core" ends with "core") 
                if field_name.endswith('_' + section_name):
                    return True
                    
                # Check if this is a nested config object
                if field_type and hasattr(field_type, 'model_fields'):
                    # Check if the remaining parts could be nested fields
                    nested_key = '_'.join(key_parts[1:])
                    nested_fields = getattr(field_type, 'model_fields', {})
                    if nested_key in nested_fields:
                        return True
        
        return False

    @classmethod
    def _map_env_key_to_config(cls, env_key: str, value: Any, model_fields: Dict) -> Optional[Dict[str, Any]]:
        """Map an environment variable key to the config structure.
        
        This method attempts to intelligently map environment variable names
        to the actual Pydantic model structure.
        """
        # Direct field match (e.g., "service_name" → service_name field)
        if env_key in model_fields:
            return {env_key: value}
        
        # Try section_field pattern (e.g., "core_name" → experimance_core.name)
        for field_name, field_info in model_fields.items():
            # Get the field type
            field_type = getattr(field_info, 'annotation', None)
            
            # Check if this is a nested config object (BaseModel subclass)
            if field_type and hasattr(field_type, 'model_fields'):
                nested_fields = getattr(field_type, 'model_fields', {})
                
                # Check if env_key matches pattern: {section}_{nested_field}
                if env_key.startswith(field_name + '_'):
                    nested_key = env_key[len(field_name) + 1:]
                    if nested_key in nested_fields:
                        return {field_name: {nested_key: value}}
                
                # Also try short section names (e.g., "core_name" for "experimance_core")
                # Extract the last part of the field name
                short_section = field_name.split('_')[-1]
                if env_key.startswith(short_section + '_'):
                    nested_key = env_key[len(short_section) + 1:]
                    if nested_key in nested_fields:
                        return {field_name: {nested_key: value}}
        
        # Fallback: treat as simple nested structure based on underscores
        key_parts = env_key.split('_')
        if len(key_parts) == 1:
            # Single part - might be a top-level field we don't recognize
            # Only warn if this looks like it could be a config field
            if env_key.isalpha() and len(env_key) > 1:
                logger.debug(f"Unrecognized environment variable for config: {env_key}")
            return None
        
        # Multi-part: create nested structure
        result = {}
        current = result
        for part in key_parts[:-1]:
            current[part] = {}
            current = current[part]
        current[key_parts[-1]] = value
        
        return result
    
    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert environment variable string to appropriate Python type."""
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle numeric values
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            # Then float
            return float(value)
        except ValueError:
            pass
        
        # Handle JSON-like values (lists, dicts)
        if value.startswith(('[', '{')):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Default to string
        return value

    @classmethod
    def from_overrides(cls: Type[T], 
                     override_config: Optional[Dict[str, Any]] = None,
                     config_file: Optional[Union[str, Path]] = None,
                     default_config: Optional[Dict[str, Any]] = None,
                     args: Optional[argparse.Namespace] = None,
                     env_prefix: Optional[str] = None) -> T:
        """Create a Config instance from multiple sources with flexible overrides.
        
        This combines the power of load_config_with_overrides with Pydantic validation.
        The priority order for configuration is:
        1. Command line args (from args parameter, highest priority)
        2. Provided override_config dictionary
        3. Environment variables (with env_prefix)
        4. Config loaded from config_file
        5. Default config (lowest priority)
        
        Args:
            override_config: Dictionary with configuration overrides
            config_file: Path to TOML configuration file
            default_config: Default configuration dictionary
            args: Command line arguments as argparse.Namespace
            env_prefix: Prefix for environment variables (e.g., "EXPERIMANCE")
                       If None, auto-detects from service_name field default
            
        Returns:
            Config instance validated by Pydantic
            
        Raises:
            ValidationError: If the merged configuration doesn't match the model
            
        Examples:
            # Basic usage
            config = MyServiceConfig.from_overrides(config_file="config.toml")
            
            # With environment variables
            # EXPERIMANCE_CORE_NAME="custom" CAMERA_FPS="15"
            config = MyServiceConfig.from_overrides(env_prefix="EXPERIMANCE")
            
            # Full override chain
            config = MyServiceConfig.from_overrides(
                config_file="config.toml",
                override_config={"debug": True},
                env_prefix="EXPERIMANCE",
                args=parsed_args
            )
        """
        # Extract environment variable overrides
        env_overrides = cls._extract_env_overrides(env_prefix)
        
        # Merge environment overrides with provided overrides
        # Command line args and explicit overrides still take precedence
        if env_overrides:
            if override_config:
                # Merge env overrides as base, with explicit overrides on top
                merged_overrides = deep_merge(env_overrides, override_config)
            else:
                merged_overrides = env_overrides
        else:
            merged_overrides = override_config
        
        # First merge all configuration sources
        merged_config = load_config_with_overrides(
            override_config=merged_overrides,
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

    @classmethod
    def from_env(cls: Type[T], 
                 config_file: Optional[Union[str, Path]] = None,
                 env_prefix: Optional[str] = None) -> T:
        """Create a Config instance with automatic environment variable support.
        
        This is a convenience method that automatically includes environment variables
        in the configuration loading process.
        
        Args:
            config_file: Path to TOML configuration file
            env_prefix: Prefix for environment variables (auto-detected if None)
            
        Returns:
            Config instance with environment variable overrides applied
            
        Examples:
            # Load config with automatic environment variable detection
            config = MyServiceConfig.from_env("config.toml")
            
            # With explicit environment prefix
            config = MyServiceConfig.from_env("config.toml", env_prefix="EXPERIMANCE")
        """
        return cls.from_overrides(
            config_file=config_file,
            env_prefix=env_prefix
        )

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

    

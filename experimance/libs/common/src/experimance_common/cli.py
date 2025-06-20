"""
Reusable command line interface utilities for Experimance services.

This module provides a standard CLI interface that all Experimance services can use
for consistent command line argument parsing, logging setup, and service execution.
"""
import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Optional, Callable, Awaitable, Any, Dict, Type, get_origin, get_args

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def extract_cli_args_from_config(config_class: Type[BaseModel], prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """Extract CLI arguments from a Pydantic config class.
    
    This function analyzes the fields of a Pydantic model and generates
    appropriate argparse arguments for them. It handles basic types like
    bool, int, float, str and recursively processes nested BaseModel fields.
    
    Args:
        config_class: Pydantic BaseModel class to extract arguments from
        prefix: Prefix for nested field names (e.g., "camera" for camera.fps)
        
    Returns:
        Dictionary mapping argument names to argparse argument configurations
    """
    cli_args = {}
    
    for field_name, field_info in config_class.model_fields.items():
        # Get the field type, handling Optional types
        field_type = field_info.annotation
        origin = get_origin(field_type)
        if origin is not None:
            # Handle Optional[Type] -> Type
            if origin is type(Optional[str]) or origin is type(None):  # Union type
                args = get_args(field_type)
                if len(args) == 2 and type(None) in args:
                    field_type = next(arg for arg in args if arg is not type(None))
        
        # Handle nested BaseModel fields recursively
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Recursively extract from nested config with prefix
            nested_prefix = f"{prefix}.{field_name}" if prefix else field_name
            nested_args = extract_cli_args_from_config(field_type, nested_prefix)
            cli_args.update(nested_args)
            continue
            
        # Only handle basic types for CLI
        if field_type not in (bool, int, float, str):
            continue
            
        # Build the argument name with prefix
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name
        # Use hyphens for CLI display, dots are only used internally for argparse dest
        arg_name = f"--{full_field_name.replace('_', '-').replace('.', '-')}"
        arg_config = {}
        
        # Set the type
        if field_type == bool:
            # For boolean fields, use tracked actions to know when they were explicitly set
            if field_info.default is True:
                arg_config['action'] = TrackedStoreFalseAction
                arg_name = f"--no-{full_field_name.replace('_', '-').replace('.', '-')}"
            else:
                arg_config['action'] = TrackedStoreTrueAction
        else:
            arg_config['type'] = field_type
            arg_config['action'] = TrackedAction
            # Don't set metavar - it just adds noise to help text
            
        # Add help text from field description
        help_text = ""
        if hasattr(field_info, 'description') and field_info.description:
            help_text = field_info.description
            
        # Add nested context to help
        if prefix:
            section_name = prefix.replace('_', ' ').replace('.', ' ').title()
            if help_text:
                help_text = f"[{section_name}] {help_text}"
            else:
                help_text = f"[{section_name}] {field_name}"
                
        # Add default value info to help
        if field_info.default is not None and field_info.default != ...:
            if help_text:
                help_text += f" (default: {field_info.default})"
            else:
                help_text = f"Default: {field_info.default}"
                
        if help_text:
            arg_config['help'] = help_text
        
        # CRITICAL: Use dots in dest so namespace_to_dict can create nested structure
        # But clean up the metavar to avoid showing confusing dotted names in help
        arg_config['dest'] = full_field_name  # Keep dots for proper nesting
        
        # Set a clean metavar based on the field type to avoid showing the dotted dest
        if field_type == bool:
            # Boolean fields don't need metavar since they're flags
            pass
        elif field_type == int:
            arg_config['metavar'] = 'N'
        elif field_type == float:
            arg_config['metavar'] = 'VALUE'
        elif field_type == str:
            arg_config['metavar'] = 'TEXT'
        else:
            arg_config['metavar'] = 'VALUE'
        
        cli_args[arg_name] = arg_config
    
    return cli_args


def setup_logging(log_level: str, service_name: str) -> None:
    """Configure logging with the specified level for a service.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service for log formatting
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format=f'%(asctime)s - {service_name} - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_service_parser(
    service_name: str,
    description: str,
    default_config_path: Optional[str] = None,
    extra_args: Optional[Dict[str, Dict[str, Any]]] = None,
    config_class: Optional[Type[BaseModel]] = None
) -> argparse.ArgumentParser:
    """Create a standard argument parser for Experimance services.
    
    Args:
        service_name: Name of the service (e.g., "Core", "Display")
        description: Description of the service
        default_config_path: Default path to config file (optional)
        extra_args: Additional arguments to add to parser
        config_class: Pydantic config class to auto-generate arguments from
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=f'Experimance {service_name} Service - {description}'
    )
    
    # Standard arguments for all services - use tracked actions to track when explicitly set
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        action=TrackedAction,
        help='Set the logging level (default: INFO)'
    )
    
    if default_config_path:
        parser.add_argument(
            '--config', '-c',
            default=default_config_path,
            action=TrackedAction,
            help=f'Path to configuration file (default: {default_config_path})'
        )
    
    # Auto-generate arguments from config class
    if config_class:
        config_args = extract_cli_args_from_config(config_class)
        for arg_name, arg_config in config_args.items():
            parser.add_argument(arg_name, **arg_config)
    
    # Add any extra arguments specific to the service
    if extra_args:
        for arg_name, arg_config in extra_args.items():
            parser.add_argument(arg_name, **arg_config)
    
    return parser


async def run_service_cli(
    service_name: str,
    description: str,
    service_runner: Callable[..., Awaitable[None]],
    default_config_path: Optional[str] = None,
    extra_args: Optional[Dict[str, Dict[str, Any]]] = None,
    config_class: Optional[Type[BaseModel]] = None
) -> None:
    """Run a service with standard CLI argument parsing and error handling.
    
    Args:
        service_name: Name of the service (e.g., "Core", "Display")
        description: Description of the service
        service_runner: Async function that runs the service
        default_config_path: Default path to config file (optional)
        extra_args: Additional arguments to add to parser
        config_class: Pydantic config class to auto-generate CLI args from
    """
    parser = create_service_parser(
        service_name=service_name,
        description=description,
        default_config_path=default_config_path,
        extra_args=extra_args,
        config_class=config_class
    )
    
    args = parser.parse_args()
    
    # Setup logging with the specified level
    setup_logging(args.log_level, service_name.upper())
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Experimance {service_name} Service with log level: {args.log_level}")
    
    # remove cli only options from args (like log-level and thos in extra-args
    # so they aren't validated by the pydantic config class    
    config_path = getattr(args, 'config', default_config_path)
    # remove config path from args if it exists
    if config_path:
        del args.config
    if args.log_level:
        del args.log_level
    for arg in extra_args or {}:
        if hasattr(args, arg.lstrip('--')):
            delattr(args, arg.lstrip('--')) 
    
    try:
        # Call the service runner with the parsed arguments
        if config_class:
            # Pass args for CLI overrides and config path
            await service_runner(args=args, config_path=config_path)
        elif default_config_path:
            await service_runner(config_path=config_path)
        else:
            await service_runner()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error in {service_name} service: {e}", exc_info=True)
        sys.exit(1)


def create_simple_main(
    service_name: str,
    description: str,
    service_runner: Callable[..., Awaitable[None]],
    default_config_path: Optional[str] = None,
    extra_args: Optional[Dict[str, Dict[str, Any]]] = None,
    config_class: Optional[Type[BaseModel]] = None
) -> Callable[[], None]:
    """Create a main() function for a service that can be used in __main__.py.
    
    Args:
        service_name: Name of the service (e.g., "Core", "Display")
        description: Description of the service
        service_runner: Async function that runs the service
        default_config_path: Default path to config file (optional)
        extra_args: Additional arguments to add to parser
        config_class: Pydantic config class to auto-generate CLI args from
        
    Returns:
        A main() function that can be called from __main__.py
    """
    def main() -> None:
        asyncio.run(run_service_cli(
            service_name=service_name,
            description=description,
            service_runner=service_runner,
            default_config_path=default_config_path,
            extra_args=extra_args,
            config_class=config_class
        ))
    
    return main


import argparse
import logging
import traceback
from typing import Dict, Any, Optional, Callable, Awaitable, Type, List

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class TrackedAction(argparse.Action):
    """Custom argparse action that tracks which arguments were explicitly provided."""
    
    def __call__(self, parser, namespace, values, option_string=None):
        # Ensure the namespace has our tracking set
        if not hasattr(namespace, '_explicitly_set'):
            namespace._explicitly_set = set()
        
        # Mark this argument as explicitly set
        namespace._explicitly_set.add(self.dest)
        
        # Set the value as normal
        setattr(namespace, self.dest, values)


class TrackedStoreTrueAction(argparse._StoreTrueAction):
    """Custom store_true action that tracks when the flag was explicitly provided."""
    
    def __call__(self, parser, namespace, values, option_string=None):
        # Ensure the namespace has our tracking set
        if not hasattr(namespace, '_explicitly_set'):
            namespace._explicitly_set = set()
        
        # Mark this argument as explicitly set
        namespace._explicitly_set.add(self.dest)
        
        # Call parent implementation
        super().__call__(parser, namespace, values, option_string)


class TrackedStoreFalseAction(argparse._StoreFalseAction):
    """Custom store_false action that tracks when the flag was explicitly provided."""
    
    def __call__(self, parser, namespace, values, option_string=None):
        # Ensure the namespace has our tracking set
        if not hasattr(namespace, '_explicitly_set'):
            namespace._explicitly_set = set()
        
        # Mark this argument as explicitly set
        namespace._explicitly_set.add(self.dest)
        
        # Call parent implementation
        super().__call__(parser, namespace, values, option_string)

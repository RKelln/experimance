"""
Reusable command line interface utilities for Experimance services.

This module provides a standard CLI interface that all Experimance services can use
for consistent command line argument parsing, logging setup, and service execution.
"""
import argparse
import asyncio
import logging
import sys
from typing import Optional, Callable, Awaitable, Any, Dict


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
    extra_args: Optional[Dict[str, Dict[str, Any]]] = None
) -> argparse.ArgumentParser:
    """Create a standard argument parser for Experimance services.
    
    Args:
        service_name: Name of the service (e.g., "Core", "Display")
        description: Description of the service
        default_config_path: Default path to config file (optional)
        extra_args: Additional arguments to add to parser
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=f'Experimance {service_name} Service - {description}'
    )
    
    # Standard arguments for all services
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    if default_config_path:
        parser.add_argument(
            '--config', '-c',
            default=default_config_path,
            help=f'Path to configuration file (default: {default_config_path})'
        )
    
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
    extra_args: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """Run a service with standard CLI argument parsing and error handling.
    
    Args:
        service_name: Name of the service (e.g., "Core", "Display")
        description: Description of the service
        service_runner: Async function that runs the service
        default_config_path: Default path to config file (optional)
        extra_args: Additional arguments to add to parser
    """
    parser = create_service_parser(
        service_name=service_name,
        description=description,
        default_config_path=default_config_path,
        extra_args=extra_args
    )
    
    args = parser.parse_args()
    
    # Setup logging with the specified level
    setup_logging(args.log_level, service_name.upper())
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Experimance {service_name} Service with log level: {args.log_level}")
    
    try:
        # Call the service runner with the parsed arguments
        if default_config_path:
            await service_runner(config_path=args.config)
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
    extra_args: Optional[Dict[str, Dict[str, Any]]] = None
) -> Callable[[], None]:
    """Create a main() function for a service that can be used in __main__.py.
    
    Args:
        service_name: Name of the service (e.g., "Core", "Display")
        description: Description of the service
        service_runner: Async function that runs the service
        default_config_path: Default path to config file (optional)
        extra_args: Additional arguments to add to parser
        
    Returns:
        A main() function that can be called from __main__.py
    """
    def main() -> None:
        asyncio.run(run_service_cli(
            service_name=service_name,
            description=description,
            service_runner=service_runner,
            default_config_path=default_config_path,
            extra_args=extra_args
        ))
    
    return main

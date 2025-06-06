#!/usr/bin/env python3
"""
Main entry point for the Experimance Image Server Service.

This script initializes and runs the image server service, which handles
image generation requests and publishes results via ZeroMQ.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from image_server import ImageServerService


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("image_server.log")
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Experimance Image Server Service"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config.toml",
        dest="config_file",
        help="Path to configuration file (default: config.toml)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--generator",
        choices=["mock", "fal", "openai", "local"],
        default="mock",
        help="Image generation strategy (default: mock)"
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Experimance Image Server Service")
    
    config = {"generator_type": args.generator}

    # Create and configure the service
    service = ImageServerService(
        config=config,
        config_file=args.config_file,
    )
    
    await service.start()
    logger.info("Image Server Service started successfully")
    await service.run() 
    logger.info("Image Server Service is complete")

    # # Set up signal handlers for graceful shutdown
    # shutdown_event = asyncio.Event()
    
    # def signal_handler(signum, frame):
    #     logger.info(f"Received signal {signum}, initiating shutdown...")
    #     shutdown_event.set()
    
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    
    # try:
    #     # Start the service
    #     await service.start()
    #     logger.info("Image Server Service started successfully")
        
    #     # Wait for shutdown signal
    #     await shutdown_event.wait()
        
    # except Exception as e:
    #     logger.error(f"Error running service: {e}")
    #     sys.exit(1)
    # finally:
    #     # Clean shutdown
    #     logger.info("Shutting down Image Server Service...")
    #     await service.stop()
    #     logger.info("Image Server Service stopped")


if __name__ == "__main__":
    asyncio.run(main())


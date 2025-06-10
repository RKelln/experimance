#!/usr/bin/env python3
"""
ZMQ Address Validation Tool for Image Server Service.

This script validates the ZMQ addresses used in the Image Server Service
and its clients, ensuring they match and are correctly configured.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from experimance_common.constants import DEFAULT_PORTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("zmq_address_validator")


def validate_addresses():
    """Validate ZMQ addresses for the Image Server Service."""
    logger.info("===== Validating ZMQ Addresses =====")
    
    # Expected addresses based on DEFAULT_PORTS
    expected = {
        # Where the image server publishes responses
        "image_server_pub": f"tcp://*:{DEFAULT_PORTS['image_server_pub']}",  # Server binds to this
        "image_server_pub_client": f"tcp://localhost:{DEFAULT_PORTS['image_server_pub']}",  # Clients connect to this
        
        # Where the image server listens for requests
        "image_request_pub": f"tcp://*:{DEFAULT_PORTS['image_request_pub']}",  # Server binds to this
        "image_request_pub_client": f"tcp://localhost:{DEFAULT_PORTS['image_request_pub']}",  # Clients connect to this
    }
    
    logger.info("Expected ZMQ Addresses:")
    logger.info(f"  Image Server publishes on: {expected['image_server_pub']} (server binding)")
    logger.info(f"  Clients connect to: {expected['image_server_pub_client']} (client connection)")
    logger.info(f"  Image Server listens on: {expected['image_request_pub']} (server binding)")
    logger.info(f"  Clients publish to: {expected['image_request_pub_client']} (client connection)")
    
    logger.info("\nPort Numbers:")
    logger.info(f"  image_server_pub: {DEFAULT_PORTS['image_server_pub']}")
    logger.info(f"  image_request_pub: {DEFAULT_PORTS['image_request_pub']}")
    
    logger.info("\nZMQ Communication Flow:")
    logger.info("  1. Client publishes RenderRequest to tcp://localhost:{DEFAULT_PORTS['image_request_pub']}")
    logger.info("  2. Server receives request on tcp://*:{DEFAULT_PORTS['image_request_pub']}")
    logger.info("  3. Server processes request and generates image")
    logger.info("  4. Server publishes ImageReady to tcp://*:{DEFAULT_PORTS['image_server_pub']}")
    logger.info("  5. Client receives ImageReady from tcp://localhost:{DEFAULT_PORTS['image_server_pub']}")
    
    # Check CLI address usage
    cli_path = Path("src/image_server/cli.py")
    if cli_path.exists():
        logger.info("\nValidating CLI Address Usage:")
        with open(cli_path, 'r') as file:
            content = file.read()
            
            events_pub_address = f"tcp://localhost:{DEFAULT_PORTS['image_request_pub']}"
            images_sub_address = f"tcp://localhost:{DEFAULT_PORTS['image_server_pub']}"
            
            if events_pub_address in content and images_sub_address in content:
                logger.info("  ✅ CLI is using the correct addresses")
            else:
                logger.error("  ❌ CLI addresses might be misconfigured!")
                if events_pub_address not in content:
                    logger.error(f"    - Missing expected events publisher address: {events_pub_address}")
                if images_sub_address not in content:
                    logger.error(f"    - Missing expected images subscriber address: {images_sub_address}")
    else:
        logger.warning("  CLI file not found in expected location.")
    
    logger.info("\nValidation complete.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Validate ZMQ addresses for Image Server Service")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        validate_addresses()
        return 0
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

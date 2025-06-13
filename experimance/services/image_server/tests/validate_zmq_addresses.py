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
    
    # Expected addresses based on unified events channel
    expected = {
        # All services use the unified events channel for PUB/SUB
        "events_server": f"tcp://*:{DEFAULT_PORTS['events']}",  # Server binds to this
        "events_client": f"tcp://localhost:{DEFAULT_PORTS['events']}",  # Clients connect to this
    }
    
    logger.info("Expected ZMQ Addresses (Unified Events Channel):")
    logger.info(f"  Image Server publishes on: {expected['events_server']} (server binding)")
    logger.info(f"  Clients connect to: {expected['events_client']} (client connection)")
    logger.info(f"  All services subscribe to: {expected['events_client']} (client connection)")
    
    logger.info("\nPort Numbers:")
    logger.info(f"  events: {DEFAULT_PORTS['events']}")
    
    logger.info("\nZMQ Communication Flow:")
    logger.info("  1. Client publishes RenderRequest to tcp://localhost:{DEFAULT_PORTS['events']}")
    logger.info("  2. Server receives request on tcp://*:{DEFAULT_PORTS['events']}")
    logger.info("  3. Server processes request and generates image")
    logger.info("  4. Server publishes ImageReady to tcp://*:{DEFAULT_PORTS['events']}")
    logger.info("  5. Client receives ImageReady from tcp://localhost:{DEFAULT_PORTS['events']}")
    logger.info("  6. All services use message type filtering to handle relevant messages")
    
    # Check CLI address usage
    cli_path = Path("src/image_server/cli.py")
    if cli_path.exists():
        logger.info("\nValidating CLI Address Usage:")
        with open(cli_path, 'r') as file:
            content = file.read()
            
            events_address = f"tcp://localhost:{DEFAULT_PORTS['events']}"
            
            if events_address in content:
                logger.info("  ✅ CLI is using the unified events channel")
            else:
                logger.error("  ❌ CLI might not be using the unified events channel!")
                logger.error(f"    - Expected events address: {events_address}")
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

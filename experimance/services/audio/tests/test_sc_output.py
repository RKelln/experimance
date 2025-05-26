#!/usr/bin/env python3
"""
A simple script to start SuperCollider and display its output.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from experimance_audio.osc_bridge import OscBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test SuperCollider Output")
    parser.add_argument("--sc-script", type=str, help="Path to SuperCollider script")
    parser.add_argument("--sclang-path", type=str, default="sclang", help="Path to SuperCollider language interpreter")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Find the script if not specified
    sc_script = args.sc_script
    if not sc_script:
        # Try to find the default script in the sc_scripts directory
        current_dir = Path(__file__).parent.parent  # Go up from src/experimance_audio to services/audio
        sc_scripts_dir = current_dir / "sc_scripts"
        default_script = sc_scripts_dir / "experimance_audio.scd"
        
        if default_script.exists():
            sc_script = str(default_script)
            logger.info(f"Using default script: {sc_script}")
        else:
            logger.error(f"No SuperCollider script path specified and default script not found at {default_script}")
            sys.exit(1)
    
    # Create OSC bridge and start SuperCollider
    osc = OscBridge()
    # Log both to file and console for the test script
    log_file_path = osc.start_supercollider(
        sc_script,
        args.sclang_path,
        log_to_file=True, 
        log_to_console=True
    )
    
    if not log_file_path:
        logger.error("Failed to start SuperCollider")
        sys.exit(1)
        
    logger.info(f"SuperCollider logs are also being saved to: {log_file_path}")
    
    logger.info("SuperCollider started. Press Ctrl+C to stop.")
    
    try:
        # Keep the script running until Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping SuperCollider...")
        osc.stop_supercollider()
        logger.info("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for OSC communication with SuperCollider.

This script provides a simple interface to test OSC communication with 
the test_osc.scd SuperCollider script.
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

from experimance_common.constants import DEFAULT_PORTS
DEFAULT_OSC_SEND_PORT = DEFAULT_PORTS.get("audio_osc_send_port", 5567)
DEFAULT_OSC_RECV_PORT = DEFAULT_PORTS.get("audio_osc_recv_port", 5568)

def main():
    parser = argparse.ArgumentParser(description="Test OSC communication with SuperCollider")
    parser.add_argument("--sc-script", type=str, help="Path to SuperCollider script")
    parser.add_argument("--sclang-path", type=str, default="sclang", help="Path to SuperCollider language interpreter")
    parser.add_argument("--osc-port", type=int, default=DEFAULT_OSC_RECV_PORT, help="SuperCollider OSC port (default: %d)" % DEFAULT_OSC_RECV_PORT)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Find the script if not specified
    sc_script = args.sc_script
    if not sc_script:
        # Try to find the test OSC script in the sc_scripts directory
        current_dir = Path(__file__).parent.parent  # Go up from src/experimance_audio to services/audio
        sc_scripts_dir = current_dir / "sc_scripts"
        test_script = sc_scripts_dir / "test_osc.scd"
        
        if test_script.exists():
            sc_script = str(test_script)
            logger.info(f"Using test script: {sc_script}")
        else:
            logger.error(f"No SuperCollider test script found at {test_script}")
            sys.exit(1)
    
    # Create OSC bridge and start SuperCollider
    # Note: We're setting a custom port to match SuperCollider's default port
    osc = OscBridge(port=args.osc_port)
    
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
    logger.info(f"Sending OSC messages to port: {args.osc_port}")
    
    # Give SuperCollider time to boot
    time.sleep(3)
    
    try:
        while True:
            print("\n----- OSC Test Menu -----")
            print("1. Send spacetime message")
            print("2. Include tag")
            print("3. Exclude tag")
            print("4. Start listening")
            print("5. Stop listening")
            print("6. Start speaking")
            print("7. Stop speaking")
            print("8. Start transition")
            print("9. Stop transition")
            print("10. Reload configs")
            print("0. Quit")
            
            choice = input("Enter your choice: ")
            
            if choice == "1":
                biome = input("Enter biome name (default: forest): ") or "forest"
                era = input("Enter era name (default: wilderness): ") or "wilderness"
                osc.send_spacetime(biome, era)
                logger.info(f"Sent spacetime: biome={biome}, era={era}")
                
            elif choice == "2":
                tag = input("Enter tag name (default: birds): ") or "birds"
                osc.include_tag(tag)
                logger.info(f"Sent include tag: {tag}")
                
            elif choice == "3":
                tag = input("Enter tag name (default: birds): ") or "birds"
                osc.exclude_tag(tag)
                logger.info(f"Sent exclude tag: {tag}")
                
            elif choice == "4":
                osc.listening(True)
                logger.info("Sent listening: start")
                
            elif choice == "5":
                osc.listening(False)
                logger.info("Sent listening: stop")
                
            elif choice == "6":
                osc.speaking(True)
                logger.info("Sent speaking: start")
                
            elif choice == "7":
                osc.speaking(False)
                logger.info("Sent speaking: stop")
                
            elif choice == "8":
                osc.transition(True)
                logger.info("Sent transition: start")
                
            elif choice == "9":
                osc.transition(False)
                logger.info("Sent transition: stop")
                
            elif choice == "10":
                osc.reload_configs()
                logger.info("Sent reload configs")
                
            elif choice == "0":
                logger.info("Quitting...")
                osc.stop_supercollider()
                break
                
            else:
                print("Invalid choice, try again")
                
            # Small delay between commands
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Stopping SuperCollider...")
        osc.stop_supercollider()
        logger.info("Done.")

if __name__ == "__main__":
    main()

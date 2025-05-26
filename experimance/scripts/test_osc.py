#!/usr/bin/env python3
"""
Simple script to test SuperCollider OSC communication.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from pythonosc import udp_client

# Import constants from experimance_common
try:
    from experimance_common.constants import DEFAULT_PORTS
    DEFAULT_OSC_SEND_PORT = DEFAULT_PORTS.get("audio_osc_send_port", 5567)
    DEFAULT_OSC_RECV_PORT = DEFAULT_PORTS.get("audio_osc_recv_port", 5568)
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import experimance_common.constants, using default ports")
    DEFAULT_OSC_SEND_PORT = 5567
    DEFAULT_OSC_RECV_PORT = 5568

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test OSC communication with SuperCollider")
    parser.add_argument("--sc-port", type=int, default=DEFAULT_OSC_RECV_PORT, 
                        help=f"SuperCollider OSC receive port (default: {DEFAULT_OSC_RECV_PORT})")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Start SuperCollider with the test script
    base_dir = Path(__file__).parent.parent  # Go up from scripts to the project root
    sc_script_path = base_dir / "services/audio/sc_scripts/test_osc.scd"
    
    if not sc_script_path.exists():
        logger.error(f"Could not find SuperCollider test script at: {sc_script_path}")
        sys.exit(1)
    
    logger.info(f"Starting SuperCollider with script: {sc_script_path}")
    
    # Start SuperCollider in a separate process
    sc_process = subprocess.Popen(
        ["sclang", str(sc_script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Function to print SuperCollider output
    def print_sc_output():
        while sc_process.poll() is None:
            if sc_process.stdout:
                line = sc_process.stdout.readline()
                if line:
                    print(f"SC: {line.rstrip()}")
            if sc_process.stderr:
                line = sc_process.stderr.readline()
                if line:
                    print(f"SC ERROR: {line.rstrip()}")
    
    # Start reading SuperCollider output in a separate thread
    import threading
    output_thread = threading.Thread(target=print_sc_output, daemon=True)
    output_thread.start()
    
    # Give SuperCollider time to boot (adjust this if needed)
    logger.info("Waiting for SuperCollider server to start...")
    time.sleep(3)
    
    # Create OSC client
    logger.info(f"Connecting to SuperCollider on port {args.sc_port}...")
    client = udp_client.SimpleUDPClient("localhost", args.sc_port)
    
    # Main menu loop
    try:
        while True:
            print("\n----- SuperCollider OSC Test Menu -----")
            print("1. Send /spacetime message")
            print("2. Send /include message")
            print("3. Send /exclude message")
            print("4. Send /listening start")
            print("5. Send /listening stop")
            print("6. Send /speaking start")
            print("7. Send /speaking stop")
            print("8. Send /transition start")
            print("9. Send /transition stop")
            print("10. Send /reload message")
            print("0. Quit")
            
            choice = input("Enter your choice: ")
            
            if choice == "1":
                biome = input("Enter biome (default: forest): ") or "forest"
                era = input("Enter era (default: wilderness): ") or "wilderness"
                client.send_message("/spacetime", [biome, era])
                logger.info(f"Sent: /spacetime {biome} {era}")
                
            elif choice == "2":
                tag = input("Enter tag to include (default: birds): ") or "birds"
                client.send_message("/include", [tag])
                logger.info(f"Sent: /include {tag}")
                
            elif choice == "3":
                tag = input("Enter tag to exclude (default: birds): ") or "birds"
                client.send_message("/exclude", [tag])
                logger.info(f"Sent: /exclude {tag}")
                
            elif choice == "4":
                client.send_message("/listening", ["start"])
                logger.info("Sent: /listening start")
                
            elif choice == "5":
                client.send_message("/listening", ["stop"])
                logger.info("Sent: /listening stop")
                
            elif choice == "6":
                client.send_message("/speaking", ["start"])
                logger.info("Sent: /speaking start")
                
            elif choice == "7":
                client.send_message("/speaking", ["stop"])
                logger.info("Sent: /speaking stop")
                
            elif choice == "8":
                client.send_message("/transition", ["start"])
                logger.info("Sent: /transition start")
                
            elif choice == "9":
                client.send_message("/transition", ["stop"])
                logger.info("Sent: /transition stop")
                
            elif choice == "10":
                client.send_message("/reload", [])
                logger.info("Sent: /reload")
                
            elif choice == "0":
                logger.info("Quitting...")
                client.send_message("/quit", [])
                logger.info("Sent quit command to SuperCollider")
                time.sleep(1)  # Give SuperCollider time to quit gracefully
                break
                
            else:
                print("Invalid choice, please try again")
            
            # Add a small delay between commands
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    finally:
        # Clean up
        if sc_process.poll() is None:
            logger.info("Terminating SuperCollider process...")
            sc_process.terminate()
            try:
                sc_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sc_process.kill()
                logger.info("Had to forcefully kill SuperCollider")
        
        logger.info("Done.")

if __name__ == "__main__":
    main()

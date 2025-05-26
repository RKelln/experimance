#!/usr/bin/env python3
"""
Test script for OscBridge class.
Sends test OSC messages and verifies they are received using oscdump.

Requirements:
- oscdump (install from liblo-tools package)
- pythonosc
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
import re
import unittest

# Add parent directory to path so we can import experimance_audio module
sys.path.append(str(Path(__file__).parent.parent.joinpath('src')))

from experimance_audio.osc_bridge import OscBridge
from experimance_common.constants import DEFAULT_PORTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class OscDumpListener:
    """Wrapper for oscdump to listen for OSC messages."""
    
    def __init__(self, port=DEFAULT_PORTS["audio_osc_recv_port"]):
        """Initialize the OSC listener.
        
        Args:
            port: OSC port to listen on
        """
        self.port = port
        self.process = None
        self.message_queue = Queue()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the oscdump process."""
        try:
            # Check if oscdump is installed
            subprocess.run(["which", "oscdump"], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            logger.error("oscdump not found. Please install liblo-tools package.")
            return False
            
        try:
            # Start oscdump as a subprocess
            cmd = ["oscdump", str(self.port)]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            logger.info(f"Started oscdump listening on port {self.port}")
            
            # Start a thread to read output
            self.running = True
            self.thread = threading.Thread(target=self._read_output)
            self.thread.daemon = True
            self.thread.start()
            
            # Give oscdump time to start
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"Failed to start oscdump: {e}")
            return False
    
    def _read_output(self):
        """Read output from oscdump and add to queue."""
        while self.running and self.process and self.process.poll() is None:
            line = self.process.stdout.readline() # type: ignore
            if line:
                self.message_queue.put(line.strip())
        
        # Read any remaining output
        if self.process and self.process.stdout:
            for line in self.process.stdout:
                if line:
                    self.message_queue.put(line.strip())
    
    def stop(self):
        """Stop the oscdump process."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
    
    def get_messages(self, timeout=5):
        """Get all messages received so far.
        
        Args:
            timeout: Time to wait for messages in seconds
            
        Returns:
            list: List of received messages
        """
        messages = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to get a message with a short timeout
                message = self.message_queue.get(timeout=0.1)
                messages.append(message)
                self.message_queue.task_done()
            except Exception:
                # No message available, continue checking until timeout
                pass
                
            # If we have any messages and haven't received a new one in a second, break
            if messages and time.time() - start_time > 1:
                break
        
        return messages
    
    def wait_for_message(self, address, timeout=5):
        """Wait for a message with a specific OSC address.
        
        Args:
            address: OSC address to wait for
            timeout: Timeout in seconds
            
        Returns:
            dict: Message details or None if timeout
        """
        start_time = time.time()
        pattern = re.compile(fr"{re.escape(address)}\s+(.*)")
        
        while time.time() - start_time < timeout:
            try:
                # Try to get a message with a short timeout
                message = self.message_queue.get(timeout=0.1)
                self.message_queue.task_done()
                
                # Check if it's the message we're waiting for
                match = pattern.search(message)
                if match:
                    return {
                        'address': address,
                        'args': match.group(1),
                        'raw': message
                    }
            except Exception:
                # No message available, continue checking until timeout
                pass
        
        return None


class TestOscBridge(unittest.TestCase):
    """Test cases for OscBridge class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Use the audio_osc_send_port for the bridge to send to
        # and listen for those messages with oscdump
        cls.test_port = DEFAULT_PORTS["audio_osc_recv_port"]
        
        # Start OSC listener
        cls.listener = OscDumpListener(port=cls.test_port)
        if not cls.listener.start():
            raise RuntimeError("Failed to start OSC listener")
            
        # Wait a bit for listener to fully initialize
        time.sleep(0.5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.listener:
            cls.listener.stop()
    
    def setUp(self):
        """Set up before each test."""
        # Create OscBridge instance
        self.bridge = OscBridge(port=self.test_port)
        
        # Clear any existing messages
        self.listener.get_messages()
    
    def tearDown(self):
        """Clean up after each test."""
        pass
    
    def test_send_spacetime(self):
        """Test sending spacetime message."""
        # Send the message
        result = self.bridge.send_spacetime("forest", "ancient")
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/spacetime")
        self.assertIsNotNone(message, "No message received for /spacetime")
        if message:  # Only check if message is not None
            self.assertIn("forest", message['args'])
            self.assertIn("ancient", message['args'])
    
    def test_include_tag(self):
        """Test sending include tag message."""
        # Send the message
        result = self.bridge.include_tag("birds")
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/include")
        self.assertIsNotNone(message, "No message received for /include")
        if message:  # Only check if message is not None
            self.assertIn("birds", message['args'])
    
    def test_exclude_tag(self):
        """Test sending exclude tag message."""
        # Send the message
        result = self.bridge.exclude_tag("water")
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/exclude")
        self.assertIsNotNone(message, "No message received for /exclude")
        if message:  # Only check if message is not None
            self.assertIn("water", message['args'])
    
    def test_listening(self):
        """Test sending listening message."""
        # Send the message (start)
        result = self.bridge.listening(True)
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/listening")
        self.assertIsNotNone(message, "No message received for /listening start")
        if message:  # Only check if message is not None
            self.assertIn("start", message['args'])
        
        # Send the message (stop)
        result = self.bridge.listening(False)
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/listening")
        self.assertIsNotNone(message, "No message received for /listening stop")
        if message:  # Only check if message is not None
            self.assertIn("stop", message['args'])
    
    def test_speaking(self):
        """Test sending speaking message."""
        # Send the message (start)
        result = self.bridge.speaking(True)
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/speaking")
        self.assertIsNotNone(message, "No message received for /speaking start")
        if message:  # Only check if message is not None
            self.assertIn("start", message['args'])
        
        # Send the message (stop)
        result = self.bridge.speaking(False)
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/speaking")
        self.assertIsNotNone(message, "No message received for /speaking stop")
        if message:  # Only check if message is not None
            self.assertIn("stop", message['args'])
    
    def test_transition(self):
        """Test sending transition message."""
        # Send the message (start)
        result = self.bridge.transition(True)
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/transition")
        self.assertIsNotNone(message, "No message received for /transition start")
        if message:  # Only check if message is not None
            self.assertIn("start", message['args'])
        
        # Send the message (stop)
        result = self.bridge.transition(False)
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/transition")
        self.assertIsNotNone(message, "No message received for /transition stop")
        if message:  # Only check if message is not None
            self.assertIn("stop", message['args'])
    
    def test_reload_configs(self):
        """Test sending reload configs message."""
        # Send the message
        result = self.bridge.reload_configs()
        self.assertTrue(result)
        
        # Wait for the message
        message = self.listener.wait_for_message("/reload")
        self.assertIsNotNone(message, "No message received for /reload")


def run_manual_test():
    """Run manual test with command line arguments."""
    parser = argparse.ArgumentParser(description='Test OscBridge with oscdump')
    parser.add_argument('--port', type=int, default=DEFAULT_PORTS["audio_osc_recv_port"],
                      help='OSC port to use')
    parser.add_argument('--message', type=str, default='/spacetime',
                      help='OSC message to send')
    parser.add_argument('--args', type=str, nargs='*', default=['temperate_forest', 'wilderness'],
                      help='Arguments for the OSC message')
    parser.add_argument('--wait', type=float, default=1.0,
                      help='Time to wait for responses in seconds')
    parser.add_argument('--no-oscdump', action='store_true',
                      help="Don't use oscdump, just send the message")
    parser.add_argument('--debug', action='store_true',
                      help='Show debug information')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print test information
    print("\n" + "="*60)
    print(f"OSC BRIDGE TEST: {args.message}")
    print("="*60)
    print(f"Port: {args.port}")
    print(f"Arguments: {args.args}")
    print(f"Wait time: {args.wait}s")
    print("-"*60)
    
    listener = None
    
    # If we're using oscdump, start the listener
    if not args.no_oscdump:
        try:
            # Check if oscdump is available
            subprocess.run(["which", "oscdump"], check=True, capture_output=True)
            
            # Start the OSC listener
            print(f"Starting oscdump to listen on port {args.port}...")
            listener = OscDumpListener(port=args.port)
            if not listener.start():
                print("\033[91mERROR: Failed to start OSC listener\033[0m")
                print("You may need to install liblo-tools: sudo apt install liblo-tools")
                return
            
        except subprocess.CalledProcessError:
            print("\033[93mWARNING: oscdump not found - can't verify message reception\033[0m")
            print("You may need to install liblo-tools: sudo apt install liblo-tools")
            args.no_oscdump = True
    
    try:
        # Create OscBridge instance
        bridge = OscBridge(port=args.port)
        
        # Send a test message based on the message type
        print(f"\nSending OSC message: {args.message} {args.args}")
        
        success = False
        if args.message == '/spacetime':
            if len(args.args) >= 2:
                success = bridge.send_spacetime(args.args[0], args.args[1])
            else:
                print("\033[91mERROR: /spacetime requires two arguments: biome era\033[0m")
        elif args.message == '/include':
            if len(args.args) >= 1:
                success = bridge.include_tag(args.args[0])
            else:
                print("\033[91mERROR: /include requires one argument: tag\033[0m")
        elif args.message == '/exclude':
            if len(args.args) >= 1:
                success = bridge.exclude_tag(args.args[0])
            else:
                print("\033[91mERROR: /exclude requires one argument: tag\033[0m")
        elif args.message == '/listening':
            if len(args.args) >= 1:
                status = args.args[0].lower() in ('true', 'start', '1', 'yes', 'on')
                success = bridge.listening(status)
                print(f"Listening status set to: {status}")
            else:
                print("\033[91mERROR: /listening requires one argument: true|false\033[0m")
        elif args.message == '/speaking':
            if len(args.args) >= 1:
                status = args.args[0].lower() in ('true', 'start', '1', 'yes', 'on')
                success = bridge.speaking(status)
                print(f"Speaking status set to: {status}")
            else:
                print("\033[91mERROR: /speaking requires one argument: true|false\033[0m")
        elif args.message == '/transition':
            if len(args.args) >= 1:
                status = args.args[0].lower() in ('true', 'start', '1', 'yes', 'on')
                success = bridge.transition(status)
                print(f"Transition status set to: {status}")
            else:
                print("\033[91mERROR: /transition requires one argument: true|false\033[0m")
        elif args.message == '/reload':
            success = bridge.reload_configs()
        elif args.message == '/quit':
            assert bridge.client is not None, "OscBridge client is not initialized"
            success = bridge.client.send_message("/quit", [])
            print("Sent quit command to SuperCollider")
        else:
            print(f"\033[91mERROR: Unknown message type: {args.message}\033[0m")
            print("Supported messages: /spacetime, /include, /exclude, /listening, /speaking, /transition, /reload, /quit")
            return
        
        if success:
            print("\033[92mMessage sent successfully\033[0m")
        else:
            print("\033[91mFailed to send message\033[0m")
        
        # Wait for a response if using oscdump
        if listener:
            print(f"\nWaiting for OSC responses (timeout: {args.wait}s)...")
            time.sleep(args.wait)
            
            # Print received messages
            messages = listener.get_messages()
            if messages:
                print("\n\033[92mReceived OSC messages:\033[0m")
                for msg in messages:
                    print(f"  \033[96m{msg}\033[0m")
            else:
                print("\n\033[91mNo OSC messages received!\033[0m")
                print("\nPossible issues:")
                print("1. SuperCollider is not running")
                print("2. SuperCollider is using a different port")
                print("3. The network connection is blocked")
                print("4. There's a typo in the message or arguments")
        else:
            if not args.no_oscdump:
                print("\nCannot verify if the message was received (oscdump not started)")
    
        # Show help for next steps
        print("\n" + "-"*60)
        print("NEXT STEPS:")
        if listener and not messages:
            print("- Try running SuperCollider with test_osc.scd")
            print("- Check that the port settings match in constants.py and SuperCollider")
            print("- Run 'oscdump 5568' manually to debug")
        print("- Try a different message with --message=<address> --args=<values...>")
        print(f"- Use a different port with --port={args.port+1}")
        print("-"*60 + "\n")
    
    finally:
        # Clean up
        if listener is not None:
            listener.stop()


def run_integrated_test():
    """Run an integrated test that starts SuperCollider and sends messages."""
    parser = argparse.ArgumentParser(description='Test OscBridge with SuperCollider')
    parser.add_argument('--port', type=int, default=DEFAULT_PORTS["audio_osc_recv_port"],
                      help='OSC port to use for sending')
    parser.add_argument('--script', type=str, default=None,
                      help='Path to SuperCollider script (default: auto-detect test_osc.scd)')
    parser.add_argument('--sclang', type=str, default="sclang",
                      help='Path to sclang executable')
    parser.add_argument('--debug', action='store_true',
                      help='Show debug information')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print test information
    print("\n" + "="*60)
    print("INTEGRATED SUPERCOLLIDER OSC TEST")
    print("="*60)
    
    # Detect script path if not provided
    sc_script_path = args.script
    if not sc_script_path:
        # Try to find the test_osc.scd script
        script_dir = Path(__file__).parent.parent
        possible_paths = [
            script_dir / "sc_scripts" / "test_osc.scd",
            script_dir.parent / "services" / "audio" / "sc_scripts" / "test_osc.scd",
            Path("/home/ryankelln/Documents/Projects/Art/experimance/installation/software/experimance/services/audio/sc_scripts/test_osc.scd")
        ]
        
        for path in possible_paths:
            if path.exists():
                sc_script_path = str(path)
                break
    
    if not sc_script_path or not Path(sc_script_path).exists():
        print("\033[91mERROR: Could not find SuperCollider script\033[0m")
        print("Please specify the path with --script")
        return
        
    print(f"SuperCollider Script: {sc_script_path}")
    print(f"SuperCollider Binary: {args.sclang}")
    print(f"OSC Port: {args.port}")
    print("-"*60)
    
    # Create OscBridge
    bridge = OscBridge(port=DEFAULT_PORTS["audio_osc_send_port"])
    
    # Start SuperCollider
    print("\nStarting SuperCollider...")
    log_path = bridge.start_supercollider(sc_script_path, args.sclang, log_to_console=True)
    
    if not log_path:
        print("\033[91mERROR: Failed to start SuperCollider\033[0m")
        return
    
    print(f"SuperCollider started! Log file: {log_path}")
    
    # Wait for SuperCollider to initialize
    print("Waiting for SuperCollider to initialize...")
    time.sleep(4)
    
    try:
        # Run a sequence of test messages
        test_sequence = [
            ("Sending spacetime context", lambda: bridge.send_spacetime("forest", "ancient")),
            ("Including 'birds' tag", lambda: bridge.include_tag("birds")),
            ("Excluding 'water' tag", lambda: bridge.exclude_tag("water")),
            ("Starting listening", lambda: bridge.listening(True)),
            ("Stopping listening", lambda: bridge.listening(False)),
            ("Starting speaking", lambda: bridge.speaking(True)),
            ("Stopping speaking", lambda: bridge.speaking(False)),
            ("Starting transition", lambda: bridge.transition(True)),
            ("Stopping transition", lambda: bridge.transition(False)),
            ("Reloading configs", lambda: bridge.reload_configs())
        ]
        
        # Run each test with a delay between them
        for i, (description, action) in enumerate(test_sequence):
            print(f"\n[{i+1}/{len(test_sequence)}] {description}...")
            if action():
                print("\033[92m✓ Message sent successfully\033[0m")
            else:
                print("\033[91m✗ Failed to send message\033[0m")
            time.sleep(1)  # Give SuperCollider time to respond
        
        print("\n\033[92mAll test messages sent!\033[0m")
        print("Check the SuperCollider output to verify reception")
        
        # Wait for user input before stopping
        input("\nPress Enter to stop SuperCollider and exit...")
    
    finally:
        # Stop SuperCollider
        print("\nStopping SuperCollider...")
        bridge.stop_supercollider()
        print("Done!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--manual':
            # Remove the --manual argument
            sys.argv.pop(1)
            run_manual_test()
        elif sys.argv[1] == '--integrated':
            # Remove the --integrated argument
            sys.argv.pop(1)
            run_integrated_test()
        else:
            # Assume it's a unittest argument
            unittest.main()
    else:
        unittest.main()

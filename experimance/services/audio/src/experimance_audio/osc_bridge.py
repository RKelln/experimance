"""
OSC bridge for communication between the Experimance Audio Service and SuperCollider.

This module provides functionality to send OSC messages to SuperCollider,
enabling control of the audio engine from the Experimance system.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from pythonosc import udp_client
from typing import Any, Dict, List, Optional, Union

from experimance_common.constants import DEFAULT_PORTS

logger = logging.getLogger(__name__)

# Default paths for SuperCollider
DEFAULT_SCLANG_PATH = "sclang"  # Assumes sclang is in PATH


class OscBridge:
    """Bridge for sending OSC messages to SuperCollider."""
    
    host: str
    port: int
    client: Optional[udp_client.SimpleUDPClient] = None
    sc_process: Optional[subprocess.Popen] = None

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORTS["audio_osc_send_port"]):
        """Initialize the OSC bridge.
        
        Args:
            host: SuperCollider host address
            port: SuperCollider OSC listening port
        """
        self.host = host
        self.port = port
        self.client = None
        self.sc_process = None
        self._connect()
    
    def _connect(self) -> bool:
        """Initialize the OSC client connection.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            self.client = udp_client.SimpleUDPClient(self.host, self.port)
            logger.info(f"OSC client initialized at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Error initializing OSC client: {e}")
            return False
    
    def send_spacetime(self, biome: str, era: str) -> bool:
        """Send spacetime context to SuperCollider.
        
        Args:
            biome: Current biome name
            era: Current era name
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/spacetime", [biome, era])
            logger.debug(f"Sent spacetime: biome={biome}, era={era}")
            return True
        except Exception as e:
            logger.error(f"Error sending spacetime OSC message: {e}")
            return False
    
    def include_tag(self, tag: str) -> bool:
        """Include a sound tag in the active set.
        
        Args:
            tag: Tag to include
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/include", [tag])
            logger.debug(f"Sent include tag: {tag}")
            return True
        except Exception as e:
            logger.error(f"Error sending include OSC message: {e}")
            return False
    
    def exclude_tag(self, tag: str) -> bool:
        """Exclude a sound tag from the active set.
        
        Args:
            tag: Tag to exclude
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/exclude", [tag])
            logger.debug(f"Sent exclude tag: {tag}")
            return True
        except Exception as e:
            logger.error(f"Error sending exclude OSC message: {e}")
            return False
    
    def listening(self, start: bool) -> bool:
        """Trigger listening UI sound effect.
        
        Args:
            start: True to start, False to stop
            
        Returns:
            bool: True if message was sent successfully
        """
        status = "start" if start else "stop"
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/listening", [status])
            logger.debug(f"Sent listening: {status}")
            return True
        except Exception as e:
            logger.error(f"Error sending listening OSC message: {e}")
            return False
    
    def speaking(self, start: bool) -> bool:
        """Trigger speaking UI sound effect.
        
        Args:
            start: True to start, False to stop
            
        Returns:
            bool: True if message was sent successfully
        """
        status = "start" if start else "stop"
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/speaking", [status])
            logger.debug(f"Sent speaking: {status}")
            return True
        except Exception as e:
            logger.error(f"Error sending speaking OSC message: {e}")
            return False
    
    def transition(self, start: bool) -> bool:
        """Trigger transition sound effect.
        
        Args:
            start: True to start, False to stop
            
        Returns:
            bool: True if message was sent successfully
        """
        status = "start" if start else "stop"
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/transition", [status])
            logger.debug(f"Sent transition: {status}")
            return True
        except Exception as e:
            logger.error(f"Error sending transition OSC message: {e}")
            return False
    
    def reload_configs(self) -> bool:
        """Trigger configuration reload in SuperCollider.
        
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/reload", [])
            logger.info("Sent reload command to SuperCollider")
            return True
        except Exception as e:
            logger.error(f"Error sending reload OSC message: {e}")
            return False
    
    def start_supercollider(self, sc_script_path: str, sclang_path: str = DEFAULT_SCLANG_PATH) -> bool:
        """Start SuperCollider with the given script.
        
        Args:
            sc_script_path: Path to the SuperCollider script to execute
            sclang_path: Path to the SuperCollider language interpreter executable
            
        Returns:
            bool: True if SuperCollider was started successfully, False otherwise
        """
        try:
            script_path = Path(sc_script_path).resolve()
            if not script_path.exists():
                logger.error(f"SuperCollider script not found: {script_path}")
                return False

            # Start SuperCollider in a non-blocking subprocess
            logger.info(f"Starting SuperCollider with script: {script_path}")
            self.sc_process = subprocess.Popen(
                [sclang_path, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            
            # Log the process ID for debugging/cleanup
            logger.info(f"SuperCollider started with PID: {self.sc_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SuperCollider: {e}")
            return False
    
    def stop_supercollider(self, timeout: float = 3.0) -> bool:
        """Stop the SuperCollider process if it's running.
        
        Args:
            timeout: Timeout in seconds to wait for graceful termination
            
        Returns:
            bool: True if SuperCollider was stopped successfully, False otherwise
        """
        if self.sc_process is None:
            logger.debug("No SuperCollider process to stop")
            return True
            
        try:
            # First try to quit gracefully by sending an OSC message
            if self.client:
                try:
                    # Send a quit command to SuperCollider
                    logger.info("Sending quit command to SuperCollider")
                    self.client.send_message("/quit", [])
                    
                    # Give SuperCollider time to quit gracefully
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if self.sc_process is not None and self.sc_process.poll() is not None:
                            logger.info("SuperCollider quit gracefully")
                            self.sc_process = None
                            return True
                        time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed to send quit command to SuperCollider: {e}")
            
            # If still running, terminate the process
            if self.sc_process is not None and self.sc_process.poll() is None:
                logger.info(f"Terminating SuperCollider process (PID: {self.sc_process.pid})")
                self.sc_process.terminate()
                
                # Wait for termination
                try:
                    self.sc_process.wait(timeout=timeout)
                    logger.info("SuperCollider terminated")
                    self.sc_process = None
                    return True
                except subprocess.TimeoutExpired:
                    # If it still doesn't terminate, kill it
                    logger.warning("SuperCollider did not terminate, killing process")
                    self.sc_process.kill() # type: ignore
                    self.sc_process.wait(timeout=1.0) # type: ignore
                    logger.info("SuperCollider killed")
                    self.sc_process = None
                    return True
                    
        except Exception as e:
            logger.error(f"Error stopping SuperCollider: {e}")
            # Make a best effort to force kill if everything else fails
            try:
                if self.sc_process is not None and self.sc_process.poll() is None:
                    self.sc_process.kill()
                    self.sc_process = None
            except Exception:
                logger.error("Failed to kill SuperCollider process")
            
        # If we get here, either the process was stopped or we couldn't stop it
        # In either case, we no longer have a reference to it
        self.sc_process = None
        return False

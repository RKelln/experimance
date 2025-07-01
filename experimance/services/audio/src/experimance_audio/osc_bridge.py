"""
OSC bridge for communication between the Experimance Audio Service and SuperCollider.

This module provides functionality to send OSC messages to SuperCollider,
enabling control of the audio engine from the Experimance system.
"""

import datetime
import logging
import os
import signal
import subprocess
import sys
import threading
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
    
    def send_spacetime(self, biome: str, era: str, tags: Optional[List[str]] = None) -> bool:
        """Send spacetime context to SuperCollider.
        This sets the current biome and era, and optionally includes tags.

        Args:
            biome: Current biome name
            era: Current era name
            tags: Optional list of tags to include in the context
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            message = [biome, era]
            if tags is not None:
                message.extend(tags)
            self.client.send_message("/spacetime", message)
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
    
    def set_volume(self, volume_type: str, value: float) -> bool:
        """Set volume level for a specific audio category.
        
        Args:
            volume_type: Type of volume to set ('master', 'environment', 'music', 'sfx')
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            # Ensure volume is in valid range
            value = max(0.0, min(1.0, value))
            self.client.send_message(f"/volume/{volume_type}", [value])
            logger.debug(f"Set {volume_type} volume: {value}")
            return True
        except Exception as e:
            logger.error(f"Error sending volume OSC message: {e}")
            return False
    
    def set_master_volume(self, value: float) -> bool:
        """Set master volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if message was sent successfully
        """
        return self.set_volume("master", value)
    
    def set_environment_volume(self, value: float) -> bool:
        """Set environment sounds volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if message was sent successfully
        """
        return self.set_volume("environment", value)
    
    def set_music_volume(self, value: float) -> bool:
        """Set music volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if message was sent successfully
        """
        return self.set_volume("music", value)
    
    def set_sfx_volume(self, value: float) -> bool:
        """Set sound effects volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if message was sent successfully
        """
        return self.set_volume("sfx", value)
    
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
            
    def request_synth_info(self) -> bool:
        """Request detailed information about currently playing synths.
        
        This is a debug endpoint that prints information about all currently active
        music loops, including their start times, current positions, and buffer details.
        
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/synth_info", [])
            logger.debug("Requested synth information from SuperCollider")
            return True
        except Exception as e:
            logger.error(f"Error sending synth_info OSC message: {e}")
            return False
    
    def start_supercollider(self, sc_script_path: str, sclang_path: str = DEFAULT_SCLANG_PATH, 
                       log_to_file: bool = True, log_to_console: bool = False) -> Optional[str]:
        """Start SuperCollider with the given script.
        
        Args:
            sc_script_path: Path to the SuperCollider script to execute
            sclang_path: Path to the SuperCollider language interpreter executable
            log_to_file: Whether to log SuperCollider output to a file
            log_to_console: Whether to log SuperCollider output to the console
            
        Returns:
            Optional[str]: Path to the log file if SuperCollider was started successfully and log_to_file is True,
                           None if SuperCollider could not be started
        """
        try:
            script_path = Path(sc_script_path).resolve()
            if not script_path.exists():
                logger.error(f"SuperCollider script not found: {script_path}")
                return None

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
            
            # Start threads to handle stdout and stderr
            log_path = self._start_output_threads(log_to_file, log_to_console)
            
            return log_path
            
        except Exception as e:
            logger.error(f"Failed to start SuperCollider: {e}")
            return None
            
    def _start_output_threads(self, log_to_file: bool = True, log_to_console: bool = False) -> Optional[str]:
        """Start threads to process SuperCollider stdout and stderr streams.
        
        Args:
            log_to_file: Whether to log SuperCollider output to a file
            log_to_console: Whether to log SuperCollider output to the console
            
        Returns:
            Optional[str]: Path to the log file if created, None otherwise
        """
        if self.sc_process is None:
            return None
        
        # Create logs directory if it doesn't exist
        log_file = None
        log_file_path = None
        
        if log_to_file:
            # Create log directory
            script_dir = Path(__file__).parent.parent.parent  # Go up from src/experimance_audio to services/audio
            log_dir = script_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = log_dir / f"supercollider_{timestamp}.log"
            log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)  # Line buffered
            logger.info(f"SuperCollider logs will be saved to: {log_file_path}")
        
        def _process_stream(stream, prefix):
            """Process each line from a stream and log it."""
            for line in iter(stream.readline, ''):
                if not line.strip():
                    continue
                    
                formatted_line = f"{prefix}: {line.rstrip()}"
                
                # Log to file if requested
                if log_to_file and log_file:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{timestamp}] {formatted_line}\n")
                    log_file.flush()  # Ensure it's written immediately
                
                # Log to console if requested
                if log_to_console:
                    logger.info(formatted_line)
            
            stream.close()
            
            # Close the log file when the thread exits
            if log_to_file and log_file:
                log_file.close()
        
        # Create and start threads for stdout and stderr
        if self.sc_process.stdout:
            stdout_thread = threading.Thread(
                target=_process_stream, 
                args=(self.sc_process.stdout, "SuperCollider"),
                daemon=True
            )
            stdout_thread.start()
            
        if self.sc_process.stderr:
            stderr_thread = threading.Thread(
                target=_process_stream, 
                args=(self.sc_process.stderr, "SuperCollider ERROR"),
                daemon=True
            )
            stderr_thread.start()
            
        return str(log_file_path) if log_file_path else None

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
                    # Send a quit command to SuperCollider - this triggers our cleanup routine
                    logger.info("Sending quit command to SuperCollider")
                    self.client.send_message("/quit", [])
                    
                    # Give SuperCollider time to run its cleanup and quit gracefully
                    logger.debug("Waiting for SuperCollider to clean up and quit gracefully...")
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if self.sc_process is not None and self.sc_process.poll() is not None:
                            logger.info("SuperCollider quit gracefully")
                            self.sc_process = None
                            # Wait another second to ensure JACK resources are released
                            time.sleep(1.0)
                            return True
                        time.sleep(0.1)
                    logger.warning("SuperCollider did not quit gracefully within timeout")
                except Exception as e:
                    logger.warning(f"Failed to send quit command to SuperCollider: {e}")
            
            # If still running, terminate the process
            if self.sc_process is not None and self.sc_process.poll() is None:
                logger.info(f"Terminating SuperCollider process (PID: {self.sc_process.pid})")
                
                # Try to send quit again with a different approach (using shell)
                try:
                    # Try using oscsend if available
                    subprocess.run(['which', 'oscsend'], check=True, capture_output=True)
                    logger.debug("Using oscsend to send quit command")
                    subprocess.run(['oscsend', 'localhost', str(self.port), '/quit'], 
                                  timeout=1.0, check=False)
                    time.sleep(1.0)  # Give it a chance to process
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.debug("oscsend not available")
                
                # Now try terminating the process
                self.sc_process.terminate()
                
                # Wait for termination
                try:
                    self.sc_process.wait(timeout=timeout)
                    logger.info("SuperCollider terminated")
                    self.sc_process = None
                    # Wait to ensure JACK resources are released
                    time.sleep(1.0)
                    return True
                except subprocess.TimeoutExpired:
                    # If it still doesn't terminate, kill it
                    logger.warning("SuperCollider did not terminate, killing process")
                    self.sc_process.kill() # type: ignore
                    self.sc_process.wait(timeout=1.0) # type: ignore
                    logger.info("SuperCollider killed")
                    self.sc_process = None
                    # Wait to ensure JACK resources are released
                    time.sleep(1.0)
                    return True
                    
        except Exception as e:
            logger.error(f"Error stopping SuperCollider: {e}")
            # Make a best effort to force kill if everything else fails
            try:
                if self.sc_process is not None and self.sc_process.poll() is None:
                    self.sc_process.kill()
                    self.sc_process = None
                    # Wait to ensure JACK resources are released
                    time.sleep(1.0)
            except Exception:
                logger.error("Failed to kill SuperCollider process")
            
        # If we get here, either the process was stopped or we couldn't stop it
        # In either case, we no longer have a reference to it
        self.sc_process = None
        
        # Try to clean up JACK connections as a last resort
        try:
            # Check if jackd is running
            jack_check = subprocess.run(['pgrep', 'jackd'], capture_output=True, text=True)
            if jack_check.returncode == 0:
                logger.debug("JACK is running, attempting to disconnect any lingering JACK clients")
                # Try to disconnect all SuperCollider connections
                subprocess.run(['jack_disconnect', '-a'], check=False, capture_output=True)
        except Exception as e:
            logger.debug(f"Error cleaning up JACK connections: {e}")
            
        return False

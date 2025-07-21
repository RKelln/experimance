"""
Main Experimance Audio Service implementation.

This service:
1. Subscribes to system events via ZeroMQ (EraChanged, etc.)
2. Communicates with SuperCollider via OSC to control audio playback
3. Manages audio state and configuration
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import pyaudio
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.base_service import BaseService
from experimance_common.health import HealthStatus
from experimance_common.zmq.config import MessageDataType
from experimance_common.zmq.services import PubSubService
from experimance_common.schemas import (
    Era, Biome, SpaceTimeUpdate, AgentControlEvent, IdleStatus, MessageBase, MessageType
)
from pydantic import ValidationError

from .config import AudioServiceConfig, DEFAULT_CONFIG_PATH
from .config_loader import AudioConfigLoader
from .osc_bridge import OscBridge, DEFAULT_SCLANG_PATH

# Configure logging
logger = logging.getLogger(__name__)

class AudioService(BaseService):
    """
    Audio Service for the Experimance interactive installation.
    
    This service subscribes to system events and controls the SuperCollider
    audio engine via OSC based on the current state of the installation.
    """
    
    def __init__(self, config: AudioServiceConfig):
        """Initialize the audio service.
        
        Args:
            config: Service configuration object.
        """
        self.config = config
            
        # Initialize base service
        super().__init__(service_name=self.config.service_name, service_type="audio")
        
        # Use ZMQ configuration from config, updating the service name to match
        self.zmq_config = self.config.zmq
        self.zmq_config.name = f"{self.config.service_name}-pubsub"
        
        # Create ZMQ service using composition
        self.zmq_service = PubSubService(self.zmq_config)
        
        # Initialize OSC bridge for communication with SuperCollider
        self.osc = OscBridge(
            host=self.config.osc.host, 
            port=self.config.osc.send_port
        )
        
        # Initialize configuration loader
        self.audio_config = AudioConfigLoader(config_dir=self.config.audio.config_dir)
        self.audio_config.load_configs()
        
        # file tracking
        self.tmp_script_path = None  # Temporary script path for SuperCollider
        self.jack_process = None     # JACK process handle

        # State tracking
        self.current_biome = None
        self.current_era = None
        self.active_tags = set()  # Track which tags are active
        
        # Volume settings (initialized from config)
        self.master_volume = self.config.audio.master_volume
        self.environment_volume = self.config.audio.environment_volume
        self.music_volume = self.config.audio.music_volume
        self.sfx_volume = self.config.audio.sfx_volume
    
    def _resolve_audio_device(self, device_identifier: str) -> Optional[str]:
        """Resolve a device identifier to a hardware address.
        
        Args:
            device_identifier: Either a full hw:X,Y address or a partial device name
            
        Returns:
            Hardware address (e.g., "hw:4,0") or None if not found
        """
        # If it's already a hw:X,Y format, return as-is
        if re.match(r'^hw:\d+,\d+$', device_identifier):
            self.record_health_check(
                "device_resolution",
                HealthStatus.HEALTHY,
                f"Device already in hardware format: {device_identifier}"
            )
            return device_identifier
            
        # If it starts with plughw:, convert to hw:
        if device_identifier.startswith('plughw:'):
            hw_addr = device_identifier.replace('plughw:', 'hw:')
            self.record_health_check(
                "device_resolution",
                HealthStatus.HEALTHY,
                f"Converted plughw to hw format: {device_identifier} -> {hw_addr}"
            )
            return hw_addr
        
        # Otherwise, try to find by partial name match
        try:
            p = pyaudio.PyAudio()
            
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    device_name = device_info['name']
                    
                    # Check if the identifier matches part of the device name
                    if device_identifier.lower() in device_name.lower():
                        # Extract hardware address from device name
                        # Look for patterns like "(hw:4,0)" in the name
                        hw_match = re.search(r'\(hw:(\d+),(\d+)\)', device_name)
                        if hw_match:
                            hw_addr = f"hw:{hw_match.group(1)},{hw_match.group(2)}"
                            logger.info(f"Resolved device '{device_identifier}' to '{hw_addr}' (device: {device_name})")
                            self.record_health_check(
                                "device_resolution",
                                HealthStatus.HEALTHY,
                                f"Successfully resolved device '{device_identifier}' to '{hw_addr}'",
                                metadata={
                                    "device_identifier": device_identifier,
                                    "resolved_address": hw_addr,
                                    "device_name": device_name,
                                    "device_index": i
                                }
                            )
                            p.terminate()
                            return hw_addr
                            
                except Exception as e:
                    logger.debug(f"Error checking device {i}: {e}")
                    continue
                    
            p.terminate()
            
            # PyAudio didn't find the device, try ALSA card fallback
            logger.debug(f"Device '{device_identifier}' not found in PyAudio, trying ALSA card fallback")
            hw_addr = self._resolve_device_via_alsa_cards(device_identifier)
            if hw_addr:
                return hw_addr
            
            # Device not found
            error_msg = f"Could not resolve device identifier '{device_identifier}' to hardware address"
            logger.warning(error_msg)
            self.record_health_check(
                "device_resolution",
                HealthStatus.WARNING,
                error_msg,
                metadata={"device_identifier": device_identifier}
            )
            return None
            
        except Exception as e:
            error_msg = f"Error resolving audio device: {e}"
            logger.error(error_msg)
            self.record_health_check(
                "device_resolution",
                HealthStatus.ERROR,
                error_msg,
                metadata={
                    "device_identifier": device_identifier,
                    "error_type": type(e).__name__
                }
            )
            return None
    
    def _resolve_device_via_alsa_cards(self, device_identifier: str) -> Optional[str]:
        """Fallback method to resolve device via ALSA /proc/asound/cards.
        
        Args:
            device_identifier: Device name to search for
            
        Returns:
            Hardware address (e.g., "hw:4,0") or None if not found
        """
        try:
            with open('/proc/asound/cards', 'r') as f:
                cards_content = f.read()
                
            # Look for lines like: " 4 [ICUSBAUDIO7D   ]: USB-Audio - ICUSBAUDIO7D"
            for line in cards_content.split('\n'):
                if device_identifier in line:
                    # Extract card number from line like " 4 [ICUSBAUDIO7D   ]:"
                    card_match = re.match(r'\s*(\d+)\s*\[', line)
                    if card_match:
                        card_number = card_match.group(1)
                        hw_addr = f"hw:{card_number},0"
                        logger.info(f"Resolved device '{device_identifier}' to '{hw_addr}' via ALSA cards fallback")
                        self.record_health_check(
                            "device_resolution",
                            HealthStatus.HEALTHY,
                            f"Successfully resolved device '{device_identifier}' to '{hw_addr}' via ALSA fallback",
                            metadata={
                                "device_identifier": device_identifier,
                                "resolved_address": hw_addr,
                                "card_number": card_number,
                                "method": "alsa_cards_fallback"
                            }
                        )
                        return hw_addr
                        
        except Exception as e:
            logger.debug(f"Error reading ALSA cards: {e}")
            
        return None
    
    def _is_jack_running(self) -> bool:
        """Check if JACK is currently running.
        
        Returns:
            True if JACK is running, False otherwise
        """
        try:
            # Check for both jackd and jackdbus processes
            jackd_result = subprocess.run(['pgrep', 'jackd'], capture_output=True, text=True)
            jackdbus_result = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
            
            jackd_running = jackd_result.returncode == 0
            jackdbus_running = jackdbus_result.returncode == 0
            
            if jackd_running or jackdbus_running:
                logger.debug(f"JACK status: jackd={jackd_running}, jackdbus={jackdbus_running}")
                return True
            return False
        except Exception as e:
            logger.debug(f"Error checking JACK status: {e}")
            return False
    
    def _stop_existing_jack(self) -> bool:
        """Stop any existing JACK processes before starting our own.
        
        Returns:
            True if all JACK processes were stopped successfully, False otherwise
        """
        try:
            # Stop jackdbus first if running
            jackdbus_result = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
            if jackdbus_result.returncode == 0:
                logger.info("Stopping existing jackdbus process")
                try:
                    # Try to stop jackdbus gracefully using jack_control
                    subprocess.run(['jack_control', 'stop'], capture_output=True, text=True, timeout=3.0)
                    subprocess.run(['jack_control', 'exit'], capture_output=True, text=True, timeout=3.0)
                    time.sleep(2.0)  # Give it more time to stop
                    
                    # Check if jackdbus is still running after graceful stop
                    check_result = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
                    if check_result.returncode == 0:
                        logger.warning("jackdbus did not stop gracefully, force killing")
                        subprocess.run(['pkill', '-TERM', 'jackdbus'], capture_output=True, text=True)
                        time.sleep(1.0)
                        
                        # Final check and force kill if still running
                        final_check = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
                        if final_check.returncode == 0:
                            logger.warning("Force killing jackdbus with SIGKILL")
                            subprocess.run(['pkill', '-KILL', 'jackdbus'], capture_output=True, text=True)
                            time.sleep(1.0)
                    
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # Fallback to pkill if jack_control is not available or times out
                    logger.warning("jack_control not available or timed out, using pkill")
                    subprocess.run(['pkill', '-TERM', 'jackdbus'], capture_output=True, text=True)
                    time.sleep(1.0)
                    subprocess.run(['pkill', '-KILL', 'jackdbus'], capture_output=True, text=True)
                    time.sleep(1.0)
            
            # Stop any jackd processes
            jackd_result = subprocess.run(['pgrep', 'jackd'], capture_output=True, text=True)
            if jackd_result.returncode == 0:
                logger.info("Stopping existing jackd process")
                subprocess.run(['pkill', '-TERM', 'jackd'], capture_output=True, text=True)
                time.sleep(1.0)
                # Force kill if still running
                final_jackd_check = subprocess.run(['pgrep', 'jackd'], capture_output=True, text=True)
                if final_jackd_check.returncode == 0:
                    subprocess.run(['pkill', '-KILL', 'jackd'], capture_output=True, text=True)
                    time.sleep(1.0)
            
            # Verify they're stopped
            final_check = self._is_jack_running()
            if not final_check:
                logger.info("All existing JACK processes stopped successfully")
                self.record_health_check(
                    "jack_cleanup",
                    HealthStatus.HEALTHY,
                    "Successfully stopped existing JACK processes"
                )
                return True
            else:
                logger.warning("Some JACK processes may still be running")
                # Get details about what's still running
                remaining_processes = []
                try:
                    jackd_check = subprocess.run(['pgrep', 'jackd'], capture_output=True, text=True)
                    if jackd_check.returncode == 0:
                        remaining_processes.append("jackd")
                    jackdbus_check = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
                    if jackdbus_check.returncode == 0:
                        remaining_processes.append("jackdbus")
                except Exception:
                    pass
                    
                self.record_health_check(
                    "jack_cleanup",
                    HealthStatus.WARNING,
                    f"Some JACK processes still running: {', '.join(remaining_processes)}",
                    metadata={"remaining_processes": remaining_processes}
                )
                return False
                
        except Exception as e:
            error_msg = f"Error stopping existing JACK processes: {e}"
            logger.error(error_msg)
            self.record_health_check(
                "jack_cleanup",
                HealthStatus.ERROR,
                error_msg,
                metadata={"error_type": type(e).__name__}
            )
            return False

    def _configure_jackdbus(self, device: str) -> bool:
        """Configure jackdbus to use the specified device.
        
        Args:
            device: Hardware device address (e.g., "hw:4,0")
            
        Returns:
            True if jackdbus was configured successfully, False otherwise
        """
        try:
            logger.info(f"Configuring jackdbus to use device: {device}")
            
            # Configure jackdbus with our settings
            output_channels = self.config.supercollider.jack_output_channels or self.config.supercollider.output_channels
            commands = [
                ['jack_control', 'ds', 'alsa'],  # Set driver to alsa
                ['jack_control', 'dps', 'device', device],  # Set device
                ['jack_control', 'dps', 'rate', str(self.config.supercollider.jack_sample_rate)],  # Set sample rate
                ['jack_control', 'dps', 'period', str(self.config.supercollider.jack_buffer_size)],  # Set buffer size
                ['jack_control', 'dps', 'nperiods', str(self.config.supercollider.jack_periods)],  # Set periods
                ['jack_control', 'dps', 'outchannels', str(output_channels)],  # Set output channels
            ]
            
            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                if result.returncode != 0:
                    logger.warning(f"Command {' '.join(cmd)} failed: {result.stderr}")
                    return False
            
            # Start jackdbus
            result = subprocess.run(['jack_control', 'start'], capture_output=True, text=True, timeout=10.0)
            if result.returncode != 0:
                logger.error(f"Failed to start jackdbus: {result.stderr}")
                return False
            
            # Wait for it to start
            start_time = time.time()
            timeout = 5.0
            while time.time() - start_time < timeout:
                status_result = subprocess.run(['jack_control', 'status'], capture_output=True, text=True)
                if status_result.returncode == 0 and 'started' in status_result.stdout.lower():
                    logger.info("jackdbus started successfully")
                    self.record_health_check(
                        "jackdbus_config",
                        HealthStatus.HEALTHY,
                        f"jackdbus configured and started with device {device}",
                        metadata={
                            "device": device,
                            "sample_rate": self.config.supercollider.jack_sample_rate,
                            "buffer_size": self.config.supercollider.jack_buffer_size
                        }
                    )
                    return True
                time.sleep(0.2)
            
            logger.error("jackdbus failed to start within timeout")
            return False
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Error configuring jackdbus: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error configuring jackdbus: {e}")
            return False

    async def _start_jack(self, device: str) -> bool:
        """Start JACK with the specified device.
        
        Args:
            device: Hardware device address (e.g., "hw:4,0")
            
        Returns:
            True if JACK started successfully, False otherwise
        """
        # First check if JACK is already running and properly configured
        if self._is_jack_running():
            logger.info("JACK is already running")
            # Check if it's jackdbus and try to verify it's using the right device
            jackdbus_result = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
            if jackdbus_result.returncode == 0:
                # jackdbus is running, check its status
                try:
                    status_result = subprocess.run(['jack_control', 'status'], capture_output=True, text=True, timeout=3.0)
                    if status_result.returncode == 0 and 'started' in status_result.stdout.lower():
                        logger.info("jackdbus is already running and started, will use existing server")
                        self.record_health_check(
                            "jack_status",
                            HealthStatus.HEALTHY,
                            "Using existing jackdbus server"
                        )
                        return True
                    else:
                        logger.info("jackdbus is running but not started, will configure it")
                        return self._configure_jackdbus(device)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("Cannot communicate with jackdbus, stopping it and starting our own")
                    if not self._stop_existing_jack():
                        logger.warning("Could not cleanly stop existing JACK, attempting to start anyway")
            else:
                logger.info("jackd is running, will stop and restart with correct device")
                if not self._stop_existing_jack():
                    logger.warning("Could not cleanly stop existing JACK, attempting to start anyway")
        
        # Try to configure jackdbus first (preferred method for Ubuntu)
        jackdbus_result = subprocess.run(['pgrep', 'jackdbus'], capture_output=True, text=True)
        if jackdbus_result.returncode == 0:
            logger.info("Attempting to configure existing jackdbus")
            if self._configure_jackdbus(device):
                return True
            else:
                logger.warning("jackdbus configuration failed, stopping it and using jackd")
                self._stop_existing_jack()
        
        # Fall back to starting our own jackd process
        try:
            # Build JACK command
            output_channels = self.config.supercollider.jack_output_channels or self.config.supercollider.output_channels
            jack_cmd = [
                'jackd',
                '-d', 'alsa',
                '-d', device,
                '-r', str(self.config.supercollider.jack_sample_rate),
                '-p', str(self.config.supercollider.jack_buffer_size),
                '-n', str(self.config.supercollider.jack_periods),
                '-P', str(output_channels)  # Set playback (output) channels
            ]
            
            logger.info(f"Starting jackd with command: {' '.join(jack_cmd)}")
            self.record_health_check(
                "jack_startup",
                HealthStatus.HEALTHY,
                f"Initiating jackd startup for device {device}",
                metadata={
                    "device": device,
                    "sample_rate": self.config.supercollider.jack_sample_rate,
                    "buffer_size": self.config.supercollider.jack_buffer_size,
                    "periods": self.config.supercollider.jack_periods
                }
            )
            
            # Start JACK in background
            self.jack_process = subprocess.Popen(
                jack_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for JACK to initialize with timeout
            start_time = time.time()
            timeout = 5.0
            
            while time.time() - start_time < timeout:
                if self._is_jack_running():
                    logger.info(f"JACK started successfully with PID: {self.jack_process.pid}")
                    self.record_health_check(
                        "jack_startup",
                        HealthStatus.HEALTHY,
                        f"JACK started successfully with PID {self.jack_process.pid}",
                        metadata={
                            "pid": self.jack_process.pid,
                            "startup_time": time.time() - start_time,
                            "device": device
                        }
                    )
                    # Give JACK a bit more time to fully initialize
                    await asyncio.sleep(1.0)
                    return True
                await asyncio.sleep(0.1)
                
            # Timeout reached
            logger.error("JACK failed to start within timeout")
            self.record_health_check(
                "jack_startup",
                HealthStatus.ERROR,
                f"JACK failed to start within {timeout}s timeout",
                metadata={"timeout": timeout, "device": device}
            )
            return False
            
        except FileNotFoundError:
            error_msg = "JACK not found - please install jackd"
            logger.error(error_msg)
            self.record_health_check(
                "jack_startup",
                HealthStatus.ERROR,
                error_msg
            )
            return False
        except Exception as e:
            error_msg = f"Error starting JACK: {e}"
            logger.error(error_msg)
            self.record_health_check(
                "jack_startup",
                HealthStatus.ERROR,
                error_msg,
                metadata={"device": device, "error_type": type(e).__name__}
            )
            return False
    
    def _stop_jack(self) -> bool:
        """Stop JACK if we started it.
        
        Returns:
            True if JACK was stopped successfully, False otherwise
        """
        if self.jack_process is None:
            logger.debug("No JACK process to stop (wasn't started by us)")
            return True
            
        try:
            logger.info("Stopping JACK...")
        
            # Try graceful termination first
            self.jack_process.terminate()
            
            try:
                self.jack_process.wait(timeout=3.0)
                logger.info("JACK terminated gracefully")
                self.record_health_check(
                    "jack_shutdown",
                    HealthStatus.HEALTHY,
                    "JACK terminated gracefully"
                )
                self.jack_process = None
                return True
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination failed
                logger.warning("JACK did not terminate gracefully, forcing kill")
                self.jack_process.kill()
                self.jack_process.wait(timeout=1.0)
                logger.info("JACK killed")
                self.record_health_check(
                    "jack_shutdown",
                    HealthStatus.WARNING,
                    "JACK had to be force-killed (did not terminate gracefully)"
                )
                self.jack_process = None
                return True
                
        except Exception as e:
            error_msg = f"Error stopping JACK: {e}"
            logger.error(error_msg)
            self.record_health_check(
                "jack_shutdown",
                HealthStatus.ERROR,
                error_msg,
                metadata={"error_type": type(e).__name__}
            )
            self.jack_process = None  # Clear reference anyway
            return False
    
    def _check_audio_system_health(self) -> Dict[str, Any]:
        """Comprehensive health check of the audio system.
        
        Returns:
            Dict containing status, message, and metadata about audio system health
        """
        metadata = {
            "jack_running": self._is_jack_running(),
            "supercollider_started": self.config.supercollider.auto_start,
            "device_configured": bool(self.config.supercollider.device),
            "surround_enabled": self.config.supercollider.enable_surround,
            "output_channels": self.config.supercollider.output_channels
        }
        
        # Check if JACK is running when expected
        if self.config.supercollider.auto_start_jack and self.config.supercollider.device:
            if not metadata["jack_running"]:
                return {
                    "status": HealthStatus.ERROR,
                    "message": "JACK should be running but is not available",
                    "metadata": metadata
                }
                
        # Check if SuperCollider process is running when expected
        if self.config.supercollider.auto_start:
            if not hasattr(self, 'osc') or not hasattr(self.osc, 'sc_process') or self.osc.sc_process is None:
                return {
                    "status": HealthStatus.ERROR,
                    "message": "SuperCollider should be running but process not found",
                    "metadata": metadata
                }
            elif self.osc.sc_process.poll() is not None:
                return {
                    "status": HealthStatus.ERROR,
                    "message": "SuperCollider process has exited unexpectedly",
                    "metadata": {**metadata, "exit_code": self.osc.sc_process.poll()}
                }
        
        # All checks passed
        status_parts = []
        if metadata["jack_running"]:
            status_parts.append("JACK active")
        if metadata["supercollider_started"]:
            status_parts.append("SuperCollider running")
        if metadata["surround_enabled"]:
            status_parts.append(f"{metadata['output_channels']} channel surround")
            
        message = "Audio system healthy: " + ", ".join(status_parts) if status_parts else "Audio system configured"
        
        return {
            "status": HealthStatus.HEALTHY,
            "message": message,
            "metadata": metadata
        }
    
    @property
    def auto_start_sc(self) -> bool:
        """Backward compatibility property for auto_start_sc."""
        return self.config.supercollider.auto_start
    
    async def start(self):
        """Start the audio service."""
        # Start ZMQ service
        await self.zmq_service.start()
        
        # Set up message handlers
        self.zmq_service.add_message_handler(MessageType.SPACE_TIME_UPDATE, self._handle_space_time_update)
        self.zmq_service.add_message_handler(MessageType.IDLE_STATUS, self._handle_idle_status)
        self.zmq_service.add_message_handler(MessageType.AGENT_CONTROL_EVENT, self._handle_agent_control_event)

        # Resolve SuperCollider script path if auto-start is enabled
        if self.config.supercollider.auto_start:
            # Check relative to this file's directory
            module_dir = Path(__file__).parent.resolve()
            service_dir = module_dir.parent.parent  # Go up from src/experimance_audio
            
            # Try to find the script in the expected sc_scripts directory
            sc_script_dir = service_dir / "sc_scripts"
            script_path = None
            # If no script path provided, look for it in standard locations
            if not self.config.supercollider.script_path:
                # Check relative to this file's directory
                module_dir = Path(__file__).parent.resolve()
                service_dir = module_dir.parent.parent  # Go up from src/experimance_audio
                
                # Try to find the script in the expected sc_scripts directory
                sc_script_dir = service_dir / "sc_scripts"
                default_script = sc_script_dir / "experimance_audio.scd"
                
                if default_script.exists():
                    # Update the config with the found path
                    script_path = default_script
                    logger.info(f"Found SuperCollider script at: {self.config.supercollider.script_path}")
                
            else: # script path set, try to find the file
                script_path = Path(self.config.supercollider.script_path)
                if not script_path.is_absolute():
                    if not script_path.exists():
                        # try under sc_script_dir
                        script_path = sc_script_dir / script_path.name
                        if not script_path.exists():
                            # try under service_dir
                            script_path = service_dir / script_path.name

            if script_path is None:
                logger.warning("No SuperCollider script path provided and couldn't find default script")
                logger.warning("SuperCollider auto-start disabled")
                # Update config to disable auto-start
                self.config.supercollider.auto_start = False
            elif not script_path.exists():
                logger.error(f"SuperCollider script not found at: {script_path}")
                logger.warning("SuperCollider auto-start disabled")
                # Update config to disable auto-start
                self.config.supercollider.auto_start = False

            # Start SuperCollider if we have a script path
            if self.config.supercollider.auto_start and script_path is not None:
                # Check if JACK is already running first
                resolved_device = None
                jack_started = True
                
                if self._is_jack_running():
                    logger.info("JACK is already running, configuring SuperCollider to connect to existing server")
                    self.record_health_check(
                        "audio_system_startup", 
                        HealthStatus.HEALTHY,
                        "JACK already running, SuperCollider will connect to existing server"
                    )
                elif self.config.supercollider.auto_start_jack and self.config.supercollider.device:
                    # Only resolve device if we need to start JACK ourselves
                    resolved_device = self._resolve_audio_device(self.config.supercollider.device)
                    if resolved_device:
                        logger.info(f"Starting JACK for device: {resolved_device}")
                        jack_started = await self._start_jack(resolved_device)
                        if not jack_started:
                            logger.error("Failed to start JACK, continuing without it")
                            self.record_health_check(
                                "audio_system_startup",
                                HealthStatus.WARNING,
                                "JACK startup failed, SuperCollider may use default audio"
                            )
                    else:
                        logger.warning(f"Could not resolve device '{self.config.supercollider.device}', skipping JACK startup")
                        self.record_health_check(
                            "audio_system_startup",
                            HealthStatus.WARNING,
                            f"Device resolution failed for '{self.config.supercollider.device}', skipping JACK startup"
                        )
                
                # Now start SuperCollider
                try:
                    self.tmp_script_path = self._modify_sclang_script(
                        script_path, 
                        temp_path=script_path.with_suffix('.tmp.scd'),
                        resolved_device=resolved_device
                    )
                    if self.osc.start_supercollider(
                        str(self.tmp_script_path), 
                        self.config.supercollider.sclang_path
                    ):
                        logger.info("SuperCollider started successfully")
                        self.record_health_check(
                            "supercollider_startup",
                            HealthStatus.HEALTHY,
                            "SuperCollider started successfully",
                            metadata={
                                "script_path": str(self.tmp_script_path),
                                "jack_enabled": jack_started,
                                "device": self.config.supercollider.device
                            }
                        )
                        # Give SuperCollider a moment to initialize
                        await asyncio.sleep(2)
                    else:
                        logger.error("Failed to start SuperCollider")
                        self.record_health_check(
                            "supercollider_startup",
                            HealthStatus.ERROR,
                            "SuperCollider startup failed"
                        )
                except Exception as e:
                    error_msg = f"Error during SuperCollider startup: {e}"
                    logger.error(error_msg)
                    self.record_health_check(
                        "supercollider_startup",
                        HealthStatus.ERROR,
                        error_msg,
                        metadata={"error_type": type(e).__name__}
                    )
        
        # Call parent start method
        await super().start()
        
        # Record comprehensive audio system health check
        audio_system_status = self._check_audio_system_health()
        self.record_health_check(
            "audio_system_status",
            audio_system_status["status"],
            audio_system_status["message"],
            metadata=audio_system_status["metadata"]
        )
        
        # Record successful startup 
        self.record_health_check(
            "service_startup",
            HealthStatus.HEALTHY,
            "Audio service started successfully"
        )
        logger.info("Audio service started")
    
    async def stop(self):
        """Stop the audio service."""
        logger.info("Stopping audio service...")
        
        # Stop ZMQ service
        if hasattr(self, 'zmq_service'):
            await self.zmq_service.stop()
        
        # Stop SuperCollider if we started it
        if self.auto_start_sc and hasattr(self, 'osc'):
            self.osc.stop_supercollider()
        
        # Stop JACK if we started it
        if hasattr(self, 'jack_process') and self.jack_process is not None:
            self._stop_jack()
        
        # Clean up temporary script file if it exists
        if self.tmp_script_path and self.tmp_script_path.exists():
            try:
                self.tmp_script_path.unlink()
                logger.debug(f"Removed temporary SuperCollider script: {self.tmp_script_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary script file: {e}")
        
        # Call parent stop method (this will automatically clean up all tasks via _clear_tasks)
        await super().stop()
        
        logger.info("Audio service stopped")
    
    async def _handle_space_time_update(self, message_data: MessageDataType):
        """Handle era changed events from the coordinator.
        
        Args:
            message_data: SPACE_TIME_UPDATE event data
        """
        try:
            # Handle both dict and MessageBase types
            try:
                update : SpaceTimeUpdate = SpaceTimeUpdate.to_message_type(message_data)  # type: ignore
            except ValidationError as e:
                self.record_error(
                    ValueError(f"Invalid RenderRequest message: {message_data}"),
                    is_fatal=False,
                    custom_message=f"Invalid RenderRequest message: {message_data}"
                )
                return
        
            logger.info(f"Era changed: {update.era}, biome: {update.biome}, tags: {update.tags}")
            
            # Update state
            self.current_era = update.era
            self.current_biome = update.biome
            
            # Send context to SuperCollider
            self.osc.send_spacetime(update.biome, update.era, update.tags)
            
            # Clear previous tags and send new ones based on context
            self.active_tags.clear()
            
            # Add biome and era as default tags
            self.active_tags.add(update.biome)
            self.active_tags.add(update.era)

            # Include all tags in SuperCollider
            if update.tags:
                for tag in update.tags:
                    self.active_tags.add(tag)
            
            # Include the default tags
            # for tag in self.active_tags:
            #     self.osc.include_tag(tag)
                
            # Signal a transition is happening
            # self.osc.transition(True)
            
            # Schedule transition end after a delay using BaseService task management
            #transition_task = self._end_transition_after_delay(5.0)  # 5 second transition
            #self.add_task(transition_task)
                
        except Exception as e:
            logger.error(f"Error handling era changed event: {e}")
    
    async def _end_transition_after_delay(self, delay_seconds: float):
        """End a transition after a specified delay.
        
        Args:
            delay_seconds: Delay in seconds before ending the transition
        """
        await asyncio.sleep(delay_seconds)
        self.osc.transition(False)
        logger.debug("Transition ended")
    
    async def _handle_idle_status(self, message_data: MessageDataType):
        """Handle idle status events from the coordinator.
        
        Args:
            message_data: IDLE_STATUS event data
        """
        try:
            # Handle both dict and MessageBase types
            if isinstance(message_data, dict):
                data = message_data
            elif isinstance(message_data, MessageBase):
                data = message_data.model_dump()
            else:
                logger.warning(f"Received unexpected message type: {type(message_data)}")
                return
                
            is_idle = data.get("status", False)
            
            logger.info(f"Idle status changed: {is_idle}")
            
            # No specific action needed for now, but could implement audio fade out/in
            # based on idle status in the future
            
        except Exception as e:
            logger.error(f"Error handling idle status event: {e}")
    
    async def _handle_agent_control_event(self, message_data: MessageDataType):
        """Handle agent control events.
        
        Args:
            message_data: AGENT_CONTROL_EVENT data
        """
        try:
            # Handle both dict and MessageBase types
            if isinstance(message_data, dict):
                data = message_data
            elif isinstance(message_data, MessageBase):
                data = message_data.model_dump()
            else:
                logger.warning(f"Received unexpected message type: {type(message_data)}")
                return
                
            # Extract AgentControlEvent data
            sub_type = data.get("sub_type")
            payload = data.get("payload", {})
            
            if not sub_type:
                logger.warning(f"Received agent control event without sub_type: {data}")
                return
                
            logger.debug(f"Agent control event: {sub_type}, payload: {payload}")
            
            # Handle different agent events
            if sub_type == "SpeechDetected":
                # User is speaking, trigger appropriate audio response
                is_speaking = payload.get("status", False)
                self.osc.speaking(is_speaking)
                
            elif sub_type == "ListeningStatus":
                # Agent is listening (or stopped listening)
                is_listening = payload.get("status", False)
                self.osc.listening(is_listening)
                
            elif sub_type == "SuggestBiome":
                # Agent suggested a biome change
                # No immediate audio response needed, will get EraChanged event later
                pass
                
            elif sub_type == "AudiencePresent":
                # Audience presence detected
                is_present = payload.get("status", False)
                if is_present:
                    # Could trigger a subtle audio cue when audience is first detected
                    pass
            
        except Exception as e:
            logger.error(f"Error handling agent control event: {e}")
    
    def add_tag(self, tag: str):
        """Add a tag to the active set and notify SuperCollider.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.active_tags:
            self.active_tags.add(tag)
            self.osc.include_tag(tag)
            logger.debug(f"Added tag: {tag}")
    
    def remove_tag(self, tag: str):
        """Remove a tag from the active tags set.
        
        Args:
            tag: The tag to remove
        """
        if tag in self.active_tags:
            self.active_tags.remove(tag)
            self.osc.exclude_tag(tag)
            logger.debug(f"Removed tag: {tag}")
    
    def set_volume(self, volume_type: str, value: float) -> bool:
        """Set volume level for a specific audio category.
        
        Args:
            volume_type: Type of volume to set ('master', 'environment', 'music', 'sfx')
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set successfully
        """
        # Ensure value is within range
        value = max(0.0, min(1.0, value))
        
        # Set the appropriate volume parameter
        if volume_type == "master":
            self.master_volume = value
            return self.osc.set_master_volume(value)
        elif volume_type == "environment":
            self.environment_volume = value
            return self.osc.set_environment_volume(value)
        elif volume_type == "music":
            self.music_volume = value
            return self.osc.set_music_volume(value)
        elif volume_type == "sfx":
            self.sfx_volume = value
            return self.osc.set_sfx_volume(value)
        else:
            logger.error(f"Unknown volume type: {volume_type}")
            return False
    
    def set_master_volume(self, value: float) -> bool:
        """Set master volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set successfully
        """
        return self.set_volume("master", value)
    
    def set_environment_volume(self, value: float) -> bool:
        """Set environment sounds volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set successfully
        """
        return self.set_volume("environment", value)
    
    def set_music_volume(self, value: float) -> bool:
        """Set music volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set successfully
        """
        return self.set_volume("music", value)
    
    def set_sfx_volume(self, value: float) -> bool:
        """Set sound effects volume level.
        
        Args:
            value: Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if volume was set successfully
        """
        return self.set_volume("sfx", value)
    
    def reload_audio_configs(self):
        """Reload audio configurations and notify SuperCollider."""
        success = self.audio_config.load_configs()
        if success:
            self.osc.reload_configs()
            logger.info("Audio configurations reloaded")
        else:
            logger.error("Failed to reload audio configurations")
    
    def _modify_sclang_script(self, script_path: Path, temp_path: Optional[Path] = None, resolved_device: Optional[str] = None) -> Optional[Path]:
        """
        Modify the SuperCollider script to set specific settings.
        Replaces lines in options and oscRecvPort
        Args:
            script_path: Path to the SuperCollider script file
            temp_path: Optional path for the temporary script file
            resolved_device: Pre-resolved hardware device address (e.g., "hw:4,0")
        """
        if not script_path.exists():
            logger.error(f"SuperCollider script not found at: {script_path}")
            return

        with open(script_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        # Determine if surround is enabled
        output_channels = self.config.supercollider.output_channels or 2
        enable_surround = self.config.supercollider.enable_surround or output_channels > 2
        
        # Track what device we actually use for logging
        device_used_in_script = self.config.supercollider.device
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('s.options.device') and self.config.supercollider.device:
                # If JACK is already running, don't set a device - let SC connect to existing JACK
                if self._is_jack_running():
                    logger.info("JACK is already running, configuring SuperCollider to connect to existing server")
                    new_lines.append('s.options.device = nil;  // Connect to existing JACK server\n')
                    device_used_in_script = "existing_jack_server"
                else:
                    # Use pre-resolved device if available, otherwise resolve now
                    device_to_use = resolved_device if resolved_device else self._resolve_audio_device(self.config.supercollider.device)
                    if device_to_use:
                        device_used_in_script = device_to_use  # Update for logging
                        new_lines.append(f's.options.device = "{device_to_use}";\n')
                        # Set input and output devices explicitly
                        new_lines.append(f's.options.inDevice = "{device_to_use}";\n')
                        new_lines.append(f's.options.outDevice = "{device_to_use}";\n')
                        # Add buffer size for better audio performance
                        new_lines.append(f's.options.hardwareBufferSize = 1024;\n')
                    else:
                        logger.error(f"Could not resolve device '{self.config.supercollider.device}', using original value")
                        new_lines.append(f's.options.device = "{self.config.supercollider.device}";\n')
            elif stripped.startswith('s.options.numOutputBusChannels'):
                new_lines.append(f's.options.numOutputBusChannels = {self.config.supercollider.output_channels};\n')
            elif stripped.startswith('s.options.numInputBusChannels'):
                new_lines.append(f's.options.numInputBusChannels = {self.config.supercollider.input_channels};\n')
            elif stripped.startswith('~oscRecvPort'):
                new_lines.append(f'~oscRecvPort = {self.config.osc.send_port};\n')
            elif stripped.startswith('~initMasterVolume'):
                new_lines.append(f'~initMasterVolume = {self.config.audio.master_volume};\n')
            elif stripped.startswith('~initEnvironmentVolume'):
                new_lines.append(f'~initEnvironmentVolume = {self.config.audio.environment_volume};\n')
            elif stripped.startswith('~initMusicVolume'):
                new_lines.append(f'~initMusicVolume = {self.config.audio.music_volume};\n')
            elif stripped.startswith('~initSfxVolume'):
                new_lines.append(f'~initSfxVolume = {self.config.audio.sfx_volume};\n')
            elif stripped.startswith('~musicFadeTime'):
                new_lines.append(f'~musicFadeTime = {self.config.audio.music_fade_time};\n')
            elif stripped.startswith('~enableSurround'):
                new_lines.append(f'~enableSurround = {str(enable_surround).lower()};\n')
            elif stripped.startswith('~surroundMode'):
                new_lines.append(f'~surroundMode = "{self.config.supercollider.surround_mode}";\n')
            elif stripped.startswith('~environmentChannels'):
                channels_str = str(self.config.supercollider.environment_channels).replace("'", "")
                new_lines.append(f'~environmentChannels = {channels_str};\n')
            elif stripped.startswith('~musicChannels'):
                channels_str = str(self.config.supercollider.music_channels).replace("'", "")
                new_lines.append(f'~musicChannels = {channels_str};\n')
            elif stripped.startswith('~sfxChannels'):
                channels_str = str(self.config.supercollider.sfx_channels).replace("'", "")
                new_lines.append(f'~sfxChannels = {channels_str};\n')
            elif stripped.startswith('~audioDir'):
                new_lines.append(f'~audioDir = "{self.config.audio.audio_dir}";\n')
            elif stripped.startswith('~configDir'):
                new_lines.append(f'~configDir = "{self.config.audio.config_dir}";\n')
            else:
                new_lines.append(line)

        if temp_path is None:
            temp_path = script_path
        
        with open(temp_path, 'w') as f:
            f.writelines(new_lines)
        
        logger.debug(f"Modified SuperCollider script saved to: {temp_path}")
        logger.info(f"Set values:"
                     f" device={device_used_in_script}, "
                     f"output_channels={output_channels}, "
                     f"input_channels={self.config.supercollider.input_channels}, "
                     f"osc_port={self.config.osc.send_port}, "
                     f"surround_enabled={enable_surround}, "
                     f"surround_mode={self.config.supercollider.surround_mode}, "
                     f"env_channels={self.config.supercollider.environment_channels}, "
                     f"music_channels={self.config.supercollider.music_channels}, "
                     f"sfx_channels={self.config.supercollider.sfx_channels}, "
                     f"master_volume={self.config.audio.master_volume}, "
                     f"environment_volume={self.config.audio.environment_volume}, "
                     f"music_volume={self.config.audio.music_volume}, "
                     f"sfx_volume={self.config.audio.sfx_volume}"
        )           

        return temp_path
    
    def check_health(self) -> Dict[str, Any]:
        """Override base class health check to use comprehensive audio system health check.
        
        Returns:
            Dict containing comprehensive audio system health status
        """
        return self._check_audio_system_health()
        
async def run_audio_service(
    config_path: str|Path = DEFAULT_CONFIG_PATH, 
    args:Optional[argparse.Namespace] = None
):
    """
    Run the Experimance Core Service.
    
    Args:
        config_path: Path to configuration file
        args: CLI arguments from argparse (if using new CLI system)
    """
    # Create config with CLI overrides
    config = AudioServiceConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    service = AudioService(config=config)
    
    await service.start()
    await service.run()


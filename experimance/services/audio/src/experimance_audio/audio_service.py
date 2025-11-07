"""
Main Experimance Audio Service implementation.

This service:
1. Subscribes to system events via ZeroMQ (EraChanged, etc.)
2. Communicates with SuperCollider via OSC to control audio playback
3. Manages audio state and configuration
"""

import asyncio
import argparse
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
from experimance_common.logger import get_log_file_path
from experimance_common.zmq.config import MessageDataType
from experimance_common.zmq.services import PubSubService
from experimance_common.schemas import (
    Era, Biome, SpaceTimeUpdate, MessageBase, MessageType
)
from pydantic import ValidationError

from .config import AudioServiceConfig, DEFAULT_CONFIG_PATH
from .config_loader import AudioConfigLoader
from .osc_bridge import OscBridge, DEFAULT_SCLANG_PATH
from experimance_common.logger import setup_logging, get_log_file_path

SERVICE_TYPE = "audio"

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
        super().__init__(service_name=self.config.service_name, service_type=SERVICE_TYPE)

        # Use ZMQ configuration from config, updating the service name to match
        self.zmq_config = self.config.zmq
        self.zmq_config.name = f"{self.config.service_name}-pubsub"
        
        # Create ZMQ service using composition
        self.zmq_service = PubSubService(self.zmq_config)
        
        # Initialize OSC bridge for communication with SuperCollider
        log_path = None
        # Use log path from config if specified, otherwise default to logs/
        if self.config.supercollider.log_path:
            log_path = Path(self.config.supercollider.log_path)
        else:
            log_path = get_log_file_path("supercollider.log")

        self.osc = OscBridge(
            host=self.config.osc.host, 
            port=self.config.osc.send_port,
            log_path=log_path
        )
        
        # Initialize configuration loader
        self.audio_config = AudioConfigLoader(config_dir=self.config.audio.config_dir)
        self.audio_config.load_configs()
        
        # file tracking
        self.tmp_script_path = None  # Temporary script path for SuperCollider

        # State tracking
        self.current_biome = None
        self.current_era = None
        self.active_tags = set()  # Track which tags are active
        
        # Volume settings (initialized from config)
        self.master_volume = self.config.audio.master_volume
        self.environment_volume = self.config.audio.environment_volume
        self.music_volume = self.config.audio.music_volume
        self.sfx_volume = self.config.audio.sfx_volume
    
    def _get_runtime_dir(self) -> str:
        """Get the XDG_RUNTIME_DIR for the current user.
        
        This is needed because:
        1. jackdbus (JACK D-Bus service) communicates via D-Bus
        2. D-Bus requires XDG_RUNTIME_DIR to find the user's session bus
        3. When running as a systemd service, this environment variable may not be set
        4. Without it, jack_control commands fail with "Cannot connect to server socket" errors
        5. The directory contains per-user runtime files like D-Bus sockets
        
        Returns:
            Runtime directory path for the current user (typically /run/user/UID)
        """
        # First check if it's already set in environment
        runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
        if runtime_dir:
            return runtime_dir
        
        # Fallback: dynamically determine based on current user
        # This is safer than hardcoding /run/user/1000 since user IDs can vary
        try:
            import pwd
            current_user = pwd.getpwuid(os.getuid())
            return f'/run/user/{current_user.pw_uid}'
        except Exception as e:
            logger.warning(f"Could not determine runtime directory: {e}")
            # Last resort fallback - but this should rarely be used
            return f'/run/user/{os.getuid()}'
    
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
    
    def _verify_audio_device_exists(self, device: str) -> bool:
        """Verify that an audio device exists and is accessible.
        
        Args:
            device: Hardware device address (e.g., "hw:4,0")
            
        Returns:
            True if device exists and is accessible, False otherwise
        """
        try:
            # Extract card number from device string like "hw:4,0"
            if device.startswith('hw:'):
                card_part = device.split(':')[1].split(',')[0]
                try:
                    card_num = int(card_part)
                    # Check if the card exists in /proc/asound/cards
                    with open('/proc/asound/cards', 'r') as f:
                        cards_content = f.read()
                        if f' {card_num} [' in cards_content:
                            logger.debug(f"Device {device} verified as accessible (card {card_num} exists)")
                            return True
                        else:
                            logger.warning(f"Device {device} not found (card {card_num} does not exist)")
                            return False
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse card number from device {device}")
                    return False
            else:
                logger.warning(f"Device {device} not in expected hw:X,Y format")
                return False
                
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Could not access /proc/asound/cards to verify device {device}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error verifying device {device}: {e}")
            return False
    
    def _is_jack_running(self) -> bool:
        """Check if JACK is currently running via jackdbus.
        
        Returns:
            True if JACK is running, False otherwise
        """
        try:
            # Ensure proper environment for jackdbus commands
            env = os.environ.copy()
            if 'XDG_RUNTIME_DIR' not in env:
                env['XDG_RUNTIME_DIR'] = self._get_runtime_dir()
            
            # Use jack_control status to check if JACK is started
            status_result = subprocess.run(['jack_control', 'status'], capture_output=True, text=True, timeout=3.0, env=env)
            if status_result.returncode == 0:
                # Check if status contains "started" 
                return 'started' in status_result.stdout.lower()
            return False
        except Exception as e:
            logger.debug(f"Error checking JACK status: {e}")
            return False
    
    def _stop_jack(self) -> bool:
        """Stop JACK via jackdbus if it's running.
        
        Returns:
            True if JACK was stopped successfully, False otherwise
        """
        try:
            # Ensure proper environment for jackdbus commands
            env = os.environ.copy()
            if 'XDG_RUNTIME_DIR' not in env:
                env['XDG_RUNTIME_DIR'] = self._get_runtime_dir()
            
            # Check if JACK is running first
            if not self._is_jack_running():
                logger.debug("JACK is not running, nothing to stop")
                return True
            
            logger.info("Stopping JACK via jackdbus")
            
            # Try to stop jackdbus gracefully using jack_control
            result = subprocess.run(['jack_control', 'stop'], capture_output=True, text=True, timeout=5.0, env=env)
            if result.returncode == 0:
                logger.info("JACK stopped successfully")
                self.record_health_check(
                    "jack_shutdown",
                    HealthStatus.HEALTHY,
                    "JACK stopped gracefully via jackdbus"
                )
                return True
            else:
                logger.warning(f"jack_control stop failed: {result.stderr}")
                self.record_health_check(
                    "jack_shutdown",
                    HealthStatus.WARNING,
                    f"jack_control stop failed: {result.stderr}"
                )
                return False
                
        except Exception as e:
            error_msg = f"Error stopping JACK: {e}"
            logger.error(error_msg)
            self.record_health_check(
                "jack_shutdown",
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
            
            # Ensure proper environment for jackdbus (critical for systemd services)
            # jackdbus requires XDG_RUNTIME_DIR to communicate with the D-Bus session bus.
            # Without this environment variable, jack_control commands will fail with
            # "Cannot connect to server socket" errors. This commonly happens when
            # running services via systemd where the user session environment isn't inherited.
            env = os.environ.copy()
            if 'XDG_RUNTIME_DIR' not in env:
                env['XDG_RUNTIME_DIR'] = self._get_runtime_dir()
            
            # Verify device exists before trying to configure it
            if not self._verify_audio_device_exists(device):
                logger.error(f"Audio device {device} not found, cannot configure JACK")
                return False
            
            # Stop JACK if it's running before making parameter changes
            # This is required because jackdbus parameter changes only take effect after restart
            if self._is_jack_running():
                logger.debug("Stopping JACK before configuration changes")
                stop_result = subprocess.run(['jack_control', 'stop'], capture_output=True, text=True, timeout=5.0, env=env)
                if stop_result.returncode != 0:
                    logger.warning(f"Failed to stop JACK before configuration: {stop_result.stderr}")
                    # Continue anyway - might work
            
            # Configure jackdbus with our settings
            # These parameter changes will only take effect when JACK is next started
            output_channels = self.config.supercollider.jack_output_channels or self.config.supercollider.output_channels
            commands = [
                ['jack_control', 'ds', 'alsa'],  # Set driver to alsa
                ['jack_control', 'dps', 'device', device],  # Set device
                ['jack_control', 'dps', 'playback', device],  # Set playback device to the same
                ['jack_control', 'dps', 'rate', str(self.config.supercollider.jack_sample_rate)],  # Set sample rate
                ['jack_control', 'dps', 'period', str(self.config.supercollider.jack_buffer_size)],  # Set buffer size
                ['jack_control', 'dps', 'nperiods', str(self.config.supercollider.jack_periods)],  # Set periods
                ['jack_control', 'dps', 'outchannels', str(output_channels)],  # Set output channels
            ]
            
            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0, env=env)
                if result.returncode != 0:
                    logger.warning(f"Command {' '.join(cmd)} failed: {result.stderr}")
                    return False
            
            # Start jackdbus
            result = subprocess.run(['jack_control', 'start'], capture_output=True, text=True, timeout=10.0, env=env)
            if result.returncode != 0:
                logger.error(f"Failed to start jackdbus: {result.stderr}")
                return False
            
            # Wait for it to start
            start_time = time.time()
            timeout = 5.0
            while time.time() - start_time < timeout:
                status_result = subprocess.run(['jack_control', 'status'], capture_output=True, text=True, env=env)
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

    def _verify_jack_configuration(self, expected_device: str) -> bool:
        """Verify that the current JACK configuration matches our requirements.
        
        Args:
            expected_device: The device we expect JACK to be using (e.g., "hw:4,0")
            
        Returns:
            True if configuration matches, False otherwise
        """
        try:
            # Ensure proper environment for jackdbus check
            # jackdbus communicates via D-Bus, which needs XDG_RUNTIME_DIR to locate
            # the user session bus socket. Without this, jack_control commands fail.
            # This is critical when running as systemd services where environment
            # variables may not be properly inherited.
            env = os.environ.copy()
            if 'XDG_RUNTIME_DIR' not in env:
                env['XDG_RUNTIME_DIR'] = self._get_runtime_dir()
            
            # Get current JACK driver parameters
            dp_result = subprocess.run(['jack_control', 'dp'], capture_output=True, text=True, timeout=3.0, env=env)
            if dp_result.returncode != 0:
                logger.debug("Could not get JACK driver parameters")
                return False
            
            config_output = dp_result.stdout
            
            # Parse the configuration output to check key parameters
            expected_output_channels = self.config.supercollider.jack_output_channels or self.config.supercollider.output_channels
            expected_sample_rate = self.config.supercollider.jack_sample_rate
            expected_buffer_size = self.config.supercollider.jack_buffer_size
            
            # Check device
            device_match = re.search(r'device:.*?:([^)]+)', config_output)
            current_device = device_match.group(1) if device_match else None
            
            # Check output channels
            outchannels_match = re.search(r'outchannels:.*?:(\d+)', config_output)
            current_output_channels = int(outchannels_match.group(1)) if outchannels_match else None
            
            # Check sample rate
            rate_match = re.search(r'rate:.*?:(\d+)', config_output)
            current_sample_rate = int(rate_match.group(1)) if rate_match else None
            
            # Check buffer size (period)
            period_match = re.search(r'period:.*?:(\d+)', config_output)
            current_buffer_size = int(period_match.group(1)) if period_match else None
            
            # Log current vs expected configuration
            logger.debug(f"JACK config verification:")
            logger.debug(f"  Device: current='{current_device}', expected='{expected_device}'")
            logger.debug(f"  Output channels: current={current_output_channels}, expected={expected_output_channels}")
            logger.debug(f"  Sample rate: current={current_sample_rate}, expected={expected_sample_rate}")
            logger.debug(f"  Buffer size: current={current_buffer_size}, expected={expected_buffer_size}")
            
            # Check if all parameters match
            config_matches = (
                current_device == expected_device and
                current_output_channels == expected_output_channels and
                current_sample_rate == expected_sample_rate and
                current_buffer_size == expected_buffer_size
            )
            
            if config_matches:
                logger.info("JACK configuration matches our requirements")
                self.record_health_check(
                    "jack_config_verification",
                    HealthStatus.HEALTHY,
                    "JACK configuration matches requirements",
                    metadata={
                        "device": current_device,
                        "output_channels": current_output_channels,
                        "sample_rate": current_sample_rate,
                        "buffer_size": current_buffer_size
                    }
                )
                return True
            else:
                logger.info("JACK configuration does not match our requirements, will reconfigure")
                self.record_health_check(
                    "jack_config_verification",
                    HealthStatus.WARNING,
                    "JACK configuration mismatch detected",
                    metadata={
                        "current_device": current_device,
                        "expected_device": expected_device,
                        "current_output_channels": current_output_channels,
                        "expected_output_channels": expected_output_channels,
                        "current_sample_rate": current_sample_rate,
                        "expected_sample_rate": expected_sample_rate,
                        "current_buffer_size": current_buffer_size,
                        "expected_buffer_size": expected_buffer_size
                    }
                )
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Could not verify JACK configuration: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error verifying JACK configuration: {e}")
            return False

    def _check_pipewire_device_conflict(self, device: str) -> bool:
        """Check if PipeWire is holding exclusive access to our target device.
        
        Args:
            device: Hardware device address (e.g., "hw:4,0")
            
        Returns:
            True if PipeWire is blocking access to the device, False otherwise
        """
        try:
            # Extract card number from device string like "hw:4,0"
            if device.startswith('hw:'):
                card_part = device.split(':')[1].split(',')[0]
                card_num = int(card_part)
                
                # Check if fuser shows any processes using the control device
                result = subprocess.run(['fuser', '-v', f'/dev/snd/controlC{card_num}'], 
                                      capture_output=True, text=True, timeout=3.0)
                
                if result.returncode == 0 and 'wireplumber' in result.stderr:
                    logger.warning(f"PipeWire/WirePlumber is holding exclusive access to {device}")
                    return True
                    
        except Exception as e:
            logger.debug(f"Error checking PipeWire device conflict: {e}")
            
        return False

    def _attempt_pipewire_device_release(self, device: str) -> bool:
        """Attempt to release a specific device from PipeWire control while keeping PipeWire running.
        
        This method tries several approaches to release just the target device without
        disrupting other audio devices (like the Yealink speakerphone for the AI agent).
        
        Args:
            device: Hardware device address (e.g., "hw:4,0")
            
        Returns:
            True if device was released successfully, False otherwise
        """
        try:
            if not self._check_pipewire_device_conflict(device):
                return True  # No conflict to resolve
                
            logger.info(f"Attempting to release {device} from PipeWire control (keeping PipeWire running)")
            
            # Extract card number from device string like "hw:4,0"
            if device.startswith('hw:'):
                card_part = device.split(':')[1].split(',')[0]
                card_num = int(card_part)
            else:
                logger.warning(f"Cannot extract card number from device {device}")
                return False
            
            # Method 1: Try to use pw-cli to suspend the specific device
            try:
                # First, list devices to find the PipeWire node ID
                list_result = subprocess.run(['pw-cli', 'list-objects'], 
                                           capture_output=True, text=True, timeout=5.0)
                if list_result.returncode == 0:
                    # Look for our specific card in the output
                    lines = list_result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if f'alsa_card.usb-' in line and f'card{card_num}' in line:
                            # Try to extract the node ID and suspend it
                            node_match = re.search(r'id (\d+)', line)
                            if node_match:
                                node_id = node_match.group(1)
                                logger.debug(f"Found PipeWire node {node_id} for card {card_num}")
                                
                                # Suspend the device node
                                suspend_result = subprocess.run(['pw-cli', 'suspend-node', node_id], 
                                                              capture_output=True, text=True, timeout=3.0)
                                if suspend_result.returncode == 0:
                                    logger.info(f"Suspended PipeWire node {node_id} for device {device}")
                                    time.sleep(0.5)  # Brief pause
                                    
                                    # Check if conflict is resolved
                                    if not self._check_pipewire_device_conflict(device):
                                        logger.info(f"Successfully released {device} from PipeWire via node suspension")
                                        self.record_health_check(
                                            "pipewire_device_release",
                                            HealthStatus.HEALTHY,
                                            f"Released device {device} from PipeWire via node suspension",
                                            metadata={"device": device, "method": "node_suspension", "node_id": node_id}
                                        )
                                        return True
                                break
            except Exception as e:
                logger.debug(f"pw-cli method failed: {e}")
            
            # Method 2: Try to kill just the WirePlumber process handling this device
            try:
                # Use lsof to find processes using the specific control device
                lsof_result = subprocess.run(['lsof', f'/dev/snd/controlC{card_num}'], 
                                           capture_output=True, text=True, timeout=3.0)
                if lsof_result.returncode == 0:
                    for line in lsof_result.stdout.split('\n')[1:]:  # Skip header
                        if 'wireplumber' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                pid = parts[1]
                                logger.debug(f"Found WirePlumber PID {pid} using device {device}")
                                
                                # Send SIGUSR1 to WirePlumber to ask it to release the device
                                # This is gentler than killing the process
                                try:
                                    subprocess.run(['kill', '-USR1', pid], timeout=2.0)
                                    time.sleep(0.5)
                                    
                                    if not self._check_pipewire_device_conflict(device):
                                        logger.info(f"Successfully released {device} via WirePlumber signal")
                                        self.record_health_check(
                                            "pipewire_device_release",
                                            HealthStatus.HEALTHY,
                                            f"Released device {device} via WirePlumber signal",
                                            metadata={"device": device, "method": "wireplumber_signal", "pid": pid}
                                        )
                                        return True
                                except Exception as e:
                                    logger.debug(f"Signal method failed: {e}")
            except Exception as e:
                logger.debug(f"lsof method failed: {e}")
            
            # Method 3: Fallback - temporarily stop only WirePlumber (not all of PipeWire)
            # This preserves other PipeWire functionality while releasing device control
            logger.info("Trying fallback: temporary WirePlumber restart")
            try:
                # Stop just WirePlumber
                stop_result = subprocess.run(['systemctl', '--user', 'stop', 'wireplumber'], 
                                           capture_output=True, text=True, timeout=5.0)
                if stop_result.returncode == 0:
                    time.sleep(1.0)  # Give it time to release the device
                    
                    # Check if device is now free
                    if not self._check_pipewire_device_conflict(device):
                        logger.info(f"Successfully released {device} by stopping WirePlumber")
                        
                        # Restart WirePlumber to restore functionality for other devices
                        restart_result = subprocess.run(['systemctl', '--user', 'start', 'wireplumber'], 
                                                      capture_output=True, text=True, timeout=5.0)
                        if restart_result.returncode == 0:
                            logger.debug("WirePlumber restarted successfully")
                        else:
                            logger.warning("WirePlumber restart failed, but device was released")
                        
                        self.record_health_check(
                            "pipewire_device_release",
                            HealthStatus.HEALTHY,
                            f"Released device {device} via WirePlumber restart",
                            metadata={"device": device, "method": "wireplumber_restart"}
                        )
                        return True
                    else:
                        # Device still conflicted, restart WirePlumber anyway
                        subprocess.run(['systemctl', '--user', 'start', 'wireplumber'], 
                                     capture_output=True, text=True, timeout=5.0)
                        logger.warning(f"WirePlumber restart did not resolve conflict for {device}")
                        
            except Exception as e:
                logger.debug(f"WirePlumber restart method failed: {e}")
            
            # All methods failed
            logger.warning(f"Could not release {device} from PipeWire control using any method")
            self.record_health_check(
                "pipewire_device_release",
                HealthStatus.WARNING,
                f"Failed to release device {device} from PipeWire control",
                metadata={"device": device, "attempted_methods": ["node_suspension", "wireplumber_signal", "wireplumber_restart"]}
            )
            return False
                
        except Exception as e:
            logger.error(f"Error attempting to release device from PipeWire: {e}")
            self.record_health_check(
                "pipewire_device_release",
                HealthStatus.ERROR,
                f"Error during device release: {e}",
                metadata={"device": device, "error_type": type(e).__name__}
            )
            return False

    async def _start_jack(self, device: str) -> bool:
        """Start or configure JACK via jackdbus with the specified device.
        
        Note: jackdbus parameter changes only take effect after stopping and restarting 
        the JACK server. The _configure_jackdbus method handles this automatically.
        
        Args:
            device: Hardware device address (e.g., "hw:4,0")
            
        Returns:
            True if JACK started successfully, False otherwise
        """
        # Check if JACK is already running with correct configuration
        if self._is_jack_running():
            logger.info("JACK is already running")
            if self._verify_jack_configuration(device):
                logger.info("JACK is running with correct configuration, will use existing server")
                self.record_health_check(
                    "jack_startup",
                    HealthStatus.HEALTHY,
                    "Using existing jackdbus server with correct configuration",
                    metadata={"device": device}
                )
                return True
            else:
                logger.info("JACK is running but configuration doesn't match, will reconfigure")
                # Stop and reconfigure
                if not self._stop_jack():
                    logger.warning("Could not stop existing JACK for reconfiguration")
                    return False
        
        # Check for PipeWire conflicts and attempt to release the device if needed
        if self._check_pipewire_device_conflict(device):
            logger.info(f"PipeWire conflict detected for {device}, attempting to release")
            if not self._attempt_pipewire_device_release(device):
                logger.error(f"Failed to release {device} from PipeWire control")
                self.record_health_check(
                    "jack_startup",
                    HealthStatus.ERROR,
                    f"PipeWire blocking access to {device}, could not release",
                    metadata={"device": device}
                )
                return False
        
        # Configure jackdbus (this will auto-start it via D-Bus if needed)
        logger.info(f"Configuring jackdbus for device: {device}")
        if self._configure_jackdbus(device):
            logger.info("JACK configured and started successfully via jackdbus")
            self.record_health_check(
                "jack_startup",
                HealthStatus.HEALTHY,
                f"JACK started successfully via jackdbus for device {device}",
                metadata={"device": device, "method": "jackdbus"}
            )
            return True
        else:
            logger.error("Failed to configure jackdbus")
            self.record_health_check(
                "jack_startup",
                HealthStatus.ERROR,
                "jackdbus configuration failed",
                metadata={"device": device}
            )
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
        self.zmq_service.add_message_handler(MessageType.PRESENCE_STATUS, self._handle_presence_status)
        self.zmq_service.add_message_handler(MessageType.SPEECH_DETECTED, self._handle_speech_detected)
        self.zmq_service.add_message_handler(MessageType.CHANGE_MAP, self._handle_change_map)

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
                    logger.info("JACK is already running")
                    # Verify that JACK configuration matches our requirements
                    if self.config.supercollider.device:
                        resolved_device = self._resolve_audio_device(self.config.supercollider.device)
                        if resolved_device and not self._verify_jack_configuration(resolved_device):
                            logger.warning("JACK configuration doesn't match requirements, reconfiguring...")
                            # Stop existing JACK and restart with correct configuration
                            if self._stop_jack():
                                jack_started = await self._start_jack(resolved_device)
                                if not jack_started:
                                    logger.error("Failed to restart JACK with correct configuration")
                                    self.record_health_check(
                                        "audio_system_startup",
                                        HealthStatus.WARNING,
                                        "JACK reconfiguration failed, continuing with existing server"
                                    )
                            else:
                                logger.warning("Could not stop existing JACK for reconfiguration")
                                self.record_health_check(
                                    "audio_system_startup",
                                    HealthStatus.WARNING,
                                    "JACK configuration mismatch but could not restart"
                                )
                        else:
                            logger.info("JACK configuration verified, will connect SuperCollider to existing server")
                            self.record_health_check(
                                "audio_system_startup", 
                                HealthStatus.HEALTHY,
                                "JACK configuration verified, SuperCollider will connect to existing server"
                            )
                    else:
                        logger.info("No specific device configured, will connect SuperCollider to existing JACK server")
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
                        self.config.supercollider.sclang_path,
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
        
        # Stop JACK if needed (only via jackdbus)
        # Note: We no longer manage jackd processes directly
        
        # Clean up temporary script file if it exists
        if False and self.tmp_script_path and self.tmp_script_path.exists():
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
            
            # end the transition sound (played during the change map)
            self.osc.transition(False)
                
        except Exception as e:
            logger.error(f"Error handling era changed event: {e}")
    

    async def _handle_change_map(self, message_data: MessageDataType):
        """Handle change map events from the coordinator.
        
        Args:
            message_data: CHANGE_MAP event data
        """
        # we don't care whats in the message, it just starts the transition sound (that ends on the next spacetime update)
        # technically this isn't the transition sound, but we're hijacking the transition sound to play during the change map
        self.osc.transition(True)

    async def _end_transition_after_delay(self, delay_seconds: float):
        """End a transition after a specified delay.
        
        Args:
            delay_seconds: Delay in seconds before ending the transition
        """
        await asyncio.sleep(delay_seconds)
        self.osc.transition(False)
        logger.debug("Transition ended")
    
    async def _handle_presence_status(self, message_data: MessageDataType):
        """Handle idle status events from the coordinator.
        
        Args:
            message_data: PRESENCE_STATUS event data
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
            
            logger.debug(f"Idle status changed: {is_idle}")
            
            # No specific action needed for now, but could implement audio fade out/in
            # based on idle status in the future
            
        except Exception as e:
            logger.error(f"Error handling idle status event: {e}")
    
    async def _handle_speech_detected(self, message_data: MessageDataType):
        """Handle speech detected events.

        Args:
            message_data: SPEECH_DETECTED data
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
                
            # Extract SpeechDetected data
            sub_type = data.get("sub_type")
            payload = data.get("payload", {})
            
            if not sub_type:
                logger.warning(f"Received speech detected event without sub_type: {data}")
                return

            logger.debug(f"Speech detected event: {sub_type}, payload: {payload}")

            # Handle different agent events
            if sub_type == "SpeechDetected":
                # User is speaking, trigger appropriate audio response
                is_speaking = payload.get("status", False)
                self.osc.speaking(is_speaking)
                
            elif sub_type == "ListeningStatus":
                # Agent is listening (or stopped listening)
                is_listening = payload.get("status", False)
                self.osc.listening(is_listening)
                
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
                # If JACK is already running, set device to nil to connect to existing JACK server
                # According to SuperCollider docs: nil device = "default:SuperCollider" 
                # which connects to the default JACK server with client name "SuperCollider"
                if self._is_jack_running():
                    logger.info("JACK is already running, SuperCollider will connect as JACK client")
                    new_lines.append('s.options.device = nil;  // Connect to existing JACK server as client\n')
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


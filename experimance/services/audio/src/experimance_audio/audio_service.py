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
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.base_service import BaseService, ServiceStatus
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

        # State tracking
        self.current_biome = None
        self.current_era = None
        self.active_tags = set()  # Track which tags are active
        
        # Volume settings (initialized from config)
        self.master_volume = self.config.audio.master_volume
        self.environment_volume = self.config.audio.environment_volume
        self.music_volume = self.config.audio.music_volume
        self.sfx_volume = self.config.audio.sfx_volume
    
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
                self.tmp_script_path = self._modify_sclang_script(script_path, temp_path=script_path.with_suffix('.tmp.scd'))
                if self.osc.start_supercollider(
                    str(self.tmp_script_path), 
                    self.config.supercollider.sclang_path
                ):
                    logger.info("SuperCollider started successfully")
                    # Give SuperCollider a moment to initialize
                    await asyncio.sleep(2)
                else:
                    logger.error("Failed to start SuperCollider")
        
        # Call parent start method
        await super().start()
        
        self.status = ServiceStatus.HEALTHY
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
            
            # Include the default tags
            for tag in self.active_tags:
                self.osc.include_tag(tag)
                
            # Signal a transition is happening
            self.osc.transition(True)
            
            # Schedule transition end after a delay using BaseService task management
            transition_task = self._end_transition_after_delay(5.0)  # 5 second transition
            self.add_task(transition_task)
                
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
    
    def _modify_sclang_script(self, script_path: Path, temp_path: Optional[Path] = None) -> Optional[Path]:
        """
        Modify the SuperCollider script to set specific settings.
        Replaces lines in options and oscRecvPort
        Args:
            script_path: Path to the SuperCollider script file
        """
        if not script_path.exists():
            logger.error(f"SuperCollider script not found at: {script_path}")
            return

        with open(script_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('s.options.device') and self.config.supercollider.device:
                new_lines.append(f's.options.device = "{self.config.supercollider.device}";\n')
            elif stripped.startswith('s.options.numOutputBusChannels'):
                new_lines.append(f's.options.numOutputBusChannels = {self.config.supercollider.output_channels};\n')
            elif stripped.startswith('s.options.numInputBusChannels'):
                new_lines.append(f's.options.numInputBusChannels = {self.config.supercollider.input_channels};\n')
            elif stripped.startswith('~oscRecvPort'):
                new_lines.append(f'~oscRecvPort = {self.config.osc.recv_port};\n')
            elif stripped.startswith('~initMasterVolume'):
                new_lines.append(f'~initMasterVolume = {self.config.audio.master_volume};\n')
            elif stripped.startswith('~initEnvironmentVolume'):
                new_lines.append(f'~initEnvironmentVolume = {self.config.audio.environment_volume};\n')
            elif stripped.startswith('~initMusicVolume'):
                new_lines.append(f'~initMusicVolume = {self.config.audio.music_volume};\n')
            elif stripped.startswith('~initSfxVolume'):
                new_lines.append(f'~initSfxVolume = {self.config.audio.sfx_volume};\n')
            else:
                new_lines.append(line)

        if temp_path is None:
            temp_path = script_path
        
        with open(temp_path, 'w') as f:
            f.writelines(new_lines)
        
        return temp_path
        
async def run_audio_service(
    config_path: str = DEFAULT_CONFIG_PATH, 
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


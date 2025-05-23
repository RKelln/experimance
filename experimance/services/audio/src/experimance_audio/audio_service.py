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
from experimance_common.service import ZmqSubscriberService
from experimance_common.schemas import Era, Biome, EraChanged, AgentControlEvent, IdleStatus
from experimance_common.zmq_utils import MessageType, ZmqTimeoutError

from .config_loader import AudioConfigLoader
from .osc_bridge import OscBridge, DEFAULT_SCLANG_PATH

# Configure logging
logger = logging.getLogger(__name__)

# Constants
EVENTS_CHANNEL_PORT = DEFAULT_PORTS["coordinator_pub"]  
AGENT_CTRL_CHANNEL_PORT = DEFAULT_PORTS["agent_pub"]    

class AudioService(ZmqSubscriberService):
    """
    Audio Service for the Experimance interactive installation.
    
    This service subscribes to system events and controls the SuperCollider
    audio engine via OSC based on the current state of the installation.
    """
    
    def __init__(self, 
                service_name: str = "audio-service",
                config_dir: Optional[str] = None,
                osc_host: str = "localhost",
                osc_port: int = DEFAULT_PORTS["audio_osc_send_port"],
                auto_start_sc: bool = True,
                sc_script_path: Optional[str] = None,
                sclang_path: str = DEFAULT_SCLANG_PATH):
        """Initialize the audio service.
        
        Args:
            service_name: Name of this service instance
            config_dir: Directory containing audio configuration files
            osc_host: SuperCollider host address
            osc_port: SuperCollider OSC listening port
            auto_start_sc: Whether to automatically start SuperCollider
            sc_script_path: Path to SuperCollider script to execute
            sclang_path: Path to SuperCollider language interpreter executable
        """
        # Initialize base service with subscription to both events and agent_ctrl channels
        super().__init__(
            service_name=service_name,
            sub_address=f"tcp://localhost:{EVENTS_CHANNEL_PORT}",
            topics=[
                MessageType.ERA_CHANGED,  # Subscribe to era changes
                MessageType.IDLE_STATUS,  # Subscribe to idle status changes
            ],
            service_type="audio"
        )
        
        # Additional subscriber for agent control events on the agent_ctrl channel
        self.agent_subscriber = None
        self.agent_sub_address = f"tcp://localhost:{AGENT_CTRL_CHANNEL_PORT}"
        
        # Initialize OSC bridge for communication with SuperCollider
        self.osc = OscBridge(host=osc_host, port=osc_port)
        
        # Store SuperCollider startup parameters
        self.auto_start_sc = auto_start_sc
        self.sc_script_path = sc_script_path
        self.sclang_path = sclang_path
        
        # Initialize configuration loader
        self.config = AudioConfigLoader(config_dir=config_dir)
        self.config.load_configs()
        
        # State tracking
        self.current_biome = None
        self.current_era = None
        self.active_tags = set()  # Track which tags are active
        
    async def start(self):
        """Start the audio service."""
        # Initialize the primary subscriber
        self.subscriber = None  # Reset in case of restart
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        await super().start()
        
        # Resolve SuperCollider script path if auto-start is enabled
        if self.auto_start_sc:
            # If no script path provided, look for it in standard locations
            if not self.sc_script_path:
                # Check relative to this file's directory
                module_dir = Path(__file__).parent.resolve()
                service_dir = module_dir.parent.parent  # Go up from src/experimance_audio
                
                # Try to find the script in the expected sc_scripts directory
                sc_script_dir = service_dir / "sc_scripts"
                default_script = sc_script_dir / "experimance_audio.scd"
                
                if default_script.exists():
                    self.sc_script_path = str(default_script)
                    logger.info(f"Found SuperCollider script at: {self.sc_script_path}")
                else:
                    logger.warning("No SuperCollider script path provided and couldn't find default script")
                    logger.warning("SuperCollider auto-start disabled")
                    self.auto_start_sc = False
            
            # Start SuperCollider if we have a script path
            if self.auto_start_sc and self.sc_script_path:
                if self.osc.start_supercollider(self.sc_script_path, self.sclang_path):
                    logger.info("SuperCollider started successfully")
                    # Give SuperCollider a moment to initialize
                    await asyncio.sleep(2)
                else:
                    logger.error("Failed to start SuperCollider")
        
        # Define wrapper functions for async handlers to work with the register_handler method
        def era_changed_wrapper(message):
            asyncio.create_task(self._handle_era_changed(message))
            
        def idle_status_wrapper(message):
            asyncio.create_task(self._handle_idle_status(message))
        
        # Register handler for EraChanged events
        self.register_handler(MessageType.ERA_CHANGED, era_changed_wrapper)
        
        # Register handler for Idle events
        self.register_handler(MessageType.IDLE_STATUS, idle_status_wrapper)
        
        # Initialize the agent control subscriber
        logger.info(f"Initializing agent control subscriber on {self.agent_sub_address}")
        from experimance_common.zmq_utils import ZmqSubscriber
        self.agent_subscriber = ZmqSubscriber(self.agent_sub_address, [MessageType.AGENT_CONTROL_EVENT])
        self.register_socket(self.agent_subscriber)
        
        # Register additional task to listen for agent control events
        self._register_task(self._listen_for_agent_events())
        
        logger.info("Audio service started")
        
    async def _listen_for_agent_events(self):
        """Listen for agent control events on the agent_ctrl channel."""
        if not self.agent_subscriber:
            logger.error("Agent subscriber not initialized")
            return
            
        while self.running:
            try:
                topic, message = await self.agent_subscriber.receive_async()
                logger.debug(f"Received agent message: {message}")
                
                # Process the agent control event
                await self._handle_agent_control_event(message)
                
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error receiving agent control event: {e}")
                
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
    
    async def _handle_era_changed(self, message: Dict[str, Any]):
        """Handle era changed events from the coordinator.
        
        Args:
            message: ERA_CHANGED event data
        """
        try:
            era = message.get("era")
            biome = message.get("biome")
            
            if not era or not biome:
                logger.warning(f"Received incomplete EraChanged event: {message}")
                return
                
            logger.info(f"Era changed: {era}, biome: {biome}")
            
            # Update state
            self.current_era = era
            self.current_biome = biome
            
            # Send context to SuperCollider
            self.osc.send_spacetime(biome, era)
            
            # Clear previous tags and send new ones based on context
            self.active_tags.clear()
            
            # Add biome and era as default tags
            self.active_tags.add(biome)
            self.active_tags.add(era)
            
            # Include the default tags
            for tag in self.active_tags:
                self.osc.include_tag(tag)
                
            # Signal a transition is happening
            self.osc.transition(True)
            
            # Schedule transition end after a delay
            asyncio.create_task(self._end_transition_after_delay(5.0))  # 5 second transition
                
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
    
    async def _handle_idle_status(self, message: Dict[str, Any]):
        """Handle idle status events from the coordinator.
        
        Args:
            message: IDLE_STATUS event data
        """
        try:
            is_idle = message.get("status", False)
            
            logger.info(f"Idle status changed: {is_idle}")
            
            # No specific action needed for now, but could implement audio fade out/in
            # based on idle status in the future
            
        except Exception as e:
            logger.error(f"Error handling idle status event: {e}")
    
    async def _handle_agent_control_event(self, message: Dict[str, Any]):
        """Handle agent control events.
        
        Args:
            message: AGENT_CONTROL_EVENT data
        """
        try:
            sub_type = message.get("sub_type")
            payload = message.get("payload", {})
            
            if not sub_type:
                logger.warning(f"Received agent control event without sub_type: {message}")
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
        """Remove a tag from the active set and notify SuperCollider.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.active_tags:
            self.active_tags.remove(tag)
            self.osc.exclude_tag(tag)
            logger.debug(f"Removed tag: {tag}")
    
    def reload_audio_configs(self):
        """Reload audio configurations and notify SuperCollider."""
        success = self.config.load_configs()
        if success:
            self.osc.reload_configs()
            logger.info("Audio configurations reloaded")
        else:
            logger.error("Failed to reload audio configurations")
    
    async def stop(self):
        """Stop the audio service and clean up resources."""
        logger.info("Stopping audio service")
        
        # Stop SuperCollider if we started it
        if self.auto_start_sc and hasattr(self.osc, 'sc_process') and self.osc.sc_process is not None:
            logger.info("Stopping SuperCollider")
            success = self.osc.stop_supercollider()
            if success:
                logger.info("SuperCollider stopped successfully")
            else:
                logger.warning("Failed to stop SuperCollider gracefully")
        
        # Stop the base service (handles ZMQ socket cleanup)
        await super().stop()
        logger.info("Audio service stopped")


async def run_audio_service(config_dir: Optional[str] = None,
                           osc_host: str = "localhost", 
                           osc_port: int = 57120,
                           auto_start_sc: bool = True,
                           sc_script_path: Optional[str] = None,
                           sclang_path: str = DEFAULT_SCLANG_PATH):
    """Run the audio service.
    
    Args:
        config_dir: Directory containing audio configuration files
        osc_host: SuperCollider host address
        osc_port: SuperCollider OSC listening port
        auto_start_sc: Whether to automatically start SuperCollider
        sc_script_path: Path to SuperCollider script to execute
        sclang_path: Path to SuperCollider language interpreter executable
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create and start the service
    service = AudioService(
        config_dir=config_dir,
        osc_host=osc_host,
        osc_port=osc_port,
        auto_start_sc=auto_start_sc,
        sc_script_path=sc_script_path,
        sclang_path=sclang_path
    )
    
    await service.start()
    logger.info("Audio service is running")
    
    try:
        await service.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error running audio service: {e}")
    finally:
        await service.stop()
        logger.info("Audio service stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experimance Audio Service")
    parser.add_argument("--config-dir", type=str, help="Directory containing audio configuration files")
    parser.add_argument("--osc-host", type=str, default="localhost", help="SuperCollider host address")
    parser.add_argument("--osc-port", type=int, default=57120, help="SuperCollider OSC port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-sc", action="store_true", help="Don't automatically start SuperCollider")
    parser.add_argument("--sc-script", type=str, help="Path to SuperCollider script (defaults to experimance_audio.scd in sc_scripts dir)")
    parser.add_argument("--sclang-path", type=str, default=DEFAULT_SCLANG_PATH, help="Path to SuperCollider language interpreter executable")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        asyncio.run(run_audio_service(
            config_dir=args.config_dir,
            osc_host=args.osc_host,
            osc_port=args.osc_port,
            auto_start_sc=not args.no_sc,
            sc_script_path=args.sc_script,
            sclang_path=args.sclang_path
        ))
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)

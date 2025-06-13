"""
Experimance Core Service: Central coordinator for the interactive art installation.

This service manages:
- Experience state machine (era progression, biome selection)
- Depth camera processing and user interaction detection
- Event publishing and coordination with other services
- Prompt generation and audio tag extraction
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.schemas import Era, Biome
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_core.config import CoreServiceConfig


logger = logging.getLogger(__name__)

# Era progression mappings
ERA_PROGRESSION = {
    Era.WILDERNESS: [Era.PRE_INDUSTRIAL],
    Era.PRE_INDUSTRIAL: [Era.EARLY_INDUSTRIAL],
    Era.EARLY_INDUSTRIAL: [Era.LATE_INDUSTRIAL],
    Era.LATE_INDUSTRIAL: [Era.MODERN],
    Era.MODERN: [Era.CURRENT],
    Era.CURRENT: [Era.FUTURE],
    Era.FUTURE: [Era.FUTURE, Era.DYSTOPIA],  # Future can loop or progress to dystopia
    Era.DYSTOPIA: [Era.RUINS],
    Era.RUINS: [Era.WILDERNESS]  # Cycle back to beginning
}

# Biome availability by era
ERA_BIOMES = {
    Era.WILDERNESS: [Biome.RAINFOREST, Biome.TEMPERATE_FOREST, Biome.BOREAL_FOREST, Biome.DECIDUOUS_FOREST, 
                     Biome.DESERT, Biome.MOUNTAIN, Biome.TUNDRA, Biome.PLAINS, Biome.RIVER, Biome.COASTAL],
    Era.PRE_INDUSTRIAL: [Biome.TEMPERATE_FOREST, Biome.DECIDUOUS_FOREST, Biome.PLAINS, Biome.RIVER, Biome.MOUNTAIN],
    Era.EARLY_INDUSTRIAL: [Biome.TEMPERATE_FOREST, Biome.DECIDUOUS_FOREST, Biome.PLAINS, Biome.RIVER, Biome.MOUNTAIN],
    Era.LATE_INDUSTRIAL: [Biome.TEMPERATE_FOREST, Biome.DESERT, Biome.PLAINS, Biome.MOUNTAIN, Biome.COASTAL],
    Era.MODERN: [Biome.TEMPERATE_FOREST, Biome.DESERT, Biome.MOUNTAIN, Biome.COASTAL, Biome.TROPICAL_ISLAND],
    Era.CURRENT: [Biome.RAINFOREST, Biome.TEMPERATE_FOREST, Biome.DESERT, Biome.MOUNTAIN, Biome.TUNDRA, 
                  Biome.COASTAL, Biome.TROPICAL_ISLAND, Biome.ARCTIC],
    Era.FUTURE: [Biome.RAINFOREST, Biome.TEMPERATE_FOREST, Biome.DESERT, Biome.MOUNTAIN, Biome.TUNDRA, 
                 Biome.COASTAL, Biome.TROPICAL_ISLAND, Biome.ARCTIC],
    Era.DYSTOPIA: [Biome.DESERT, Biome.TUNDRA, Biome.ARCTIC],  # Limited biomes after dystopia
    Era.RUINS: [Biome.RAINFOREST, Biome.TEMPERATE_FOREST, Biome.SWAMP, Biome.PLAINS]  # Nature reclaiming
}


class ExperimanceCoreService(ZmqPublisherSubscriberService):
    """
    Central coordinator service for the Experimance interactive art installation.
    
    Manages the experience state machine, coordinates with other services via ZMQ,
    and drives the narrative progression through different eras of human development.
    """

    def __init__(self, config_path: str = "config.toml"):
        """
        Initialize the Experimance Core Service.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration using the common config system
        self.config = CoreServiceConfig.from_overrides(config_file=config_path)
        
        # Extract service configuration
        service_name = self.config.experimance_core.name
        heartbeat_interval = self.config.experimance_core.heartbeat_interval
        
        # Initialize ZMQ addresses using unified events channel
        pub_address = f"tcp://*:{DEFAULT_PORTS['events']}"
        sub_address = f"tcp://localhost:{DEFAULT_PORTS['events']}"
        
        # Initialize parent service
        super().__init__(
            service_name=service_name,
            pub_address=pub_address,
            sub_address=sub_address,
            subscribe_topics=[],  # Will subscribe to all messages and filter by type
            publish_topic=f"{service_name}.heartbeat",
            service_type="core_coordinator"
        )
        
        # State machine variables
        self.current_era: Era = Era.WILDERNESS
        self.current_biome: Biome = Biome.TEMPERATE_FOREST
        self.user_interaction_score: float = 0.0
        self.idle_timer: float = 0.0
        self.audience_present: bool = False
        self.era_progression_timer: float = 0.0
        self.session_start_time: datetime = datetime.now()
        
        # Internal state
        self.last_depth_map: Optional[Any] = None
        self._message_handlers: Dict[str, Any] = {}
        
        # State management constants
        self.AVAILABLE_ERAS = list(Era)
        self.AVAILABLE_BIOMES = list(Biome)
        self.Era = Era  # Expose enum class
        self.Biome = Biome  # Expose enum class
        
        logger.info(f"Experimance Core Service initialized: {service_name}")

    async def start(self):
        """Start the service and initialize components."""
        logger.info("Starting Experimance Core Service")
        
        # Initialize message handlers
        self._register_message_handlers()
        
        # Register background tasks - best practice is to add tasks in start()
        self.add_task(self._main_event_loop())
        self.add_task(self._depth_processing_task())
        self.add_task(self._state_machine_task())
        
        # Call parent start - always call super().start() LAST
        await super().start()
        
        logger.info("Experimance Core Service started successfully")


    def _register_message_handlers(self):
        """Register handlers for different message types."""
        self._message_handlers = {
            "ImageReady": self._handle_image_ready,
            "AgentControl": self._handle_agent_control,
            "AudioStatus": self._handle_audio_status,
        }
        
        # Register handlers with the parent service
        for message_type, handler in self._message_handlers.items():
            self.register_handler(message_type, handler)
            
        logger.debug(f"Registered {len(self._message_handlers)} message handlers")

    # State Management Methods
    
    async def transition_to_era(self, new_era: str) -> bool:
        """
        Transition to a new era and publish EraChanged event.
        
        Args:
            new_era: The era to transition to
            
        Returns:
            True if transition was successful, False otherwise
        """
        if not self.is_valid_era(new_era):
            logger.warning(f"Invalid era: {new_era}")
            return False
            
        if not self.can_transition_to_era(new_era):
            logger.warning(f"Cannot transition from {self.current_era} to {new_era}")
            return False
            
        old_era = self.current_era
        self.current_era = Era(new_era)
        self.era_progression_timer = 0.0
        
        # Select appropriate biome for new era
        self.select_biome_for_era(new_era)
        
        # Publish EraChanged event
        await self._publish_era_changed_event(old_era, new_era)
        
        logger.info(f"Transitioned from {old_era} to {new_era}")
        return True
    
    def can_transition_to_era(self, target_era: str) -> bool:
        """Check if transition to target era is allowed."""
        current_era_enum = Era(self.current_era)
        target_era_enum = Era(target_era)
        return target_era_enum in ERA_PROGRESSION.get(current_era_enum, [])
    
    async def progress_era(self) -> bool:
        """Progress to the next era in the timeline."""
        current_era_enum = Era(self.current_era)
        possible_next_eras = ERA_PROGRESSION.get(current_era_enum, [])
        
        if not possible_next_eras:
            logger.warning(f"No progression defined for era: {self.current_era}")
            return False
            
        # For Future era, use probability to decide between looping and progressing
        if current_era_enum == Era.FUTURE and len(possible_next_eras) > 1:
            import random
            # 70% chance to stay in Future, 30% to progress to dystopia
            next_era = random.choices(possible_next_eras, weights=[0.7, 0.3])[0]
        else:
            # Simple progression for other eras
            next_era = possible_next_eras[0]
            
        return await self.transition_to_era(next_era.value)
    
    def get_next_era(self) -> Optional[str]:
        """Get the next possible era without transitioning."""
        current_era_enum = Era(self.current_era)
        possible_next_eras = ERA_PROGRESSION.get(current_era_enum, [])
        
        if not possible_next_eras:
            return None
            
        if current_era_enum == Era.FUTURE and len(possible_next_eras) > 1:
            import random
            return random.choices(possible_next_eras, weights=[0.7, 0.3])[0].value
        else:
            return possible_next_eras[0].value
    
    def select_biome_for_era(self, era: str) -> str:
        """Select an appropriate biome for the given era."""
        era_enum = Era(era)
        available_biomes = ERA_BIOMES.get(era_enum, [Biome.TEMPERATE_FOREST])
        
        # If current biome is available in new era, keep it
        current_biome_enum = Biome(self.current_biome)
        if current_biome_enum in available_biomes:
            return self.current_biome
            
        # Otherwise, select randomly from available biomes
        import random
        new_biome = random.choice(available_biomes)
        self.current_biome = new_biome
        logger.info(f"Selected biome {self.current_biome} for era {era}")
        return self.current_biome
    
    def update_idle_timer(self, delta_time: float):
        """Update the idle timer."""
        self.idle_timer += delta_time
    
    def should_reset_to_wilderness(self) -> bool:
        """Check if system should reset to wilderness due to idle timeout."""
        idle_timeout = self.config.state_machine.idle_timeout
        return self.idle_timer >= idle_timeout and self.current_era != Era.WILDERNESS.value
    
    async def reset_to_wilderness(self):
        """Reset the system to wilderness state."""
        old_era = self.current_era
        self.current_era = Era.WILDERNESS
        self.current_biome = Biome.TEMPERATE_FOREST
        self.idle_timer = 0.0
        self.user_interaction_score = 0.0
        self.audience_present = False
        
        # Publish EraChanged event
        await self._publish_era_changed_event(old_era, Era.WILDERNESS.value)
        
        logger.info("System reset to wilderness due to idle timeout")
    
    def calculate_interaction_score(self, interaction_intensity: float):
        """Calculate and update user interaction score."""
        # Simple scoring: weight recent interactions more heavily
        decay_factor = 0.9
        self.user_interaction_score = (self.user_interaction_score * decay_factor + 
                                     interaction_intensity * (1 - decay_factor))
        
        # Clamp to [0, 1] range
        self.user_interaction_score = max(0.0, min(1.0, self.user_interaction_score))
        
        # Reset idle timer on interaction
        if interaction_intensity > 0.1:
            self.idle_timer = 0.0
            
        logger.debug(f"Updated interaction score: {self.user_interaction_score:.3f}")
    
    def save_state(self) -> Dict[str, Any]:
        """Save current state to dictionary."""
        return {
            "current_era": self.current_era,
            "current_biome": self.current_biome,
            "user_interaction_score": self.user_interaction_score,
            "idle_timer": self.idle_timer,
            "audience_present": self.audience_present,
            "era_progression_timer": self.era_progression_timer,
            "session_start_time": self.session_start_time.isoformat()
        }
    
    def load_state(self, state_data: Dict[str, Any]):
        """Load state from dictionary."""
        # Handle enum values from string data
        era_str = state_data.get("current_era", Era.WILDERNESS.value)
        self.current_era = Era(era_str) if isinstance(era_str, str) else era_str
        
        biome_str = state_data.get("current_biome", Biome.TEMPERATE_FOREST.value)
        self.current_biome = Biome(biome_str) if isinstance(biome_str, str) else biome_str
        
        self.user_interaction_score = state_data.get("user_interaction_score", 0.0)
        self.idle_timer = state_data.get("idle_timer", 0.0)
        self.audience_present = state_data.get("audience_present", False)
        self.era_progression_timer = state_data.get("era_progression_timer", 0.0)
        
        if "session_start_time" in state_data:
            from datetime import datetime
            self.session_start_time = datetime.fromisoformat(state_data["session_start_time"])
        
        # Validate loaded state
        self.validate_and_correct_state()
        
        logger.info("State loaded successfully")
    
    def is_valid_era(self, era: str) -> bool:
        """Check if era is valid."""
        try:
            Era(era)
            return True
        except ValueError:
            return False
    
    def is_valid_biome(self, biome: str) -> bool:
        """Check if biome is valid."""
        try:
            Biome(biome)
            return True
        except ValueError:
            return False
    
    def validate_and_correct_state(self):
        """Validate current state and correct if invalid."""
        # Validate era
        if not self.is_valid_era(self.current_era):
            logger.warning(f"Invalid era {self.current_era}, resetting to wilderness")
            self.current_era = Era.WILDERNESS
        
        # Validate biome
        if not self.is_valid_biome(self.current_biome):
            logger.warning(f"Invalid biome {self.current_biome}, resetting to temperate forest")
            self.current_biome = Biome.TEMPERATE_FOREST
        
        # Validate biome is available for current era
        era_enum = Era(self.current_era) if isinstance(self.current_era, str) else self.current_era
        biome_enum = Biome(self.current_biome)
        available_biomes = ERA_BIOMES.get(era_enum, [Biome.TEMPERATE_FOREST])
        
        if biome_enum not in available_biomes:
            logger.warning(f"Biome {self.current_biome} not available for era {self.current_era}")
            self.select_biome_for_era(self.current_era)
        
        # Validate numeric ranges
        self.user_interaction_score = max(0.0, min(1.0, self.user_interaction_score))
        self.idle_timer = max(0.0, self.idle_timer)
        self.era_progression_timer = max(0.0, self.era_progression_timer)
    
    async def _publish_era_changed_event(self, old_era: str, new_era: str):
        """Publish EraChanged event."""
        event = {
            "type": MessageType.ERA_CHANGED.value,
            "old_era": old_era,
            "new_era": new_era,
            "current_biome": self.current_biome,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Simply await the publish operation
            success = await self.publish_message(event)
            if success:
                logger.debug(f"Published era change event: {old_era} -> {new_era}")
            else:
                self.record_error(Exception(f"Failed to publish era change event: {old_era} -> {new_era}"), is_fatal=False)
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Error publishing era change event")

    # Message Handler Methods
    async def _handle_image_ready(self, message: Dict[str, Any]):
        """Handle ImageReady messages from image server."""
        logger.debug(f"Received ImageReady message: {message.get('request_id')}")
        # TODO: Implement image ready handling logic

    async def _handle_agent_control(self, message: Dict[str, Any]):
        """Handle AgentControl messages from agent service."""
        logger.debug(f"Received AgentControl message: {message.get('sub_type')}")
        # TODO: Implement agent control handling logic

    async def _handle_audio_status(self, message: Dict[str, Any]):
        """Handle AudioStatus messages from audio service.""" 
        logger.debug(f"Received AudioStatus message: {message.get('status')}")
        # TODO: Implement audio status handling logic

    async def _main_event_loop(self):
        """Primary event coordination loop."""
        logger.info("Main event loop started")
        
        while self.running:
            try:
                # Main coordination logic will go here
                
                # Use _sleep_if_running() to respect shutdown requests
                if not await self._sleep_if_running(0.1):  # Prevent tight loop
                    break
                
            except Exception as e:
                self.record_error(e, is_fatal=False, 
                                custom_message="Error in main event loop")

    async def _depth_processing_task(self):
        """Handle depth camera data processing."""
        logger.info("Depth processing task started")
        
        while self.running:
            try:
                # Depth processing logic will go here
                
                # Use _sleep_if_running() to respect shutdown requests
                if not await self._sleep_if_running(0.033):  # ~30 Hz
                    break
                
            except Exception as e:
                self.record_error(e, is_fatal=False,
                                custom_message="Error in depth processing")

    async def _state_machine_task(self):
        """Handle era progression and state machine logic."""
        logger.info("State machine task started")
        
        while self.running:
            try:
                # State machine logic will go here
                
                # Use _sleep_if_running() to respect shutdown requests
                if not await self._sleep_if_running(1.0):  # 1 Hz for state updates
                    break
                
            except Exception as e:
                self.record_error(e, is_fatal=False,
                                custom_message="Error in state machine")

    async def stop(self):
        """Stop the service gracefully."""
        logger.info("Stopping Experimance Core Service")
        await super().stop()
        logger.info("Experimance Core Service stopped")


async def run_experimance_core_service(config_path: str = "config.toml"):
    """
    Run the Experimance Core Service.
    
    Args:
        config_path: Path to configuration file
    """
    service = ExperimanceCoreService(config_path=config_path)
    
    await service.start()
    await service.run()



if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    asyncio.run(run_experimance_core_service(config_path))

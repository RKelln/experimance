"""
Experimance Core Service: Central coordinator for the interactive art installation.

This service manages:
- Experience state machine (era progression, biome selection)
- Depth camera processing and user interaction detection
- Event publishing and coordination with other services
- Prompt generation and audio tag extraction
"""
import argparse
import asyncio
import logging
import time
import sys
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from experimance_common.constants import DEFAULT_PORTS, TICK
from experimance_common.schemas import Era, Biome
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_core.config import (
    CoreServiceConfig, 
    CameraState,
    DepthFrame,
    DEFAULT_CONFIG_PATH,
    CAMERA_RESET_TIMEOUT
)
from experimance_core.depth_factory import create_depth_processor


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

    def __init__(self, config: CoreServiceConfig):
        """
        Initialize the Experimance Core Service.
        
        Args:
            config: Pre-configured CoreServiceConfig instance
        """
        self.config = config
        
        # Store visualization flag for easy access  
        self.visualize = self.config.visualize
        
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
        
        # Depth processing state
        self._depth_processor: Optional[Any] = None
        self.previous_depth_image: Optional[np.ndarray] = None
        self.last_processed_frame: Optional[np.ndarray] = None  # Reference frame for change detection
        self.hand_detected: bool = False
        self.depth_difference_score: float = 0.0
        self.change_map: Optional[np.ndarray] = None  # Binary change map for display service
        
        # Retry control for depth processing
        self.depth_retry_count = 0
        self.max_depth_retries = 5
        self.depth_retry_delay = 1.0  # Start with 1 second
        self.max_depth_retry_delay = 30.0  # Cap at 30 seconds
        self.last_depth_warning_time = 0
        self._camera_state = CameraState.DISCONNECTED
        
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
        
        # Initialize depth processing (non-blocking on failure)
        try:
            await self._initialize_depth_processor()
        except Exception as e:
            logger.warning(f"Initial depth processing setup failed: {e}")
            logger.info("Depth processing will be retried during runtime")
        
        # Call parent start - always call super().start() LAST
        await super().start()
        
        logger.info("Experimance Core Service started successfully")

    def _create_camera_config(self):
        """Get the camera configuration from the service configuration."""
        return self.config.camera

    async def _initialize_depth_processor(self):
        """Initialize depth processor using the new robust camera system."""
        try:
            # Get camera config from service config
            camera_config = self._create_camera_config()
            
            # Create depth processor using the factory
            self._depth_processor = create_depth_processor(
                camera_config=camera_config,
                mock_path=None  # Use real camera by default
            )
            
            # Initialize the processor
            success = await self._depth_processor.initialize()
            if success:
                self._camera_state = CameraState.READY
                logger.info("Depth processor initialized successfully")
            else:
                self._camera_state = CameraState.ERROR
                raise Exception("Depth processor initialization returned False")
                
        except Exception as e:
            self._camera_state = CameraState.ERROR
            logger.error(f"Failed to initialize depth processor: {e}")
            raise

    async def _initialize_depth_processor_with_retry(self):
        """Initialize depth processor with retry logic and rate-limited warnings."""
        import time
        
        current_time = time.time()
        
        # Only log warning if enough time has passed since last warning
        if (self.last_depth_warning_time == 0 or 
            current_time - self.last_depth_warning_time > self.depth_retry_delay):
            
            try:
                # Get camera config from service config
                camera_config = self._create_camera_config()
                
                # Create depth processor using the factory
                self._depth_processor = create_depth_processor(
                    camera_config=camera_config,
                    mock_path=None  # Use real camera by default
                )
                
                # Initialize the processor
                success = await self._depth_processor.initialize()
                if success:
                    self._camera_state = CameraState.READY
                    logger.info("Depth processor initialized successfully")
                    
                    # Reset retry count on success
                    self.depth_retry_count = 0
                    self.depth_retry_delay = 1.0
                    self.last_depth_warning_time = 0
                    return True
                else:
                    raise Exception("Depth processor initialization returned False")
                
            except Exception as e:
                self.depth_retry_count += 1
                self.last_depth_warning_time = current_time
                self._camera_state = CameraState.ERROR
                
                if self.depth_retry_count <= self.max_depth_retries:
                    logger.warning(f"Depth processor initialization failed (attempt {self.depth_retry_count}/{self.max_depth_retries}): {e}")
                    
                    # Exponential backoff with cap
                    self.depth_retry_delay = min(self.depth_retry_delay * 2, self.max_depth_retry_delay)
                    logger.info(f"Will retry depth initialization in {self.depth_retry_delay:.1f} seconds")
                else:
                    logger.error(f"Depth processor initialization failed after {self.max_depth_retries} attempts. Last error: {e}")
                    logger.info("Will continue attempting every 30 seconds...")
                    self.depth_retry_delay = self.max_depth_retry_delay
                    
                return False
        
        return False

    async def _process_depth_frame(self, depth_frame: DepthFrame):
        """
        Process a single depth frame with smart filtering - only process when:
        1. No hands are detected
        2. Significant change compared to last processed frame
        3. Generate change maps for display service
        """
        try:
            # Extract data from DepthFrame
            depth_image = depth_frame.depth_image
            hand_detected = depth_frame.hand_detected
            
            # Update hand detection state (handle None case)
            if hand_detected is not None and self.hand_detected != hand_detected:
                self.hand_detected = hand_detected
                logger.debug(f"Hand detection changed: {hand_detected}")
                
                # Publish interaction sound trigger
                await self._publish_interaction_sound(hand_detected)
            
            # Always update the previous frame for interaction scoring
            if depth_image is not None:
                self.previous_depth_image = depth_image.copy()
            
            # Early exit if hands are detected - we don't process frames with hands
            if hand_detected:
                #logger.debug("Skipping frame processing - hands detected")
                # Still show visualization even when hands detected
                self._visualize_depth_processing(depth_frame, 0.0)
                return
            
            # Initialize change score for this frame
            change_score = 0.0
            
            # Calculate change compared to last PROCESSED frame (not just previous frame)
            if self.last_processed_frame is not None and depth_image is not None:
                # Create eroded mask to reduce edge noise
                mask = self._create_comparison_mask(depth_image)
                
                # Calculate difference with noise reduction
                change_score, change_map = self._calculate_change_with_mask(
                    self.last_processed_frame, depth_image, mask
                )
                
                # Store change map for display service
                self.change_map = change_map
                
                # Only process if change is significant enough
                change_threshold = getattr(self.config.camera, 'significant_change_threshold', 0.02)
                if change_score < change_threshold:
                    #logger.debug(f"Change too small ({change_score:.4f}), skipping frame")
                    # Show visualization even for small changes
                    self._visualize_depth_processing(depth_frame, change_score)
                    return
                
                logger.debug(f"Significant change detected ({change_score:.4f}), processing frame")
                
                # Update depth difference score for interaction calculations
                self.depth_difference_score = change_score
                
                # Calculate interaction intensity 
                interaction_intensity = change_score
                
                # Update interaction score
                self.calculate_interaction_score(interaction_intensity)
                
                # Publish change map to display service
                await self._publish_change_map(change_map, change_score)
                
                # Publish video mask for visualization if significant interaction
                if interaction_intensity > 0.1:
                    await self._publish_video_mask()
            
            # This frame becomes our new reference frame
            if depth_image is not None:
                self.last_processed_frame = depth_image.copy()
                self.last_depth_map = depth_image.copy()
                logger.debug("Updated reference frame for change detection")
            
            # Visualization for debugging (always show for processed frames)
            self._visualize_depth_processing(depth_frame, change_score)
            
        except Exception as e:
            logger.error(f"Error processing depth frame: {e}")

    def _create_comparison_mask(self, depth_image: np.ndarray) -> np.ndarray:
        """Create a mask for comparison that reduces edge noise."""
        h, w = depth_image.shape[:2]
        
        # Create a mask that excludes the edges where noise is common
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Erode the mask to exclude noisy edges
        try:
            edge_erosion = getattr(self.config.camera, 'edge_erosion_pixels', 10)
            # Handle case where this might be a MagicMock in tests
            if hasattr(edge_erosion, '_mock_name'):
                edge_erosion = 10
        except (AttributeError, TypeError):
            edge_erosion = 10
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_erosion, edge_erosion))
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask

    def _calculate_change_with_mask(self, ref_frame: np.ndarray, current_frame: np.ndarray, 
                                   mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate change between frames using mask to reduce noise.
        
        Returns:
            Tuple of (change_score, binary_change_map)
        """
        # Resize frames for comparison
        small_size = (128, 128)
        small_ref = cv2.resize(ref_frame.astype(np.uint8), small_size)
        small_current = cv2.resize(current_frame.astype(np.uint8), small_size)
        small_mask = cv2.resize(mask, small_size)
        
        # Calculate absolute difference
        diff = cv2.absdiff(small_ref, small_current)
        
        # Apply threshold to get binary difference
        change_threshold = getattr(self.config.camera, 'change_threshold', 30)
        _, binary_diff = cv2.threshold(diff, change_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply mask to exclude noisy regions
        binary_diff = cv2.bitwise_and(binary_diff, small_mask)
        
        # Morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
        
        # Calculate change score
        changed_pixels = cv2.countNonZero(binary_diff)
        mask_pixels = cv2.countNonZero(small_mask)
        
        if mask_pixels == 0:
            change_score = 0.0
        else:
            change_score = changed_pixels / mask_pixels
        
        # Create full-resolution change map for display service
        change_map = cv2.resize(binary_diff, (current_frame.shape[1], current_frame.shape[0]))
        
        return change_score, change_map

    async def _publish_change_map(self, change_map: np.ndarray, change_score: float):
        """Publish change map to display service."""
        event = {
            "type": MessageType.CHANGE_MAP.value,
            "change_score": change_score,
            "has_change_map": True,  # Indicates binary change map is available
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            success = await self.publish_message(event)
            if success:
                logger.debug(f"Published change map: score={change_score:.4f}")
            else:
                logger.warning("Failed to publish change map event")
        except Exception as e:
            logger.error(f"Error publishing change map: {e}")

    async def _publish_interaction_sound(self, hand_detected: bool):
        """Publish interaction sound command based on hand detection."""
        event = {
            "type": "AudioCommand",  # Custom message type for audio commands
            "trigger": "interaction_start" if hand_detected else "interaction_stop",
            "hand_detected": hand_detected,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            success = await self.publish_message(event)
            if success:
                logger.debug(f"Published interaction sound: {'start' if hand_detected else 'stop'}")
            else:
                logger.warning("Failed to publish interaction sound command")
        except Exception as e:
            logger.error(f"Error publishing interaction sound: {e}")

    async def _publish_video_mask(self):
        """Publish video mask event for depth difference visualization."""
        event = {
            "type": MessageType.VIDEO_MASK.value,
            "interaction_score": self.user_interaction_score,
            "depth_difference_score": self.depth_difference_score,
            "hand_detected": self.hand_detected,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            success = await self.publish_message(event)
            if success:
                logger.debug(f"Published video mask: score={self.user_interaction_score:.3f}")
            else:
                logger.warning("Failed to publish video mask event")
        except Exception as e:
            logger.error(f"Error publishing video mask: {e}")

    async def _publish_idle_state_changed(self):
        """Publish idle state changed event."""
        event = {
            "type": MessageType.IDLE_STATUS.value,
            "idle_duration": self.idle_timer,
            "current_era": self.current_era,
            "current_biome": self.current_biome,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            success = await self.publish_message(event)
            if success:
                logger.debug(f"Published idle state: {self.idle_timer:.1f}s")
            else:
                logger.warning("Failed to publish idle state event")
        except Exception as e:
            logger.error(f"Error publishing idle state: {e}")

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
        self.select_biome_for_era(self.current_era)
        
        # Publish EraChanged event
        await self._publish_era_changed_event(old_era, self.current_era)
        
        logger.info(f"Transitioned from {old_era} to {self.current_era}")
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
    
    def select_biome_for_era(self, era: str | Era) -> str:
        """Select an appropriate biome for the given era."""
        if isinstance(era, str):
            era_enum = Era(era)
        elif isinstance(era, Era):
            era_enum = era
        else:
            self.record_error(f"Invalid era type: {type(era)}")

        available_biomes = ERA_BIOMES.get(era_enum, [Biome.TEMPERATE_FOREST])
        
        # If current biome is available in new era, keep it
        current_biome_enum = Biome(self.current_biome)
        if current_biome_enum in available_biomes:
            return self.current_biome
            
        # Otherwise, select randomly from available biomes
        import random
        new_biome = random.choice(available_biomes)
        self.current_biome = new_biome
        logger.info(f"Selected biome {self.current_biome} for era {era_enum}")
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
                # Check if we should publish a render request
                # This should happen when there's significant interaction or era changes
                if (self.user_interaction_score > self.config.state_machine.interaction_threshold or
                    self.era_progression_timer < 2.0):  # Recently changed era
                    
                    await self._publish_render_request()
                
                # Periodic state logging for debugging
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.info(f"State: era={self.current_era}, biome={self.current_biome}, "
                               f"interaction={self.user_interaction_score:.3f}, "
                               f"idle={self.idle_timer:.1f}s, hand_detected={self.hand_detected}")
                
                # Use _sleep_if_running() to respect shutdown requests
                if not await self._sleep_if_running(0.5):  # Check twice per second
                    break
                
            except Exception as e:
                self.record_error(e, is_fatal=False, 
                                custom_message="Error in main event loop")

    async def _publish_render_request(self):
        """Publish a render request to the image server."""
        event = {
            "type": MessageType.RENDER_REQUEST.value,
            "current_era": self.current_era,
            "current_biome": self.current_biome,
            "interaction_score": self.user_interaction_score,
            "seed": int(time.time()),  # Use timestamp as seed for variability
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            success = await self.publish_message(event)
            if success:
                logger.debug(f"Published render request for {self.current_era}/{self.current_biome}")
            else:
                logger.warning("Failed to publish render request")
        except Exception as e:
            logger.error(f"Error publishing render request: {e}")

    async def _depth_processing_task(self):
        """Handle depth camera data processing with improved retry logic using robust camera."""
        logger.info("Depth processing task started")
        
        while self.running:
            try:
                if self._depth_processor is not None:
                    try:
                        # Get next depth frame
                        depth_frame = await self._depth_processor.get_processed_frame()
                        
                        if not self.running:
                            break
                        
                        if depth_frame is not None:
                            await self._process_depth_frame(depth_frame)
                        
                        # Brief yield to allow other tasks to run
                        await asyncio.sleep(TICK)
                        
                    except asyncio.CancelledError:
                        logger.info("Depth processing task was cancelled")
                        break
                        
                    except Exception as e:
                        logger.error(f"Error in depth processing loop: {e}")
                        # Reset the processor to attempt recovery
                        if self._depth_processor:
                            try:
                                self._depth_processor.stop()
                            except:
                                pass
                        self._depth_processor = None
                        self._camera_state = CameraState.ERROR
                
                # Try to initialize or reinitialize depth processing
                if self._depth_processor is None:
                    try:
                        # Only timeout on initialization to prevent hanging during reset/init
                        success = await asyncio.wait_for(
                            self._initialize_depth_processor_with_retry(),
                            timeout=CAMERA_RESET_TIMEOUT  # 1 minute timeout for initialization only
                        )
                        if not success:
                            # Wait for the retry delay before trying again
                            if not await self._sleep_if_running(self.depth_retry_delay):
                                break
                    except asyncio.TimeoutError:
                        logger.warning("Depth processor initialization timed out")
                        if not await self._sleep_if_running(self.depth_retry_delay):
                            break
                    except asyncio.CancelledError:
                        logger.info("Depth processor initialization was cancelled")
                        break
                
            except asyncio.CancelledError:
                logger.info("Depth processing task cancelled")
                break
            except Exception as e:
                self.record_error(e, is_fatal=False,
                                custom_message="Error in depth processing task")
                # Sleep briefly before retrying to avoid tight error loops
                if not await self._sleep_if_running(1.0):
                    break

    async def _state_machine_task(self):
        """Handle era progression and state machine logic."""
        logger.info("State machine task started")
        
        last_update = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                delta_time = current_time - last_update
                last_update = current_time
                
                # Update timers
                self.update_idle_timer(delta_time)
                self.era_progression_timer += delta_time
                
                # Check for idle reset to wilderness
                if self.should_reset_to_wilderness():
                    await self.reset_to_wilderness()
                
                # Check for era progression based on interaction
                elif (self.user_interaction_score > self.config.state_machine.interaction_threshold and
                      self.era_progression_timer >= self.config.state_machine.era_min_duration):
                    
                    # Check if we can progress to next era
                    next_era = self.get_next_era()
                    if next_era and next_era != self.current_era:
                        logger.info(f"Era progression triggered by interaction: {self.user_interaction_score:.3f}")
                        await self.progress_era()
                
                # Publish idle state changes if needed
                if self.idle_timer > 0 and int(self.idle_timer) % 10 == 0:  # Every 10 seconds of idle
                    await self._publish_idle_state_changed()
                
                # Use _sleep_if_running() to respect shutdown requests
                if not await self._sleep_if_running(1.0):  # 1 Hz for state updates
                    break
                
            except Exception as e:
                self.record_error(e, is_fatal=False,
                                custom_message="Error in state machine")

    async def stop(self):
        """Stop the service gracefully."""
        logger.info("Stopping Experimance Core Service")
        # debugging print the caller of stop
        import traceback
        logger.info(f"Stop called from: {traceback.format_stack()[-2].strip()}")
        
        # Clean up visualization window if it was used
        if self.visualize:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logger.warning(f"Error cleaning up visualization window: {e}")
        
        # Call parent stop first to ensure proper shutdown sequence
        await super().stop()
        
        # Clean up depth processing
        if self._depth_processor is not None:
            try:
                self._depth_processor.stop()
                self._depth_processor = None
                self._camera_state = CameraState.DISCONNECTED
            except Exception as e:
                logger.error(f"Error cleaning up depth processor: {e}")
        
        logger.info("Experimance Core Service stopped")

    def _visualize_depth_processing(self, depth_frame: DepthFrame, change_score: float = 0.0):
        """
        Display depth frame with processing flags for visual debugging.
        
        Args:
            depth_frame: Current depth frame data
            change_score: Current change detection score
        """
        if not self.visualize:
            return
            
        try:
            depth_image = depth_frame.depth_image
            hand_detected = depth_frame.hand_detected
            
            if depth_image is None:
                return
                
            # Normalize depth image for display (0-255)
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            cv2.normalize(depth_image, depth_normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Convert to 3-channel for color overlays
            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Create info overlay
            overlay = depth_color.copy()
            
            # Add status text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Status indicators
            status_text = [
                f"Hand Detected: {'YES' if hand_detected else 'NO'}",
                f"Change Score: {change_score:.4f}",
                f"Era: {self.current_era.value}",
                f"Biome: {self.current_biome.value}",
                f"Interaction Score: {self.user_interaction_score:.3f}",
                f"Idle Timer: {self.idle_timer:.1f}s",
                f"Camera State: {self._camera_state.value}",
            ]
            
            # Color coding
            hand_color = (0, 0, 255) if hand_detected else (0, 255, 0)  # Red if hands, green if no hands
            change_color = (0, 255, 255) if change_score > 0.02 else (128, 128, 128)  # Yellow if significant change
            
            # Draw status text
            y_offset = 30
            for i, text in enumerate(status_text):
                color = hand_color if i == 0 else change_color if i == 1 else (255, 255, 255)
                cv2.putText(overlay, text, (10, y_offset + i * 25), font, font_scale, color, thickness)
            
            # Show change map if available
            if hasattr(self, 'change_map') and self.change_map is not None:
                # Resize change map to match depth image
                change_resized = cv2.resize(self.change_map.astype(np.uint8) * 255, 
                                          (depth_image.shape[1], depth_image.shape[0]))
                change_colored = cv2.applyColorMap(change_resized, cv2.COLORMAP_HOT)
                
                # Blend change map with depth image
                alpha = 0.3
                cv2.addWeighted(overlay, 1 - alpha, change_colored, alpha, 0, overlay)
            
            # Add frame border based on processing state
            border_color = (0, 0, 255) if hand_detected else (0, 255, 0)  # Red if hands, green if processing
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1]-1, overlay.shape[0]-1), border_color, 3)
            
            # Show the visualization
            cv2.imshow('Experimance Core - Depth Processing', overlay)
            
            # Non-blocking key check (1ms wait)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Visualization quit requested (press 'q' again or Ctrl+C to stop service)")
                
        except Exception as e:
            logger.warning(f"Visualization error: {e}")
        
async def run_experimance_core_service(
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
    config = CoreServiceConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    service = ExperimanceCoreService(config=config)
    
    await service.start()
    await service.run()

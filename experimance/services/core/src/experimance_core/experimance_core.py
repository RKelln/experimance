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
import random
import time
import traceback
import sys
import uuid
from collections import deque
from experimance_core.depth_processor import DepthProcessor
from experimance_core.depth_visualizer import DepthVisualizer
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from experimance_common.constants import DEFAULT_PORTS, TICK, IMAGE_TRANSPORT_MODES
from experimance_common.schemas import Era, Biome, ContentType, RenderRequest
from experimance_common.base_service import BaseService
from experimance_common.zmq.services import ControllerService
from experimance_common.zmq.config import MessageType, MessageDataType, ControllerServiceConfig
from experimance_common.zmq.zmq_utils import prepare_image_message
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

class ExperimanceCoreService(BaseService):
    """
    Central coordinator service for the Experimance interactive art installation.
    
    Manages the experience state machine, coordinates with other services via ZMQ,
    and drives the narrative progression through different eras of human development.
    
    Uses composition pattern with ControllerService for ZMQ communication.
    """

    def __init__(self, config: CoreServiceConfig):
        """
        Initialize the Experimance Core Service.
        
        Args:
            config: Pre-configured CoreServiceConfig instance
        """
        self.config = config
        
        # Initialize base service
        super().__init__(
            service_name=config.service_name,
            service_type="core_coordinator"
        )
        
        # Initialize ZMQ controller service using composition
        self.zmq_service = ControllerService(config=config.zmq)
        
        # State machine variables
        self.current_era: Era = Era.WILDERNESS
        self.current_biome: Biome = Biome.TEMPERATE_FOREST
        self.user_interaction_score: float = 0.0
        self.idle_timer: float = 0.0
        self.audience_present: bool = False
        self.era_progression_timer: float = 0.0
        self.session_start_time: datetime = datetime.now()
        
        # Track era changes for transition decisions
        self._last_era: Optional[Era] = Era.WILDERNESS
        self._pending_transition: Optional[Dict[str, Any]] = None  # Store pending transition data
        
        # Internal state
        self.last_depth_map: Optional[Any] = None

        # Depth processing state
        self._depth_processor: Optional[DepthProcessor] = None
        self._depth_visualizer: Optional[DepthVisualizer] = None
        self.previous_depth_image: Optional[np.ndarray] = None
        self.last_processed_frame: Optional[np.ndarray] = None  # Reference frame for change detection
        self.hand_detected: bool = False
        self.depth_difference_score: float = 0.0
        self.change_map: Optional[np.ndarray] = None  # Binary change map for display service
        
        # Change score smoothing with queue (take minimum to reduce artifacts)
        queue_size = self.config.experimance_core.change_smoothing_queue_size
        self.change_score_queue: deque = deque(maxlen=queue_size)
        
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
        
        logger.info(f"Experimance Core Service initialized: {self.service_name}")



    async def start(self):
        """Start the service and initialize components."""
        logger.info("Starting Experimance Core Service")
        
        # Initialize message handlers
        self._register_message_handlers()
        
        # Start the ZMQ service
        await self.zmq_service.start()
        
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
        camera_config = self.config.camera
        return camera_config

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
            
            # Initialize visualizer if debug mode is enabled
            if camera_config.debug_mode:
                logger.info("🎬 Debug mode enabled, initializing depth visualizer")
                self._depth_visualizer = DepthVisualizer(
                    window_name="Experimance Core - Depth Debug",
                    window_size=(1200, 800)
                )
                self._depth_visualizer.create_window()
            
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
                
                # Clear change score queue when hand state changes to reset smoothing
                self.change_score_queue.clear()
                logger.debug("Cleared change score queue due to hand state change")
            
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
            raw_change_score = 0.0
            
            # Calculate change compared to last PROCESSED frame (not just previous frame)
            if self.last_processed_frame is not None and depth_image is not None:
                # Create eroded mask to reduce edge noise
                mask = self._create_comparison_mask(depth_image)
                
                # Calculate difference with noise reduction
                raw_change_score, change_map = self._calculate_change_with_mask(
                    self.last_processed_frame, depth_image, mask
                )
                
                # Add raw change score to queue for smoothing
                self.change_score_queue.append(raw_change_score)
                
                # Use minimum from queue to reduce artifacts from hand entry/exit
                if self.change_score_queue.maxlen is not None and len(self.change_score_queue) >= self.change_score_queue.maxlen:
                    smoothed_change_score = min(self.change_score_queue)
                else:
                    smoothed_change_score = 0
                
                logger.debug(f"Change scores - raw: {raw_change_score:.4f}, smoothed (min): {smoothed_change_score:.4f}, queue: {list(self.change_score_queue)}")
                
                # Only process if smoothed change is significant enough
                change_threshold = getattr(self.config.camera, 'significant_change_threshold', 0.01)
                if smoothed_change_score < change_threshold:
                    logger.debug(f"Smoothed change too small ({smoothed_change_score:.4f}), skipping frame")
                    # Show visualization even for small changes
                    self._visualize_depth_processing(depth_frame, smoothed_change_score)
                    return
                
                logger.debug(f"Significant smoothed change detected ({smoothed_change_score:.4f}), processing frame")
                
                self.change_score_queue.clear()  # Clear queue after processing significant change

                # Store change map for display service
                self.change_map = change_map

                # Update depth difference score for interaction calculations (use smoothed score)
                self.depth_difference_score = smoothed_change_score
                
                # Calculate interaction intensity (currently just using smoothed change score)
                interaction_intensity = smoothed_change_score
                
                # Update interaction score
                self.calculate_interaction_score(interaction_intensity)
                
                # Update the generated image for display service
                # TODO: This should be a separate task in the future


                # Publish change map to display service (with smoothed score)
                await self._publish_change_map(change_map, smoothed_change_score)
            
            # This frame becomes our new reference frame
            if depth_image is not None:
                self.last_processed_frame = depth_image.copy()
                self.last_depth_map = depth_image.copy()
                logger.debug("Updated reference frame for change detection")
            
            # Visualization for debugging (always show for processed frames)
            # Use smoothed score if available, otherwise raw score
            display_score = smoothed_change_score if 'smoothed_change_score' in locals() else raw_change_score
            self._visualize_depth_processing(depth_frame, display_score, self.depth_difference_score)
            
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
        try:
            # Use the new enum-based image utilities
            message = prepare_image_message(
                image_data=change_map,
                target_address=f"tcp://localhost:{DEFAULT_PORTS['events']}",
                transport_mode=IMAGE_TRANSPORT_MODES["AUTO"],  # Auto-detect optimal transport
                type=MessageType.CHANGE_MAP.value,
                change_score=change_score,
                has_change_map=True,
                timestamp=datetime.now().isoformat(),
                mask_id=f"change_map_{int(time.time() * 1000)}"
            )
            
            success = await self.zmq_service.publish(message)
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
            success = await self.zmq_service.publish(event)
            if success:
                logger.debug(f"Published interaction sound: {'start' if hand_detected else 'stop'}")
            else:
                logger.warning("Failed to publish interaction sound command")
        except Exception as e:
            logger.error(f"Error publishing interaction sound: {e}")

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
            success = await self.zmq_service.publish(event)
            if success:
                logger.debug(f"Published idle state: {self.idle_timer:.1f}s")
            else:
                logger.warning("Failed to publish idle state event")
        except Exception as e:
            logger.error(f"Error publishing idle state: {e}")

    def _register_message_handlers(self):
        """Register handlers for different message types."""
        # Register PubSub message handlers
        self.zmq_service.add_message_handler(MessageType.AGENT_CONTROL_EVENT, self._zmq_handle_agent_control)
        self.zmq_service.add_message_handler("AudioStatus", self._zmq_handle_audio_status)
        
        # Register worker response handler for responses from push/pull workers
        self.zmq_service.add_response_handler(self._handle_worker_response)
        
        logger.info("Message handlers registered with ZMQ service")
        

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
        # decay_factor = 0.9
        # self.user_interaction_score = (self.user_interaction_score * decay_factor + 
        #                              interaction_intensity * (1 - decay_factor))
        
        # Clamp to [0, 1] range
        #self.user_interaction_score = max(0.0, min(1.0, self.user_interaction_score))
        
        # for testing just add up all the intensities
        self.user_interaction_score += interaction_intensity

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
            success = await self.zmq_service.publish(event)
            if success:
                logger.debug(f"Published era change event: {old_era} -> {new_era}")
            else:
                self.record_error(Exception(f"Failed to publish era change event: {old_era} -> {new_era}"), is_fatal=False)
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Error publishing era change event")

    # Message Handler Methods
    async def _handle_image_ready(self, message: Dict[str, Any]):
        """Handle ImageReady messages from image server."""
        request_id = message.get('request_id')
        logger.debug(f"Received ImageReady message: {request_id}")
        
        try:
            # Check if this is a response to our render request
            if not request_id:
                logger.warning("ImageReady message missing request_id")
                return
            
            # Determine if we need a transition
            needs_transition = await self._should_request_transition(message)
            
            if needs_transition:
                # Request transition from transition service
                await self._request_transition(message)
            else:
                # Send directly to display service
                await self._send_display_media(message)
                
        except Exception as e:
            logger.error(f"Error handling ImageReady message: {e}")

    async def _handle_image_ready_task(self, task: MessageDataType):
        """Handle IMAGE_READY tasks received from the image server via PULL socket.
        
        Args:
            task: The IMAGE_READY message from the image server
        """
        try:
            # Convert to dict if it's a Pydantic model
            if hasattr(task, 'model_dump') and callable(getattr(task, 'model_dump')):
                task_dict = task.model_dump()  # type: ignore
            elif isinstance(task, dict):
                task_dict = task
            else:
                logger.warning(f"Unexpected task type: {type(task)}")
                return
                
            if task_dict.get("type") == MessageType.IMAGE_READY.value:
                logger.debug(f"Received IMAGE_READY task: {task_dict.get('request_id')}")
                # Delegate to existing message handler
                await self._handle_image_ready(task_dict)
            else:
                logger.warning(f"Received unexpected task type from image server: {task_dict.get('type')}")
        except Exception as e:
            logger.error(f"Error handling image ready task: {e}")
            self.record_error(e, is_fatal=False, custom_message=f"Error handling IMAGE_READY task: {e}")

    async def _should_request_transition(self, image_message: Dict[str, Any]) -> bool:
        """
        Determine if a transition is needed based on current state.
        
        Args:
            image_message: The ImageReady message from image server
            
        Returns:
            True if a transition should be requested
        """
        # For now, implement simple logic - can be made more sophisticated
        
        # Always transition on era changes
        if hasattr(self, '_last_era') and self._last_era != self.current_era:
            logger.info(f"Era changed from {getattr(self, '_last_era', None)} to {self.current_era}, requesting transition")
            return True
        
        # Transition on significant interaction (major changes to the sand table)
        if self.user_interaction_score > self.config.state_machine.interaction_threshold * 2:
            logger.info(f"High interaction score ({self.user_interaction_score:.3f}), requesting transition")
            return True
        
        # For now, default to no transition for minor changes
        return False

    async def _request_transition(self, image_message: Dict[str, Any]):
        """
        Request a transition from the transition service.
        
        Args:
            image_message: The ImageReady message containing the new image
        """
        try:
            # Determine transition type based on context
            transition_type = self._get_transition_type()
            
            # For now, we'll implement a simple approach where we directly
            # send to display with transition info. Later this can be expanded
            # to actually communicate with a transition service.
            
            logger.info(f"Requesting {transition_type} transition")
            
            # Send to display with transition metadata
            await self._send_display_media(image_message, transition_type=transition_type)
            
        except Exception as e:
            logger.error(f"Error requesting transition: {e}")
            # Fallback: send without transition
            await self._send_display_media(image_message)

    def _get_transition_type(self) -> str:
        """
        Determine the appropriate transition type based on current context.
        
        Returns:
            Transition type string
        """
        # Era changes get more dramatic transitions
        if hasattr(self, '_last_era') and self._last_era != self.current_era:
            era_transitions = {
                "wilderness": "fade",
                "pre_industrial": "slide", 
                "early_industrial": "fade",
                "late_industrial": "morph",
                "modern": "slide",
                "current": "fade",
                "future": "morph",
                "dystopia": "fade",
                "ruins": "fade"
            }
            return era_transitions.get(self.current_era.value, "fade")
        
        # High interaction gets subtle transitions
        if self.user_interaction_score > self.config.state_machine.interaction_threshold:
            return "fade"
        
        return "fade"  # Default

    async def _send_display_media(self, image_message: Dict[str, Any], transition_type: Optional[str] = None):
        """
        Send DISPLAY_MEDIA message to display service.
        
        Args:
            image_message: The ImageReady message from image server
            transition_type: Optional transition type
        """
        try:
            # Create DISPLAY_MEDIA message
            display_message = {
                "type": MessageType.DISPLAY_MEDIA.value,
                "content_type": ContentType.IMAGE.value,
                "request_id": image_message.get('request_id'),
                "timestamp": datetime.now().isoformat(),
                "era": self.current_era.value,
                "biome": self.current_biome.value,
                "interaction_score": self.user_interaction_score
            }
            
            # Add transition info if specified
            if transition_type:
                display_message["transition_type"] = transition_type
                display_message["transition_duration"] = 2.0  # Default 2 seconds
            
            # Copy image transport fields from the ImageReady message
            image_fields = ["uri", "image_data", "image_format", "image_id", "mask_id"]
            for field in image_fields:
                if field in image_message:
                    display_message[field] = image_message[field]
            
            # Publish to display service
            success = await self.zmq_service.publish(display_message)
            
            if success:
                logger.debug(f"Sent DISPLAY_MEDIA to display service (transition: {transition_type})")
                
                # Update state tracking
                self._last_era = self.current_era
                
            else:
                logger.warning("Failed to send DISPLAY_MEDIA message")
                
        except Exception as e:
            logger.error(f"Error sending DISPLAY_MEDIA: {e}")

    async def _handle_agent_control(self, message: Dict[str, Any]):
        """Handle AgentControl messages from agent service."""
        logger.debug(f"Received AgentControl message: {message.get('sub_type')}")
        # TODO: Implement agent control handling logic

    async def _handle_audio_status(self, message: Dict[str, Any]):
        """Handle AudioStatus messages from audio service.""" 
        logger.debug(f"Received AudioStatus message: {message.get('status')}")
        # TODO: Implement audio status handling logic

    async def _handle_transition_ready(self, message: Dict[str, Any]):
        """Handle TRANSITION_READY messages from transition service."""
        logger.debug(f"Received TRANSITION_READY message: {message.get('transition_id')}")
        
        if self._pending_transition is None:
            logger.warning("Received TRANSITION_READY but no pending transition")
            return
        
        try:
            # Create DISPLAY_MEDIA message with transition
            transition_type = message.get("transition_type", "fade")
            sequence_path = message.get("sequence_path")
            video_path = message.get("video_path")
            duration = message.get("duration", 2.0)
            
            if sequence_path:
                # Image sequence transition
                display_media = self._create_display_media(
                    content_type=ContentType.IMAGE_SEQUENCE,
                    transition_type=transition_type,
                    sequence_path=sequence_path,
                    duration=duration,
                    final_image_id=self._pending_transition["target_image_id"],
                    final_image_uri=self._pending_transition["target_image_uri"],
                    era=self.current_era.value,
                    biome=self.current_biome.value,
                    interaction_score=self.user_interaction_score
                )
            elif video_path:
                # Video transition
                display_media = self._create_display_media(
                    content_type=ContentType.VIDEO,
                    transition_type=transition_type,
                    video_path=video_path,
                    duration=duration,
                    final_image_id=self._pending_transition["target_image_id"],
                    final_image_uri=self._pending_transition["target_image_uri"],
                    era=self.current_era.value,
                    biome=self.current_biome.value,
                    interaction_score=self.user_interaction_score
                )
            else:
                logger.warning("TRANSITION_READY message missing sequence_path or video_path")
                return
            
            # Publish DISPLAY_MEDIA
            await self.zmq_service.publish(display_media)
            logger.info(f"Published DISPLAY_MEDIA with {transition_type} transition")
            
            # Clear pending transition
            self._pending_transition = None
            
        except Exception as e:
            logger.error(f"Error handling TRANSITION_READY: {e}")
            self._pending_transition = None
    
    def _create_display_media(self, content_type: ContentType, **kwargs) -> Dict[str, Any]:
        """Create a DISPLAY_MEDIA message with the given content type and parameters."""
        display_media = {
            "type": MessageType.DISPLAY_MEDIA.value,
            "content_type": content_type.value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add all additional parameters
        for key, value in kwargs.items():
            if value is not None:
                display_media[key] = value
        
        return display_media
    
    def _needs_transition(self) -> bool:
        """Check if a transition is needed based on era change."""
        if self._last_era is None:
            return False  # First image, no transition needed
        
        return self.current_era != self._last_era

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

    async def _publish_render_request(self, message: Optional[RenderRequest] = None):
        """Publish a render request to the image server."""
        
        if message is None:
            # Create a default render request using the proper schema
            prompt = self._generate_prompt_for_era_biome(self.current_era, self.current_biome)
            
            request = RenderRequest(
                request_id=str(uuid.uuid4()),
                era=self.current_era,
                biome=self.current_biome,
                prompt=prompt,
                seed=int(time.time())  # Use timestamp as seed for variability
            )
        else:
            request = message
        
        try:
            # Send the task to the image server worker via PUSH socket
            await self.zmq_service.send_work_to_worker("image_server", request)
            logger.debug(f"Sent render request for {request.era}/{request.biome}")
            logger.debug(f"Request ID: {request.request_id}")
        except Exception as e:
            logger.error(f"Error sending render request: {e}")
            
    def _generate_prompt_for_era_biome(self, era: Era, biome: Biome) -> str:
        """Generate an appropriate prompt for the given era and biome."""
        # Map eras to descriptive periods
        era_descriptions = {
            Era.WILDERNESS: "pristine wilderness",
            Era.PRE_INDUSTRIAL: "pre-industrial settlement", 
            Era.EARLY_INDUSTRIAL: "early industrial development",
            Era.LATE_INDUSTRIAL: "industrial landscape",
            Era.MODERN: "modern development",
            Era.CURRENT: "contemporary landscape",
            Era.FUTURE: "futuristic landscape",
            Era.DYSTOPIA: "dystopian wasteland",
            Era.RUINS: "overgrown ruins"
        }
        
        # Map biomes to environmental descriptions
        biome_descriptions = {
            Biome.RAINFOREST: "dense rainforest",
            Biome.TEMPERATE_FOREST: "temperate forest",
            Biome.BOREAL_FOREST: "boreal forest",
            Biome.DECIDUOUS_FOREST: "deciduous forest",
            Biome.DESERT: "arid desert",
            Biome.MOUNTAIN: "mountainous terrain",
            Biome.TUNDRA: "arctic tundra",
            Biome.PLAINS: "open plains",
            Biome.RIVER: "river valley",
            Biome.COASTAL: "coastal region",
            Biome.TROPICAL_ISLAND: "tropical island",
            Biome.ARCTIC: "arctic landscape",
            Biome.SWAMP: "wetland swamp"
        }
        
        era_desc = era_descriptions.get(era, "landscape")
        biome_desc = biome_descriptions.get(biome, "natural area")
        
        # Create a descriptive prompt for satellite imagery
        prompt = f"Satellite view of {biome_desc} in {era_desc}, aerial perspective, detailed topography"
        
        return prompt

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
                            # start_time = time.perf_counter()
                            await self._process_depth_frame(depth_frame)
                            # end_time = time.perf_counter()
                            # duration_ms = (end_time - start_time) * 1000
                            # if self.config.camera.verbose_performance:
                            #     logger.info(f"Processed depth frame in {duration_ms:.1f} ms")
                            
                            # Display frame in visualizer if debug mode is enabled
                            if self._depth_visualizer is not None:
                                # Non-blocking visualization (don't let window events affect service)
                                if not self._depth_visualizer.display_frame(depth_frame, show_fps=False):
                                    logger.info("Depth visualizer window closed, disabling visualization")
                                    self._depth_visualizer.destroy_window()
                                    self._depth_visualizer = None
                        
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
        """Stop the service and clean up resources."""
        logger.info("Stopping Experimance Core Service")
        
        try:
            # Stop ZMQ service first to ensure subscriber tasks are cancelled
            if self.zmq_service and self.zmq_service.is_running():
                logger.debug("Stopping ZMQ service...")
                await self.zmq_service.stop()
                logger.debug("ZMQ service stopped")
            
            # Clean up depth processor
            if self._depth_processor:
                self._depth_processor.stop()
                self._depth_processor = None
            
            # Clean up depth visualizer
            if self._depth_visualizer:
                self._depth_visualizer.destroy_window()
                self._depth_visualizer = None
                
            logger.info("Core service components stopped")
            
        except Exception as e:
            logger.error(f"Error during Core service component shutdown: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
            # Continue with base service stop even if components fail
        
        # Stop base service last
        await super().stop()
        
        logger.info("Experimance Core Service stopped")

    def _visualize_depth_processing(self, depth_frame: DepthFrame, 
                                    change_score: float = 0.0, significant_change: float = 0.0):
        """
        Display depth frame with processing flags for visual debugging.
        
        Args:
            depth_frame: Current depth frame data
            change_score: Current change detection score
            significant_change: Last significant change score (for visualization)
            change_map: Optional binary change map for visualization
        """
        if not self.config.visualize:
            return
            
        try:
            depth_image = depth_frame.depth_image
            hand_detected = depth_frame.hand_detected
            
            if depth_image is None:
                return
                
            # Normalize depth image for display (0-255)
            #depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            #cv2.normalize(depth_image, depth_normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Convert to 3-channel for color overlays
            depth_color = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            
            # Create info overlay
            overlay = depth_color.copy()
            
            # Add status text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Status indicators
            queue_size = self.config.experimance_core.change_smoothing_queue_size
            queue_info = f"Queue({len(self.change_score_queue)}/{queue_size}): " + str([f"{x:.3f}" for x in list(self.change_score_queue)])
            queue_min = min(self.change_score_queue) if len(self.change_score_queue) > 0 else 0.0
            
            status_text = [
                f"Hand Detected: {'YES' if hand_detected else 'NO'}",
                f"Change Score (raw): {change_score:.4f}",
                f"Change Score (min): {queue_min:.4f}",
                f"Last significant change: {significant_change:.4f}",
                queue_info,
                f"Era: {self.current_era.value}",
                f"Biome: {self.current_biome.value}",
                f"Interaction Score: {self.user_interaction_score:.3f}",
                f"Idle Timer: {self.idle_timer:.1f}s",
                f"Camera State: {self._camera_state.value}",
            ]
            
            # Color coding
            hand_color = (0, 0, 255) if hand_detected else (0, 255, 0)  # Red if hands, green if no hands
            change_color = (0, 255, 255) if queue_min > 0.02 else (128, 128, 128)  # Yellow if significant change
            queue_color = (255, 255, 255)  # White for queue info
            
            # Draw status text
            y_offset = 30
            for i, text in enumerate(status_text):
                if i == 0:
                    color = hand_color
                elif i == 1 or i == 2:
                    color = change_color
                elif i == 3:
                    color = queue_color
                else:
                    color = (255, 255, 255)
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
            if hand_detected:
                border_color = (0, 0, 255)  # Red if hands detected
            elif queue_min > 0.02:
                border_color = (0, 255, 255)  # Cyan if significant change detected
            else:
                border_color = (0, 255, 0)  # Green if processing normally
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1]-1, overlay.shape[0]-1), border_color, 3)
            
            # Show the visualization
            cv2.imshow('Experimance Core - Depth Processing', overlay)
            
            # Non-blocking key check (1ms wait)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Visualization quit requested (press 'q' again or Ctrl+C to stop service)")
                
        except Exception as e:
            logger.warning(f"Visualization error: {e}")
        
# ZMQ Message Handler Adapters
    # These methods adapt between the new ZMQ signature (topic, data) and the old message handler signature
    
    def _zmq_handle_agent_control(self, topic: str, data: MessageDataType):
        """ZMQ adapter for agent control messages."""
        try:
            if isinstance(data, dict):
                # Create a task for async execution
                asyncio.create_task(self._handle_agent_control(data))
            else:
                logger.warning(f"Unexpected agent control data type: {type(data)}")
        except Exception as e:
            logger.error(f"Error handling agent control message: {e}")
    
    def _zmq_handle_audio_status(self, topic: str, data: MessageDataType):
        """ZMQ adapter for audio status messages."""
        try:
            if isinstance(data, dict):
                # Create a task for async execution
                asyncio.create_task(self._handle_audio_status(data))
            else:
                logger.warning(f"Unexpected audio status data type: {type(data)}")
        except Exception as e:
            logger.error(f"Error handling audio status message: {e}")

    async def _handle_worker_response(self, worker_name: str, response_data: MessageDataType):
        """Handle responses from workers via pull sockets.
        
        Args:
            worker_name: Name of the worker that sent the response
            response_data: The response data from the worker
        """
        try:
            logger.debug(f"Received response from worker '{worker_name}'")
            
            # Route to appropriate handler based on worker name
            if worker_name == "image_server":
                await self._handle_image_ready_task(response_data)
            elif worker_name == "audio":
                # TODO: Add audio worker response handler when available
                logger.debug(f"Received audio worker response: {response_data}")
            elif worker_name == "display":
                # TODO: Add display worker response handler when available  
                logger.debug(f"Received display worker response: {response_data}")
            else:
                logger.warning(f"Unknown worker response from '{worker_name}': {response_data}")
                
        except Exception as e:
            logger.error(f"Error handling worker response from '{worker_name}': {e}")
            self.record_error(e, is_fatal=False, custom_message=f"Error handling worker response from {worker_name}")


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
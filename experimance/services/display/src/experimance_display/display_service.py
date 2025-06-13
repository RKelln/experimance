#!/usr/bin/env python3
"""
Display Service for the Experimance project.

This service handles all visual rendering for the installation, including:
- Satellite landscape images with crossfade transitions
- Masked video overlays responding to sand interaction
- Text overlays for agent communication
- Custom transition videos between images

The service is designed to be testable independently of ZMQ by providing
a direct interface for triggering display updates.
"""

import asyncio
import logging
import argparse
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import pyglet
from pyglet import clock
from pyglet.window import key

from experimance_common.zmq.subscriber import ZmqSubscriberService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import DEFAULT_PORTS, TICK
from experimance_common.base_service import ServiceState

from .config import DisplayServiceConfig
from .renderers.layer_manager import LayerManager
from .renderers.image_renderer import ImageRenderer
from .renderers.video_overlay_renderer import VideoOverlayRenderer
from .renderers.text_overlay_manager import TextOverlayManager
from .renderers.debug_overlay_renderer import DebugOverlayRenderer

logger = logging.getLogger(__name__)


class DisplayService(ZmqSubscriberService):
    """Main display service that renders the Experimance visual output.
    
    This service subscribes to multiple ZMQ channels and coordinates all visual
    rendering through a layered approach. It also provides a direct interface
    for testing without ZMQ.
    """
    
    def __init__(
        self,
        config: DisplayServiceConfig,
        service_name: Optional[str] = None,
    ):
        """Initialize the Display Service.
        
        Args:
            config: Service configuration object
            service_name: Name of this service instance
        """
        self.config = config
        if service_name is not None:
            self.config.service_name = service_name
        if self.config.service_name is None:
            self.config.service_name = "display-service"
        
        # Initialize ZMQ subscriber service for images channel
        super().__init__(
            service_name=self.config.service_name,
            service_type="display",
            sub_address=self.config.zmq.core_sub_address,
            topics=[
                MessageType.IMAGE_READY,
                MessageType.TRANSITION_READY,
                MessageType.LOOP_READY,
                MessageType.TEXT_OVERLAY,
                MessageType.REMOVE_TEXT,
                MessageType.VIDEO_MASK,
                MessageType.ERA_CHANGED,
            ]
        )
        
        # Pyglet window and rendering components
        self.window = None
        self.layer_manager = None
        self.image_renderer = None
        self.video_overlay_renderer = None
        self.text_overlay_manager = None
        
        # Frame timing
        self.target_fps = 30  # 30fps as per requirements
        self.frame_timer = 0.0
        self.frame_count = 0
        self.fps_display_timer = 0.0
        
        # Direct interface for testing (non-ZMQ control)
        self._direct_handlers: Dict[str, Callable] = {}
        
        logger.info(f"DisplayService initialized: {self.config.service_name}")
    
    async def start(self):
        """Start the display service."""
        logger.info("Starting DisplayService...")
        
        # Initialize pyglet window
        self._initialize_window()
        
        # Initialize rendering components
        self._initialize_renderers()
        
        # Register ZMQ message handlers
        self._register_zmq_handlers()
        
        # Register direct interface handlers for testing
        self._register_direct_handlers()
        
        # Register background tasks before calling super().start()
        self.add_task(self._run_pyglet_loop())
        
        # Start the base ZMQ service
        await super().start()
        
        # DON'T schedule pyglet clock for frame updates - we'll handle this manually
        # clock.schedule_interval(self._update_frame, 1.0 / self.target_fps)

        logger.info(f"DisplayService started on {self.window.width}x{self.window.height}" if self.window else "DisplayService started (no window)")
        
        # Show title screen if enabled
        await self._show_title_screen()
        
        # Show debug text if enabled
        if self.config.display.debug_text:
            await self._show_debug_text()
    
    def _initialize_window(self):
        """Initialize the pyglet window."""
        try:
            # Skip window creation in headless mode
            if self.config.display.headless:
                logger.info("Running in headless mode - no window will be created")
                # Create a mock window object for headless mode
                self.window = self._create_headless_window()
                return
            
            if self.config.display.fullscreen:
                # Get primary monitor for fullscreen
                display = pyglet.display.get_display()
                screen = display.get_default_screen()
                self.window = pyglet.window.Window(
                    fullscreen=True,
                    screen=screen,
                    vsync=self.config.display.vsync
                )
            else:
                # Windowed mode
                width, height = self.config.display.resolution
                self.window = pyglet.window.Window(
                    width=width,
                    height=height,
                    caption=f"Experimance Display - {self.config.service_name}",
                    vsync=self.config.display.vsync  # Re-enable V-sync
                )
            
            # Register window event handlers
            self.window.on_draw = self._on_draw
            self.window.on_key_press = self._on_key_press
            self.window.on_close = self._on_close
            
            logger.info(f"Window initialized: {self.window.width}x{self.window.height}, fullscreen={self.config.display.fullscreen}")
            
        except Exception as e:
            logger.error(f"Failed to initialize window: {e}", exc_info=True)
            self.record_error(e, is_fatal=True)
            raise
    
    def _create_headless_window(self):
        """Create a mock window object for headless mode."""
        class HeadlessWindow:
            def __init__(self, width: int, height: int):
                self.width = width
                self.height = height
                self.fullscreen = False
                self.has_exit = False
                
            def clear(self):
                """Mock clear operation."""
                pass
                
            def close(self):
                """Mock close operation."""
                self.has_exit = True
                
            def flip(self):
                """Mock flip operation."""
                pass
                
            def switch_to(self):
                """Mock switch_to operation."""
                pass
                
            def dispatch_events(self):
                """Mock dispatch_events operation."""
                pass
                
            def dispatch_event(self, event_name, *args):
                """Mock dispatch_event operation."""
                pass
                
            def set_fullscreen(self, fullscreen: bool):
                """Mock set_fullscreen operation."""
                self.fullscreen = fullscreen
        
        width, height = self.config.display.resolution
        return HeadlessWindow(width, height)
    
    def _initialize_renderers(self):
        """Initialize all rendering components."""
        try:
            # Ensure window is created first
            if not self.window:
                error_msg = "Cannot initialize renderers without window"
                logger.error(error_msg)
                self.record_error(RuntimeError(error_msg), is_fatal=True)
                return
                
            window_size = (self.window.width, self.window.height)
            
            # Create layer manager to coordinate rendering order
            self.layer_manager = LayerManager(
                window_size=window_size,
                config=self.config
            )
            
            # Create individual renderers
            self.image_renderer = ImageRenderer(
                window_size=window_size,
                config=self.config.rendering,
                transitions_config=self.config.transitions
            )
            
            self.video_overlay_renderer = VideoOverlayRenderer(
                window_size=window_size,
                config=self.config.rendering,
                transitions_config=self.config.transitions
            )
            
            self.text_overlay_manager = TextOverlayManager(
                window_size=window_size,
                config=self.config.text_styles,
                transitions_config=self.config.transitions
            )
            
            # Create debug overlay renderer
            self.debug_overlay_renderer = DebugOverlayRenderer(
                window_size=window_size,
                config=self.config,
                layer_manager=self.layer_manager
            )
            
            # Register renderers with layer manager (re-enable all renderers)
            self.layer_manager.register_renderer("background", self.image_renderer)
            self.layer_manager.register_renderer("video_overlay", self.video_overlay_renderer)
            self.layer_manager.register_renderer("text_overlay", self.text_overlay_manager)
            self.layer_manager.register_renderer("debug_overlay", self.debug_overlay_renderer)
            
            logger.info("Rendering components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize renderers: {e}", exc_info=True)
            self.record_error(e, is_fatal=True)
            raise
    
    def _register_zmq_handlers(self):
        """Register ZMQ message handlers."""
        self.register_handler(MessageType.IMAGE_READY, self._handle_image_ready)
        self.register_handler(MessageType.TRANSITION_READY, self._handle_transition_ready)
        self.register_handler(MessageType.LOOP_READY, self._handle_loop_ready)
        self.register_handler(MessageType.TEXT_OVERLAY, self._handle_text_overlay)
        self.register_handler(MessageType.REMOVE_TEXT, self._handle_remove_text)
        self.register_handler(MessageType.VIDEO_MASK, self._handle_video_mask)
        self.register_handler(MessageType.ERA_CHANGED, self._handle_era_changed)
        
        logger.info("ZMQ message handlers registered")
    
    def _register_direct_handlers(self):
        """Register direct interface handlers for testing."""
        self._direct_handlers = {
            "image_ready": self._handle_image_ready,
            "text_overlay": self._handle_text_overlay,
            "remove_text": self._handle_remove_text,
            "video_mask": self._handle_video_mask,
            "transition_ready": self._handle_transition_ready,
            "loop_ready": self._handle_loop_ready,
            "era_changed": self._handle_era_changed,
        }
        logger.info("Direct interface handlers registered")
    
    # ZMQ Message Handlers
    
    async def _handle_image_ready(self, message: Dict[str, Any]):
        """Handle ImageReady messages."""
        try:
            logger.debug(f"Received ImageReady: {message}")
            
            # Validate message
            if not self._validate_image_ready(message):
                return
            
            # Pass to image renderer
            if self.image_renderer:
                await self.image_renderer.handle_image_ready(message)
            
        except Exception as e:
            logger.error(f"Error handling ImageReady: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_transition_ready(self, message: Dict[str, Any]):
        """Handle TransitionReady messages."""
        try:
            logger.debug(f"Received TransitionReady: {message}")
            
            # Pass to image renderer for custom transitions
            if self.image_renderer:
                await self.image_renderer.handle_transition_ready(message)
            
        except Exception as e:
            logger.error(f"Error handling TransitionReady: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_loop_ready(self, message: Dict[str, Any]):
        """Handle LoopReady messages (future enhancement)."""
        try:
            logger.debug(f"Received LoopReady: {message}")
            
            # Future: pass to image renderer for animated loops
            # if self.image_renderer:
            #     await self.image_renderer.handle_loop_ready(message)
            logger.info("LoopReady support not yet implemented")
            
        except Exception as e:
            logger.error(f"Error handling LoopReady: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_text_overlay(self, message: Dict[str, Any]):
        """Handle TextOverlay messages."""
        try:
            logger.info(f"Processing TextOverlay message: {message}")
            
            # Validate message
            if not self._validate_text_overlay(message):
                logger.error("TextOverlay validation failed")
                return
            
            logger.info("TextOverlay validation passed")
            
            # Pass to text overlay manager
            if self.text_overlay_manager:
                logger.info("Passing to text overlay manager")
                await self.text_overlay_manager.handle_text_overlay(message)
                logger.info("Text overlay manager processed message")
            else:
                logger.error("Text overlay manager not initialized")
            
        except Exception as e:
            logger.error(f"Error handling TextOverlay: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_remove_text(self, message: Dict[str, Any]):
        """Handle RemoveText messages."""
        try:
            logger.debug(f"Received RemoveText: {message}")
            
            # Pass to text overlay manager
            if self.text_overlay_manager:
                await self.text_overlay_manager.handle_remove_text(message)
            
        except Exception as e:
            logger.error(f"Error handling RemoveText: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_video_mask(self, message: Dict[str, Any]):
        """Handle VideoMask messages."""
        try:
            logger.debug(f"Received VideoMask: {message}")
            
            # Pass to video overlay renderer
            if self.video_overlay_renderer:
                await self.video_overlay_renderer.handle_video_mask(message)
            
        except Exception as e:
            logger.error(f"Error handling VideoMask: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_era_changed(self, message: Dict[str, Any]):
        """Handle EraChanged messages."""
        try:
            logger.debug(f"Received EraChanged: {message}")
            
            # Future: could trigger era-specific display changes
            logger.info(f"Era changed to: {message.get('era', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error handling EraChanged: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    # Direct Interface for Testing
    
    def trigger_display_update(self, update_type: str, data: Dict[str, Any]):
        """Direct interface for triggering display updates (for testing).
        
        Args:
            update_type: Type of update (e.g., "image_ready", "text_overlay")
            data: Update data in the same format as ZMQ messages
        """
        if update_type in self._direct_handlers:
            # Schedule the handler to run in the next frame
            clock.schedule_once(
                lambda dt: asyncio.create_task(self._direct_handlers[update_type](data)),
                0
            )
            logger.debug(f"Scheduled direct update: {update_type}")
        else:
            logger.warning(f"Unknown update type: {update_type}")
    
    # Validation Methods
    
    def _validate_image_ready(self, message: Dict[str, Any]) -> bool:
        """Validate ImageReady message."""
        required_fields = ["image_id", "uri"]
        for field in required_fields:
            if field not in message:
                logger.error(f"ImageReady missing required field: {field}")
                return False
        return True
    
    def _validate_text_overlay(self, message: Dict[str, Any]) -> bool:
        """Validate TextOverlay message."""
        required_fields = ["text_id", "content"]
        for field in required_fields:
            if field not in message:
                logger.error(f"TextOverlay missing required field: {field}")
                return False
        return True
    
    # Pyglet Event Handlers
    
    def _update_frame(self, dt):
        """Update frame timing and components."""

        if not self.config.display.profile:
            # fast path
            if self.layer_manager:
                self.layer_manager.update(dt)
            return
        
        # slow with timing analysis
        frame_start = time.perf_counter()
        
        self.frame_timer += dt
        self.frame_count += 1
        self.fps_display_timer += dt
        
        # Update all rendering components
        if self.layer_manager:
            self.layer_manager.update(dt)
        
        frame_time = time.perf_counter() - frame_start
        
        # Track frame processing times for adaptive timing analysis
        if hasattr(self, 'processing_times'):
            self.processing_times.append(frame_time)
            # Keep only last 100 measurements for rolling average
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
        else:
            self.processing_times = [frame_time]
    
        # Display FPS every second (always log, not just in debug mode)
        if self.fps_display_timer >= 1.0:
            fps = self.frame_count / self.fps_display_timer
            
            # Calculate timing statistics
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            min_time = min(self.processing_times)
            
            # Calculate adaptive timing statistics if available
            adaptive_info = ""
            if hasattr(self, 'adaptive_stats') and self.adaptive_stats['processing_times']:
                avg_proc = sum(self.adaptive_stats['processing_times']) / len(self.adaptive_stats['processing_times'])
                avg_sleep = sum(self.adaptive_stats['sleep_times']) / len(self.adaptive_stats['sleep_times'])
                target_frame_time = 1.0 / self.target_fps
                adaptive_info = f" - Adaptive: proc={avg_proc*1000:.2f}ms, sleep={avg_sleep*1000:.2f}ms, target={target_frame_time*1000:.2f}ms"
            
            logger.info(f"FPS: {fps:.1f} (target: {self.target_fps}, headless: {self.config.display.headless}) - "
                    f"Frame time: avg={avg_time*1000:.2f}ms, max={max_time*1000:.2f}ms, min={min_time*1000:.2f}ms{adaptive_info}")
            self.fps_display_timer = 0.0
            self.frame_count = 0
    
    def _on_draw(self):
        """Pyglet draw handler."""
        import time

        draw_start = time.perf_counter()
        
        if self.window:
            self.window.clear()
        
        render_start = time.perf_counter()
        if self.layer_manager:
            self.layer_manager.render()
        render_time = time.perf_counter() - render_start
        
        total_draw_time = time.perf_counter() - draw_start
        
        # Log draw times occasionally for performance analysis
        if hasattr(self, '_draw_count'):
            self._draw_count += 1
        else:
            self._draw_count = 1
            
        # Log every 30 frames (once per second at 30fps)
        if self.config.display.profile and self._draw_count % 30 == 0:
            logger.debug(f"Draw timing - Total: {total_draw_time*1000:.2f}ms, Render: {render_time*1000:.2f}ms")
    
    def _on_key_press(self, symbol, modifiers):
        """Pyglet key press handler."""
        if symbol == key.ESCAPE or symbol == key.Q:
            # Graceful shutdown
            logger.info("Exit key pressed, shutting down...")
            # Schedule shutdown in the next frame to avoid blocking the event handler
            clock.schedule_once(lambda dt: self.request_stop(), 0)
        elif symbol == key.F11:
            # Toggle fullscreen (for testing)
            if self.window:
                self.window.set_fullscreen(not self.window.fullscreen)
        elif symbol == key.F1:
            # Toggle debug overlay
            self.config.display.debug_overlay = not self.config.display.debug_overlay
            logger.info(f"Debug overlay: {self.config.display.debug_overlay}")
    
    def _on_close(self):
        """Pyglet window close handler."""
        logger.info("Window close event, shutting down...")
        # Schedule shutdown in the next frame to avoid blocking the event handler
        clock.schedule_once(lambda dt: self.request_stop(), 0)
    
    async def _run_pyglet_loop(self):
        """Run the pyglet event loop in an async-friendly way."""
        import time
        logger.info("Starting pyglet event loop...")

        try:
            while self.running:
                frame_start = time.perf_counter()
                
                # In headless mode, just run a simple update loop
                if self.config.display.headless:
                    # Calculate frame time for target FPS
                    frame_dt = 1.0 / self.target_fps  # e.g., 1/30 = 0.033 seconds per frame
                    
                    # Manually call update frame for FPS measurement
                    self._update_frame(frame_dt)
                    
                    # Check for headless shutdown conditions
                    if self.window and hasattr(self.window, 'has_exit') and self.window.has_exit:
                        logger.info("Headless window marked for exit, shutting down...")
                        self.request_stop()
                        break
                    
                    # Calculate actual processing time and adaptive sleep
                    processing_time = time.perf_counter() - frame_start
                    sleep_time = max(0, frame_dt - processing_time)
                    
                    # Track adaptive timing for analysis
                    if self.config.display.profile:
                        if hasattr(self, 'adaptive_stats'):
                            self.adaptive_stats['processing_times'].append(processing_time)
                            self.adaptive_stats['sleep_times'].append(sleep_time)
                            # Keep only last 100 measurements
                            if len(self.adaptive_stats['processing_times']) > 100:
                                self.adaptive_stats['processing_times'].pop(0)
                                self.adaptive_stats['sleep_times'].pop(0)
                        else:
                            self.adaptive_stats = {
                                'processing_times': [processing_time],
                                'sleep_times': [sleep_time]
                            }
                    
                    # Sleep for remaining time to maintain target FPS
                    await asyncio.sleep(sleep_time)
                    continue
                
                # Regular pyglet loop for windowed mode
                # Calculate frame time for target FPS
                frame_dt = 1.0 / self.target_fps
                
                # Process pyglet events
                pyglet.clock.tick()
                
                # Manually handle frame updates at target FPS
                self._update_frame(frame_dt)
                
                # Create a copy of the windows list to avoid iteration issues during window closure
                windows = list(pyglet.app.windows)
                for window in windows:
                    if window.has_exit:
                        continue
                    window.switch_to()
                    window.dispatch_events()
                    window.dispatch_event('on_draw')
                    window.flip()

                # Check if all windows are closed
                if not pyglet.app.windows or all(w.has_exit for w in pyglet.app.windows):
                    logger.info("All windows closed, shutting down...")
                    self.request_stop()
                    break
                
                # Calculate actual processing time and adaptive sleep
                processing_time = time.perf_counter() - frame_start
                sleep_time = max(0.0001, frame_dt - processing_time)
                
                if self.config.display.profile:
                    # Track adaptive timing for analysis
                    if hasattr(self, 'adaptive_stats'):
                        self.adaptive_stats['processing_times'].append(processing_time)
                        self.adaptive_stats['sleep_times'].append(sleep_time)
                        # Keep only last 100 measurements
                        if len(self.adaptive_stats['processing_times']) > 100:
                            self.adaptive_stats['processing_times'].pop(0)
                            self.adaptive_stats['sleep_times'].pop(0)
                    else:
                        self.adaptive_stats = {
                            'processing_times': [processing_time],
                            'sleep_times': [sleep_time]
                        }
                
                # Sleep for remaining time to maintain target FPS
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in pyglet loop: {e}", exc_info=True)
            self.record_error(e, is_fatal=True)
        
        logger.info("Pyglet event loop finished")
    
    async def stop(self):
        """Stop the display service."""
        logger.info("Stopping DisplayService...")
        # Stop ZMQ service (must be first)
        await super().stop()
        
        try:
            # Stop clock updates
            clock.unschedule(self._update_frame)
            
            # Clean up rendering components
            if self.layer_manager:
                await self.layer_manager.cleanup()
            
            # Close pyglet window
            if self.window:
                self.window.close()
                self.window = None
        
            logger.info("DisplayService stopped")
            
        except Exception as e:
            logger.error(f"Error during DisplayService shutdown: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
            raise
    
    async def _show_title_screen(self):
        """Display the title screen if enabled in configuration."""
        if not self.config.title_screen.enabled:
            logger.debug("Title screen disabled in configuration")
            return
        
        if not self.text_overlay_manager:
            logger.warning("Cannot show title screen: TextOverlayManager not initialized")
            return
        
        try:
            logger.info(f"Showing title screen: '{self.config.title_screen.text}'")
            
            # Create title screen text message (positioning is handled by the "title" text style)
            title_message = {
                "text_id": "_title_screen",
                "content": self.config.title_screen.text,
                "speaker": "title",
                "duration": self.config.title_screen.duration,
                # Use title screen specific fade out duration
                "fade_duration": self.config.title_screen.fade_duration
            }
            
            logger.debug(f"Title message: {title_message}")
            
            # Display the title text
            await self.text_overlay_manager.handle_text_overlay(title_message)
            
            logger.debug(f"Title screen will be visible for {self.config.title_screen.duration} seconds")
            
        except Exception as e:
            logger.error(f"Error showing title screen: {e}", exc_info=True)
            # Don't let title screen errors prevent startup
    
    async def _show_debug_text(self):
        """Display debug text in all positions with different speakers."""
        if not self.text_overlay_manager:
            logger.warning("Cannot show debug text: TextOverlayManager not initialized")
            return
        
        try:
            logger.info("Showing debug text in all positions")
            
            # Get all available positions from text overlay manager
            positions = [
                "top_left", "top_center", "top_right",
                "center_left", "center", "center_right", 
                "bottom_left", "bottom_center", "bottom_right"
            ]
            
            # Different speakers to test
            speakers = ["agent", "system", "debug", "title"]
            
            # Create debug text for each position
            for i, position in enumerate(positions):
                speaker = speakers[i % len(speakers)]  # Cycle through speakers
                
                debug_message = {
                    "text_id": f"debug_{position}",
                    "content": f"{speaker.upper()}\n{position}",
                    "speaker": speaker,
                    "duration": None,  # Infinite duration
                    "style": {
                        "position": position,
                        "max_width": None  # Clear max_width to prevent layout issues
                    }
                }
                
                logger.debug(f"Adding debug text: {position} ({speaker})")
                await self.text_overlay_manager.handle_text_overlay(debug_message)
            
            logger.info(f"Debug text displayed in {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"Error showing debug text: {e}", exc_info=True)


async def main():
    """Main entry point for running the display service."""
    parser = argparse.ArgumentParser(description="Experimance Display Service")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default="services/display/config.toml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="display-service",
        help="Service instance name"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level"
    )
    parser.add_argument(
        "--windowed", "-w",
        action="store_true",
        help="Run in windowed mode (overrides config)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no window, for testing)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug overlay"
    )
    parser.add_argument(
        "--debug-text",
        action="store_true",
        help="Show test text in all positions with different speakers"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record and display profiling/performance info"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = DisplayServiceConfig.from_overrides(
        config_file=args.config,
        override_config=vars(args)
    )
    
    # Debug print to check settings
    logger.info(f"Config loaded from {args.config}:")
    logger.info(f"  - Display: fullscreen={config.display.fullscreen}, resolution={config.display.resolution}")
    logger.info(f"  - Rendering: backend={config.rendering.backend}, vsync={config.display.vsync}")
    
    # Override fullscreen if windowed mode requested
    if args.windowed:
        config.display.fullscreen = False
        logger.info(f"Windowed mode requested: fullscreen={config.display.fullscreen}")
    
    # Override headless mode if requested
    if args.headless:
        config.display.headless = True
        logger.info(f"Headless mode enabled: headless={config.display.headless}")
    
    # Override debug if requested
    if args.debug:
        config.display.debug_overlay = True
        logger.info(f"Debug overlay enabled: debug_overlay={config.display.debug_overlay}")
    
    if args.profile:
        config.display.profile = True
        logger.info(f"Profiling enabled: profile={config.display.profile}")

    # Override debug text if requested
    if args.debug_text:
        config.display.debug_text = True
        logger.info("Debug text enabled: will show test text in all positions")
    
    # Create and start the service
    service = DisplayService(
        config=config,
        service_name=args.name,
    )
    
    await service.start()
    logger.info(f"Display service '{args.name}' started successfully")
    
    # Run the main service loop (handles both ZMQ and pyglet)
    await service.run()


def main_sync():
    """Synchronous entry point that can handle asyncio properly."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())

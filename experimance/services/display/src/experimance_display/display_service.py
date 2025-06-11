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
import signal
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import pyglet
from pyglet import clock
from pyglet.window import key

from experimance_common.zmq.subscriber import ZmqSubscriberService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.base_service import ServiceState

from .config import DisplayServiceConfig
from .renderers.layer_manager import LayerManager
from .renderers.image_renderer import ImageRenderer
from .renderers.video_overlay_renderer import VideoOverlayRenderer
from .renderers.text_overlay_manager import TextOverlayManager

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
            sub_address=self.config.zmq.images_sub_address,
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
        
        # Schedule pyglet clock for frame updates
        clock.schedule_interval(self._update_frame, 1.0 / self.target_fps)
        
        logger.info(f"DisplayService started on {self.window.width}x{self.window.height}" if self.window else "DisplayService started (no window)")
    
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
                    vsync=self.config.display.vsync
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
            
            # Register renderers with layer manager
            self.layer_manager.register_renderer("background", self.image_renderer)
            self.layer_manager.register_renderer("video_overlay", self.video_overlay_renderer)
            self.layer_manager.register_renderer("text_overlay", self.text_overlay_manager)
            
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
            logger.debug(f"Received TextOverlay: {message}")
            
            # Validate message
            if not self._validate_text_overlay(message):
                return
            
            # Pass to text overlay manager
            if self.text_overlay_manager:
                await self.text_overlay_manager.handle_text_overlay(message)
            
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
        self.frame_timer += dt
        self.frame_count += 1
        self.fps_display_timer += dt
        
        # Update all rendering components
        if self.layer_manager:
            self.layer_manager.update(dt)
        
        # Display FPS every second if debug is enabled
        if self.config.display.debug_overlay and self.fps_display_timer >= 1.0:
            fps = self.frame_count / self.fps_display_timer
            logger.debug(f"FPS: {fps:.1f}")
            self.fps_display_timer = 0.0
            self.frame_count = 0
    
    def _on_draw(self):
        """Pyglet draw handler."""
        if self.window:
            self.window.clear()
        
        if self.layer_manager:
            self.layer_manager.render()
    
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
    
    # Service Lifecycle
    
    async def run(self):
        """Run the display service main loop."""
        logger.info("Starting display service main loop...")
        
        try:
            # Start the base service (which manages our registered tasks)
            await super().run()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            self.record_error(e, is_fatal=True)
    
    async def _run_pyglet_loop(self):
        """Run the pyglet event loop in an async-friendly way."""
        logger.info("Starting pyglet event loop...")
        
        try:
            while self.running:
                # In headless mode, just run a simple update loop
                if self.config.display.headless:
                    # Update components without rendering
                    if self.layer_manager:
                        self.layer_manager.update(0.001)  # Simulate 1ms time step
                    
                    # Check for headless shutdown conditions
                    if self.window and hasattr(self.window, 'has_exit') and self.window.has_exit:
                        logger.info("Headless window marked for exit, shutting down...")
                        self.request_stop()
                        break
                        
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.001)
                    continue
                
                # Regular pyglet loop for windowed mode
                # Process pyglet events
                pyglet.clock.tick()
                
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
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in pyglet loop: {e}", exc_info=True)
            self.record_error(e, is_fatal=True)
        
        logger.info("Pyglet event loop finished")
    
    async def stop(self):
        """Stop the display service."""
        logger.info("Stopping DisplayService...")
        
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
            
            # Stop ZMQ service
            await super().stop()
            
            logger.info("DisplayService stopped")
            
        except Exception as e:
            logger.error(f"Error during DisplayService shutdown: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
            raise


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
    
    # Create and start the service
    service = DisplayService(
        config=config,
        service_name=args.name,
    )
    
    # Signal handlers are automatically set up by the base service
    # No custom signal handling needed - the base service handles SIGINT/SIGTERM properly
    
    try:
        await service.start()
        logger.info(f"Display service '{args.name}' started successfully")
        
        # Run the main service loop (handles both ZMQ and pyglet)
        await service.run()
        
    except KeyboardInterrupt:
        logger.info("Received Ctrl+C, shutting down...")
        # stop() will be called in the finally block of run()
    except Exception as e:
        logger.error(f"Failed to start display service: {e}", exc_info=True)
        # Ensure cleanup on error
        if service.state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
            await service.stop()


def main_sync():
    """Synchronous entry point that can handle asyncio properly."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())

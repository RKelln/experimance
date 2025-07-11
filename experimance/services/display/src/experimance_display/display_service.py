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

from experimance_common.schemas import ContentType, MessageType, DisplayText, RemoveText
from experimance_display.pyglet_test import MainWindow
from pydantic import ValidationError
import pyglet
from pyglet import clock
from pyglet.window import key

from experimance_common.base_service import BaseService
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import MessageDataType
from experimance_common.constants import DEFAULT_PORTS, TICK, DISPLAY_SERVICE_DIR
from experimance_common.service_state import ServiceState

from .config import DisplayServiceConfig
from .renderers.layer_manager import LayerManager
from .renderers.image_renderer import ImageRenderer
from .renderers.panorama_renderer import PanoramaRenderer
from .renderers.video_overlay_renderer import VideoOverlayRenderer
from .renderers.mask_renderer import MaskRenderer
from .renderers.text_overlay_manager import TextOverlayManager
from .renderers.debug_overlay_renderer import DebugOverlayRenderer
from .renderers.panorama_renderer import PanoramaRenderer

logger = logging.getLogger(__name__)


class DisplayService(BaseService):
    """Main display service that renders the Experimance visual output.
    
    This service subscribes to multiple ZMQ channels and coordinates all visual
    rendering through a layered approach. It also provides a direct interface
    for testing without ZMQ.
    """
    
    def __init__(
        self,
        config: DisplayServiceConfig
    ):
        """Initialize the Display Service.
        
        Args:
            config: Service configuration object
        """
        # Initialize base service
        super().__init__(
            service_name=config.service_name,
            service_type="display"
        )
        
        self.config = config
        logger.debug(f"Initializing DisplayService with config: {self.config}")

        # Initialize ZMQ service using composition
        self.zmq_service = PubSubService(config=config.zmq)
        
        # Pyglet window and rendering components
        self.window = None
        self.layer_manager = None
        self.image_renderer = None
        self.panorama_renderer = None
        self.video_overlay_renderer = None
        self.mask_renderer = None
        self.text_overlay_manager = None
        
        # Frame timing
        self.target_fps = 30  # 30fps as per requirements
        self.frame_timer = 0.0
        self.frame_count = 0
        self.fps_display_timer = 0.0
        
        # Direct interface for testing (non-ZMQ control)
        self._direct_handlers: Dict[str, Callable] = {}
        
        logger.info(f"DisplayService initialized with strategy: {self.config.service_name}")
    
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
        
        # Add rendering task before calling super().start()
        self.add_task(self._run_pyglet_loop())
        
        # Start the ZMQ service
        logger.info(f"🔌 Starting ZMQ service with config: {self.config.zmq}")
        await self.zmq_service.start()
        
        if self.config.zmq.subscriber:
            logger.info(f"✅ ZMQ service started, subscriber listening on {self.config.zmq.subscriber.address}:{self.config.zmq.subscriber.port}")
            logger.info(f"📡 Subscribed to topics: {self.config.zmq.subscriber.topics}")
        else:
            logger.info("✅ ZMQ service started (no subscriber configured)")
        
        # Start the base service
        await super().start()
        
        # DON'T schedule pyglet clock for frame updates - we'll handle this manually
        # clock.schedule_interval(self._update_frame, 1.0 / self.target_fps)

        logger.info(f"DisplayService started on {self.window.width}x{self.window.height}" if self.window else "DisplayService started (no window)")
        
        # Show title screen if enabled
        await self._show_title_screen()

        # wait 5 sec then fade out the video
        # if self.video_overlay_renderer is not None and self.config.video_overlay.enabled:
            
        #     async def fade_out_video_overlay():
        #         logger.info("Waiting 5 seconds before fading out video overlay")
        #         await asyncio.sleep(5)
        #         if self.video_overlay_renderer:
        #             self.video_overlay_renderer.hide_overlay()
        #     self.add_task(asyncio.create_task(fade_out_video_overlay()))

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
                screens = display.get_screens()
                if not screens:
                    raise RuntimeError("No screens available for fullscreen mode")
                if self.config.display.monitor >= len(screens):
                    raise ValueError(f"Invalid monitor index {self.config.display.monitor} for available screens: {len(screens)}")
                screen = screens[self.config.display.monitor]
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
        class HeadlessWindow(pyglet.window.BaseWindow):
            def __init__(self, width: int, height: int):
                self.width = width
                self.height = height
                #self.fullscreen = False
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
                self._fullscreen = fullscreen
        
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
                
            #window_size = (self.window.width, self.window.height)
            
            batch = pyglet.graphics.Batch()
            layer_count = 0

            # Create layer manager to coordinate rendering order
            self.layer_manager = LayerManager(
                config=self.config,
                window=self.window,
                batch=batch,
            )
            
            # Create individual renderers
            # Choose between standard image renderer and panorama renderer
            if self.config.panorama.enabled:
                self.panorama_renderer = PanoramaRenderer(
                    config=self.config,
                    window=self.window,
                    batch=batch,
                    order=(layer_count := layer_count + 1),
                )
                self.layer_manager.register_renderer("panorama", self.panorama_renderer)
                # Set image_renderer to None when using panorama mode
                self.image_renderer = None
                logger.info("Using panorama rendering mode")
            else:
                self.image_renderer = ImageRenderer(
                    config=self.config,
                    window=self.window,
                    batch=batch,
                    order=(layer_count := layer_count + 1),  # Increment layer count for each renderer,
                )
                self.layer_manager.register_renderer("background", self.image_renderer)
                # Set panorama_renderer to None when using standard mode
                self.panorama_renderer = None
                logger.info("Using standard image rendering mode")
            
            if self.config.video_overlay.enabled:
                self.video_overlay_renderer = VideoOverlayRenderer(
                    config=self.config,
                    window=self.window,
                    batch=batch,
                    order=(layer_count := layer_count + 1),
                )
                self.layer_manager.register_renderer("video_overlay", self.video_overlay_renderer)
            
            # Add circular mask renderer (between video and text)
            if self.config.display.mask:
                self.mask_renderer = MaskRenderer(
                    config=self.config,
                    window=self.window,
                    batch=batch,
                    order=(layer_count := layer_count + 1),
                )
                self.layer_manager.register_renderer("mask", self.mask_renderer)
            
            self.text_overlay_manager = TextOverlayManager(
                config=self.config,
                window=self.window,
                batch=batch,
                order=(layer_count := layer_count + 1),
            )
            self.layer_manager.register_renderer("text_overlay", self.text_overlay_manager)
            
            # Create debug overlay renderer
            self.debug_overlay_renderer = DebugOverlayRenderer(
                config=self.config,
                window=self.window,
                batch=batch,
                layer_manager=self.layer_manager,
                order=(layer_count := layer_count + 1),
            )
            self.layer_manager.register_renderer("debug_overlay", self.debug_overlay_renderer)
            
            logger.info("Rendering components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize renderers: {e}", exc_info=True)
            self.record_error(e, is_fatal=True)
            raise
    
    def _register_zmq_handlers(self):
        """Register ZMQ message handlers using the composition pattern."""

        self.zmq_service.add_message_handler(MessageType.DISPLAY_MEDIA, self._handle_display_media)
        self.zmq_service.add_message_handler(MessageType.CHANGE_MAP, self._handle_video_mask)
        self.zmq_service.add_message_handler(MessageType.DISPLAY_TEXT, self._handle_text_overlay)
        self.zmq_service.add_message_handler(MessageType.REMOVE_TEXT, self._handle_remove_text)
        
        logger.info("ZMQ message handlers registered using composition pattern")
    
    def _register_direct_handlers(self):
        """Register direct interface handlers for testing."""
        self._direct_handlers = {
            "text_overlay": self._handle_text_overlay,
            "remove_text": self._handle_remove_text,
            "change_map": self._handle_video_mask,
            "display_media": self._handle_display_media,
            "set_mask_visibility": self._handle_set_mask_visibility,
        }
        logger.info("Direct interface handlers registered")
    
    # ZMQ Message Handlers
    async def _handle_display_media(self, message: MessageDataType):
        """Handle DisplayMedia messages."""
        try:
            logger.info(f"🎯 RECEIVED DisplayMedia message: {message}")
            
            if message is None:
                logger.error("Received None message in handle_display_media")
                return
            if message.get("type") != MessageType.DISPLAY_MEDIA:
                logger.error(f"Invalid display media message type: {message.get('type')}")
                return
            
            logger.debug(f"Received DisplayMedia: {message.get('content_type')}")
            
            # Route to appropriate renderer based on configuration and content type
            match message.get("content_type"):
                case ContentType.IMAGE:
                    logger.info(f"📸 Routing IMAGE to renderer (panorama={bool(self.panorama_renderer)})")
                    # Route to panorama renderer if enabled, otherwise standard image renderer
                    if self.panorama_renderer:
                        self.panorama_renderer.handle_display_media(message)
                    elif self.image_renderer:
                        await self.image_renderer.handle_display_media(message)

                case ContentType.VIDEO:
                    logger.info("🎬 Routing VIDEO to video overlay renderer")
                    if self.video_overlay_renderer:
                        logger.debug("Handling video overlay")
                        #await self.video_overlay_renderer.handle_display_media(message)
            
                case ContentType.IMAGE_SEQUENCE:
                    logger.info("🎞️ Routing IMAGE_SEQUENCE to image renderer")
                    if self.image_renderer:
                        logger.debug("Handling video sequence")
                        #await self.image_renderer.handle_image_sequence(message)

                case ContentType.DEBUG_DEPTH:
                    logger.info("🔧 Routing DEBUG_DEPTH to debug renderer")
                    # Debug images go to standard image renderer even in panorama mode
                    if self.image_renderer:
                        await self.image_renderer.handle_display_media(message)
                    elif self.panorama_renderer:
                        self.panorama_renderer.handle_display_media(message)

            # now that we've recieved a display media message, we can remove the change map
            if self.video_overlay_renderer is not None:
                logger.debug("Removing video mask after display media")
                self.video_overlay_renderer.hide_overlay()

        except Exception as e:
            logger.error(f"Error handling DisplayMedia: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)

    async def _handle_text_overlay(self, message: MessageDataType):
        """Handle TextOverlay messages."""
        try:
            logger.info(f"Processing TextOverlay message: {message}")
            
            try:
                display_text: DisplayText = DisplayText.to_message_type(message)  # type: ignore
            except ValidationError as e:
                self.record_error(
                    ValueError(f"Invalid DisplayText message: {message}"),
                    is_fatal=False,
                    custom_message=f"Invalid DisplayText message: {message}"
                )
                return

            # Pass to text overlay manager
            if self.text_overlay_manager:
                await self.text_overlay_manager.handle_text_overlay(display_text)
            else:
                self.record_error(Exception("Text overlay manager not initialized"), is_fatal=False)
            
        except Exception as e:
            self.record_error(e, is_fatal=False)
    
    async def _handle_remove_text(self, message: MessageDataType):
        """Handle RemoveText messages."""
        try:
            logger.debug(f"Received RemoveText: {message}")
            
            # Pass to text overlay manager
            if self.text_overlay_manager:
                await self.text_overlay_manager.handle_remove_text(message)
            
        except Exception as e:
            logger.error(f"Error handling RemoveText: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
    
    async def _handle_video_mask(self, message: MessageDataType):
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
    
    def trigger_display_update(self, update_type: str, data: MessageDataType):
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
    
    def _validate_image_ready(self, message: MessageDataType) -> bool:
        """Validate ImageReady message."""
        required_fields = ["image_id", "uri"]
        for field in required_fields:
            if field not in message:
                logger.error(f"ImageReady missing required field: {field}")
                return False
        return True
    
    def _validate_text_overlay(self, message: MessageDataType) -> bool:
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
        
        try:
            # Stop ZMQ service first to ensure subscriber tasks are cancelled
            if self.zmq_service and self.zmq_service.is_running():
                logger.debug("Stopping ZMQ service...")
                await self.zmq_service.stop()
                logger.debug("ZMQ service stopped")
            
            # Stop clock updates
            clock.unschedule(self._update_frame)
            
            # Clean up rendering components
            if self.layer_manager:
                await self.layer_manager.cleanup()
            
            # Close pyglet window
            if self.window:
                self.window.close()
                self.window = None
        
            logger.info("DisplayService components stopped")
            
        except Exception as e:
            logger.error(f"Error during DisplayService component shutdown: {e}", exc_info=True)
            self.record_error(e, is_fatal=False)
            # Continue with base service stop even if components fail
        
        # Stop base service last
        await super().stop()
        
        logger.info("DisplayService stopped")
    
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
            title_message = DisplayText(
                text_id = "_title_screen",
                content = self.config.title_screen.text,
                speaker = "title",
                duration = self.config.title_screen.duration,
                # Use title screen specific fade out duration
                fade_duration = self.config.title_screen.fade_duration
            )
            
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

                debug_message = DisplayText(
                    text_id=f"debug_{position}",
                    content=f"{speaker.upper()}\n{position}",
                    speaker=speaker,
                    duration=None,  # Infinite duration
                    style={
                        "position": position,
                        "max_width": None  # Clear max_width to prevent layout issues
                    }
                )

                logger.debug(f"Adding debug text: {position} ({speaker})")
                await self.text_overlay_manager.handle_text_overlay(debug_message)
            
            logger.info(f"Debug text displayed in {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"Error showing debug text: {e}", exc_info=True)
    
    # Mask Control Methods
    
    def set_mask_visibility(self, visible: bool):
        """Set the mask visibility.
        
        Args:
            visible: Whether the mask should be visible
        """
        if self.mask_renderer:
            self.mask_renderer.set_visible(visible)
            logger.info(f"Mask visibility updated: {visible}")
        else:
            logger.warning("Cannot set mask visibility: mask renderer not initialized")
    
    def set_mask_opacity(self, opacity: float):
        """Set the mask opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        if self.mask_renderer:
            self.mask_renderer.set_opacity(opacity)
            logger.info(f"Mask opacity updated: {opacity}")
        else:
            logger.warning("Cannot set mask opacity: mask renderer not initialized")
    
    async def _handle_set_mask_visibility(self, message: MessageDataType):
        """Handle direct mask visibility updates."""
        try:
            visible = message.get("visible", True)
            self.set_mask_visibility(visible)
            
        except Exception as e:
            logger.error(f"Error handling mask visibility update: {e}", exc_info=True)
        

async def run_display_service(
    config_path: str = "config.toml", 
    args: Optional[argparse.Namespace] = None
) -> None:
    """
    Run the Experimance Display Service with CLI integration.
    
    Args:
        config_path: Path to configuration file
        args: CLI arguments from argparse (for config overrides)
    """
    # Create config with CLI overrides
    config = DisplayServiceConfig.from_overrides(
        config_file=config_path,
        args=args  # CLI args automatically override config values
    )
    
    service = DisplayService(
        config=config
    )
    await service.start()
    await service.run()

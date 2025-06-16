"""
Depth Processor Visualization Module

This module provides a reusable depth frame visualization component that can be used
by both the test scripts and the core service for debugging and monitoring.
"""
import time
from typing import Optional, Callable, Any
import cv2
import numpy as np

from experimance_core.config import DepthFrame


class DepthVisualizer:
    """
    A reusable visualization component for depth processing debugging.
    
    This class creates a composite view showing all processing steps in a grid layout.
    It can be used both in test scripts and in the core service when debug mode is enabled.
    """
    
    def __init__(
        self, 
        window_name: str = "Depth Debug Visualization",
        window_size: tuple = (1200, 800),
        grid_size: tuple = (2, 3),
        cell_size: tuple = (400, 400)
    ):
        """
        Initialize the depth visualizer.
        
        Args:
            window_name: Name of the OpenCV window
            window_size: Window size (width, height)
            grid_size: Grid layout (rows, cols)
            cell_size: Size of each grid cell (width, height)
        """
        self.window_name = window_name
        self.window_size = window_size
        self.grid_rows, self.grid_cols = grid_size
        self.cell_width, self.cell_height = cell_size
        self.window_created = False
        
        # State for interactive mode
        self.paused = False
        self.start_time = time.time()
        self.frame_count = 0
        
    def create_window(self):
        """Create and configure the visualization window."""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, *self.window_size)
            self.window_created = True
    
    def destroy_window(self):
        """Destroy the visualization window."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
    
    def _resize_preserve_aspect(self, img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        """Resize image preserving aspect ratio to fit within max dimensions."""
        if img is None or img.size == 0:
            return np.zeros((max_height, max_width, 3), dtype=np.uint8)
        
        h, w = img.shape[:2]
        
        # Calculate scale factor to fit within max dimensions
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and center the image
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        y_offset = (max_height - new_h) // 2
        x_offset = (max_width - new_w) // 2
        
        # Place resized image on canvas
        if len(resized.shape) == 2:  # Grayscale
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _add_image_to_grid(self, composite: np.ndarray, img: Optional[np.ndarray], row: int, col: int, title: str):
        """Add an image to the specified grid position with title."""
        if img is None:
            # Create placeholder
            placeholder = np.zeros((self.cell_height-40, self.cell_width-20, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Data", (self.cell_width//2-50, self.cell_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            img_resized = placeholder
        else:
            # Ensure image is the right type
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:  # Already BGR
                pass
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Resize preserving aspect ratio (leaving space for title)
            img_resized = self._resize_preserve_aspect(img, self.cell_width-20, self.cell_height-40)
        
        # Calculate position
        y_start = row * self.cell_height + 30
        y_end = y_start + (self.cell_height-40)
        x_start = col * self.cell_width + 10
        x_end = x_start + (self.cell_width-20)
        
        # Add image to composite
        composite[y_start:y_end, x_start:x_end] = img_resized
        
        # Add title
        cv2.putText(composite, title, (x_start, row * self.cell_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add border
        cv2.rectangle(composite, (x_start-2, y_start-2), (x_end+2, y_end+2), (64, 64, 64), 1)
    
    def create_composite_image(self, frame: DepthFrame) -> np.ndarray:
        """Create a composite image showing all processing steps."""
        # Create blank composite image
        composite = np.zeros((self.grid_rows * self.cell_height, self.grid_cols * self.cell_width, 3), dtype=np.uint8)
        
        # Add all images to grid
        self._add_image_to_grid(composite, frame.depth_image, 0, 0, "Final Output")
        
        if frame.has_debug_images:
            self._add_image_to_grid(composite, frame.raw_depth_image, 0, 1, "Raw Depth")
            self._add_image_to_grid(composite, frame.importance_mask, 0, 2, "Importance Mask")
            self._add_image_to_grid(composite, frame.masked_image, 1, 0, "Masked Image")
            self._add_image_to_grid(composite, frame.change_diff_image, 1, 1, "Change Diff")
            self._add_image_to_grid(composite, frame.hand_detection_image, 1, 2, "Hand Detection")
        else:
            # Show placeholders
            for i, title in enumerate(["Raw Depth", "Importance Mask", "Masked Image", "Change Diff", "Hand Detection"]):
                row = (i + 1) // 3
                col = (i + 1) % 3
                self._add_image_to_grid(composite, None, row, col, f"{title} (Debug Off)")
        
        # Add info overlay
        info_y = self.grid_rows * self.cell_height - 20
        info_color = (0, 255, 255)  # Yellow
        cv2.putText(composite, f"Frame: {frame.frame_number}", (10, info_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
        cv2.putText(composite, f"Hand: {frame.hand_detected}", (150, info_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
        cv2.putText(composite, f"Change: {frame.change_score:.3f}", (280, info_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
        cv2.putText(composite, "Press 'q'=quit, 'p'=pause, 's'=save", (450, info_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
        
        return composite
    
    def display_frame(self, frame: DepthFrame, show_fps: bool = True) -> bool:
        """
        Display a single frame in the visualization window.
        
        Args:
            frame: The depth frame to display
            show_fps: Whether to print FPS info to console
            
        Returns:
            True if should continue, False if user requested quit
        """
        if not self.paused:
            self.frame_count += 1
            
            # Create and display composite image
            composite = self.create_composite_image(frame)
            self.create_window()  # Ensure window exists
            cv2.imshow(self.window_name, composite)
            
            # Show frame info in terminal
            if show_fps:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                print(f"\r📊 Frame {frame.frame_number}: "
                      f"Hand={frame.hand_detected}, "
                      f"Change={frame.change_score:.3f}, "
                      f"FPS={fps:.1f}, "
                      f"Elapsed={elapsed:.1f}s", end="", flush=True)
        
        # Handle key presses and window close
        key = cv2.waitKey(1) & 0xFF
        
        # Check if window was closed
        if self.window_created and cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\n🚪 Window closed by user")
            return False
            
        if key == ord('q'):
            print("\n👋 Quit requested")
            return False
        elif key == ord('p'):
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            print(f"\n⏸️  {status}")
        elif key == ord('s'):
            # Save current images
            timestamp = int(time.time())
            composite = self.create_composite_image(frame)
            cv2.imwrite(f"debug_composite_{timestamp}.png", composite)
            cv2.imwrite(f"debug_final_{timestamp}.png", frame.depth_image)
            if frame.has_debug_images and frame.raw_depth_image is not None:
                cv2.imwrite(f"debug_raw_{timestamp}.png", frame.raw_depth_image)
                if frame.importance_mask is not None:
                    cv2.imwrite(f"debug_mask_{timestamp}.png", frame.importance_mask)
                if frame.masked_image is not None:
                    cv2.imwrite(f"debug_masked_{timestamp}.png", frame.masked_image)
            print(f"\n💾 Saved debug images with timestamp {timestamp}")
        
        return True
    
    def get_stats(self) -> dict:
        """Get current visualization statistics."""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "fps": fps,
            "paused": self.paused
        }


class DepthVisualizationContext:
    """
    Context manager for depth visualization that handles window lifecycle.
    
    Example usage:
        with DepthVisualizationContext() as visualizer:
            async for frame in processor.stream_frames():
                if not visualizer.display_frame(frame):
                    break
    """
    
    def __init__(self, **kwargs):
        """Initialize with visualizer arguments."""
        self.visualizer_kwargs = kwargs
        self.visualizer = None
    
    def __enter__(self) -> DepthVisualizer:
        """Enter the context and create visualizer."""
        self.visualizer = DepthVisualizer(**self.visualizer_kwargs)
        return self.visualizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and cleanup."""
        if self.visualizer:
            self.visualizer.destroy_window()
            stats = self.visualizer.get_stats()
            print(f"\n📈 Visualization Stats: {stats['frame_count']} frames "
                  f"in {stats['elapsed_time']:.1f}s ({stats['fps']:.1f} FPS)")

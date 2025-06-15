from experimance_core.robust_camera import (
    CameraConfig,
    DepthProcessor,
    DepthFrame,
    mask_bright_area,
    simple_obstruction_detect,
    is_blank_frame,
    detect_difference
)
from experimance_core.mock_depth_processor import MockDepthProcessor
from experimance_core.depth_factory import create_depth_processor
from experimance_core.config import DEFAULT_CONFIG_PATH, DEFAULT_CAMERA_CONFIG_DIR
import asyncio
import cv2
import numpy as np
import time
import pyrealsense2 as rs
import logging
import argparse
import traceback
import toml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
$ uv run python services/core/tests/test_camera.py
This script provides a standalone testing suite for the robust camera module, 
allowing users to test both mock and real camera functionality, 
as well as image processing functions.
"""

def create_test_config(verbose: bool = False, safe_mode: bool = False) -> CameraConfig:
    """Create a test configuration, loading values from config.toml if available."""
    # Try to load configuration from the default config path
    config_path = Path(DEFAULT_CONFIG_PATH)
    depth_config = {}
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                toml_config = toml.load(f)
                depth_config = toml_config.get('depth_processing', {})
                print(f"📋 Loaded configuration from {config_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load {config_path}: {e}")
            print("📋 Using default configuration")
    else:
        print(f"📋 No config file found at {config_path}, using defaults")
    
    # Extract values from config with defaults
    resolution = tuple(depth_config.get('resolution', [640, 480]))
    fps = depth_config.get('fps', 30)
    min_depth = depth_config.get('min_depth', 0.0)
    max_depth = depth_config.get('max_depth', 10.0)
    change_threshold = depth_config.get('change_threshold', 60)
    output_size = tuple(depth_config.get('output_size', [1024, 1024]))
    camera_config_path = depth_config.get('camera_config_path', None)
    if camera_config_path:
        camera_config_path = str(Path(DEFAULT_CAMERA_CONFIG_DIR) / camera_config_path)
    
    # In safe mode, skip advanced config
    if safe_mode:
        print("🛡️  Safe mode enabled - skipping advanced camera configuration")
        camera_config_path = None
    else:
        # For testing, skip the JSON config file if it doesn't exist
        if camera_config_path and not Path(camera_config_path).exists():
            print(f"⚠️  Camera config file {camera_config_path} not found, skipping advanced config")
            camera_config_path = None
    
    # Extract filter settings
    enable_filters = depth_config.get('enable_filters', True)
    spatial_filter = depth_config.get('spatial_filter', True)
    temporal_filter = depth_config.get('temporal_filter', True)
    decimation_filter = depth_config.get('decimation_filter', False)
    hole_filling_filter = depth_config.get('hole_filling_filter', True)
    threshold_filter = depth_config.get('threshold_filter', False)
    
    # Spatial filter settings
    spatial_filter_magnitude = depth_config.get('spatial_filter_magnitude', 2.0)
    spatial_filter_alpha = depth_config.get('spatial_filter_alpha', 0.5)
    spatial_filter_delta = depth_config.get('spatial_filter_delta', 20.0)
    spatial_filter_hole_fill = depth_config.get('spatial_filter_hole_fill', 1)
    
    # Temporal filter settings
    temporal_filter_alpha = depth_config.get('temporal_filter_alpha', 0.4)
    temporal_filter_delta = depth_config.get('temporal_filter_delta', 20.0)
    temporal_filter_persistence = depth_config.get('temporal_filter_persistence', 3)
    
    # Decimation filter settings
    decimation_filter_magnitude = depth_config.get('decimation_filter_magnitude', 2)
    
    # Hole filling filter settings
    hole_filling_mode = depth_config.get('hole_filling_mode', 1)
    
    # Threshold filter settings
    threshold_filter_min = depth_config.get('threshold_filter_min', 0.15)
    threshold_filter_max = depth_config.get('threshold_filter_max', 4.0)
    
    return CameraConfig(
        resolution=resolution,
        fps=fps,
        min_depth=min_depth,
        max_depth=max_depth,
        change_threshold=change_threshold,
        output_resolution=output_size,
        json_config_path=camera_config_path,
        detect_hands=True,
        crop_to_content=True,
        warm_up_frames=5,
        max_retries=3,
        verbose_performance=verbose,  # Enable performance logging if verbose
        skip_advanced_config=safe_mode,  # Skip advanced config in safe mode
        # Filter settings
        enable_filters=enable_filters,
        spatial_filter=spatial_filter,
        temporal_filter=temporal_filter,
        decimation_filter=decimation_filter,
        hole_filling_filter=hole_filling_filter,
        threshold_filter=threshold_filter,
        spatial_filter_magnitude=spatial_filter_magnitude,
        spatial_filter_alpha=spatial_filter_alpha,
        spatial_filter_delta=spatial_filter_delta,
        spatial_filter_hole_fill=spatial_filter_hole_fill,
        temporal_filter_alpha=temporal_filter_alpha,
        temporal_filter_delta=temporal_filter_delta,
        temporal_filter_persistence=temporal_filter_persistence,
        decimation_filter_magnitude=decimation_filter_magnitude,
        hole_filling_mode=hole_filling_mode,
        threshold_filter_min=threshold_filter_min,
        threshold_filter_max=threshold_filter_max
    )


async def test_mock_processor(duration: int = 0, verbose: bool = False, safe_mode: bool = False):
    """Test the mock depth processor for a specified duration."""
    if duration > 0:
        print(f"🧪 Testing MockDepthProcessor for {duration} seconds...")
    else:
        print("🧪 Testing MockDepthProcessor (infinite duration - press Ctrl+C to stop)...")
    
    config = create_test_config(verbose, safe_mode)
    processor = MockDepthProcessor(config)
    
    success = await processor.initialize()
    if not success:
        print("❌ Failed to initialize mock processor")
        return
    
    print("✅ Mock processor initialized")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        async for frame in processor.stream_frames():
            frame_count += 1
            elapsed = time.time() - start_time
            
            print(f"Frame {frame_count}: "
                  f"hands={frame.hand_detected}, "
                  f"change={frame.change_score:.3f}, "
                  f"shape={frame.depth_image.shape}")
            
            if duration > 0 and elapsed >= duration:
                break
                
        print(f"✅ Processed {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count/elapsed:.1f} fps)")
              
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        processor.stop()


async def test_real_camera(duration: int = 0, verbose: bool = False, safe_mode: bool = False):
    """Test the real camera for a specified duration."""
    if duration > 0:
        print(f"📷 Testing RealSense camera for {duration} seconds...")
    else:
        print("📷 Testing RealSense camera (infinite duration - press Ctrl+C to stop)...")
    print("⚠️  Make sure RealSense camera is connected!")
    if safe_mode:
        print("🛡️  Running in safe mode - advanced configuration disabled")
    
    config = create_test_config(verbose, safe_mode)
    processor = DepthProcessor(config)
    
    print("🔄 Initializing camera...")
    success = await processor.initialize()
    if not success:
        print("❌ Failed to initialize camera - check connection and drivers")
        return
    
    print("✅ Camera initialized successfully")
    
    frame_count = 0
    hand_detections = 0
    start_time = time.time()
    
    try:
        async for frame in processor.stream_frames():
            frame_count += 1
            elapsed = time.time() - start_time
            
            if frame.hand_detected:
                hand_detections += 1
                # print(f"👋 Frame {frame_count}: HANDS DETECTED! "
                #       f"change={frame.change_score:.3f}")
            elif frame_count % 30 == 0:  # Print every 30 frames
                print(f"Frame {frame_count}: "
                      f"change={frame.change_score:.3f}, "
                      f"fps={frame_count/elapsed:.1f}")
            
            if duration > 0 and elapsed >= duration:
                break
                
        print(f"✅ Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"👋 Hand detections: {hand_detections}/{frame_count} "
              f"({100*hand_detections/frame_count:.1f}%)")
              
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Camera error: {e}")
    finally:
        processor.stop()


async def test_image_functions():
    """Test the standalone image processing functions."""
    print("🖼️  Testing image processing functions...")
    
    # Create a test image
    test_image = np.zeros((480, 640), dtype=np.uint8)
    
    # Add a bright center region
    center_y, center_x = 240, 320
    cv2.circle(test_image, (center_x, center_y), 100, (200,), -1)
    
    # Test mask_bright_area
    print("Testing mask_bright_area...")
    mask = mask_bright_area(test_image)
    bright_pixels = cv2.countNonZero(mask)
    print(f"✅ Bright area mask created: {bright_pixels} bright pixels")
    
    # Test obstruction detection
    print("Testing simple_obstruction_detect...")
    obstruction = simple_obstruction_detect(test_image)
    print(f"✅ Obstruction detection: {obstruction}")
    
    # Test blank frame detection
    print("Testing is_blank_frame...")
    blank_result = is_blank_frame(test_image)
    print(f"✅ Blank frame detection: {blank_result}")
    
    # Test difference detection
    print("Testing detect_difference...")
    test_image2 = test_image.copy()
    cv2.circle(test_image2, (center_x + 50, center_y), 50, (150,), -1)
    
    diff_score, _ = detect_difference(test_image, test_image2)
    print(f"✅ Difference detection: {diff_score} different pixels")
    
    print("✅ All image processing functions tested")


def show_camera_info():
    """Show information about connected RealSense cameras."""
    print("📷 Scanning for RealSense cameras...")
    
    try:
        ctx = rs.context() # type: ignore
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ No RealSense cameras found")
            print("   - Check USB connection")
            print("   - Check if camera is being used by another process")
            print("   - Try: sudo udevadm control --reload-rules && sudo udevadm trigger")
            return
        
        for i, device in enumerate(devices):
            print(f"\n📷 Camera {i + 1}:")
            print(f"   Name: {device.get_info(rs.camera_info.name)}") # type: ignore 
            print(f"   Serial: {device.get_info(rs.camera_info.serial_number)}") # type: ignore
            print(f"   Firmware: {device.get_info(rs.camera_info.firmware_version)}") # type: ignore
            
            # Show available sensors
            sensors = device.query_sensors()
            print(f"   Sensors: {len(sensors)} available")
            for j, sensor in enumerate(sensors):
                print(f"     {j + 1}. {sensor.get_info(rs.camera_info.name)}") # type: ignore
        
        print(f"\n✅ Found {len(devices)} RealSense camera(s)")
        
    except Exception as e:
        print(f"❌ Error scanning cameras: {e}")


async def interactive_mode():
    """Interactive mode for testing different camera functions."""
    print("\n🎮 Interactive Camera Testing Mode")
    print("=" * 40)
    
    while True:
        print("\nChoose a test:")
        print("1. 📷 Test real camera (10s)")
        print("2. 🧪 Test mock processor (10s)")
        print("3. 🖼️  Test image functions")
        print("4. 📋 Show camera info")
        print("5. 🏃 Custom duration test")
        print("6. 🎬 Test camera visualization")
        print("0. 🚪 Exit")
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == "1":
                await test_real_camera()
            elif choice == "2":
                await test_mock_processor()
            elif choice == "3":
                await test_image_functions()
            elif choice == "4":
                show_camera_info()
            elif choice == "5":
                duration = int(input("Enter duration in seconds: "))
                camera_type = input("Real camera or mock? (r/m): ").lower()
                if camera_type.startswith('r'):
                    await test_real_camera(duration)
                else:
                    await test_mock_processor(duration)
            elif choice == "6":
                duration = int(input("Enter duration for visualization test (seconds): "))
                camera_type = input("Mock camera or real camera? (m/r): ").lower()
                use_mock = camera_type.startswith('m')
                await test_camera_visualization(use_mock, duration)
            elif choice == "0" or choice.lower() == "q":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice, please try again")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except ValueError:
            print("❌ Invalid input, please try again")
        except Exception as e:
            print(f"❌ Error: {e}")


async def test_camera_visualization(use_mock: bool = True, duration: int = 0):
    """
    Test camera with debug visualization showing intermediate processing steps.
    
    Args:
        use_mock: Whether to use mock camera (True) or real camera (False)
        duration: How long to run the visualization test (0 = infinite)
    """
    print(f"🔍 Testing Camera Visualization ({'Mock' if use_mock else 'Real'} Camera)")
    if duration > 0:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: infinite (close window or press 'q' to quit)")
    print("Press 'q' to quit, 'p' to pause/unpause, 's' to save current images")
    print("=" * 60)
    
    # Use the same config loading as other tests, but enable debug mode and adjust for visualization
    config = create_test_config(verbose=True)  # Enable verbose for visualization
    
    # Override specific settings for better visualization experience
    config.output_resolution = (512, 512)  # Smaller for display
    config.warm_up_frames = 3  # Faster warmup for testing
    config.debug_mode = True  # Enable debug images
    config.verbose_performance = True  # Show performance stats
    
    # Create processor
    if use_mock:
        processor = MockDepthProcessor(config)
    else:
        processor = DepthProcessor(config)
    
    # Initialize
    if not await processor.initialize():
        print("❌ Failed to initialize camera processor")
        return
    
    print("✅ Camera processor initialized")
    print("Opening composite visualization window...")
    
    # Create single composite window
    cv2.namedWindow("Camera Debug Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Debug Visualization", 1200, 800)
    
    def create_composite_image(frame: DepthFrame) -> np.ndarray:
        """Create a composite image showing all processing steps."""
        # Define grid size (2 rows x 3 columns)
        rows, cols = 2, 3
        cell_height, cell_width = 400, 400
        
        # Create blank composite image
        composite = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
        
        def resize_preserve_aspect(img, max_width, max_height):
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
        
        def add_image_to_grid(img, row, col, title):
            """Add an image to the specified grid position with title."""
            if img is None:
                # Create placeholder
                placeholder = np.zeros((cell_height-40, cell_width-20, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Data", (cell_width//2-50, cell_height//2), 
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
                img_resized = resize_preserve_aspect(img, cell_width-20, cell_height-40)
            
            # Calculate position
            y_start = row * cell_height + 30
            y_end = y_start + (cell_height-40)
            x_start = col * cell_width + 10
            x_end = x_start + (cell_width-20)
            
            # Add image to composite
            composite[y_start:y_end, x_start:x_end] = img_resized
            
            # Add title
            cv2.putText(composite, title, (x_start, row * cell_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add border
            cv2.rectangle(composite, (x_start-2, y_start-2), (x_end+2, y_end+2), (64, 64, 64), 1)
        
        # Add all images to grid
        add_image_to_grid(frame.depth_image, 0, 0, "Final Output")
        
        if frame.has_debug_images:
            add_image_to_grid(frame.raw_depth_image, 0, 1, "Raw Depth")
            add_image_to_grid(frame.importance_mask, 0, 2, "Importance Mask")
            add_image_to_grid(frame.masked_image, 1, 0, "Masked Image")
            add_image_to_grid(frame.change_diff_image, 1, 1, "Change Diff")
            add_image_to_grid(frame.hand_detection_image, 1, 2, "Hand Detection")
        else:
            # Show placeholders
            for i, title in enumerate(["Raw Depth", "Importance Mask", "Masked Image", "Change Diff", "Hand Detection"]):
                row = (i + 1) // 3
                col = (i + 1) % 3
                add_image_to_grid(None, row, col, f"{title} (Debug Off)")
        
        # Add info overlay
        info_y = rows * cell_height - 20
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
    
    try:
        start_time = time.time()
        frame_count = 0
        paused = False
        
        print("🎬 Starting visualization stream...")
        
        async for frame in processor.stream_frames():
            if not paused:
                frame_count += 1
                
                # Create and display composite image
                composite = create_composite_image(frame)
                cv2.imshow("Camera Debug Visualization", composite)
                
                # Show frame info in terminal
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"\r📊 Frame {frame.frame_number}: "
                      f"Hand={frame.hand_detected}, "
                      f"Change={frame.change_score:.3f}, "
                      f"FPS={fps:.1f}, "
                      f"Elapsed={elapsed:.1f}s", end="", flush=True)
            
            # Handle key presses and window close
            key = cv2.waitKey(1) & 0xFF
            
            # Check if window was closed
            if cv2.getWindowProperty("Camera Debug Visualization", cv2.WND_PROP_VISIBLE) < 1:
                print("\n🚪 Window closed by user")
                break
                
            if key == ord('q'):
                print("\n👋 Quit requested")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n⏸️  {status}")
            elif key == ord('s'):
                # Save current images
                timestamp = int(time.time())
                # Save composite image
                cv2.imwrite(f"debug_composite_{timestamp}.png", composite)
                # Save individual images
                cv2.imwrite(f"debug_final_{timestamp}.png", frame.depth_image)
                if frame.has_debug_images and frame.raw_depth_image is not None:
                    cv2.imwrite(f"debug_raw_{timestamp}.png", frame.raw_depth_image)
                    if frame.importance_mask is not None:
                        cv2.imwrite(f"debug_mask_{timestamp}.png", frame.importance_mask)
                    if frame.masked_image is not None:
                        cv2.imwrite(f"debug_masked_{timestamp}.png", frame.masked_image)
                print(f"\n💾 Saved debug images (including composite) with timestamp {timestamp}")
            
            # Check duration
            if duration > 0 and time.time() - start_time >= duration:
                print(f"\n⏰ Duration limit ({duration}s) reached")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        processor.stop()
        cv2.destroyAllWindows()
        
        final_elapsed = time.time() - start_time
        final_fps = frame_count / final_elapsed if final_elapsed > 0 else 0
        print(f"\n📈 Final Stats: {frame_count} frames in {final_elapsed:.1f}s ({final_fps:.1f} FPS)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Robust Camera Module - Standalone Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_camera.py                      # Interactive mode
  python test_camera.py --mock               # Test mock processor (infinite)
  python test_camera.py --real               # Test real camera (infinite)
  python test_camera.py --mock --visualize   # Visual debug with mock camera
  python test_camera.py --real --visualize   # Visual debug with real camera
  python test_camera.py --info               # Show camera info
  python test_camera.py --functions          # Test image functions
  python test_camera.py --real --duration 30 # Test real camera for 30s
  python test_camera.py --visualize          # Visual debug (defaults to mock)
  python test_camera.py --mock --duration 0  # Infinite duration (default)
        """
    )
    
    parser.add_argument("--mock", action="store_true", 
                       help="Use mock depth processor")
    parser.add_argument("--real", action="store_true", 
                       help="Use real RealSense camera")
    parser.add_argument("--visualize", action="store_true", 
                       help="Visual debug mode - show intermediate processing steps")
    parser.add_argument("--info", action="store_true", 
                       help="Show connected camera information")
    parser.add_argument("--functions", action="store_true", 
                       help="Test standalone image processing functions")
    parser.add_argument("--interactive", action="store_true", 
                       help="Launch interactive testing mode")
    parser.add_argument("--duration", type=int, default=0, 
                       help="Test duration in seconds (default: 0 = infinite)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--safe", action="store_true", 
                       help="Safe mode - skip advanced camera configuration")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.mock and args.real:
        print("❌ Error: Cannot specify both --mock and --real")
        parser.print_help()
        exit(1)
    
    # Determine mode for display
    camera_type = "Real Camera" if args.real else "Mock Camera"
    mode = ""
    if args.visualize:
        mode = f"Visual Debug ({camera_type})"
    elif args.mock or (not args.real and not args.info and not args.functions and not args.interactive):
        mode = "Mock Processor Test"
    elif args.real:
        mode = "Real Camera Test"
    elif args.info:
        mode = "Camera Information"
    elif args.functions:
        mode = "Function Testing"
    elif args.interactive:
        mode = "Interactive Mode"
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Robust Camera Module - Standalone Testing")
    print("=" * 50)
    if mode:
        print(f"📋 Mode: {mode}")
        print("-" * 50)
    
    async def main():
        # uv run python services/core/tests/test_camera.py
        
        # Determine camera type
        use_real_camera = args.real
        
        if args.interactive:
            await interactive_mode()
        elif args.visualize:
            # Visualization mode - use real camera if --real specified, otherwise mock
            await test_camera_visualization(use_mock=not use_real_camera, duration=args.duration)
        elif args.mock or (not args.real and not args.info and not args.functions):
            # Mock mode (explicit --mock or default when no other mode specified)
            await test_mock_processor(args.duration, args.verbose, args.safe)
        elif args.real:
            # Real camera mode
            await test_real_camera(args.duration, args.verbose, args.safe)
        elif args.info:
            show_camera_info()
        elif args.functions:
            await test_image_functions()
        else:
            # Fallback: show help and run interactive mode
            parser.print_help()
            print("\n🎮 Starting interactive mode...")
            await interactive_mode()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

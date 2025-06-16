from experimance_core.config import CoreServiceConfig, DEFAULT_CONFIG_PATH
from experimance_core.depth_processor import (
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
from experimance_core.depth_visualizer import DepthVisualizationContext
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

def create_test_config(verbose: bool = False, safe_mode: bool = False):
    """Create a test configuration, loading values from config.toml if available."""
    # Try to load configuration from the default config path
    config_path = Path(DEFAULT_CONFIG_PATH)
    
    if config_path.exists():
        try:
            # Load the full service config and extract camera config
            service_config = CoreServiceConfig.from_overrides(config_file=str(config_path))
            camera_config = service_config.camera
            
            # Override specific settings for testing
            if verbose:
                camera_config.verbose_performance = True
            if safe_mode:
                camera_config.skip_advanced_config = True
                camera_config.json_config_path = None
            
            print(f"📋 Loaded configuration from {config_path}")
            print(f"🔍 Camera depth range: {camera_config.min_depth}m - {camera_config.max_depth}m")
            return camera_config
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load {config_path}: {e}")
            print("📋 Using default configuration")
    else:
        print(f"📋 No config file found at {config_path}, using defaults")
    
    # Fallback to creating a default camera config
    from experimance_core.config import CameraConfig
    default_config = CameraConfig(
        verbose_performance=verbose,
        skip_advanced_config=safe_mode
    )
    print(f"🔍 Default camera depth range: {default_config.min_depth}m - {default_config.max_depth}m")
    return default_config


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
    obstruction = simple_obstruction_detect(test_image, debug_save=True, debug_prefix="test_script")
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
    
    try:
        with DepthVisualizationContext() as visualizer:
            print("🎬 Starting visualization stream...")
            
            async for frame in processor.stream_frames():
                # Display frame and handle user input
                if not visualizer.display_frame(frame):
                    break
                
                # Check duration
                if duration > 0 and time.time() - visualizer.start_time >= duration:
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

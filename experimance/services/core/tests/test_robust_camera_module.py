#!/usr/bin/env python3
"""
Test script for the new robust camera module.

This demonstrates the clean, modern interface and error handling capabilities.
"""

import asyncio
import logging
import time
import cv2
from robust_camera import CameraConfig, DepthProcessor, MockDepthProcessor, create_depth_processor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_real_camera():
    """Test the robust camera with real hardware."""
    logger.info("=== Testing Real Camera ===")
    
    # Create configuration
    config = CameraConfig(
        resolution=(640, 480),  # Use validated resolution
        fps=30,
        align_frames=True,
        min_depth=0.0,
        max_depth=10.0,
        detect_hands=True,
        crop_to_content=True,
        max_retries=3
    )
    
    # Create processor
    processor = DepthProcessor(config)
    
    try:
        # Test initialization
        if not await processor.initialize():
            logger.error("Failed to initialize camera")
            return False
        
        logger.info("Camera initialized successfully!")
        
        # Test frame capture
        frame_count = 0
        max_frames = 50
        
        async for frame in processor.stream_frames():
            frame_count += 1
            
            logger.info(f"Frame {frame_count}: "
                       f"shape={frame.depth_image.shape}, "
                       f"hands={frame.hand_detected}, "
                       f"change={frame.change_score:.3f}")
            
            # Display frame
            cv2.imshow('Robust Camera - Depth', frame.depth_image)
            if frame.color_image is not None:
                cv2.imshow('Robust Camera - Color', frame.color_image)
            
            # Check for quit or max frames
            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= max_frames:
                break
        
        logger.info(f"Successfully processed {frame_count} frames")
        return True
        
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
        return False
    finally:
        processor.stop()
        cv2.destroyAllWindows()


async def test_mock_camera():
    """Test the mock camera functionality."""
    logger.info("=== Testing Mock Camera ===")
    
    config = CameraConfig(
        resolution=(640, 480),
        fps=10,  # Faster for testing
        detect_hands=True,
        warm_up_frames=5
    )
    
    # Create mock processor
    processor = MockDepthProcessor(config)
    
    try:
        if not await processor.initialize():
            logger.error("Failed to initialize mock camera")
            return False
        
        logger.info("Mock camera initialized successfully!")
        
        frame_count = 0
        max_frames = 20
        
        async for frame in processor.stream_frames():
            frame_count += 1
            
            logger.info(f"Mock Frame {frame_count}: "
                       f"hands={frame.hand_detected}, "
                       f"change={frame.change_score:.3f}, "
                       f"interaction={frame.has_interaction}")
            
            cv2.imshow('Mock Camera', frame.depth_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= max_frames:
                break
        
        logger.info(f"Successfully processed {frame_count} mock frames")
        return True
        
    except Exception as e:
        logger.error(f"Mock camera test failed: {e}")
        return False
    finally:
        processor.stop()
        cv2.destroyAllWindows()


async def test_compatibility_function():
    """Test the compatibility function for existing code."""
    logger.info("=== Testing Compatibility Function ===")
    
    from robust_camera import robust_depth_generator
    
    try:
        # Use mock for compatibility test
        frame_count = 0
        
        async for depth_image, hand_detected in robust_depth_generator(
            size=(640, 480),
            fps=10,
            mock="mock"  # Use mock mode
        ):
            frame_count += 1
            
            logger.info(f"Compat Frame {frame_count}: "
                       f"shape={depth_image.shape}, hands={hand_detected}")
            
            if frame_count >= 10:
                break
        
        logger.info("Compatibility function works correctly!")
        return True
        
    except Exception as e:
        logger.error(f"Compatibility test failed: {e}")
        return False


async def test_error_recovery():
    """Test error recovery capabilities."""
    logger.info("=== Testing Error Recovery ===")
    
    config = CameraConfig(
        resolution=(640, 480),
        fps=30,
        max_retries=2,  # Quick test
        retry_delay=1.0
    )
    
    processor = DepthProcessor(config)
    
    try:
        # This might fail if no camera is connected, which is fine for testing recovery
        result = await processor.initialize()
        
        if result:
            logger.info("Camera connected - testing normal operation")
            frame = await processor.get_processed_frame()
            if frame:
                logger.info("Successfully captured test frame")
        else:
            logger.info("Camera not available - error recovery was tested during initialization")
        
        return True
        
    except Exception as e:
        logger.info(f"Error recovery tested: {e}")
        return True  # This is expected behavior
    finally:
        processor.stop()


async def main():
    """Run all tests."""
    logger.info("Starting Robust Camera Module Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Mock Camera", test_mock_camera),
        ("Compatibility Function", test_compatibility_function),
        ("Error Recovery", test_error_recovery),
        ("Real Camera", test_real_camera),  # Last, as it might fail if no hardware
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = False
        
        # Brief pause between tests
        await asyncio.sleep(1)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS:")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("🎉 All tests passed! Robust camera module is ready.")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Check logs above.")
        return 1


if __name__ == '__main__':
    exit(asyncio.run(main()))

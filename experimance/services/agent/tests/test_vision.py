#!/usr/bin/env python3
"""
Test script for vision components in the Experimance Agent Service.

This script allows testing the webcam, VLM, and audience detection components
independently to verify proper installation and functionality.
"""

import asyncio
import logging
import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.config import VisionConfig
from agent.vision import WebcamManager, VLMProcessor, AudienceDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_webcam():
    """Test webcam capture functionality."""
    print("=" * 60)
    print("TESTING WEBCAM CAPTURE")
    print("=" * 60)
    
    config = VisionConfig()
    webcam = WebcamManager(config)
    
    try:
        await webcam.start()
        print(f"✓ Webcam initialized: {webcam.get_capture_info()}")
        
        # Capture a few frames
        for i in range(3):
            print(f"Capturing frame {i+1}...")
            frame = await webcam.capture_frame()
            if frame is not None:
                print(f"✓ Frame {i+1} captured: {frame.shape}")
                
                # Save frame for inspection
                success = await webcam.save_frame(frame, f"test_frame_{i+1}.jpg")
                if success:
                    print(f"✓ Frame saved as test_frame_{i+1}.jpg")
            else:
                print(f"✗ Failed to capture frame {i+1}")
            
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"✗ Webcam test failed: {e}")
    finally:
        await webcam.stop()
        print("Webcam test completed\n")


async def test_vlm():
    """Test VLM processing functionality."""
    print("=" * 60)
    print("TESTING VLM PROCESSING")
    print("=" * 60)
    
    config = VisionConfig(vlm_enabled=True, vlm_device="cpu")  # Use CPU for testing
    vlm = VLMProcessor(config)
    
    try:
        print("Loading VLM model (this may take a while)...")
        await vlm.start()
        print(f"✓ VLM initialized: {vlm.get_status()}")
        
        # Test with webcam if available
        webcam_config = VisionConfig()
        webcam = WebcamManager(webcam_config)
        
        try:
            await webcam.start()
            frame = await webcam.capture_frame()
            
            if frame is not None:
                print("Testing VLM analysis...")
                rgb_frame = webcam.preprocess_for_vlm(frame)
                
                # Test different analysis types
                for analysis_type in ["scene_description", "audience_detection"]:
                    print(f"Running {analysis_type} analysis...")
                    result = await vlm.analyze_scene(rgb_frame, analysis_type)
                    
                    if result.get("success", False):
                        print(f"✓ {analysis_type}: {result['description']}")
                        if analysis_type == "audience_detection":
                            print(f"  Audience detected: {result.get('audience_detected', 'unknown')}")
                    else:
                        print(f"✗ {analysis_type} failed: {result.get('error', 'unknown')}")
            else:
                print("✗ No webcam frame available for VLM testing")
                
        except Exception as e:
            print(f"Webcam unavailable for VLM testing: {e}")
        finally:
            if webcam:
                await webcam.stop()
            
    except Exception as e:
        print(f"✗ VLM test failed: {e}")
    finally:
        await vlm.stop()
        print("VLM test completed\n")


async def test_audience_detection():
    """Test audience detection functionality."""
    print("=" * 60)
    print("TESTING AUDIENCE DETECTION")
    print("=" * 60)
    
    config = VisionConfig(
        audience_detection_enabled=True,
        vlm_enabled=True,
        vlm_device="cpu"  # Use CPU for testing
    )
    
    webcam = WebcamManager(config)
    vlm = VLMProcessor(config)
    detector = AudienceDetector(config)
    
    try:
        # Initialize components
        await webcam.start()
        await vlm.start()
        await detector.start()
        
        print("✓ All components initialized")
        print(f"Webcam: {webcam.get_capture_info()}")
        print(f"VLM: {vlm.get_status()}")
        print(f"Detector: {detector.get_detection_stats()}")
        
        # Run detection loop
        print("\nRunning audience detection (press Ctrl+C to stop)...")
        
        try:
            for i in range(5):  # Test 5 detection cycles
                frame = await webcam.capture_frame()
                if frame is not None:
                    print(f"\nDetection cycle {i+1}:")

                    result = await detector.detect_audience(frame, webcam_manager=webcam, vlm=vlm)

                    if result.get("success", False):
                        detected = result["audience_detected"]
                        confidence = result["confidence"]
                        method = result["method_used"]
                        
                        print(f"  Audience detected: {detected}")
                        print(f"  Confidence: {confidence:.2f}")
                        print(f"  Method: {method}")
                        print(f"  Detection time: {result['detection_time']:.2f}s")
                    else:
                        print(f"  Detection failed: {result.get('error', 'unknown')}")
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\nDetection loop stopped by user")
            
        # Show final stats
        print(f"\nFinal detection statistics:")
        stats = detector.get_detection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"✗ Audience detection test failed: {e}")
    finally:
        await detector.stop()
        await vlm.stop()
        await webcam.stop()
        print("Audience detection test completed\n")


async def main():
    """Run all vision component tests."""
    print("Experimance Agent Vision Components Test")
    print("=" * 60)
    
    try:
        # Test webcam first
        await test_webcam()
        
        # Test VLM (skip if no CUDA and model is large)
        try:
            await test_vlm()
        except Exception as e:
            print(f"Skipping VLM test due to error: {e}\n")
        
        # Test integrated audience detection
        try:
            await test_audience_detection()
        except Exception as e:
            print(f"Skipping audience detection test due to error: {e}\n")
            
        print("=" * 60)
        print("All tests completed!")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

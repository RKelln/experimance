#!/usr/bin/env python3
"""
Test script for CPU-optimized audience detection.

This script tests the fast OpenCV-based audience detection that works well on CPU
without requiring GPU acceleration or heavy VLM models.
"""

import asyncio
import sys
from pathlib import Path
import time
import cv2

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experimance_agent.config import VisionConfig
from experimance_agent.vision import WebcamManager, CPUAudienceDetector


async def test_cpu_detection():
    """Test CPU-optimized audience detection."""
    print("=" * 60)
    print("TESTING CPU AUDIENCE DETECTION")
    print("=" * 60)
    
    # Configure for CPU detection
    config = VisionConfig(
        webcam_enabled=True,
        webcam_device_id=0,  # Use the first detected camera
        webcam_width=640,
        webcam_height=480,
        audience_detection_enabled=True,
        detection_method="cpu",
        cpu_performance_mode="balanced"
    )
    
    webcam = WebcamManager(config)
    detector = CPUAudienceDetector(config)
    
    try:
        # Initialize components
        print("Initializing webcam...")
        await webcam.start()
        print(f"✓ Webcam: {webcam.get_capture_info()}")
        
        print("Initializing CPU detector...")
        await detector.start()
        print("✓ CPU audience detector initialized")
        
        # Test different performance modes
        performance_modes = ["fast", "balanced", "accurate"]
        
        for mode in performance_modes:
            print(f"\n--- Testing {mode.upper()} mode ---")
            detector.set_performance_mode(mode)
            
            # Run several detection cycles
            detection_times = []
            results = []
            
            for i in range(5):
                frame = await webcam.capture_frame()
                if frame is not None:
                    start_time = time.time()
                    result = await detector.detect_audience(frame)
                    detection_time = time.time() - start_time
                    
                    detection_times.append(detection_time)
                    results.append(result)
                    
                    if result.get("success", False):
                        detected = result["audience_detected"]
                        confidence = result["confidence"]
                        method = result.get("method_used", "unknown")
                        
                        print(f"  Cycle {i+1}: {detected} (conf: {confidence:.2f}, time: {detection_time*1000:.1f}ms, method: {method})")
                    else:
                        print(f"  Cycle {i+1}: FAILED - {result.get('error', 'unknown')}")
                
                await asyncio.sleep(0.5)  # Brief pause between detections
            
            # Show performance stats for this mode
            avg_time = sum(detection_times) / len(detection_times)
            max_time = max(detection_times)
            print(f"  Performance: avg={avg_time*1000:.1f}ms, max={max_time*1000:.1f}ms")
        
        # Show final detector stats
        print(f"\n--- Final Statistics ---")
        stats = detector.get_detection_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Test visual feedback (save a frame with detections)
        print(f"\nTesting visual feedback...")
        frame = await webcam.capture_frame()
        if frame is not None:
            # Save original frame
            cv2.imwrite("cpu_detection_test_original.jpg", frame)
            
            # Run detection to get detailed results
            result = await detector.detect_audience(frame)
            if result.get("success", False):
                person_detection = result.get("person_detection", {})
                motion_detection = result.get("motion_detection", {})
                
                print(f"✓ Test frame saved as 'cpu_detection_test_original.jpg'")
                print(f"  Person detection: {person_detection.get('persons_detected', False)} ({person_detection.get('person_count', 0)} persons)")
                print(f"  Motion detection: {motion_detection.get('motion_detected', False)} (intensity: {motion_detection.get('motion_intensity', 0):.3f})")
        
        print("\n✓ CPU detection test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"✗ CPU detection test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await detector.stop()
        await webcam.stop()
        print("Components stopped")


async def benchmark_cpu_detection():
    """Benchmark CPU detection performance."""
    print("\n" + "=" * 60)
    print("CPU DETECTION PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    config = VisionConfig(
        webcam_enabled=True,
        webcam_device_id=0,
        webcam_width=640,
        webcam_height=480,
        audience_detection_enabled=True,
        detection_method="cpu",
        cpu_performance_mode="balanced"
    )
    
    webcam = WebcamManager(config)
    detector = CPUAudienceDetector(config)
    
    try:
        await webcam.start()
        await detector.start()
        
        # Capture a test frame
        frame = await webcam.capture_frame()
        if frame is None:
            print("✗ No frame available for benchmarking")
            return
        
        print(f"Benchmarking with frame size: {frame.shape}")
        
        # Test each performance mode
        modes = ["fast", "balanced", "accurate"]
        benchmark_results = {}
        
        for mode in modes:
            print(f"\nBenchmarking {mode.upper()} mode...")
            detector.set_performance_mode(mode)
            
            # Warm up
            for _ in range(3):
                await detector.detect_audience(frame)
            
            # Benchmark runs
            times = []
            for i in range(20):
                start_time = time.perf_counter()
                result = await detector.detect_audience(frame)
                end_time = time.perf_counter()
                
                if result.get("success", False):
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/20 runs...")
            
            # Calculate statistics
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                benchmark_results[mode] = {
                    "avg_ms": avg_time,
                    "min_ms": min_time,
                    "max_ms": max_time,
                    "fps": 1000 / avg_time
                }
                
                print(f"  Results: avg={avg_time:.1f}ms, min={min_time:.1f}ms, max={max_time:.1f}ms")
                print(f"  Theoretical max FPS: {1000/avg_time:.1f}")
        
        # Summary
        print(f"\n--- BENCHMARK SUMMARY ---")
        for mode, results in benchmark_results.items():
            print(f"{mode.upper():>8}: {results['avg_ms']:5.1f}ms avg ({results['fps']:4.1f} FPS)")
        
        # Recommendation
        best_mode = min(benchmark_results.keys(), key=lambda m: benchmark_results[m]['avg_ms'])
        print(f"\nRecommended mode for real-time use: {best_mode.upper()}")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
    finally:
        await detector.stop()
        await webcam.stop()


async def main():
    """Run CPU detection tests."""
    print("Experimance Agent CPU Detection Testing")
    print("=" * 60)
    
    try:
        # Basic functionality test
        await test_cpu_detection()
        
        # Performance benchmark
        await benchmark_cpu_detection()
        
        print("\n" + "=" * 60)
        print("All CPU detection tests completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

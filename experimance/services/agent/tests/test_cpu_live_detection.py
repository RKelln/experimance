#!/usr/bin/env python3
"""
Live CPU audience detection test for the Experimance Agent Service.

This script runs continuous audience detection with colorful output
to test from different locations. Use Ctrl+C to stop.
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Literal

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.config import VisionConfig
from agent.vision import WebcamManager, CPUAudienceDetector

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Background colors for high visibility
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    BG_MAGENTA = '\033[105m'
    BG_CYAN = '\033[106m'

def clear_screen():
    """Clear the terminal screen."""
    print('\033[2J\033[H', end='')

def print_large_status(detected: bool, confidence: float):
    """Print large, visible status that can be seen from a distance."""
    clear_screen()
    
    # Create large ASCII art status
    if detected:
        status_color = Colors.BG_GREEN + Colors.BOLD + Colors.WHITE
        status_text = "AUDIENCE DETECTED"
        emoji = "👥"
    else:
        status_color = Colors.BG_RED + Colors.BOLD + Colors.WHITE
        status_text = "NO AUDIENCE"
        emoji = "🚫"
    
    # Calculate confidence bar
    bar_length = 40
    filled_length = int(bar_length * confidence)
    confidence_bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}              EXPERIMANCE LIVE AUDIENCE DETECTION TEST{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    # Large status display
    print(f"{status_color}                                                                 {Colors.RESET}")
    print(f"{status_color}    {emoji}  {status_text:^50}  {emoji}    {Colors.RESET}")
    print(f"{status_color}                                                                 {Colors.RESET}\n")
    
    # Confidence display
    confidence_color = Colors.GREEN if confidence > 0.7 else Colors.YELLOW if confidence > 0.4 else Colors.RED
    print(f"{Colors.BOLD}CONFIDENCE: {confidence_color}{confidence:.2f} ({confidence*100:.1f}%){Colors.RESET}")
    print(f"{Colors.BOLD}CONFIDENCE BAR: [{confidence_color}{confidence_bar}{Colors.RESET}]")
    
    return status_text, confidence

def print_detection_details(result: dict):
    """Print detailed detection information."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}DETECTION DETAILS:{Colors.RESET}")
    print(f"  Detection Time: {Colors.CYAN}{result.get('detection_time', 0)*1000:.1f}ms{Colors.RESET}")
    print(f"  Method Used: {Colors.MAGENTA}{result.get('method_used', 'unknown')}{Colors.RESET}")
    
    # Person detection details
    person_info = result.get('person_detection', {})
    person_count = person_info.get('person_count', 0)
    person_confidence = person_info.get('confidence', 0.0)
    person_color = Colors.GREEN if person_count > 0 else Colors.RED
    print(f"  Person Detection: {person_color}{person_count} persons (conf: {person_confidence:.2f}){Colors.RESET}")
    
    # Motion detection details
    motion_info = result.get('motion_detection', {})
    motion_detected = motion_info.get('motion_detected', False)
    motion_intensity = motion_info.get('motion_intensity', 0.0)
    motion_color = Colors.GREEN if motion_detected else Colors.RED
    print(f"  Motion Detection: {motion_color}{'YES' if motion_detected else 'NO'} (intensity: {motion_intensity:.3f}){Colors.RESET}")
    
    # Performance info
    performance = result.get('performance', {})
    avg_time = performance.get('avg_detection_time', 0) * 1000
    frame_scale = performance.get('frame_scale', 1.0)
    print(f"  Average Detection Time: {Colors.YELLOW}{avg_time:.1f}ms{Colors.RESET}")
    print(f"  Frame Scale: {Colors.YELLOW}{frame_scale:.1f}x{Colors.RESET}")
    
def print_instructions():
    """Print usage instructions."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}INSTRUCTIONS:{Colors.RESET}")
    print(f"  • Move around in front of the camera to test detection")
    print(f"  • Stand still to test static person detection")
    print(f"  • Move out of frame to test absence detection")
    print(f"  • Press {Colors.BOLD}Ctrl+C{Colors.RESET} to stop")
    print(f"  • Status updates every 0.5 seconds")

async def run_live_detection(performance_mode: str = "accurate"):
    """Run continuous live detection with visual feedback."""
    
    # Configure logging to be less verbose during live testing
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(levelname)s: %(message)s'
    )
    
    print(f"{Colors.BOLD}{Colors.GREEN}Initializing Live CPU Audience Detection...{Colors.RESET}")
    print(f"{Colors.BOLD}Performance Mode: {Colors.CYAN}{performance_mode}{Colors.RESET}")
    
    # Create configuration
    config = VisionConfig(
        webcam_enabled=True,
        webcam_device_id=0,
        webcam_width=640,
        webcam_height=360,
        webcam_fps=30,
        audience_detection_enabled=True,
        audience_detection_interval=0.5,  # Fast updates for live testing
        vlm_enabled=False,  # CPU only
        cpu_performance_mode=performance_mode  # type: ignore
    )
    
    # Initialize components
    webcam_manager = WebcamManager(config)
    cpu_detector = CPUAudienceDetector(config)
    
    try:
        # Start components
        await webcam_manager.start()
        await cpu_detector.start()
        
        # Set performance mode
        cpu_detector.set_performance_mode(performance_mode)
        
        print(f"{Colors.BOLD}{Colors.GREEN}✓ Components initialized successfully{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}Starting live detection in 3 seconds...{Colors.RESET}")
        await asyncio.sleep(3)
        
        clear_screen()
        print_instructions()
        
        # Detection loop
        detection_count = 0
        start_time = time.time()
        
        while True:
            try:
                # Capture frame
                frame = await webcam_manager.capture_frame()
                if frame is None:
                    print(f"{Colors.RED}Failed to capture frame from webcam{Colors.RESET}")
                    await asyncio.sleep(1)
                    continue
                
                # Perform detection
                result = await cpu_detector.detect_audience(frame)
                
                if not result.get("success", False):
                    print(f"{Colors.RED}Detection failed: {result.get('error', 'Unknown error')}{Colors.RESET}")
                    await asyncio.sleep(1)
                    continue
                
                # Display results
                detected = result.get("audience_detected", False)
                confidence = result.get("confidence", 0.0)
                
                status_text, conf_display = print_large_status(detected, confidence)
                print_detection_details(result)
                
                # Show running statistics
                detection_count += 1
                elapsed_time = time.time() - start_time
                fps = detection_count / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\n{Colors.BOLD}{Colors.BLUE}STATISTICS:{Colors.RESET}")
                print(f"  Detections: {Colors.CYAN}{detection_count}{Colors.RESET}")
                print(f"  Running Time: {Colors.CYAN}{elapsed_time:.1f}s{Colors.RESET}")
                print(f"  Average FPS: {Colors.CYAN}{fps:.1f}{Colors.RESET}")
                
                print(f"\n{Colors.BOLD}{Colors.MAGENTA}Press Ctrl+C to stop{Colors.RESET}")
                
                # Update every 0.5 seconds
                await asyncio.sleep(config.audience_detection_interval)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Stopping live detection...{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error during detection: {e}{Colors.RESET}")
                await asyncio.sleep(1)
    
    finally:
        # Cleanup
        print(f"{Colors.YELLOW}Cleaning up...{Colors.RESET}")
        await cpu_detector.stop()
        await webcam_manager.stop()
        
        # Final statistics
        stats = cpu_detector.get_detection_stats()
        print(f"\n{Colors.BOLD}{Colors.GREEN}FINAL STATISTICS:{Colors.RESET}")
        print(f"  Total Detections: {Colors.CYAN}{stats['total_detections']}{Colors.RESET}")
        print(f"  Average Detection Time: {Colors.CYAN}{stats['avg_detection_time']*1000:.1f}ms{Colors.RESET}")
        print(f"  Max Detection Time: {Colors.CYAN}{stats['max_detection_time']*1000:.1f}ms{Colors.RESET}")
        print(f"  Performance Mode: {Colors.CYAN}{performance_mode}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}Live detection test completed!{Colors.RESET}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Live CPU audience detection test with colorful output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Modes:
  fast      - Fastest detection, lower accuracy (scale: 0.3x)
  balanced  - Good balance of speed and accuracy (scale: 0.5x) 
  accurate  - Best accuracy, slower detection (scale: 0.7x)

Usage Examples:
  python test_cpu_live_detection.py                    # Use accurate mode
  python test_cpu_live_detection.py --mode fast        # Use fast mode
  python test_cpu_live_detection.py --mode balanced    # Use balanced mode
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['fast', 'balanced', 'accurate'],
        default='accurate',
        help='Performance mode for detection (default: accurate)'
    )
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.BLUE}Experimance Live CPU Audience Detection Test{Colors.RESET}")
    print(f"{Colors.BOLD}Performance Mode: {Colors.CYAN}{args.mode}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*50}{Colors.RESET}")
    
    try:
        asyncio.run(run_live_detection(args.mode))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Test failed: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()

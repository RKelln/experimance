#!/usr/bin/env python3
"""
Test the AudioManager with a real audio file.
"""

import asyncio
import logging
from pathlib import Path
import sys
import tempfile
import json

# Add the service source directory to sys.path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from audio_manager import AudioManager, AudioTrack
except ImportError:
    print("Could not import AudioManager - make sure you're running from the correct directory")
    sys.exit(1)

# Set up logging to capture debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class MPVEventLogger:
    """Helper to capture log messages and check for mpv errors."""
    def __init__(self):
        self.messages = []
        self.handler = logging.Handler()
        self.handler.emit = self._capture_message
        
    def _capture_message(self, record):
        """Capture log messages."""
        self.messages.append(record.getMessage())
        
    def start_capture(self):
        """Start capturing log messages."""
        logging.getLogger('audio_manager').addHandler(self.handler)
        
    def stop_capture(self):
        """Stop capturing log messages."""
        logging.getLogger('audio_manager').removeHandler(self.handler)
        
    def get_mpv_errors(self):
        """Get any mpv error messages."""
        return [msg for msg in self.messages if 'mpv volume command failed' in msg.lower()]
        
    def get_event_skips(self):
        """Get messages about skipping mpv events."""
        return [msg for msg in self.messages if 'skipping mpv event' in msg.lower()]


async def test_mpv_ipc_event_handling():
    """Test that mpv IPC events are properly handled and don't cause errors."""
    print("\n=== Testing MPV IPC Event Handling ===")
    
    # Find a real audio file
    audio_file = Path(__file__).parent.parent.parent.parent.parent / "media" / "audio" / "cartesia_sophie.wav"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return False
    
    # Set up event logger to capture debug messages
    event_logger = MPVEventLogger()
    event_logger.start_capture()
    
    try:
        manager = AudioManager(default_volume=0.8, crossfade_duration=2.0)
        
        print("1. Starting audio track and monitoring for IPC events...")
        
        # Start a track that will likely generate seek/playback-restart events
        success = await manager.play_audio(
            f"file://{audio_file.absolute()}", 
            loop=True, 
            fade_in=True,
            fade_in_duration=3.0  # Longer fade to increase chance of events
        )
        
        if not success:
            print("❌ Failed to start audio track")
            return False
            
        print("2. Letting audio play for 5 seconds to capture any mpv events...")
        await asyncio.sleep(5)
        
        print("3. Starting crossfade to generate more IPC activity...")
        success2 = await manager.play_audio(
            f"file://{audio_file.absolute()}", 
            loop=True,
            crossfade=True,
            fade_in=True,
            fade_in_duration=2.0
        )
        
        if not success2:
            print("❌ Failed to start crossfade")
            return False
            
        print("4. Waiting for crossfade to complete...")
        await asyncio.sleep(4)
        
        # Clean up
        await manager.cleanup()
        
    finally:
        event_logger.stop_capture()
    
    # Analyze captured messages
    mpv_errors = event_logger.get_mpv_errors()
    event_skips = event_logger.get_event_skips()
    
    print(f"\nResults:")
    print(f"  - MPV command errors: {len(mpv_errors)}")
    print(f"  - MPV events skipped: {len(event_skips)}")
    
    if mpv_errors:
        print(f"❌ Found {len(mpv_errors)} mpv command errors:")
        for error in mpv_errors[:3]:  # Show first 3 errors
            print(f"    {error}")
        return False
    
    if event_skips:
        print(f"✅ Successfully skipped {len(event_skips)} mpv events (expected behavior)")
        print("Sample skipped events:")
        for skip in event_skips[:3]:  # Show first 3 skips
            print(f"    {skip}")
    
    print("✅ No mpv command errors detected - IPC event handling working correctly!")
    return True


async def test_volume_command_reliability():
    """Test that volume commands work reliably even with mpv events."""
    print("\n=== Testing Volume Command Reliability ===")
    
    # Find a real audio file
    audio_file = Path(__file__).parent.parent.parent.parent.parent / "media" / "audio" / "cartesia_sophie.wav"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return False
    
    # Set up event logger
    event_logger = MPVEventLogger()
    event_logger.start_capture()
    
    try:
        # Create a single track to test direct volume commands
        track = AudioTrack(f"file://{audio_file.absolute()}", loop=True, volume=0.8)
        
        # Start the track at low volume
        success = await track.start(start_volume=0.1)
        if not success:
            print("❌ Failed to start audio track")
            return False
            
        print("1. Track started, testing rapid volume changes...")
        
        # Test rapid volume changes that might generate events
        volumes = [0.2, 0.5, 0.3, 0.7, 0.4, 0.8, 0.1, 0.6]
        
        for i, vol in enumerate(volumes):
            result = await track._send_mpv_command(["set_property", "volume", int(vol * 100)])
            print(f"   Volume {i+1}: {vol} -> Result: {result is not None}")
            await asyncio.sleep(0.2)  # Short delay between commands
        
        print("2. Testing fade in operation...")
        await track.fade_in(duration=2.0)
        
        print("3. Testing fade out operation...")
        await track.fade_out(duration=2.0)
        
        # Clean up
        await track.stop()
        
    finally:
        event_logger.stop_capture()
    
    # Check for errors
    mpv_errors = event_logger.get_mpv_errors()
    
    if mpv_errors:
        print(f"❌ Found {len(mpv_errors)} mpv command errors during volume testing")
        return False
    else:
        print("✅ All volume commands executed without errors!")
        return True


async def test_real_audio():
    """Test with a real audio file."""
    print("=== Testing AudioManager with Real Audio and Smooth Fading ===")
    
    # Find a real audio file
    audio_file = Path(__file__).parent.parent.parent.parent.parent / "media" / "audio" / "cartesia_sophie.wav"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return
    
    print(f"Testing with: {audio_file}")
    
    # Use shorter crossfade for testing
    manager = AudioManager(default_volume=0.8, crossfade_duration=3.0)
    
    print("\n1. Testing track that starts at full volume (no fade, no crossfade)...")
    success = await manager.play_audio(
        f"file://{audio_file.absolute()}", 
        loop=False, 
        crossfade=False,  # No crossfade
        fade_in=False     # No fade in - should start at full volume immediately
    )
    print(f"Play audio result: {success}")
    
    if success:
        print("Audio started at full volume, waiting 3 seconds...")
        await asyncio.sleep(3)
        
        print("\n2. Testing crossfade with fade in (both tracks should start at 0 volume)...")
        success2 = await manager.play_audio(
            f"file://{audio_file.absolute()}", 
            loop=True, 
            crossfade=True,   # Crossfade - new track starts at 0
            fade_in=True,     # Fade in - new track starts at 0
            fade_in_duration=2.0
        )
        print(f"Crossfade result: {success2}")
        
        print("Letting crossfade complete (4 seconds)...")
        await asyncio.sleep(6)
        
        print("\n3. Testing manual fade out...")
        success3 = await manager.play_audio(
            f"file://{audio_file.absolute()}", 
            loop=True, 
            crossfade=False,  # No crossfade
            fade_in=False     # No fade in - should start at full volume immediately
        )
        await manager.fade_out_all(10.0)

        print("Done!")
    else:
        print("Failed to start audio")
    
    await manager.cleanup()


async def main():
    """Run all tests."""
    print("🧪 AudioManager IPC Event Handling Tests")
    print("=" * 50)
    
    # Test 1: IPC event handling
    test1_passed = await test_mpv_ipc_event_handling()
    
    # Test 2: Volume command reliability
    test2_passed = await test_volume_command_reliability()
    
    # Test 3: Regular audio manager test
    await test_real_audio()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  - IPC Event Handling: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  - Volume Commands:     {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"  - Basic Audio Tests:   ✅ COMPLETED")
    
    if test1_passed and test2_passed:
        print("\n🎉 All critical tests PASSED! MPV IPC is working correctly.")
    else:
        print("\n⚠️  Some tests failed - check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())

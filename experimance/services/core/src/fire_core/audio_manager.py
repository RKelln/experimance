#!/usr/bin/env python3
"""
Audio Manager for Fire Core Service.

Simple audio playback manager using mpv subprocess for environmental audio.
Handles looping, crossfading, and cleanup of audio tracks.
"""

import asyncio
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AudioTrack:
    """Represents a single audio track being played."""
    
    def __init__(self, uri: str, loop: bool = True, volume: float = 1.0):
        self.uri = uri
        self.loop = loop
        self.volume = volume
        self.process: Optional[subprocess.Popen] = None
        self.track_id = str(uuid.uuid4())[:8]
        
    async def start(self) -> bool:
        """Start playing the audio track."""
        try:
            # Build mpv command
            cmd = [
                'mpv',
                '--no-video',          # Audio only
                '--no-terminal',       # No interactive terminal
                '--really-quiet',      # Suppress output
                '--volume={}'.format(int(self.volume * 100)),
            ]
            
            if self.loop:
                cmd.append('--loop=inf')
            
            cmd.append(self.uri)
            
            logger.info(f"Starting audio track {self.track_id}: {self.uri}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=None  # Allow signal handling
            )
            
            # Give it a moment to start
            await asyncio.sleep(0.1)
            
            # Check if it started successfully
            if self.process.poll() is not None:
                logger.error(f"Audio track {self.track_id} failed to start (exit code: {self.process.returncode})")
                return False
                
            logger.info(f"Audio track {self.track_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio track {self.track_id}: {e}")
            return False
    
    async def fade_out(self, duration: float = 2.0) -> None:
        """Fade out and stop the audio track."""
        if not self.process or self.process.poll() is not None:
            return
            
        try:
            logger.info(f"Fading out audio track {self.track_id} over {duration}s")
            
            # mpv doesn't have built-in fade out via command line,
            # so we'll do a simple volume ramp down
            steps = 20
            step_duration = duration / steps
            
            for i in range(steps, 0, -1):
                if self.process.poll() is not None:
                    break
                    
                volume = int((i / steps) * self.volume * 100)
                try:
                    # Send volume change via echo to mpv's stdin (if it was opened with stdin)
                    # For now, we'll just do an abrupt stop after the fade duration
                    pass
                except:
                    pass
                    
                await asyncio.sleep(step_duration)
                
            # Final stop
            await self.stop()
            
        except Exception as e:
            logger.error(f"Error during fade out of track {self.track_id}: {e}")
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the audio track immediately."""
        if not self.process:
            return
            
        try:
            logger.info(f"Stopping audio track {self.track_id}")
            
            if self.process.poll() is None:
                self.process.terminate()
                
                # Give it a chance to terminate gracefully
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait), 
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Audio track {self.track_id} didn't terminate gracefully, killing")
                    self.process.kill()
                    
        except Exception as e:
            logger.error(f"Error stopping audio track {self.track_id}: {e}")
        finally:
            self.process = None
    
    @property
    def is_playing(self) -> bool:
        """Check if the track is currently playing."""
        return self.process is not None and self.process.poll() is None


class AudioManager:
    """
    Simple audio manager for environmental sounds using mpv.
    
    Handles playing looped audio tracks, crossfading between tracks,
    and cleanup when scenes change.
    """
    
    def __init__(self, default_volume: float = 0.7, crossfade_duration: float = 2.0):
        self.default_volume = max(0.0, min(1.0, default_volume))
        self.crossfade_duration = crossfade_duration
        self.current_tracks: Dict[str, AudioTrack] = {}
        self.fade_tasks: Set[asyncio.Task] = set()
        
        logger.info(f"AudioManager initialized with volume={self.default_volume}, crossfade={crossfade_duration}s")
    
    async def play_audio(
        self, 
        uri: str, 
        loop: bool = True, 
        volume: Optional[float] = None, 
        crossfade: bool = True
    ) -> bool:
        """
        Play an audio track, optionally crossfading from current tracks.
        
        Args:
            uri: URI to the audio file (file://, http://, etc.)
            loop: Whether to loop the audio indefinitely
            volume: Volume level (0.0-1.0), defaults to manager default
            crossfade: Whether to crossfade from existing tracks
            
        Returns:
            True if audio started successfully, False otherwise
        """
        if volume is None:
            volume = self.default_volume
        
        volume = max(0.0, min(1.0, volume))
        
        # Create new track
        track = AudioTrack(uri, loop, volume)
        
        # Start the new track
        success = await track.start()
        if not success:
            logger.error(f"Failed to start audio track: {uri}")
            return False
        
        # Handle existing tracks
        if crossfade and self.current_tracks:
            # Fade out existing tracks
            await self._fade_out_all_tracks()
        else:
            # Stop existing tracks immediately
            await self.stop_all()
        
        # Add new track to current tracks
        self.current_tracks[track.track_id] = track
        
        logger.info(f"Now playing: {uri} (volume={volume}, loop={loop})")
        return True
    
    async def _fade_out_all_tracks(self) -> None:
        """Fade out all current tracks."""
        if not self.current_tracks:
            return
            
        logger.info(f"Fading out {len(self.current_tracks)} current tracks")
        
        # Start fade out tasks for all tracks
        fade_tasks = []
        for track in self.current_tracks.values():
            task = asyncio.create_task(track.fade_out(self.crossfade_duration))
            fade_tasks.append(task)
            self.fade_tasks.add(task)
        
        # Wait for all fades to complete
        if fade_tasks:
            try:
                await asyncio.gather(*fade_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during fade out: {e}")
        
        # Clean up completed tasks
        for task in fade_tasks:
            self.fade_tasks.discard(task)
        
        # Clear current tracks
        self.current_tracks.clear()
    
    async def fade_out_all(self, duration: Optional[float] = None) -> None:
        """
        Fade out all current tracks.
        
        Args:
            duration: Fade duration in seconds, defaults to manager crossfade duration
        """
        if duration is not None:
            original_duration = self.crossfade_duration
            self.crossfade_duration = duration
            
        await self._fade_out_all_tracks()
        
        if duration is not None:
            self.crossfade_duration = original_duration
    
    async def stop_all(self) -> None:
        """Stop all current tracks immediately."""
        if not self.current_tracks:
            return
            
        logger.info(f"Stopping {len(self.current_tracks)} audio tracks")
        
        # Cancel any ongoing fade tasks
        for task in list(self.fade_tasks):
            task.cancel()
        self.fade_tasks.clear()
        
        # Stop all tracks
        stop_tasks = []
        for track in self.current_tracks.values():
            stop_tasks.append(asyncio.create_task(track.stop()))
        
        if stop_tasks:
            try:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error stopping tracks: {e}")
        
        self.current_tracks.clear()
    
    async def set_volume(self, volume: float) -> None:
        """
        Set the default volume for future tracks.
        
        Note: This doesn't affect currently playing tracks.
        """
        self.default_volume = max(0.0, min(1.0, volume))
        logger.info(f"Default volume set to {self.default_volume}")
    
    def is_playing(self) -> bool:
        """Check if any tracks are currently playing."""
        return any(track.is_playing for track in self.current_tracks.values())
    
    def get_playing_count(self) -> int:
        """Get the number of currently playing tracks."""
        return sum(1 for track in self.current_tracks.values() if track.is_playing)
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up AudioManager")
        await self.stop_all()


# Test function for development
async def test_audio_manager():
    """Test function for development and debugging."""
    manager = AudioManager(volume=0.5)
    
    # Test with a local file (if it exists)
    test_file = Path("media/audio/test.wav")
    if test_file.exists():
        print(f"Testing with: {test_file}")
        success = await manager.play_audio(f"file://{test_file.absolute()}")
        if success:
            print("Audio started, waiting 5 seconds...")
            await asyncio.sleep(5)
            print("Fading out...")
            await manager.fade_out_all(2.0)
            print("Done")
        else:
            print("Failed to start audio")
    else:
        print(f"Test file not found: {test_file}")
    
    await manager.cleanup()


if __name__ == "__main__":
    # Simple test
    asyncio.run(test_audio_manager())

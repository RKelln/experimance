#!/usr/bin/env python3
"""
Audio Manager for Fire Core Service.

Simple audio playback manager using mpv subprocess for environmental audio.
Handles looping, crossfading, and cleanup of audio tracks.
"""

import asyncio
import json
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
        self.ipc_socket_path: Optional[str] = None
        self.ipc_reader: Optional[asyncio.StreamReader] = None
        self.ipc_writer: Optional[asyncio.StreamWriter] = None
        self._ipc_lock: Optional[asyncio.Lock] = None
        
    async def _send_mpv_command(self, command: list) -> Optional[dict]:
        """Send a command to mpv via IPC."""
        if not self.ipc_writer:
            return None
            
        try:
            cmd_data = {"command": command}
            cmd_json = json.dumps(cmd_data) + "\n"
            
            # Initialize lock if needed
            if self._ipc_lock is None:
                self._ipc_lock = asyncio.Lock()
            
            async with self._ipc_lock:
                self.ipc_writer.write(cmd_json.encode())
                await self.ipc_writer.drain()
                
                # Read response, skipping events
                if self.ipc_reader:
                    # Try to read response within timeout, but skip events
                    max_attempts = 5
                    for _ in range(max_attempts):
                        try:
                            # Check reader is still available before each read
                            if not self.ipc_reader:
                                break
                                
                            response_line = await asyncio.wait_for(
                                self.ipc_reader.readline(), 
                                timeout=0.2
                            )
                            if response_line:
                                response = json.loads(response_line.decode())
                                
                                # Skip events, return command responses only
                                if "event" in response:
                                    #logger.debug(f"Skipping mpv event: {response}")
                                    continue
                                    
                                # This should be a command response
                                return response
                        except asyncio.TimeoutError:
                            # No more responses, return None
                            break
                    
        except Exception as e:
            logger.debug(f"Error sending mpv command {command}: {e}")
            
        return None
        
    async def start(self, start_volume: Optional[float] = None) -> bool:
        """Start playing the audio track.
        
        Args:
            start_volume: Initial volume to start at (0.0-1.0). If None, uses track volume.
        """
        try:
            # Create temporary socket for IPC
            temp_dir = Path(tempfile.gettempdir())
            self.ipc_socket_path = str(temp_dir / f"mpv_ipc_{self.track_id}.sock")
            
            # Determine starting volume
            if start_volume is not None:
                initial_volume = max(0.0, min(1.0, start_volume))
            else:
                initial_volume = self.volume
            
            # Build mpv command for background audio playbook (no window/interface)
            cmd = [
                'mpv',                 # Use mpv from PATH
                '--no-config',         # Don't load user config files
                '--no-video',          # Audio only, no video processing
                '--no-terminal',       # No interactive terminal
                '--really-quiet',      # Suppress all output
                '--no-input-default-bindings',  # Disable keyboard shortcuts
                '--no-osc',            # No on-screen controller
                '--no-osd-bar',        # No on-screen display bar
                '--keep-open=no',      # Exit when playback finishes
                '--force-window=no',   # Don't create any window
                '--volume={}'.format(int(initial_volume * 100)),
                '--audio-channels=stereo',  # Force stereo output for mono compatibility
                f'--input-ipc-server={self.ipc_socket_path}',  # Enable IPC
            ]
        
            if self.loop:
                cmd.append('--loop=inf')
            
            cmd.append(self.uri)
            #cmd.append("media/audio/audio-test-5a-43676.mp3")
            
            logger.info(f"Starting audio track {self.track_id}: {self.uri} (initial volume: {initial_volume})")
            logger.debug(f"mpv command: {' '.join(cmd)}")

            # Capture stderr for debugging when things go wrong
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, #DEVNULL
                stderr=subprocess.PIPE, #DEVNULL
                preexec_fn=None  # Allow signal handling
            )
            
            # Give mpv time to start and create the socket
            await asyncio.sleep(0.2)
            
            # Check if it started successfully
            if self.process.poll() is not None:
                # Capture any error output for debugging
                try:
                    stdout, stderr = self.process.communicate(timeout=1.0)
                    stdout_str = stdout.decode('utf-8', errors='ignore').strip() if stdout else ""
                    stderr_str = stderr.decode('utf-8', errors='ignore').strip() if stderr else ""
                    
                    error_msg = f"Audio track {self.track_id} failed to start (exit code: {self.process.returncode})"
                    if stderr_str:
                        error_msg += f"\nSTDERR: {stderr_str}"
                    if stdout_str:
                        error_msg += f"\nSTDOUT: {stdout_str}"
                    
                    logger.error(error_msg)
                except subprocess.TimeoutExpired:
                    logger.error(f"Audio track {self.track_id} failed to start (exit code: {self.process.returncode}) - no error output captured")
                except Exception as e:
                    logger.error(f"Audio track {self.track_id} failed to start (exit code: {self.process.returncode}) - error capturing output: {e}")
                
                return False
            
            # Connect to IPC socket
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(self.ipc_socket_path),
                    timeout=2.0
                )
                self.ipc_reader = reader
                self.ipc_writer = writer
                logger.debug(f"Connected to mpv IPC for track {self.track_id}")
            except Exception as e:
                logger.warning(f"Failed to connect to mpv IPC for track {self.track_id}: {e}")
                # Continue without IPC - basic functionality will still work
                
            logger.info(f"Audio track {self.track_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio track {self.track_id}: {e}")
            await self.stop()  # Clean up on failure
            return False
    
    async def fade_out(self, duration: float = 2.0) -> None:
        """Fade out and stop the audio track."""
        if not self.process or self.process.poll() is not None:
            return
            
        try:
            logger.info(f"Fading out audio track {self.track_id} over {duration}s")
            
            # If we have IPC connection, do smooth volume fade
            if self.ipc_writer and self.ipc_reader:
                await self._smooth_volume_fade(duration)
            else:
                # Fallback: just wait then stop
                logger.debug(f"No IPC connection for track {self.track_id}, using simple fade")
                await asyncio.sleep(duration)
                
            # Final stop
            await self.stop()
            
        except Exception as e:
            logger.error(f"Error during fade out of track {self.track_id}: {e}")
            await self.stop()
    
    async def _smooth_volume_fade(self, duration: float) -> None:
        """Perform smooth volume fade using mpv IPC."""
        try:
            steps = max(20, int(duration * 10))  # At least 20 steps, more for longer fades
            step_duration = duration / steps
            
            start_volume = int(self.volume * 100)
            
            for i in range(steps, 0, -1):
                if not self.process or self.process.poll() is not None:
                    break
                    
                # Calculate volume for this step
                volume_ratio = i / steps
                current_volume = int(start_volume * volume_ratio)
                
                # Send volume command to mpv
                result = await self._send_mpv_command(["set_property", "volume", current_volume])
                if result and result.get("error") and result.get("error") != "success":
                    logger.debug(f"mpv volume command failed: {result}")
                    
                await asyncio.sleep(step_duration)
                
        except Exception as e:
            logger.debug(f"Error during smooth fade: {e}")
            # Continue with stop anyway
    
    async def stop(self) -> None:
        """Stop the audio track immediately."""
        try:
            logger.info(f"Stopping audio track {self.track_id}")
            
            # Close IPC connection first
            if self.ipc_writer:
                try:
                    self.ipc_writer.close()
                    await self.ipc_writer.wait_closed()
                except:
                    pass
                self.ipc_writer = None
                self.ipc_reader = None
            
            # Clean up IPC socket file
            if self.ipc_socket_path and Path(self.ipc_socket_path).exists():
                try:
                    Path(self.ipc_socket_path).unlink()
                except:
                    pass
                self.ipc_socket_path = None
            
            # Stop the process
            if self.process:
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
                        
                self.process = None
                    
        except Exception as e:
            logger.error(f"Error stopping audio track {self.track_id}: {e}")
        finally:
            self.process = None
            self.ipc_writer = None
            self.ipc_reader = None
            self.ipc_socket_path = None
    
    async def fade_in(self, duration: float = 2.0) -> None:
        """Fade in the audio track from 0 to target volume."""
        if not self.process or self.process.poll() is not None:
            return
            
        if not self.ipc_writer or not self.ipc_reader:
            logger.debug(f"No IPC connection for track {self.track_id}, skipping fade in")
            return
            
        try:
            logger.info(f"Fading in audio track {self.track_id} over {duration}s")
            
            # Start at 0 volume
            await self._send_mpv_command(["set_property", "volume", 0])
            
            steps = max(20, int(duration * 10))  # At least 20 steps
            step_duration = duration / steps
            target_volume = int(self.volume * 100)
            
            for i in range(1, steps + 1):
                if not self.process or self.process.poll() is not None:
                    break
                    
                # Calculate volume for this step
                volume_ratio = i / steps
                current_volume = int(target_volume * volume_ratio)
                
                # Send volume command to mpv
                result = await self._send_mpv_command(["set_property", "volume", current_volume])
                if result and result.get("error") and result.get("error") != "success":
                    logger.debug(f"mpv volume command failed: {result}")
                    
                await asyncio.sleep(step_duration)
                
        except Exception as e:
            logger.debug(f"Error during fade in: {e}")
    
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
        crossfade: bool = True,
        fade_in: bool = True,
        fade_in_duration: Optional[float] = None
    ) -> bool:
        """
        Play an audio track, optionally crossfading from current tracks.
        
        Args:
            uri: URI to the audio file (file://, http://, etc.)
            loop: Whether to loop the audio indefinitely
            volume: Volume level (0.0-1.0), defaults to manager default
            crossfade: Whether to crossfade from existing tracks
            fade_in: Whether to fade in the new track
            fade_in_duration: Duration for fade in (defaults to crossfade_duration)
            
        Returns:
            True if audio started successfully, False otherwise
        """
        if volume is None:
            volume = max(0.0, min(1.0, self.default_volume))
        else:
            volume = max(0.0, min(1.0, volume))
        
        if fade_in_duration is None:
            fade_in_duration = self.crossfade_duration
        
        # Create new track
        track = AudioTrack(uri, loop, volume)
        
        # Determine starting volume:
        # - If crossfading or fading in, start at 0
        # - Otherwise start at target volume
        start_volume = 0.0 if (crossfade or fade_in) else volume
        
        # Start the new track
        success = await track.start(start_volume=start_volume)
        if not success:
            logger.error(f"Failed to start audio track: {uri}")
            return False
        
        # Handle crossfading and existing tracks
        if crossfade and self.current_tracks:
            # Get list of tracks to fade out BEFORE adding new track
            tracks_to_fade_out = list(self.current_tracks.values())
            
            # Add new track to current tracks immediately
            self.current_tracks[track.track_id] = track
            
            # Start fade out of OLD tracks only
            fade_out_task = asyncio.create_task(self._fade_out_tracks(tracks_to_fade_out))
            
            # Start fade in of new track
            if fade_in:
                fade_in_task = asyncio.create_task(track.fade_in(fade_in_duration))
                # Let both run concurrently
            
        else:
            # Stop existing tracks immediately
            await self.stop_all()
            
            # Add new track to current tracks
            self.current_tracks[track.track_id] = track
            
            # Optionally fade in the new track (even if not crossfading)
            if fade_in:
                asyncio.create_task(track.fade_in(fade_in_duration))
        
        logger.info(f"Now playing: {uri} (volume={volume}, loop={loop}, fade_in={fade_in})")
        return True
    
    async def _fade_out_tracks(self, tracks: list) -> None:
        """Fade out a specific list of tracks."""
        if not tracks:
            return
            
        logger.info(f"Fading out {len(tracks)} current tracks")
        
        # Start fade out tasks for specified tracks
        fade_tasks = []
        for track in tracks:
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
        
        # Remove the faded tracks from current_tracks
        for track in tracks:
            self.current_tracks.pop(track.track_id, None)
    
    async def _fade_out_all_tracks(self) -> None:
        """Fade out all current tracks."""
        if not self.current_tracks:
            return
        
        tracks_to_fade = list(self.current_tracks.values())
        await self._fade_out_tracks(tracks_to_fade)
    
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
    manager = AudioManager(default_volume=0.5)
    
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

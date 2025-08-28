#!/usr/bin/env python3
"""Interactive multi-channel audio delay calibration tool.

This script helps calibrate delays for the multi-channel audio transport by:
1. Playing a click track on different channels
2. Recording audio to detect timing differences
3. Providing interactive controls to adjust delays
4. Auto-calibrating delays based on microphone feedback

Usage:
    python scripts/test_multi_channel_audio.py
    p        print("Commands:")
        print("  p <bpm> <duration> - Play click track (default: p 120 5)")
        print("  t <channel>        - Play test tone on specific channel")
        print("  d <channel> <ms>   - Set delay for channel (in milliseconds)")
        print("  v <channel> <vol>  - Set volume for channel (0.0-1.0)")
        print("  s                  - Show current settings")
        print("  c                  - Play click track on all channels")
        print("  voice <filepath>   - Load voice audio file for testing")
        print("  play voice         - Play loaded voice audio")
        print("  r <duration>       - Record and analyze timing (auto-calibrate)")
        print("  w <filename>       - Write current config to file")
        print("  i [channels]       - Interactive mode with live delay adjustment (e.g., 'i 0,1')")
        print("  q                  - Quit")ts/test_multi_channel_audio.py --auto-calibrate
    python scripts/test_multi_channel_audio.py --config projects/fire/agent.toml
"""

import argparse
import asyncio
import logging
import math
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "libs" / "common" / "src"))
sys.path.insert(0, str(project_root / "services" / "agent" / "src"))

try:
    import numpy as np
    import pyaudio
    from scipy import signal
    from scipy.io import wavfile
    import toml
    import wave  # Add wave module for WAV file reading
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install numpy scipy pyaudio toml")
    sys.exit(1)

from experimance_common.audio_utils import (
    list_audio_devices,
    find_audio_device_by_name,
    resolve_audio_device_index,
    suppress_audio_errors
)

# Import DelayBuffer from the actual transport implementation
from agent.backends.pipecat.audio_utils import DelayBuffer, create_delay_buffers

logger = logging.getLogger(__name__)

class ClickTrackGenerator:
    """Generates precise click tracks for timing calibration."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
    def generate_click(self, duration_ms: int = 3, frequency: int = 1000) -> np.ndarray:
        """Generate a single click sound."""
        samples = int(duration_ms * self.sample_rate / 1000)
        t = np.linspace(0, duration_ms / 1000, samples, False)
        
        # Generate a click with sharp attack and decay
        click = np.sin(2 * np.pi * frequency * t)
        envelope = np.exp(-t * 50)  # Exponential decay
        
        return click * envelope
    
    def generate_click_track(self, bpm: int = 120, duration_seconds: int = 10, 
                           click_frequency: int = 1000) -> np.ndarray:
        """Generate a click track at specified BPM."""
        samples_per_beat = int(60 * self.sample_rate / bpm)
        total_samples = int(duration_seconds * self.sample_rate)
        
        click = self.generate_click(frequency=click_frequency)
        track = np.zeros(total_samples)
        
        # Place clicks at beat intervals
        for beat in range(int(duration_seconds * bpm / 60)):
            start_idx = beat * samples_per_beat
            end_idx = min(start_idx + len(click), total_samples)
            
            if start_idx < total_samples:
                track[start_idx:end_idx] += click[:end_idx-start_idx]
        
        return track

class VoiceAudioLoader:
    """Helper class to load and manage voice audio files for testing."""
    
    @staticmethod
    def load_wav_file(filepath: str) -> Tuple[np.ndarray, int]:
        """Load a WAV file and return audio data and sample rate."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        with wave.open(filepath, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            num_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            
            logger.info(f"Loading voice audio: {filepath}")
            logger.info(f"  Sample rate: {sample_rate}Hz, Channels: {num_channels}, Duration: {num_frames / sample_rate:.2f}s")
            
            # Read all frames
            audio_data = wav_file.readframes(num_frames)
            
            # Convert bytes to numpy array based on sample width
            if sample_width == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int32).astype(np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            return audio_array, sample_rate
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if original_rate == target_rate:
            return audio_data
        
        logger.info(f"Resampling audio from {original_rate}Hz to {target_rate}Hz")
        
        # Simple linear interpolation resampling
        ratio = target_rate / original_rate
        new_length = int(len(audio_data) * ratio)
        
        resampled = np.interp(
            np.linspace(0, len(audio_data), new_length),
            np.arange(len(audio_data)),
            audio_data
        ).astype(np.int16)
        
        return resampled
    
    @staticmethod
    def create_looped_audio(audio_data: np.ndarray, loop_count: int = 5) -> np.ndarray:
        """Create a looped version of the audio data."""
        looped = np.tile(audio_data, loop_count)
        logger.info(f"Created {loop_count}x looped audio: {len(looped)} samples")
        return looped

class MultiChannelAudioTester:
    """Interactive multi-channel audio delay tester."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.sample_rate = 48000
        self.chunk_size = 1024
        self.channels = 4  # Default channels
        
        # Audio devices
        self.output_device_index: Optional[int] = None
        self.input_device_index: Optional[int] = None
        
        # Delay configuration
        self.channel_delays = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.channel_volumes = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        
        # Delay buffers - will be created after sample rate is set
        self.delay_buffers: Dict[int, DelayBuffer] = {}
        self.max_delay_seconds = 1.0
        
        # Audio objects
        self.pa = None
        self.output_stream = None
        self.input_stream = None
        
        # Click track
        self.click_generator = ClickTrackGenerator(self.sample_rate)
        self.current_track = None
        self.track_queue = queue.Queue()
        self.is_playing = False
        
        # Recording for auto-calibration
        self.recording_data = []
        self.is_recording = False
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Interactive mode state
        self.interactive_mode = False
        self.selected_channels = [0, 1]  # Default channels for interactive mode
        self.continuous_track = None
        self.interactive_bpm = 120
        
        # Voice audio for testing
        self.voice_audio = None
        self.voice_sample_rate = None
        self.use_voice_audio = False
            
    def load_config(self, config_path: str):
        """Load configuration from agent TOML file."""
        try:
            config = toml.load(config_path)
            pipecat_config = config.get('backend_config', {}).get('pipecat', {})
            
            # Load device settings
            device_name = pipecat_config.get('aggregate_device_name')
            if device_name:
                device_idx = find_audio_device_by_name(device_name, input_device=False)
                if device_idx is not None:
                    self.output_device_index = device_idx
                    logger.info(f"Using output device from config: {device_name} (index {device_idx})")
            
            # Load channel configuration
            self.channels = pipecat_config.get('output_channels', 4)
            self.channel_delays = pipecat_config.get('channel_delays', {})
            self.channel_volumes = pipecat_config.get('channel_volumes', {})
            
            # Ensure all channels have delays and volumes
            for ch in range(self.channels):
                if ch not in self.channel_delays:
                    self.channel_delays[ch] = 0.0
                if ch not in self.channel_volumes:
                    self.channel_volumes[ch] = 1.0
                    
            logger.info(f"Loaded config: {self.channels} channels, delays: {self.channel_delays}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def initialize_audio(self):
        """Initialize PyAudio streams."""
        try:
            with suppress_audio_errors():
                self.pa = pyaudio.PyAudio()
                
                # Prefer PipeWire device if not explicitly specified
                if self.output_device_index is None:
                    pw_idx = find_audio_device_by_name("pipewire", input_device=False)
                    if pw_idx is not None:
                        self.output_device_index = pw_idx
                        logger.info(f"Using output device: pipewire (index {pw_idx})")
                    else:
                        logger.warning("No output device specified, using PortAudio default")

                # Log selected/default device info for clarity
                try:
                    if self.output_device_index is not None:
                        dev_info = self.pa.get_device_info_by_index(self.output_device_index)
                    else:
                        dev_info = self.pa.get_default_output_device_info()
                    host_info = self.pa.get_host_api_info_by_index(dev_info.get("hostApi", 0))
                    logger.info(
                        "PortAudio output -> name='%s' index=%s host='%s' maxOut=%s",
                        dev_info.get("name"),
                        dev_info.get("index"),
                        host_info.get("name"),
                        dev_info.get("maxOutputChannels"),
                    )
                except Exception as e:
                    logger.debug(f"Unable to query PortAudio device info: {e}")
                
                # Initialize output stream
                self.output_stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.output_device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._output_callback
                )
                
                logger.info(f"Initialized audio output: {self.channels} channels @ {self.sample_rate}Hz")
                
                # Initialize delay buffers after sample rate is set
                self._initialize_delay_buffers()
                
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def _initialize_delay_buffers(self):
        """Initialize delay buffers for all channels."""
        self.delay_buffers = create_delay_buffers(
            self.channel_delays,
            self.sample_rate,
            self.max_delay_seconds
        )
    
    def _update_delay_buffer(self, channel: int, delay_seconds: float):
        """Update delay buffer for a specific channel."""
        max_delay_samples = int(self.max_delay_seconds * self.sample_rate)
        delay_samples = int(delay_seconds * self.sample_rate)
        
        # Create new delay buffer for this channel
        buffer = DelayBuffer(max_delay_samples)
        buffer.set_delay(delay_samples)
        self.delay_buffers[channel] = buffer
        
        logger.info(f"Updated delay buffer for channel {channel}: "
                   f"{delay_seconds:.3f}s ({delay_samples} samples)")
    
    def initialize_input(self, input_device_name: Optional[str] = None):
        """Initialize audio input for auto-calibration."""
        try:
            if input_device_name:
                self.input_device_index = find_audio_device_by_name(input_device_name, input_device=True)
                
            self.input_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,  # Mono input
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._input_callback
            )
            
            logger.info(f"Initialized audio input for auto-calibration")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio input: {e}")
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """Audio output callback - plays multi-channel audio with delays."""
        try:
            if not self.track_queue.empty():
                mono_chunk = self.track_queue.get_nowait()
                multi_channel_chunk = self._process_multi_channel(mono_chunk, frame_count)
                return (multi_channel_chunk, pyaudio.paContinue)
            else:
                # Return silence
                silence = np.zeros((frame_count, self.channels), dtype=np.int16)
                return (silence.tobytes(), pyaudio.paContinue)
                
        except Exception as e:
            logger.error(f"Output callback error: {e}")
            return (None, pyaudio.paComplete)
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback - records for auto-calibration."""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.recording_data.extend(audio_data)
        
        return (None, pyaudio.paContinue)
    
    def _process_multi_channel(self, mono_data: np.ndarray, frame_count: int) -> bytes:
        """Convert mono audio to multi-channel with delays and volumes using proper DelayBuffer."""
        if len(mono_data) < frame_count:
            # Pad with zeros if not enough data
            padded = np.zeros(frame_count, dtype=np.int16)
            padded[:len(mono_data)] = mono_data
            mono_data = padded
        
        # Convert to float32 for processing
        mono_float = mono_data.astype(np.float32)
        
        # Create multi-channel output
        multi_channel = np.zeros((frame_count, self.channels), dtype=np.float32)
        
        for ch in range(self.channels):
            volume = self.channel_volumes.get(ch, 1.0)
            
            if volume > 0.0:
                # Get delay buffer for this channel
                delay_buffer = self.delay_buffers.get(ch)
                
                if delay_buffer is not None:
                    # Apply delay using DelayBuffer
                    delayed_audio = delay_buffer.process_chunk(mono_float)
                    multi_channel[:, ch] = delayed_audio * volume
                else:
                    # No delay buffer, just apply volume
                    multi_channel[:, ch] = mono_float * volume
        
        # Convert to int16 interleaved format
        multi_channel = np.clip(multi_channel, -32768, 32767).astype(np.int16)
        return multi_channel.tobytes()
    
    def generate_and_queue_track(self, bpm: int = 120, duration: int = 5, 
                                frequency: int = 1000, channel: Optional[int] = None):
        """Generate click track and queue it for playback."""
        track = self.click_generator.generate_click_track(bpm, duration, frequency)
        
        # Normalize and convert to int16
        track = np.clip(track * 32767 * 0.5, -32768, 32767).astype(np.int16)
        
        # Queue audio chunks
        chunk_size = self.chunk_size
        for i in range(0, len(track), chunk_size):
            chunk = track[i:i + chunk_size]
            self.track_queue.put(chunk)
        
        logger.info(f"Queued {duration}s click track at {bpm} BPM, {frequency}Hz")
    
    def play_test_tone(self, channel: int, duration: int = 2, frequency: int = 1000):
        """Play a test tone on a specific channel."""
        logger.info(f"Playing test tone on channel {channel}")
        
        # Temporarily set only one channel to full volume
        original_volumes = self.channel_volumes.copy()
        for ch in range(self.channels):
            self.channel_volumes[ch] = 1.0 if ch == channel else 0.0
        
        # Generate and play tone
        self.generate_and_queue_track(bpm=120, duration=duration, frequency=frequency)
        time.sleep(duration + 0.5)
        
        # Restore original volumes
        self.channel_volumes = original_volumes
    
    def interactive_delay_adjustment(self):
        """Interactive delay adjustment interface."""
        print("\n" + "="*60)
        print("INTERACTIVE MULTI-CHANNEL AUDIO DELAY CALIBRATION")
        print("="*60)
        print(f"Channels: {self.channels}")
        print(f"Sample Rate: {self.sample_rate}Hz")
        print(f"Current delays (ms): {', '.join(f'Ch{ch}: {delay*1000:.1f}' for ch, delay in self.channel_delays.items())}")
        print("\nCommands:")
        print("  p <bpm> <duration> - Play click track (default: p 120 5)")
        print("  t <channel>        - Play test tone on specific channel")
        print("  d <channel> <ms>   - Set delay for channel (in milliseconds)")
        print("  v <channel> <vol>  - Set volume for channel (0.0-1.0)")
        print("  s                  - Show current settings")
        print("  c                  - Play click track on all channels")
        print("  r <duration>       - Record and analyze timing (auto-calibrate)")
        print("  w <filename>       - Write current config to file")
        print("  i [channels]       - Interactive mode with live delay adjustment (e.g., 'i 0,1')")
        print("  voice <filepath>   - Load voice audio file for testing")
        print("  play voice         - Play loaded voice audio")
        print("  q                  - Quit")
        print("-"*60)
        
        while True:
            try:
                cmd = input("\n> ").strip().lower().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'q':
                    break
                    
                elif cmd[0] == 'p':
                    bpm = int(cmd[1]) if len(cmd) > 1 else 120
                    duration = int(cmd[2]) if len(cmd) > 2 else 5
                    self.generate_and_queue_track(bpm, duration)
                    
                elif cmd[0] == 't':
                    if len(cmd) > 1:
                        channel = int(cmd[1])
                        if 0 <= channel < self.channels:
                            self.play_test_tone(channel)
                        else:
                            print(f"Channel must be 0-{self.channels-1}")
                    else:
                        print("Usage: t <channel>")
                        
                elif cmd[0] == 'd':
                    if len(cmd) > 2:
                        channel = int(cmd[1])
                        delay_ms = float(cmd[2])
                        if 0 <= channel < self.channels:
                            self.channel_delays[channel] = delay_ms / 1000.0
                            self._update_delay_buffer(channel, delay_ms / 1000.0)
                            print(f"Set channel {channel} delay to {delay_ms:.1f}ms")
                        else:
                            print(f"Channel must be 0-{self.channels-1}")
                    else:
                        print("Usage: d <channel> <delay_ms>")
                        
                elif cmd[0] == 'v':
                    if len(cmd) > 2:
                        channel = int(cmd[1])
                        volume = float(cmd[2])
                        if 0 <= channel < self.channels and 0.0 <= volume <= 1.0:
                            self.channel_volumes[channel] = volume
                            print(f"Set channel {channel} volume to {volume:.2f}")
                        else:
                            print(f"Channel must be 0-{self.channels-1}, volume 0.0-1.0")
                    else:
                        print("Usage: v <channel> <volume>")
                        
                elif cmd[0] == 's':
                    self.show_settings()
                    
                elif cmd[0] == 'c':
                    print("Playing click track on all channels...")
                    self.generate_and_queue_track(120, 5)
                
                elif cmd[0] == 'voice':
                    if len(cmd) > 1:
                        voice_path = cmd[1]
                        self.load_voice_audio(voice_path)
                    else:
                        print("Usage: voice <filepath>")
                        print("Example: voice media/audio/cartesia_sophie.wav")
                
                elif cmd[0] == 'play' and len(cmd) > 1 and cmd[1] == 'voice':
                    if self.voice_audio is not None:
                        print("Playing voice audio...")
                        self.play_voice_audio()
                    else:
                        print("No voice audio loaded. Use 'voice <filepath>' first.")
                    
                elif cmd[0] == 'r':
                    duration = int(cmd[1]) if len(cmd) > 1 else 3
                    if self.input_stream:
                        self.auto_calibrate_delays(duration)
                    else:
                        print("Input stream not available. Start with --auto-calibrate option.")
                        
                elif cmd[0] == 'w':
                    filename = cmd[1] if len(cmd) > 1 else "calibrated_delays.toml"
                    self.write_config(filename)
                
                elif cmd[0] == 'i':
                    # Interactive mode
                    if len(cmd) > 1:
                        try:
                            # Parse selected channels
                            self.selected_channels = [int(ch.strip()) for ch in cmd[1].split(',')]
                            self.selected_channels = [ch for ch in self.selected_channels if 0 <= ch < self.channels]
                            print(f"Selected channels for interactive mode: {self.selected_channels}")
                        except ValueError:
                            print("Invalid channel format. Use: i 0,1,2")
                            continue
                    else:
                        self.selected_channels = [0, 1]  # Default
                    
                    self.run_interactive_mode()
                    
                else:
                    print("Unknown command. Available: p, t, d, v, s, c, r, w, i, voice, q")
                    print("Usage: voice <filepath> - Load voice audio file")
                    print("       play voice - Play loaded voice audio") 
                    print("       i [channels] - Interactive mode (e.g., 'i 0,1' for channels 0 and 1)")
                    
            except (ValueError, IndexError) as e:
                print(f"Invalid command format: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception as e:
                logger.error(f"Command error: {e}")
    
    def show_settings(self):
        """Display current delay and volume settings."""
        print("\nCurrent Settings:")
        print("-" * 40)
        for ch in range(self.channels):
            delay_ms = self.channel_delays.get(ch, 0.0) * 1000
            volume = self.channel_volumes.get(ch, 1.0)
            print(f"Channel {ch}: {delay_ms:6.1f}ms delay, {volume:4.2f} volume")
        print("-" * 40)
    
    def run_interactive_mode(self):
        """Run interactive mode with continuous audio and live delay adjustment."""
        print("\n" + "="*60)
        print("INTERACTIVE DELAY ADJUSTMENT MODE")
        print("="*60)
        print(f"Selected channels: {self.selected_channels}")
        
        # Check if we have voice audio loaded
        if self.voice_audio is not None:
            print("Using loaded voice audio for testing")
            audio_type = "voice"
            current_audio_mode = "voice"
        else:
            print("Using click track audio for testing")
            audio_type = "clicks"  
            current_audio_mode = "clicks"
            
        print("Keyboard controls:")
        print("  ,/.    - Adjust delay ±1ms on ALL selected channels")
        print("  </>    - Adjust delay ±5ms on ALL selected channels") 
        print("  1-4    - Toggle channel in/out of selection")
        if audio_type == "clicks":
            print("  f/s    - Click interval faster/slower")
        print("  v      - Toggle voice/clicks mode (if voice loaded)")
        print("  r      - Reset all selected channels to 0ms")
        print("  ESC/q  - Exit interactive mode")
        print("-"*60)
        
        # Start continuous audio based on type
        if audio_type == "voice":
            self.start_continuous_voice()
        else:
            self.start_continuous_clicks()
        
        # Setup terminal for single-key input
        import termios
        import tty
        import select
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        current_channel = self.selected_channels[0] if self.selected_channels else 0
        
        try:
            tty.setcbreak(fd)
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~termios.ECHO  # Disable echo
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            
            last_update = time.time()
            
            while True:
                # Check for keyboard input
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    key = sys.stdin.read(1)
                    
                    if key in ['\x1b', 'q']:  # ESC or q
                        break
                    elif key == ',':
                        self._adjust_selected_channels(-1)
                    elif key == '.':
                        self._adjust_selected_channels(+1)
                    elif key == '<':
                        self._adjust_selected_channels(-5)
                    elif key == '>':
                        self._adjust_selected_channels(+5)
                    elif key == 'f':
                        if current_audio_mode == "clicks":
                            self.interactive_bpm = min(200, self.interactive_bpm + 10)
                            self.restart_continuous_clicks()
                            print(f"\rBPM: {self.interactive_bpm}", end='', flush=True)
                    elif key == 's':
                        if current_audio_mode == "clicks":
                            self.interactive_bpm = max(60, self.interactive_bpm - 10)
                            self.restart_continuous_clicks()
                            print(f"\rBPM: {self.interactive_bpm}", end='', flush=True)
                    elif key == 'v':
                        # Toggle between voice and clicks mode
                        if self.voice_audio is not None:
                            if current_audio_mode == "voice":
                                current_audio_mode = "clicks"
                                self.stop_continuous_voice()
                                self.start_continuous_clicks()
                                print(f"\rSwitched to clicks mode", end='', flush=True)
                            else:
                                current_audio_mode = "voice"
                                self.stop_continuous_clicks()  
                                self.start_continuous_voice()
                                print(f"\rSwitched to voice mode", end='', flush=True)
                        else:
                            print(f"\rNo voice audio loaded", end='', flush=True)
                    elif key == 'r':
                        # Reset all selected channels to 0
                        for ch in self.selected_channels:
                            self._adjust_channel_delay(ch, -self.channel_delays.get(ch, 0.0) * 1000)
                        print(f"\rReset all delays", end='', flush=True)
                
                # Update status display
                now = time.time()
                if now - last_update > 0.5:
                    last_update = now
                    all_delays_str = ', '.join(f'Ch{ch}:{self.channel_delays.get(ch, 0.0)*1000:.1f}ms' 
                                             for ch in range(self.channels))
                    selected_str = f"[{','.join(map(str, self.selected_channels))}]"
                    mode_str = f"Mode: {current_audio_mode}"
                    if current_audio_mode == "clicks":
                        mode_str += f" BPM: {self.interactive_bpm}"
                    status = f"\rSelected: {selected_str} | All: {all_delays_str} | {mode_str}"
                    # Clear line and write status
                    sys.stdout.write("\r\033[K" + status)
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            pass
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            # Stop whatever audio mode is currently running
            if current_audio_mode == "voice":
                self.stop_continuous_voice()
            else:
                self.stop_continuous_clicks()
            print("\n\nExited interactive mode")
    
    def _adjust_channel_delay(self, channel: int, delta_ms: float):
        """Adjust delay for a specific channel and update the delay buffer."""
        if channel not in range(self.channels):
            return
        
        current_delay_ms = self.channel_delays.get(channel, 0.0) * 1000
        new_delay_ms = max(0.0, current_delay_ms + delta_ms)
        new_delay_seconds = new_delay_ms / 1000.0
        
        self.channel_delays[channel] = new_delay_seconds
        self._update_delay_buffer(channel, new_delay_seconds)
    
    def _adjust_selected_channels(self, delta_ms: float):
        """Adjust delay for all selected channels simultaneously."""
        for channel in self.selected_channels:
            self._adjust_channel_delay(channel, delta_ms)
        
        # Update status display with all channels
        all_delays_str = ', '.join([f"Ch{ch}: {self.channel_delays.get(ch, 0.0)*1000:.1f}ms" 
                                   for ch in range(self.channels)])
        selected_str = f"[{','.join(map(str, self.selected_channels))}]"
        print(f"\rSelected: {selected_str} | All: {all_delays_str}     ", end='', flush=True)
    
    def start_continuous_clicks(self):
        """Start continuous click track for interactive mode."""
        # Generate a very long click track
        duration = 300  # 5 minutes worth
        track = self.click_generator.generate_click_track(self.interactive_bpm, duration, 1000)
        track = np.clip(track * 32767 * 0.5, -32768, 32767).astype(np.int16)
        
        # Queue the track
        chunk_size = self.chunk_size
        for i in range(0, len(track), chunk_size):
            chunk = track[i:i + chunk_size]
            self.track_queue.put(chunk)
            
        self.is_playing = True
        logger.info(f"Started continuous clicks at {self.interactive_bpm} BPM")
    
    def restart_continuous_clicks(self):
        """Restart continuous clicks with new BPM."""
        self.stop_continuous_clicks()
        time.sleep(0.1)  # Brief pause
        self.start_continuous_clicks()
    
    def stop_continuous_clicks(self):
        """Stop continuous click track."""
        self.is_playing = False
        # Clear the queue
        while not self.track_queue.empty():
            try:
                self.track_queue.get_nowait()
            except queue.Empty:
                break
    
    def start_continuous_voice(self):
        """Start continuous voice track for interactive mode."""
        if self.voice_audio is None:
            print("No voice audio loaded! Falling back to clicks.")
            self.start_continuous_clicks()
            return
            
        # Get voice audio and prepare for looping
        voice_audio = self.voice_audio.copy()
        
        # Add a brief pause between loops (0.5 seconds)
        pause_samples = int(0.5 * self.sample_rate)
        pause = np.zeros(pause_samples, dtype=voice_audio.dtype)
        
        # Create a loop that runs for about 5 minutes
        loop_with_pause = np.concatenate([voice_audio, pause])
        total_duration = 300  # 5 minutes
        loop_duration = len(loop_with_pause) / self.sample_rate
        num_loops = int(total_duration / loop_duration) + 1
        
        # Create extended track
        extended_track = np.tile(loop_with_pause, num_loops)
        
        # Queue the track - just like clicks!
        chunk_size = self.chunk_size
        for i in range(0, len(extended_track), chunk_size):
            chunk = extended_track[i:i + chunk_size]
            self.track_queue.put(chunk)
            
        self.is_playing = True
        logger.info("Started continuous voice audio for interactive mode")
    
    def stop_continuous_voice(self):
        """Stop continuous voice track (same as clicks)."""
        self.stop_continuous_clicks()
        
    def restart_continuous_voice(self):
        """Restart continuous voice audio."""
        self.stop_continuous_voice()
        time.sleep(0.1)  # Brief pause
        self.start_continuous_voice()
    
    def auto_calibrate_delays(self, duration: int = 3):
        """Attempt to auto-calibrate delays using microphone feedback."""
        print(f"\nAuto-calibrating delays using microphone...")
        print("Make sure microphone can hear all speakers.")
        print(f"Recording for {duration} seconds...")
        
        # Clear previous recording
        self.recording_data = []
        
        # Start recording
        self.is_recording = True
        
        # Play click track
        self.generate_and_queue_track(bpm=60, duration=duration, frequency=2000)
        
        # Wait for recording to complete
        time.sleep(duration + 1)
        
        # Stop recording
        self.is_recording = False
        
        if len(self.recording_data) > 0:
            self._analyze_recording()
        else:
            print("No audio data recorded.")
    
    def _analyze_recording(self):
        """Analyze recorded audio to detect timing differences."""
        print("Analyzing recorded audio for timing differences...")
        
        recording = np.array(self.recording_data, dtype=np.int16)
        
        # Simple peak detection to find click timings
        # This is a basic implementation - could be improved with more sophisticated analysis
        
        # Apply high-pass filter to isolate clicks
        nyquist = self.sample_rate // 2
        b, a = signal.butter(4, 1000 / nyquist, btype='high')
        filtered = signal.filtfilt(b, a, recording.astype(np.float32))
        
        # Find peaks
        peaks, _ = signal.find_peaks(np.abs(filtered), height=np.max(np.abs(filtered)) * 0.3, distance=self.sample_rate//4)
        
        if len(peaks) > 1:
            # Calculate intervals between peaks
            intervals = np.diff(peaks) / self.sample_rate
            avg_interval = np.mean(intervals)
            
            print(f"Detected {len(peaks)} clicks")
            print(f"Average interval: {avg_interval:.3f}s")
            print(f"Expected interval: 1.000s (60 BPM)")
            
            # This is a simplified analysis - in practice, you'd need to:
            # 1. Play clicks on individual channels
            # 2. Measure arrival times for each channel
            # 3. Calculate relative delays
            # 4. Adjust delays to align all channels
            
            timing_error = abs(avg_interval - 1.0)
            if timing_error < 0.01:
                print("✓ Timing looks good!")
            else:
                print(f"⚠ Timing error detected: {timing_error*1000:.1f}ms")
                print("Manual adjustment may be needed.")
        else:
            print("Could not detect sufficient clicks for analysis.")
            print("Try increasing volume or moving microphone closer to speakers.")
    
    def write_config(self, filename: str):
        """Write current delay configuration to TOML file."""
        config = {
            'backend_config': {
                'pipecat': {
                    'multi_channel_output': True,
                    'output_channels': self.channels,
                    'channel_delays': self.channel_delays,
                    'channel_volumes': self.channel_volumes
                }
            }
        }
        
        try:
            with open(filename, 'w') as f:
                toml.dump(config, f)
            print(f"Configuration written to {filename}")
        except Exception as e:
            logger.error(f"Failed to write config: {e}")
    
    def load_voice_audio(self, voice_file_path: str):
        """Load voice audio for testing."""
        try:
            audio_data, original_rate = VoiceAudioLoader.load_wav_file(voice_file_path)
            
            # Resample if needed
            if original_rate != self.sample_rate:
                audio_data = VoiceAudioLoader.resample_audio(
                    audio_data, original_rate, self.sample_rate
                )
            
            # Create looped version (3 loops for ~15 seconds)
            self.voice_audio = VoiceAudioLoader.create_looped_audio(audio_data, loop_count=3)
            
            print(f"Voice audio loaded successfully: {len(self.voice_audio)} samples")
            print(f"Duration: {len(self.voice_audio) / self.sample_rate:.2f} seconds")
            
        except Exception as e:
            print(f"Failed to load voice audio: {e}")
            logger.error(f"Voice audio loading error: {e}")
            self.voice_audio = None
    
    def play_voice_audio(self):
        """Play loaded voice audio through multi-channel system."""
        if self.voice_audio is None:
            print("No voice audio loaded")
            return
        
        print("Playing voice audio on all channels...")
        
        # Convert to the same format as click tracks
        audio_data = np.clip(self.voice_audio * 0.5, -32768, 32767).astype(np.int16)
        
        # Queue the audio in chunks
        chunk_size = self.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            self.track_queue.put(chunk)
        
        self.is_playing = True
        print(f"Queued {len(audio_data)/self.sample_rate:.2f}s of voice audio")
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        if self.input_stream:
            self.input_stream.stop_stream() 
            self.input_stream.close()
            
        if self.pa:
            self.pa.terminate()
            
        logger.info("Audio resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description="Multi-channel audio delay calibration tool")
    parser.add_argument('--config', '-c', help="Agent configuration file to load")
    parser.add_argument('--file', '-f', help="Audio file to load for testing (e.g., media/audio/cartesia_sophie.wav)")
    parser.add_argument('--auto-calibrate', '-a', action='store_true', 
                       help="Enable auto-calibration with microphone")
    parser.add_argument('--input-device', '-i', help="Input device name for auto-calibration")
    parser.add_argument('--output-device', '-o', help="Output device name")
    parser.add_argument('--list-devices', '-l', action='store_true', help="List audio devices and exit")
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help="Enable verbose logging (INFO level)")
    
    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.list_devices:
        print("Available Audio Devices:")
        print("-" * 50)
        devices = list_audio_devices()
        for device in devices:
            channels_info = f"{device['max_input_channels']}in/{device['max_output_channels']}out"
            print(f"Index {device['index']:2}: {device['name']} [{channels_info}]")
        return
    
    # Create tester
    tester = MultiChannelAudioTester(args.config)
    
    try:
        # Override output device if specified
        if args.output_device:
            device_idx = find_audio_device_by_name(args.output_device, input_device=False)
            if device_idx is not None:
                tester.output_device_index = device_idx
                logger.info(f"Using output device: {args.output_device} (index {device_idx})")
        
        # Initialize audio
        tester.initialize_audio()
        
        # Load audio file if provided
        if args.file:
            print(f"Loading audio file: {args.file}")
            tester.load_voice_audio(args.file)
        
        # Initialize input for auto-calibration if requested
        if args.auto_calibrate:
            tester.initialize_input(args.input_device)
        
        # Start interactive interface
        tester.interactive_delay_adjustment()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()

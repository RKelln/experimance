"""
Audio generation base classes for the image server.

This module provides abstract base classes for audio generators,
similar to the ImageGenerator pattern but for audio generation.
"""

import asyncio
import logging
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Set
import numpy as np
import torch
import torchaudio

from image_server.generators.config import BaseGeneratorConfig
from image_server.generators.audio.audio_config import BaseAudioGeneratorConfig
from experimance_common.logger import configure_external_loggers

# Configure logging
logger = logging.getLogger(__name__)

VALID_AUDIO_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'm4a']


class AudioGeneratorCapabilities:
    """Defines the capabilities that audio generators can support."""
    
    # Core generation capabilities
    TEXT_TO_AUDIO = "text_to_audio"           # Generate audio from text prompts
    ENVIRONMENTAL_SOUNDS = "environmental_sounds"  # Generate environmental/ambient audio
    SEAMLESS_LOOPS = "seamless_loops"         # Create seamlessly looping audio
    CUSTOM_DURATION = "custom_duration"       # Support custom audio durations
    SEMANTIC_CACHING = "semantic_caching"     # Intelligent caching with similarity matching
    
    # Quality and processing features
    LOUDNESS_NORMALIZATION = "loudness_normalization"  # EBU R128 loudness normalization
    CLAP_SCORING = "clap_scoring"             # CLAP similarity scoring for quality assessment
    VARIANT_GENERATION = "variant_generation"  # Generate multiple variants per prompt
    BACKGROUND_PREFETCH = "background_prefetch"  # Background prefetching of variants
    
    @classmethod
    def all_capabilities(cls) -> Set[str]:
        """Get all defined capabilities."""
        return {
            value for name, value in cls.__dict__.items() 
            if isinstance(value, str) and not name.startswith('_')
        }


class AudioNormalizer:
    """Utility class for audio loudness normalization using EBU R128 standard."""
    
    @staticmethod
    def normalize_loudness(
        audio_path: str,
        target_lufs: float = -23.0,
        true_peak_dbfs: float = -2.0,
        output_path: Optional[str] = None,
        temp_dir: Optional[str] = None
    ) -> str:
        """
        Normalize audio loudness using ffmpeg-normalize with EBU R128 standard.
        
        Args:
            audio_path: Path to input audio file
            target_lufs: Target loudness in LUFS (default -23.0 for EBU R128)
            true_peak_dbfs: True peak limit in dBFS (default -2.0)
            output_path: Output path (if None, creates normalized version)
            temp_dir: Temporary directory for processing
            
        Returns:
            Path to normalized audio file
            
        Raises:
            RuntimeError: If normalization fails
        """
        try:
            input_path = Path(audio_path)
            
            if output_path is None:
                # Create output path with _normalized suffix
                output_path = input_path.with_stem(f"{input_path.stem}_normalized")
            else:
                output_path = Path(output_path)
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build ffmpeg-normalize command
            cmd = [
                "ffmpeg-normalize",
                str(input_path),
                "--target-level", str(target_lufs),
                "--true-peak", str(true_peak_dbfs),
                "--loudness-range-target", "7.0",  # EBU R128 recommendation
                "--keep-loudness-range-target",
                "--offset", "0",
                "--dual-mono",
                "-c:a", "libmp3lame",  # Use MP3 codec
                "-b:a", "192k",        # 192 kbps bitrate
                "-o", str(output_path)
            ]
            
            # Run normalization
            logger.debug(f"Running audio normalization: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Audio normalized successfully: {output_path}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio normalization failed: {e}")
            logger.error(f"Command: {' '.join(e.cmd)}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Audio normalization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during audio normalization: {e}")
            raise RuntimeError(f"Audio normalization failed: {e}")
    
    @staticmethod
    def make_seamless_loop(
        audio_tensor: torch.Tensor,
        sample_rate: int,
        tail_duration_s: float = 1.5
    ) -> torch.Tensor:
        """
        Make audio seamlessly loopable using equal-power crossfade.
        
        Args:
            audio_tensor: Audio tensor [channels, samples] or [samples]
            sample_rate: Sample rate of the audio
            tail_duration_s: Duration of crossfade in seconds
            
        Returns:
            Seamlessly looping audio tensor
            
        Raises:
            ValueError: If tail duration is too large for audio length
        """
        # Ensure tensor is 2D [channels, samples]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        channels, total_samples = audio_tensor.shape
        crossfade_samples = int(tail_duration_s * sample_rate)
        
        if crossfade_samples <= 0 or 2 * crossfade_samples >= total_samples:
            raise ValueError(
                f"Crossfade duration ({tail_duration_s}s = {crossfade_samples} samples) "
                f"is too large for audio length ({total_samples} samples)"
            )
        
        # Extract head and tail sections for crossfade
        head = audio_tensor[:, :crossfade_samples].clone()
        tail = audio_tensor[:, -crossfade_samples:].clone()
        
        # Create equal-power crossfade curves
        t = torch.linspace(0.0, 1.0, crossfade_samples, device=audio_tensor.device)
        fade_in = torch.sin(0.5 * torch.pi * t)    # Rising curve
        fade_out = torch.cos(0.5 * torch.pi * t)   # Falling curve
        
        # Apply crossfade
        blended = head * fade_out + tail * fade_in
        
        # Create output: original audio minus tail, with crossfaded start
        output = audio_tensor[:, :-crossfade_samples].clone()
        output[:, :crossfade_samples] = blended
        
        return output.squeeze(0) if channels == 1 else output


class AudioGenerator(ABC):
    """Abstract base class for audio generation strategies."""
    
    # Generator capabilities - subclasses should override this
    supported_capabilities: Set[str] = set()
    
    def __init__(self, config: BaseAudioGeneratorConfig, output_dir: str = "/tmp", max_concurrent: int = 1, **kwargs):
        """Initialize the audio generator.
        
        Args:
            config: Audio generator configuration
            output_dir: Directory to save generated audio files
            max_concurrent: Maximum number of concurrent generations (default 1 for thread-safety)
            **kwargs: Additional configuration options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Initialize queuing mechanism for thread-safe generation
        self._generation_queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_task = None
        self._is_running = False
        self._pending_requests = {}  # Track pending requests for cancellation
        
        # Audio-specific setup
        self.normalizer = AudioNormalizer() if config.normalize_loudness else None
        
        self._configure(config, **kwargs)
    
    def _configure(self, config, **kwargs):
        """Configure generator-specific settings.
        
        Subclasses can override this to handle their specific configuration.
        """
        pass

    def supports_capability(self, capability: str) -> bool:
        """Check if this generator supports a specific capability.
        
        Args:
            capability: Capability to check (use AudioGeneratorCapabilities constants)
            
        Returns:
            True if the generator supports the capability
        """
        return capability in self.supported_capabilities
    
    def get_supported_capabilities(self) -> Set[str]:
        """Get all capabilities supported by this generator.
        
        Returns:
            Set of supported capability strings
        """
        return set(self.supported_capabilities)
    
    @classmethod
    def supports_capability_class(cls, capability: str) -> bool:
        """Class method to check capability without instantiating.
        
        Args:
            capability: Capability to check (use AudioGeneratorCapabilities constants)
            
        Returns:
            True if the generator class supports the capability
        """
        return capability in getattr(cls, 'supported_capabilities', set())

    async def _process_generation_queue(self):
        """Process generation requests from the queue."""
        while self._is_running:
            try:
                # Get next request from queue
                request_data = await self._generation_queue.get()
                
                if request_data is None:  # Shutdown signal
                    break
                
                request_id, prompt, kwargs, future = request_data
                
                # Check if request was cancelled
                if request_id not in self._pending_requests:
                    self._generation_queue.task_done()
                    continue
                
                # Process request with semaphore for concurrency control
                async with self._semaphore:
                    try:
                        if not future.cancelled():
                            result = await self._generate_audio_impl(prompt, **kwargs)
                            future.set_result(result)
                    except Exception as e:
                        if not future.cancelled():
                            future.set_exception(e)
                    finally:
                        # Clean up pending request
                        self._pending_requests.pop(request_id, None)
                        self._generation_queue.task_done()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio generation queue processor: {e}")

    async def generate_audio(self, prompt: str, **kwargs) -> str:
        """Generate audio using the queue system for thread safety.
        
        This method queues the request and returns when generation is complete.
        
        Args:
            prompt: Text description of the audio to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Path to the generated audio file
        """
        if not self._is_running:
            await self.start()
        
        # Create unique request ID and future for this request
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        # Add to pending requests for tracking
        self._pending_requests[request_id] = future
        
        # Queue the request
        await self._generation_queue.put((request_id, prompt, kwargs, future))
        
        # Wait for completion
        try:
            return await future
        except asyncio.CancelledError:
            # Clean up if cancelled
            self._pending_requests.pop(request_id, None)
            raise

    @abstractmethod
    async def _generate_audio_impl(self, prompt: str, **kwargs) -> str:
        """Internal implementation of audio generation.
        
        This is the method that subclasses should implement instead of generate_audio.
        
        Args:
            prompt: Text description of the audio to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Path to the generated audio file
            
        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        pass

    async def start(self):
        """Start the generator and queue processing.
        
        This method can be overridden by subclasses to implement pre-warming logic.
        """
        if not self._is_running:
            self._is_running = True
            self._queue_task = asyncio.create_task(self._process_generation_queue())
            logger.debug(f"{self.__class__.__name__}: Audio generation queue processor started")

    async def stop(self):
        """Stop the generator and queue processing.
        
        This method should be extended by subclasses to handle their specific cleanup.
        """
        if self._is_running:
            self._is_running = False
            
            # Cancel all pending requests
            for future in self._pending_requests.values():
                if not future.cancelled():
                    future.cancel()
            self._pending_requests.clear()
            
            # Signal queue processor to stop and wait for completion
            if self._queue_task:
                await self._generation_queue.put(None)  # Signal to stop
                try:
                    await asyncio.wait_for(self._queue_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"{self.__class__.__name__}: Queue processor didn't stop cleanly, cancelling")
                    self._queue_task.cancel()
                    try:
                        await self._queue_task
                    except asyncio.CancelledError:
                        pass
            
            logger.debug(f"{self.__class__.__name__}: Audio generator stopped")

    def _validate_prompt(self, prompt: str):
        """Validate the prompt for audio generation.
        
        Args:
            prompt: Text prompt to validate
            
        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt.strip()) < 3:
            raise ValueError("Prompt must be at least 3 characters long")

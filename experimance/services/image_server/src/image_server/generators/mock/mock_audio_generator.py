"""
Mock audio generator for testing purposes.
"""

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Optional, List
import numpy as np
import soundfile as sf

from image_server.generators.audio.audio_generator import AudioGenerator, AudioGeneratorCapabilities
from image_server.generators.config import BaseGeneratorConfig
from .mock_audio_generator_config import MockAudioGeneratorConfig

# Configure logging
logger = logging.getLogger(__name__)


class MockAudioGenerator(AudioGenerator):
    """Mock audio generator for testing purposes.
    
    Can either generate simple placeholder audio (tones, silence, noise) or 
    use existing audio files from a specified directory for more realistic testing.
    """
    
    # Mock generator supports most capabilities for testing purposes
    supported_capabilities = {
        AudioGeneratorCapabilities.TEXT_TO_AUDIO,
        AudioGeneratorCapabilities.ENVIRONMENTAL_SOUNDS,
        AudioGeneratorCapabilities.SEAMLESS_LOOPS,
        AudioGeneratorCapabilities.CUSTOM_DURATION,
        AudioGeneratorCapabilities.SEMANTIC_CACHING,
        AudioGeneratorCapabilities.LOUDNESS_NORMALIZATION,
    }
    
    def _configure(self, config: BaseGeneratorConfig, **kwargs):
        """Configure mock audio generator settings."""
        self.config = MockAudioGeneratorConfig(**{
            **config.model_dump(),
            **kwargs
        })
        
        # If using existing audio files, validate the directory and collect audio files
        self._existing_audio: List[Path] = []
        if self.config.use_existing_audio and self.config.existing_audio_dir:
            self._load_existing_audio()
            
        logger.info(f"MockAudioGenerator configured: use_existing={self.config.use_existing_audio}, "
                   f"placeholder_type={self.config.placeholder_type}")
    
    def _load_existing_audio(self) -> None:
        """Load list of existing audio files from the configured directory."""
        if not self.config.existing_audio_dir or not self.config.existing_audio_dir.exists():
            logger.warning(f"Existing audio directory not found: {self.config.existing_audio_dir}")
            self.config.use_existing_audio = False
            return
            
        # Find all audio files (common formats)
        from image_server.generators.audio.audio_generator import VALID_AUDIO_EXTENSIONS
        self._existing_audio = [
            f for f in self.config.existing_audio_dir.rglob("*")
            if f.suffix.lower().lstrip('.') in VALID_AUDIO_EXTENSIONS and f.is_file()
        ]
        
        logger.info(f"Found {len(self._existing_audio)} existing audio files in {self.config.existing_audio_dir}")
        
        if not self._existing_audio:
            logger.warning("No audio files found in existing audio directory, falling back to placeholder generation")
            self.config.use_existing_audio = False
    
    async def _generate_audio_impl(self, prompt: str, **kwargs) -> str:
        """Generate mock audio based on configuration."""
        logger.info(f"Generating mock audio for prompt: '{prompt[:100]}...'")
        
        # Simulate generation time
        if self.config.generation_delay_s > 0:
            await asyncio.sleep(self.config.generation_delay_s)
        
        if self.config.use_existing_audio and self._existing_audio:
            return await self._copy_existing_audio(prompt, **kwargs)
        else:
            return await self._generate_placeholder_audio(prompt, **kwargs)
    
    async def _copy_existing_audio(self, prompt: str, **kwargs) -> str:
        """Copy an existing audio file for testing."""
        # Select audio file based on hash of prompt for consistency
        file_index = hash(prompt) % len(self._existing_audio)
        source_file = self.config.existing_audio_dir / self._existing_audio[file_index]
        
        # Create unique output filename
        timestamp = int(time.time() * 1000)
        output_filename = f"mock_audio_{timestamp}_{hash(prompt) % 10000}.wav"
        output_path = Path(self.output_dir) / output_filename
        
        # Copy the file
        import shutil
        shutil.copy2(source_file, output_path)
        
        logger.info(f"Copied existing audio: {source_file.name} -> {output_path.name}")
        return str(output_path)
    
    async def _generate_placeholder_audio(self, prompt: str, **kwargs) -> str:
        """Generate placeholder audio based on configuration."""
        # Get duration from kwargs or config
        duration = kwargs.get('duration_s', self.config.duration_s)
        sample_rate = kwargs.get('sample_rate', self.config.sample_rate)
        
        # Generate audio data based on placeholder type
        if self.config.placeholder_type == "silence":
            audio_data = np.zeros(int(duration * sample_rate), dtype=np.float32)
            
        elif self.config.placeholder_type == "noise":
            # Generate white noise at a reasonable volume
            audio_data = np.random.normal(0, 0.1, int(duration * sample_rate)).astype(np.float32)
            
        elif self.config.placeholder_type == "tone":
            # Generate a sine wave tone
            t = np.linspace(0, duration, int(duration * sample_rate), False)
            # Vary frequency slightly based on prompt for some variation
            freq_variation = (hash(prompt) % 100) / 100.0  # 0-1
            frequency = self.config.tone_frequency * (0.8 + 0.4 * freq_variation)  # ±20% variation
            audio_data = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
            
            # Add fade in/out to avoid clicks
            fade_samples = int(0.1 * sample_rate)  # 100ms fade
            audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        else:
            raise ValueError(f"Unknown placeholder type: {self.config.placeholder_type}")
        
        # Create unique output filename
        timestamp = int(time.time() * 1000)
        output_filename = f"mock_audio_{self.config.placeholder_type}_{timestamp}.wav"
        output_path = Path(self.output_dir) / output_filename
        
        # Write audio file
        sf.write(str(output_path), audio_data, sample_rate)
        
        # Add metadata if requested
        if self.config.include_prompt_metadata:
            try:
                import soundfile as sf
                # Note: WAV files support limited metadata, but we can try
                with sf.SoundFile(str(output_path), 'r+') as f:
                    # This might not work for all formats, but it's a mock generator
                    pass
            except Exception as e:
                logger.debug(f"Could not add metadata to audio file: {e}")
        
        logger.info(f"Generated {self.config.placeholder_type} placeholder audio: "
                   f"{duration}s at {sample_rate}Hz -> {output_path.name}")
        
        return str(output_path)
    
    async def stop(self):
        """Stop the mock audio generator."""
        logger.info("MockAudioGenerator stopped")

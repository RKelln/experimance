"""
Mock audio generator configuration for testing purposes.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field

from image_server.generators.audio.audio_config import BaseAudioGeneratorConfig
from experimance_common.constants import GENERATED_AUDIO_DIR


class MockAudioGeneratorConfig(BaseAudioGeneratorConfig):
    """Configuration for mock audio generation used in testing.
    
    Similar to MockImageGenerator, this can either generate simple placeholder 
    audio files or use existing audio files from a specified directory.
    """
    
    strategy: Literal["mock_audio"] = "mock_audio"
    
    # Mock-specific configuration
    use_existing_audio: bool = Field(
        default=False, 
        description="Use existing audio files instead of generating placeholders"
    )
    
    existing_audio_dir: Optional[Path] = Field(
        default=GENERATED_AUDIO_DIR,
        description="Directory containing existing audio files to use for testing"
    )
    
    placeholder_type: Literal["tone", "silence", "noise"] = Field(
        default="tone",
        description="Type of placeholder audio to generate: tone, silence, or noise"
    )
    
    tone_frequency: float = Field(
        default=440.0,
        description="Frequency in Hz for tone placeholder (A4 = 440Hz)"
    )
    
    generation_delay_s: float = Field(
        default=1.0,
        description="Artificial delay to simulate generation time"
    )
    
    include_prompt_metadata: bool = Field(
        default=True,
        description="Include prompt text in audio file metadata"
    )

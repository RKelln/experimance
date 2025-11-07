"""
Monkey patch for SileroVADAnalyzer to fix sample rate override issues.

The issue: SileroVADAnalyzer validates sample rates before calling super().set_sample_rate(),
which prevents the base class logic (_init_sample_rate or sample_rate) from working.
This breaks resampling workflows where the transport tries to set device rate (48kHz)
but the VAD should stay at its initialized rate (16kHz).

This patch makes SileroVADAnalyzer respect the base class _init_sample_rate logic.
"""

import logging
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger = logging.getLogger(__name__)

# Store the original set_sample_rate method
_original_silero_set_sample_rate = SileroVADAnalyzer.set_sample_rate


def patched_silero_set_sample_rate(self, sample_rate: int):
    """
    Patched SileroVADAnalyzer.set_sample_rate that respects _init_sample_rate.
    
    This allows the base class logic to work properly:
    - If initialized with a specific sample rate, that rate is preserved
    - Only validates the sample rate that will actually be used
    - Enables proper resampling workflows: Device(48kHz) → Filter → VAD(16kHz)
    """
    # Use the same logic as the base class
    target_rate = self._init_sample_rate or sample_rate
    
    # Only validate the rate we're actually going to use
    if target_rate != 16000 and target_rate != 8000:
        raise ValueError(
            f"Silero VAD sample rate needs to be 16000 or 8000 (target rate: {target_rate})"
        )
    
    # Call the base class method directly (which will set target_rate correctly)
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer
    VADAnalyzer.set_sample_rate(self, sample_rate)
    
    logger.debug(f"SileroVAD: set_sample_rate({sample_rate}) → using {self._sample_rate}Hz")


def apply_silero_vad_patch():
    """Apply the monkey patch to SileroVADAnalyzer."""
    logger.info("Applying SileroVADAnalyzer sample rate fix (respects _init_sample_rate)")
    SileroVADAnalyzer.set_sample_rate = patched_silero_set_sample_rate


def remove_silero_vad_patch():
    """Remove the monkey patch and restore original behavior."""
    logger.info("Removing SileroVADAnalyzer sample rate patch")
    SileroVADAnalyzer.set_sample_rate = _original_silero_set_sample_rate

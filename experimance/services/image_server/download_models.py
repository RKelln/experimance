#!/usr/bin/env python3
"""
Download and cache audio generation models.

This script pre-downloads the required models for audio generation,
which can take some time on first run. Models are stored in the 
centralized MODELS_DIR location.
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any

# Import constants for model directory
sys.path.append(str(Path(__file__).parent.parent.parent / "libs" / "common" / "src"))
from experimance_common.constants import MODELS_DIR


def setup_model_cache():
    """Setup HuggingFace cache to use the centralized models directory."""
    # Set up audio models subdirectory
    audio_models_dir = MODELS_DIR / "audio"
    audio_models_dir.mkdir(exist_ok=True)
    
    # Configure HuggingFace cache location
    hf_cache_dir = audio_models_dir / "huggingface_cache"
    hf_cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables for HuggingFace
    os.environ['TRANSFORMERS_CACHE'] = str(hf_cache_dir)
    os.environ['HF_HOME'] = str(hf_cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(hf_cache_dir)
    
    print(f"📁 Models will be cached in: {hf_cache_dir}")
    return hf_cache_dir


def setup_logging():
    """Setup logging for model download progress."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def download_clap_model():
    """Download CLAP model for audio-text similarity."""
    print("📥 Downloading CLAP model...")
    print("   Model: laion/clap-htsat-unfused (~600MB)")
    
    try:
        from transformers.models.clap.modeling_clap import ClapModel
        from transformers.models.clap.processing_clap import ClapProcessor
        
        # Download processor
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        print("   ✓ CLAP processor downloaded")
        
        # Download model
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        print("   ✓ CLAP model downloaded")
        
        return True
        
    except Exception as e:
        print(f"   ✗ CLAP download failed: {e}")
        return False


def download_bge_model():
    """Download BGE model for semantic text embeddings (optional)."""
    print("📥 Downloading BGE embeddings model...")
    print("   Model: BAAI/bge-small-en-v1.5 (~130MB)")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download model
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("   ✓ BGE model downloaded")
        
        return True
        
    except Exception as e:
        print(f"   ✗ BGE download failed: {e}")
        print("   (This is optional - semantic caching will be disabled)")
        return False


def download_tangoflux_model():
    """Download TangoFlux model for audio generation."""
    print("📥 Downloading TangoFlux model...")
    print("   Model: declare-lab/TangoFlux (~2.8GB)")
    print("   This may take several minutes...")
    
    try:
        # Import TangoFlux (will trigger model download on first use)
        from tangoflux import TangoFluxInference
        
        # Initialize to trigger download
        inference = TangoFluxInference(device="cpu")  # Use CPU for initial download
        print("   ✓ TangoFlux model downloaded")
        
        return True
        
    except Exception as e:
        print(f"   ✗ TangoFlux download failed: {e}")
        return False


def check_disk_space():
    """Check available disk space for model downloads."""
    try:
        import shutil
        
        # Check space in home directory (where HuggingFace models are cached)
        home = Path.home()
        total, used, free = shutil.disk_usage(home)
        
        free_gb = free / (1024**3)
        required_gb = 4.0  # Approximate total for all models
        
        print(f"💾 Disk space check:")
        print(f"   Available: {free_gb:.1f}GB")
        print(f"   Required: ~{required_gb}GB")
        
        if free_gb < required_gb:
            print(f"   ⚠ Warning: Low disk space!")
            return False
        else:
            print(f"   ✓ Sufficient space available")
            return True
            
    except Exception as e:
        print(f"   ⚠ Could not check disk space: {e}")
        return True  # Continue anyway


def main():
    """Download all required models."""
    setup_logging()
    
    print("🎵 Audio Generation Model Downloader")
    print("=" * 40)
    print()
    
    # Setup model cache directory
    cache_dir = setup_model_cache()
    
    # Check disk space first
    if not check_disk_space():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    print()
    
    # Download models
    results = []
    
    # Essential models
    print("Downloading essential models...")
    results.append(("CLAP", download_clap_model()))
    results.append(("TangoFlux", download_tangoflux_model()))
    
    print()
    
    # Optional models
    print("Downloading optional models...")
    results.append(("BGE (optional)", download_bge_model()))
    
    # Summary
    print()
    print("=" * 40)
    print("DOWNLOAD SUMMARY:")
    
    essential_passed = 0
    optional_passed = 0
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        
        if success:
            if "optional" in name.lower():
                optional_passed += 1
            else:
                essential_passed += 1
    
    print()
    
    if essential_passed == 2:  # CLAP + TangoFlux
        print("🎉 Essential models downloaded successfully!")
        print("   Audio generation is ready to use.")
        
        if optional_passed > 0:
            print(f"   Bonus: {optional_passed} optional model(s) also available.")
        
        print()
        print("Next steps:")
        print("1. Run: python test_audio_deps.py")
        print("2. Start image server with: PROJECT_ENV=fire uv run -m image_server")
        return 0
        
    else:
        print("❌ Some essential models failed to download.")
        print("   Please check your internet connection and try again.")
        print("   See INSTALL_AUDIO_DEPS.md for troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

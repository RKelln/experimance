#!/usr/bin/env python3
"""
Subprocess worker for audio generation with isolated GPU environment.

This allows running audio generation on a different GPU than the main image server
by setting CUDA_VISIBLE_DEVICES before importing PyTorch.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_gpu_environment(gpu_id: Optional[int]):
    """Set CUDA_VISIBLE_DEVICES before any CUDA imports."""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id} for audio generation")
    else:
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Set CUDA_VISIBLE_DEVICES='' (CPU mode) for audio generation")


def run_audio_generation(config_data: Dict[str, Any], prompt: str, **kwargs) -> Dict[str, Any]:
    """Run audio generation in isolated environment."""
    try:
        # Import everything after GPU environment is set
        import asyncio
        from image_server.generators.audio.prompt2audio import Prompt2AudioGenerator
        from image_server.generators.audio.audio_config import Prompt2AudioConfig
        
        # Parse configuration
        config = Prompt2AudioConfig(**config_data)
        
        # Create generator
        generator = Prompt2AudioGenerator(config=config)
        
        # Run generation
        async def generate():
            await generator.start()
            try:
                result_path = await generator._generate_audio_impl(prompt, **kwargs)
                return {"success": True, "audio_path": result_path}
            finally:
                await generator.stop()
        
        # Execute the generation
        result = asyncio.run(generate())
        logger.info(f"Audio generation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def main():
    """Main entry point for subprocess worker."""
    if len(sys.argv) != 2:
        logger.error("Usage: subprocess_worker.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        # Load job parameters
        with open(json_file, 'r') as f:
            job_data = json.load(f)
        
        # Set GPU environment first
        gpu_id = job_data.get('gpu_id')
        set_gpu_environment(gpu_id)
        
        # Extract parameters
        config_data = job_data['config']
        prompt = job_data['prompt']
        kwargs = job_data.get('kwargs', {})
        
        # Run generation
        result = run_audio_generation(config_data, prompt, **kwargs)
        
        # Write result back to JSON file
        result_file = json_file.replace('.json', '_result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        if result.get('success'):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Subprocess worker failed: {e}", exc_info=True)
        # Write error result
        try:
            result_file = json_file.replace('.json', '_result.json')
            with open(result_file, 'w') as f:
                json.dump({"success": False, "error": str(e)}, f)
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

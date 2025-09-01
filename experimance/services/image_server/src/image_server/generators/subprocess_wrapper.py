#!/usr/bin/env python3
"""
Generic subprocess wrapper for running generators in separate processes with different GPU settings.

This wrapper allows any generator (image or audio) to be run in a subprocess with its own
CUDA_VISIBLE_DEVICES setting, solving multi-GPU allocation issues while maintaining
the same interface as the original generator.
"""

import asyncio
import json
import logging
import os
import pickle
import subprocess
import tempfile
import uuid
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
from pydantic import Field
import sys

from image_server.generators.generator import ImageGenerator, GeneratorCapabilities
from image_server.generators.audio.audio_generator import AudioGenerator, AudioGeneratorCapabilities
from image_server.generators.config import BaseGeneratorConfig
from image_server.generators.audio.audio_config import BaseAudioGeneratorConfig

logger = logging.getLogger(__name__)


class SubprocessGeneratorConfig(BaseGeneratorConfig):
    """Configuration for subprocess generator wrapper."""
    wrapped_generator_class: str
    wrapped_generator_config: Dict[str, Any]
    cuda_visible_devices: Optional[str] = None
    python_executable: str = Field(default_factory=lambda: sys.executable)
    timeout_seconds: int = 300
    max_retries: int = 3


class SubprocessAudioGeneratorConfig(BaseAudioGeneratorConfig):
    """Configuration for subprocess audio generator wrapper."""
    wrapped_generator_class: str
    wrapped_generator_config: Dict[str, Any]
    cuda_visible_devices: Optional[str] = None
    python_executable: str = Field(default_factory=lambda: sys.executable)
    timeout_seconds: int = 300
    max_retries: int = 3


class SubprocessImageGenerator(ImageGenerator):
    """Subprocess wrapper for image generators."""
    
    def __init__(self, config: SubprocessGeneratorConfig, **kwargs):
        """Initialize subprocess image generator wrapper."""
        self.subprocess_config = config
        self._wrapped_capabilities: Set[str] = set()
        
        # Initialize with empty capabilities initially - will be populated after subprocess setup
        super().__init__(config, **kwargs)
        
    def _configure(self, config: SubprocessGeneratorConfig, **kwargs):
        """Configure the subprocess wrapper."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="subprocess_generator_"))
        self.temp_dir.mkdir(exist_ok=True)
        
        # Fetch capabilities from wrapped generator
        self._fetch_wrapped_capabilities()
    
    def _fetch_wrapped_capabilities(self):
        """Fetch capabilities from the wrapped generator class."""
        try:
            # Create a subprocess to get capabilities without initializing the heavy generator
            script = f"""
import sys
sys.path.append('{os.getcwd()}')
from {self.subprocess_config.wrapped_generator_class.rsplit('.', 1)[0]} import {self.subprocess_config.wrapped_generator_class.rsplit('.', 1)[1]}
print(list(getattr({self.subprocess_config.wrapped_generator_class.rsplit('.', 1)[1]}, 'supported_capabilities', set())))
"""
            
            env = os.environ.copy()
            if self.subprocess_config.cuda_visible_devices is not None:
                env['CUDA_VISIBLE_DEVICES'] = self.subprocess_config.cuda_visible_devices
            
            result = subprocess.run(
                [self.subprocess_config.python_executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse the capabilities from subprocess output
                capabilities_str = result.stdout.strip()
                if capabilities_str:
                    import ast
                    self._wrapped_capabilities = set(ast.literal_eval(capabilities_str))
                    # Update our supported capabilities
                    self.supported_capabilities = self._wrapped_capabilities
                    logger.info(f"Subprocess wrapper loaded capabilities: {self._wrapped_capabilities}")
            else:
                logger.warning(f"Failed to fetch capabilities: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Could not fetch wrapped generator capabilities: {e}")
    
    @property
    def supported_capabilities(self) -> Set[str]:
        """Get supported capabilities from wrapped generator."""
        return self._wrapped_capabilities
    
    @supported_capabilities.setter
    def supported_capabilities(self, value: Set[str]):
        """Set supported capabilities."""
        self._wrapped_capabilities = value

    async def _generate_image_impl(self, prompt: str, **kwargs) -> str:
        """Generate image using subprocess wrapper."""
        return await self._run_subprocess_generation("generate_image", prompt, **kwargs)

    async def _run_subprocess_generation(self, method_name: str, *args, **kwargs) -> str:
        """Run generation in subprocess with proper GPU isolation."""
        request_id = str(uuid.uuid4())
        
        for attempt in range(self.subprocess_config.max_retries):
            try:
                # Create communication files
                input_file = self.temp_dir / f"input_{request_id}_{attempt}.pkl"
                output_file = self.temp_dir / f"output_{request_id}_{attempt}.pkl"
                
                # Prepare subprocess data
                subprocess_data = {
                    'generator_class': self.subprocess_config.wrapped_generator_class,
                    'generator_config': self.subprocess_config.wrapped_generator_config,
                    'output_dir': str(self.output_dir),
                    'method_name': method_name,
                    'args': args,
                    'kwargs': kwargs,
                    'output_file': str(output_file)
                }
                
                # Save input data
                with open(input_file, 'wb') as f:
                    pickle.dump(subprocess_data, f)
                
                # Prepare environment
                env = os.environ.copy()
                if self.subprocess_config.cuda_visible_devices is not None:
                    env['CUDA_VISIBLE_DEVICES'] = self.subprocess_config.cuda_visible_devices
                    logger.debug(f"Setting CUDA_VISIBLE_DEVICES={self.subprocess_config.cuda_visible_devices} for subprocess")
                
                # Create subprocess script
                script_content = self._get_subprocess_script()
                script_file = self.temp_dir / f"generator_script_{request_id}_{attempt}.py"
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                # Run subprocess
                logger.debug(f"Starting subprocess generation (attempt {attempt + 1}/{self.subprocess_config.max_retries})")
                process = await asyncio.create_subprocess_exec(
                    self.subprocess_config.python_executable,
                    str(script_file),
                    str(input_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.subprocess_config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Subprocess generation timed out after {self.subprocess_config.timeout_seconds}s")
                    process.kill()
                    await process.wait()
                    raise RuntimeError("Generation timed out")
                
                # Check process result
                if process.returncode != 0:
                    error_msg = f"Subprocess failed with return code {process.returncode}"
                    if stderr:
                        error_msg += f": {stderr.decode()}"
                    logger.error(error_msg)
                    if stdout:
                        logger.debug(f"Subprocess stdout: {stdout.decode()}")
                    
                    if attempt < self.subprocess_config.max_retries - 1:
                        logger.info(f"Retrying subprocess generation (attempt {attempt + 2}/{self.subprocess_config.max_retries})")
                        continue
                    else:
                        raise RuntimeError(error_msg)
                
                # Read result
                if not output_file.exists():
                    raise RuntimeError("Subprocess completed but output file not found")
                
                with open(output_file, 'rb') as f:
                    result = pickle.load(f)
                
                if 'error' in result:
                    raise RuntimeError(f"Subprocess generation failed: {result['error']}")
                
                logger.info(f"Subprocess generation completed successfully: {result['output_path']}")
                return result['output_path']
                
            except Exception as e:
                logger.error(f"Subprocess generation attempt {attempt + 1} failed: {e}")
                if attempt < self.subprocess_config.max_retries - 1:
                    continue
                else:
                    raise RuntimeError(f"All subprocess generation attempts failed: {e}")
            finally:
                # Clean up temporary files
                for temp_file in [input_file, output_file, script_file]:
                    if temp_file.exists():
                        temp_file.unlink()
        
        # This should never be reached due to the raise in the except block
        raise RuntimeError("Unexpected end of subprocess generation method")

    def _get_subprocess_script(self) -> str:
        """Get the subprocess script content."""
        return """#!/usr/bin/env python3
import sys
import pickle
import asyncio
import logging
from pathlib import Path

# Configure logging to capture subprocess output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        input_file = sys.argv[1]
        
        # Load input data
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        # Import and create generator
        module_name, class_name = data['generator_class'].rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        generator_class = getattr(module, class_name)
        
        # Create config object (assume it's a dict that can be passed as kwargs)
        config_data = data['generator_config']
        if hasattr(generator_class, '__annotations__') and 'config' in generator_class.__init__.__annotations__:
            # Try to determine config class from type hints
            config_type = generator_class.__init__.__annotations__['config']
            if hasattr(config_type, '__origin__'):  # Handle Optional[ConfigClass]
                config_type = config_type.__args__[0]
            config = config_type(**config_data)
        else:
            # Fall back to passing config as dict
            config = config_data
        
        # Create generator
        generator = generator_class(config, output_dir=data['output_dir'])
        
        # Start generator if needed
        await generator.start()
        
        try:
            # Call the requested method
            method = getattr(generator, data['method_name'])
            result = await method(*data['args'], **data['kwargs'])
            
            # Save result
            output_data = {'output_path': result}
            
        finally:
            # Stop generator
            await generator.stop()
        
        # Write result
        with open(data['output_file'], 'wb') as f:
            pickle.dump(output_data, f)
            
        logger.info(f"Subprocess generation completed: {result}")
        
    except Exception as e:
        logger.error(f"Subprocess generation failed: {e}")
        # Save error
        try:
            with open(data['output_file'], 'wb') as f:
                pickle.dump({'error': str(e)}, f)
        except:
            pass
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
"""

    async def stop(self):
        """Stop the generator and clean up."""
        await super().stop()
        
        # Clean up temp directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


class SubprocessAudioGenerator(AudioGenerator):
    """Subprocess wrapper for audio generators."""
    
    def __init__(self, config: SubprocessAudioGeneratorConfig, **kwargs):
        """Initialize subprocess audio generator wrapper."""
        self.subprocess_config = config
        self._wrapped_capabilities: Set[str] = set()
        
        # Initialize with empty capabilities initially - will be populated after subprocess setup
        super().__init__(config, **kwargs)
        
    def _configure(self, config: SubprocessGeneratorConfig, **kwargs):
        """Configure the subprocess wrapper."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="subprocess_audio_generator_"))
        self.temp_dir.mkdir(exist_ok=True)
        
        # Fetch capabilities from wrapped generator
        self._fetch_wrapped_capabilities()
    
    def _fetch_wrapped_capabilities(self):
        """Fetch capabilities from the wrapped generator class."""
        try:
            # Create a subprocess to get capabilities without initializing the heavy generator
            script = f"""
import sys
sys.path.append('{os.getcwd()}')
from {self.subprocess_config.wrapped_generator_class.rsplit('.', 1)[0]} import {self.subprocess_config.wrapped_generator_class.rsplit('.', 1)[1]}
print(list(getattr({self.subprocess_config.wrapped_generator_class.rsplit('.', 1)[1]}, 'supported_capabilities', set())))
"""
            
            env = os.environ.copy()
            if self.subprocess_config.cuda_visible_devices is not None:
                env['CUDA_VISIBLE_DEVICES'] = self.subprocess_config.cuda_visible_devices
            
            result = subprocess.run(
                [self.subprocess_config.python_executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse the capabilities from subprocess output
                capabilities_str = result.stdout.strip()
                if capabilities_str:
                    import ast
                    self._wrapped_capabilities = set(ast.literal_eval(capabilities_str))
                    # Update our supported capabilities
                    self.supported_capabilities = self._wrapped_capabilities
                    logger.info(f"Subprocess audio wrapper loaded capabilities: {self._wrapped_capabilities}")
            else:
                logger.warning(f"Failed to fetch audio capabilities: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Could not fetch wrapped audio generator capabilities: {e}")
    
    @property
    def supported_capabilities(self) -> Set[str]:
        """Get supported capabilities from wrapped generator."""
        return self._wrapped_capabilities
    
    @supported_capabilities.setter
    def supported_capabilities(self, value: Set[str]):
        """Set supported capabilities."""
        self._wrapped_capabilities = value

    async def _generate_audio_impl(self, prompt: str, **kwargs) -> str:
        """Generate audio using subprocess wrapper."""
        return await self._run_subprocess_generation("generate_audio", prompt, **kwargs)

    async def _run_subprocess_generation(self, method_name: str, *args, **kwargs) -> str:
        """Run generation in subprocess with proper GPU isolation."""
        request_id = str(uuid.uuid4())
        
        for attempt in range(self.subprocess_config.max_retries):
            try:
                # Create communication files
                input_file = self.temp_dir / f"input_{request_id}_{attempt}.pkl"
                output_file = self.temp_dir / f"output_{request_id}_{attempt}.pkl"
                
                # Prepare subprocess data
                subprocess_data = {
                    'generator_class': self.subprocess_config.wrapped_generator_class,
                    'generator_config': self.subprocess_config.wrapped_generator_config,
                    'output_dir': str(self.output_dir),
                    'method_name': method_name,
                    'args': args,
                    'kwargs': kwargs,
                    'output_file': str(output_file)
                }
                
                # Save input data
                with open(input_file, 'wb') as f:
                    pickle.dump(subprocess_data, f)
                
                # Prepare environment
                env = os.environ.copy()
                if self.subprocess_config.cuda_visible_devices is not None:
                    env['CUDA_VISIBLE_DEVICES'] = self.subprocess_config.cuda_visible_devices
                    logger.debug(f"Setting CUDA_VISIBLE_DEVICES={self.subprocess_config.cuda_visible_devices} for audio subprocess")
                
                # Create subprocess script
                script_content = self._get_subprocess_script()
                script_file = self.temp_dir / f"audio_generator_script_{request_id}_{attempt}.py"
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                # Run subprocess
                logger.debug(f"Starting subprocess audio generation (attempt {attempt + 1}/{self.subprocess_config.max_retries})")
                process = await asyncio.create_subprocess_exec(
                    self.subprocess_config.python_executable,
                    str(script_file),
                    str(input_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.subprocess_config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Subprocess audio generation timed out after {self.subprocess_config.timeout_seconds}s")
                    process.kill()
                    await process.wait()
                    raise RuntimeError("Audio generation timed out")
                
                # Check process result
                if process.returncode != 0:
                    error_msg = f"Audio subprocess failed with return code {process.returncode}"
                    if stderr:
                        error_msg += f": {stderr.decode()}"
                    logger.error(error_msg)
                    if stdout:
                        logger.debug(f"Audio subprocess stdout: {stdout.decode()}")
                    
                    if attempt < self.subprocess_config.max_retries - 1:
                        logger.info(f"Retrying subprocess audio generation (attempt {attempt + 2}/{self.subprocess_config.max_retries})")
                        continue
                    else:
                        raise RuntimeError(error_msg)
                
                # Read result
                if not output_file.exists():
                    raise RuntimeError("Audio subprocess completed but output file not found")
                
                with open(output_file, 'rb') as f:
                    result = pickle.load(f)
                
                if 'error' in result:
                    raise RuntimeError(f"Subprocess audio generation failed: {result['error']}")
                
                logger.info(f"Subprocess audio generation completed successfully: {result['output_path']}")
                return result['output_path']
                
            except Exception as e:
                logger.error(f"Subprocess audio generation attempt {attempt + 1} failed: {e}")
                if attempt < self.subprocess_config.max_retries - 1:
                    continue
                else:
                    raise RuntimeError(f"All subprocess audio generation attempts failed: {e}")
            finally:
                # Clean up temporary files
                for temp_file in [input_file, output_file, script_file]:
                    if temp_file.exists():
                        temp_file.unlink()
        
        # This should never be reached due to the raise in the except block
        raise RuntimeError("Unexpected end of subprocess audio generation method")

    def _get_subprocess_script(self) -> str:
        """Get the subprocess script content for audio generation."""
        return """#!/usr/bin/env python3
import sys
import pickle
import asyncio
import logging
from pathlib import Path

# Configure logging to capture subprocess output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        input_file = sys.argv[1]
        
        # Debug: Check CUDA environment
        import os
        import torch
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"CUDA device {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # Load input data
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        # Import and create generator
        module_name, class_name = data['generator_class'].rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        generator_class = getattr(module, class_name)
        
        # Create config object - use direct instantiation with the provided class
        config_data = data['generator_config']
        
        # Import the config class that should be used
        if data['generator_class'] == 'image_server.generators.audio.prompt2audio.Prompt2AudioGenerator':
            from image_server.generators.audio.prompt2audio import Prompt2AudioConfig
            config = Prompt2AudioConfig(**config_data)
        else:
            # For other generators, try to use type hints or fall back to dict
            if hasattr(generator_class, '__annotations__') and 'config' in generator_class.__init__.__annotations__:
                config_type = generator_class.__init__.__annotations__['config']
                if hasattr(config_type, '__origin__'):  # Handle Optional[ConfigClass]
                    config_type = config_type.__args__[0]
                config = config_type(**config_data)
            else:
                # Fall back to passing config as dict
                config = config_data
        
        # Create generator
        generator = generator_class(config, output_dir=data['output_dir'])
        
        # Start generator if needed
        await generator.start()
        
        try:
            # Call the requested method
            method = getattr(generator, data['method_name'])
            result = await method(*data['args'], **data['kwargs'])
            
            # Save result
            output_data = {'output_path': result}
            
        finally:
            # Stop generator
            await generator.stop()
        
        # Write result
        with open(data['output_file'], 'wb') as f:
            pickle.dump(output_data, f)
            
        logger.info(f"Subprocess audio generation completed: {result}")
        
    except Exception as e:
        logger.error(f"Subprocess audio generation failed: {e}")
        # Save error
        try:
            with open(data['output_file'], 'wb') as f:
                pickle.dump({'error': str(e)}, f)
        except:
            pass
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
"""

    async def stop(self):
        """Stop the generator and clean up."""
        await super().stop()
        
        # Clean up temp directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


def create_subprocess_wrapper(
    generator_class: str,
    generator_config: Dict[str, Any],
    generator_type: str = "auto",
    cuda_visible_devices: Optional[str] = None,
    output_dir: Optional[str] = None,
    **wrapper_kwargs
) -> Union[SubprocessImageGenerator, SubprocessAudioGenerator]:
    """Factory function to create subprocess wrapper for any generator.
    
    Args:
        generator_class: Fully qualified class name (e.g., "image_server.generators.fal.FalGenerator")
        generator_config: Configuration dict for the wrapped generator
        generator_type: Type of generator ("image", "audio", or "auto" to detect)
        cuda_visible_devices: GPU devices to make visible (e.g., "0" or "1,2")
        output_dir: Output directory for generated files
        **wrapper_kwargs: Additional arguments for the wrapper (timeout, retries, etc.)
    
    Returns:
        Configured subprocess wrapper instance
    """
    # Auto-detect generator type if needed
    if generator_type == "auto":
        try:
            module_name = generator_class.rsplit('.', 1)[0]
            if 'audio' in module_name.lower():
                generator_type = "audio"
            else:
                generator_type = "image"
        except:
            generator_type = "image"  # Default to image
    
    # Create appropriate config and wrapper
    if generator_type == "audio":
        subprocess_config = SubprocessAudioGeneratorConfig(
            wrapped_generator_class=generator_class,
            wrapped_generator_config=generator_config,
            cuda_visible_devices=cuda_visible_devices,
            strategy="subprocess",  # Add required strategy field
            **wrapper_kwargs
        )
        wrapper = SubprocessAudioGenerator(subprocess_config)
    else:
        subprocess_config = SubprocessGeneratorConfig(
            wrapped_generator_class=generator_class,
            wrapped_generator_config=generator_config,
            cuda_visible_devices=cuda_visible_devices,
            strategy="subprocess",  # Add required strategy field
            **wrapper_kwargs
        )
        wrapper = SubprocessImageGenerator(subprocess_config)
    
    # Set output directory if provided
    if output_dir:
        wrapper.output_dir = Path(output_dir)
        wrapper.output_dir.mkdir(parents=True, exist_ok=True)
    
    return wrapper

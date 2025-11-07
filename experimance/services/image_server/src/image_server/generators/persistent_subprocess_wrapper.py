#!/usr/bin/env python3
"""
Persistent subprocess wrapper for generators with model persistence.

This wrapper starts a long-running worker process that keeps models loaded
in memory between requests, dramatically improving performance for subsequent
generations.

Uses subprocess instead of multiprocessing for better CUDA isolation.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import Field
import sys

if sys.version_info >= (3, 8):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from asyncio.subprocess import Process

from image_server.generators.audio.audio_generator import AudioGenerator
from image_server.generators.audio.audio_config import BaseAudioGeneratorConfig

logger = logging.getLogger(__name__)


class PersistentSubprocessAudioGeneratorConfig(BaseAudioGeneratorConfig):
    """Configuration for persistent subprocess audio generator wrapper."""
    wrapped_generator_class: str
    wrapped_generator_config: Dict[str, Any]
    cuda_visible_devices: Optional[str] = None
    python_executable: str = Field(default_factory=lambda: sys.executable)
    timeout_seconds: int = 300
    max_retries: int = 3
    startup_timeout_seconds: int = 120  # Time to wait for worker to load models


class PersistentSubprocessAudioGenerator(AudioGenerator):
    """Subprocess wrapper for audio generators with persistent worker."""
    
    def __init__(self, config: PersistentSubprocessAudioGeneratorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.subprocess_config = config
        self.worker_process: Optional["Process"] = None
        self.temp_dir: Optional[Path] = None
        self.request_file: Optional[Path] = None
        self.response_file: Optional[Path] = None
        self.shutdown_file: Optional[Path] = None

    async def start(self):
        """Start the persistent worker process."""
        await super().start()
        
        # Use subprocess for better CUDA isolation
        import tempfile
        import json
        
        # Create temporary directory for communication files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="persistent_audio_worker_"))
        self.request_file = self.temp_dir / "request.json"
        self.response_file = self.temp_dir / "response.json"
        self.shutdown_file = self.temp_dir / "shutdown.flag"
        
        logger.info(f"Starting persistent audio worker with CUDA_VISIBLE_DEVICES={self.subprocess_config.cuda_visible_devices}")
        
        # Create worker script
        worker_script = self._create_worker_script()
        script_file = self.temp_dir / "worker.py"
        with open(script_file, 'w') as f:
            f.write(worker_script)
        
        # Prepare environment
        env = os.environ.copy()
        if self.subprocess_config.cuda_visible_devices:
            env['CUDA_VISIBLE_DEVICES'] = self.subprocess_config.cuda_visible_devices
        
        # Start the subprocess worker
        self.worker_process = await asyncio.create_subprocess_exec(
            self.subprocess_config.python_executable,
            str(script_file),
            str(self.temp_dir),
            self.subprocess_config.wrapped_generator_class,
            json.dumps(self.subprocess_config.wrapped_generator_config, default=str),
            str(self.output_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Wait for worker to signal it's ready
        try:
            await asyncio.wait_for(
                self._wait_for_worker_ready(),
                timeout=self.subprocess_config.startup_timeout_seconds
            )
            logger.info("Persistent audio worker started and models loaded")
        except asyncio.TimeoutError:
            raise RuntimeError(f"Worker failed to start within {self.subprocess_config.startup_timeout_seconds} seconds")

    async def _wait_for_worker_ready(self):
        """Wait for worker to create ready signal."""
        if not self.temp_dir:
            raise RuntimeError("Temp directory not initialized")
            
        ready_file = self.temp_dir / "ready.flag"
        while not ready_file.exists():
            await asyncio.sleep(0.1)
        # Read any startup error
        error_file = self.temp_dir / "error.txt"
        if error_file.exists():
            with open(error_file, 'r') as f:
                error = f.read().strip()
            raise RuntimeError(f"Worker startup error: {error}")

    def _create_worker_script(self) -> str:
        """Create the persistent worker script."""
        return '''#!/usr/bin/env python3
import os
import sys
import json
import time
import traceback
from pathlib import Path

def main():
    # Get arguments
    temp_dir = Path(sys.argv[1])
    generator_class_str = sys.argv[2]
    generator_config_json = sys.argv[3]
    output_dir = sys.argv[4]
    
    generator_config = json.loads(generator_config_json)
    
    # Communication files
    request_file = temp_dir / "request.json"
    response_file = temp_dir / "response.json"
    shutdown_file = temp_dir / "shutdown.flag"
    ready_file = temp_dir / "ready.flag"
    error_file = temp_dir / "error.txt"
    
    generator = None
    
    try:
        # Check CUDA environment first
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"Worker: CUDA_VISIBLE_DEVICES = {cuda_visible}")
        
        # Import torch early to check CUDA setup
        import torch
        if torch.cuda.is_available():
            print(f"Worker: CUDA available, device count: {torch.cuda.device_count()}")
            print(f"Worker: Current device: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                print(f"Worker: Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("Worker: CUDA not available")
        
        # Import and create generator
        print(f"Worker: Loading generator {generator_class_str}")
        module_name, class_name = generator_class_str.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        generator_class = getattr(module, class_name)
        
        # Create config
        if generator_class_str == 'image_server.generators.audio.prompt2audio.Prompt2AudioGenerator':
            from image_server.generators.audio.prompt2audio import Prompt2AudioConfig
            config = Prompt2AudioConfig(**generator_config)
        else:
            # Generic fallback
            config = generator_config
        
        # Create generator
        generator = generator_class(config, output_dir=output_dir)
        
        # Start generator (loads models)
        print("Worker: Starting generator and loading models...")
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def start_generator():
            await generator.start()
            
        loop.run_until_complete(start_generator())
        print("Worker: Generator started, models loaded")
        
        # Check memory usage after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"Worker: GPU {i} - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        
        # Signal ready
        ready_file.touch()
        
        # Process requests
        while not shutdown_file.exists():
            if request_file.exists():
                try:
                    # Read request
                    with open(request_file, 'r') as f:
                        request = json.load(f)
                    
                    # Remove request file
                    request_file.unlink()
                    
                    method_name = request.get('method')
                    args = request.get('args', [])
                    kwargs = request.get('kwargs', {})
                    
                    print(f"Worker: Processing {method_name} with prompt '{args[0] if args else 'N/A'}'")
                    
                    # Execute method
                    method = getattr(generator, method_name)
                    
                    async def run_method():
                        result = await method(*args, **kwargs)
                        
                        # Try to get generation metadata if available
                        metadata = {}
                            
                        if hasattr(generator, 'get_last_generation_metadata'):
                            try:
                                metadata = generator.get_last_generation_metadata()
                            except Exception as e:
                                print(f"Worker: Error getting metadata: {e}", file=sys.stderr)
                                metadata = {}
                        
                        return result, metadata
                        
                    result, metadata = loop.run_until_complete(run_method())
                    
                    # Write response with metadata
                    response = {
                        'type': 'success', 
                        'result': str(result),
                        'metadata': metadata
                    }
                    with open(response_file, 'w') as f:
                        json.dump(response, f, default=str)
                    
                    print(f"Worker: Completed request, result: {result}, metadata: {metadata}")
                    
                except Exception as e:
                    print(f"Worker: Error processing request: {e}")
                    traceback.print_exc()
                    response = {'type': 'error', 'error': str(e)}
                    with open(response_file, 'w') as f:
                        json.dump(response, f)
            else:
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Worker: Startup error: {e}")
        traceback.print_exc()
        with open(error_file, 'w') as f:
            f.write(str(e))
        return
    
    finally:
        print("Worker: Shutting down")
        if generator:
            try:
                loop.run_until_complete(generator.stop())
            except:
                pass

if __name__ == "__main__":
    main()
'''

    async def stop(self):
        """Stop the persistent worker process."""
        if self.worker_process and not self.worker_process.returncode:
            logger.info("Stopping persistent audio worker...")
            
            # Signal shutdown
            if self.shutdown_file:
                self.shutdown_file.touch()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(self.worker_process.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("Worker didn't shutdown gracefully, terminating")
                self.worker_process.terminate()
                try:
                    await asyncio.wait_for(self.worker_process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.error("Worker didn't terminate, killing")
                    self.worker_process.kill()
            
            logger.info("Persistent audio worker stopped")
        
        # Cleanup temp directory
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Could not cleanup temp directory: {e}")

    async def _generate_audio_impl(self, prompt: str, **kwargs) -> str:
        """Generate audio using persistent worker."""
        if not self.worker_process or self.worker_process.returncode is not None:
            raise RuntimeError("Worker process is not running")
        
        if not self.request_file or not self.response_file:
            raise RuntimeError("Communication files not initialized")
        
        # Prepare request
        request = {
            'method': 'generate_audio',
            'args': [prompt],
            'kwargs': kwargs
        }
        
        # Write request
        with open(self.request_file, 'w') as f:
            import json
            json.dump(request, f)
        
        # Wait for response with timeout handling and subprocess restart
        try:
            response = await self._wait_for_response()
        except asyncio.TimeoutError as e:
            logger.error(f"Audio generation timed out: {e}")
            logger.info("Killing and restarting subprocess due to timeout")
            
            # Kill the stuck subprocess
            await self._restart_worker_after_timeout()
            
            # Re-raise the original timeout error since this request failed
            raise RuntimeError("Audio generation timed out and subprocess was restarted") from e
        
        if response.get('type') == 'error':
            raise RuntimeError(f"Worker error: {response.get('error')}")
        
        result = response.get('result')
        if result is None:
            raise RuntimeError("Worker returned no result")
        
        # Store metadata for potential retrieval
        self._last_generation_metadata = response.get('metadata', {})
        
        return str(result)

    async def _restart_worker_after_timeout(self):
        """Kill and restart the worker subprocess after a timeout."""
        try:
            # Force kill the stuck subprocess
            if self.worker_process and self.worker_process.returncode is None:
                logger.info("Force killing stuck subprocess")
                self.worker_process.kill()
                try:
                    await asyncio.wait_for(self.worker_process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning("Process didn't die after kill signal")
            
            # Clean up existing state
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    logger.warning(f"Could not cleanup temp directory during restart: {e}")
            
            # Restart the worker
            await self.start()
            logger.info("Successfully restarted audio generation subprocess")
            
        except Exception as restart_error:
            logger.error(f"Failed to restart subprocess: {restart_error}")
            raise RuntimeError(f"Failed to restart subprocess after timeout: {restart_error}") from restart_error
    
    def get_last_generation_metadata(self) -> dict:
        """Get metadata from the last generation."""
        metadata = getattr(self, '_last_generation_metadata', {})
        logger.debug(f"PersistentSubprocessWrapper returning metadata: {metadata}")
        return metadata

    async def _wait_for_response(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Wait for worker to write response file."""
        if timeout is None:
            timeout = self.subprocess_config.timeout_seconds
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.response_file and self.response_file.exists():
                try:
                    with open(self.response_file, 'r') as f:
                        import json
                        response = json.load(f)
                    
                    # Remove response file
                    self.response_file.unlink()
                    return response
                except (json.JSONDecodeError, FileNotFoundError):
                    # File might be being written, wait a bit more
                    await asyncio.sleep(0.1)
                    continue
            
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"Worker response timeout after {timeout} seconds")


def create_persistent_subprocess_wrapper(
    generator_class: str,
    generator_config: Dict[str, Any],
    generator_type: str,
    cuda_visible_devices: Optional[str] = None,
    timeout_seconds: int = 300,
    max_retries: int = 3,
    startup_timeout_seconds: int = 120,
    output_dir: Optional[Path] = None,
    **kwargs
) -> PersistentSubprocessAudioGenerator:
    """Factory function to create a persistent subprocess audio generator wrapper."""
    
    if generator_type != "audio":
        raise ValueError(f"Unsupported generator type: {generator_type}")
    
    if output_dir is None:
        output_dir = Path("media/images/generated/audio")
    
    # Create subprocess wrapper config
    wrapper_config = PersistentSubprocessAudioGeneratorConfig(
        strategy=generator_config.get('strategy', 'prompt2audio'),
        wrapped_generator_class=generator_class,
        wrapped_generator_config=generator_config,
        cuda_visible_devices=cuda_visible_devices,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        startup_timeout_seconds=startup_timeout_seconds
    )
    
    return PersistentSubprocessAudioGenerator(wrapper_config, output_dir=output_dir, **kwargs)

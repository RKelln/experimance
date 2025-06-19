#!/usr/bin/env python3
"""
End-to-end integration test for the complete Experimance pipeline.

This test uses actual services (not mocks) to verify the full flow:
Core → Image Server (mock mode) → Core → Display

Tests the complete DISPLAY_MEDIA message flow with real ZMQ communication.
"""

import asyncio
import json
import logging
import os
import pytest
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import zmq
import zmq.asyncio

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.schemas import DisplayMedia, ImageReady, RenderRequest
from experimance_common.zmq.zmq_utils import MessageType, deserialize_message


logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages starting and stopping real services for integration testing."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.processes: Dict[str, subprocess.Popen] = {}
        self.service_configs: Dict[str, Path] = {}
        
    async def start_image_server(self) -> None:
        """Start the image server in mock mode with integration test config."""
        config_path = self.workspace_root / "services/image_server/config_integration_test.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"Integration test config not found: {config_path}")
            
        self.service_configs["image_server"] = config_path
        
        cmd = [
            "uv", "run", "-m", "image_server.image_service",
            "--config", str(config_path)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.workspace_root / "services/image_server",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes["image_server"] = process
        logger.info(f"Started image server with PID {process.pid}")
        
        # Give it time to start up
        await asyncio.sleep(2.0)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Image server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
    
    async def start_display(self) -> None:
        """Start the display service."""
        config_path = self.workspace_root / "services/display/config.toml"
        
        cmd = [
            "uv", "run", "-m", "experimance_display.display_service",
            "--config", str(config_path) if config_path.exists() else None
        ]
        
        # Remove None values
        cmd = [arg for arg in cmd if arg is not None]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.workspace_root / "services/display",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes["display"] = process
        logger.info(f"Started display service with PID {process.pid}")
        
        # Give it time to start up
        await asyncio.sleep(2.0)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Display service failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
    
    async def start_core(self, temp_config_path: Path) -> None:
        """Start the core service with test configuration."""
        cmd = [
            "uv", "run", "-m", "experimance_core.experimance_core",
            "--config", str(temp_config_path)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.workspace_root / "services/core",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes["core"] = process
        logger.info(f"Started core service with PID {process.pid}")
        
        # Give it time to start up
        await asyncio.sleep(3.0)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Core service failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
    
    async def stop_all(self) -> None:
        """Stop all started services."""
        for service_name, process in self.processes.items():
            if process.poll() is None:  # Still running
                logger.info(f"Stopping {service_name} (PID {process.pid})")
                process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {service_name}")
                    process.kill()
                    process.wait()
        
        self.processes.clear()
    
    def get_service_logs(self, service_name: str) -> tuple[str, str]:
        """Get stdout and stderr from a service."""
        if service_name not in self.processes:
            return "", ""
        
        process = self.processes[service_name]
        try:
            stdout, stderr = process.communicate(timeout=1.0)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            return "Service still running", ""


class MessageCapture:
    """Captures and analyzes messages from the events channel."""
    
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self.subscriber = None
        self.captured_messages: List[Dict] = []
        self.running = False
    
    async def start_capturing(self) -> None:
        """Start capturing messages from the events channel."""
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://localhost:{DEFAULT_PORTS['events']}")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        
        self.running = True
        logger.info("Started message capture")
    
    async def capture_loop(self) -> None:
        """Main capture loop - run this in a background task."""
        while self.running and self.subscriber:
            try:
                # Non-blocking receive with timeout
                message_parts = await asyncio.wait_for(
                    self.subscriber.recv_multipart(),
                    timeout=0.1
                )
                
                if len(message_parts) >= 2:
                    message_type = message_parts[0].decode('utf-8')
                    message_data = json.loads(message_parts[1].decode('utf-8'))
                    
                    message_with_meta = {
                        "type": message_type,
                        "data": message_data,
                        "timestamp": time.time()
                    }
                    
                    self.captured_messages.append(message_with_meta)
                    logger.debug(f"Captured: {message_type}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error capturing message: {e}")
                break
    
    async def stop_capturing(self) -> None:
        """Stop capturing messages."""
        self.running = False
        if self.subscriber:
            self.subscriber.close()
        self.context.term()
        logger.info("Stopped message capture")
    
    def get_messages_by_type(self, message_type: str) -> List[Dict]:
        """Get all captured messages of a specific type."""
        return [msg for msg in self.captured_messages if msg["type"] == message_type]
    
    def wait_for_message_type(self, message_type: str, timeout: float = 10.0) -> Optional[Dict]:
        """Wait for a specific message type to appear."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            messages = self.get_messages_by_type(message_type)
            if messages:
                return messages[-1]  # Return the most recent one
            time.sleep(0.1)
        
        return None


@pytest.fixture
def workspace_root():
    """Get the workspace root directory."""
    # Navigate up from utils/tests to find the workspace root
    current_dir = Path(__file__).parent
    while current_dir.parent != current_dir:
        if (current_dir / "pyproject.toml").exists() and (current_dir / "services").exists():
            return current_dir
        current_dir = current_dir.parent
    
    raise RuntimeError("Could not find workspace root")


@pytest.fixture
async def service_manager(workspace_root):
    """Service manager fixture with cleanup."""
    manager = ServiceManager(workspace_root)
    yield manager
    await manager.stop_all()


@pytest.fixture
async def message_capture():
    """Message capture fixture with cleanup."""
    capture = MessageCapture()
    yield capture
    await capture.stop_capturing()


def create_test_core_config(temp_dir: Path) -> Path:
    """Create a minimal core service configuration for testing."""
    config_content = """
[core]
idle_timeout = 60
wilderness_reset = 300
sensor_gain = 1.0
log_level = "INFO"

[state]
initial_era = "modern"
initial_biome = "coastal"
accumulated_interaction_score = 0.0

[zmq]
events_port = 5555
"""
    
    config_path = temp_dir / "test_core_config.toml"
    config_path.write_text(config_content)
    return config_path


@pytest.mark.asyncio
async def test_full_pipeline_integration(service_manager: ServiceManager, message_capture: MessageCapture, workspace_root: Path):
    """
    Test the complete pipeline: Core → Image Server → Core → Display
    
    This test verifies:
    1. Core service can publish RenderRequest messages
    2. Image server receives and processes RenderRequest
    3. Image server publishes ImageReady message
    4. Core receives ImageReady and publishes DisplayMedia
    5. Display service receives and processes DisplayMedia
    """
    
    # Create temporary config for core
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        core_config_path = create_test_core_config(temp_dir_path)
        
        # Start message capture first
        await message_capture.start_capturing()
        capture_task = asyncio.create_task(message_capture.capture_loop())
        
        try:
            # Start services in order
            logger.info("Starting image server...")
            await service_manager.start_image_server()
            
            logger.info("Starting display service...")
            await service_manager.start_display()
            
            logger.info("Starting core service...")
            await service_manager.start_core(core_config_path)
            
            # Wait for services to fully initialize
            await asyncio.sleep(5.0)
            
            # Verify all services are still running
            for service_name, process in service_manager.processes.items():
                assert process.poll() is None, f"{service_name} service crashed"
            
            # Now trigger the pipeline by simulating an era change in core
            # We'll send a depth update to trigger image generation
            context = zmq.asyncio.Context()
            
            # Publish a depth update to trigger image generation
            depth_publisher = context.socket(zmq.PUB)
            depth_publisher.bind(f"tcp://*:{DEFAULT_PORTS['depth']}")
            await asyncio.sleep(1.0)  # Let binding complete
            
            depth_data = {
                "hand_detected": False,
                "depth_map_png": None  # Core will use cached or generate new
            }
            
            await depth_publisher.send_multipart([
                b"depth_update",
                json.dumps(depth_data).encode('utf-8')
            ])
            
            logger.info("Sent depth update to trigger pipeline")
            
            # Wait for and verify the message flow
            # 1. Should see RenderRequest
            render_request_msg = message_capture.wait_for_message_type("RenderRequest", timeout=15.0)
            assert render_request_msg is not None, "No RenderRequest message captured"
            
            render_request_data = render_request_msg["data"]
            assert "era" in render_request_data
            assert "biome" in render_request_data
            assert "prompt" in render_request_data
            
            logger.info(f"✓ Captured RenderRequest: era={render_request_data['era']}, biome={render_request_data['biome']}")
            
            # 2. Should see ImageReady
            image_ready_msg = message_capture.wait_for_message_type("ImageReady", timeout=20.0)
            assert image_ready_msg is not None, "No ImageReady message captured"
            
            image_ready_data = image_ready_msg["data"]
            assert "image_id" in image_ready_data
            assert "era" in image_ready_data
            assert "biome" in image_ready_data
            
            # Should have image data in one of the expected formats
            has_image_data = any(key in image_ready_data for key in ["uri", "image_data", "file_path"])
            assert has_image_data, "ImageReady message missing image data"
            
            logger.info(f"✓ Captured ImageReady: image_id={image_ready_data['image_id']}")
            
            # 3. Should see DisplayMedia
            display_media_msg = message_capture.wait_for_message_type("DisplayMedia", timeout=10.0)
            assert display_media_msg is not None, "No DisplayMedia message captured"
            
            display_media_data = display_media_msg["data"]
            assert "content_type" in display_media_data
            assert display_media_data["content_type"] == "image"
            assert "era" in display_media_data
            assert "biome" in display_media_data
            
            # Should have image data in some format
            has_image_data = any(key in display_media_data for key in ["uri", "image_data", "file_path"])
            assert has_image_data, "DisplayMedia message missing image data"
            
            logger.info(f"✓ Captured DisplayMedia: content_type={display_media_data['content_type']}")
            
            # Verify the era and biome are consistent through the pipeline
            assert render_request_data["era"] == image_ready_data["era"]
            assert render_request_data["biome"] == image_ready_data["biome"]
            assert image_ready_data["era"] == display_media_data["era"]
            assert image_ready_data["biome"] == display_media_data["biome"]
            
            logger.info("✓ Era and biome consistency verified through pipeline")
            
            # Cleanup publisher
            depth_publisher.close()
            context.term()
            
        finally:
            # Stop capture task
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_image_server_mock_configuration(workspace_root: Path):
    """Test that the image server can be configured for mock mode with existing images."""
    config_path = workspace_root / "services/image_server/config_integration_test.toml"
    assert config_path.exists(), f"Integration test config not found: {config_path}"
    
    # Read and verify the config
    import toml
    config = toml.load(config_path)
    
    assert "generator" in config
    assert config["generator"]["default_strategy"] == "mock"
    
    mock_config = config["mock"]
    assert "use_existing_images" in mock_config
    assert mock_config["use_existing_images"] is True
    
    # Verify the images directory exists and has content
    images_dir = Path(mock_config["existing_images_dir"])
    if not images_dir.is_absolute():
        images_dir = workspace_root / images_dir
    
    assert images_dir.exists(), f"Images directory not found: {images_dir}"
    
    # Should have some images
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    assert len(image_files) > 0, f"No images found in {images_dir}"
    
    logger.info(f"✓ Found {len(image_files)} images in {images_dir}")


@pytest.mark.asyncio
async def test_service_startup_health(service_manager: ServiceManager, workspace_root: Path):
    """Test that all services can start up without errors."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        core_config_path = create_test_core_config(temp_dir_path)
        
        # Start each service and verify it doesn't crash
        await service_manager.start_image_server()
        await asyncio.sleep(2.0)
        assert service_manager.processes["image_server"].poll() is None
        
        await service_manager.start_display()
        await asyncio.sleep(2.0)
        assert service_manager.processes["display"].poll() is None
        
        await service_manager.start_core(core_config_path)
        await asyncio.sleep(3.0)
        assert service_manager.processes["core"].poll() is None
        
        # All services should still be running
        for service_name, process in service_manager.processes.items():
            assert process.poll() is None, f"{service_name} service crashed"
        
        logger.info("✓ All services started successfully and remain stable")


if __name__ == "__main__":
    # Allow running this test directly
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the tests
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])

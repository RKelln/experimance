#!/usr/bin/env python3
"""
Unit tests for the Sohkepayin Core Service.

Tests the core service functionality including:
- State management
- Message handling
- Configuration loading
- Service lifecycle
"""

import pytest
import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, ANY

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sohkepayin_core.sohkepayin_core import SohkepayinCoreService, CoreState, ActiveRequest
from sohkepayin_core.config import SohkepayinCoreConfig, LLMConfig, ImagePrompt
from sohkepayin_core.tiler import TileSpec
from experimance_common.schemas import StoryHeard
from experimance_common.schemas_base import ImageReady


class TestSohkepayinCoreService:
    """Test the SohkepayinCoreService class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        # Use default config and override just what we need
        return SohkepayinCoreConfig(
            service_name="test_sohkepayin_core",
            heartbeat_interval=1.0,
            llm=LLMConfig(
                provider="mock",
                model="test-model"
            )
        )
    
    @pytest.fixture
    def mock_service(self, mock_config):
        """Create a mock service instance."""
        with patch('sohkepayin_core.sohkepayin_core.get_llm_provider') as mock_llm_factory:
            mock_llm = Mock()
            mock_llm_factory.return_value = mock_llm
            
            service = SohkepayinCoreService(mock_config)
            service.zmq_service = Mock()
            service.zmq_service.start = AsyncMock()
            service.zmq_service.stop = AsyncMock()
            service.zmq_service.publish = AsyncMock()
            service.zmq_service.send_work_to_worker = AsyncMock()
            
            return service
    
    def test_service_initialization(self, mock_config):
        """Test service initialization."""
        with patch('sohkepayin_core.sohkepayin_core.get_llm_provider') as mock_llm_factory:
            mock_llm = Mock()
            mock_llm_factory.return_value = mock_llm
            
            service = SohkepayinCoreService(mock_config)
            
            assert service.config == mock_config
            assert service.core_state == CoreState.IDLE
            assert service.current_request is None
            assert service.pending_image_requests == {}
            assert service.llm == mock_llm
            assert service.prompt_builder is not None
            assert service.tiler is not None
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, mock_service):
        """Test state transitions."""
        service = mock_service
        
        # Test transition to listening
        await service._transition_to_state(CoreState.LISTENING)
        assert service.core_state == CoreState.LISTENING
        
        # Test transition to base image
        await service._transition_to_state(CoreState.BASE_IMAGE)
        assert service.core_state == CoreState.BASE_IMAGE
        
        # Test transition to tiles
        await service._transition_to_state(CoreState.TILES)
        assert service.core_state == CoreState.TILES
        
        # Test returning to listening clears requests
        service.current_request = ActiveRequest(
            request_id="test",
            base_prompt=Mock()
        )
        service.pending_image_requests["test"] = "base"
        
        await service._transition_to_state(CoreState.LISTENING)
        assert service.core_state == CoreState.LISTENING
        assert service.current_request is None
        assert service.pending_image_requests == {}
    
    @pytest.mark.asyncio
    async def test_handle_story_heard(self, mock_service):
        """Test handling StoryHeard messages."""
        service = mock_service
        
        # Mock the prompt builder
        mock_prompt = ImagePrompt(
            prompt="test panorama prompt",
            negative_prompt="test negative"
        )
        service.prompt_builder.build_prompt = AsyncMock(return_value=mock_prompt)
        
        # Create test story
        story = StoryHeard(content="A beautiful mountain landscape with snow-capped peaks.")
        
        # Handle the story
        await service._handle_story_heard("test_topic", story.model_dump())
        
        # Verify state transition
        assert service.core_state == CoreState.BASE_IMAGE
        
        # Verify request was created
        assert service.current_request is not None
        assert service.current_request.base_prompt == mock_prompt
        assert len(service.current_request.tiles) > 0
        
        # Verify clear display was sent
        service.zmq_service.publish.assert_called()
        
        # Verify base image request was sent
        service.zmq_service.send_work_to_worker.assert_called_with("image_server", ANY)
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_base(self, mock_service):
        """Test handling ImageReady for base image."""
        service = mock_service
        
        # Set up active request
        mock_prompt = ImagePrompt(prompt="test prompt", negative_prompt="test negative")
        mock_tile = TileSpec(x=0, y=0, width=800, height=600, tile_index=0, total_tiles=1)
        service.current_request = ActiveRequest(
            request_id="test_req",
            base_prompt=mock_prompt,
            tiles=[mock_tile]  # At least one tile
        )
        service.pending_image_requests["test_req_base"] = "base"
        
        # Create ImageReady message
        image_ready = ImageReady(
            request_id="test_req_base",
            uri="file:///test/image.png"
        )
        
        # Handle the message
        await service._handle_image_ready(image_ready)
        
        # Verify state transition
        assert service.core_state == CoreState.TILES
        
        # Verify base image marked as ready
        assert service.current_request.base_image_ready is True
        
        # Verify display message sent
        service.zmq_service.publish.assert_called()
        
        # Verify tile requests sent
        assert service.zmq_service.send_work_to_worker.call_count >= 1  # At least tiles were sent
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_tile(self, mock_service):
        """Test handling ImageReady for tile image."""
        service = mock_service
        
        # Set up active request with tiles
        mock_tile = TileSpec(
            x=100, y=200, width=800, height=600,
            tile_index=0, total_tiles=1
        )
        mock_tile.final_x = 100
        mock_tile.final_y = 200
        
        service.current_request = ActiveRequest(
            request_id="test_req",
            base_prompt=Mock(),
            tiles=[mock_tile],
            total_tiles=1
        )
        service.pending_image_requests["test_req_tile_0"] = "tile_0"
        
        # Create ImageReady message
        image_ready = ImageReady(
            request_id="test_req_tile_0",
            uri="file:///test/tile.png"
        )
        
        # Handle the message
        current_request = service.current_request  # Save reference before state change
        await service._handle_image_ready(image_ready)
        
        # Verify tile marked as completed (check saved reference)
        assert 0 in current_request.completed_tiles
        assert current_request.completed_tiles[0] == "file:///test/tile.png"
        
        # Verify display message sent with position
        service.zmq_service.publish.assert_called()
        
        # Since this completes all tiles, should return to listening and clear request
        assert service.core_state == CoreState.LISTENING
    
    @pytest.mark.asyncio
    async def test_handle_unknown_image_ready(self, mock_service):
        """Test handling ImageReady for unknown request."""
        service = mock_service
        
        # Create ImageReady for unknown request
        image_ready = ImageReady(
            request_id="unknown_request",
            uri="file:///test/image.png"
        )
        
        # Should not crash
        await service._handle_image_ready(image_ready)
        
        # Should remain in current state
        assert service.core_state == CoreState.IDLE


class TestActiveRequest:
    """Test the ActiveRequest dataclass."""
    
    def test_active_request_creation(self):
        """Test creating an ActiveRequest."""
        mock_prompt = Mock()
        
        request = ActiveRequest(
            request_id="test_123",
            base_prompt=mock_prompt
        )
        
        assert request.request_id == "test_123"
        assert request.base_prompt == mock_prompt
        assert request.tiles == []
        assert request.base_image_ready is False
        assert request.completed_tiles == {}
        assert request.total_tiles == 0
    
    def test_active_request_with_tiles(self):
        """Test ActiveRequest with tiles."""
        mock_prompt = Mock()
        mock_tiles = [
            TileSpec(x=0, y=0, width=100, height=100, tile_index=0, total_tiles=2),
            TileSpec(x=50, y=0, width=100, height=100, tile_index=1, total_tiles=2)
        ]
        
        request = ActiveRequest(
            request_id="test_123",
            base_prompt=mock_prompt,
            tiles=mock_tiles,
            total_tiles=2
        )
        
        assert len(request.tiles) == 2
        assert request.total_tiles == 2


class TestCoreState:
    """Test the CoreState enum."""
    
    def test_core_state_values(self):
        """Test CoreState enum values."""
        assert CoreState.IDLE.value == "idle"
        assert CoreState.LISTENING.value == "listening"
        assert CoreState.BASE_IMAGE.value == "base_image"
        assert CoreState.TILES.value == "tiles"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

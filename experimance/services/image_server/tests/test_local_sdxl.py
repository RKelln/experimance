import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_server.generators.local.sdxl_generator import SDXLLightningGenerator, SDXLLightningConfig, LocalSDXLGenerator
from image_server.generators.config import BaseGeneratorConfig


class DummyResult:
    def __init__(self):
        # create a 1x1 white image lazily to avoid PIL import issues inside dummy pipeline
        from PIL import Image
        self.images = [Image.new("RGB", (1, 1), color=(255, 255, 255))]


class MockComponent:
    def to(self, *args, **kwargs):
        return self
    def set_attn_processor(self, *args, **kwargs):
        pass


def dummy_pipeline_factory(*args, **kwargs):
    # Returns a callable emulating the diffusers pipeline signature accepting any args/kwargs
    class DummyPipeline:
        def __init__(self):
            # Mock pipeline attributes
            class MockScheduler:
                pass
            self.scheduler = MockScheduler()
            self.vae = MockComponent()
            self.text_encoder = MockComponent()
            self.text_encoder_2 = MockComponent()
            self.tokenizer = MockComponent()
            self.tokenizer_2 = MockComponent()
            self.unet = MockComponent()
            
        def __call__(self, **kwargs):  # noqa: D401
            return DummyResult()
            
        def to(self, device):
            return self
            
        def enable_attention_slicing(self, slice_size):
            pass
            
        def enable_xformers_memory_efficient_attention(self):
            pass
            
        def enable_vae_slicing(self):
            pass
            
    return DummyPipeline()


class DummyBaseConfig(BaseGeneratorConfig):  # minimal concrete for abstract usage
    strategy: str = "mock"  # not used but required


def test_generate_image_with_injected_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = SDXLLightningConfig(
        pipeline_factory=dummy_pipeline_factory,
        controlnet_id=None,
    )
    gen = SDXLLightningGenerator(config=DummyBaseConfig(), output_dir=str(tmp_path), lightning_config=cfg)

    # Run generation
    out_path = asyncio.run(gen.generate_image("a test prompt"))

    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"


def test_scheduler_override(tmp_path: Path):
    # Provide a scheduler config to ensure it is accepted (no real scheduling occurs in dummy pipeline)
    cfg = SDXLLightningConfig(
        pipeline_factory=dummy_pipeline_factory,
        controlnet_id=None,
        scheduler_config={"dummy": True},
    )
    gen = SDXLLightningGenerator(config=DummyBaseConfig(), output_dir=str(tmp_path), lightning_config=cfg)
    # Should not raise
    out_path = asyncio.run(gen.generate_image("another prompt"))
    assert Path(out_path).exists()


def test_model_type_detection():
    """Test the model type detection logic."""
    gen = LocalSDXLGenerator(config=DummyBaseConfig(), output_dir="/tmp")
    
    # Test URL detection
    model_type, path_or_id, filename = gen._detect_model_type("https://example.com/model.safetensors")
    assert model_type == "url"
    assert path_or_id == "https://example.com/model.safetensors"
    assert filename == "model.safetensors"
    
    # Test HuggingFace ID detection
    model_type, path_or_id, filename = gen._detect_model_type("stabilityai/stable-diffusion-xl-base-1.0")
    assert model_type == "hf_id"
    assert path_or_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert filename is None
    
    # Test local file detection
    model_type, path_or_id, filename = gen._detect_model_type("my_model.safetensors")
    assert model_type == "file"
    assert path_or_id == "my_model.safetensors"
    assert filename == "my_model.safetensors"


def test_image_to_image_generation(tmp_path: Path):
    """Test image-to-image generation with PIL Image input."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig, LocalSDXLGenerator
    
    cfg = LocalSDXLConfig(
        model="test_model.safetensors",
        pipeline_factory=dummy_pipeline_factory,
        controlnet_id=None,  # No ControlNet for this test
        steps=4,
        width=512,
        height=512
    )
    gen = LocalSDXLGenerator(cfg)
    gen.output_dir = tmp_path  # Use Path object, not string
    
    # Create a test input image
    input_image = Image.new("RGB", (512, 512), color="blue")
    
    # Run image-to-image generation
    out_path = asyncio.run(gen.generate_image(
        "transform this into a sunset scene", 
        image=input_image,
        strength=0.7
    ))
    
    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"


def test_controlnet_generation(tmp_path: Path):
    """Test ControlNet generation with depth map."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig, LocalSDXLGenerator
    
    # Mock the ControlNet pipeline creation to avoid loading real models
    def mock_controlnet_pipeline(*args, **kwargs):
        # Return the dummy pipeline but mark it as a ControlNet pipeline
        from unittest.mock import MagicMock
        
        # Create a pipeline that behaves like StableDiffusionXLControlNetPipeline for isinstance checks
        mock_pipeline = MagicMock()
        mock_pipeline.__class__.__name__ = 'StableDiffusionXLControlNetPipeline'
        mock_pipeline.__class__.__module__ = 'diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl'
        
        # Mock the __call__ method to return a realistic result
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), color="red")]
        mock_pipeline.return_value = mock_result
        
        # Mock all the other methods
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = MagicMock()
        mock_pipeline.enable_xformers_memory_efficient_attention = MagicMock()
        mock_pipeline.enable_vae_slicing = MagicMock()
        
        return mock_pipeline
    
    cfg = LocalSDXLConfig(
        model="test_model.safetensors",
        pipeline_factory=mock_controlnet_pipeline,  # Use mock ControlNet pipeline
        steps=4,
        width=512,
        height=512
    )
    gen = LocalSDXLGenerator(cfg)
    gen.output_dir = tmp_path  # Use Path object, not string
    
    # Create a test depth map
    depth_map = Image.new("RGB", (512, 512), color="gray")
    
    # Run ControlNet generation
    out_path = asyncio.run(gen.generate_image(
        "a portrait with depth control",
        depth_map=depth_map
    ))
    
    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"


def test_text_to_image_generation(tmp_path: Path):
    """Test standard text-to-image generation (no image input)."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig, LocalSDXLGenerator
    
    cfg = LocalSDXLConfig(
        model="test_model.safetensors",
        pipeline_factory=dummy_pipeline_factory,
        controlnet_id=None,  # No ControlNet for txt2img
        steps=4,
        width=512,
        height=512
    )
    gen = LocalSDXLGenerator(cfg)
    gen.output_dir = tmp_path  # Use Path object, not string
    
    # Run text-to-image generation
    out_path = asyncio.run(gen.generate_image("a beautiful landscape"))
    
    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"


def test_controlnet_overrides_img2img(tmp_path: Path):
    """Test that ControlNet depth map takes precedence over img2img image."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig, LocalSDXLGenerator
    
    # Mock the ControlNet pipeline creation to avoid loading real models  
    def mock_controlnet_pipeline(*args, **kwargs):
        from unittest.mock import MagicMock
        
        # Create a pipeline that behaves like StableDiffusionXLControlNetPipeline for isinstance checks
        mock_pipeline = MagicMock()
        mock_pipeline.__class__.__name__ = 'StableDiffusionXLControlNetPipeline'
        mock_pipeline.__class__.__module__ = 'diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl'
        
        # Mock the __call__ method to return a realistic result
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), color="red")]
        mock_pipeline.return_value = mock_result
        
        # Mock all the other methods
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = MagicMock()
        mock_pipeline.enable_xformers_memory_efficient_attention = MagicMock()
        mock_pipeline.enable_vae_slicing = MagicMock()
        
        return mock_pipeline
    
    cfg = LocalSDXLConfig(
        model="test_model.safetensors",
        pipeline_factory=mock_controlnet_pipeline,  # Use mock ControlNet pipeline
        steps=4,
        width=512,
        height=512
    )
    gen = LocalSDXLGenerator(cfg)
    gen.output_dir = tmp_path  # Use Path object, not string
    
    # Create both an input image and depth map
    input_image = Image.new("RGB", (512, 512), color="blue")
    depth_map = Image.new("RGB", (512, 512), color="gray")
    
    # Run generation with both - ControlNet should take precedence
    out_path = asyncio.run(gen.generate_image(
        "a portrait with depth control",
        image=input_image,  # This should be overridden by depth_map
        depth_map=depth_map,
        strength=0.8  # This should be removed for ControlNet
    ))
    
    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"


# ========== Additional Unit Tests for Image-to-Image Functionality ==========

def test_image_parameter_handling():
    """Test that the generate_image method properly handles image and depth_map parameters."""
    # Create a generator with mock config
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig
    
    config = LocalSDXLConfig(model="test_model.safetensors", controlnet_id=None)
    generator = LocalSDXLGenerator(config)
    generator.output_dir = Path("/tmp")
    
    # Mock the pipeline
    mock_pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (512, 512), color="red")]
    mock_pipeline.return_value = mock_result
    generator._pipeline = mock_pipeline
    
    # Mock _get_output_path to return a test path
    with patch.object(generator, '_get_output_path', return_value="/tmp/test.png"):
        # Test text-to-image (no image parameter)
        asyncio.run(generator.generate_image("test prompt"))
        
        # Verify the pipeline was called with correct parameters
        call_args = mock_pipeline.call_args[1]  # Get keyword arguments
        assert "prompt" in call_args
        assert call_args["prompt"] == "test prompt"
        assert "width" in call_args
        assert "height" in call_args
        assert "image" not in call_args  # No image parameter for txt2img
        assert "strength" not in call_args  # No strength for txt2img


def test_image_to_image_parameter_handling():
    """Test that image-to-image parameters are properly set."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig
    
    config = LocalSDXLConfig(model="test_model.safetensors", controlnet_id=None)
    generator = LocalSDXLGenerator(config)
    generator.output_dir = Path("/tmp")
    
    # Mock the pipeline
    mock_pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (512, 512), color="red")]
    mock_pipeline.return_value = mock_result
    generator._pipeline = mock_pipeline
    
    # Create test input image
    input_image = Image.new("RGB", (512, 512), color="blue")
    
    with patch.object(generator, '_get_output_path', return_value="/tmp/test.png"):
        # Test image-to-image generation
        asyncio.run(generator.generate_image(
            "test prompt", 
            image=input_image, 
            strength=0.7
        ))
        
        # Verify the pipeline was called with image and strength
        call_args = mock_pipeline.call_args[1]
        assert call_args["prompt"] == "test prompt"
        assert call_args["image"] is input_image
        assert call_args["strength"] == 0.7


def test_controlnet_parameter_handling():
    """Test that ControlNet depth map parameters are properly handled."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig
    from diffusers import StableDiffusionXLControlNetPipeline
    
    config = LocalSDXLConfig(model="test_model.safetensors")
    generator = LocalSDXLGenerator(config)
    generator.output_dir = Path("/tmp")
    
    # Create a mock that satisfies isinstance checks
    mock_pipeline = MagicMock(spec=StableDiffusionXLControlNetPipeline)
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (512, 512), color="red")]
    mock_pipeline.return_value = mock_result
    generator._pipeline = mock_pipeline
    
    # Create test depth map
    depth_map = Image.new("RGB", (512, 512), color="gray")
    
    with patch.object(generator, '_get_output_path', return_value="/tmp/test.png"):
        # Test ControlNet generation
        asyncio.run(generator.generate_image(
            "test prompt",
            depth_map=depth_map
        ))
        
        # Verify the pipeline was called with the depth map as image
        call_args = mock_pipeline.call_args[1]
        assert call_args["prompt"] == "test prompt"
        assert call_args["image"] is depth_map  # ControlNet uses image parameter for depth map
        assert "strength" not in call_args  # No strength for ControlNet


def test_controlnet_overrides_img2img_parameter_handling():
    """Test that ControlNet depth map overrides img2img image parameter."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig
    from diffusers import StableDiffusionXLControlNetPipeline
    
    config = LocalSDXLConfig(model="test_model.safetensors")
    generator = LocalSDXLGenerator(config)
    generator.output_dir = Path("/tmp")
    
    # Create a mock that satisfies isinstance checks
    mock_pipeline = MagicMock(spec=StableDiffusionXLControlNetPipeline)
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (512, 512), color="red")]
    mock_pipeline.return_value = mock_result
    generator._pipeline = mock_pipeline
    
    # Create test images
    input_image = Image.new("RGB", (512, 512), color="blue")
    depth_map = Image.new("RGB", (512, 512), color="gray")
    
    # Patch isinstance in the sdxl_generator module specifically
    with patch('image_server.generators.local.sdxl_generator.isinstance') as mock_isinstance, \
         patch.object(generator, '_get_output_path', return_value="/tmp/test.png"):
        
        # Make isinstance return True for our mock ControlNet pipeline
        mock_isinstance.side_effect = lambda obj, classinfo: (
            True if obj is mock_pipeline and classinfo is StableDiffusionXLControlNetPipeline
            else isinstance.__wrapped__(obj, classinfo)
        )
        
        # Test generation with both image and depth_map - ControlNet should win
        asyncio.run(generator.generate_image(
            "test prompt",
            image=input_image,  # This should be overridden
            depth_map=depth_map,
            strength=0.8  # This should be removed
        ))
        
        # Verify ControlNet depth map takes precedence
        call_args = mock_pipeline.call_args[1]
        assert call_args["prompt"] == "test prompt"
        assert call_args["image"] is depth_map  # ControlNet depth map wins
        assert call_args["image"] is not input_image  # img2img image is overridden
        assert "strength" not in call_args  # Strength removed for ControlNet


def test_generation_mode_logging():
    """Test that the correct generation mode is logged."""
    from image_server.generators.local.sdxl_generator import LocalSDXLConfig
    from diffusers import StableDiffusionXLControlNetPipeline
    
    config = LocalSDXLConfig(model="test_model.safetensors", controlnet_id=None)
    generator = LocalSDXLGenerator(config)
    generator.output_dir = Path("/tmp")
    
    # Mock the pipeline
    mock_pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (512, 512), color="red")]
    mock_pipeline.return_value = mock_result
    generator._pipeline = mock_pipeline
    
    with patch.object(generator, '_get_output_path', return_value="/tmp/test.png"), \
         patch('image_server.generators.local.sdxl_generator.logger') as mock_logger:
        
        # Test txt2img logging
        asyncio.run(generator.generate_image("test prompt"))
        
        # Check that txt2img was logged
        info_calls = [call for call in mock_logger.info.call_args_list if 'txt2img' in str(call)]
        assert len(info_calls) > 0
        
        # Reset mock and test img2img logging
        mock_logger.reset_mock()
        input_image = Image.new("RGB", (512, 512), color="blue")
        asyncio.run(generator.generate_image("test prompt", image=input_image))
        
        # Check that img2img was logged
        info_calls = [call for call in mock_logger.info.call_args_list if 'img2img' in str(call)]
        assert len(info_calls) > 0
        
        # Reset mock and test ControlNet logging
        mock_logger.reset_mock()
        
        # Create a mock ControlNet pipeline
        controlnet_pipeline = MagicMock(spec=StableDiffusionXLControlNetPipeline)
        controlnet_pipeline.return_value = mock_result
        generator._pipeline = controlnet_pipeline
        
        # Patch isinstance in the sdxl_generator module specifically
        with patch('image_server.generators.local.sdxl_generator.isinstance') as mock_isinstance_inner:
            # Make isinstance return True for our mock ControlNet pipeline
            mock_isinstance_inner.side_effect = lambda obj, classinfo: (
                True if obj is controlnet_pipeline and classinfo is StableDiffusionXLControlNetPipeline
                else isinstance.__wrapped__(obj, classinfo)
            )
            
            depth_map = Image.new("RGB", (512, 512), color="gray")
            asyncio.run(generator.generate_image("test prompt", depth_map=depth_map))
        
        # Check that controlnet was logged
        info_calls = [call for call in mock_logger.info.call_args_list if 'controlnet' in str(call)]
        assert len(info_calls) > 0

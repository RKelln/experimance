import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

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

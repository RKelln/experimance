
from image_server.generators.generator import ImageGenerator, MockImageGenerator
from image_server.generators.fal.fal_comfy_generator import FalComfyGenerator
from image_server.generators.openai.openai_generator import OpenAIGenerator
from image_server.generators.local.sdxl_generator import LocalSDXLGenerator

# Factory function to create generators
def create_generator(strategy: str, **kwargs) -> ImageGenerator:
    """Factory function to create image generators.
    
    Args:
        strategy: Generator strategy ("mock", "fal", "openai", "local")
        **kwargs: Configuration options for the generator
        
    Returns:
        Configured ImageGenerator instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    generators = {
        "mock": MockImageGenerator,
        "fal": FalComfyGenerator,
        "openai": OpenAIGenerator,
        "local": LocalSDXLGenerator
    }
    
    if strategy not in generators:
        raise ValueError(f"Unsupported generator strategy: {strategy}. "
                        f"Available strategies: {list(generators.keys())}")
    
    return generators[strategy](**kwargs)

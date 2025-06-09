from typing import Optional, Literal

from image_server.generators.config import BaseGeneratorConfig

class MockGeneratorConfig(BaseGeneratorConfig):
    strategy: Literal["mock"] = "mock"
    image_size: tuple = (1024, 1024)
    background_color: tuple = (100, 150, 200)
    text_color: tuple = (255, 255, 255)
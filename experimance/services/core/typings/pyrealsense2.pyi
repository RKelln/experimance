"""
Type stubs for pyrealsense2 - Common functions used in Experimance.
This provides basic type hints for the most commonly used RealSense functions.
"""

from typing import Any, List, Optional
import numpy as np

class context:
    def query_devices(self) -> List[Any]: ...

class pipeline:
    def start(self, config: Any) -> Any: ...
    def stop(self) -> None: ...
    def wait_for_frames(self) -> Any: ...

class config:
    def enable_stream(self, stream: Any, width: int, height: int, format: Any, fps: int) -> None: ...

class colorizer:
    def __init__(self, color_scheme: int = 0) -> None: ...
    def colorize(self, frame: Any) -> Any: ...
    def set_option(self, option: Any, value: Any) -> None: ...

class align:
    def __init__(self, stream: Any) -> None: ...
    def process(self, frames: Any) -> Any: ...

# Stream types
class stream:
    depth: Any
    color: Any

# Formats
class format:
    z16: Any
    bgr8: Any

# Camera info
class camera_info:
    name: Any
    serial_number: Any
    firmware_version: Any
    product_id: Any
    usb_type_descriptor: Any

# Options
class option:
    visual_preset: Any
    min_distance: Any
    max_distance: Any
    color_scheme: Any

# Other commonly used functions and classes
def load_json(path: str) -> bool: ...

# Filters
class decimation_filter:
    def __init__(self) -> None: ...
    def set_option(self, option: Any, value: Any) -> None: ...

class threshold_filter:
    def __init__(self, min_dist: float = 0.0, max_dist: float = 4.0) -> None: ...

class spatial_filter:
    def __init__(self) -> None: ...
    def set_option(self, option: Any, value: Any) -> None: ...

class temporal_filter:
    def __init__(self) -> None: ...

class hole_filling_filter:
    def __init__(self) -> None: ...
    def set_option(self, option: Any, value: Any) -> None: ...

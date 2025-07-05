"""Flow management module for the Experimance agent service."""

from .base_flow_manager import BaseFlowManager, ConfigurableFlowManager, create_flow_manager
from .experimance_flows import ExperimanceFlowManager
from .simple_flows import SimpleFlowManager

# Import adapter components (optional dependencies)
try:
    from .openai_realtime_adapter import OpenAIRealtimeAdapter, is_openai_realtime_adapter_available
    from .adapter_registry import register_custom_adapters, get_adapter_for_llm, is_adapter_registration_available
    ADAPTER_SUPPORT_AVAILABLE = True
    
    # Register adapters immediately when the module is imported
    register_custom_adapters()
except ImportError:
    ADAPTER_SUPPORT_AVAILABLE = False

__all__ = [
    "BaseFlowManager", 
    "ConfigurableFlowManager", 
    "ExperimanceFlowManager", 
    "SimpleFlowManager",
    "create_flow_manager"
]

# Add adapter exports if available
if ADAPTER_SUPPORT_AVAILABLE:
    __all__.extend([
        "OpenAIRealtimeAdapter",
        "is_openai_realtime_adapter_available", 
        "register_custom_adapters",
        "get_adapter_for_llm",
        "is_adapter_registration_available"
    ])

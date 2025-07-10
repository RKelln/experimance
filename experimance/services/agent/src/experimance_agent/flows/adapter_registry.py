"""
Adapter registry for custom LLM adapters in Experimance.

This module handles the registration of custom adapters with pipecat-flows,
allowing for seamless integration of new LLM services like OpenAI Realtime Beta.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Track registration status
_ADAPTERS_REGISTERED = False

try:
    # Import the module but not the function directly to avoid reference issues
    import pipecat_flows.adapters
    from pipecat.services.openai_realtime_beta import OpenAIRealtimeBetaLLMService
    PIPECAT_FLOWS_AVAILABLE = True
    
    # Patch immediately on import, before any other code can cache the function reference
    _immediate_patch_applied = False
    
except ImportError:
    PIPECAT_FLOWS_AVAILABLE = False
    _immediate_patch_applied = False
    logger.warning("pipecat-flows not available for custom adapter registration")

from .openai_realtime_adapter import (
    OpenAIRealtimeAdapter, 
    is_openai_realtime_adapter_available
)


def register_custom_adapters():
    """Register custom adapters with pipecat-flows.
    
    This function patches the create_adapter function to include our custom adapters.
    It should be called once during application startup.
    """
    global _ADAPTERS_REGISTERED
    
    if _ADAPTERS_REGISTERED or _immediate_patch_applied:
        logger.debug("Custom adapters already registered (or immediate patch applied)")
        # Try deferred patching in case new modules were loaded
        _apply_deferred_patch()
        return
    
    if not PIPECAT_FLOWS_AVAILABLE:
        logger.warning("Cannot register custom adapters: pipecat-flows not available")
        return
    
    # If immediate patch wasn't applied, apply it now
    _apply_immediate_patch()
    # Also apply deferred patch for any modules loaded after immediate patch
    _apply_deferred_patch()
    _ADAPTERS_REGISTERED = True


def _apply_immediate_patch():
    """Apply the patch immediately when the module is imported."""
    global _immediate_patch_applied
    
    if _immediate_patch_applied or not PIPECAT_FLOWS_AVAILABLE:
        return
    
    try:
        import pipecat_flows.adapters as adapter_module
        original_create_adapter = adapter_module.create_adapter
        
        def patched_create_adapter(llm) -> Any:
            """Patched create_adapter function that includes our custom adapters."""
            llm_type = type(llm).__name__
            
            # Check for our custom OpenAI Realtime Beta adapter FIRST
            if llm_type == "OpenAIRealtimeBetaLLMService":
                # Import here to avoid circular imports
                from .openai_realtime_adapter import OpenAIRealtimeAdapter, is_openai_realtime_adapter_available
                
                if is_openai_realtime_adapter_available():
                    return OpenAIRealtimeAdapter()
                else:
                    logger.warning("⚠️  IMMEDIATE PATCH: OpenAI Realtime adapter requested but not available")
            
            # Fall back to the original create_adapter function
            #logger.debug(f"IMMEDIATE PATCH: Falling back to original create_adapter for {llm_type}")
            return original_create_adapter(llm)
        
        # Replace the create_adapter function immediately
        adapter_module.create_adapter = patched_create_adapter
        
        # CRITICAL: Also patch any existing imports in other modules
        import sys
        
        # Patch in the main adapters module
        pipecat_flows_adapters = sys.modules.get('pipecat_flows.adapters')
        if pipecat_flows_adapters:
            pipecat_flows_adapters.create_adapter = patched_create_adapter # type:ignore[assignment]
        
        # CRITICAL: Patch in the manager module which imports create_adapter directly
        pipecat_flows_manager = sys.modules.get('pipecat_flows.manager')
        if pipecat_flows_manager:
            # The manager module imports: from .adapters import create_adapter
            # So we need to patch that reference too
            pipecat_flows_manager.create_adapter = patched_create_adapter # type:ignore[assignment]
        
        # Store the patched function for later use
        _apply_immediate_patch.patched_function = patched_create_adapter
        
        _immediate_patch_applied = True
        
    except Exception as e:
        logger.error(f"❌ IMMEDIATE PATCH: Failed to register custom adapters: {e}")
        import traceback
        traceback.print_exc()


def _apply_deferred_patch():
    """Apply patches to modules that weren't loaded during immediate patching."""
    if not PIPECAT_FLOWS_AVAILABLE or not _immediate_patch_applied:
        return
    
    if not hasattr(_apply_immediate_patch, 'patched_function'):
        return
    
    patched_function = _apply_immediate_patch.patched_function
    
    try:
        import sys
        
        # Patch manager module if it wasn't available before
        pipecat_flows_manager = sys.modules.get('pipecat_flows.manager')
        if pipecat_flows_manager and not hasattr(pipecat_flows_manager, '_experimance_patched'):
            pipecat_flows_manager.create_adapter = patched_function # type:ignore[assignment]
            pipecat_flows_manager._experimance_patched = True # type:ignore[assignment]
        
    except Exception as e:
        logger.error(f"❌ DEFERRED PATCH: Failed: {e}")
        print(f"❌ DEFERRED PATCH: Failed: {e}")


# Apply the patch immediately when this module is imported
if PIPECAT_FLOWS_AVAILABLE:
    _apply_immediate_patch()


def ensure_adapters_registered():
    """Ensure adapters are registered - call this before using flows."""
    if not _ADAPTERS_REGISTERED:
        register_custom_adapters()
    else:
        # Even if already registered, try deferred patching for new modules
        _apply_deferred_patch()


def get_adapter_for_llm(llm: Any) -> Optional[Any]:
    """Get the appropriate adapter for an LLM service.
    
    This is a convenience function that ensures adapters are registered
    and then returns the appropriate adapter.
    
    Args:
        llm: The LLM service instance
        
    Returns:
        The appropriate adapter instance, or None if not available
    """
    if not PIPECAT_FLOWS_AVAILABLE:
        return None
    
    # Ensure adapters are registered
    register_custom_adapters()
    
    try:
        import pipecat_flows.adapters
        return pipecat_flows.adapters.create_adapter(llm)
    except Exception as e:
        logger.error(f"Failed to get adapter for LLM {type(llm).__name__}: {e}")
        return None


def is_adapter_registration_available() -> bool:
    """Check if adapter registration is available."""
    return PIPECAT_FLOWS_AVAILABLE

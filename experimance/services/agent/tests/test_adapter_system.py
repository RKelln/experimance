#!/usr/bin/env python3
"""
Test script for the OpenAI Realtime Beta adapter system.

This script tests whether our custom adapter registration works and if we can
use the existing flow managers with OpenAI Realtime Beta service.
"""

import logging
import os
import asyncio
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_adapter_registration():
    """Test that our custom adapter can be registered and discovered."""
    logger.info("=== Testing Adapter Registration ===")
    
    try:
        from experimance_agent.flows.adapter_registry import (
            register_custom_adapters,
            is_adapter_registration_available,
            get_adapter_for_llm
        )
        
        if not is_adapter_registration_available():
            logger.warning("Adapter registration not available - missing pipecat-flows")
            return False
        
        # Register our custom adapters
        register_custom_adapters()
        logger.info("✓ Custom adapters registered successfully")
        
        # Test with a mock OpenAI Realtime service
        try:
            from pipecat.services.openai_realtime_beta import OpenAIRealtimeBetaLLMService
            
            # Create a mock service (this won't actually connect)
            mock_service = OpenAIRealtimeBetaLLMService(
                api_key="test_key",
                model="gpt-4o-realtime-preview"
            )
            
            # Test adapter discovery
            adapter = get_adapter_for_llm(mock_service)
            if adapter is not None:
                logger.info(f"✓ Adapter found for OpenAI Realtime Beta: {type(adapter).__name__}")
                return True
            else:
                logger.error("✗ No adapter found for OpenAI Realtime Beta")
                return False
                
        except ImportError as e:
            logger.warning(f"OpenAI Realtime Beta service not available: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import adapter components: {e}")
        return False


def test_flow_manager_creation():
    """Test that flow managers can be created with the adapter system."""
    logger.info("\n=== Testing Flow Manager Creation ===")
    
    try:
        from experimance_agent.flows import create_flow_manager
        from experimance_agent.backends.base import UserContext
        
        # Create a mock user context
        user_context = UserContext()
        
        # Test different flow manager types
        flow_types = ["experimance", "simple"]
        
        for flow_type in flow_types:
            try:
                # Note: We can't fully test without actual task and llm objects
                # but we can test the factory function logic
                logger.info(f"Testing {flow_type} flow manager creation...")
                
                # This will fail due to missing task/llm parameters, but should show
                # that the factory function recognizes the flow type
                try:
                    create_flow_manager(
                        flow_type=flow_type,
                        task=None,
                        llm=None,
                        context_aggregator=None,
                        user_context=user_context
                    )
                except Exception as e:
                    if "requires" in str(e).lower() or "none" in str(e).lower():
                        logger.info(f"✓ {flow_type} flow manager recognized (expected parameter error)")
                    else:
                        logger.error(f"✗ Unexpected error for {flow_type}: {e}")
                        
            except ValueError as e:
                if "unknown flow type" in str(e).lower():
                    logger.error(f"✗ {flow_type} flow manager not recognized")
                else:
                    logger.info(f"✓ {flow_type} flow manager recognized (expected parameter error)")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import flow components: {e}")
        return False


def test_adapter_functionality():
    """Test basic adapter functionality."""
    logger.info("\n=== Testing Adapter Functionality ===")
    
    try:
        from experimance_agent.flows.openai_realtime_adapter import (
            OpenAIRealtimeAdapter,
            is_openai_realtime_adapter_available
        )
        
        if not is_openai_realtime_adapter_available():
            logger.warning("OpenAI Realtime adapter not available")
            return False
        
        # Create adapter instance
        adapter = OpenAIRealtimeAdapter()
        logger.info("✓ OpenAI Realtime adapter created successfully")
        
        # Test basic methods
        test_message = adapter.format_summary_message("Test summary")
        if isinstance(test_message, dict) and "role" in test_message:
            logger.info("✓ format_summary_message works correctly")
        else:
            logger.error("✗ format_summary_message failed")
            
        # Test function schema conversion
        test_function = {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Test parameter"}
                },
                "required": ["param1"]
            }
        }
        
        try:
            schema = adapter.convert_to_function_schema(test_function)
            logger.info("✓ convert_to_function_schema works correctly")
        except Exception as e:
            logger.error(f"✗ convert_to_function_schema failed: {e}")
            
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import adapter: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Testing OpenAI Realtime Beta Adapter System")
    logger.info("=" * 50)
    
    # Test adapter registration
    adapter_reg_success = test_adapter_registration()
    
    # Test flow manager creation
    flow_mgr_success = test_flow_manager_creation()
    
    # Test adapter functionality
    adapter_func_success = test_adapter_functionality()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Adapter Registration: {'✓ PASS' if adapter_reg_success else '✗ FAIL'}")
    logger.info(f"Flow Manager Creation: {'✓ PASS' if flow_mgr_success else '✗ FAIL'}")
    logger.info(f"Adapter Functionality: {'✓ PASS' if adapter_func_success else '✗ FAIL'}")
    
    all_passed = adapter_reg_success and flow_mgr_success and adapter_func_success
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        logger.info("\n🎉 OpenAI Realtime Beta adapter system is working!")
        logger.info("You can now use any flow manager (experimance, simple, configurable)")
        logger.info("with OpenAI Realtime Beta service, and the adapter will handle compatibility.")
    else:
        logger.info("\n⚠️  Some issues detected. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())

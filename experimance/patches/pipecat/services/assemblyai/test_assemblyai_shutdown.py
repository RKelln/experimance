#!/usr/bin/env python3
"""
Test script to verify AssemblyAI shutdown fix works correctly.
This script creates a minimal pipeline and tests shutdown behavior.
"""

import asyncio
import signal
import sys
import time
from unittest.mock import Mock

# Mock the required pipecat components for testing
class MockFrame:
    pass

class MockStartFrame(MockFrame):
    pass

class MockEndFrame(MockFrame):
    pass

class MockCancelFrame(MockFrame):
    pass

# Test the shutdown timing
async def test_assemblyai_shutdown():
    """Test that AssemblyAI STT service shuts down quickly."""
    print("Testing AssemblyAI STT shutdown behavior...")
    
    try:
        # Import the patched service
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
        
        # Create service with mock API key
        service = AssemblyAISTTService(
            api_key="test_key_dummy",
            connection_params=AssemblyAIConnectionParams()
        )
        
        # Mock the websocket connection to avoid actual network calls
        service._websocket = Mock()
        service._websocket.closed = False
        service._websocket.send = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._websocket.close = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._connected = True
        
        # Create a proper mock task that can be cancelled
        async def mock_receive_task():
            try:
                await asyncio.sleep(10)  # Long-running task
            except asyncio.CancelledError:
                pass
        
        service._receive_task = asyncio.create_task(mock_receive_task())
        
        # Test disconnect timing (forced shutdown)
        start_time = time.time()
        await service._disconnect(force=True)
        end_time = time.time()
        
        forced_disconnect_time = end_time - start_time
        print(f"Forced disconnect completed in {forced_disconnect_time:.2f} seconds")
        
        # Reset for graceful shutdown test
        service._connected = True
        service._websocket = Mock()
        service._websocket.closed = False
        service._websocket.send = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._websocket.close = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._receive_task = asyncio.create_task(mock_receive_task())
        
        # Test graceful shutdown
        start_time = time.time()
        await service._disconnect(force=False)
        end_time = time.time()
        
        graceful_disconnect_time = end_time - start_time
        print(f"Graceful disconnect completed in {graceful_disconnect_time:.2f} seconds")
        
        # Both should be fast, but forced should be faster
        if forced_disconnect_time < 0.1 and graceful_disconnect_time < 2.0:
            print("✅ SUCCESS: Both forced and graceful shutdowns completed quickly")
            print(f"   - Forced: {forced_disconnect_time:.2f}s (should be < 0.1s)")  
            print(f"   - Graceful: {graceful_disconnect_time:.2f}s (should be < 2s)")
            return True
        else:
            print("❌ FAILURE: Shutdown took too long")
            print(f"   - Forced: {forced_disconnect_time:.2f}s")
            print(f"   - Graceful: {graceful_disconnect_time:.2f}s")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import AssemblyAI service: {e}")
        print("Make sure pipecat is installed with assemblyai support")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals for testing."""
    print(f"\n🔄 Received signal {signum}, testing shutdown...")
    asyncio.create_task(test_and_exit())

async def test_and_exit():
    """Run test and exit."""
    success = await test_assemblyai_shutdown()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("AssemblyAI STT Shutdown Test")
    print("=" * 40)
    
    # Set up signal handler to test SIGINT behavior
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Press Ctrl+C to test shutdown behavior, or wait for automatic test...")
    print("(Automatic test will run in 3 seconds)")
    
    # Run automatic test after delay
    async def auto_test():
        await asyncio.sleep(3)
        await test_and_exit()
    
    try:
        asyncio.run(auto_test())
    except KeyboardInterrupt:
        # This should be handled by signal handler
        pass

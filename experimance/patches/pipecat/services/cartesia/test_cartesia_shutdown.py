#!/usr/bin/env python3
"""
Test script to verify Cartesia TTS shutdown fix works correctly.
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
async def test_cartesia_shutdown():
    """Test that Cartesia TTS service shuts down quickly."""
    print("Testing Cartesia TTS shutdown behavior...")
    
    try:
        # Import the patched service
        from pipecat.services.cartesia.tts import CartesiaTTSService
        
        # Create service with mock API key
        service = CartesiaTTSService(
            api_key="test_key_dummy",
            voice_id="test_voice_id"
        )
        
        # Mock the websocket connection to avoid actual network calls
        service._websocket = Mock()
        service._websocket.closed = False
        service._websocket.open = True
        service._websocket.send = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._websocket.close = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._connected = True
        
        # Create a proper mock task that can be cancelled
        async def mock_receive_task():
            try:
                # Simulate long-running task that waits to be cancelled
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                print("  → Receive task cancelled properly")
                raise
            
        service._receive_task = asyncio.create_task(mock_receive_task())
        
        # Mock the cancel_task method
        async def mock_cancel_task(task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        service.cancel_task = mock_cancel_task
        
        print("🔄 Testing graceful shutdown...")
        start_time = time.time()
        await service._disconnect(force=False)
        graceful_time = time.time() - start_time
        print(f"  ✅ Graceful shutdown: {graceful_time:.3f}s")
        
        # Reset for forced shutdown test
        service._connected = True
        service._context_id = "test_context_123"
        service._receive_task = asyncio.create_task(mock_receive_task())
        
        print("⚡ Testing forced shutdown...")
        start_time = time.time()
        await service._disconnect(force=True)
        forced_time = time.time() - start_time
        print(f"  ⚡ Forced shutdown: {forced_time:.3f}s")
        
        # Evaluate results
        print(f"\n📊 Results:")
        print(f"   - Graceful shutdown: {graceful_time:.3f}s")
        print(f"   - Forced shutdown: {forced_time:.3f}s")
        
        # Forced shutdown should be significantly faster (skips context cleanup)
        if forced_time < graceful_time:
            print("✅ SUCCESS: Forced shutdown is faster than graceful shutdown")
            return True
        else:
            print("⚠️  WARNING: Forced shutdown should be faster")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import pipecat components: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def test_cancel_frame_shutdown():
    """Test that cancel frame triggers forced shutdown."""
    print("\n🔄 Testing CancelFrame triggers forced shutdown...")
    
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.frames.frames import CancelFrame
        
        service = CartesiaTTSService(
            api_key="test_key_dummy",
            voice_id="test_voice_id"
        )
        
        # Mock websocket
        service._websocket = Mock()
        service._websocket.closed = False
        service._websocket.close = Mock(return_value=asyncio.create_task(asyncio.sleep(0.01)))
        service._connected = True
        
        # Mock receive task
        async def mock_receive_task():
            await asyncio.sleep(10)
            
        service._receive_task = asyncio.create_task(mock_receive_task())
        
        # Mock cancel_task
        async def mock_cancel_task(task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        service.cancel_task = mock_cancel_task
        service.stop_all_metrics = Mock(return_value=asyncio.create_task(asyncio.sleep(0.001)))
        
        # Test cancel frame
        start_time = time.time()
        await service.cancel(CancelFrame())
        cancel_time = time.time() - start_time
        
        print(f"  ⚡ Cancel frame shutdown: {cancel_time:.3f}s")
        
        if cancel_time < 0.1:  # Should be very fast
            print("✅ SUCCESS: Cancel frame shutdown is fast")
            return True
        else:
            print("⚠️  WARNING: Cancel frame shutdown should be faster")
            return False
            
    except Exception as e:
        print(f"❌ Cancel frame test failed: {e}")
        return False

async def main():
    """Main test runner."""
    print("🧪 Cartesia TTS Shutdown Fix Test Suite")
    print("=" * 50)
    
    results = []
    
    # Run basic shutdown test
    try:
        result1 = await test_cartesia_shutdown()
        results.append(result1)
    except Exception as e:
        print(f"❌ Basic shutdown test failed: {e}")
        results.append(False)
    
    # Run cancel frame test
    try:
        result2 = await test_cancel_frame_shutdown()
        results.append(result2)
    except Exception as e:
        print(f"❌ Cancel frame test failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 Cartesia TTS shutdown fix is working correctly")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("🔧 The shutdown fix may need additional work")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)

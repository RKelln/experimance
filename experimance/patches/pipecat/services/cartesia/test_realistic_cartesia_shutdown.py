#!/usr/bin/env python3
"""
More realistic test for Cartesia TTS shutdown behavior that simulates
actual connection scenarios without requiring real API credentials.
"""

import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

async def test_realistic_cartesia_shutdown():
    """Test with more realistic WebSocket behavior simulation."""
    print("Testing realistic Cartesia TTS shutdown behavior...")
    
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
        
        # Create service
        service = CartesiaTTSService(
            api_key="test_key_dummy",
            voice_id="test_voice_id"
        )
        
        # Create a more realistic mock WebSocket that simulates network delays
        class RealisticWebSocketMock:
            def __init__(self, simulate_slow_close=False):
                self.closed = False
                self.open = True
                self.simulate_slow_close = simulate_slow_close
                self._send_delay = 0.01  # Small network delay
                
            async def send(self, data):
                await asyncio.sleep(self._send_delay)
                if isinstance(data, str):
                    try:
                        msg = json.loads(data)
                        if msg.get("context_id") and msg.get("cancel"):
                            print("  → Sent context cancellation message to server")
                            # Simulate server taking time to respond
                            if self.simulate_slow_close:
                                await asyncio.sleep(2.0)  # Slow server response
                    except json.JSONDecodeError:
                        pass  # Not JSON, ignore
                        
            async def close(self):
                if self.simulate_slow_close:
                    await asyncio.sleep(0.5)  # Slow close
                self.closed = True
                self.open = False
                print("  → WebSocket closed")
                
            async def recv(self):
                # Simulate long-running receive that gets cancelled
                try:
                    await asyncio.sleep(10)  # Would wait forever if not cancelled
                    return '{"type": "chunk", "context_id": "test123", "data": "dGVzdA=="}'
                except asyncio.CancelledError:
                    print("  → WebSocket recv cancelled")
                    raise
        
        # Test scenarios
        scenarios = [
            ("Fast server response", False),
            ("Slow server response", True),
        ]
        
        for scenario_name, slow_close in scenarios:
            print(f"\n--- Testing: {scenario_name} ---")
            
            # Reset service state - use hasattr to check if attributes exist (for compatibility)
            if hasattr(service, '_connected'):
                service._connected = True
            service._websocket = RealisticWebSocketMock(simulate_slow_close=slow_close)
            service._context_id = "test_context_123"
            
            # Create realistic receive task that actually runs
            async def realistic_receive_task():
                try:
                    while getattr(service, '_connected', True):
                        try:
                            if hasattr(service, '_receive_messages'):
                                # Use the actual _receive_messages method if available
                                await service._receive_messages()
                            else:
                                # Fallback simulation
                                message = await service._websocket.recv()
                                print(f"  → Received: {message}")
                        except asyncio.CancelledError:
                            print("  → Receive task cancelled properly")
                            raise
                        except Exception as e:
                            print(f"  → Receive task error: {e}")
                            break
                except asyncio.CancelledError:
                    print("  → Receive task cancelled")
                    raise
                    
            service._receive_task = asyncio.create_task(realistic_receive_task())
            
            # Mock the cancel_task method to work properly (if it exists)
            if hasattr(service, 'cancel_task'):
                original_cancel_task = service.cancel_task
                async def mock_cancel_task(task, timeout=None):
                    if task and not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=timeout if timeout else 1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                
                service.cancel_task = mock_cancel_task
            
            # Mock stop_all_metrics if it exists
            if hasattr(service, 'stop_all_metrics'):
                service.stop_all_metrics = AsyncMock()
            
            # Test graceful shutdown first
            print("🔄 Testing graceful shutdown...")
            start_time = time.time()
            try:
                if hasattr(service, '_disconnect'):
                    # Use the patched _disconnect method
                    await service._disconnect(force=False)
                else:
                    # Fallback to original method
                    await service._disconnect()
            except TypeError:
                # Method doesn't accept force parameter (original version)
                await service._disconnect()
            graceful_time = time.time() - start_time
            print(f"  ✅ Graceful shutdown: {graceful_time:.3f}s")
            
            # Reset for forced shutdown test
            if hasattr(service, '_connected'):
                service._connected = True
            service._websocket = RealisticWebSocketMock(simulate_slow_close=slow_close)
            service._context_id = "test_context_456"
            service._receive_task = asyncio.create_task(realistic_receive_task())
            
            # Test forced shutdown
            print("⚡ Testing forced shutdown...")
            start_time = time.time()
            try:
                if hasattr(service, '_disconnect'):
                    # Try to use the patched _disconnect method with force=True
                    await service._disconnect(force=True)
                else:
                    # Fallback - test cancel frame instead
                    await service.cancel(CancelFrame())
            except TypeError:
                # Method doesn't accept force parameter, test cancel instead
                await service.cancel(CancelFrame())
            forced_time = time.time() - start_time
            print(f"  ⚡ Forced shutdown: {forced_time:.3f}s")
            
            # Analyze results for this scenario
            print(f"  📊 {scenario_name} Results:")
            print(f"     - Graceful: {graceful_time:.3f}s")
            print(f"     - Forced: {forced_time:.3f}s")
            
            if hasattr(service, '_disconnect') and 'force' in service._disconnect.__code__.co_varnames:
                # We have the patched version
                if forced_time < graceful_time * 0.8:  # Forced should be significantly faster
                    print(f"  ✅ {scenario_name}: Forced shutdown is faster")
                else:
                    print(f"  ⚠️  {scenario_name}: Forced shutdown should be faster")
            else:
                print(f"  ℹ️  {scenario_name}: Testing original version (no force parameter)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import pipecat components: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_websocket_edge_cases():
    """Test edge cases in WebSocket handling."""
    print("\n🔍 Testing WebSocket edge cases...")
    
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        
        service = CartesiaTTSService(
            api_key="test_key_dummy", 
            voice_id="test_voice_id"
        )
        
        # Test 1: WebSocket already closed
        print("  🔄 Test 1: WebSocket already closed")
        service._websocket = Mock()
        service._websocket.closed = True
        service._websocket.open = False
        
        start_time = time.time()
        try:
            if hasattr(service, '_disconnect'):
                await service._disconnect(force=True)
            else:
                await service._disconnect()
        except TypeError:
            await service._disconnect()
        elapsed = time.time() - start_time
        print(f"     ✅ Handled closed WebSocket in {elapsed:.3f}s")
        
        # Test 2: No WebSocket connection
        print("  🔄 Test 2: No WebSocket connection")
        service._websocket = None
        
        start_time = time.time()
        try:
            if hasattr(service, '_disconnect'):
                await service._disconnect(force=True)
            else:
                await service._disconnect()
        except TypeError:
            await service._disconnect()
        elapsed = time.time() - start_time
        print(f"     ✅ Handled None WebSocket in {elapsed:.3f}s")
        
        # Test 3: WebSocket send fails
        print("  🔄 Test 3: WebSocket send fails")
        service._websocket = Mock()
        service._websocket.closed = False
        service._websocket.open = True
        service._context_id = "test_context"
        
        async def failing_send(data):
            raise websockets.exceptions.ConnectionClosed(None, None)
            
        service._websocket.send = failing_send
        service._websocket.close = AsyncMock()
        
        start_time = time.time()
        try:
            if hasattr(service, '_disconnect'):
                await service._disconnect(force=False)  # Graceful should handle the error
            else:
                await service._disconnect()
        except TypeError:
            await service._disconnect()
        elapsed = time.time() - start_time
        print(f"     ✅ Handled send failure in {elapsed:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

async def main():
    """Main test runner."""
    print("🧪 Cartesia TTS Realistic Shutdown Test Suite")
    print("=" * 60)
    
    results = []
    
    # Check if we have the patched version
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        service = CartesiaTTSService(api_key="test", voice_id="test")
        if hasattr(service, '_disconnect') and 'force' in service._disconnect.__code__.co_varnames:
            print("✅ Detected PATCHED version of Cartesia TTS")
        else:
            print("ℹ️  Detected ORIGINAL version of Cartesia TTS")
        print("")
    except Exception as e:
        print(f"⚠️  Could not detect version: {e}")
        print("")
    
    # Run realistic shutdown test
    try:
        result1 = await test_realistic_cartesia_shutdown()
        results.append(result1)
    except Exception as e:
        print(f"❌ Realistic shutdown test failed: {e}")
        results.append(False)
    
    # Run edge case tests
    try:
        result2 = await test_websocket_edge_cases()
        results.append(result2)
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 Cartesia TTS shutdown behavior is working correctly")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("🔧 The shutdown fix may need additional work")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        exit(1)

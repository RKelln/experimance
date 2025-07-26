#!/usr/bin/env python3
"""
More realistic test for AssemblyAI shutdown behavior that simulates
actual connection scenarios without requiring real API credentials.
"""

import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

async def test_realistic_assemblyai_shutdown():
    """Test with more realistic WebSocket behavior simulation."""
    print("Testing realistic AssemblyAI STT shutdown behavior...")
    
    try:
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
        from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
        
        # Create service
        service = AssemblyAISTTService(
            api_key="test_key_dummy",
            connection_params=AssemblyAIConnectionParams()
        )
        
        # Create a more realistic mock WebSocket that simulates network delays
        class RealisticWebSocketMock:
            def __init__(self, simulate_slow_close=False):
                self.closed = False
                self.simulate_slow_close = simulate_slow_close
                self._send_delay = 0.01  # Small network delay
                
            async def send(self, data):
                await asyncio.sleep(self._send_delay)
                if isinstance(data, str):
                    msg = json.loads(data)
                    if msg.get("type") == "Terminate":
                        print("  → Sent termination message to server")
                        # Simulate server taking time to respond
                        if self.simulate_slow_close:
                            await asyncio.sleep(2.0)  # Slow server response
                        
            async def close(self):
                if self.simulate_slow_close:
                    await asyncio.sleep(0.5)  # Slow close
                self.closed = True
                print("  → WebSocket closed")
                
            async def recv(self):
                # Simulate long-running receive that gets cancelled
                try:
                    await asyncio.sleep(10)  # Would wait forever if not cancelled
                    return '{"type": "Turn", "transcript": "test"}'
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
            
            # Reset service state
            service._connected = True
            service._websocket = RealisticWebSocketMock(simulate_slow_close=slow_close)
            service._audio_buffer = bytearray(b"some audio data")
            
            # Create realistic receive task that actually runs
            async def realistic_receive_task():
                try:
                    while service._connected:
                        try:
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
            
            # Mock the cancel_task method to work properly
            async def mock_cancel_task(task):
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            service.cancel_task = mock_cancel_task
            
            # Test forced shutdown (CancelFrame scenario)
            print("  Testing forced shutdown...")
            start_time = time.time()
            await service._disconnect(force=True)
            forced_time = time.time() - start_time
            
            # Reset for graceful test
            service._connected = True
            service._websocket = RealisticWebSocketMock(simulate_slow_close=slow_close)
            service._receive_task = asyncio.create_task(realistic_receive_task())
            
            # Test graceful shutdown (EndFrame scenario) 
            print("  Testing graceful shutdown...")
            start_time = time.time()
            await service._disconnect(force=False)
            graceful_time = time.time() - start_time
            
            print(f"  Results for {scenario_name}:")
            print(f"    - Forced: {forced_time:.2f}s")
            print(f"    - Graceful: {graceful_time:.2f}s")
            
            # Verify expectations
            if scenario_name == "Fast server response":
                if forced_time < 0.1 and graceful_time < 1.5:
                    print(f"    ✅ PASS: Both shutdowns were fast enough")
                else:
                    print(f"    ❌ FAIL: Shutdowns too slow")
            else:  # Slow server response
                if forced_time < 0.1:  # Forced should still be fast
                    print(f"    ✅ PASS: Forced shutdown bypassed slow server")
                else:
                    print(f"    ❌ FAIL: Forced shutdown was too slow")
                    
                if graceful_time < 2.0:  # Should timeout quickly instead of waiting
                    print(f"    ✅ PASS: Graceful shutdown timed out appropriately")
                else:
                    print(f"    ❌ FAIL: Graceful shutdown waited too long")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Realistic AssemblyAI STT Shutdown Test")
    print("=" * 50)
    
    success = asyncio.run(test_realistic_assemblyai_shutdown())
    exit(0 if success else 1)

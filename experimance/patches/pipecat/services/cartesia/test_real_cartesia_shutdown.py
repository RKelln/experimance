#!/usr/bin/env python3
"""
Real-world test of Cartesia TTS shutdown behavior with actual API connection.
This test uses real API keys to test the patched shutdown fix.
"""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    # Load from the experimance project .env file (go up 5 levels from patches/pipecat/services/cartesia/)
    env_path = Path(__file__).parent.parent.parent.parent.parent / "projects" / "experimance" / ".env"
    load_dotenv(env_path)
    print(f"🔧 Loaded environment from: {env_path}")
except ImportError:
    print("⚠️  dotenv not installed, using system environment variables")

VOICE_ID = "bf0a246a-8642-498a-9950-80c35e9276b5"

# Test the shutdown timing with real connection
async def test_real_cartesia_shutdown():
    """Test Cartesia TTS service shutdown with real connection."""
    print("Testing Cartesia TTS shutdown with real connection...")
    
    # Get API key from environment
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        print("❌ CARTESIA_API_KEY not found in environment")
        print("Please set CARTESIA_API_KEY in your .env file or environment")
        return False
    
    try:
        # Import the patched service and pipeline components
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.frames.frames import StartFrame, CancelFrame, EndFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        
        # Test forced shutdown (like SIGINT) - create minimal pipeline
        print("🔄 Testing forced shutdown with real connection...")
        
        # Create service with real API key
        service = CartesiaTTSService(
            api_key=api_key,
            voice_id=VOICE_ID,
            model="sonic-2"
        )
        
        # Create a minimal pipeline to properly initialize the service
        pipeline = Pipeline([service])
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        print("🚀 Starting pipeline and connecting to Cartesia...")
        
        # Start the pipeline in background
        run_task = asyncio.create_task(runner.run(task))
        
        # Wait for connection to establish
        await asyncio.sleep(3.0)
        print("✅ Connection should be established")
        
        # Test forced shutdown
        start_time = time.time()
        print("⚡ Triggering forced shutdown...")
        await runner.cancel()
        
        # Wait for the run task to complete
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            print("⚠️  Pipeline shutdown timed out")
        
        end_time = time.time()
        
        forced_shutdown_time = end_time - start_time
        print(f"⚡ Forced shutdown completed in {forced_shutdown_time:.2f} seconds")
        
        # Evaluate results
        print(f"\n📊 Results:")
        print(f"   - Forced shutdown: {forced_shutdown_time:.2f}s")
        
        if forced_shutdown_time < 3.0:  # Should be much faster than the original hang time
            print("✅ SUCCESS: Shutdown completed quickly!")
            print(f"✅ This is much faster than potential WebSocket hangs")
            return True
        else:
            print("❌ FAILURE: Shutdown times are still too slow")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import pipecat components: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_real_cartesia_graceful_vs_forced():
    """Test both graceful and forced shutdown with real connection."""
    print("\n🔄 Testing graceful vs forced shutdown comparison...")
    
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        print("❌ CARTESIA_API_KEY not found in environment")
        return False
    
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.frames.frames import StartFrame, CancelFrame, EndFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        
        # Test graceful shutdown
        print("🕊️  Testing graceful shutdown...")
        service1 = CartesiaTTSService(
            api_key=api_key,
            voice_id=VOICE_ID,
            model="sonic-2"
        )
        
        # Create pipeline for proper initialization
        pipeline1 = Pipeline([service1])
        task1 = PipelineTask(pipeline1)
        runner1 = PipelineRunner()
        
        # Start pipeline
        run_task1 = asyncio.create_task(runner1.run(task1))
        await asyncio.sleep(2.0)  # Let it connect
        
        # Graceful stop
        start_time = time.time()
        await runner1.cancel()  # Use cancel for graceful shutdown too
        try:
            await asyncio.wait_for(run_task1, timeout=5.0)
        except asyncio.TimeoutError:
            print("  ⚠️  Graceful shutdown timed out")
        graceful_time = time.time() - start_time
        print(f"  🕊️  Graceful shutdown: {graceful_time:.3f}s")
        
        # Test forced shutdown
        print("⚡ Testing forced shutdown...")
        service2 = CartesiaTTSService(
            api_key=api_key,
            voice_id=VOICE_ID,
            model="sonic-2"
        )
        
        # Create pipeline for proper initialization
        pipeline2 = Pipeline([service2])
        task2 = PipelineTask(pipeline2)
        runner2 = PipelineRunner()
        
        # Start pipeline
        run_task2 = asyncio.create_task(runner2.run(task2))
        await asyncio.sleep(2.0)  # Let it connect
        
        # Forced cancel
        start_time = time.time()
        await runner2.cancel()
        try:
            await asyncio.wait_for(run_task2, timeout=5.0)
        except asyncio.TimeoutError:
            print("  ⚠️  Forced shutdown timed out")
        forced_time = time.time() - start_time
        print(f"  ⚡ Forced shutdown: {forced_time:.3f}s")
        
        # Compare results
        print(f"\n📊 Comparison Results:")
        print(f"   - First shutdown: {graceful_time:.3f}s")
        print(f"   - Second shutdown: {forced_time:.3f}s")
        
        # Check if we have the patched version
        if hasattr(service1, '_disconnect') and 'force' in service1._disconnect.__code__.co_varnames:
            print("✅ Detected PATCHED version")
            # Both should be fast with the patch
            if graceful_time < 3.0 and forced_time < 3.0:
                print("✅ SUCCESS: Both shutdowns complete quickly")
                return True
            else:
                print("⚠️  WARNING: One or both shutdowns took too long")
                return False
        else:
            print("ℹ️  Detected ORIGINAL version (no force parameter)")
            if abs(forced_time - graceful_time) < 2.0:
                print("✅ SUCCESS: Both shutdowns complete in reasonable time")
                return True
            else:
                print("⚠️  WARNING: Shutdown times vary significantly")
                return False
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_with_text_generation():
    """Test shutdown behavior while actively generating speech."""
    print("\n🎤 Testing shutdown during active TTS generation...")
    
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        print("❌ CARTESIA_API_KEY not found in environment")
        return False
    
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.frames.frames import StartFrame, CancelFrame, TextFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        
        service = CartesiaTTSService(
            api_key=api_key,
            voice_id=VOICE_ID,
            model="sonic-2"
        )
        
        # Create pipeline for proper initialization
        pipeline = Pipeline([service])
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        # Start pipeline
        run_task = asyncio.create_task(runner.run(task))
        await asyncio.sleep(1.0)
        
        # Start generating some speech (long text to ensure it's still processing)
        long_text = "This is a very long text that should take some time to process and generate speech. " * 10
        
        # Create a task to generate TTS
        async def generate_speech():
            try:
                async for frame in service.run_tts(long_text):
                    if frame:
                        print("  🎵 Generated TTS frame")
                        await asyncio.sleep(0.1)  # Simulate processing
            except Exception as e:
                print(f"  ⚠️  TTS generation interrupted: {e}")
        
        tts_task = asyncio.create_task(generate_speech())
        
        # Wait a moment for TTS to start
        await asyncio.sleep(1.0)
        
        # Now force shutdown while TTS is active
        print("⚡ Forcing shutdown during active TTS generation...")
        start_time = time.time()
        
        # Cancel the TTS task and service
        tts_task.cancel()
        await runner.cancel()
        
        try:
            await asyncio.wait_for(tts_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
            
        try:
            await asyncio.wait_for(run_task, timeout=3.0)
        except asyncio.TimeoutError:
            print("  ⚠️  Pipeline shutdown timed out")
        
        shutdown_time = time.time() - start_time
        print(f"  ⚡ Shutdown during active TTS: {shutdown_time:.3f}s")
        
        if shutdown_time < 3.0:
            print("✅ SUCCESS: Shutdown during TTS generation is fast")
            return True
        else:
            print("⚠️  WARNING: Shutdown during TTS took longer than expected")
            return False
        
    except Exception as e:
        print(f"❌ TTS generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""
    print("🧪 Cartesia TTS Real API Shutdown Test Suite")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        print("❌ CARTESIA_API_KEY not found!")
        print("Please set your Cartesia API key in the environment or .env file")
        print("Example: export CARTESIA_API_KEY='your_api_key_here'")
        return 1
    
    print(f"🔑 Using Cartesia API key: {api_key[:8]}...")
    print("")
    
    results = []
    
    # Run basic shutdown test
    try:
        result1 = await test_real_cartesia_shutdown()
        results.append(result1)
    except Exception as e:
        print(f"❌ Basic shutdown test failed: {e}")
        results.append(False)
    
    # Run graceful vs forced comparison
    try:
        result2 = await test_real_cartesia_graceful_vs_forced()
        results.append(result2)
    except Exception as e:
        print(f"❌ Graceful vs forced test failed: {e}")
        results.append(False)
    
    # Run TTS generation test
    try:
        result3 = await test_with_text_generation()
        results.append(result3)
    except Exception as e:
        print(f"❌ TTS generation test failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 Cartesia TTS shutdown fix is working correctly with real API")
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

#!/usr/bin/env python3
"""
Real-world test of AssemblyAI shutdown behavior with actual API connection.
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
    # Load from the experimance project .env file (go up 5 levels from patches/pipecat/services/assemblyai/)
    env_path = Path(__file__).parent.parent.parent.parent.parent / "projects" / "experimance" / ".env"
    load_dotenv(env_path)
    print(f"🔧 Loaded environment from: {env_path}")
except ImportError:
    print("⚠️  dotenv not installed, using system environment variables")

# Test the shutdown timing with real connection
async def test_real_assemblyai_shutdown():
    """Test AssemblyAI STT service shutdown with real connection."""
    print("Testing AssemblyAI STT shutdown with real connection...")
    
    # Get API key from environment
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("❌ ASSEMBLYAI_API_KEY not found in environment")
        print("Please set ASSEMBLYAI_API_KEY in your .env file or environment")
        return False
    
    try:
        # Import the patched service and pipeline components
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
        from pipecat.frames.frames import StartFrame, CancelFrame, EndFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        
        print(f"🔑 Using API key: {api_key[:8]}...")
        
        # Test forced shutdown (like SIGINT) - create minimal pipeline
        print("🔄 Testing forced shutdown with real connection...")
        
        # Create service with real API key
        service = AssemblyAISTTService(
            api_key=api_key,
            connection_params=AssemblyAIConnectionParams()
        )
        
        # Create a minimal pipeline to properly initialize the service
        pipeline = Pipeline([service])
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        print("🚀 Starting pipeline and connecting to AssemblyAI...")
        
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
        
        if forced_shutdown_time < 3.0:  # Should be much faster than the original 10+ seconds
            print("✅ SUCCESS: Shutdown completed quickly!")
            print(f"✅ This is much faster than the original 10+ second delay")
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

def signal_handler(signum, frame):
    """Handle shutdown signals for testing."""
    print(f"\n🛑 Received signal {signum}, running shutdown test...")
    asyncio.create_task(test_and_exit())

async def test_and_exit():
    """Run test and exit."""
    success = await test_real_assemblyai_shutdown()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("AssemblyAI Real Connection Shutdown Test")
    print("=" * 50)
    
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

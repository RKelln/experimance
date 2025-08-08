# Feed the Fires Integration Testing

This directory contains test scripts for validating the Feed the Fires project's service integration, specifically testing the story-to-image pipeline using real services.

## Test Scripts

### 1. Full Integration Test (`test_fire_integration.py`)

Comprehensive test that validates the complete pipeline:
- Sends `StoryHeard` messages to core service
- Monitors image generation from image_server
- Monitors display updates from display service
- Uses test data from `services/core/tests/test_stories.json`
- Provides detailed reporting on success/failure rates

### 2. Quick Test (`test_fire_quick.py`)

Simple script for manual testing and debugging:
- Send individual `StoryHeard` messages
- Interactive mode for custom stories
- Predefined test stories
- Immediate feedback

## Prerequisites

### Environment Setup
```bash
# Set the project environment
export PROJECT_ENV=fire
```

### Required Services
Before running tests, start these services in separate terminals:

```bash
# Terminal 1: Core service
export PROJECT_ENV=fire
cd /path/to/experimance
uv run -m fire_core

# Terminal 2: Image server
export PROJECT_ENV=fire  
cd /path/to/experimance
uv run -m image_server

# Terminal 3: Display service
export PROJECT_ENV=fire
cd /path/to/experimance  
uv run -m experimance_display
```

## Usage

### Full Integration Test

```bash
export PROJECT_ENV=fire
uv run python utils/tests/test_fire_integration.py
```

This test will:
1. Load test stories from JSON file
2. Send `StoryHeard` messages to core (via port 5557 - agent port)
3. Monitor for `ImageReady` messages (port 5556 - image_results)
4. Monitor for `DisplayMedia` messages (port 5558 - events)
5. Report success rates for the complete pipeline

### Quick Test Options

```bash
export PROJECT_ENV=fire

# Interactive menu
uv run python utils/tests/test_fire_quick.py

# Send a custom story directly
uv run python utils/tests/test_fire_quick.py "I remember the old oak tree in my backyard"
```

## Network Configuration

The tests use the following ZMQ connections based on `DEFAULT_PORTS`:

- **Publisher → Core Subscriber**: Port 5557 (`agent`)
  - Sends `StoryHeard` messages
- **Image Results Subscriber**: Port 5556 (`image_results`) 
  - Monitors `ImageReady` messages
- **Events Subscriber**: Port 5558 (`events`)
  - Monitors `DisplayMedia` messages

## Test Data Format

The integration test uses `services/core/tests/test_stories.json` with this structure:

```json
[
    {
        "context": "LLM: \"Hi there!...\"\nUser: \"Um, last summer I sat by the canals...\"",
        "prompt": "Venice canal at sunset in summer...",
        "negative_prompt": "no tourists, no modern boats"
    }
]
```

## Expected Flow

1. **StoryHeard** → Core Service
   - Core receives story from user
   - Core processes with LLM to generate image prompt
   
2. **RenderRequest** → Image Server
   - Core sends image generation request
   - Image server generates image
   
3. **ImageReady** → Core Service  
   - Image server notifies completion
   - Core receives generated image path
   
4. **DisplayMedia** → Display Service
   - Core sends display instruction
   - Display service shows the image

## Monitoring and Debugging

### Log Levels
Both scripts use INFO level logging by default. For more detail:

```python
# In the script, change:
logger = setup_logger(__name__, level=logging.DEBUG)
```

### Service Logs
Monitor individual service logs in their terminals to see:
- Core: Story processing, LLM interaction, image requests
- Image Server: Image generation progress, completion
- Display: Media display commands

### Network Issues
If tests fail to connect:
1. Verify all services are running
2. Check port conflicts: `netstat -tlnp | grep :555`
3. Verify PROJECT_ENV is set in all terminals
4. Check firewall settings if running across machines

## Troubleshooting

### "Could not import schemas" Error
- Ensure `PROJECT_ENV=fire` is set
- Verify you're in the experimance project root
- Check that fire project files exist in `projects/fire/`

### No Messages Received
- Confirm all three services are running and healthy
- Check service logs for errors during startup
- Verify ZMQ port configuration matches
- Try the quick test first to isolate issues

### Partial Pipeline Success
- Image generated but no display: Check display service logs
- No image generated: Check image_server logs and configuration
- Stories not processed: Check core service LLM configuration

## Test Results Interpretation

### Integration Test Output
```
Total stories sent: 4
Images received: 4/4
Display messages received: 3/4  
Complete pipeline success: 3/4
```

- **Images received**: Core → Image Server working
- **Display messages**: Image Server → Display working  
- **Complete pipeline**: End-to-end functionality

A healthy system should achieve 90%+ success rates, with occasional failures due to:
- LLM processing issues
- Image generation timeouts
- Network latency

# Feed the Fires Core Service

The core orchestration service for the Feed the Fires interactive art installation. This service manages the complete pipeline from audience stories and real-time conversations to immersive panoramic visualizations with intelligent interruption and request queueing.

## Quick Start

```bash
# Set project environment
export PROJECT_ENV=fire

# Install the service (from experimance root)
uv pip install -e services/core

# Run the service
uv run -m fire_core

# Test with CLI utility
uv run -m fire_core.cli --conversation forest_memories --delay 2
```

## What it does

1. **Listens for Stories**: Receives `StoryHeard` messages from the agent service
2. **Processes Live Transcripts**: Accumulates `TranscriptUpdate` messages for real-time conversation analysis
3. **Analyzes Content**: Uses LLM to infer environmental settings and build rich image prompts
4. **Smart Interruption**: Intelligently manages overlapping requests with "base images always complete" policy
5. **Generates Base Images**: Creates initial panoramic visualizations from enhanced prompts
6. **Creates Tiles**: Generates high-resolution tiles for seamless display
7. **Queues Requests**: Manages multiple image generation requests with proper prioritization

## State Machine

### Core States
- **Idle**: Initial state, waiting for first input
- **Listening**: Ready to receive stories, transcripts, and location updates
- **BaseImage**: Generating base panorama image (cannot be interrupted)
- **Tiles**: Generating high-resolution tiles (can be cancelled for new requests)

### Request States  
- **QUEUED**: Request created and waiting to be processed
- **PROCESSING_LLM**: LLM analyzing content (can be interrupted)
- **WAITING_BASE**: Base image being generated (protected from interruption)
- **BASE_READY**: Base image completed and displayed
- **WAITING_TILES**: Tile images being generated (tiles can be cancelled)
- **COMPLETED**: All processing finished
- **CANCELLED**: Request was interrupted or discarded

## Configuration

Main config file: `projects/fire/core.toml`

Key settings:
- **Panorama dimensions**: Base image size before mirroring
- **Tile constraints**: Max size, overlap, megapixel limits
- **LLM settings**: Provider, model, timeouts
- **ZMQ ports**: Communication endpoints

## Architecture

### Smart Interruption System

**"Base Images Always Complete" Policy**
- Base panorama images never get cancelled once generation starts
- Tile generation can be interrupted for higher priority requests
- LLM processing can be cancelled when new transcripts arrive
- This ensures responsive behavior while minimizing wasted computational work

**Request Priority System**
1. **New transcript-based requests** take priority over queued requests
2. **Running base images** always complete but future tiles get cancelled
3. **LLM processing** can be interrupted but allows graceful completion
4. **Queue management** ensures fair processing of accumulated requests

### Components

- **Transcript Accumulator**: Collects streaming conversation updates
- **LLM Processing**: Background analysis using ActiveRequest state management
- **Request Queue**: FIFO queue with intelligent interruption capabilities
- **Tiler**: Calculates optimal tiling strategy for seamless display
- **State Machine**: Orchestrates the complete pipeline with proper cancellation

### Message Flow

#### Story-based Flow
```
StoryHeard → LLM Analysis → Base Image Request → Base Image Ready →
Tile Requests → Tile Images Ready → Complete
```

#### Transcript-based Flow  
```
TranscriptUpdate → Accumulator → Background LLM → Smart Interruption →
Queue Management → Base Image → Tiles → Complete
```

#### Interruption Flow
```
New Transcript → Cancel Old LLM → Create New Request → Smart Queue →
Base Completes → New Base Starts → Enhanced Prompt
```

### Tiling Strategy

- Minimize number of tiles while staying under megapixel limits
- Ensure minimum overlap for seamless blending
- Apply edge masking for smooth composition
- Calculate optimal positioning for display service

## Environment Variables

- `PROJECT_ENV=fire`: Enable Feed the Fires mode
- `OPENAI_API_KEY`: API key for OpenAI LLM

## Development

### Testing with CLI Utility

The service includes a comprehensive CLI testing utility:

```bash
# Interactive menu mode
uv run -m fire_core.cli --interactive

# Send conversation sequences
uv run -m fire_core.cli --conversation forest_memories --delay 2
uv run -m fire_core.cli --conversation desert_journey --delay 1.5

# Send single story
uv run -m fire_core.cli --story "I walked through ancient redwoods..."

# Send direct prompt (debug mode)  
uv run -m fire_core.cli --prompt "mystical forest with golden light"

# Send single transcript
uv run -m fire_core.cli --transcript "The forest felt magical" --speaker-id user
```

### Conversation Testing

Test realistic conversation flows with built-in scenarios:
- `forest_memories`: 6-message conversation building forest imagery
- `desert_journey`: 7-message conversation creating desert landscapes  
- `mountain_reflection`: 7-message conversation about alpine lakes

Each conversation demonstrates:
- Progressive prompt enhancement as more details emerge
- Smart interruption when new messages arrive during processing
- LLM decision-making for insufficient vs ready content
- Queue management with proper request prioritization

### Debug Commands

```bash
# Run with mock LLM (no API key needed)
uv run -m fire_core --llm-provider mock

# Debug mode with detailed logging
uv run -m fire_core --log-level DEBUG

# List available test content
uv run -m fire_core.cli --list-conversations
uv run -m fire_core.cli --list-stories
uv run -m fire_core.cli --list-prompts
```

### Adding New Biomes

1. Add biome to `projects/fire/schemas.py`
2. Update biome templates in `prompt_builder.py`
3. Test with mock stories

### Adding New LLM Providers

1. Implement `LLMProvider` interface in `llm.py`
2. Add provider configuration options
3. Update `LLMManager` factory

## Integration

### Message Types
- **Input**: 
  - `StoryHeard` from agent service (complete stories)
  - `TranscriptUpdate` from agent service (real-time conversation)
  - `UpdateLocation` from agent service (location changes)
  - Debug prompts via updates channel
- **Output**: 
  - `RenderRequest` to image_server service
  - `DisplayMedia` to display service
  - Health status and metrics

### ZMQ Communication

**Agent Channel (5557)**
- Binds to receive stories and transcripts from agent service
- Handles `StoryHeard` and `TranscriptUpdate` messages
- Non-blocking message reception with background processing

**Updates Channel (5556)**  
- Binds to receive direct prompts and status updates
- Supports debug/testing scenarios
- Used by CLI utility for development

**Controller Channel (5555)**
- Publishes `DisplayMedia` messages to display service
- Sends `RenderRequest` work to image_server service
- Manages worker response handling

### Real-time Behavior

**Transcript Processing**
- Accumulates conversation messages in sessions
- Triggers LLM analysis when sufficient user content available
- Ignores agent messages to avoid processing AI responses
- Maintains conversation context across multiple exchanges

**Smart Queueing**
- New transcript requests interrupt ongoing LLM processing
- Base image generation always completes (artistic integrity)
- Tile generation gets cancelled for responsive updates
- Queue processes accumulated requests after timeouts

**Error Recovery**
- Graceful handling of image server unavailability  
- Automatic retry and queue processing after timeouts
- ZMQ connection resilience with proper cleanup
- Comprehensive logging for debugging and monitoring

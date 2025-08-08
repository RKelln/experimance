# Sohkepayin Core Service

The core orchestration service for the Sohkepayin interactive art installation. This service manages the complete pipeline from audience stories to immersive panoramic visualizations.

## Quick Start

```bash
# Set project environment
export PROJECT_ENV=sohkepayin

# Install the service (from experimance root)
uv pip install -e services/core

# Run the service
uv run -m sohkepayin_core
```

## What it does

1. **Listens for Stories**: Receives `StoryHeard` messages from the agent service
2. **Analyzes Content**: Uses LLM to infer environmental settings (biome, emotion, atmosphere)
3. **Generates Base Image**: Creates initial panoramic visualization
4. **Creates Tiles**: Generates high-resolution tiles for seamless display
5. **Sends to Display**: Delivers images with positioning to the display service

## State Machine

- **Idle**: Initial state, waiting for first story
- **Listening**: Ready to receive stories and location updates
- **BaseImage**: Generating base panorama image
- **Tiles**: Generating high-resolution tiles

## Configuration

Main config file: `projects/sohkepayin/core.toml`

Key settings:
- **Panorama dimensions**: Base image size before mirroring
- **Tile constraints**: Max size, overlap, megapixel limits
- **LLM settings**: Provider, model, timeouts
- **ZMQ ports**: Communication endpoints

## Architecture

### Components

- **LLM Manager**: Story analysis using OpenAI GPT-4o
- **Prompt Builder**: Converts analysis to image generation prompts
- **Tiler**: Calculates optimal tiling strategy
- **State Machine**: Orchestrates the complete pipeline

### Message Flow

```
StoryHeard → LLM Analysis → Base Image Request → Base Image Ready →
Tile Requests → Tile Images Ready → Complete
```

### Tiling Strategy

- Minimize number of tiles while staying under megapixel limits
- Ensure minimum overlap for seamless blending
- Apply edge masking for smooth composition
- Calculate optimal positioning for display service

## Environment Variables

- `PROJECT_ENV=sohkepayin`: Enable Sohkepayin mode
- `OPENAI_API_KEY`: API key for OpenAI LLM

## Development

### Testing

```bash
# Run with mock LLM (no API key needed)
uv run -m sohkepayin_core --llm-provider mock

# Debug mode
uv run -m sohkepayin_core --log-level DEBUG
```

### Adding New Biomes

1. Add biome to `projects/sohkepayin/schemas.py`
2. Update biome templates in `prompt_builder.py`
3. Test with mock stories

### Adding New LLM Providers

1. Implement `LLMProvider` interface in `llm.py`
2. Add provider configuration options
3. Update `LLMManager` factory

## Integration

- **Input**: `StoryHeard`, `UpdateLocation` from agent service
- **Output**: `RenderRequest` to image_server, `DisplayMedia` to display service
- **Ports**: See `projects/sohkepayin/core.toml` for ZMQ configuration

# Agent Service Implementation Plan

## Overview
Create a modular agent service that handles speech-to-speech conversation with the audience, integrates webcam feeds for audience detection and scene understanding, and provides tool calling capabilities for controlling other services.

## Architecture Components

### 1. Main Service (`agent.py`)
- [x] Extend BaseService using PubSubService composition
- [x] Bind publisher to port 5557 (DEFAULT_PORTS["agent"]) 
- [x] Subscribe to events port 5555 (DEFAULT_PORTS["events"]) from core
- [x] Manage agent backend lifecycle
- [ ] Handle webcam integration
- [ ] Coordinate transcript display

### 2. Configuration (`config.py`)
- [x] Extend BaseServiceConfig
- [x] Agent backend selection (livekit, hume, ultravox)
- [x] Webcam configuration
- [x] Vision model configuration
- [x] ZMQ pub/sub configuration
- [x] Transcript display settings

### 3. Modular Agent Backends (`backends/`)

#### Base Interface (`backends/base.py`)
- [x] Abstract AgentBackend class
- [x] Standardized lifecycle methods (start, stop, connect)
- [x] Conversation handling interface
- [x] Tool calling interface
- [x] Transcript access interface
- [x] Event callbacks for agent actions

#### LiveKit Backend (`backends/livekit_backend.py`)
- [ ] Implement AgentBackend interface
- [ ] Integrate existing livekit_prototype.py code
- [ ] Handle LiveKit room management
- [ ] Implement tool calling for biome control
- [ ] Transcript extraction and management
- [ ] Agent conversation state management

#### Future Backends
- [ ] Hume.ai backend (`backends/hume_backend.py`)
- [ ] Ultravox backend (`backends/ultravox_backend.py`)

### 4. Vision Processing (`vision/`)

#### Webcam Manager (`vision/webcam.py`)
- [ ] Camera capture and frame processing
- [ ] Audience detection using computer vision
- [ ] Periodic image capture for analysis
- [ ] Integration with vision language models

#### Vision Language Model (`vision/vlm.py`)
- [ ] Local VLM integration (e.g., llama-vision, moondream)
- [ ] Scene description generation
- [ ] Audience analysis
- [ ] Context awareness for agent interactions

### 5. Transcript Management (`transcript/`)

#### Transcript Handler (`transcript/handler.py`)
- [ ] Real-time transcript processing
- [ ] Display text generation and formatting
- [ ] Text overlay timing and management
- [ ] Speaker attribution and styling

### 6. Tools Integration (`tools/`)

#### Biome Control (`tools/biome.py`)
- [ ] AgentControlEvent generation for SuggestBiome
- [ ] Integration with agent tool calling
- [ ] Validation of biome suggestions

#### Audience Detection (`tools/audience.py`)
- [ ] AgentControlEvent generation for AudiencePresent
- [ ] Computer vision integration
- [ ] Presence state management

## Message Flow Implementation

### Publishing Messages
1. **AgentControlEvent** → Core (port 5557)
   - SuggestBiome: When agent decides to change biome
   - AudiencePresent: When webcam detects audience presence/absence
   - SpeechDetected: When agent is actively speaking

2. **DisplayText/RemoveText** → Display (direct push to display service)
   - Real-time transcript display
   - Agent responses and prompts
   - Conversation context

### Subscribing to Messages
1. **Events** ← Core (port 5555)
   - SpaceTimeUpdate: Current era/biome state
   - System status updates
   - Interaction state changes

## Implementation Phases

### Phase 1: Core Service Structure ✅ COMPLETE
- [x] Create agent.py with BaseService + PubSubService
- [x] Create config.py with agent-specific configuration
- [x] Set up ZMQ pub/sub on correct ports
- [x] Create modular backend interface

### Phase 2: LiveKit Integration 🔄
- [ ] Migrate livekit_prototype.py to backends/livekit_backend.py
- [ ] Implement AgentBackend interface for LiveKit
- [ ] Add tool calling for biome control
- [ ] Integrate transcript handling

### Phase 3: Vision Integration 🔄
- [ ] Implement webcam capture and processing
- [ ] Add audience detection capabilities
- [ ] Integrate local vision language model
- [ ] Connect vision insights to agent conversations

### Phase 4: Transcript Display 🔄
- [ ] Real-time transcript processing
- [ ] DisplayText message generation
- [ ] Text styling and timing management
- [ ] Speaker attribution

### Phase 5: Tool Integration 🔄
- [ ] Biome suggestion tool implementation
- [ ] Audience detection event publishing
- [ ] Speech state tracking and publishing

### Phase 6: Testing & Integration 📋
- [ ] Unit tests for all components
- [ ] Integration tests with core/display services
- [ ] End-to-end conversation testing
- [ ] Performance optimization

### Phase 7: Additional Backends 📋
- [ ] Research and implement Hume.ai backend
- [ ] Research and implement Ultravox backend
- [ ] Backend comparison and selection logic

## Technical Considerations

### ZMQ Patterns
- Agent binds as publisher on port 5557 (others subscribe)
- Agent subscribes to events on port 5555 from core
- Direct push messages to display service for text overlays

### Error Handling
- Graceful backend failures with automatic retry
- Vision processing error recovery
- Network disconnection handling for remote backends

### Performance
- Efficient webcam frame processing
- Minimal latency for real-time conversation
- Async processing for all I/O operations

### Configuration Management
- Environment-based backend selection
- Runtime configuration for vision models
- Flexible tool enabling/disabling

## Dependencies to Add
- OpenCV for webcam processing
- Local VLM library (e.g., transformers, ollama)
- Computer vision libraries for audience detection
- Additional agent backend dependencies as needed

## Files to Create
```
services/agent/src/experimance_agent/
├── agent.py                    # Main service class
├── config.py                   # Configuration definitions
├── backends/
│   ├── __init__.py
│   ├── base.py                 # Abstract backend interface
│   └── livekit_backend.py      # LiveKit implementation
├── vision/
│   ├── __init__.py
│   ├── webcam.py              # Camera capture and processing
│   └── vlm.py                 # Vision language model
├── transcript/
│   ├── __init__.py
│   └── handler.py             # Transcript processing and display
├── tools/
│   ├── __init__.py
│   ├── biome.py               # Biome control tools
│   └── audience.py            # Audience detection tools
└── __main__.py                # CLI entry point
```

This plan provides a comprehensive, modular approach to building the agent service with clear separation of concerns and extensibility for future agent backends.

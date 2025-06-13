# Experimance Core Service Design

## Overview

The Core service is the central coordinator for the Experimance interactive art installation. It manages the experience state machine, processes depth camera data, coordinates with all other services via ZMQ, and drives the narrative progression through different eras of human development.

## Architecture

### Service Type
- **Publisher/Subscriber Service**: Extends `ZMQPublisherSubscriberService` from `experimance_common.service`
- **Central Coordinator**: Publishes events to all services and subscribes to coordination responses
- **Async Processing**: Built on `asyncio` for concurrent operations

### Key Responsibilities

1. **State Management**: Era progression, biome selection, user interaction tracking
2. **Depth Processing**: Integration with depth camera via `depth_finder` module
3. **Event Publishing**: Coordinates all services via ZMQ events
4. **Event Coordination**: Receives status updates and coordinates service interactions
5. **Prompt Generation**: Creates text-to-image prompts via `prompter` module
6. **Audio Tag Extraction**: Derives environmental audio tags from generated prompts
7. **Experience Flow**: Manages timing, transitions, and narrative arc

## State Machine

### Era States
```
Wilderness → Pre-industrial → Early Industrial → Late Industrial → 
Early Modern → Modern → AI/Future ⟷ Post-apocalyptic → Ruins → [drift back to Wilderness]
```

### Core State Variables
- `current_era`: Current era in the timeline
- `current_biome`: Geographic/environmental setting (forest, desert, urban, etc.)
- `user_interaction_score`: Calculated from depth changes and sensor data [0,1]
- `idle_timer`: Time since last meaningful interaction
- `audience_present`: Boolean from agent face detection
- `last_depth_map`: For change detection
- `era_progression_timer`: Controls automatic advancement

### Transition Logic
- **Forward Progression**: Driven by `user_interaction_score` above threshold
- **Idle Drift**: Returns toward Wilderness after `idle_timeout`
- **Agent Influence**: Biome suggestions from conversational agent
- **Branching**: AI era can go to Post-apocalyptic or loop within AI themes

## ZMQ Communication

### Published Events (to `events` channel - tcp://*:5555)

#### EraChanged
```json
{
  "event_type": "EraChanged",
  "timestamp": "2025-06-13T10:30:00Z",
  "era": "modern",
  "biome": "urban",
  "user_interaction_score": 0.7,
  "transition_reason": "user_interaction" | "idle_drift" | "agent_suggestion"
}
```

#### RenderRequest  
```json
{
  "event_type": "RenderRequest",
  "timestamp": "2025-06-13T10:30:00Z",
  "request_id": "uuid-string",
  "prompt": "generated text prompt",
  "negative_prompt": "negative prompt text",
  "depth_map_base64": "base64-encoded-png",
  "era": "modern",
  "biome": "urban",
  "seed": 12345
}
```

#### AudioCommand
```json
{
  "event_type": "AudioCommand",
  "timestamp": "2025-06-13T10:30:00Z",
  "command_type": "spacetime" | "include_tags" | "exclude_tags" | "trigger",
  "era": "modern",
  "biome": "urban",
  "tags_to_include": ["church", "urban", "traffic"],
  "tags_to_exclude": ["birds", "forest"],
  "trigger": "interaction_start" | "interaction_stop" | "transition"
}
```

#### VideoMask (Depth Difference Visualization)
```json
{
  "event_type": "VideoMask",
  "timestamp": "2025-06-13T10:30:00Z",
  "mask_id": "uuid-string",
  "mask_type": "depth_difference",
  "depth_map_base64": "base64-encoded-png",
  "interaction_score": 0.8
}
```

#### IdleStateChanged
```json
{
  "event_type": "IdleStateChanged", 
  "timestamp": "2025-06-13T10:30:00Z",
  "is_idle": true,
  "idle_duration": 45.2
}
```

### Subscribed Events (from `events` channel - tcp://*:5555)

#### ImageReady (from Image Server)
```json
{
  "event_type": "ImageReady",
  "timestamp": "2025-06-13T10:30:00Z",
  "request_id": "uuid-string",
  "image_id": "uuid-string",
  "uri": "file:///path/to/image.png",
  "generation_time_ms": 2500
}
```

#### AgentControl (from Agent Service)
```json
{
  "event_type": "AgentControl",
  "timestamp": "2025-06-13T10:30:00Z",
  "sub_type": "AudiencePresent" | "SuggestBiome" | "ConversationState",
  "audience_present": true,
  "biome_suggestion": "desert",
  "conversation_active": false
}
```

#### AudioStatus (from Audio Service)
```json
{
  "event_type": "AudioStatus",
  "timestamp": "2025-06-13T10:30:00Z",
  "status": "ready" | "transitioning" | "error",
  "active_tags": ["urban", "church", "traffic"],
  "current_era": "modern",
  "current_biome": "urban"
}
```

## Internal Modules

### depth_finder.py
- **Current State**: Existing module, use as-is initially
- **Purpose**: RealSense depth camera processing, hand detection, change detection
- **Integration**: Called as internal module, may be split to separate service later
- **Key Functions**: `depth_generator()`, `detect_difference()`, `simple_obstruction_detect()`

### prompter.py
- **Current State**: Existing module, use as-is initially  
- **Purpose**: Generate text-to-image prompts based on era/biome/location
- **Integration**: Synchronous calls from main service loop
- **Key Classes**: `PromptData`, `StringListData`
- **Data Sources**: JSON files in `/data/` (locations.json, anthropocene.json)
- **Audio Tag Support**: Extract semantic tags from generated prompts for environmental audio

## Audio Tag Extraction

### Tag Extraction Pipeline
1. **Prompt Generation**: Create text-to-image prompt via `prompter` module
2. **Keyword Extraction**: Simple keyword matching against known audio tags
3. **Tag Filtering**: Remove previous era/biome tags and add new ones
4. **Audio Command**: Send `AudioCommand` message with updated tag lists

### Known Audio Tags (from audio config)
- **Biome Tags**: `desert`, `forest`, `urban`, `mountain`, `coastal`, `farmland`
- **Era Tags**: `pre_industrial`, `industrial`, `modern`, `ai_future`, `post_apocalyptic`
- **Feature Tags**: `church`, `bells`, `traffic`, `birds`, `water`, `machinery`, `wind`
- **Activity Tags**: `busy`, `quiet`, `construction`, `farming`, `shipping`

### Tag Extraction Logic
```python
def extract_audio_tags(prompt: str, known_tags: Set[str]) -> List[str]:
    """Extract audio tags from generated prompt using keyword matching."""
    prompt_lower = prompt.lower()
    found_tags = []
    
    for tag in known_tags:
        # Simple keyword matching (can be enhanced later)
        if tag in prompt_lower:
            found_tags.append(tag)
    
    return found_tags

def update_audio_tags(new_tags: List[str], previous_tags: List[str]) -> Tuple[List[str], List[str]]:
    """Calculate tags to include and exclude for audio system."""
    tags_to_include = list(set(new_tags))
    tags_to_exclude = list(set(previous_tags) - set(new_tags))
    
    return tags_to_include, tags_to_exclude
```

## Configuration

### Service Configuration (`config.toml`)
```toml
[experimance_core]
name = "experimance_core"
publish_port = 5555
heartbeat_interval = 3.0

[state_machine]
idle_timeout = 45.0           # seconds before idle drift starts
wilderness_reset = 300.0      # seconds to full reset to wilderness
interaction_threshold = 0.3   # minimum score to trigger era advancement
era_min_duration = 10.0       # minimum time in era before advancement

[depth_processing]
change_threshold = 50         # pixel difference threshold
min_depth = 0.49             # meters
max_depth = 0.56             # meters
resolution = [1280, 720]     # depth camera resolution
output_size = [1024, 1024]   # processed depth map size

[sensors]
gain = 1.8                   # amplification for vibe sensors
osc_port = 8000             # OSC input port for sensor data

[audio]
zmq_address = "tcp://*:5560" # ZMQ address for audio service
tag_config_path = "config/audio_tags.json"  # Known audio tags configuration
interaction_sound_duration = 2.0  # seconds for interaction sound effects

[prompting]
data_path = "data/"
locations_file = "locations.json"
developments_file = "anthropocene.json"
random_strategy = "shuffle"  # or "choice" or null

[persistence]
initial_state_path = "saved_data/default_state.json"
save_state_interval = 30.0   # seconds between state saves
```

### Initial State File (`saved_data/default_state.json`)
```json
{
  "era": "wilderness",
  "biome": "forest", 
  "user_interaction_score": 0.0,
  "idle_timer": 0.0,
  "audience_present": false,
  "era_progression_timer": 0.0,
  "session_start_time": "2025-06-13T10:00:00Z"
}
```

## Service Implementation Pattern

### Class Structure
```python
class ExperimanceCoreService(ZMQPublisherSubscriberService):
    def __init__(self, config_path: str = "config.toml"):
        # Initialize with config, state, and modules
        
    async def start(self):
        # Initialize depth processing, load state, start tasks
        # Register message handlers for subscribed events
        
    async def main_loop(self):
        # Primary event loop: process depth data, update state, publish events
        
    async def depth_processing_task(self):
        # Handle depth camera data and change detection
        
    async def state_machine_task(self):
        # Era progression logic and idle management
        
    async def _handle_image_ready(self, message):
        # Process ImageReady messages for coordination
        
    async def _handle_agent_control(self, message):
        # Process agent control events
        
    async def _handle_audio_status(self, message):
        # Process audio status updates
```

### Key Async Tasks
1. **Main Event Loop**: Coordinates all processing and state management
2. **Depth Processing**: Continuous depth camera monitoring and difference detection
3. **State Machine**: Era progression, idle timers, and interaction scoring
4. **Message Handlers**: Process responses from other services
5. **Audio Tag Management**: Extract and coordinate environmental audio tags
6. **Periodic State Save**: Persistence for crash recovery

### Interaction Sound Management
- **Hand Detection**: Continuous sound while `hand_detected` is true from depth processing
- **Sand Sensors**: Future integration for actual touch detection
- **Interaction Scoring**: Convert depth changes to user interaction score
- **Sound Triggers**: Send `AudioCommand` with `trigger: "interaction_start/stop"`

## Data Flow

```
Depth Camera → depth_finder → Change Detection → user_interaction_score →
State Machine → Era/Biome Update → prompter → Text Prompts →
RenderRequest Event → ZMQ Publish → image_server, display, audio services
```

## Error Handling

### Critical Errors (Fatal)
- Depth camera initialization failure
- ZMQ publisher socket binding failure
- Configuration file parse errors

### Recoverable Errors (Non-Fatal)
- Temporary depth camera disconnection
- Agent service unavailable
- State file corruption (fall back to defaults)
- Prompt generation failures (use fallback prompts)

### Recovery Strategies
- Automatic depth camera reconnection with exponential backoff
- Graceful degradation when subsystems unavailable
- State persistence every 30 seconds for crash recovery
- Health check publishing for monitoring

## Performance Considerations

### Timing Requirements
- Depth processing: 15-30 Hz for responsive interaction
- State updates: 1-5 Hz for smooth era transitions  
- Event publishing: As needed, not on fixed schedule
- Agent responsiveness: < 200ms for natural conversation

### Resource Management
- Async processing prevents blocking operations
- Depth image processing on separate thread if needed
- Memory-efficient depth map storage (PNG compression)
- Configurable processing resolution for performance tuning

## Testing Strategy

### Unit Tests
- State machine logic with mock inputs
- Prompt generation with known data sets
- Configuration loading and validation
- Error handling scenarios

### Integration Tests  
- ZMQ message publishing/receiving
- Depth camera mock data processing
- Agent control event handling
- Full experience flow simulation

### Performance Tests
- Sustained operation over installation runtime (weeks)
- Memory leak detection
- Depth processing latency measurement
- Event publishing throughput

## Future Considerations

### Modularity
- `depth_finder` may become separate service for scaling
- `prompter` could be enhanced with LLM integration
- State persistence could use Redis for multi-instance deployments

### Monitoring
- Prometheus metrics for operational visibility
- Health check endpoints for infrastructure monitoring
- Performance metrics collection and alerting

### Extensibility
- Plugin architecture for new era types
- Dynamic biome loading from external sources
- A/B testing framework for experience variations

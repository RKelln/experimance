# Core Service Architecture

## Overview

The Core Service is the central coordinator for the Experimance installation. It owns the era/biome state machine, processes depth-camera input, generates AI image prompts, and dispatches work to every other service via ZMQ.

There are two separate core implementations that share this package:

- **`experimance_core`** – Depth-camera driven, era-based narrative for the Experimance project.
- **`fire_core`** – Story/transcript driven, panoramic-image pipeline for the Feed the Fires project.

Both follow the same structural pattern: `BaseService` + `ControllerService` composition, Pydantic-validated config, and project-aware TOML loading.

---

## Service Composition Pattern

```
ExperimanceCoreService (BaseService)
  └── ControllerService (from experimance_common.zmq.services)
        ├── Publisher  – broadcasts events on port 5555
        ├── Subscriber – receives events on port 5555
        └── Workers    – push/pull sockets per downstream service
```

The `ControllerService` is configured entirely through `ControllerServiceConfig` from `experimance_common.zmq.config`. Service logic never touches raw ZMQ sockets.

---

## ZMQ Communication

### Port Map

Defined in `libs/common/src/experimance_common/constants_base.py`:

| Channel                  | Port | Direction                    |
|--------------------------|------|------------------------------|
| `events`                 | 5555 | Pub/sub, all services        |
| `agent`                  | 5557 | Agent publisher → all        |
| `image_requests`         | 5564 | Core → image_server (push)   |
| `image_results`          | 5565 | image_server → core (pull)   |

### Published Events (port 5555)

#### `PresenceStatus`
Audience detection state broadcast to all services.

```json
{
  "type": "PresenceStatus",
  "idle": false,
  "present": true,
  "hand": true,
  "voice": false,
  "touch": false,
  "conversation": true,
  "person_count": 1,
  "presence_duration": 45.3,
  "hand_duration": 12.5,
  "voice_duration": 3.2,
  "touch_duration": 0.0,
  "timestamp": "2025-11-10T12:30:00.123456"
}
```

#### `SpaceTimeUpdate`
Era/biome change notification (replaces the old `EraChanged` event).

```json
{
  "type": "SpaceTimeUpdate",
  "era": "modern",
  "biome": "urban",
  "tags": ["urban", "traffic", "city"],
  "timestamp": "2025-11-10T12:30:00Z"
}
```

#### `DisplayMedia`
Instructs the display service to show a generated image.

```json
{
  "type": "DisplayMedia",
  "content_type": "image",
  "request_id": "uuid-string",
  "uri": "file:///path/to/image.png",
  "era": "modern",
  "biome": "urban",
  "fade_in": 0.5,
  "fade_out": 0.3
}
```

#### `CHANGE_MAP` (`MessageType.CHANGE_MAP`)
Real-time interaction mask sent to the display service.

```json
{
  "type": "CHANGE_MAP",
  "change_score": 0.8,
  "has_change_map": true,
  "image_data": "<base64-encoded-binary-mask>",
  "timestamp": "2025-11-10T12:30:00Z",
  "mask_id": "change_map_1699632000123"
}
```

### Pushed Work (image_requests, port 5564)

#### `RenderRequest`
Triggers AI image generation. Sent to `image_server` via push socket.

```json
{
  "type": "RenderRequest",
  "request_id": "uuid-string",
  "era": "modern",
  "biome": "urban",
  "prompt": "A bustling modern cityscape with glass towers...",
  "negative_prompt": "blurry, low quality",
  "seed": 12345,
  "width": 1024,
  "height": 1024,
  "depth_map": {
    "image_data": "<base64-encoded-png>",
    "uri": null
  }
}
```

### Subscribed Events (port 5555)

| Message type       | Source            | Purpose                                  |
|--------------------|-------------------|------------------------------------------|
| `REQUEST_BIOME`    | agent service     | Request a specific biome change          |
| `AUDIENCE_PRESENT` | agent/vision      | Person count from vision system          |
| `SPEECH_DETECTED`  | audio service     | Human or agent speech notification       |

### Pull Responses (image_results, port 5565)

| Message type | Source       | Purpose                          |
|--------------|--------------|----------------------------------|
| `ImageReady` | image_server | Generated image URI and metadata |

---

## State Machine (Experimance project)

### Era Sequence

```
Wilderness → Pre-industrial → Early Industrial → Late Industrial →
Modern (1970s) → Current (2020s) → AI/Future
                                        ↕
                               Post-apocalyptic → Ruins → [back to Wilderness]
```

### Core State Variables

| Variable                 | Description                                            |
|--------------------------|--------------------------------------------------------|
| `current_era`            | Active era in the timeline                             |
| `current_biome`          | Geographic/environmental setting                       |
| `user_interaction_score` | Normalised [0,1] score calculated from depth changes   |
| `idle_timer`             | Seconds since last meaningful interaction              |
| `audience_present`       | Detected by the vision or depth system                 |
| `era_progression_timer`  | Controls automatic advancement                         |

### Transition Rules

- **Forward**: `user_interaction_score` exceeds `state_machine.interaction_threshold` for `state_machine.era_min_duration` seconds.
- **Idle drift**: No interaction for `presence.idle_threshold` seconds → drift back toward Wilderness.
- **Agent influence**: `REQUEST_BIOME` message from agent service can suggest a biome change.
- **Branching**: AI era can transition to Post-apocalyptic or loop internally.

---

## Internal Module Map

### experimance_core (`src/experimance_core/`)

| Module                  | Responsibility                                             |
|-------------------------|------------------------------------------------------------|
| `experimance_core.py`   | Main service class and state machine orchestration         |
| `config.py`             | Pydantic configuration models (`CoreServiceConfig`, etc.)  |
| `realsense_camera.py`   | Low-level RealSense camera wrapper and recovery logic      |
| `depth_processor.py`    | High-level depth processing pipeline (`DepthProcessor`)    |
| `mock_depth_processor.py` | Mock depth generator for development and CI              |
| `depth_visualizer.py`   | OpenCV visualisation helpers for debugging                 |
| `depth_utils.py`        | Utility functions for depth analysis and mask creation     |
| `depth_factory.py`      | Factory that constructs the correct depth processor        |
| `presence.py`           | Audience presence detection with hysteresis               |
| `prompt_generator.py` / `prompter.py` | LLM prompt generation from era/biome state |
| `camera_utils.py`       | Low-level USB / hardware utility functions                 |

### fire_core (`src/fire_core/`)

| Module                  | Responsibility                                                        |
|-------------------------|-----------------------------------------------------------------------|
| `fire_core.py`          | Main service: request queue, LLM orchestration, state monitor        |
| `config.py`             | Pydantic config models for the fire project                           |
| `llm.py`                | LLM provider interface (OpenAI, mock, etc.)                           |
| `llm_prompt_builder.py` | Prompt composition and template substitution                          |
| `audio_manager.py`      | Audio generation / playback orchestration                             |
| `tiler.py`              | Optimal tile strategy for panoramic display                           |
| `prompt_logger.py`      | Structured logging for LLM inputs/outputs                             |
| `cli.py`                | CLI testing utilities for conversations and debug scenarios           |
| `__main__.py`           | Service entry point                                                   |

---

## Error Handling

### Fatal errors (service exits)
- ZMQ publisher socket bind failure
- Configuration file parse errors

### Recoverable errors (logged, service continues)
- Camera disconnection → exponential-backoff retry (see [depth-camera.md](depth-camera.md))
- Agent service unavailable → graceful degradation
- Prompt generation failure → fallback to cached/default prompt
- `ImageReady` timeout → state resets, next request queued

### Health Reporting

`BaseService` writes a JSON health file automatically. Camera state, ZMQ connectivity, and worker status are included. Check `BaseService.health_check()` in `libs/common` for the full schema.

---

## Configuration Files

| File                              | Purpose                                    |
|-----------------------------------|--------------------------------------------|
| `services/core/config.toml`       | Default/example values for the service     |
| `projects/experimance/core.toml`  | Production overrides for experimance       |
| `projects/fire/core.toml`         | Production overrides for fire              |

All config keys map directly to Pydantic model fields. Override any key at runtime with `--section-name-field-name` CLI flags.

---

## Async Task Structure

```
start()
  ├── zmq_service.start()
  ├── _depth_processing_task()   [continuous camera loop]
  ├── _state_machine_task()      [era/presence logic]
  ├── _periodic_task()           [housekeeping]
  └── zmq_service message handlers (registered via add_message_handler)
```

All tasks run on a single `asyncio` event loop. Blocking I/O (e.g. camera hardware calls) is wrapped in `asyncio.to_thread`.

---

## Data Flow

```
RealSense Camera
  → realsense_camera.py   (raw frames, error recovery)
  → depth_processor.py    (change detection, hand detection, DepthFrame)
  → experimance_core.py   (interaction scoring, state transitions)
  → prompter.py           (era/biome → text prompt)
  → ZMQ push              (RenderRequest → image_server)
  → ZMQ pull              (ImageReady ← image_server)
  → ZMQ pub               (DisplayMedia → display service)
                          (SpaceTimeUpdate → all services)
                          (PresenceStatus  → all services)
```

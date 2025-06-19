# Experimance — Technical Design

> **Purpose**
> Provide a concise, implementation‑ready reference for developers & AI coding assistants building *Experimance*: an interactive sand‑table where audience actions steer an AI‑generated satellite narrative of human development.

---

## 1  Experience Overview

A large bowl of white sand has AI generated satellite images projected on it. As the audience interacts with the sand the images update```

Complete schema definitions live at `/schemas`.

---

## 5.1 DISPLAY_MEDIA Message Flow and Design

### Overview

The DISPLAY_MEDIA message type represents a significant architectural improvement in how media content is delivered to the display service. Instead of the display service directly handling ImageReady messages, the core `experimance` service now acts as a coordinator, receiving ImageReady messages and then intelligently deciding how to present the content to the display.

### Architecture Benefits

1. **Centralized Display Logic**: All display decisions happen in one place (core service)
2. **Flexible Content Types**: Support for images, image sequences, and videos 
3. **Rich Transition Support**: Built-in fade effects and transition management
4. **Context Preservation**: Era and biome information travels with display requests
5. **Robust Image Transport**: Uses modern enum-based image transport utilities

### Message Flow

TODO: make into diagram?

core -> RENDER_REQUEST (push)  -> image server
core <- IMAGE_READY (pull)     <- image server
core -> DISPLAY_MEDIA (pubsub) -> display

With future:
core -> TRANSITION_REQUEST (push) -> transition service
core <- TRANSITION_READY (pull)   <- transition serrvice
core -> DISPLAY_MEDIA (pubsub) -> display

and possible:
core -> LOOP_REQUEST (push) -> image server
core <- LOOP_READY (pull)   <- image server
core -> DISPLAY_MEDIA (pubsub) -> display

and additional core events:
core -> CHANGE_MAP (push) -> display

core <- AGENT_CONTROL_EVENT (???) <- agent
agent -> TEXT_OVERLAY (push)      -> display
agent -> REMOVE_TEXT (push)       -> display


### Transition Logic

The core service evaluates several factors when an ImageReady message is received:

1. **Era Progression**: Did the era change from the last image?
2. **Interaction State**: Is there active user interaction that warrants immediate display?
3. **Content Timing**: Should there be a smooth transition between images?

Based on these factors, the core service creates a DisplayMedia message with appropriate:
- **Content Type**: Usually "image" for generated content
- **Transition Effects**: Fade-in/out durations based on interaction context
- **Transport Mode**: Automatically selected based on image size and connection type

### Image Transport Integration

DISPLAY_MEDIA messages leverage the modernized image transport utilities:

- **Enum-Based Transport**: Uses `ImageLoadFormat` enum instead of boolean flags
- **Automatic Mode Selection**: Chooses between FILE_PATH, BASE64, NUMPY, or PIL based on content and destination
- **Size-Based Optimization**: Large images use file paths, smaller ones use direct transport
- **Error Handling**: Robust fallback mechanisms for transport failures

### Example Scenarios

**Scenario 1: Era Change with Transition**
```json
{
  "type": "DisplayMedia",
  "content_type": "image",
  "image_data": "<base64_encoded_data>",
  "fade_in": 1.0,
  "fade_out": 0.5,
  "era": "modern",
  "biome": "coastal",
  "source_request_id": "abc-123"
}
```

**Scenario 2: Direct Image Display (Active Interaction)**
```json
{
  "type": "DisplayMedia", 
  "content_type": "image",
  "uri": "file:///var/cache/experimance/frame_1723.png",
  "fade_in": 0.2,
  "era": "ai_future",
  "biome": "urban"
}
```

---

## 6  Hardware & I/Oessing from untouched wilderness through many eras of human development to the modern day and then beyond to an imagined abstract AI-infused cityscape or post-apocalyptic ruins depending on the audience's interactions with the sand. A depth camera is used to observe the topology of the sand which informs the generated images. Sensors on the sand's contain detect how gently the sand is moved. 

As the audience looks at the piece it sees them with a webcam and introduces itself - the audience can talk to it and it invites them to play with the sand.

Soft music plays from nearby speakers along with environmental audio that matches the imagery in the sand (sounds of animals fading to the sounds of human habitation). 


### Eras
The piece has different states, based of eras of human development:

**Wilderness** (no humans)
**Pre-industrial** (villages)
**Early industrial** (1600s)
**Late industrial** (1800s)
**Early modern** (1900-1960)
**Modern** (2000-2020)
**AI/Future** (2030+)
**Post-apocalyptic**
**Ruins**

### Typical interaction Flow

1. Audience approaches the table → *agent* greets them (presence detected via webcam face‑tracking **or** hand‑over‑sand pickup from the depth camera).
2. Depth camera & sand‑edge sensors detect movement intensity. `experimance` calculates `user_interaction_score`.
3. **Era state machine** steps forward through the timeline (Wilderness ⇢ … ⇢ AI). From the AI era it may either loop within AI, or continue to Post‑apocalyptic and Ruins before drifting back toward Wilderness.
   * `experimance` (using the in-process `prompting` module) builds a text‑to‑image prompt and other necessary data.
   * `experimance` publishes a `RenderRequest` on the `events` bus (see Section 2 & 5).
   * *image\_server* receives the `RenderRequest`, renders the frame (local SDXL or remote fal.ai), and publishes `ImageReady` on the `events` bus.
   * `experimance` receives `ImageReady`, evaluates transition requirements, and publishes `DisplayMedia` to *display* service.
   * *display* receives `DisplayMedia` and updates the projection with appropriate transitions (fade, dissolve, etc.).
   * *audio* cross‑fades music & ambience to match the new era (triggered by `EraChanged` from `experimance`).
4. When idle > *IDLE\_TIMEOUT* (configurable, default = 45 s) the scene drifts back toward Wilderness.
* Processes `AgentControl` messages (e.g., `SuggestBiome`) from the `agent` to influence `biome` state.
* **Depth Difference Visualization**: Generates and publishes `VideoMask` messages with depth difference images for sand interaction feedback.
* **Interaction Sound Management**: Triggers continuous interaction sounds while hands are detected over the sand.it may either loop within AI, or continue to Post‑apocalyptic and Ruins before drifting back toward Wilderness.
   * `experimance` (using the in-process `prompting` module) builds a text‑to‑image prompt and extracts audio tags.
   * `experimance` publishes `RenderRequest` and `AudioCommand` on the unified `events` bus (see Section 2 & 5).
   * *image\_server* receives the `RenderRequest`, renders the frame (local SDXL or remote fal.ai), and publishes `ImageReady` back on the `events` bus.
   * *display* receives `ImageReady` and updates the projection.
   * *audio* receives `AudioCommand` and cross‑fades music & ambience to match the new era and environmental tags.
   * `experimance` coordinates timing and publishes `VideoMask` for depth difference visualization on sand.> **Purpose**
> Provide a concise, implementation‑ready reference for developers & AI coding assistants building *Experimance*: an interactive sand‑table where audience actions steer an AI‑generated satellite narrative of human development.

---

## 1  Experience Overview

A large bowl of white sand has AI generated satellite images projected on it. As the audience interacts with the sand the images update, progressing from untouched wilderness through many eras of human development to the modern day and then beyond to an imagined abstract AI-infused cityscape or post-apocalyptic ruins depending on the audience's interactions with the sand. A depth camera is used to observe the topology of the sand which informs the generated images. Sensors on the sand's contain detect how gently the sand is moved. 

As the audience looks at the piece it sees them with a webcam and introduces itself - the audience can talk to it and it invites them to play with the sand.

Soft music plays from nearby speakers along with environmental audio that matches the imagery in the sand (sounds of animals fading to the sounds of human habitation). 


### Eras
The piece has different states, based of eras of human development:

**Wilderness** (no humans)
**Pre-industrial** (villages)
**Early industrial** (1600s)
**Late industrial** (1800s)
**Early modern** (1900-1960)
**Modern** (2000-2020)
**AI/Future** (2030+)
**Post-apocalyptic**
**Ruins**

### Typical interaction Flow

1. Audience approaches the table → *agent* greets them (presence detected via webcam face‑tracking **or** hand‑over‑sand pickup from the depth camera).
2. Depth camera & sand‑edge sensors detect movement intensity. `experimance` calculates `user_interaction_score`.
3. **Era state machine** steps forward through the timeline (Wilderness ⇢ … ⇢ AI). From the AI era it may either loop within AI, or continue to Post‑apocalyptic and Ruins before drifting back toward Wilderness.
   * `experimance` (using the in-process `prompting` module) builds a text‑to‑image prompt and other necessary data.
   * `experimance` publishes a `RenderRequest` on the `events` bus (see Section 2 & 5).
   * *image\_server* receives the `RenderRequest`, renders the frame (local SDXL or remote fal.ai), and publishes `ImageReady` on the `images` bus.
   * *display* receives `ImageReady` and updates the projection.
   * *audio* cross‑fades music & ambience to match the new era (triggered by `EraChanged` from `experimance`).
4. When idle > *IDLE\_TIMEOUT* (configurable, default = 45 s) the scene drifts back toward Wilderness.

---

## 2  ZeroMQ Connection Map

**Message Bus**: ZeroMQ PUB/SUB (JSON packets) with unified events channel.

### Primary Communication Channels

| Channel / Port                     | Pattern     | Publishers                           | Subscribers                                 | Notes                                                                                                          |
| ---------------------------------- | ----------- | ------------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `events` (`tcp://*:5555`)          | PUB ▶ SUB   | **All services** *(multi-publisher)* | **All services** *(selective subscription)* | **Unified coordination channel**: All inter-service communication flows here with message type filtering.      |
| `depth/heightmap` (`tcp://*:5556`) | PUB ▶ SUB   | `depth_proc` *(PUB)*                 | `experimance` *(SUB)*                       | ~15 Hz JSON: `{ "hand_detected": bool, "depth_map_png": "<base64_optional>" }`. PNG only if changed & no hand. |
| `transitions` (`tcp://*:5561`)     | PUSH ▶ PULL | `experimance` *(PUSH)*               | `transition_worker.rs` *(PULL)*             | `TransitionRequest` work distribution (optional performance optimization)                                      |
| `loops` (`tcp://*:5562`)           | PUSH ▶ PULL | `experimance` *(PUSH)*               | `animate_worker` *(PULL)*                   | `LoopRequest` work distribution (optional performance optimization)                                            |

### Events Channel Message Flow

**Published by Core (`experimance`)**:
- `EraChanged` → `audio`, `display`, `agent`
- `RenderRequest` → `image_server`
- `AudioCommand` → `audio`
- `VideoMask` → `display`
- `DisplayMedia` → `display`
- `IdleStateChanged` → all services

**Published by Services**:
- `image_server` → `ImageReady` → `experimance`
- `agent` → `AgentControl` → `experimance`, `audio`
- `audio` → `AudioStatus` → `experimance`

> **Key Architecture Change**: Multiple publishers can publish to the same ZMQ address. This enables a **unified events channel** where all coordination messages flow through `tcp://*:5555`, eliminating the need for separate ports per service relationship.

> All sockets use `ZMQ_CONFLATE=1` where only the latest message matters (e.g., `depth/heightmap`).
> Heartbeats (`ZMQ_HEARTBEAT_IVL=3000`) keep connections resilient over the month‑long run.

The table defines a **star topology**: `experimance` sits at the center (`events` bus) while high‑bandwidth or service‑specific streams (depth, prompts, images) get their own ports so they can be shunted to a different host if needed (e.g., GPU server).
**Schema** snippets live in `/schemas/*.json`.

---

## 3 Physical Setup & Installation

### 3.1 Hardware Mounting

* An articulated arm mounted to a desk/table provides the mounting point for both the depth camera and projector.
* Both devices should be positioned to point straight down at the sand surface to minimize calibration issues.
* The depth camera and projector should be positioned as close as possible to each other to minimize parallax issues.

### 3.2 Depth Camera Calibration

* The `depth_proc` service requires a measurement range in centimeters (defined in its configuration).
* This range must be measured manually during setup and represents the min/max distance from the camera to the sand surface.
* Example configuration in `depth_proc.toml`:
  ```toml
  depth_min_cm = 40    # Minimum distance to detect (closest point)
  depth_max_cm = 55    # Maximum distance to detect (furthest point)
  ```
* Detailed algorithm documentation for depth map change detection will be maintained in a separate technical document.

### 3.3 Physical Dimensions

* Projector output: 1080p (1920×1080) resolution projected onto the sand surface.
* Sand container: Circular bowl approximately 25cm in diameter.
* Minimum height for mounting: 40-60cm above sand surface (dependent on projector and depth camera specifications).

### 3.4 Environment Requirements

* Low ambient light preferred for optimal projection visibility.
* Stable mounting surface to prevent vibrations affecting depth sensing ad sand sensors.

---

## 4  Micro‑services

### 4.1 `experimance.py`

*Language*: Python 3.11 — `asyncio`
*Responsibilities*

* Holds global state: `era`, `biome`, `idle_timer`, `audience_present`, `user_interaction_score`.
* Loads initial state from a JSON configuration file (allows for reproducible testing and defined starting states).
* Converts depth‑map Δ (from `depth/heightmap` channel) and sensor RMS (from OSC) into **user_interaction_score ∈ [0,1]**.
* Drives the era state machine:
  * Gentle interaction progresses the era slowly.
  * More active, rough interaction progresses more quickly.
    * Enough intensity (i.e., `user_interaction_score` accumulating past a certain threshold) locks in a future progression towards Post-apocalyptic and Ruins eras. Specific mechanics TBD.
* Publishes `EraChanged` (containing new `era` and `biome`) and `RenderRequest` (containing full prompt and depth map) events on the `events` bus.
* Processes `AgentControlEvent` messages (e.g., `SuggestBiome`) from the `agent` to influence `biome` state.

*Config example* (`experimance.toml`):

```toml
idle_timeout = 45           # seconds
wilderness_reset = 300      # seconds to full reset
sensor_gain = 1.8           # tweak for vibe sensors
initial_state_path = "saved_data/default_state.json"  # Path to initial state JSON
```

*State Management*:
* While persistent state across restarts is not required, the service loads its initial state at startup from `initial_state_path`.
* This state file contains era, biome, and any other necessary parameters.
* For testing, different state files can be specified (e.g., to start directly in the AI era).
* Example state file:
  ```json
  {
    "era": "wilderness", 
    "biome": "temperate_forest",
    "accumulated_interaction_score": 0.0,
    "era_history": []
  }
  ```

### 4.2 `prompting` (in‑process module)

* Lives inside `services/experimance/prompting/` (imported as `from prompting import build_prompt`).
* Templates in `/prompts/{era}/{biome}.jinja2`; optional llama‑3‑8B variations if local GPU is present.
* Called synchronously by `experimance` when it needs a new prompt—no ZeroMQ traffic required.
* **Optional future split**: if you want a language‑model farm, swap the direct call for a `REQ/REP` RPC over `tcp://*:5560`.

### 4.3 `image_server.py` `image_server.py`

* Strategy pattern: `LocalSDXL`, `RemoteFal`, `MockGenerator`.
* Receives `RenderRequest` from the `events` bus.
* Emits `ImageReady` (JSON with `type: "ImageReady"`, UUID, and PNG path/URL) on the `images` bus.
* Async processing of image generation requests.

### 4.4 `display`

* Default: OpenGL full‑screen window (PyGLFW) + shader‑based cross‑fade.
* **Performance alternative**: Godot or Rust SDL app with ZeroMQ image stream & OSC control (TBD; investigate Godot ZMQ/OSC plugins).
* Subscribes to the `events` channel (`tcp://*:5555`). Differentiates incoming messages (`ImageReady`, `TransitionReady`, `LoopReady`) by inspecting the `type` field in the JSON payload.
* Shader‑based cross-dissolve between images and video.
* Receives either:
  * **Image** to quickly crossfade to (generated in response to sand interation, i.e. new depth map)
  * **Transition** (video/image series), crossfade to and stay on final frame/image after playing once
  * **Loop** (video/image series): crossfade to and repeat forever 
* Implementation notes:

  * **Textures:** keep three GPU textures resident: `current`, `transition_frame`, `loop_video`.

    * Cross-fade shader mixes `current` ↔ `transition_frame`.
    * When transition ends, promote `transition_frame` to `current`.
    * If a `LoopReady` arrives, hand texture ownership to a video-decoder thread that pushes frames into `loop_video`.
    * When a new transition starts, stop the decoder and revert to stills.
  * **Transport buffering:** enable `ZMQ_CONFLATE=1` on the `images` SUB socket for the *transition frames* topic; you’ll only ever upload the newest frame each render tick, so late frames are silently dropped.
  * **Latency knobs:**

    * Transition length default ≈ 3 s (90 frames @ 30 fps).
    * Generate frames at *display* resolution; no scaling in the shader → smoother.
    * If `transition_worker` can’t keep up, fallback to simple GPU linear fade performed directly in `display`—have `experimance` include a `"style":"simple"` hint when frame budget is tight.

### 4.5 `agent`

* Using LiveKit or pipecat libraries for voice-to-voice LLM interaction.
  * RAG system for recalling details about how and why the work was made, artist's thoughts on AI, and LLMs own take on the piece (its thoughts of itself).
* Welcomes user to the art and explains they can interact with sand or talk with the piece.
* Uses tool use/function calling to send `AgentControlEvent` messages (e.g., `{ "event_type": "SuggestBiome", "payload": { "biome_suggestion": "desert" } }`) to `experimance` via the `agent_ctrl` channel when the user indicates a location preference.
* Additional capabilities:
  * Face detection → open‑mouth gating to reduce false speech triggers?
* Dialogue policy (JSON rules):
  * greet on face seen for > 1 s (sends `AgentControlEvent` like `{ "event_type": "AudiencePresent", "payload": { "status": true } }`).
  * if user mentions "where I’m from" → triggers tool use to suggest biome.

### 4.6 `audio`

* SuperCollider backend (preferred, open‑source).
* Layers: `music`, `ambience`, `ui`.
* Auto‑ducking music & ambience via side‑chain when the **agent or audience** speaks.

### 4.7 `depth_proc.py`

> *Optional split‑out service if RealSense processing overwhelms the Pi*

* Captures raw frames from RealSense D455, computes height‑map (C++/pybind11 or Rust for vectorized filters).
* Runs at 15-30 fps.
* Detects hand presence.
* Publishes a JSON message on the `depth/heightmap` topic (`tcp://*:5556`):
  `{ "hand_detected": <boolean>, "depth_map_png": "<base64_encoded_png_string_optional>" }`
  The `depth_map_png` is only included if the depth map has changed significantly since the last publication AND no hand is currently detected (to avoid imprinting hands on the map).
* `experimance` subscribes to this channel and uses `hand_detected` for interaction logic (e.g., pausing certain processes) and `depth_map_png` for generating `RenderRequest` messages.
* Settings (via TOML): bilateral filter radius, hole‑fill level, target FPS, change threshold for depth map publication.
* Can run **either** on Raspberry Pi 5 (ARM64 build) or mini‑PC GPU (CUDA accelerated) depending on benchmark.

> `experimance` subscribes to `depth/heightmap` and no longer touches hardware directly when this service is enabled.

---

## 5  Inter‑service Message Schemas (excerpt)

```jsonc
// --- Schemas for unified `events` channel (tcp://*:5555) --- 
// All services publish and subscribe to this channel with message type filtering

// EraChanged (published by experimance → audio, display, agent)
{
  "type": "EraChanged",
  "era": "ai_future", // string, e.g., "wilderness", "pre_industrial", etc.
  "biome": "coastal"  // string, e.g., "desert", "forest", "mountains"
}

// RenderRequest (published by experimance → image_server)
{
  "type": "RenderRequest",
  "request_id": "<uuid_string>",
  "era": "ai_future",
  "biome": "coastal",
  "prompt": "Low‑orbit satellite view of a coastal smart‑city …",
  "depth_map_png": "<base64_encoded_png_string_1024x1024_optional>"
}

// AudioCommand (published by experimance → audio)
{
  "type": "AudioCommand",
  "command_type": "spacetime" | "include_tags" | "exclude_tags" | "trigger",
  "era": "ai_future",
  "biome": "coastal",
  "tags_to_include": ["urban", "traffic", "technology"],
  "tags_to_exclude": ["birds", "nature"],
  "trigger": "interaction_start" | "interaction_stop" | "transition"
}

// VideoMask (published by experimance → display)
{
  "type": "VideoMask",
  "mask_id": "<uuid_string>",
  "mask_type": "depth_difference",
  "depth_map_png": "<base64_encoded_png_string>",
  "interaction_score": 0.7
}

// ImageReady (published by image_server → experimance, display)
{
  "type": "ImageReady",
  "request_id": "<uuid_string>",
  "image_id": "<uuid_string>",
  "uri": "file:///var/cache/experimance/frame_1723.png"
}

// DisplayMedia (published by experimance → display)
{
  "type": "DisplayMedia",
  "content_type": "image",  // "image", "image_sequence", "video"
  
  // For IMAGE content_type
  "image_data": "<base64_or_numpy_array>",  // Image data in base64 or raw format
  "uri": "file:///path/to/image.png",       // Optional file URI
  
  // For IMAGE_SEQUENCE content_type
  "sequence_path": "/path/to/sequence/",    // Directory with numbered images
  
  // For VIDEO content_type
  "video_path": "/path/to/video.mp4",       // Path to video file
  
  // Display properties
  "duration": 3.0,                          // Duration in seconds (sequences/videos)
  "loop": false,                            // Whether to loop content
  "fade_in": 0.5,                          // Fade in duration in seconds
  "fade_out": 0.5,                         // Fade out duration in seconds
  
  // Context information
  "era": "ai_future",                       // Current era
  "biome": "coastal",                       // Current biome
  "source_request_id": "<uuid_string>"      // Links to original RenderRequest
}

// AgentControl (published by agent → experimance, audio)
{
  "type": "AgentControl",
  "sub_type": "SuggestBiome" | "AudiencePresent" | "ConversationState",
  "biome_suggestion": "desert",
  "audience_present": true,
  "conversation_active": false
}

// AudioStatus (published by audio → experimance)
{
  "type": "AudioStatus",
  "status": "ready" | "transitioning" | "error",
  "active_tags": ["urban", "traffic", "technology"],
  "current_era": "ai_future",
  "current_biome": "coastal"
}

// IdleStateChanged (published by experimance → all services)
{
  "type": "IdleStateChanged",
  "is_idle": true,
  "idle_duration": 45.2
}

// --- Schemas for `events` channel (tcp://*:5555) --- 
// All messages on this channel should include a "type" field for the display to dispatch on.

// ImageReady (image_server → display)
{
  "type": "ImageReady",
  "request_id": "<uuid_string_optional>", // Corresponds to RenderRequest if applicable
  "image_id": "<uuid_string>", // Unique ID for this specific image/frame
  "uri": "file:///var/cache/experimance/frame_1723.png" // or http://, data: etc.
}

// TransitionReady (transition_worker → display)
{
  "type": "TransitionReady",
  "transition_id": "<uuid_string>",
  "uri": "file:///var/cache/experimance/transitions/uuid_0000.avif", // Path to video or image sequence manifest
  "is_video": true, // boolean, true if URI points to a video file, false if image sequence/manifest
  "loop": false, // Should always be false for transitions
  "final_frame_uri": "file:///var/cache/experimance/transitions/uuid_0063.avif" // URI of the last frame to hold
}

// LoopReady (animate_worker → display)
{
  "type": "LoopReady",
  "loop_id": "<uuid_string>",
  "uri": "file:///var/cache/experimance/loops/loop_animation.mp4",
  "is_video": true,
  "duration_s": 10.5 // Optional: duration in seconds
}

// --- Schemas for `events` channel (tcp://*:5555) --- 

// AgentControlEvent (agent → experimance, audio)
{
  "type": "AgentControlEvent", 
  "sub_type": "SuggestBiome", // e.g., "SuggestBiome", "AudiencePresent", "SpeechDetected"
  "payload": {
    // Payload structure varies based on sub_type
    // Example for SuggestBiome:
    // "biome_suggestion": "desert"
    // Example for AudiencePresent:
    // "status": true 
  }
}

// --- Schemas for PUSH/PULL channels --- 

// TransitionRequest (experimance → transition_worker on tcp://*:5561)
{
  "type": "TransitionRequest",
  "request_id": "<uuid_string>",
  "from_image_uri": "file:///var/cache/experimance/current_stable_frame.png",
  "to_image_uri": "file:///var/cache/experimance/next_stable_frame.png",
  "style": "dissolve", // e.g., "dissolve", "morph", "wipe"
  "duration_frames": 90 // Number of frames for the transition
}

// LoopRequest (experimance → animate_worker on tcp://*:5562)
{
  "type": "LoopRequest",
  "request_id": "<uuid_string>",
  "still_image_uri": "file:///var/cache/experimance/current_stable_frame.png",
  "style": "subtle_wind_animation" // Hint for the animation style
}

```

Complete schema definitions live at `/schemas`.

---

## 6  Hardware & I/O

| Component      | Model / Spec                    | Interface     | Notes                                           |
| -------------- | ------------------------------- | ------------- | ----------------------------------------------- |
| Projector      | 1920×1080                       | HDMI / NDI    |                                                 |
| Depth Camera   | Intel RealSense D415            | USB 3.2       |                                                 |
| Vibe Sensors   | Piezo + HX711 amp (edge mounts) | Arduino → OSC | `experimance` service listens for OSC messages. |
| GPU            | NVIDIA RTX 4090 (VRAM 24 GB)    | PCIe 4.0      |                                                 |
| Mic & Speakers | USB audio + 2.1 monitors        | USB/C Jack    |                                                 |

## 7 Transition & Animation Pipeline — Design Sketch

| Phase                         | What happens                                                                                                                                                                                                                                                                                                                                                                                                                   | Why this variant is fast & flexible                                                                                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 Stable-frame render**     | `experimance` requests still image from `image_server` by publishing `RenderRequest` on `events` bus → `image_server` eventually publishes `ImageReady` (PNG) on `images` bus.                                                                                                                                                                                                                                                 | Same path you already have; one high-quality key-frame per era step. Non-blocking.                                                                                                            |
| **2 Transition job**          | `experimance` immediately publishes`TransitionRequest {from_uri, to_uri, style, frames}` on a new ZeroMQ **PUSH** queue (`tcp://*:5561`).`transition_worker.rs` (compiled with SIMD + WGPU) runs **PULL** workers (1 GPU thread each) that generate e.g. 32- or 64-frame morphs and write them to `/var/cache/experimance/transitions/uuid_%04d.avif` *or* a single H.265 (`-tune zerolatency`) clip.                          | *PUSH/PULL* lets you add more workers or move them to another host without touching the rest of the graph. Rust/WGPU can blend two 1024² textures → 60 fps on even modest GPUs.               |
| **3 Display swap**            | Worker publishes `TransitionReady {uri, is_video, loop=false}` on the `images` PUB channel (`tcp://*:5558`).`display` sees it, preloads the file/frames, cross-fades to the transition, then auto-plays the sequence.                                                                                                                                                                                                          | Keeps all media deliveries unified on one PUB topic; display remains stateless beyond “what’s current/next?”.                                                                                 |
| **4 Idle era drift**          | If `experimance` decides to drift eras (no depth activity, maybe voice), *repeat steps 1-3*. The *from_uri* for the next transition is simply the *to_uri* of the last one.                                                                                                                                                                                                                                                    |                                                                                                                                                                                               |
| **5 Optional loop animation** | When hardware budget allows: • `experimance` publishes `LoopRequest {still_uri, style}` on another PUSH queue (`tcp://*:5562`). • `animate_worker` (either a small SD-video model or cloud service) returns `LoopReady {video_uri, duration_s}` on `images` channel. • `display` switches its texture source from “last still” to looping video (ffmpeg → GL texture or Godot’s VideoPlayer) until the next `TransitionReady`. | Isolation means you can add fancy generative loops later without re-wiring the main display logic. If the loop service isn’t running, nothing breaks—`display` just stays on the still image. |

## 8  Dev & Ops

### 8.1 Repository Layout

```
experimance/                # monorepo root (git)
├─ services/                # independently runnable micro‑services
│  ├─ experimance/          # ↳ each has its own `pyproject.toml` and `config.toml` (example)
│  ├─ image_server/
│  ├─ display/
│  ├─ agent/
│  └─ audio/
├─ libs/                    # shared pure‑python utilities
│  └─ common/
├─ infra/                   # ops artefacts
│  ├─ docker/               # per‑service Dockerfiles (x86‑64 + ARM64)
│  ├─ compose.yaml          # optional docker‑compose stack
│  ├─ ansible/              # bare‑metal provisioning playbooks
│  └─ grafana/              # dashboards + Prometheus rules
└─ scripts/                 # helper dev scripts (`./scripts/dev <svc>`)
```

* **Package manager**: [`uv`](https://github.com/astral-sh/uv) handles lock‑files & virtual‑envs.

  ```bash
  uv pip install -r services/experimance/requirements.txt
  uv venv .venv
  ```
* Shared lint & type‑check config at repo root (`ruff.toml`, `mypy.ini`).

### 8.2 Continuous Integration

* GitHub Actions matrix: `ubuntu‑latest` (x86‑64) & `buildjet‑arm64` (for Pi).
* Steps: `uv pip install`, Ruff, Mypy, pytest, package build.
* Artifacts: multi‑arch Docker images pushed to GHCR (tags: `:main‑amd64`, `:main‑arm64`).

### 8.3 Deployment Targets

Consolidated deployment strategy:

| Host               | Role / Expected Load                                                                                                                                       | Method                                                 | Notes                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Mini‑PC x86**    | GPU‑bound tasks: `image_server` (local SDXL), `transition_worker`, `animate_worker`, optional heavy LLM for `agent`, optional `depth_proc` (CUDA).         | Docker (NVIDIA Container Runtime)                      | Primary compute node.                                                  |
| **Raspberry Pi5**  | Orchestration & I/O: `experimance`, `display` (if lightweight), `audio` (if local to agent), `depth_proc` (if lightweight), OSC listener for vibe sensors. | Bare‑metal Python via `systemd` or lightweight Docker. | Handles real-time interaction and less intensive services.             |
| **Cloud (fal.ai)** | Primary remote image generation (fallback/alternative for `image_server`), primary STT/TTS/LLM for `agent`.                                                | HTTPS API                                              | For heavy lifting if local GPU is insufficient or for specific models. |

* `compose.yaml` can be used for local development to simulate the multi-host setup or to deploy a full stack on a single powerful machine.
* Production will likely mix bare-metal/`systemd` for Pi services and Docker for GPU-heavy services on the Mini-PC to balance ease of deployment and resource access.

### 8.4 Observability & Remote Access

* **Metrics**: Prometheus Python client (`opentelemetry‑exporter‑otlp` → Prometheus remote‑write).
* **Dashboards**: Self‑hosted Grafana on the Mini‑PC (open‑source edition → \$0). Dashboards and alert rules stored under `infra/grafana/` and provisioned via `ansible`.
* **Networking / Remote Debug**: [Tailscale](https://tailscale.com) mesh‑VPN.

  * Allows SSH & web‑UI access to Pi and Mini‑PC without port‑fiddling.
  * Cheap (free tier up to 20 devices) and reliable for a month‑long remote install.
* **Alerting**: Grafana contact point → email or Matrix message if critical services restart repeatedly.

### 8.5 Local Developer Comfort

* `make dev ENV=experimance` spins up virtual‑env, installs deps via `uv`, starts autoreload.
* VS Code workspace in repo root with multi‑service debug launch configs.
* `pre‑commit` hooks enforce formatting & lint before pushes.

---

## 9  Open Questions / TODO  Open Questions / TODO

1. Decide control‑net injection point: pre‑baked vs runtime.
2. Investigate StreamDiffusion or Latent Consistency Models for 30 fps preview option.
3. Audio service strategy: While the `agent` machine (likely Mini-PC if using local STT/TTS/LLM) needs direct access to mic/speakers for low-latency voice interaction, the `audio` service (controlling overall soundscape, music) can still be a separate entity. It could run on the Pi and receive `EraChanged` events, or even on the Mini-PC if co-located with the agent. The key is that the `agent`'s voice I/O is local to its processing.
4. Long‑term storage of generated images & metadata for archival.
5. Configuration Management: Adopt a per-service `config.toml` approach. For secrets (API keys), use environment variables (e.g., sourced from a `.env` file not committed to git, or injected by the deployment system).
6. System Resilience: Services should auto-restart on crash (e.g., via `systemd` or Docker restart policies). While perfect idempotency isn't strictly required for all messages, critical state changes or resource-intensive requests should be designed to minimize issues if re-processed (e.g., `RenderRequest` could include a UUID; `image_server` could check if an image for that UUID is already being generated or exists). A global "reset all services" script/mechanism could be useful for development or unrecoverable states.

---

## 10 Testing Strategy

### 10.1 Testing Approach

* **Unit Testing**: Each service component will have pytest-based unit tests with mocking of dependencies.
* **Integration Testing**: Simulated end-to-end workflows testing communication between services.
* **Performance Testing**: Latency benchmarks for critical paths, particularly the image generation and transition pipelines.

### 10.2 Test Framework

* **pytest** as the primary testing framework due to its simplicity, powerful fixtures, and parameterization capabilities.
* **pytest-mock** for mocking dependencies.
* **pytest-asyncio** for testing asynchronous components.

```bash
# Example test command
uv run pytest -xvs services/experimance/tests/
```

### 10.3 Mocking Strategy

* Every service is designed to be testable in isolation with mocked dependencies.
* ZeroMQ communication is mocked by intercepting socket operations:

```python
# Example mock for ZMQ subscriber
@pytest.fixture
def mock_zmq_subscriber(mocker):
    mock_sub = mocker.patch("zmq.Context.socket")
    # Configure mock to return test messages when recv is called
    mock_sub.return_value.recv_multipart.return_value = [
        b'EraChanged', 
        json.dumps({"era": "modern", "biome": "coastal"}).encode('utf-8')
    ]
    return mock_sub
```

* Image data can be replaced with fixed test patterns.
* A `MockGenerator` strategy in `image_server.py` returns pre-generated test images.

### 10.4 Test Data

* Sample depth maps stored in `test/data/depth_maps/`
* Mock images for each era in `test/data/mock_images/`
* Test videos for transitions in `test/data/transitions/`

### 10.5 Runnable Modules

Each service module is designed to be directly runnable with test functionality:

```python
# services/experimance/image_server.py
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock generator")
    parser.add_argument("--test-file", type=str, help="Test with specific depth map file")
    args = parser.parse_args()
    
    # Run standalone test mode
    server = ImageServer(use_mock=args.mock)
    asyncio.run(server.serve_test(test_file=args.test_file))
```

## 11 Edge Cases & Robustness

### 11.1 Multiple Users & Disruptions

| Edge Case                     | Detection Method                              | Response                                                                 |
| ----------------------------- | --------------------------------------------- | ------------------------------------------------------------------------ |
| Multiple users at once        | Face detection count > 1                      | Normal operation; agent addresses group                                  |
| Hand covering depth camera    | 100% of depth map within hand detection range | Pause image generation, log warning, continue with previous stable image |
| Sand surface covered/obscured | Depth map variance below threshold            | Maintain current era, log warning                                        |
| Excessive/violent interaction | user_interaction_score > 0.95                 | Cap interaction effect, potentially accelerate to post-apocalyptic       |

### 11.2 Technical Failures

| Failure Type               | Detection             | Recovery                                                                              |
| -------------------------- | --------------------- | ------------------------------------------------------------------------------------- |
| Image generation timeout   | No response after 10s | Log error, retry once, then fall back to nearest cached image for era                 |
| ZeroMQ message loss        | Heartbeat failure     | Automatic reconnection via ZMQ_RECONNECT_IVL                                          |
| GPU out of memory          | CUDA error in logs    | Reset `image_server` service, fall back to remote generation                          |
| Depth camera disconnection | Device error          | Log critical error, continue with last valid depth map, show alert on admin dashboard |

### 11.3 Resource Management

| Resource                      | Estimated Usage          | Management Strategy                                                           |
| ----------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| GPU Memory                    | 8-12GB for local SDXL    | Reserve 10-15GB for generation; 2GB for display; 1GB for transition rendering |
| Disk Space                    | ~500MB/hour of operation | Implement rolling cleanup of old images and convert to video                  |
| Network (if using remote API) | ~2-5 MB per image        | Compress depth maps before transmission; cache frequent generations           |
| CPU (Pi)                      | ~40-60%                  | Monitor thermal throttling; consider cooling solution                         |

### 11.4 Performance Targets

* Image generation latency: Optimize from current ~2s toward target 500ms
* End-to-end interaction to visual response: <500ms ideal
* Transition rendering: Real-time (30fps)
* Audio transition: Synchronized within 100ms of visual change

---

© 2025 Experimance — MIT‑licensed where applicable.

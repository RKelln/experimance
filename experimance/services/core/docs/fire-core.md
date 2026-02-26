# Fire Core Service

## What it does

`fire_core` is the central orchestrator for the **Feed the Fires** interactive art installation. It listens for audience stories and live conversation transcripts from the agent service, uses an LLM to infer environmental settings from the content, then drives a panoramic image pipeline: base image → tiles → display.

Key behaviours:
- **Story-based flow** – a complete `StoryHeard` message triggers one full render cycle.
- **Transcript-based flow** – streaming `TranscriptUpdate` messages accumulate until there is enough user content to begin LLM analysis.
- **Smart interruption** – base images always complete; tile generation and LLM processing can be cancelled if a higher-priority request arrives.
- **Request queueing** – multiple simultaneous requests are queued and processed in order.

---

## Quick Start

```bash
# Set active project
scripts/project fire

# Install (from repo root)
uv pip install -e services/core

# Run the service
uv run -m fire_core

# Run with mock LLM (no API key needed)
uv run -m fire_core --llm-provider mock

# Debug mode
uv run -m fire_core --log-level DEBUG

# See all options
uv run -m fire_core --help
```

### CLI test utility

The included CLI tool lets you drive the service without a real agent:

```bash
# Interactive menu
uv run -m fire_core.cli --interactive

# Send a pre-built conversation sequence
uv run -m fire_core.cli --conversation forest_memories --delay 2
uv run -m fire_core.cli --conversation desert_journey --delay 1.5
uv run -m fire_core.cli --conversation mountain_reflection

# Send a single story
uv run -m fire_core.cli --story "I walked through ancient redwoods..."

# Send a direct prompt (debug, bypasses LLM)
uv run -m fire_core.cli --prompt "mystical forest with golden light"

# Send a single transcript line
uv run -m fire_core.cli --transcript "The forest felt magical" --speaker-id user

# List available test content
uv run -m fire_core.cli --list-conversations
uv run -m fire_core.cli --list-stories
uv run -m fire_core.cli --list-prompts
```

---

## Environment Assumptions

- Linux
- `OPENAI_API_KEY` in `projects/fire/.env` (unless using `--llm-provider mock`)
- `image_server` service running and reachable on port 5564
- `display` service running and reachable on port 5555

---

## Configuration

Main config: `projects/fire/core.toml`

Key settings:

| Section    | Key                   | Description                               |
|------------|-----------------------|-------------------------------------------|
| `[tiler]`  | `max_tile_megapixels` | Maximum megapixels per tile               |
| `[tiler]`  | `min_overlap`         | Minimum overlap fraction for seamless blend |
| `[llm]`    | `provider`            | `openai` or `mock`                        |
| `[llm]`    | `model`               | OpenAI model name (e.g. `gpt-4o`)         |
| `[llm]`    | `timeout`             | Request timeout in seconds                |

Override any key at runtime with `--section-key` flags.

---

## Request State Machine

All state transitions are centralised in `_state_monitor_task()`. There is no global service state; each request tracks its own lifecycle.

```
QUEUED → PROCESSING_LLM → WAITING_BASE → BASE_READY → WAITING_TILES → WAITING_AUDIO → COMPLETED
                                                                         ↓ (if interrupted)
                                                                       CANCELLED
```

| State            | Description                                                          |
|------------------|----------------------------------------------------------------------|
| `QUEUED`         | Request created, waiting to be picked up                             |
| `PROCESSING_LLM` | LLM is analysing content (can be interrupted by new transcript)      |
| `WAITING_BASE`   | Base panorama is being generated (protected from cancellation)       |
| `BASE_READY`     | Base image displayed; tile requests are being sent                   |
| `WAITING_TILES`  | Tiles generating (can be cancelled for higher-priority request)      |
| `WAITING_AUDIO`  | Base and tiles complete; waiting for audio to finish                 |
| `COMPLETED`      | All processing finished                                              |
| `CANCELLED`      | Interrupted or superseded by a newer request                         |

### Interruption policy

- **LLM processing** – cancelled immediately when a new transcript session arrives.
- **Base image** – never cancelled once started (ensures something always displays).
- **Tile generation** – cancelled for a higher-priority request; base stays visible.

---

## Message Flow

### Story-based

```
StoryHeard → LLM Analysis → Queue Request → RenderRequest (base) →
ImageReady (base) → RenderRequest × N (tiles) → ImageReady × N → DisplayMedia × N → COMPLETED
```

### Transcript-based

```
TranscriptUpdate → Accumulator → (sufficient user content?) →
Background LLM → Smart Interruption → Queue → State Monitor → [story-based flow above]
```

---

## ZMQ Communication

| Channel                | Port | Role                                                |
|------------------------|------|-----------------------------------------------------|
| Agent channel          | 5557 | Binds; receives `StoryHeard`, `TranscriptUpdate`, `UpdateLocation` |
| Updates channel        | 5556 | Binds; receives debug prompts from CLI              |
| Controller channel     | 5555 | Publishes `DisplayMedia`; pushes `RenderRequest` to image_server |
| image_requests         | 5564 | Push (work → image_server)                          |
| image_results          | 5565 | Pull (results ← image_server)                       |

### Input messages

| Type              | Source        | Description                              |
|-------------------|---------------|------------------------------------------|
| `StoryHeard`      | agent service | Complete audience story                  |
| `TranscriptUpdate`| agent service | Streaming conversation fragment          |
| `UpdateLocation`  | agent service | Location context change                  |

### Output messages

| Type           | Destination    | Description                        |
|----------------|----------------|------------------------------------|
| `RenderRequest`| image_server   | Image generation request           |
| `DisplayMedia` | display service| Instruct display to show an image  |

---

## Tiling Strategy

1. Calculate optimal number of tiles to cover the panorama while staying under `max_tile_megapixels`.
2. Apply minimum `min_overlap` to ensure seamless blending at edges.
3. Generate edge masks for smooth composition.
4. Each tile is sent as a separate `RenderRequest` referencing the base image as context.
5. Tile positions are included in the `DisplayMedia` message so the display service places them correctly.

---

## Development

### Adding a new biome

1. Add the biome value to `projects/fire/schemas.py` (`Biome` enum).
2. Add a prompt template in `fire_core/llm_prompt_builder.py`.
3. Test with `uv run -m fire_core.cli --story "..."`.

### Adding a new LLM provider

1. Implement the `LLMProvider` interface in `src/fire_core/llm.py`.
2. Add a config option for the provider name.
3. Register the provider in `LLMManager` factory.

---

## Error Handling

| Scenario                          | Behaviour                                              |
|-----------------------------------|--------------------------------------------------------|
| image_server unavailable          | Request times out, transitions to CANCELLED, queue continues |
| LLM API error                     | Retried up to configured limit, then request CANCELLED |
| Display service unavailable       | `DisplayMedia` dropped, logged as warning              |
| ZMQ connection lost               | `ControllerService` reconnects automatically           |

---

## Files Touched

| File                                   | Role                                        |
|----------------------------------------|---------------------------------------------|
| `src/fire_core/fire_core.py`           | Main service class and state monitor        |
| `src/fire_core/config.py`              | Pydantic config models                      |
| `src/fire_core/llm.py`                 | LLM provider interface and implementations  |
| `src/fire_core/llm_prompt_builder.py`  | Prompt composition logic                    |
| `src/fire_core/audio_manager.py`       | Audio generation / playback coordination    |
| `src/fire_core/tiler.py`               | Tile strategy calculation                   |
| `src/fire_core/prompt_logger.py`       | Structured LLM input/output logging         |
| `src/fire_core/cli.py`                 | CLI test utility                            |
| `src/fire_core/__main__.py`            | Entry point                                 |
| `projects/fire/core.toml`              | Production configuration                    |
| `projects/fire/schemas.py`             | Project-specific message schemas and enums  |
| `tests/test_tiler_new.py`              | Tiler unit tests                            |
| `tests/test_llm_integration.py`        | LLM integration tests                       |

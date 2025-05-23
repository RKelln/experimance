# Experimance Audio System — Technical Design Document

---

## 1. Overview

The Experimance audio system provides a responsive, layered soundscape tightly coupled to dynamically generated satellite visuals of biomes and eras. It is collaborative, modular, and supports both live/manual and automated workflows, with separation of control (Python CLI/OSC) and sound engine (SuperCollider).

---

## 2. Architecture

- **SuperCollider**
  - Receives OSC messages to update current context (biome, era), add/remove tags (for sound inclusion/exclusion), and trigger events (e.g., /listening, /transition).
  - Loads and manages environmental audio layers and music loops from config files.
  - Applies default logic based on context and responds to explicit include/exclude overrides.
  - Supports hot-reloading of configs for rapid testing and creative iteration.

- **Python CLI/Bridge**
  - Interactive and/or scriptable tool for sending OSC messages.
  - Allows for manual testing, session recording/playback, and full automation mirroring installation logic.

---

## 3. Sound Configurations

### A. Environmental Audio Layers

- **Each entry:**
  - `path`: Relative file path to audio file
  - `tags`: List of tags (e.g., ["desert", "pre_industrial", "birds"])
  - `prompt`: Text description for text-to-audio or creative documentation
  - `volume`: Optional, default 1.0

Example (layers.json):
```json
[
  {
    "path": "audio/env/birds_desert.wav",
    "tags": ["desert", "pre_industrial", "birds"],
    "prompt": "Sparse desert bird calls in early morning wind",
    "volume": 0.8
  },
  {
    "path": "audio/env/church_bells.wav",
    "tags": ["church", "bells"],
    "prompt": "Distant church bells echoing in a small town",
    "volume": 1.0
  }
]
```
---

### B. Triggered Sound Effects

- **Each entry:**
  - `trigger`: Name of the event
  - `path`: Audio file
  - `prompt`: Description
  - `volume`: Optional

Example (triggers.json):
```json
[
  {
    "trigger": "transition",
    "path": "audio/sfx/transition_swipe.wav",
    "prompt": "Quick rising whoosh for image transition",
    "volume": 1.0
  },
  {
    "trigger": "ui_select",
    "path": "audio/sfx/ui_select.wav",
    "prompt": "Short synthetic bleep for user selection"
  }
]
```
---

### C. Music Loops (Layered, Era-based, Slot-ordered)

- Music is defined as a list of ordered slots per era.
- On era change, each slot in the previous era crossfades smoothly into the slot in the new era (rhythm/length/beat must match per slot across eras).
- The slots list is not labeled by instrument, allowing flexible creative interpretation by the musician—order matters.

Example (music_loops.json):
```json
{
  "era_loops": {
    "pre_industrial": [
      { "path": "audio/music/pre_industrial_loop1.wav", "prompt": "Folk drone, organic swells" },
      { "path": "audio/music/pre_industrial_loop2.wav", "prompt": "Hand percussion groove" },
      { "path": "audio/music/pre_industrial_loop3.wav", "prompt": "Pastoral melody phrase" }
    ],
    "industrial": [
      { "path": "audio/music/industrial_loop1.wav", "prompt": "Mechanical synth drone" },
      { "path": "audio/music/industrial_loop2.wav", "prompt": "Steam-driven drums" },
      { "path": "audio/music/industrial_loop3.wav", "prompt": "Repetitive piano arpeggios" }
    ],
    "current": [
      { "path": "audio/music/current_loop1.wav", "prompt": "Pulsing synth bass" },
      { "path": "audio/music/current_loop2.wav", "prompt": "Electronic rhythm kit" },
      { "path": "audio/music/current_loop3.wav", "prompt": "Ambient lead melody" }
    ]
  }
}
```
---

## 4. OSC Command Patterns

- `/spacetime <biome> <era>` — Set main context
- `/include <tag>` — Add a sound tag to active set
- `/exclude <tag>` — Remove a tag from active set
- `/listening <start|stop>` — Trigger UI/interaction SFX
- `/speaking <start|stop>` — Trigger UI/interaction SFX
- `/transition <start|stop>` — Scene/era/biome transition cue
- `/reload` — Reload configs in SuperCollider

Note: the `listening` and `speaking` tiggers may overlap, so other sounds should be ducked as long as one of them is started. 

---

## 5. Workflow and Extensibility

- Sound designer/musician edits JSON configs and adds new audio files.
- SC script hot-reloads configs and manages tag-based filtering, triggering, and crossfading.
- Python CLI/Bridge drives installation logic, can record/playback OSC sessions, and allows for manual creative testing.

---

## 6. Example Project Layout
```
experimance_audio/
├── src/experimance_audio/       # location for Python
├── sc_scripts/
│    └── experimance_audio.scd   # SuperCollider script
├── audio/
│    ├── music/
│    ├── sfx/
│    └── environments/
├── config/
│    ├── layers.json
│    ├── triggers.json
│    └── music_loops.json
└── logs/
     └── osc_recordings/
```
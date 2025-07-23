# Experimance Audio Configuration

This directory contains the configuration files for the Experimance audio system. These JSON files define the audio layers, trigger sounds, and music loops used by the SuperCollider script.

## Configuration Files

### 1. `layers.json`

Contains environmental audio layers that play continuously based on the current biome, era, or other active tags.

Format:
```json
[
  // Continuous ambience (“bed”)
  {
    "path": "environment/wilderness/temperate_forest_ambience.wav",
    "prompt": "Songbirds, wind in trees, distant animals",
    "tags": ["temperate_forest", "wilderness", "ambience"],
    "interval": "loop",
    "requires": ["temperate_forest"],
    "requires_any": ["wilderness", "pre_industrial"],
    "volume": 0.8
  },
  // Spot effect with moderate repetition
  {
    "path": "environment/current/culture_sports_fields.wav",
    "prompt": "Whistle, cheering, sports play sounds",
    "tags": ["current", "culture", "sports", "fields"],
    "interval": "frequent",
    "requires": ["current", "sports"],
    "volume": 1.0,
    "weight": 2
  }
]
```

#### Data Schema

##### **Required fields**

- **`path`** *(string)*  
  The relative file path of the audio file.

- **`prompt`** *(string)*  
  A human-readable description of the sound for UIs and text-to-audio models.

- **`tags`** *(array of strings)*  
  Descriptive metadata—these are for search, browsing, and categorization only.  
  Tags might include biome names ("forest", "lake"), feature types ("water", "ambience"), or other descriptors.

- **`interval`** *(string)*  
  Specifies the repetition type:
  - `"loop"`: continuously and seamlessly looped with crossfading
  - `"frequent"`: played many times per minute
  - `"occasional"`: played a few times per minute
  - `"rare"`: played only a couple of times per minute at most

##### **Optional audio fields**

- **`volume`** *(number)*  
  Playback volume (0.0 to 1.0, default: 1.0)

- **`crossfade_time`** *(number)*  
  For `"loop"` interval only: Duration in seconds for seamless crossfading between the end and beginning of the audio file (default: 2.0). This creates smooth, unnoticeable transitions in looping environmental sounds.

##### **Optional gating fields (to control playback eligibility)**

- **`requires`** *(array of strings)*  
  Only play this sound if **all** listed tags are present in the system's current set of active tags.  
  Leave this field out (or as an empty array) if there is no such requirement.  
  **Typical use:** biome or feature tags specifying a required location, state, or condition.

- **`requires_any`** *(array of strings)*  
  Only play this sound if **at least one** of these tags is present in the active set.  
  Leave this field out (or as an empty array) if there is no such restriction.  
  **Typical use:** eras or situations (e.g. "wilderness", "modern", "dystopia") that define a subset of possible contexts.

- **`requires_none`** *(array of strings)*  
  Never play this sound if **any** of these tags are present in the active set.  
  Generally used so that sounds that are similar, that share `requires` cannot be played at the same time (e.g. only allow one type of market for each era).

##### Tags and Requires fields

- `tags` serve as both metadata **and** primary selectors for playback:  
  - If any of a sound's `tags` match active tags in the controller, the sound is considered for playback.
  - Further gating is applied using `requires`, `requires_any`, and `requires_none`.

- **Order of logic:**
  1. At least one tag in `tags` is active.
  2. All `requires` tags are present (if provided).
  3. At least one `requires_any` tag is present (if provided).
  4. No `requires_none` tag is present (if provided).
  5. If multiple sounds are eligible, selection may be randomized or weighted.

- If `weight` is omitted, an eligible sound always plays (subject to how your scheduler handles concurrency).


##### **Logic for sound activation**

For a sound to be eligible:
1. **All** tags in `requires` must be present in the set of current active tags (if `requires` is present).
2. **At least one** tag in `requires_any` must be present (if `requires_any` is present).
3. **None** of the tags in `requires_none` can be present (if `requires_none` is present).

If a field is missing or an empty array, it is ignored (i.e., imposes no restriction).

##### **Duplicate entries for biomes or special gating**

If a sound should be available for multiple biomes with otherwise identical conditions, duplicate the entry, using a different `requires` for each relevant biome.

---

##### Example Entries

```json
[
  {
    "path": "environment/nature/river_ambience.wav",
    "prompt": "River ambience: gentle water flow, occasional splash, birds nearby",
    "tags": ["river", "nature", "ambience", "water"],
    "requires": ["river"],
    "requires_none": ["coastal"],
    "interval": "loop"
  },
  {
    "path": "environment/nature/forest_ambience.wav",
    "prompt": "Forest ambience: wind in trees, songbirds, distant animals",
    "tags": ["forest", "nature", "ambience"],
    "requires": ["forest"],
    "requires_any": [
      "wilderness", "pre_industrial", "early_industrial", "dystopia", "ruins"
    ],
    "interval": "loop",
    "crossfade_time": 3.0
  },
  {
    "path": "environment/nature/rainforest_rain.wav",
    "prompt": "Rainfall in a dense rainforest: heavy drops, thick leaf canopy, birds",
    "tags": ["rainforest", "rain", "ambience"],
    "requires": ["rainforest", "rain"],
    "requires_any": ["wilderness", "pre_industrial", "dystopia"]
    "interval": "loop",
  },
  {
    "path": "environment/city/subway_station.wav",
    "prompt": "Subway ambience: arriving train, crowd murmur, announcements",
    "tags": ["city", "subway", "transport", "ambience"],
    "requires": ["subway"],
    "requires_any": ["modern", "current", "future"]
    "interval": "loop",
  }
]
```


### 2. `triggers.json`

Contains one-shot sound effects that play in response to specific events.

Format:
```json
[
  {
    "trigger": "event_name",
    "path": "relative/path/to/audio.wav",
    "volume": 1.0
  },
  ...
]
```

- `trigger`: Event name that triggers this sound (e.g., "transition", "listening", "speaking")
- `path`: Relative path to the audio file from the audio directory
- `volume`: Playback volume (0.0 to 1.0)

### 3. `music_loops.json`

Contains era-specific music loops that provide background music for each era.

Format:
```json
{
  "era_loops": {
    "era_name": [
      {
        "path": "relative/path/to/audio.wav",
        "prompt": "Description of this audio layer",
        "volume": 0.7
      },
      ...
    ],
    ...
  }
}
```

- `era_name`: Name of the era (e.g., "wilderness", "pre_industrial")
- `path`: Relative path to the audio file from the audio directory
- `prompt`: Description of the audio for reference/documentation
- `volume`: Playback volume (0.0 to 1.0)

## Missing Audio Files

If an audio file specified in these configurations is not found, the system will automatically generate placeholder sounds:

- For music loops: Era-specific procedurally generated music
- For environmental layers and triggers: A warning will be printed

## Updating Configurations

Changes to these configuration files can be loaded at runtime by sending the `/reload` OSC message to SuperCollider:

```bash
uv run -m experimance_audio.cli reload
```

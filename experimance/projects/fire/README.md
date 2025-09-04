# Feed the Fires – Project Overview

## What is Feed the Fires?

*Feed the Fires* is an immersive 360‑projection artwork in which audience stories are transformed, in real‑time, into dream‑like panoramas that wrap an entire rectangular room.

The codebase extends the *Experimance* architecture.  Four micro‑services cooperate via a ZeroMQ event bus:

| Service            | Role                                                 | Key Messages                    |
| ------------------ | ---------------------------------------------------- | ------------------------------- |
| **`fire_core`**    | Builds prompts, orchestrates rendering, routes media | `RenderRequest`, `DisplayMedia` |
| **`image_server`** | Generates images (fast pass + 6 tiles)               | `ImageReady`                    |
| **`display`**      | Assembles & renders panorama, shader squashes × 6    | —                               |
| **`agent`**        | Collect audience stories                             | `StoryHeard` ➜ Core             |

```mermaid
graph TD
    A[StoryHeard] --> B(Core)
    B -->|fast pass| C(RenderRequest strip)
    C --> D(ImageServer)
    D -->|ImageReady strip| E(Display)
    B -->|6 RenderRequests| D
    D -->|DisplayMedia (any order)| E
```

### Data flow (numbers = seconds elapsed)

1. **0 s** – Story finish → half length strip request.
2. **\~1 s** – Strip arrives, blurred on screen.
3. **≤5 s** – Three tiles fade‑in. Mirror to reach full length.

---

# Core Service

**Package path:** `services/core/src/fire_core/`

## Quick start

```bash
# Set project to fire
scripts/project fire

# in separate terminals
uv run -m fire_core
uv run -m image_server
uv run -m experimance_display
uv run -m fire_agent
```


## Responsibilities

Creates a panorama image based on a story told by the audience. The story is obtained from 
the `agent` service then passed to the `core` that manges the rendering of the
panorama using `image_server` and then sends the pieces of the panorma to `display`.

Geenerally the panoram is half the total wall length and mirrored (in `display`).
So first a base image is generated for the entire half length, then tiles of that 
are generated individually at higher resolution using the base image as a reference 
to ensure they tile seamlessly.

* Listen:
  * `StoryHeard` ➜ call `infer_location()` (cloud LLM) to create a prompt.
    * Send `DisplayMedia` with `ContentType.CLEAR` to `display` to clear panorama.
  * `UpdateLocation` ➜ call `update_location()` to modify the existing prompt.
* Build prompt → dispatch `RenderRequest` at N width * 1 height aspect.
* On full aspect image ready → send `DisplayMedia` with no `position`.
* Calculate tiles and use those as base/reference images.
* Fire N hi‑res `RenderRequest`s, one for for each tile.
* Send each each tile as `DisplayMedia` with `position` info to `display`.


## Package layout

```
services/core/src/fire_core/
├─ fire_core.py  # Idle ▸ Listening ▸ BaseImage ▸ Tiles
├─ llm.py               # OpenAI / … wrapper
├─ tiler.py             # Manages tiling
└─ prompt_builder.py    # Creates or updates text-to-image prompts based on narratives
```

# Tiled images

The core service has to intelligently manage image tiling, given a base image. 
It will overlap the tiles and mask out the edges to help with blending before sending
the correctly positioned tiles to display service.

Given a base image, a maximum tile size and minimum amount of overlap it will create 
these tiles by cropping the base image. It uses these for image to image generation,
then applies a mask to the edges of the generated images to create transparent PNGs
that can be sent directly to the display wioth proper positioning.

---

# Display Service – `services/display/README.md`

**Package path:** `services/display/src/display/`

## Quick start

```bash
uv pip install -e services/display
uv fire_display --config projects/fire/display.toml
```

## Rendering pipeline

1. **BaseImage** – fast‑pass base image, starting with large dynamic blur σ→0 over N s.
2. **Tiles** – each new tile is rescaled and positioned and alpha‑fades 0→1 in 3 s.
3. **Projection shader** – optional mirror, rescale to screen size

## Key config (`config.toml`)

```toml
[panorama]
enabled = true
rescale = "width" # or "height" or "shortest", applies to tiles as well
output_width = 1920 # final size of panorama after mirroring
output_height = 1080
start_blur = 5.0 # sigma
end_blur = 0.0
blur_duration = 10.0 #seconds
mirror = true # mirror image horizontally

[panorama.tiles]
width = 1920   # actual image size of tiles rescaled to this size in panorama
height = 1080
fade_duration = 3.0
```


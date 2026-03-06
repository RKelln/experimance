# Circular Marquee Text

Characters travel clockwise around the circular display mask, appearing one at a time as a write head advances.  A configurable gap zone keeps clear space ahead of the cursor.  Multiple sentences can be queued to stream continuously around the circle.

## Quick start

```bash
# Single looping message
uv run python -m experimance_display.cli marquee "Hello world"

# One-shot sentence, medium speed
uv run python -m experimance_display.cli marquee "One-time message" --no-loop --write-speed 12

# Queue several sentences back-to-back (each starts where the previous ended)
uv run python -m experimance_display.cli marquee "First sentence" --no-loop
uv run python -m experimance_display.cli marquee "Second sentence" --no-loop
uv run python -m experimance_display.cli marquee "Third sentence" --no-loop
```

## CLI reference

```
uv run python -m experimance_display.cli marquee <content> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `content` | optional (random if omitted) | Text to display |
| `--id` | auto UUID | Unique text ID — reuse to replace content on a live item |
| `--duration` | infinite | Remove item after this many seconds |
| `--font-size` | 28 | Character size in pixels |
| `--font-name` | `"Arial"` | Font family |
| `--write-speed` | 8.0 | Characters written per second |
| `--no-loop` | *(loop on)* | Stop after one pass instead of repeating |
| `--no-follow` | *(follow on)* | Start at `start_angle` instead of picking up from previous item |
| `--gap-slots` | 6 | Clear-space slots kept ahead of the write head |
| `--radius-scale` | 0.97 | Position of the text arc relative to mask radius (1.0 = exactly on edge) |
| `--char-spacing` | 1.0 | Angular slot-width multiplier — >1.0 spreads chars apart |
| `--start-angle` | 0.0 | Starting position in degrees CW from 12 o'clock (ignored when follow is active) |

## Follow mode and queueing

By default `follow=True`.  The renderer tracks where each item's last character was placed and starts the next item from that slot.

**If a no-loop item is still printing when a new message arrives**, the new message is queued.  It starts automatically, with no gap, the moment the current item's characters finish fading away.  This makes it straightforward to stream a sequence of sentences:

```bash
uv run python -m experimance_display.cli marquee "Line one" --no-loop
uv run python -m experimance_display.cli marquee "Line two" --no-loop   # queued, starts right after
uv run python -m experimance_display.cli marquee "Line three" --no-loop # queued behind line two
```

To disable follow and always start at the same position:

```bash
uv run python -m experimance_display.cli marquee "Always from top" --no-follow --start-angle 0
```

## ZMQ message API

Send a `DisplayText` message with `speaker = "marquee"`:

```json
{
    "type":     "DisplayText",
    "text_id":  "my-id",
    "speaker":  "marquee",
    "content":  "Your text here",
    "duration": 30.0,
    "style": {
        "font_name":              "Arial",
        "font_size":              28,
        "color":                  [255, 255, 255, 200],
        "write_speed":            8.0,
        "loop":                   false,
        "gap_slots":              6,
        "char_fade_in_duration":  0.05,
        "gap_fade_out_duration":  0.15,
        "item_fade_out_duration": 1.0,
        "radius_scale":           0.97,
        "char_spacing":           1.0,
        "start_angle":            0.0,
        "follow":                 true
    }
}
```

All `style` fields are optional — any field omitted inherits from `[text_styles.marquee]` in `display.toml`.

Remove an item early:

```json
{"type": "RemoveText", "text_id": "my-id"}
```

Sending a `DisplayText` with a `text_id` that already exists replaces the content immediately.

## Configuration (`display.toml`)

```toml
[text_styles.marquee]
font_name               = "Arial"
font_size               = 28
color                   = [255, 255, 255, 200]
write_speed             = 8.0
loop                    = true
gap_slots               = 6
char_fade_in_duration   = 0.05
gap_fade_out_duration   = 0.15
item_fade_out_duration  = 1.0
radius_scale            = 0.97
char_spacing            = 1.0
start_angle             = 0.0
follow                  = true
```

### Field reference

| Field | Default | Description |
|-------|---------|-------------|
| `font_name` | `"Arial"` | Font family name |
| `font_size` | `28` | Character size in pixels |
| `color` | `[255,255,255,255]` | RGBA text colour |
| `write_speed` | `8.0` | Characters written per second |
| `loop` | `true` | Repeat content continuously; set `false` for one-shot sentences |
| `gap_slots` | `6` | Slots ahead of write head kept invisible — larger values create more clear space |
| `char_fade_in_duration` | `0.05` | Seconds each new character fades in |
| `gap_fade_out_duration` | `0.15` | Seconds a character takes to fade out when the write head catches up to it |
| `item_fade_out_duration` | `1.0` | Seconds all characters take to fade out when the whole item expires |
| `radius_scale` | `0.97` | Arc radius as a fraction of the mask circle radius.  Values slightly below 1.0 keep text inside the mask edge |
| `char_spacing` | `1.0` | Slot-width multiplier.  `>1.0` spreads characters further apart; `<1.0` packs them tighter |
| `start_angle` | `0.0` | Default starting position in degrees CW from 12 o'clock.  Ignored when `follow=true` and a previous item has run |
| `follow` | `true` | Start new items where the previous one's last character ended |

## How it works

### Circle geometry

The arc exactly matches the `MaskRenderer` circular mask: centre `(w//2, h//2)`, radius `min(w,h)//2`.  `radius_scale` shifts the character centres inward from that edge.

### Slot layout

The circle is divided into equal angular slots.  Slot width is estimated as:

```
char_width_px  = font_size × 0.6 × char_spacing
total_slots    = circumference / char_width_px
slot_angle_rad = 2π / total_slots
```

Slot 0 begins at `start_angle` degrees clockwise from 12 o'clock.  Slots increase clockwise.

### Write head

Each `update()` tick the write head advances by `write_speed × dt` slots.  At each new slot the next character from `content` is placed as a pyglet `Label`, rotated to face inward.  The character immediately starts fading in.

When `loop=False` and `content` is exhausted, the write head keeps advancing empty slots until the gap zone has swept past all placed characters, then the item auto-expires.

### Gap zone

The `gap_slots` slots immediately ahead of the write head are the gap zone.  Any character whose slot enters the zone starts a fast fade-out (`gap_fade_out_duration`).  This produces the visible clear space in front of the cursor.

### Expiry

- **Duration expiry**: when `duration` seconds elapse, `start_expiry()` is called — all characters begin fading out simultaneously using `item_fade_out_duration`.
- **Auto-expiry (no-loop)**: once all characters have faded, the item is removed and the follow angle is snapshotted.
- **Explicit removal**: `RemoveText` triggers the same expiry path.

The follow-mode angle snapshot is taken at the moment expiry begins (not after the fade completes), so queued items start with no perceptible gap.

### Queueing

When a new no-loop `follow` message arrives and another no-loop `follow` item is still writing, the message is held in an internal FIFO queue.  As soon as the active item auto-expires the next queued message is dispatched via `handle_text_overlay`, picking up the snapshotted angle.

### Coordinate system

pyglet uses OpenGL upward-Y coordinates.  12 o'clock = angle `+π/2`.  Clockwise on screen = decreasing angle.  Character rotation:

```
angle(slot)    = π/2 − radians(start_angle) − slot × char_spacing_rad
rotation_deg   = 90 − degrees(angle)
```

## Implementation

| File | Role |
|------|------|
| `renderers/marquee_text_renderer.py` | Renderer, `MarqueeItem`, `MarqueeChar` |
| `config.py` — `MarqueeStyleConfig` | All config fields and defaults |
| `display_service.py` | Routes `speaker="marquee"` messages to the renderer |
| `cli.py` — `marquee` subcommand | CLI wrapper |
| `projects/experimance/display.toml` | Per-installation config values |

#!/usr/bin/env python3
"""
Marquee Text Renderer for the Display Service.

Full documentation: ``services/display/docs/marquee.md``

Renders text character-by-character around a circular arc that matches the
display mask geometry.  Each character is a pyglet Label positioned and
rotated to face inward, travelling clockwise from 12 o'clock.

Architecture overview
---------------------
``MarqueeTextRenderer`` is a ``LayerRenderer`` registered as the
``"marquee_text"`` layer.  It maintains a dict of active ``MarqueeItem``
objects and a FIFO queue for pending no-loop messages.

``MarqueeItem`` holds all per-item state: content, rendering params, write-head
position, per-character ``MarqueeChar`` instances (each wrapping a
``pyglet.text.Label``), and expiry state.

Each ``update(dt)`` tick:
  1. Check whole-item expiry (duration elapsed or ``RemoveText`` received).
  2. Advance the write head — place characters one slot at a time.
  3. Auto-expire no-loop items once all characters have faded away.
  4. Apply gap-zone fade-out to characters inside the clear-space ahead of
     the write head.
  5. Tick per-character fade-in / fade-out animations, removing fully-faded
     chars from the dict.
  6. Flush the queue if a blocking no-loop item just finished.

Coordinate system
-----------------
pyglet uses OpenGL upward-Y coordinates.  12 o'clock is angle +π/2.
Clockwise on screen = decreasing angle.  Character rotation is:

    rotation_deg = 90 - degrees(angle)

Slot 0 starts at ``start_angle`` degrees clockwise from 12 o'clock:

    angle(slot) = π/2 - radians(start_angle) - slot * char_spacing_rad

Follow mode
-----------
When ``follow=True`` (default), a new item picks up exactly where the
previous one's last written character ended.  If a no-loop item is still
printing when the next message arrives, the new message is queued and starts
automatically once the current item's gap zone has cleared.

Message API
-----------
Send a ``DisplayText`` ZMQ message::

    {
        "type":     "DisplayText",
        "text_id":  "my-id",          # unique string; reuse to replace content
        "speaker":  "marquee",
        "content":  "Your text here",
        "duration": 30.0,             # optional; None = infinite
        "style": {                    # all fields optional — override config
            "font_name":              "Arial",
            "font_size":              28,
            "color":                  [255, 255, 255, 200],
            "write_speed":            8.0,
            "loop":                   True,
            "gap_slots":              6,
            "char_fade_in_duration":  0.05,
            "gap_fade_out_duration":  0.15,
            "item_fade_out_duration": 1.0,
            "radius_scale":           0.97,
            "char_spacing":           1.0,
            "start_angle":            0.0,
            "follow":                 True
        }
    }

Remove with ``RemoveText``::

    {"type": "RemoveText", "text_id": "my-id"}

CLI usage
---------
See ``services/display/docs/marquee.md``.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyglet
from pyglet.text import Label

from experimance_common.schemas import DisplayText
from experimance_common.zmq.config import MessageDataType
from experimance_display.config import DisplayServiceConfig
from .layer_manager import LayerRenderer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal data models
# ---------------------------------------------------------------------------


@dataclass
class MarqueeChar:
    """State for a single character placed on the arc."""

    char: str
    label: Label
    slot_index: int  # integer slot index around the circle
    opacity: float = 0.0
    # fade-in state
    is_fading_in: bool = True
    fade_in_progress: float = 0.0
    # fade-out state (gap zone or item expiry)
    is_fading_out: bool = False
    fade_out_progress: float = 0.0

    def start_fade_out(self):
        if not self.is_fading_out:
            self.is_fading_out = True
            self.is_fading_in = False
            self.fade_out_progress = 0.0

    @property
    def fully_faded(self) -> bool:
        return self.is_fading_out and self.fade_out_progress >= 1.0


@dataclass
class MarqueeItem:
    """State for one active marquee text item."""

    text_id: str
    content: str  # full string to write
    creation_time: float

    # rendering params (merged from config + per-message overrides)
    font_name: str
    font_size: int
    color: Tuple[int, int, int, int]
    write_speed: float  # chars/sec
    loop: bool
    gap_slots: int
    char_fade_in_duration: float
    gap_fade_out_duration: float
    item_fade_out_duration: float
    radius_scale: float
    char_spacing: float  # multiplier on estimated char width for slot spacing
    start_angle: float  # degrees clockwise from 12 o'clock
    follow: bool  # whether this item was started in follow mode

    # geometry (set by renderer after creation)
    total_slots: int = 0  # total number of slots around the full circle
    char_spacing_rad: float = 0.0  # angular width of one slot in radians

    # runtime state
    content_index: int = 0  # next character in content[] to place
    write_head_slot: int = 0  # slot index the write head is at
    write_timer: float = 0.0  # accumulator for write_speed timing
    last_written_slot: int = 0  # slot where the most recent character was placed

    # alive characters keyed by slot_index
    chars: Dict[int, MarqueeChar] = field(default_factory=dict)

    # True once at least one character has been placed; guards fully_gone so
    # an item with an empty chars dict (not yet started) isn't immediately
    # considered done when is_expiring is set.
    has_written: bool = False

    # whole-item expiry fade
    duration: Optional[float] = None
    is_expiring: bool = False
    expire_progress: float = 0.0

    def is_expired(self) -> bool:
        if self.duration is None:
            return False
        return (time.time() - self.creation_time) >= self.duration

    def start_expiry(self):
        if not self.is_expiring:
            self.is_expiring = True
            # kick off fade-out on every visible char
            for ch in self.chars.values():
                ch.start_fade_out()

    @property
    def fully_gone(self) -> bool:
        """True when the item is expiring and all chars have faded out.

        If ``chars`` is empty the item is done regardless of whether it ever
        placed a character (handles both the gap-zone-swept case and an
        immediate RemoveText before writing started).  If chars are present,
        every one must be fully faded.
        """
        if not self.is_expiring:
            return False
        if not self.chars:
            return True
        return all(c.fully_faded for c in self.chars.values())


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class MarqueeTextRenderer(LayerRenderer):
    """Renders text items character-by-character around a circular arc."""

    def __init__(
        self,
        config: DisplayServiceConfig,
        window: pyglet.window.BaseWindow,
        batch: pyglet.graphics.Batch,
        order: int = 3,
    ):
        super().__init__(config=config, window=window, batch=batch, order=order)

        self._items: Dict[str, MarqueeItem] = {}
        self._visible = True
        self._opacity = 1.0

        # Capture the running event loop at init time so update() (which is
        # called synchronously from the pyglet clock) can safely schedule
        # coroutines without relying on the deprecated get_event_loop().
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        # Pending messages (DisplayText objects) waiting for active no-loop
        # items to finish before starting.
        self._queue: List[DisplayText] = []

        # Tracks the write-head position (degrees CW from 12 o'clock) of the
        # most recently completed/removed item so follow-mode items can pick up
        # where the last one left off.  None = no prior history.
        self._last_write_head_angle: Optional[float] = None

        # Pre-compute circle geometry once; update on resize
        self._cx, self._cy, self._radius = self._compute_circle()

        logger.info(
            f"MarqueeTextRenderer initialised — circle centre=({self._cx},{self._cy}) "
            f"radius={self._radius}"
        )

    # ------------------------------------------------------------------
    # LayerRenderer interface
    # ------------------------------------------------------------------

    @property
    def is_visible(self) -> bool:
        return self._visible and bool(self._items)

    @property
    def opacity(self) -> float:
        return self._opacity

    def resize(self, new_size: Tuple[int, int]):
        self._cx, self._cy, self._radius = self._compute_circle()
        # Rebuild all char labels at new positions
        for item in self._items.values():
            self._rebuild_item_labels(item)

    def update(self, dt: float):
        expired_ids: List[str] = []

        for item in self._items.values():
            # 1. Whole-item expiry
            if item.is_expiring:
                if item.fully_gone:
                    expired_ids.append(item.text_id)
                    continue
                # individual chars handle their own fade-out below
            elif item.is_expired():
                self._snapshot_write_head(item)
                item.start_expiry()

            if item.is_expiring:
                # update individual chars and skip write-head logic
                self._update_chars(item, dt)
                continue

            # 2. Advance write head, applying gap fadeout after each character
            # so chars written earlier in the same tick aren't immediately swept.
            item.write_timer += dt
            chars_to_write = int(item.write_timer * item.write_speed)
            if chars_to_write > 0:
                item.write_timer -= chars_to_write / item.write_speed
                for _ in range(chars_to_write):
                    self._write_next_char(item)
                    self._apply_gap_fadeout(item)

            # Auto-expire no-loop items once content is written and all
            # characters have been swept away by the gap zone.
            # Guard: only trigger if the write head has actually moved past all
            # content (content_index > 0 ensures at least one char was placed).
            if (
                not item.loop
                and item.content_index > 0
                and item.content_index >= len(item.content)
                and not item.chars
            ):
                logger.info(
                    f"MarqueeTextRenderer: auto-expiring {item.text_id!r} "
                    f"(no-loop, content done, chars empty, head={item.write_head_slot})"
                )
                self._snapshot_write_head(item)
                item.start_expiry()

            # 3. Update per-char animation
            self._update_chars(item, dt)

        for tid in expired_ids:
            self._delete_item(tid)

        # After deletions, flush the queue if no no-loop follow blocker remains.
        if self._queue and self._find_noloop_follow_blocker() is None:
            next_msg = self._queue.pop(0)
            self._loop.call_soon(
                lambda m=next_msg: asyncio.ensure_future(self.handle_text_overlay(m))
            )

    async def cleanup(self):
        self._queue.clear()
        for item in list(self._items.values()):
            self._delete_item(item.text_id)
        logger.info("MarqueeTextRenderer cleanup complete")

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def handle_text_overlay(self, message: DisplayText):
        """Handle a DisplayText message (speaker must be 'marquee')."""
        try:
            text_id = message.get("text_id")
            content = message.get("content")
            if not text_id or not content or not content.strip():
                logger.error("MarqueeTextRenderer: missing text_id or content")
                return

            logger.info(
                f"MarqueeTextRenderer: handle_text_overlay received text_id={text_id!r} "
                f"content={content!r} active_items={list(self._items.keys())} "
                f"last_angle={self._last_write_head_angle}"
            )

            # If there is an active no-loop follow item still writing (not yet
            # expiring), queue this message so it starts right after.
            blocker = self._find_noloop_follow_blocker()
            if blocker is not None:
                logger.debug(
                    f"MarqueeTextRenderer: queuing {text_id!r} behind active item {blocker!r}"
                )
                self._queue.append(message)
                return

            # Always append a trailing space — acts as a natural word gap and
            # ensures the loop doesn't run the last character directly into the
            # first character on wrap-around.
            if not content.endswith(" "):
                content = content + " "

            duration = message.get("duration", None)
            style_overrides = message.get("style") or {}

            # Resolve style from config + overrides
            cfg = self.config.text_styles.marquee
            font_name = style_overrides.get("font_name", cfg.font_name)
            font_size = style_overrides.get("font_size", cfg.font_size)
            raw_color = style_overrides.get("color", cfg.color)
            color = tuple(raw_color) if not isinstance(raw_color, tuple) else raw_color
            write_speed = float(style_overrides.get("write_speed", cfg.write_speed))
            loop = bool(style_overrides.get("loop", cfg.loop))
            gap_slots = int(style_overrides.get("gap_slots", cfg.gap_slots))
            char_fade_in = float(
                style_overrides.get("char_fade_in_duration", cfg.char_fade_in_duration)
            )
            gap_fade_out = float(
                style_overrides.get("gap_fade_out_duration", cfg.gap_fade_out_duration)
            )
            item_fade_out = float(
                style_overrides.get("item_fade_out_duration", cfg.item_fade_out_duration)
            )
            radius_scale = float(style_overrides.get("radius_scale", cfg.radius_scale))
            char_spacing = float(style_overrides.get("char_spacing", cfg.char_spacing))
            follow = bool(style_overrides.get("follow", cfg.follow))

            # Resolve start angle:
            # - Explicit override always wins.
            # - follow=True: use snapshotted angle from the last finished item.
            # - Otherwise fall back to cfg.start_angle.
            if "start_angle" in style_overrides:
                start_angle = float(style_overrides["start_angle"])
            elif follow and self._last_write_head_angle is not None:
                start_angle = self._last_write_head_angle
                logger.debug(
                    f"MarqueeTextRenderer: follow mode — starting {text_id!r} "
                    f"at {start_angle:.1f}° (from previous write head)"
                )
            else:
                start_angle = float(cfg.start_angle)

            if text_id in self._items:
                # Replace content on existing item (streaming update).
                # Apply any new style overrides so the restarted item uses the
                # latest font, colour, speed, and geometry settings.
                logger.debug(f"MarqueeTextRenderer: replacing content for {text_id!r}")
                old = self._items[text_id]
                # Delete existing pyglet labels before discarding refs
                for ch in old.chars.values():
                    try:
                        ch.label.delete()
                    except Exception:
                        pass
                old.chars = {}
                old.content = content
                old.content_index = 0
                old.write_head_slot = 0
                old.last_written_slot = 0
                old.has_written = False
                old.write_timer = 0.0
                old.is_expiring = False
                old.creation_time = time.time()
                old.duration = duration
                # Style fields
                old.font_name = font_name
                old.font_size = font_size
                old.color = color
                old.write_speed = write_speed
                old.loop = loop
                old.gap_slots = gap_slots
                old.char_fade_in_duration = char_fade_in
                old.gap_fade_out_duration = gap_fade_out
                old.item_fade_out_duration = item_fade_out
                old.follow = follow
                # Recompute geometry if radius_scale or char_spacing changed
                if old.radius_scale != radius_scale or old.char_spacing != char_spacing:
                    old.radius_scale = radius_scale
                    old.char_spacing = char_spacing
                    radius = self._radius * radius_scale
                    old.total_slots, old.char_spacing_rad = self._compute_slots(
                        font_size, radius, char_spacing
                    )
                return

            radius = self._radius * radius_scale
            total_slots, char_spacing_rad = self._compute_slots(font_size, radius, char_spacing)

            item = MarqueeItem(
                text_id=text_id,
                content=content,
                creation_time=time.time(),
                font_name=font_name,
                font_size=font_size,
                color=color,
                write_speed=write_speed,
                loop=loop,
                gap_slots=gap_slots,
                char_fade_in_duration=char_fade_in,
                gap_fade_out_duration=gap_fade_out,
                item_fade_out_duration=item_fade_out,
                radius_scale=radius_scale,
                char_spacing=char_spacing,
                start_angle=start_angle,
                follow=follow,
                total_slots=total_slots,
                char_spacing_rad=char_spacing_rad,
                duration=duration,
            )
            self._items[text_id] = item
            logger.info(
                f"MarqueeTextRenderer: added {text_id!r} "
                f"({total_slots} slots, write_speed={write_speed}/s, loop={loop}, "
                f"start_angle={start_angle:.1f}°, follow={follow})"
            )

        except Exception:
            logger.exception("MarqueeTextRenderer: error in handle_text_overlay")

    async def handle_remove_text(self, message: MessageDataType):
        """Handle a RemoveText message."""
        try:
            text_id = message.get("text_id")
            if text_id in self._items:
                logger.info(f"MarqueeTextRenderer: removing {text_id!r}")
                item = self._items[text_id]
                self._snapshot_write_head(item)
                item.start_expiry()
            # silently ignore unknown IDs
        except Exception:
            logger.exception("MarqueeTextRenderer: error in handle_remove_text")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_noloop_follow_blocker(self) -> Optional[str]:
        """Return the text_id of the first active no-loop follow item that is
        still writing (not yet expiring), or None if the queue can run."""
        for item in self._items.values():
            if item.follow and not item.loop and not item.is_expiring:
                return item.text_id
        return None

    def _compute_circle(self) -> Tuple[int, int, int]:
        """Return (cx, cy, radius) matching the MaskRenderer circular mask."""
        w, h = self.window.width, self.window.height
        cx = w // 2
        cy = h // 2
        radius = min(w, h) // 2
        return cx, cy, radius

    def _compute_slots(
        self, font_size: int, radius: float, char_spacing: float = 1.0
    ) -> Tuple[int, float]:
        """
        Estimate how many character slots fit around the circle and the
        angular spacing per slot.

        We use a rough monospace-ish estimate: character width ≈ font_size * 0.6.
        ``char_spacing`` is a multiplier — values > 1.0 spread characters apart,
        < 1.0 packs them tighter.
        """
        char_width_px = font_size * 0.6 * char_spacing
        circumference = 2.0 * math.pi * radius
        total_slots = max(4, int(circumference / char_width_px))
        char_spacing_rad = (2.0 * math.pi) / total_slots
        return total_slots, char_spacing_rad

    def _slot_to_angle(self, slot_index: int, item: MarqueeItem) -> float:
        """Convert a slot index to a radian angle on the circle.

        Pyglet uses OpenGL coordinates where Y increases *upward*.

        Slot 0 starts at ``start_angle`` degrees clockwise from 12 o'clock.
        In upward-Y space, 12 o'clock is +π/2.  Clockwise on screen means
        *decreasing* angle, so we subtract both the start offset and the
        slot progression.
        """
        start_rad = math.radians(item.start_angle)
        return math.pi / 2.0 - start_rad - slot_index * item.char_spacing_rad

    def _slot_to_screen(self, slot_index: int, item: MarqueeItem) -> Tuple[float, float, float]:
        """Return (x, y, rotation_degrees) for a slot.

        Characters are oriented so their **baseline faces outward** and their
        top (ascender) points toward the circle centre — i.e. reading inward.

        Pyglet ``rotation`` is counter-clockwise in degrees.

        At 12 o'clock (angle = +π/2):
          - the character sits at the top, upright → rotation = 0°
        At 3 o'clock (angle = 0):
          - the character should be rotated 90° CCW → rotation = 90°
        General formula: rotation = 90° − angle_in_degrees
          (equivalently: -(angle_deg - 90))
        """
        radius = self._radius * item.radius_scale
        angle = self._slot_to_angle(slot_index, item)
        x = self._cx + radius * math.cos(angle)
        y = self._cy + radius * math.sin(angle)
        rotation_deg = 90.0 - math.degrees(angle)
        return x, y, rotation_deg

    def _make_char_label(
        self, char: str, slot_index: int, item: MarqueeItem, opacity_byte: int = 0
    ) -> Label:
        x, y, rot = self._slot_to_screen(slot_index, item)
        color_with_alpha = (*item.color[:3], opacity_byte)
        label = Label(
            text=char,
            font_name=item.font_name,
            font_size=item.font_size,
            color=color_with_alpha,
            x=x,
            y=y,
            anchor_x="center",
            anchor_y="center",
            rotation=rot,
            batch=self.batch,
            group=self,
        )
        return label

    def _write_next_char(self, item: MarqueeItem):
        """Place the next character from item.content at the write head slot."""
        if item.content_index >= len(item.content):
            if item.loop:
                item.content_index = 0
            else:
                # Content exhausted — keep advancing the write head so the gap
                # zone sweeps over the last characters, then stop.
                item.write_head_slot = (item.write_head_slot + 1) % item.total_slots
                return

        char = item.content[item.content_index]
        item.content_index += 1
        slot = item.write_head_slot

        # If the slot is occupied, start fading out the old character
        if slot in item.chars:
            item.chars[slot].start_fade_out()

        # Place new character (starts invisible, fades in)
        label = self._make_char_label(char, slot, item, opacity_byte=0)
        item.chars[slot] = MarqueeChar(
            char=char,
            label=label,
            slot_index=slot,
            opacity=0.0,
            is_fading_in=True,
        )
        item.has_written = True
        item.last_written_slot = slot

        # Advance write head
        item.write_head_slot = (item.write_head_slot + 1) % item.total_slots

    def _apply_gap_fadeout(self, item: MarqueeItem):
        """
        Any char whose slot falls within gap_slots *ahead* of the write head
        (i.e. slots the write head is about to reach) should start fading out,
        creating visible clear space in front of the cursor.
        """
        gap = item.gap_slots
        total = item.total_slots
        head = item.write_head_slot
        for offset in range(1, gap + 1):
            # slots [head, head+1, ..., head+gap-1] are in the clear-space zone
            ahead_slot = (head + offset - 1) % total
            if ahead_slot in item.chars:
                ch = item.chars[ahead_slot]
                if not ch.is_fading_out:
                    ch.start_fade_out()

    def _update_chars(self, item: MarqueeItem, dt: float):
        """Update per-character fade animation and remove fully-faded chars."""
        slots_to_remove = []
        for slot, ch in item.chars.items():
            fade_out_dur = (
                item.gap_fade_out_duration if not item.is_expiring else item.item_fade_out_duration
            )

            if ch.is_fading_in:
                if item.char_fade_in_duration <= 0:
                    ch.opacity = 1.0
                    ch.is_fading_in = False
                    ch.fade_in_progress = 1.0
                else:
                    ch.fade_in_progress += dt / item.char_fade_in_duration
                    if ch.fade_in_progress >= 1.0:
                        ch.fade_in_progress = 1.0
                        ch.is_fading_in = False
                    ch.opacity = ch.fade_in_progress

            elif ch.is_fading_out:
                if fade_out_dur <= 0:
                    ch.opacity = 0.0
                    ch.fade_out_progress = 1.0
                else:
                    ch.fade_out_progress += dt / fade_out_dur
                    if ch.fade_out_progress >= 1.0:
                        ch.fade_out_progress = 1.0
                        ch.opacity = 0.0
                    else:
                        ch.opacity = 1.0 - ch.fade_out_progress

            else:
                ch.opacity = 1.0

            # Apply opacity to label; use item.color as the source of truth
            # for RGB rather than reading it back from the label.
            alpha = int(ch.opacity * 255)
            r, g, b = item.color[:3]
            ch.label.color = (r, g, b, alpha)

            if ch.fully_faded:
                slots_to_remove.append(slot)

        for slot in slots_to_remove:
            item.chars.pop(slot, None)

    def _rebuild_item_labels(self, item: MarqueeItem):
        """Rebuild all char labels after a geometry change (resize).

        Because ``total_slots`` may change after a resize, all existing slot
        indices would be invalid.  The safest approach is to delete every
        existing label and reset the write-head so the item starts fresh from
        its current position.  This is a visible reset, but it avoids rendering
        artifacts from stale slot indices.
        """
        radius = self._radius * item.radius_scale
        item.total_slots, item.char_spacing_rad = self._compute_slots(
            item.font_size, radius, item.char_spacing
        )
        # Delete all existing pyglet labels
        for ch in item.chars.values():
            try:
                ch.label.delete()
            except Exception:
                pass
        item.chars = {}
        # Reset write-head state so the item re-writes cleanly from slot 0
        item.write_head_slot = 0
        item.content_index = 0
        item.last_written_slot = 0
        item.has_written = False

    def _delete_item(self, text_id: str):
        """Remove an item and delete all its pyglet labels."""
        item = self._items.pop(text_id, None)
        if item:
            for ch in item.chars.values():
                try:
                    ch.label.delete()
                except Exception:
                    pass
            logger.debug(f"MarqueeTextRenderer: deleted item {text_id!r}")

    def _snapshot_write_head(self, item: MarqueeItem):
        """Record the write-head angle for follow-mode continuity.

        Uses the slot immediately after the last written character — not the
        blind-advancing write_head_slot which may have drifted far ahead
        through the gap zone.
        """
        if item.follow:
            next_slot = (item.last_written_slot + 1) % item.total_slots
            head_angle_rad = self._slot_to_angle(next_slot, item)
            raw_deg = math.degrees(math.pi / 2.0 - head_angle_rad)
            self._last_write_head_angle = raw_deg % 360.0
            logger.debug(
                f"MarqueeTextRenderer: snapshotted write head at "
                f"{self._last_write_head_angle:.1f}° from item {item.text_id!r} "
                f"(last_written_slot={item.last_written_slot})"
            )

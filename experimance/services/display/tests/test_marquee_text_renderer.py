#!/usr/bin/env python3
"""
Unit tests for MarqueeTextRenderer.

Tests are grouped by concern and deliberately avoid pyglet window creation
(all pyglet.text.Label calls are mocked).  The goal is to lock in the
behaviour fixed during the bug-squash session so regressions are caught
immediately.

Covered areas
-------------
- MarqueeItem / MarqueeChar data-model invariants
  - fully_gone with empty chars (vacuous-true guard)
  - fully_gone with chars present but not yet faded
  - fully_gone after immediate RemoveText (before writing)
  - has_written flag lifecycle
- Renderer helper logic (no window needed)
  - _compute_slots geometry
  - _slot_to_angle / _slot_to_screen coordinate maths
  - _apply_gap_fadeout marks correct slots
  - _write_next_char places char, advances write head, sets has_written
  - _write_next_char loop wrap
  - _write_next_char no-loop: advances head past end without placing chars
  - _update_chars fade-in and fade-out progression, label colour update
  - _update_chars removes fully-faded chars
  - _snapshot_write_head angle calculation
- handle_text_overlay (async, label creation mocked)
  - rejects missing text_id
  - rejects whitespace-only content
  - creates new item with resolved style
  - content replacement deletes old labels and applies new style
  - content replacement resets has_written / write-head
  - geometry recomputed on radius_scale change
  - queues message when no-loop follow blocker is active
- handle_remove_text
  - triggers expiry on known item
  - silently ignores unknown ID
- update() lifecycle
  - expired item is removed after fully_gone
  - auto-expire for no-loop item (content done, chars empty)
  - queue flushed once blocker expires
  - item with duration is expired after time elapses
"""

import asyncio
import math
import time
from dataclasses import replace
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.display.src.experimance_display.renderers.marquee_text_renderer import (
    MarqueeChar,
    MarqueeItem,
    MarqueeTextRenderer,
)
from services.display.src.experimance_display.config import (
    DisplayServiceConfig,
    DisplayConfig,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_label():
    label = MagicMock()
    label.color = (255, 255, 255, 0)
    return label


def _make_char(slot=0, opacity=1.0, fading_in=False, fading_out=False, fade_out_progress=0.0):
    label = _make_mock_label()
    ch = MarqueeChar(
        char="A",
        label=label,
        slot_index=slot,
        opacity=opacity,
        is_fading_in=fading_in,
        fade_out_progress=fade_out_progress,
    )
    if fading_out:
        ch.is_fading_out = True
        ch.is_fading_in = False
    return ch


def _make_item(**kwargs) -> MarqueeItem:
    """Create a minimal MarqueeItem with sensible defaults."""
    defaults = dict(
        text_id="test",
        content="Hello ",
        creation_time=time.time(),
        font_name="Arial",
        font_size=28,
        color=(255, 255, 255, 255),
        write_speed=8.0,
        loop=True,
        gap_slots=3,
        char_fade_in_duration=0.05,
        gap_fade_out_duration=0.15,
        item_fade_out_duration=1.0,
        radius_scale=0.97,
        char_spacing=1.0,
        start_angle=0.0,
        follow=True,
        total_slots=40,
        char_spacing_rad=(2 * math.pi) / 40,
    )
    defaults.update(kwargs)
    return MarqueeItem(**defaults)


def _make_renderer(width=800, height=600, loop=None):
    """Create a MarqueeTextRenderer with all pyglet deps mocked out.

    ``pyglet.graphics.Group.__init__`` stores ``order`` as a read-only
    property, so we patch Group.__init__ to be a no-op and set the
    private attribute directly.
    """
    config = DisplayServiceConfig(display=DisplayConfig(headless=True, resolution=(width, height)))
    window = Mock()
    window.width = width
    window.height = height
    batch = Mock()

    import pyglet.graphics

    # Patch Group.__init__ so we don't need a real display context, and
    # patch the asyncio helpers so __init__ doesn't touch an event loop.
    with (
        patch.object(pyglet.graphics.Group, "__init__", return_value=None),
        patch("asyncio.get_running_loop", side_effect=RuntimeError),
        patch("asyncio.get_event_loop", return_value=MagicMock()),
    ):
        renderer = MarqueeTextRenderer.__new__(MarqueeTextRenderer)
        # Set the private slot that pyglet.graphics.Group.order reads from
        renderer._order = 3
        renderer.config = config
        renderer.window = window
        renderer.batch = batch
        renderer._items = {}
        renderer._visible = True
        renderer._opacity = 1.0
        renderer._loop = MagicMock()
        renderer._queue = []
        renderer._last_write_head_angle = None
        renderer._cx = width // 2
        renderer._cy = height // 2
        renderer._radius = min(width, height) // 2

    return renderer


# ---------------------------------------------------------------------------
# MarqueeChar tests
# ---------------------------------------------------------------------------


class TestMarqueeChar:
    def test_fully_faded_requires_fading_out(self):
        ch = _make_char(opacity=0.0, fading_out=False)
        assert ch.fully_faded is False

    def test_fully_faded_requires_progress_complete(self):
        ch = _make_char(fading_out=True, fade_out_progress=0.99)
        assert ch.fully_faded is False

    def test_fully_faded_true_when_done(self):
        ch = _make_char(fading_out=True, fade_out_progress=1.0)
        assert ch.fully_faded is True

    def test_start_fade_out_idempotent(self):
        ch = _make_char()
        ch.start_fade_out()
        ch.start_fade_out()
        assert ch.is_fading_out is True
        assert ch.fade_out_progress == 0.0  # not reset on second call


# ---------------------------------------------------------------------------
# MarqueeItem.fully_gone tests
# ---------------------------------------------------------------------------


class TestMarqueeItemFullyGone:
    def test_not_expiring_never_fully_gone(self):
        item = _make_item()
        assert item.fully_gone is False

    def test_expiring_empty_chars_is_fully_gone(self):
        """Empty chars dict while expiring = done (gap zone swept everything,
        or RemoveText before writing started)."""
        item = _make_item()
        item.is_expiring = True
        item.chars = {}
        assert item.fully_gone is True

    def test_expiring_with_unfaded_chars_not_fully_gone(self):
        item = _make_item()
        item.is_expiring = True
        item.chars = {0: _make_char(slot=0, opacity=0.5, fading_out=True, fade_out_progress=0.5)}
        assert item.fully_gone is False

    def test_expiring_with_all_chars_faded_is_fully_gone(self):
        item = _make_item()
        item.is_expiring = True
        item.chars = {0: _make_char(slot=0, fading_out=True, fade_out_progress=1.0)}
        assert item.fully_gone is True

    def test_immediate_remove_before_writing_is_fully_gone(self):
        """has_written=False, is_expiring=True, chars={} — should be gone immediately."""
        item = _make_item()
        assert item.has_written is False
        item.is_expiring = True
        assert item.fully_gone is True  # no chars → done regardless of has_written

    def test_start_expiry_kicks_off_char_fadeouts(self):
        item = _make_item()
        ch0 = _make_char(slot=0)
        ch1 = _make_char(slot=1)
        item.chars = {0: ch0, 1: ch1}
        item.start_expiry()
        assert item.is_expiring is True
        assert ch0.is_fading_out is True
        assert ch1.is_fading_out is True


# ---------------------------------------------------------------------------
# Renderer geometry helpers
# ---------------------------------------------------------------------------


class TestComputeSlots:
    def test_slot_count_increases_with_radius(self):
        r = _make_renderer()
        slots_small, _ = r._compute_slots(28, 200)
        slots_large, _ = r._compute_slots(28, 400)
        assert slots_large > slots_small

    def test_slot_count_decreases_with_font_size(self):
        r = _make_renderer()
        slots_small_font, _ = r._compute_slots(14, 300)
        slots_large_font, _ = r._compute_slots(56, 300)
        assert slots_small_font > slots_large_font

    def test_char_spacing_rad_covers_full_circle(self):
        r = _make_renderer()
        total, spacing = r._compute_slots(28, 300)
        assert abs(total * spacing - 2 * math.pi) < 1e-9

    def test_minimum_four_slots(self):
        # Absurdly large font on a tiny radius
        r = _make_renderer()
        total, _ = r._compute_slots(500, 1)
        assert total == 4

    def test_char_spacing_multiplier(self):
        r = _make_renderer()
        slots_1x, _ = r._compute_slots(28, 300, char_spacing=1.0)
        slots_2x, _ = r._compute_slots(28, 300, char_spacing=2.0)
        assert slots_1x > slots_2x


class TestSlotToAngle:
    def test_slot_zero_at_12_oclock_with_zero_start_angle(self):
        r = _make_renderer()
        item = _make_item(start_angle=0.0)
        angle = r._slot_to_angle(0, item)
        assert abs(angle - math.pi / 2) < 1e-9

    def test_slot_increases_go_clockwise(self):
        """Increasing slot index should give decreasing angle (clockwise)."""
        r = _make_renderer()
        item = _make_item(start_angle=0.0)
        a0 = r._slot_to_angle(0, item)
        a1 = r._slot_to_angle(1, item)
        assert a1 < a0

    def test_start_angle_offsets_correctly(self):
        r = _make_renderer()
        item_0 = _make_item(start_angle=0.0)
        item_90 = _make_item(start_angle=90.0)
        angle_0 = r._slot_to_angle(0, item_0)
        angle_90 = r._slot_to_angle(0, item_90)
        assert abs(angle_0 - angle_90 - math.pi / 2) < 1e-9


class TestSlotToScreen:
    def test_slot_zero_at_top_for_zero_start_angle(self):
        r = _make_renderer(width=800, height=600)
        item = _make_item(start_angle=0.0, radius_scale=1.0)
        x, y, rot = r._slot_to_screen(0, item)
        # 12 o'clock: x ≈ cx, y ≈ cy + radius
        assert abs(x - r._cx) < 1.0
        assert abs(y - (r._cy + r._radius)) < 1.0

    def test_rotation_at_12_oclock_is_zero(self):
        r = _make_renderer()
        item = _make_item(start_angle=0.0, radius_scale=1.0)
        _, _, rot = r._slot_to_screen(0, item)
        assert abs(rot) < 1e-6


# ---------------------------------------------------------------------------
# _apply_gap_fadeout
# ---------------------------------------------------------------------------


class TestApplyGapFadeout:
    def test_chars_in_gap_zone_start_fading(self):
        r = _make_renderer()
        item = _make_item(gap_slots=3, total_slots=40)
        item.write_head_slot = 10
        # Place chars at slots 10, 11, 12 (head + 0..2 = gap zone)
        for s in [10, 11, 12]:
            item.chars[s] = _make_char(slot=s)
        r._apply_gap_fadeout(item)
        assert item.chars[10].is_fading_out is True
        assert item.chars[11].is_fading_out is True
        assert item.chars[12].is_fading_out is True

    def test_chars_outside_gap_zone_untouched(self):
        r = _make_renderer()
        item = _make_item(gap_slots=3, total_slots=40)
        item.write_head_slot = 10
        item.chars[13] = _make_char(slot=13)
        item.chars[9] = _make_char(slot=9)
        r._apply_gap_fadeout(item)
        assert item.chars[13].is_fading_out is False
        assert item.chars[9].is_fading_out is False

    def test_gap_zone_wraps_around_circle(self):
        r = _make_renderer()
        item = _make_item(gap_slots=3, total_slots=10)
        item.write_head_slot = 9  # Near the end
        # Slots 9, 0, 1 should be in the gap zone
        for s in [9, 0, 1]:
            item.chars[s] = _make_char(slot=s)
        r._apply_gap_fadeout(item)
        assert item.chars[9].is_fading_out is True
        assert item.chars[0].is_fading_out is True
        assert item.chars[1].is_fading_out is True

    def test_already_fading_chars_not_reset(self):
        r = _make_renderer()
        item = _make_item(gap_slots=3, total_slots=40)
        item.write_head_slot = 10
        ch = _make_char(slot=10, fading_out=True, fade_out_progress=0.5)
        item.chars[10] = ch
        r._apply_gap_fadeout(item)
        # fade_out_progress should NOT be reset to 0
        assert ch.fade_out_progress == 0.5


# ---------------------------------------------------------------------------
# _write_next_char
# ---------------------------------------------------------------------------


class TestWriteNextChar:
    def _patch_label(self, renderer):
        mock_label = _make_mock_label()
        renderer._make_char_label = Mock(return_value=mock_label)
        return mock_label

    def test_places_char_and_advances_head(self):
        r = _make_renderer()
        item = _make_item(content="AB ", total_slots=40, loop=False)
        self._patch_label(r)
        r._write_next_char(item)
        assert item.write_head_slot == 1
        assert 0 in item.chars
        assert item.chars[0].char == "A"

    def test_sets_has_written(self):
        r = _make_renderer()
        item = _make_item(content="AB ", total_slots=40)
        self._patch_label(r)
        assert item.has_written is False
        r._write_next_char(item)
        assert item.has_written is True

    def test_loop_wraps_content_index(self):
        r = _make_renderer()
        item = _make_item(content="AB ", total_slots=40, loop=True)
        item.content_index = 3  # at end
        self._patch_label(r)
        r._write_next_char(item)
        # Should have reset content_index to 0, then placed content[0]
        assert item.chars[0].char == "A"

    def test_no_loop_advances_head_past_end(self):
        r = _make_renderer()
        item = _make_item(content="AB ", total_slots=40, loop=False)
        item.content_index = 3  # exhausted
        item.write_head_slot = 5
        self._patch_label(r)
        r._write_next_char(item)
        # Head advances, but no new char is placed
        assert item.write_head_slot == 6
        assert len(item.chars) == 0

    def test_occupying_slot_starts_fadeout_on_old_char(self):
        r = _make_renderer()
        item = _make_item(content="B ", total_slots=40)
        old_ch = _make_char(slot=0)
        item.chars[0] = old_ch
        self._patch_label(r)
        r._write_next_char(item)
        assert old_ch.is_fading_out is True

    def test_updates_last_written_slot(self):
        r = _make_renderer()
        item = _make_item(content="ABC ", total_slots=40)
        self._patch_label(r)
        r._write_next_char(item)
        assert item.last_written_slot == 0
        r._write_next_char(item)
        assert item.last_written_slot == 1


# ---------------------------------------------------------------------------
# _update_chars
# ---------------------------------------------------------------------------


class TestUpdateChars:
    def test_fade_in_progresses(self):
        r = _make_renderer()
        item = _make_item(char_fade_in_duration=1.0)
        ch = _make_char(slot=0, fading_in=True, opacity=0.0)
        ch.fade_in_progress = 0.0
        item.chars = {0: ch}
        r._update_chars(item, dt=0.5)
        assert abs(ch.fade_in_progress - 0.5) < 1e-6
        assert abs(ch.opacity - 0.5) < 1e-6

    def test_fade_in_completes(self):
        r = _make_renderer()
        item = _make_item(char_fade_in_duration=0.1)
        ch = _make_char(slot=0, fading_in=True, opacity=0.0)
        ch.fade_in_progress = 0.0
        item.chars = {0: ch}
        r._update_chars(item, dt=0.2)
        assert ch.is_fading_in is False
        assert ch.opacity == 1.0

    def test_fade_out_progresses(self):
        r = _make_renderer()
        item = _make_item(gap_fade_out_duration=1.0)
        ch = _make_char(slot=0, fading_out=True, opacity=1.0)
        ch.fade_out_progress = 0.0
        item.chars = {0: ch}
        r._update_chars(item, dt=0.5)
        assert abs(ch.fade_out_progress - 0.5) < 1e-6
        assert abs(ch.opacity - 0.5) < 1e-6

    def test_fully_faded_char_removed(self):
        r = _make_renderer()
        item = _make_item(gap_fade_out_duration=0.1)
        ch = _make_char(slot=0, fading_out=True, opacity=0.1)
        ch.fade_out_progress = 0.0
        item.chars = {0: ch}
        r._update_chars(item, dt=0.2)
        assert 0 not in item.chars

    def test_label_colour_uses_item_color_not_label(self):
        """RGB for label colour update must come from item.color, not label.color."""
        r = _make_renderer()
        item = _make_item(color=(100, 150, 200, 255), char_fade_in_duration=0.0)
        ch = _make_char(slot=0, fading_in=True, opacity=0.0)
        ch.fade_in_progress = 0.0
        # Deliberately set label.color to different values to prove we DON'T read them
        ch.label.color = (0, 0, 0, 0)
        item.chars = {0: ch}
        r._update_chars(item, dt=0.1)
        # Should have written (100, 150, 200, alpha) — not (0, 0, 0, alpha)
        written_color = ch.label.color
        assert written_color[0] == 100
        assert written_color[1] == 150
        assert written_color[2] == 200

    def test_instant_fade_in_zero_duration(self):
        r = _make_renderer()
        item = _make_item(char_fade_in_duration=0.0)
        ch = _make_char(slot=0, fading_in=True, opacity=0.0)
        item.chars = {0: ch}
        r._update_chars(item, dt=0.016)
        assert ch.opacity == 1.0
        assert ch.is_fading_in is False

    def test_item_expiry_uses_item_fade_out_duration(self):
        r = _make_renderer()
        item = _make_item(item_fade_out_duration=2.0, gap_fade_out_duration=0.1)
        item.is_expiring = True
        ch = _make_char(slot=0, fading_out=True, opacity=1.0)
        ch.fade_out_progress = 0.0
        item.chars = {0: ch}
        r._update_chars(item, dt=1.0)  # 1s of 2s item fade-out
        # Should use 2.0s duration, so progress = 0.5
        assert abs(ch.fade_out_progress - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# _snapshot_write_head
# ---------------------------------------------------------------------------


class TestSnapshotWriteHead:
    def test_snapshot_stored_for_follow_item(self):
        r = _make_renderer()
        item = _make_item(follow=True, start_angle=0.0, total_slots=40)
        item.last_written_slot = 0
        r._snapshot_write_head(item)
        assert r._last_write_head_angle is not None

    def test_snapshot_not_stored_for_non_follow_item(self):
        r = _make_renderer()
        item = _make_item(follow=False, total_slots=40)
        r._snapshot_write_head(item)
        assert r._last_write_head_angle is None

    def test_snapshot_angle_in_0_360_range(self):
        r = _make_renderer()
        item = _make_item(follow=True, start_angle=0.0, total_slots=40)
        for slot in [0, 10, 20, 39]:
            item.last_written_slot = slot
            r._snapshot_write_head(item)
            assert 0.0 <= r._last_write_head_angle < 360.0


# ---------------------------------------------------------------------------
# handle_text_overlay (async)
# ---------------------------------------------------------------------------


class TestHandleTextOverlay:
    def _make_renderer_with_label_mock(self):
        r = _make_renderer()
        mock_label = _make_mock_label()
        r._make_char_label = Mock(return_value=mock_label)
        return r

    @pytest.mark.asyncio
    async def test_rejects_missing_text_id(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay({"content": "hello"})
        assert len(r._items) == 0

    @pytest.mark.asyncio
    async def test_rejects_missing_content(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay({"text_id": "a"})
        assert len(r._items) == 0

    @pytest.mark.asyncio
    async def test_rejects_whitespace_only_content(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay({"text_id": "a", "content": "   "})
        assert len(r._items) == 0

    @pytest.mark.asyncio
    async def test_creates_new_item(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay({"text_id": "a", "content": "Hello", "speaker": "marquee"})
        assert "a" in r._items
        assert r._items["a"].content == "Hello "  # trailing space appended

    @pytest.mark.asyncio
    async def test_trailing_space_not_duplicated(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay({"text_id": "a", "content": "Hello ", "speaker": "marquee"})
        assert r._items["a"].content == "Hello "

    @pytest.mark.asyncio
    async def test_style_overrides_applied_to_new_item(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay(
            {
                "text_id": "a",
                "content": "Hi",
                "speaker": "marquee",
                "style": {"font_size": 42, "write_speed": 3.0, "loop": False},
            }
        )
        item = r._items["a"]
        assert item.font_size == 42
        assert item.write_speed == 3.0
        assert item.loop is False

    @pytest.mark.asyncio
    async def test_content_replacement_deletes_old_labels(self):
        r = self._make_renderer_with_label_mock()
        # Create an item with chars already in place
        await r.handle_text_overlay({"text_id": "a", "content": "Old", "speaker": "marquee"})
        item = r._items["a"]
        old_label = _make_mock_label()
        old_ch = _make_char(slot=0)
        old_ch.label = old_label
        item.chars = {0: old_ch}

        # Replace content
        await r.handle_text_overlay({"text_id": "a", "content": "New", "speaker": "marquee"})
        old_label.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_content_replacement_resets_write_head(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay({"text_id": "a", "content": "Old", "speaker": "marquee"})
        item = r._items["a"]
        item.write_head_slot = 15
        item.content_index = 5
        item.has_written = True

        await r.handle_text_overlay({"text_id": "a", "content": "New", "speaker": "marquee"})
        assert item.write_head_slot == 0
        assert item.content_index == 0
        assert item.has_written is False

    @pytest.mark.asyncio
    async def test_content_replacement_applies_new_style(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay(
            {
                "text_id": "a",
                "content": "Old",
                "speaker": "marquee",
                "style": {"font_size": 28, "write_speed": 8.0},
            }
        )
        await r.handle_text_overlay(
            {
                "text_id": "a",
                "content": "New",
                "speaker": "marquee",
                "style": {"font_size": 42, "write_speed": 2.0},
            }
        )
        item = r._items["a"]
        assert item.font_size == 42
        assert item.write_speed == 2.0

    @pytest.mark.asyncio
    async def test_content_replacement_recomputes_geometry_on_radius_change(self):
        r = self._make_renderer_with_label_mock()
        await r.handle_text_overlay(
            {
                "text_id": "a",
                "content": "Old",
                "speaker": "marquee",
                "style": {"radius_scale": 0.9},
            }
        )
        old_slots = r._items["a"].total_slots

        await r.handle_text_overlay(
            {
                "text_id": "a",
                "content": "New",
                "speaker": "marquee",
                "style": {"radius_scale": 0.5},
            }
        )
        new_slots = r._items["a"].total_slots
        # Smaller radius → fewer slots
        assert new_slots < old_slots

    @pytest.mark.asyncio
    async def test_queues_when_noloop_follow_blocker_active(self):
        r = self._make_renderer_with_label_mock()
        # First message: no-loop follow item (becomes the blocker)
        await r.handle_text_overlay(
            {
                "text_id": "blocker",
                "content": "Block",
                "speaker": "marquee",
                "style": {"loop": False, "follow": True},
            }
        )
        # Second message: should be queued
        msg2 = {
            "text_id": "queued",
            "content": "Queue me",
            "speaker": "marquee",
            "style": {"loop": False, "follow": True},
        }
        await r.handle_text_overlay(msg2)

        assert "queued" not in r._items
        assert len(r._queue) == 1
        assert r._queue[0] is msg2


# ---------------------------------------------------------------------------
# handle_remove_text
# ---------------------------------------------------------------------------


class TestHandleRemoveText:
    @pytest.mark.asyncio
    async def test_triggers_expiry_on_known_item(self):
        r = _make_renderer()
        item = _make_item(text_id="x")
        r._items["x"] = item
        await r.handle_remove_text({"text_id": "x"})
        assert item.is_expiring is True

    @pytest.mark.asyncio
    async def test_silently_ignores_unknown_id(self):
        r = _make_renderer()
        # Should not raise
        await r.handle_remove_text({"text_id": "nonexistent"})
        assert len(r._items) == 0


# ---------------------------------------------------------------------------
# update() integration
# ---------------------------------------------------------------------------


class TestUpdate:
    def _renderer_with_item(self, **item_kwargs):
        r = _make_renderer()
        item = _make_item(**item_kwargs)
        r._items[item.text_id] = item
        return r, item

    def test_expired_item_removed_after_fully_gone(self):
        r, item = self._renderer_with_item(duration=0.001)
        item.is_expiring = True
        item.has_written = True
        item.chars = {}  # all faded away
        r.update(dt=0.016)
        assert "test" not in r._items

    def test_item_with_duration_starts_expiring(self):
        r, item = self._renderer_with_item()
        item.duration = 0.001
        item.creation_time = time.time() - 1.0  # already expired
        r.update(dt=0.016)
        assert item.is_expiring is True

    def test_auto_expire_noloop_item_when_content_done_and_chars_empty(self):
        r, item = self._renderer_with_item(loop=False)
        item.content_index = len(item.content)  # exhausted
        item.chars = {}
        r.update(dt=0.0)
        assert item.is_expiring is True

    def test_auto_expire_not_triggered_before_any_content_written(self):
        """content_index == 0 means nothing written yet; must not auto-expire."""
        r, item = self._renderer_with_item(loop=False)
        item.content_index = 0
        item.chars = {}
        r.update(dt=0.0)
        assert item.is_expiring is False

    def test_queue_flushed_when_blocker_expires(self):
        r = _make_renderer()

        # Blocker item: no-loop follow, already expiring with chars gone
        blocker = _make_item(text_id="blocker", loop=False, follow=True)
        blocker.is_expiring = True
        blocker.chars = {}
        r._items["blocker"] = blocker

        # Pending message in queue
        pending = {
            "text_id": "next",
            "content": "Next",
            "speaker": "marquee",
            "style": {"loop": False, "follow": True},
        }
        r._queue.append(pending)

        r.update(dt=0.016)

        # Blocker should be deleted; queue should be flushed via self._loop
        assert "blocker" not in r._items
        assert len(r._queue) == 0
        r._loop.call_soon.assert_called_once()

    def test_write_head_advances_with_dt(self):
        r = _make_renderer()
        item = _make_item(write_speed=10.0, loop=True)
        r._items["test"] = item
        r._make_char_label = Mock(return_value=_make_mock_label())
        # 0.3s at 10 chars/s → 3 chars
        r.update(dt=0.3)
        assert item.content_index == 3


# ---------------------------------------------------------------------------
# _rebuild_item_labels (resize)
# ---------------------------------------------------------------------------


class TestRebuildItemLabels:
    def test_resize_clears_chars_and_resets_write_head(self):
        r = _make_renderer()
        item = _make_item()
        # Simulate some chars already placed
        for i in range(3):
            ch = _make_char(slot=i)
            item.chars[i] = ch
        item.write_head_slot = 5
        item.content_index = 3
        item.has_written = True

        r._rebuild_item_labels(item)

        assert item.chars == {}
        assert item.write_head_slot == 0
        assert item.content_index == 0
        assert item.has_written is False

    def test_resize_deletes_pyglet_labels(self):
        r = _make_renderer()
        item = _make_item()
        labels = []
        for i in range(3):
            lbl = _make_mock_label()
            labels.append(lbl)
            ch = _make_char(slot=i)
            ch.label = lbl
            item.chars[i] = ch

        r._rebuild_item_labels(item)

        for lbl in labels:
            lbl.delete.assert_called_once()

    def test_resize_recomputes_total_slots(self):
        r = _make_renderer(width=800, height=600)
        item = _make_item(radius_scale=1.0, font_size=28, char_spacing=1.0)
        old_slots = item.total_slots

        # Simulate window resize to double size
        r.window.width = 1600
        r.window.height = 1200
        r._radius = 600

        r._rebuild_item_labels(item)
        # Larger radius → more slots
        assert item.total_slots > old_slots

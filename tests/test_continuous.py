"""Cursor, click, drag, scroll and two-hand zoom."""

from __future__ import annotations

import numpy as np
import pytest
from synth import make_state, pinch_hand, pose, synth_hand

from gesturekit import geometry as geo
from gesturekit.gestures.base import RecognizerContext
from gesturekit.gestures.continuous import (
    ContinuousRecognizer,
    PointerConfig,
    PointerRecognizer,
    ScrollRecognizer,
    ZoomRecognizer,
)


def feed(rec, frames):
    ctx = RecognizerContext()
    out = []
    for state in frames:
        ctx.push(state)
        out += rec.update(ctx)
    return out


def pointing(center=(0.5, 0.5), t=1000.0, scale=0.26):
    return make_state(
        synth_hand(curls=(0.9, 0.0, 1.0, 1.0, 1.0), center=center, scale=scale,
                   thumb_across=True),
        timestamp=t,
    )


# ---------------------------------------------------------------------------
# Pointer
# ---------------------------------------------------------------------------

def test_cursor_moves_with_index_finger():
    events = feed(PointerRecognizer(), [pointing((0.4, 0.5), 1000 + i * 0.03) for i in range(6)])
    moves = [e for e in events if e.name == "cursor_move"]
    assert moves
    assert 0.0 <= moves[-1].data["x"] <= 1.0
    assert 0.0 <= moves[-1].data["y"] <= 1.0


def test_cursor_output_is_normalised_and_clamped():
    """Hand at the extreme frame edge still maps inside the screen."""
    events = feed(PointerRecognizer(), [pointing((0.02, 0.02), 1000 + i * 0.03) for i in range(6)])
    move = [e for e in events if e.name == "cursor_move"][-1]
    assert move.data["x"] == pytest.approx(0.0, abs=1e-6)
    assert move.data["y"] == pytest.approx(0.0, abs=1e-6)


def test_cursor_region_maps_to_full_range():
    cfg = PointerConfig(region_x0=0.2, region_x1=0.8, region_y0=0.2, region_y1=0.8)
    left = feed(PointerRecognizer(cfg), [pointing((0.20, 0.5), 1000 + i * 0.03) for i in range(8)])
    right = feed(PointerRecognizer(cfg), [pointing((0.80, 0.5), 1000 + i * 0.03) for i in range(8)])
    assert [e for e in left if e.name == "cursor_move"][-1].data["x"] < 0.35
    assert [e for e in right if e.name == "cursor_move"][-1].data["x"] > 0.65


def test_cursor_silent_when_index_is_curled():
    frames = [make_state(pose("closed_fist"), timestamp=1000 + i * 0.03) for i in range(6)]
    assert feed(PointerRecognizer(), frames) == []


def test_quick_pinch_emits_click_not_drag():
    rec = PointerRecognizer(PointerConfig(drag_hold=0.30))
    frames = [pointing(t=1000 + i * 0.03) for i in range(3)]
    frames += [make_state(pinch_hand(others_open=False), timestamp=1000.10 + i * 0.03)
               for i in range(3)]
    frames += [pointing(t=1000.22 + i * 0.03) for i in range(3)]
    names = [e.name for e in feed(rec, frames)]
    assert "click" in names
    assert "drag_start" not in names


def test_held_pinch_becomes_a_drag():
    rec = PointerRecognizer(PointerConfig(drag_hold=0.25))
    frames = [pointing(t=1000 + i * 0.03) for i in range(3)]
    frames += [make_state(pinch_hand(others_open=False), timestamp=1000.10 + i * 0.05)
               for i in range(12)]
    frames += [pointing(t=1001.0 + i * 0.03) for i in range(3)]
    names = [e.name for e in feed(rec, frames)]
    assert "drag_start" in names
    assert "drag_end" in names
    assert names.index("drag_start") < names.index("drag_end")
    assert "click" not in names


def test_drag_ends_when_hand_disappears():
    """A lost hand must release the button; a stuck drag is a disaster."""
    rec = PointerRecognizer(PointerConfig(drag_hold=0.2))
    ctx = RecognizerContext()
    for i in range(12):
        ctx.push(make_state(pinch_hand(others_open=False), timestamp=1000 + i * 0.05))
        rec.update(ctx)
    released = rec.update(RecognizerContext())
    assert [e.name for e in released] == ["drag_end"]


def test_pinch_hysteresis_prevents_click_storm():
    """A pinch hovering at the threshold must not machine-gun clicks."""
    rec = PointerRecognizer(PointerConfig(pinch_on=0.32, pinch_off=0.42))
    frames = []
    for i in range(24):
        pts = pinch_hand(others_open=False)
        # nudge the index tip so the distance oscillates right at the boundary
        pts[geo.INDEX_TIP, :2] += np.array([0.0, 0.012 * (i % 2)], np.float32)
        frames.append(make_state(pts, timestamp=1000 + i * 0.04))
    clicks = [e for e in feed(rec, frames) if e.name == "click"]
    assert len(clicks) <= 1


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------

def scroll_pose(center, t):
    return make_state(
        synth_hand(curls=(0.9, 0.0, 0.0, 1.0, 1.0), center=center, scale=0.26,
                   thumb_across=True),
        timestamp=t,
    )


def test_scroll_emits_on_two_finger_drag():
    frames = [scroll_pose((0.5, 0.5 - i * 0.02), 1000 + i * 0.04) for i in range(8)]
    events = feed(ScrollRecognizer(), frames)
    assert events and all(e.name == "scroll" for e in events)
    assert events[-1].data["dy"] > 0          # moving up scrolls up


def test_scroll_direction_inverts_correctly():
    down = feed(ScrollRecognizer(),
                [scroll_pose((0.5, 0.5 + i * 0.02), 1000 + i * 0.04) for i in range(8)])
    assert down[-1].data["dy"] < 0


def test_scroll_requires_the_right_pose():
    frames = [make_state(pose("open_palm"), timestamp=1000 + i * 0.04) for i in range(8)]
    assert feed(ScrollRecognizer(), frames) == []


def test_scroll_deadzone_ignores_micro_movement():
    frames = [scroll_pose((0.5, 0.5 + i * 0.0005), 1000 + i * 0.04) for i in range(10)]
    assert feed(ScrollRecognizer(deadzone=0.35), frames) == []


# ---------------------------------------------------------------------------
# Zoom
# ---------------------------------------------------------------------------

def two_pinches(gap, t=1000.0):
    a = make_state(pinch_hand(others_open=False, center=(0.5 - gap / 2, 0.5)), timestamp=t)
    b = make_state(pinch_hand(others_open=False, center=(0.5 + gap / 2, 0.5)), timestamp=t)
    return [a, b]


def test_zoom_in_on_hands_separating():
    rec = ZoomRecognizer()
    rec.update_two(two_pinches(0.20))
    events = rec.update_two(two_pinches(0.45, t=1000.3))
    assert [e.name for e in events] == ["zoom_in"]


def test_zoom_out_on_hands_converging():
    rec = ZoomRecognizer()
    rec.update_two(two_pinches(0.45))
    events = rec.update_two(two_pinches(0.20, t=1000.3))
    assert [e.name for e in events] == ["zoom_out"]


def test_zoom_requires_both_hands_pinching():
    rec = ZoomRecognizer()
    open_hands = [
        make_state(pose("open_palm", center=(0.3, 0.5))),
        make_state(pose("open_palm", center=(0.7, 0.5))),
    ]
    rec.update_two(open_hands)
    assert rec.update_two(open_hands) == []


def test_zoom_needs_two_hands():
    assert ZoomRecognizer().update_two(two_pinches(0.3)[:1]) == []


def test_zoom_deadzone():
    rec = ZoomRecognizer(deadzone=0.2)
    rec.update_two(two_pinches(0.30))
    assert rec.update_two(two_pinches(0.31, t=1000.2)) == []


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_scroll_takes_priority_over_pointer():
    """Overlapping poses: scrolling must not also drag the cursor."""
    rec = ContinuousRecognizer()
    frames = [scroll_pose((0.5, 0.5 - i * 0.02), 1000 + i * 0.04) for i in range(8)]
    names = {e.name for e in feed(rec, frames)}
    assert "scroll" in names
    assert "cursor_move" not in names

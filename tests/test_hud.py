"""HUD rendering: must never crash, never mutate frame geometry."""

from __future__ import annotations

import numpy as np
import pytest
from synth import make_state, pose

from gesturekit.hud import HUD, draw_help_screen, panel
from gesturekit.types import FrameResult, GestureEvent, Handedness


@pytest.fixture
def frame():
    return np.full((480, 640, 3), 40, np.uint8)


def test_panel_only_touches_its_rectangle(frame):
    before = frame.copy()
    panel(frame, 10, 10, 100, 60, (200, 0, 0), 0.5)
    assert not np.array_equal(frame[10:60, 10:100], before[10:60, 10:100])
    np.testing.assert_array_equal(frame[200:, 200:], before[200:, 200:])


def test_panel_clips_out_of_bounds(frame):
    panel(frame, -50, -50, 5000, 5000, (0, 0, 0), 0.5)      # must not raise
    panel(frame, 700, 700, 800, 800, (0, 0, 0), 0.5)        # fully outside
    assert frame.shape == (480, 640, 3)


def test_draw_with_no_hands(frame):
    out = HUD().draw(frame, FrameResult(fps=30.0), profile="presentation", backend="dry-run")
    assert out.shape == (480, 640, 3)


def test_draw_with_hands_and_events(frame):
    result = FrameResult(
        hands=[make_state(pose("peace"))],
        events=[GestureEvent("peace", 0.9, Handedness.RIGHT, 1000.0)],
        fps=29.5,
        latency_ms=22.0,
        inference_ms=8.0,
    )
    hud = HUD(show_scores=True)
    hud.push_events(result.events, {"peace": "Blackout"})
    out = hud.draw(frame, result, profile="presentation", backend="pynput",
                   scores={"peace": 0.9, "three": 0.2})
    assert out.shape == (480, 640, 3)


def test_draw_handles_two_hands(frame):
    result = FrameResult(hands=[
        make_state(pose("peace", center=(0.3, 0.5))),
        make_state(pose("rock", center=(0.7, 0.5)), handedness=Handedness.LEFT),
    ])
    assert HUD().draw(frame, result).shape == (480, 640, 3)


def test_arm_countdown_renders(frame):
    out = HUD().draw(frame, FrameResult(), armed=True, arm_remaining=1.7, require_arm=True)
    assert out.shape == (480, 640, 3)


def test_unarmed_state_renders(frame):
    assert HUD().draw(frame, FrameResult(), armed=False, require_arm=True) is not None


def test_toast_renders_and_expires(frame):
    hud = HUD()
    hud.notify("profile: desktop", "good", duration=10.0)
    hud.draw(frame, FrameResult())
    assert hud.toast is not None
    hud.notify("gone", "info", duration=-1.0)
    hud.draw(frame, FrameResult())
    assert hud.toast is None


def test_hand_at_frame_edge_does_not_crash(frame):
    for center in ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)):
        result = FrameResult(hands=[make_state(pose("open_palm", center=center))])
        HUD().draw(frame.copy(), result)


def test_tiny_frame_does_not_crash():
    small = np.zeros((64, 64, 3), np.uint8)
    HUD().draw(small, FrameResult(hands=[make_state(pose("peace"))]))


def test_continuous_events_are_not_listed():
    hud = HUD()
    hud.push_events(
        [GestureEvent("cursor_move", 1.0, Handedness.RIGHT, 1.0, kind="continuous")], {}
    )
    assert len(hud._recent) == 0


def test_help_screen_renders():
    img = draw_help_screen(1280, 720, [("peace", "Blackout"), ("rock", "Pen")], "presentation")
    assert img.shape == (720, 1280, 3)
    assert img.any()


def test_help_screen_with_many_bindings():
    bindings = [(f"gesture_{i}", f"action {i}") for i in range(80)]
    assert draw_help_screen(1280, 720, bindings, "test").shape == (720, 1280, 3)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_both_themes(theme, frame):
    hud = HUD(theme=theme)
    assert hud.draw(frame, FrameResult(hands=[make_state(pose("peace"))])) is not None


def test_toggles_are_respected(frame):
    hud = HUD(show_landmarks=False, show_trail=False, show_scores=False)
    plain = hud.draw(frame.copy(), FrameResult(hands=[make_state(pose("peace"))]))
    hud.show_landmarks = True
    drawn = hud.draw(frame.copy(), FrameResult(hands=[make_state(pose("peace"))]))
    assert not np.array_equal(plain, drawn)

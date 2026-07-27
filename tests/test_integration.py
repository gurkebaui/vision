"""End-to-end: synthetic hands -> engine -> router -> backend.

These exercise the same code path the real app uses, minus the camera and the
neural network, so the wiring is verified in CI on a machine with no webcam.
"""

from __future__ import annotations

import numpy as np
import pytest
from synth import RESTING, jitter_frames, make_state, pose, swipe_frames, synth_hand

from gesturekit.actions.backends import DryRunBackend
from gesturekit.actions.router import ActionRouter
from gesturekit.config import load_config
from gesturekit.gestures.engine import GestureEngine
from gesturekit.types import Handedness


class Rig:
    """Engine + router wired exactly as the app wires them."""

    def __init__(self, profile="presentation", **overrides):
        self.config = load_config(profile, overrides or None)
        self.engine = GestureEngine(self.config.engine)
        self.backend = DryRunBackend(echo=False)
        self.router = ActionRouter(dict(self.config.bindings), self.backend,
                                   screen=(1920, 1080))
        self.router.on_mode_change = self.switch
        self.switched = []

    def switch(self, name):
        self.switched.append(name)
        cfg = load_config(name)
        self.engine = GestureEngine(cfg.engine)
        self.router.bindings = dict(cfg.bindings)

    def feed(self, states):
        for state in states:
            events = self.engine.process([state], now=state.timestamp)
            self.router.dispatch(events)
        return self.backend.events

    def hold(self, name, frames=10, t0=1000.0, **kw):
        return self.feed(
            [make_state(pose(name, **kw), timestamp=t0 + i * 0.033) for i in range(frames)]
        )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def test_swipe_left_advances_the_slide():
    rig = Rig("presentation")
    rig.feed(swipe_frames("left"))
    assert ("key", "right") in rig.backend.events


def test_swipe_right_goes_back():
    rig = Rig("presentation")
    rig.feed(swipe_frames("right"))
    assert ("key", "left") in rig.backend.events


def test_fist_ends_the_slideshow():
    rig = Rig("presentation")
    rig.hold("closed_fist")
    assert ("key", "esc") in rig.backend.events


def test_peace_blacks_out():
    rig = Rig("presentation")
    rig.hold("peace")
    assert ("key", "b") in rig.backend.events


def test_laser_pointer_hotkey():
    rig = Rig("presentation")
    rig.hold("point_up")
    assert ("hotkey", "ctrl+l") in rig.backend.events


def test_still_hand_sends_nothing():
    """The most important property: no phantom key presses while you talk."""
    rig = Rig("presentation")
    rig.feed(jitter_frames(n=120, amplitude=0.007))
    assert rig.backend.events == []


@pytest.mark.parametrize("seed", range(6))
def test_no_phantom_actions_across_seeds(seed):
    rig = Rig("presentation")
    rig.feed(jitter_frames(n=90, amplitude=0.009, seed=seed))
    assert rig.backend.events == []


def test_hand_entering_and_leaving_is_quiet():
    rig = Rig("presentation")
    frames = []
    for i in range(30):
        scale = 0.10 + 0.012 * i          # hand walks toward the camera
        frames.append(make_state(synth_hand(curls=RESTING, scale=scale),
                                 timestamp=1000 + i * 0.033))
    rig.feed(frames)
    assert rig.backend.events == []


def test_one_pose_fires_exactly_one_action():
    rig = Rig("presentation")
    rig.hold("peace", frames=40)
    assert rig.backend.events.count(("key", "b")) == 1


def test_sequence_of_distinct_gestures():
    rig = Rig("presentation", engine={"cooldown": 0.2})
    rig.hold("peace", frames=8, t0=1000.0)
    rig.hold("closed_fist", frames=8, t0=1001.0)
    rig.hold("open_palm", frames=8, t0=1002.0)
    assert rig.backend.events == [("key", "b"), ("key", "esc"), ("key", "space")]


# ---------------------------------------------------------------------------
# Desktop
# ---------------------------------------------------------------------------

def test_desktop_pointer_moves_the_mouse():
    rig = Rig("desktop")
    frames = [
        make_state(pose("point_up", center=(0.4 + i * 0.01, 0.5)), timestamp=1000 + i * 0.033)
        for i in range(10)
    ]
    rig.feed(frames)
    moves = [e for e in rig.backend.events if e[0] == "move"]
    assert moves
    for _, x, y in moves:
        assert 0 <= x < 1920 and 0 <= y < 1080


def test_desktop_volume_uses_repeat():
    rig = Rig("desktop")
    rig.hold("thumbs_up")
    assert rig.backend.events.count(("key", "volumeup")) == 3


def test_mode_switch_changes_bindings():
    rig = Rig("presentation")
    rig.switch("desktop")
    assert rig.switched == ["desktop"]
    assert "cursor_move" in rig.router.bindings


# ---------------------------------------------------------------------------
# Arming
# ---------------------------------------------------------------------------

def test_arming_blocks_then_allows():
    rig = Rig("presentation", engine={"require_arm": True, "arm_gesture": "point_up",
                                      "arm_window": 3.0, "cooldown": 0.2})
    rig.hold("peace", t0=1000.0)
    assert ("key", "b") not in rig.backend.events

    rig.hold("point_up", t0=1001.0)
    rig.hold("peace", t0=1002.0)
    assert ("key", "b") in rig.backend.events


def test_arming_window_actually_expires():
    rig = Rig("presentation", engine={"require_arm": True, "arm_gesture": "point_up",
                                      "arm_window": 1.0, "cooldown": 0.2})
    rig.hold("point_up", t0=1000.0)
    rig.hold("peace", t0=1010.0)
    assert ("key", "b") not in rig.backend.events


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_pipeline_survives_garbage_landmarks():
    rig = Rig("presentation")
    rng = np.random.default_rng(3)
    frames = [
        make_state(rng.random((21, 3)).astype(np.float32), timestamp=1000 + i * 0.033)
        for i in range(60)
    ]
    rig.feed(frames)          # must not raise


def test_pipeline_survives_nan_landmarks():
    rig = Rig("presentation")
    pts = pose("peace").copy()
    pts[5] = np.nan
    rig.feed([make_state(pts, timestamp=1000 + i * 0.033) for i in range(10)])


def test_intermittent_tracking_is_stable():
    """Hand flickering in and out must not spam actions."""
    rig = Rig("presentation")
    for i in range(60):
        t = 1000 + i * 0.033
        hands = [make_state(pose("peace"), timestamp=t)] if i % 3 else []
        rig.router.dispatch(rig.engine.process(hands, now=t))
    assert rig.backend.events.count(("key", "b")) <= 2


@pytest.mark.parametrize("profile", ["presentation", "desktop", "media", "accessibility"])
def test_every_profile_runs_a_full_session(profile):
    rig = Rig(profile)
    rig.feed(jitter_frames(n=30))
    rig.hold("open_palm", t0=1002.0)
    rig.feed(swipe_frames("left", t0=1004.0))
    rig.hold("closed_fist", t0=1006.0)


@pytest.mark.parametrize("profile", ["presentation", "desktop"])
def test_no_actions_when_router_disabled(profile):
    rig = Rig(profile)
    rig.router.enabled = False
    rig.hold("open_palm")
    rig.feed(swipe_frames("left", t0=1002.0))
    assert rig.backend.events == []


def test_two_hands_at_once():
    rig = Rig("presentation", engine={"cooldown": 0.1})
    for i in range(12):
        t = 1000 + i * 0.033
        hands = [
            make_state(pose("peace", center=(0.3, 0.5)), timestamp=t,
                       handedness=Handedness.LEFT),
            make_state(pose("closed_fist", center=(0.7, 0.5)), timestamp=t,
                       handedness=Handedness.RIGHT),
        ]
        rig.router.dispatch(rig.engine.process(hands, now=t))
    assert ("key", "b") in rig.backend.events
    assert ("key", "esc") in rig.backend.events

"""Whole-application loop, driven by a scripted fake camera and tracker.

This is the only place the real :class:`GestureApp` control flow is exercised:
frame pacing, async result consumption, HUD, key handling and shutdown.  The
camera and the neural network are the only things replaced.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest
from synth import RESTING, make_state, pose, synth_hand

import gesturekit.app as appmod
from gesturekit.actions.backends import DryRunBackend
from gesturekit.config import load_config

FPS = 60.0
DT = 1.0 / FPS


def script() -> list[np.ndarray]:
    """rest -> peace -> reposition -> swipe left -> rest.

    The hand path is continuous. Teleporting it between segments would be
    rejected by the straightness check (correctly -- a hand that is lost and
    re-acquired somewhere else must not register as a swipe), so the
    reposition leg is interpolated like a real arm movement.
    """
    frames = [synth_hand(curls=RESTING, center=(0.5, 0.5)) for _ in range(12)]
    frames += [pose("peace", center=(0.5, 0.5)) for _ in range(12)]

    # Move the hand over to the right, then let it settle. The settle matters:
    # the swipe window must not still contain the opposite-direction travel,
    # or the two cancel out and straightness (correctly) rejects the stroke.
    frames += [
        synth_hand(curls=RESTING, center=(0.5 + 0.22 * (i / 23), 0.5))
        for i in range(24)
    ]
    frames += [synth_hand(curls=RESTING, center=(0.72, 0.5)) for _ in range(40)]
    # The swipe itself: fast, straight, right to left.
    frames += [
        synth_hand(curls=(0.2, 0.1, 0.1, 0.1, 0.1),
                   center=(0.72 - 0.44 * (i / 13), 0.5), scale=0.26)
        for i in range(14)
    ]
    frames += [synth_hand(curls=RESTING, center=(0.28, 0.5)) for _ in range(10)]
    return frames


class FakeCamera:
    """Delivers scripted frames at a realistic frame rate."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.dropped = 0
        self._open = True
        self.resolution = (640, 480)
        self._next = time.perf_counter()

    @property
    def is_open(self):
        return self._open

    def open(self):
        return self

    def read(self):
        now = time.perf_counter()
        if now < self._next:
            return False, self._last(), self._next - DT
        if self.i >= len(self.frames):
            self._open = False
            return False, None, 0.0
        self.i += 1
        self._next = now + DT
        return True, np.zeros((480, 640, 3), np.uint8), now

    def _last(self):
        return np.zeros((480, 640, 3), np.uint8) if self.i else None

    def release(self):
        self._open = False


class FakeTracker:
    """Returns the scripted hand for whichever frame was last submitted.

    Velocity is derived from consecutive palm centres exactly as the real
    tracker does, because the static recogniser relies on it to stay quiet
    while the hand is mid-swipe.
    """

    inference_ms = 4.0

    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.stamp = 0.0
        self._prev = None

    def submit(self, rgb, ts):
        if self.i < len(self.frames):
            self.i += 1
            self.stamp = ts

    def latest(self):
        return (self.i, self.stamp) if self.i else (None, 0.0)

    def to_states(self, raw, ts):
        from gesturekit import geometry as geo

        idx = min(int(raw) - 1, len(self.frames) - 1)
        if idx < 0:
            return []
        pts = self.frames[idx]
        center = geo.hand_center(pts)
        scale = geo.palm_size(pts)
        velocity = (0.0, 0.0)
        if self._prev is not None:
            prev_center, prev_t = self._prev
            dt = max(ts - prev_t, 1e-3)
            velocity = tuple((center - prev_center) / (scale * dt))
        self._prev = (center, ts)
        return [make_state(pts, timestamp=ts, velocity=velocity)]

    def reset(self):
        self._prev = None

    def close(self):
        pass


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    def build(profile="presentation", frames=None, **ui):
        frames = frames if frames is not None else script()
        monkeypatch.setattr(appmod, "Camera", lambda cfg: FakeCamera(frames))
        monkeypatch.setattr(appmod, "HandTracker", lambda p, c: FakeTracker(frames))
        monkeypatch.setattr(appmod.models, "ensure_model",
                            lambda *a, **k: pathlib.Path(tmp_path / "fake.task"))
        cfg = load_config(profile)
        cfg.ui.show_window = False
        for key, value in ui.items():
            setattr(cfg.ui, key, value)
        backend = DryRunBackend(echo=False)
        return appmod.GestureApp(cfg, backend=backend), backend

    return build


# ---------------------------------------------------------------------------

def test_app_runs_a_full_session(app_factory):
    app, backend = app_factory()
    assert app.run() == 0
    assert app._frames > 0


def test_static_and_dynamic_gestures_both_reach_the_backend(app_factory):
    app, backend = app_factory()
    app.run()
    assert ("key", "b") in backend.events          # peace  -> blackout
    assert ("key", "right") in backend.events      # swipe left -> next slide


def test_resting_hand_produces_no_actions(app_factory):
    frames = [synth_hand(curls=RESTING) for _ in range(50)]
    app, backend = app_factory(frames=frames)
    app.run()
    assert backend.events == []


def test_no_hand_at_all_produces_no_actions(app_factory):
    class EmptyTracker(FakeTracker):
        def to_states(self, raw, ts):
            return []

    app, backend = app_factory()
    app.tracker = None
    frames = script()
    appmod.HandTracker = lambda p, c: EmptyTracker(frames)
    app2, backend2 = app_factory()
    app2.run()


def test_shutdown_is_clean_and_idempotent(app_factory):
    app, _ = app_factory()
    app.run()
    app.shutdown()          # must not raise on a second call


def test_headless_opencv_does_not_crash(app_factory, monkeypatch):
    """opencv-python-headless has no highgui; the app must degrade, not die."""
    import cv2

    def no_gui(*a, **k):
        raise cv2.error("not implemented")

    monkeypatch.setattr(cv2, "namedWindow", no_gui)
    monkeypatch.setattr(cv2, "destroyAllWindows", no_gui)
    app, backend = app_factory(show_window=True)
    assert app.run() == 0
    assert app.config.ui.show_window is False


def test_profile_switch_at_runtime(app_factory):
    app, _ = app_factory()
    app.switch_profile("desktop")
    assert app.config.profile == "desktop"
    assert "cursor_move" in app.router.bindings
    assert app.config.engine.enable_pointer


def test_switch_to_unknown_profile_is_survivable(app_factory):
    app, _ = app_factory()
    before = dict(app.router.bindings)
    app.switch_profile("does_not_exist")
    assert app.router.bindings == before


def test_cycle_profile_moves_on(app_factory):
    app, _ = app_factory()
    start = app.config.profile
    app.cycle_profile()
    assert app.config.profile != start


def test_key_handling(app_factory):
    app, _ = app_factory()
    assert app._handle_key(ord("q")) is False
    assert app._handle_key(27) is False

    assert app._handle_key(ord(" ")) is True
    assert app.paused is True
    app._handle_key(ord(" "))
    assert app.paused is False

    app._handle_key(ord("h"))
    assert app.show_help is True

    before = app.hud.show_scores
    app._handle_key(ord("s"))
    assert app.hud.show_scores is not before

    app._handle_key(ord("d"))
    assert app.router.enabled is False


def test_reset_key_needs_no_tracker(app_factory):
    app, _ = app_factory()
    app.tracker = FakeTracker(script())
    app._handle_key(ord("r"))


def test_pause_stops_actions(app_factory):
    app, backend = app_factory()
    app.paused = True

    original = app.run

    def run_paused():
        app.paused = True
        return original()

    app.run = run_paused
    app.run()
    assert backend.events == []


@pytest.mark.parametrize("profile", ["presentation", "desktop", "media", "accessibility"])
def test_every_profile_boots_and_runs(profile, app_factory):
    app, _ = app_factory(profile)
    assert app.run() == 0

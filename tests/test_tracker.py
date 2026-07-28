"""Tracker result conversion.

The MediaPipe task itself needs model weights and a camera, so these tests
drive :meth:`HandTracker.to_states` -- the pure conversion/filtering logic --
with objects shaped exactly like the ones MediaPipe 1.0 returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from synth import pose

from gesturekit.tracker import HandTracker, TrackerConfig, TrackerError
from gesturekit.types import Handedness

# -- MediaPipe-shaped stand-ins ---------------------------------------------

@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0


@dataclass
class FakeCategory:
    category_name: str
    score: float
    index: int = 0
    display_name: str = ""


@dataclass
class FakeResult:
    hand_landmarks: list = field(default_factory=list)
    handedness: list = field(default_factory=list)
    hand_world_landmarks: list = field(default_factory=list)
    gestures: list = field(default_factory=list)


def as_landmarks(pts: np.ndarray) -> list:
    return [FakeLandmark(float(x), float(y), float(z)) for x, y, z in pts]


def result_for(name="peace", handed="Right", gesture=None, score=0.99):
    res = FakeResult(
        hand_landmarks=[as_landmarks(pose(name))],
        handedness=[[FakeCategory(handed, score)]],
    )
    if gesture:
        res.gestures = [[FakeCategory(gesture[0], gesture[1])]]
    return res


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    """A HandTracker with the MediaPipe task construction stubbed out."""
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"x" * 1024)
    monkeypatch.setattr(HandTracker, "_build", lambda self: None)
    t = HandTracker(model, TrackerConfig())
    t._live = True
    return t


# ---------------------------------------------------------------------------

def test_missing_model_file_raises(tmp_path):
    with pytest.raises(TrackerError, match="not found"):
        HandTracker(tmp_path / "nope.task")


def test_none_result_yields_no_hands(tracker):
    assert tracker.to_states(None, 1000.0) == []


def test_empty_result_yields_no_hands(tracker):
    assert tracker.to_states(FakeResult(), 1000.0) == []


def test_converts_landmarks_to_state(tracker):
    states = tracker.to_states(result_for("peace"), 1000.0)
    assert len(states) == 1
    s = states[0]
    assert s.points.shape == (21, 3)
    assert s.normalized.shape == (21, 3)
    assert s.curls.shape == (5,)
    assert s.extended.shape == (5,)
    assert s.handedness is Handedness.RIGHT
    assert s.score == pytest.approx(0.99)
    assert s.palm_size > 0


def test_handedness_mapping(tracker):
    assert tracker.to_states(result_for(handed="Left"), 1.0)[0].handedness is Handedness.LEFT
    assert tracker.to_states(result_for(handed="Right"), 2.0)[0].handedness is Handedness.RIGHT
    assert tracker.to_states(result_for(handed="?"), 3.0)[0].handedness is Handedness.UNKNOWN


def test_canned_gesture_label_is_captured(tracker):
    s = tracker.to_states(result_for(gesture=("Victory", 0.88)), 1000.0)[0]
    assert s.label == "Victory"
    assert s.label_score == pytest.approx(0.88)


def test_missing_handedness_is_tolerated(tracker):
    res = FakeResult(hand_landmarks=[as_landmarks(pose("peace"))])
    assert tracker.to_states(res, 1.0)[0].handedness is Handedness.UNKNOWN


def test_short_landmark_list_is_skipped(tracker):
    res = FakeResult(hand_landmarks=[as_landmarks(pose("peace")[:5])],
                     handedness=[[FakeCategory("Right", 0.9)]])
    assert tracker.to_states(res, 1.0) == []


def test_velocity_zero_on_first_frame(tracker):
    s = tracker.to_states(result_for(), 1000.0)[0]
    assert float(np.linalg.norm(s.velocity)) == 0.0


def test_velocity_measured_in_palm_widths_per_second(tracker):
    """Same physical motion at different distances gives the same velocity."""
    speeds = []
    for scale in (0.20, 0.40):
        tracker.reset()
        tracker._prev_center.clear()
        tracker._pos_filters.clear()
        tracker._vel_filters.clear()
        for i in range(8):
            shift = 0.9 * scale * i * 0.05
            pts = pose("peace", scale=scale, center=(0.3 + shift, 0.5))
            res = FakeResult(hand_landmarks=[as_landmarks(pts)],
                             handedness=[[FakeCategory("Right", 0.9)]])
            state = tracker.to_states(res, 1000.0 + i * 0.05)[0]
        speeds.append(state.speed)
    assert speeds[0] == pytest.approx(speeds[1], rel=0.25)


def test_landmarks_are_filtered_over_time(tracker):
    """Noise in, smooth out."""
    rng = np.random.default_rng(0)
    base = pose("peace")
    last = None
    deltas = []
    for i in range(30):
        pts = base + rng.normal(0, 0.004, base.shape).astype(np.float32)
        res = FakeResult(hand_landmarks=[as_landmarks(pts)],
                         handedness=[[FakeCategory("Right", 0.9)]])
        state = tracker.to_states(res, 1000.0 + i * 0.033)[0]
        if last is not None:
            deltas.append(float(np.abs(state.points - last).mean()))
        last = state.points.copy()
    assert np.mean(deltas[10:]) < 0.004


def test_two_hands_get_independent_filters(tracker):
    res = FakeResult(
        hand_landmarks=[as_landmarks(pose("peace", center=(0.3, 0.5))),
                        as_landmarks(pose("rock", center=(0.7, 0.5)))],
        handedness=[[FakeCategory("Left", 0.9)], [FakeCategory("Right", 0.9)]],
    )
    states = tracker.to_states(res, 1000.0)
    assert len(states) == 2
    assert {s.handedness for s in states} == {Handedness.LEFT, Handedness.RIGHT}
    assert len(tracker._pos_filters) == 2


def test_reset_clears_filter_state(tracker):
    tracker.to_states(result_for(), 1000.0)
    assert tracker._prev_center
    tracker.reset()
    assert tracker._prev_center == {}


def test_close_is_safe_without_task(tracker):
    tracker.close()
    tracker.close()


def test_context_manager(tracker):
    with tracker as t:
        assert t is tracker

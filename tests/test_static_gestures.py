"""Static hand-shape recognition: accuracy, robustness and stability."""

from __future__ import annotations

import numpy as np
import pytest
from synth import POSES, make_state, pose

from gesturekit.gestures.base import RecognizerContext
from gesturekit.gestures.static import (
    GestureStabilizer,
    StaticRecognizer,
    score_all,
)

ALL_POSES = list(POSES) + ["pinch", "ok"]
#: Poses whose meaning does not depend on which way the hand points.
ORIENTATION_FREE = [
    "open_palm", "closed_fist", "peace", "rock", "call_me", "three", "four", "pinch", "ok",
]


@pytest.mark.parametrize("name", ALL_POSES)
def test_every_pose_is_top1(name):
    scores = score_all(make_state(pose(name)))
    best = max(scores, key=scores.get)
    assert best == name, f"{name} misread as {best} ({scores[best]:.2f})"


@pytest.mark.parametrize("name", ALL_POSES)
def test_correct_pose_scores_confidently(name):
    assert score_all(make_state(pose(name)))[name] > 0.62


@pytest.mark.parametrize("name", ALL_POSES)
def test_winner_has_a_clear_margin(name):
    scores = sorted(score_all(make_state(pose(name))).values(), reverse=True)
    assert scores[0] - scores[1] > 0.08


@pytest.mark.parametrize("name", ORIENTATION_FREE)
@pytest.mark.parametrize("angle", [-45, -25, 25, 45])
def test_robust_to_hand_rotation(name, angle):
    scores = score_all(make_state(pose(name, rotation_deg=angle)))
    assert max(scores, key=scores.get) == name


@pytest.mark.parametrize("name", ORIENTATION_FREE)
@pytest.mark.parametrize("scale", [0.14, 0.22, 0.42, 0.60])
def test_robust_to_hand_distance(name, scale):
    scores = score_all(make_state(pose(name, scale=scale)))
    assert max(scores, key=scores.get) == name


@pytest.mark.parametrize("name", ORIENTATION_FREE)
def test_robust_to_frame_position(name):
    for center in ((0.25, 0.35), (0.75, 0.35), (0.5, 0.7)):
        scores = score_all(make_state(pose(name, center=center)))
        assert max(scores, key=scores.get) == name


def test_accuracy_under_landmark_noise():
    """MediaPipe landmarks are noisy; recognition must survive that."""
    rng = np.random.default_rng(7)
    hits = total = 0
    for name in ALL_POSES:
        base = pose(name)
        for _ in range(40):
            pts = base.copy()
            pts[:, :2] += rng.normal(0, 0.006, (21, 2))
            scores = score_all(make_state(pts))
            hits += max(scores, key=scores.get) == name
            total += 1
    assert hits / total > 0.95, f"only {hits}/{total} correct under noise"


def test_scores_are_bounded():
    for name in ALL_POSES:
        for value in score_all(make_state(pose(name))).values():
            assert 0.0 <= value <= 1.0


def test_canned_label_boosts_but_does_not_override():
    """Google's 7-class model is a hint, not a veto -- it cannot force a wrong answer."""
    honest = score_all(make_state(pose("peace")))
    lied_to = score_all(make_state(pose("peace"), label="Closed_Fist", label_score=0.99))
    assert max(lied_to, key=lied_to.get) == "peace"
    assert lied_to["closed_fist"] >= honest["closed_fist"]


def test_canned_label_raises_confidence_of_agreeing_gesture():
    plain = score_all(make_state(pose("closed_fist")))
    boosted = score_all(make_state(pose("closed_fist"), label="Closed_Fist", label_score=0.95))
    assert boosted["closed_fist"] >= plain["closed_fist"]


# ---------------------------------------------------------------------------
# Stabiliser
# ---------------------------------------------------------------------------

def test_stabilizer_requires_consecutive_frames():
    s = GestureStabilizer(min_frames=4, min_score=0.6, margin=0.1)
    scores = {"open_palm": 0.9, "peace": 0.2}
    assert s.update(scores) is None
    assert s.update(scores) is None
    assert s.update(scores) is None
    hit = s.update(scores)
    assert hit is not None and hit[0] == "open_palm"


def test_stabilizer_does_not_repeat_while_held():
    s = GestureStabilizer(min_frames=2, min_score=0.6, margin=0.1)
    scores = {"open_palm": 0.9}
    fires = [s.update(scores) for _ in range(20)]
    assert sum(f is not None for f in fires) == 1


def test_stabilizer_rejects_ambiguous_frames():
    """Two gestures scoring the same is a coin flip -- emit nothing."""
    s = GestureStabilizer(min_frames=2, min_score=0.5, margin=0.15)
    for _ in range(10):
        assert s.update({"peace": 0.71, "three": 0.70}) is None


def test_stabilizer_rejects_low_scores():
    s = GestureStabilizer(min_frames=2, min_score=0.7, margin=0.05)
    for _ in range(10):
        assert s.update({"open_palm": 0.5}) is None


def test_stabilizer_releases_and_can_refire():
    s = GestureStabilizer(min_frames=2, min_score=0.6, margin=0.1, release_frames=2)
    hot = {"open_palm": 0.9}
    s.update(hot)
    assert s.update(hot) is not None
    for _ in range(3):
        s.update({})
    s.update(hot)
    assert s.update(hot) is not None


def test_single_frame_spike_is_ignored():
    """v1 fired on one good frame; a flicker must not trigger an action."""
    s = GestureStabilizer(min_frames=4, min_score=0.6, margin=0.1)
    assert s.update({"closed_fist": 0.99}) is None
    for _ in range(5):
        s.update({"open_palm": 0.3, "peace": 0.28})
    assert s.active is None


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------

def test_recognizer_emits_once_per_pose():
    rec = StaticRecognizer(min_frames=3, min_score=0.6)
    ctx = RecognizerContext()
    events = []
    for i in range(15):
        ctx.push(make_state(pose("peace"), timestamp=1000 + i * 0.033))
        events += rec.update(ctx)
    assert [e.name for e in events] == ["peace"]


def test_recognizer_suppresses_shapes_during_fast_motion():
    """A hand in flight is mid-swipe; its shape must not also fire."""
    rec = StaticRecognizer(min_frames=2, min_score=0.6, max_speed=1.8)
    ctx = RecognizerContext()
    events = []
    for i in range(10):
        ctx.push(make_state(pose("open_palm"), timestamp=1000 + i * 0.033,
                            velocity=(4.0, 0.0)))
        events += rec.update(ctx)
    assert events == []


def test_recognizer_handles_empty_context():
    assert StaticRecognizer().update(RecognizerContext()) == []

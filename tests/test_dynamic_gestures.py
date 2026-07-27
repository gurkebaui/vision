"""Motion gestures: swipes must fire on real strokes and never on noise."""

from __future__ import annotations

import numpy as np
import pytest
from synth import circle_frames, jitter_frames, make_state, pose, swipe_frames, synth_hand

from gesturekit.gestures.base import RecognizerContext
from gesturekit.gestures.dynamic import DynamicRecognizer, SwipeConfig, WaveRecognizer


def run(frames, recognizer=None):
    rec = recognizer or DynamicRecognizer()
    ctx = RecognizerContext()
    events = []
    for state in frames:
        ctx.push(state)
        events += rec.update(ctx)
    return events


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_swipe_detected_in_each_direction(direction):
    events = run(swipe_frames(direction))
    assert [e.name for e in events] == [f"swipe_{direction}"]
    assert events[0].confidence > 0.5


def test_swipe_reports_useful_metadata():
    ev = run(swipe_frames("left"))[0]
    assert ev.kind == "dynamic"
    assert ev.data["distance"] > 1.0
    assert ev.data["straightness"] > 0.8


def test_still_hand_never_swipes():
    """v1's two-frame delta test fired on landmark noise. This must not."""
    assert run(jitter_frames(n=90, amplitude=0.006)) == []


@pytest.mark.parametrize("seed", range(8))
def test_no_false_swipes_across_noise_seeds(seed):
    assert run(jitter_frames(n=60, amplitude=0.008, seed=seed)) == []


def test_slow_drift_is_not_a_swipe():
    """Moving the hand slowly across the frame is repositioning, not a command."""
    frames = swipe_frames("right", n=60, duration=3.0, distance=0.45)
    assert run(frames) == []


def test_wobbly_path_is_rejected():
    """Big travel but a curvy path -- straightness must veto it."""
    frames = []
    for i in range(16):
        f = i / 15
        x = 0.3 + 0.4 * f
        y = 0.5 + 0.16 * np.sin(f * 9)
        frames.append(make_state(synth_hand(center=(x, y), scale=0.26),
                                 timestamp=1000 + f * 0.3))
    names = [e.name for e in run(frames)]
    assert "swipe_right" not in names


def test_curling_fingers_does_not_fake_a_swipe():
    """A v1 bug: the 21-point centroid shifted when fingers closed."""
    frames = []
    for i in range(20):
        c = i / 19.0
        frames.append(
            make_state(synth_hand(curls=(c, c, c, c, c), center=(0.5, 0.5), scale=0.28),
                       timestamp=1000 + i * 0.02)
        )
    assert run(frames) == []


def test_refractory_prevents_double_fire():
    rec = DynamicRecognizer(SwipeConfig(refractory=0.6))
    ctx = RecognizerContext()
    events = []
    for state in swipe_frames("right", n=24, duration=0.35):
        ctx.push(state)
        events += rec.update(ctx)
    assert len(events) == 1


def test_two_separate_swipes_both_fire():
    rec = DynamicRecognizer(SwipeConfig(refractory=0.4))
    ctx = RecognizerContext()
    events = []
    for state in swipe_frames("right", t0=1000.0):
        ctx.push(state)
        events += rec.update(ctx)
    for state in swipe_frames("left", t0=1002.0):
        ctx.push(state)
        events += rec.update(ctx)
    assert [e.name for e in events] == ["swipe_right", "swipe_left"]


def test_swipe_threshold_is_distance_invariant():
    """The same physical gesture must work near and far from the camera."""
    for scale, distance in ((0.18, 0.26), (0.40, 0.58)):
        frames = [
            make_state(
                synth_hand(center=(0.5 + distance * (i / 13 - 0.5), 0.5), scale=scale),
                timestamp=1000 + (i / 13) * 0.3,
            )
            for i in range(14)
        ]
        assert [e.name for e in run(frames)] == ["swipe_right"]


def test_short_history_is_safe():
    rec = DynamicRecognizer()
    ctx = RecognizerContext()
    for i in range(3):
        ctx.push(make_state(pose("open_palm"), timestamp=1000 + i * 0.03))
        assert rec.update(ctx) == []


@pytest.mark.parametrize("clockwise", [True, False])
def test_circle_detection(clockwise):
    rec = DynamicRecognizer(enable_circles=True)
    names = [e.name for e in run(circle_frames(clockwise=clockwise), rec)]
    assert names and names[-1] == ("circle_cw" if clockwise else "circle_ccw")


def test_circle_disabled_when_configured():
    rec = DynamicRecognizer(enable_circles=False)
    assert [e.name for e in run(circle_frames(), rec) if "circle" in e.name] == []


def test_jitter_is_not_a_circle():
    rec = DynamicRecognizer(enable_circles=True)
    assert run(jitter_frames(n=80, amplitude=0.01), rec) == []


# ---------------------------------------------------------------------------
# Wave
# ---------------------------------------------------------------------------

def test_wave_detected():
    rec = WaveRecognizer()
    ctx = RecognizerContext()
    events = []
    for i in range(40):
        t = i * 0.04
        x = 0.5 + 0.15 * np.sin(t * 11)     # ~1.2 palm widths peak-to-peak
        ctx.push(make_state(synth_hand(curls=(0, 0, 0, 0, 0), center=(x, 0.5), scale=0.26),
                            timestamp=1000 + t))
        events += rec.update(ctx)
    assert any(e.name == "wave" for e in events)


def test_wave_needs_an_open_hand():
    rec = WaveRecognizer()
    ctx = RecognizerContext()
    events = []
    for i in range(40):
        t = i * 0.04
        x = 0.5 + 0.15 * np.sin(t * 11)
        ctx.push(make_state(synth_hand(curls=(1, 1, 1, 1, 1), center=(x, 0.5), scale=0.26),
                            timestamp=1000 + t))
        events += rec.update(ctx)
    assert not any(e.name == "wave" for e in events)


def test_still_hand_does_not_wave():
    rec = WaveRecognizer()
    ctx = RecognizerContext()
    events = []
    for state in jitter_frames(n=60):
        ctx.push(state)
        events += rec.update(ctx)
    assert events == []

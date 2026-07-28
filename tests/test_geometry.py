"""Geometry invariance and correctness."""

from __future__ import annotations

import numpy as np
import pytest
from synth import pose, synth_hand

from gesturekit import geometry as geo


def test_landmark_indices_consistent():
    assert geo.FINGER_CHAINS.shape == (5, 4)
    assert list(geo.TIPS) == [4, 8, 12, 16, 20]
    assert len(geo.HAND_CONNECTIONS) == 21


def test_to_array_from_objects():
    class P:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    pts = geo.to_array([P(0.1, 0.2, 0.3), P(0.4, 0.5, 0.6)])
    assert pts.shape == (2, 3)
    assert pts.dtype == np.float32
    np.testing.assert_allclose(pts[1], [0.4, 0.5, 0.6], rtol=1e-6)


def test_to_array_handles_empty_and_arrays():
    assert geo.to_array([]).shape == (0, 3)
    src = np.zeros((21, 3), np.float64)
    assert geo.to_array(src).dtype == np.float32


def test_palm_size_scales_linearly():
    small = synth_hand(scale=0.2)
    big = synth_hand(scale=0.4)
    assert geo.palm_size(big) == pytest.approx(2 * geo.palm_size(small), rel=1e-4)


def test_normalize_is_translation_invariant():
    a = synth_hand(center=(0.3, 0.3))
    b = synth_hand(center=(0.7, 0.6))
    np.testing.assert_allclose(geo.normalize(a), geo.normalize(b), atol=1e-5)


def test_normalize_is_scale_invariant():
    a = synth_hand(scale=0.15)
    b = synth_hand(scale=0.55)
    np.testing.assert_allclose(geo.normalize(a), geo.normalize(b), atol=1e-4)


@pytest.mark.parametrize("angle", [-90, -45, -15, 15, 45, 90, 175])
def test_normalize_is_rotation_invariant(angle):
    a = synth_hand(rotation_deg=0)
    b = synth_hand(rotation_deg=angle)
    np.testing.assert_allclose(
        geo.normalize(a)[:, :2], geo.normalize(b)[:, :2], atol=1e-4
    )


def test_curl_ordering():
    straight = geo.finger_curl(synth_hand(curls=(0, 0, 0, 0, 0)))
    folded = geo.finger_curl(synth_hand(curls=(1, 1, 1, 1, 1)))
    assert np.all(folded > straight)
    assert straight.max() < 0.2
    assert folded[1:].min() > 0.5


def test_curl_is_monotonic_in_bend():
    values = [float(geo.finger_curl(synth_hand(curls=(0, c, 0, 0, 0)))[1])
              for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)


def test_fingers_extended_matches_pose():
    assert geo.fingers_extended(pose("open_palm")).sum() == 5
    assert geo.fingers_extended(pose("closed_fist")).sum() <= 1
    ext = geo.fingers_extended(pose("peace"))
    assert ext[1] and ext[2] and not ext[3] and not ext[4]


def test_hand_center_ignores_finger_curl():
    """The v1 bug: curling fingers moved the 'centre' and faked a swipe."""
    open_c = geo.hand_center(synth_hand(curls=(0, 0, 0, 0, 0)))
    fist_c = geo.hand_center(synth_hand(curls=(1, 1, 1, 1, 1)))
    assert float(np.linalg.norm(open_c - fist_c)) < 0.02


def test_pinch_distance_scale_invariant():
    near = geo.pinch_distance(synth_hand(scale=0.2))
    far = geo.pinch_distance(synth_hand(scale=0.5))
    assert near == pytest.approx(far, rel=0.02)


def test_pointing_direction_is_unit_and_correct():
    d = geo.pointing_direction(pose("point_up"))
    assert float(np.linalg.norm(d)) == pytest.approx(1.0, abs=1e-5)
    assert d[1] < -0.7                       # up = negative y
    assert geo.pointing_direction(pose("point_down"))[1] > 0.7
    assert geo.pointing_direction(pose("point_left"))[0] < -0.7
    assert geo.pointing_direction(pose("point_right"))[0] > 0.7


@pytest.mark.parametrize("angle", [-60, -30, 0, 30, 60])
def test_hand_roll_tracks_rotation(angle):
    # Measured relative to the hand's own resting roll: the middle knuckle is
    # not exactly above the wrist on a real (or realistic) hand, so there is a
    # small constant offset. What matters is that roll tracks rotation 1:1.
    baseline = geo.hand_roll(synth_hand(rotation_deg=0))
    rolled = geo.hand_roll(synth_hand(rotation_deg=angle))
    assert (rolled - baseline) == pytest.approx(angle, abs=2.0)


def test_bounding_box_within_unit_square():
    x0, y0, x1, y1 = geo.bounding_box(synth_hand())
    assert 0.0 <= x0 < x1 <= 1.0
    assert 0.0 <= y0 < y1 <= 1.0


def test_degenerate_input_does_not_crash():
    empty = np.zeros((0, 3), np.float32)
    assert geo.palm_size(empty) > 0
    assert geo.finger_curl(empty).shape == (5,)
    assert geo.hand_center(empty).shape == (2,)
    assert geo.normalize(empty).shape == (0, 3)
    assert geo.bounding_box(empty) == (0.0, 0.0, 0.0, 0.0)

"""Tests for 1-Euro Filter jitter suppression and high-speed tracking response."""

from __future__ import annotations

import math
import random

from vision_control.filters import LandmarkSmoother, OneEuroFilter, Point2DOneEuroFilter


def test_one_euro_suppresses_stationary_distant_jitter():
    rng = random.Random(42)
    f = OneEuroFilter(min_cutoff=1.1, beta=1.5)

    raw_vals = []
    filt_vals = []
    for i in range(90):
        t = i / 30.0
        noisy = 0.50 + rng.gauss(0.0, 0.008)  # ~10px jitter at 1280px
        raw_vals.append(noisy)
        filt_vals.append(f.filter(noisy, t))

    # Ignore first 10 warmup samples
    raw_var = sum((x - 0.50) ** 2 for x in raw_vals[10:]) / len(raw_vals[10:])
    filt_var = sum((x - 0.50) ** 2 for x in filt_vals[10:]) / len(filt_vals[10:])
    assert filt_var < raw_var * 0.25


def test_one_euro_tracks_fast_swipe_step_cleanly():
    f = OneEuroFilter(min_cutoff=1.2, beta=1.8)
    for i in range(10):
        f.filter(0.2, i / 30.0)
    # Rapid flick from 0.2 -> 0.8 over 5 frames
    val = 0.2
    for j in range(1, 6):
        target = 0.2 + 0.6 * (j / 5.0)
        val = f.filter(target, (10 + j) / 30.0)
    assert val > 0.68


def test_gap_resets_without_stale_lag():
    p2 = Point2DOneEuroFilter()
    p2.filter(0.1, 0.1, 1.0)
    # Gap > 0.45s snaps immediately to new coordinate
    sx, sy = p2.filter(0.85, 0.75, 2.0)
    assert math.isclose(sx, 0.85, abs_tol=1e-5)
    assert math.isclose(sy, 0.75, abs_tol=1e-5)


def test_landmark_smoother_21_points():
    sm = LandmarkSmoother(num_landmarks=21)
    pts = [(0.5, 0.5, 0.0) for _ in range(21)]
    out = sm.filter(pts, 0.033)
    assert len(out) == 21
    assert out[8] == (0.5, 0.5, 0.0)

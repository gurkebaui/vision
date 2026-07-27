"""Filter behaviour: jitter rejection without lag on fast motion."""

from __future__ import annotations

import numpy as np
import pytest

from gesturekit.filters import EMA, Hysteresis, LandmarkFilter, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f(0.7, 0.0) == pytest.approx(0.7)


def test_reduces_jitter_on_static_signal():
    rng = np.random.default_rng(1)
    f = OneEuroFilter(min_cutoff=0.8, beta=0.01)
    raw, filt = [], []
    for i in range(200):
        x = 1.0 + rng.normal(0, 0.02)
        raw.append(x)
        filt.append(f(x, i / 60.0))
    # Ignore the warm-up while the filter converges.
    assert np.std(filt[40:]) < np.std(raw[40:]) * 0.4


def test_tracks_fast_motion_without_large_lag():
    """The whole point of One Euro: heavy smoothing at rest, light in motion."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.7)
    out = 0.0
    for i in range(60):
        t = i / 60.0
        out = f(t * 10.0, t)          # ramp at 10 units/second
    assert abs(out - 10.0) < 0.6


def test_array_filtering_preserves_shape():
    f = OneEuroFilter()
    for i in range(10):
        out = f(np.random.rand(21, 3).astype(np.float32), i / 60)
    assert out.shape == (21, 3)


def test_landmark_filter_smooths_and_resets():
    lf = LandmarkFilter()
    base = np.zeros((21, 3), np.float32)
    for i in range(30):
        noisy = base + np.random.normal(0, 0.01, (21, 3)).astype(np.float32)
        out = lf(noisy, i / 60)
    assert float(np.abs(out).max()) < 0.02
    lf.reset()
    spike = np.ones((21, 3), np.float32)
    np.testing.assert_allclose(lf(spike, 100.0), spike)


def test_zero_and_negative_dt_do_not_explode():
    f = OneEuroFilter()
    f(1.0, 5.0)
    assert np.isfinite(f(1.1, 5.0))
    assert np.isfinite(f(1.2, 4.0))


def test_hysteresis_prevents_flicker():
    """A value hovering at the threshold must not toggle every frame."""
    h = Hysteresis(on_at=0.30, off_at=0.42, invert=True)
    states = [h.update(v) for v in (0.50, 0.35, 0.29, 0.35, 0.41, 0.45, 0.35)]
    assert states == [False, False, True, True, True, False, False]


def test_hysteresis_normal_direction():
    h = Hysteresis(on_at=0.7, off_at=0.5)
    assert [h.update(v) for v in (0.4, 0.6, 0.75, 0.6, 0.45)] == [
        False, False, True, True, False
    ]


def test_ema():
    e = EMA(0.5)
    assert e.update(10) == 10
    assert e.update(20) == pytest.approx(15)

"""Temporal filters for jitter-free tracking.

MediaPipe landmarks jitter by a few pixels even for a perfectly still hand.
v1 fed those raw coordinates straight into threshold comparisons, which is why
its swipe detector fired on noise and its cursor would have been unusable.

Two tools live here:

``OneEuroFilter``
    Adaptive low-pass filter (Casiez, Roussel & Vogel, CHI 2012).  It filters
    hard when the hand is slow -- killing jitter -- and barely at all when the
    hand is fast, so quick motions do not lag behind.  A fixed EMA cannot do
    both at once.

``Hysteresis``
    Schmitt trigger for booleans.  Prevents a value sitting exactly on a
    threshold from flickering on/off every frame.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

Number = Union[float, np.ndarray]


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Adaptive low-pass filter that works on scalars or NumPy arrays.

    Args:
        min_cutoff: Cutoff frequency at rest (Hz).  Lower = smoother but laggier.
        beta: Speed coefficient.  Higher = less lag while moving fast.
        d_cutoff: Cutoff for the derivative estimate.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[Number] = None
        self._dx_prev: Optional[Number] = None
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x: Number, t: float) -> Number:
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32, copy=True)

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1.0 / 60.0
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        if isinstance(cutoff, np.ndarray):
            tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
            a = 1.0 / (1.0 + tau / dt)
        else:
            a = _alpha(float(cutoff), dt)

        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class LandmarkFilter:
    """One Euro filter applied to a whole ``(21, 3)`` landmark array."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05):
        self._f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def reset(self) -> None:
        self._f.reset()

    def __call__(self, pts: np.ndarray, t: float) -> np.ndarray:
        if pts.size == 0:
            return pts
        out = self._f(pts, t)
        return out if isinstance(out, np.ndarray) else pts


class Hysteresis:
    """Boolean latch with separate on/off thresholds (Schmitt trigger).

    ``on_at`` must be the "more extreme" value.  For a pinch (small distance
    means pinched) pass ``on_at < off_at`` and set ``invert=True``.
    """

    __slots__ = ("on_at", "off_at", "invert", "state")

    def __init__(self, on_at: float, off_at: float, invert: bool = False, initial: bool = False):
        self.on_at = float(on_at)
        self.off_at = float(off_at)
        self.invert = bool(invert)
        self.state = bool(initial)

    def reset(self, state: bool = False) -> None:
        self.state = bool(state)

    def update(self, value: float) -> bool:
        if self.invert:
            if self.state:
                if value > self.off_at:
                    self.state = False
            elif value < self.on_at:
                self.state = True
        else:
            if self.state:
                if value < self.off_at:
                    self.state = False
            elif value > self.on_at:
                self.state = True
        return self.state


class EMA:
    """Plain exponential moving average, for display values like FPS."""

    __slots__ = ("alpha", "value")

    def __init__(self, alpha: float = 0.1, initial: Optional[float] = None):
        self.alpha = float(alpha)
        self.value = initial

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

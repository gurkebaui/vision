"""
1-Euro Filter (Casiez, Roussel & Vogel, CHI 2012) for jitter-free, low-latency
hand landmark and air-mouse cursor smoothing at long camera distances (1m–5m+).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def _smoothing_factor(dt: float, cutoff_hz: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-4))
    return 1.0 / (1.0 + tau / max(dt, 1e-4))


class OneEuroFilter:
    """Adaptive first-order low-pass filter balancing steady precision vs high-speed lag."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 1.6,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = float(t)
            return float(x)

        dt = max(float(t) - self._t_prev, 1e-4)
        # If there was a long tracking gap (>0.45s), snap cleanly without stale lag
        if dt > 0.45:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = float(t)
            return float(x)

        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (float(x) - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(dt, cutoff)
        x_hat = a * float(x) + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = float(t)
        return x_hat


class Point2DOneEuroFilter:
    """2D cursor / laser-pointer filter with optional micro-deadzone suppression."""

    def __init__(
        self,
        min_cutoff: float = 1.15,
        beta: float = 1.45,
        d_cutoff: float = 1.0,
        deadzone: float = 0.0012,
    ) -> None:
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.deadzone = float(deadzone)
        self._last_out: Optional[Tuple[float, float]] = None

    def update_params(self, min_cutoff: float, beta: float) -> None:
        self.fx.min_cutoff = float(min_cutoff)
        self.fy.min_cutoff = float(min_cutoff)
        self.fx.beta = float(beta)
        self.fy.beta = float(beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self._last_out = None

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        sx = self.fx.filter(x, t)
        sy = self.fy.filter(y, t)
        if self._last_out is not None and self.deadzone > 0.0:
            if math.hypot(sx - self._last_out[0], sy - self._last_out[1]) < self.deadzone:
                return self._last_out
        self._last_out = (sx, sy)
        return sx, sy


class LandmarkSmoother:
    """Smooths 21 3D hand landmarks independently using 1-Euro filters."""

    def __init__(
        self,
        num_landmarks: int = 21,
        min_cutoff: float = 1.4,
        beta: float = 1.8,
        d_cutoff: float = 1.0,
    ) -> None:
        self.num_landmarks = num_landmarks
        self._filters: List[Tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter]] = [
            (
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff),
                OneEuroFilter(min_cutoff=min_cutoff * 0.85, beta=beta, d_cutoff=d_cutoff),
            )
            for _ in range(num_landmarks)
        ]

    def reset(self) -> None:
        for fx, fy, fz in self._filters:
            fx.reset()
            fy.reset()
            fz.reset()

    def filter(
        self, landmarks: Sequence[Tuple[float, float, float]], t: float
    ) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        for idx, (x, y, z) in enumerate(landmarks):
            if idx < self.num_landmarks:
                fx, fy, fz = self._filters[idx]
                out.append((fx.filter(x, t), fy.filter(y, t), fz.filter(z, t)))
            else:
                out.append((float(x), float(y), float(z)))
        return out

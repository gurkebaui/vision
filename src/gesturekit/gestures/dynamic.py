"""Motion gestures: swipes, circles, waves.

v1's swipe test compared the hand centre between two consecutive frames and
fired if the delta exceeded 0.05.  That is unreliable for three reasons: a
single-frame delta is dominated by landmark noise, the "centre" moved when
fingers curled, and there was no check that the motion was actually a straight
purposeful stroke rather than a wobble.

This implementation analyses the whole recent trajectory:

* **Displacement** must exceed a distance measured in palm widths, so the
  threshold means the same thing near and far from the camera.
* **Straightness** = net displacement / path length.  A wobble has a long path
  and small net displacement, so it scores low and is rejected.
* **Peak speed** must clear a floor, so slow drifting never counts.
* A **refractory period** after each fire prevents one physical swipe from
  emitting three events as the hand decelerates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..types import GestureEvent
from .base import Recognizer, RecognizerContext


@dataclass
class SwipeConfig:
    min_distance: float = 1.15      # palm widths of net travel
    min_speed: float = 2.2          # palm widths / second, peak
    max_duration: float = 0.55      # seconds; longer = a drift, not a swipe
    min_straightness: float = 0.82  # net / path length
    axis_ratio: float = 1.7         # dominant axis must beat the other by this
    refractory: float = 0.55        # seconds of silence after firing
    require_open_hand: bool = False


class DynamicRecognizer(Recognizer):
    """Detects directional swipes and circular motions."""

    name = "dynamic"

    def __init__(self, config: Optional[SwipeConfig] = None, enable_circles: bool = True):
        self.config = config or SwipeConfig()
        self.enable_circles = enable_circles
        self._last_fire = 0.0
        self._last_circle = 0.0

    def reset(self) -> None:
        self._last_fire = 0.0
        self._last_circle = 0.0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _straightness(path: np.ndarray) -> float:
        if path.shape[0] < 3:
            return 0.0
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total = float(seg.sum())
        if total < 1e-6:
            return 0.0
        net = float(np.linalg.norm(path[-1] - path[0]))
        return net / total

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        hand = ctx.current
        if hand is None or len(ctx.states) < 5:
            return []

        cfg = self.config
        now = hand.timestamp
        if now - self._last_fire < cfg.refractory:
            return []

        path = ctx.trajectory(cfg.max_duration)
        stamps = ctx.timestamps(cfg.max_duration)
        if path.shape[0] < 5:
            return []

        scale = max(hand.palm_size, 1e-6)
        disp = (path[-1] - path[0]) / scale
        dist = float(np.linalg.norm(disp))
        duration = float(stamps[-1] - stamps[0])
        if duration <= 1e-3:
            return []

        events: list[GestureEvent] = []

        # -- straight swipes ------------------------------------------------
        peak_speed = 0.0
        if path.shape[0] >= 2:
            d = np.linalg.norm(np.diff(path, axis=0), axis=1) / scale
            dt = np.maximum(np.diff(stamps), 1e-3)
            peak_speed = float((d / dt).max())

        straight = self._straightness(path)
        if (
            dist >= cfg.min_distance
            and peak_speed >= cfg.min_speed
            and straight >= cfg.min_straightness
        ):
            ax, ay = abs(float(disp[0])), abs(float(disp[1]))
            direction = None
            if ax > ay * cfg.axis_ratio:
                direction = "swipe_right" if disp[0] > 0 else "swipe_left"
            elif ay > ax * cfg.axis_ratio:
                direction = "swipe_down" if disp[1] > 0 else "swipe_up"

            if direction and (not cfg.require_open_hand or hand.n_extended >= 3):
                conf = float(
                    np.clip(
                        0.45 * min(dist / (cfg.min_distance * 1.8), 1.0)
                        + 0.30 * min(peak_speed / (cfg.min_speed * 1.8), 1.0)
                        + 0.25 * straight,
                        0.0,
                        1.0,
                    )
                )
                self._last_fire = now
                events.append(
                    GestureEvent(
                        name=direction,
                        confidence=conf,
                        hand=hand.handedness,
                        timestamp=now,
                        kind="dynamic",
                        data={
                            "distance": dist,
                            "speed": peak_speed,
                            "straightness": straight,
                            "duration": duration,
                        },
                    )
                )
                return events

        # -- circles ---------------------------------------------------------
        if self.enable_circles and now - self._last_circle > 1.0:
            circle = self._detect_circle(ctx, scale)
            if circle:
                name, conf, meta = circle
                self._last_circle = now
                self._last_fire = now
                events.append(
                    GestureEvent(
                        name=name,
                        confidence=conf,
                        hand=hand.handedness,
                        timestamp=now,
                        kind="dynamic",
                        data=meta,
                    )
                )
        return events

    def _detect_circle(self, ctx: RecognizerContext, scale: float):
        """Total signed turning angle ~ +/-2pi with a stable radius."""
        path = ctx.trajectory(1.4)
        if path.shape[0] < 14:
            return None

        centroid = path.mean(axis=0)
        rel = path - centroid
        radii = np.linalg.norm(rel, axis=1) / scale
        mean_r = float(radii.mean())
        if mean_r < 0.55 or mean_r > 4.0:
            return None
        if float(radii.std()) / max(mean_r, 1e-6) > 0.38:   # not round enough
            return None

        ang = np.arctan2(rel[:, 1], rel[:, 0])
        d = np.diff(ang)
        d = (d + math.pi) % (2 * math.pi) - math.pi          # unwrap to (-pi, pi]
        total = float(d.sum())
        if abs(total) < 1.75 * math.pi:
            return None

        name = "circle_cw" if total > 0 else "circle_ccw"    # +y is down
        conf = float(np.clip(abs(total) / (2 * math.pi), 0.0, 1.0))
        return name, conf, {"turn": total, "radius": mean_r}


class WaveRecognizer(Recognizer):
    """Detects a side-to-side wave (handy as a wake/sleep toggle)."""

    name = "wave"

    def __init__(self, min_reversals: int = 3, window: float = 1.5, min_amplitude: float = 0.85):
        self.min_reversals = int(min_reversals)
        self.window = float(window)
        self.min_amplitude = float(min_amplitude)
        self._last = 0.0

    def reset(self) -> None:
        self._last = 0.0

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        hand = ctx.current
        if hand is None or hand.timestamp - self._last < 1.5:
            return []
        path = ctx.trajectory(self.window)
        if path.shape[0] < 12:
            return []

        x = path[:, 0] / max(hand.palm_size, 1e-6)
        amp = float(x.max() - x.min())
        if amp < self.min_amplitude:
            return []

        dx = np.diff(x)
        dx = dx[np.abs(dx) > 0.015]
        if dx.size < 4:
            return []
        reversals = int((np.diff(np.sign(dx)) != 0).sum())
        if reversals < self.min_reversals:
            return []
        if hand.n_extended < 3:      # a wave is done with an open hand
            return []

        self._last = hand.timestamp
        return [
            GestureEvent(
                name="wave",
                confidence=float(np.clip(0.5 + 0.1 * reversals, 0.0, 1.0)),
                hand=hand.handedness,
                timestamp=hand.timestamp,
                kind="dynamic",
                data={"reversals": reversals, "amplitude": amp},
            )
        ]

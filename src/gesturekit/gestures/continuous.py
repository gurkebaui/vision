"""Continuous / streaming gestures.

These do not fire once -- they stream state every frame for as long as the
pose is held.  They power the features v1 had no answer for: an actual mouse
cursor, click-and-drag, scrolling, two-hand zoom, and a laser pointer.

Each detector is a small state machine with hysteresis on entry/exit so a
borderline pinch cannot rattle the button on and off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import geometry as geo
from ..filters import Hysteresis, OneEuroFilter
from ..types import GestureEvent, Handedness, HandState
from .base import Recognizer, RecognizerContext


@dataclass
class PointerConfig:
    """Cursor mapping parameters.

    The active region is a sub-rectangle of the frame, expanded to the full
    screen.  Without it you would have to reach the physical edge of the
    camera's view -- where tracking is worst -- to reach the screen edge.
    """

    region_x0: float = 0.18
    region_y0: float = 0.14
    region_x1: float = 0.82
    region_y1: float = 0.78
    smoothing: float = 1.0      # One Euro min_cutoff; lower = smoother
    beta: float = 0.012         # higher = snappier on fast moves
    pinch_on: float = 0.32
    pinch_off: float = 0.42
    drag_hold: float = 0.28     # seconds pinched before it becomes a drag


class PointerRecognizer(Recognizer):
    """Index-finger cursor with pinch-to-click and pinch-hold-to-drag."""

    name = "pointer"

    def __init__(self, config: Optional[PointerConfig] = None, enabled: bool = True):
        self.config = config or PointerConfig()
        self.enabled = enabled
        self._filter = OneEuroFilter(min_cutoff=self.config.smoothing, beta=self.config.beta)
        self._pinch = Hysteresis(self.config.pinch_on, self.config.pinch_off, invert=True)
        self._pinch_start: Optional[float] = None
        self._dragging = False
        self._click_pending = False

    def reset(self) -> None:
        self._filter.reset()
        self._pinch.reset(False)
        self._pinch_start = None
        self._dragging = False
        self._click_pending = False

    def _map(self, pt: np.ndarray) -> np.ndarray:
        c = self.config
        x = (float(pt[0]) - c.region_x0) / max(c.region_x1 - c.region_x0, 1e-6)
        y = (float(pt[1]) - c.region_y0) / max(c.region_y1 - c.region_y0, 1e-6)
        return np.array([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)], dtype=np.float32)

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        hand = ctx.current
        if not self.enabled or hand is None:
            # Hand lost mid-drag: release immediately. A stuck mouse button is
            # far worse than a missed drag.
            if self._dragging:
                self._dragging = False
                self._pinch.reset(False)
                self._pinch_start = None
                self._filter.reset()
                return [self._event_static("drag_end", 1.0, {})]
            return []

        # The cursor follows the index fingertip while the finger is out.
        # Pinching necessarily bends the index, so an active pinch/drag keeps
        # tracking alive -- otherwise the pointer would cut out at the exact
        # moment you click, and a drag could never move anywhere.
        pinching = self._pinch.state or hand.pinch < self.config.pinch_off
        if hand.curls[1] > 0.55 and not pinching:
            self._filter.reset()
            if self._dragging:
                self._dragging = False
                self._pinch.reset(False)
                self._pinch_start = None
                return [self._event("drag_end", 1.0, hand, {})]
            return []

        tip = hand.points[geo.INDEX_TIP, :2]
        smooth = self._filter(tip.astype(np.float32), hand.timestamp)
        pos = self._map(np.asarray(smooth, dtype=np.float32))

        events: list[GestureEvent] = [
            self._event("cursor_move", 1.0, hand, {"x": float(pos[0]), "y": float(pos[1])})
        ]

        pinched = self._pinch.update(hand.pinch)
        if pinched and self._pinch_start is None:
            self._pinch_start = hand.timestamp
            self._click_pending = True
        elif pinched and self._pinch_start is not None:
            held = hand.timestamp - self._pinch_start
            if held >= self.config.drag_hold and not self._dragging:
                self._dragging = True
                self._click_pending = False
                events.append(
                    self._event("drag_start", 0.95, hand, {"x": float(pos[0]), "y": float(pos[1])})
                )
        elif not pinched and self._pinch_start is not None:
            held = hand.timestamp - self._pinch_start
            self._pinch_start = None
            if self._dragging:
                self._dragging = False
                events.append(
                    self._event("drag_end", 0.95, hand, {"x": float(pos[0]), "y": float(pos[1])})
                )
            elif self._click_pending and held < self.config.drag_hold:
                events.append(
                    self._event("click", 0.9, hand, {"x": float(pos[0]), "y": float(pos[1])})
                )
            self._click_pending = False

        return events

    @staticmethod
    def _event(name: str, conf: float, hand: Optional[HandState], data: dict) -> GestureEvent:
        return GestureEvent(
            name=name,
            confidence=conf,
            hand=hand.handedness if hand else Handedness.UNKNOWN,
            timestamp=hand.timestamp if hand else 0.0,
            kind="continuous",
            data=data,
        )

    @staticmethod
    def _event_static(name: str, conf: float, data: dict) -> GestureEvent:
        """Event for when no hand is present (e.g. a forced drag release)."""
        return GestureEvent(
            name=name,
            confidence=conf,
            hand=Handedness.UNKNOWN,
            timestamp=time.perf_counter(),
            kind="continuous",
            data=data,
        )


class ScrollRecognizer(Recognizer):
    """Two-finger (index+middle) vertical/horizontal scrolling."""

    name = "scroll"

    def __init__(self, sensitivity: float = 1.0, deadzone: float = 0.35, enabled: bool = True):
        self.sensitivity = float(sensitivity)
        self.deadzone = float(deadzone)
        self.enabled = enabled
        self._anchor: Optional[np.ndarray] = None
        self._active = False

    def reset(self) -> None:
        self._anchor = None
        self._active = False

    @staticmethod
    def is_scroll_pose(h: HandState) -> bool:
        return bool(
            h.curls[1] < 0.4 and h.curls[2] < 0.4 and h.curls[3] > 0.55 and h.curls[4] > 0.55
        )

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        hand = ctx.current
        if not self.enabled or hand is None or not self.is_scroll_pose(hand):
            self._anchor = None
            self._active = False
            return []

        mid = (hand.points[geo.INDEX_TIP, :2] + hand.points[geo.MIDDLE_TIP, :2]) / 2.0
        if self._anchor is None:
            self._anchor = mid.copy()
            return []

        delta = (mid - self._anchor) / max(hand.palm_size, 1e-6)
        if float(np.linalg.norm(delta)) < self.deadzone:
            return []

        self._anchor = mid.copy()
        dx = float(delta[0]) * self.sensitivity
        dy = float(delta[1]) * self.sensitivity
        return [
            GestureEvent(
                name="scroll",
                confidence=0.9,
                hand=hand.handedness,
                timestamp=hand.timestamp,
                kind="continuous",
                data={"dx": dx, "dy": -dy},  # screen up = positive scroll
            )
        ]


class ZoomRecognizer(Recognizer):
    """Two-hand pinch-to-zoom: distance between the hands drives the level."""

    name = "zoom"

    def __init__(self, sensitivity: float = 1.0, deadzone: float = 0.18, enabled: bool = True):
        self.sensitivity = float(sensitivity)
        self.deadzone = float(deadzone)
        self.enabled = enabled
        self._baseline: Optional[float] = None

    def reset(self) -> None:
        self._baseline = None

    def update_two(self, hands: list[HandState]) -> list[GestureEvent]:
        if not self.enabled or len(hands) < 2:
            self._baseline = None
            return []

        a, b = hands[0], hands[1]
        # Both hands must be pinching, which makes the gesture intentional.
        if a.pinch > 0.4 or b.pinch > 0.4:
            self._baseline = None
            return []

        scale = max((a.palm_size + b.palm_size) / 2.0, 1e-6)
        dist = float(np.linalg.norm(a.center - b.center)) / scale
        if self._baseline is None:
            self._baseline = dist
            return []

        ratio = dist / max(self._baseline, 1e-6)
        if abs(ratio - 1.0) < self.deadzone:
            return []

        self._baseline = dist
        return [
            GestureEvent(
                name="zoom_in" if ratio > 1.0 else "zoom_out",
                confidence=0.9,
                hand=a.handedness,
                timestamp=a.timestamp,
                kind="continuous",
                data={"ratio": ratio, "steps": max(1, int(abs(ratio - 1.0) / 0.18))},
            )
        ]

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        return []


class ContinuousRecognizer(Recognizer):
    """Bundles pointer, scroll and zoom behind one interface."""

    name = "continuous"

    def __init__(
        self,
        pointer: Optional[PointerRecognizer] = None,
        scroll: Optional[ScrollRecognizer] = None,
        zoom: Optional[ZoomRecognizer] = None,
    ):
        self.pointer = pointer or PointerRecognizer()
        self.scroll = scroll or ScrollRecognizer()
        self.zoom = zoom or ZoomRecognizer()

    def reset(self) -> None:
        self.pointer.reset()
        self.scroll.reset()
        self.zoom.reset()

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        # Scrolling wins over pointing. The two poses overlap (both have the
        # index finger out), so the decision is made on the *pose*, not on
        # whether scroll happened to emit this frame -- otherwise the first
        # few frames of a scroll would still jump the cursor.
        hand = ctx.current
        if hand is not None and self.scroll.enabled and self.scroll.is_scroll_pose(hand):
            events = self.scroll.update(ctx)
            self.pointer.reset()
            return events

        self.scroll.reset()
        return self.pointer.update(ctx)

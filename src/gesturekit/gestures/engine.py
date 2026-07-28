"""Gesture engine: fans hands out to recognisers and arbitrates the results.

Also implements **arming** -- the modernised version of the original
"lift a finger, then you have 3 seconds" idea.

v1's arming was effectively broken: it required a fist in the recent buffer,
then only ever set the flag from ``point_up``, and the 3-second window was
checked against a variable initialised to the integer ``1``, so the very first
comparison used a 1970 timestamp.  Here arming is an explicit state machine
with a configurable trigger gesture, a real countdown, visible feedback, and
the option to switch it off entirely for always-on control.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..types import GestureEvent, Handedness, HandState
from .base import RecognizerContext
from .continuous import (
    ContinuousRecognizer,
    PointerConfig,
    PointerRecognizer,
    ScrollRecognizer,
    ZoomRecognizer,
)
from .dynamic import DynamicRecognizer, SwipeConfig, WaveRecognizer
from .static import StaticRecognizer


@dataclass
class GestureEngineConfig:
    # arming
    require_arm: bool = False
    arm_gesture: str = "point_up"
    arm_window: float = 3.0
    disarm_gesture: str = "closed_fist"

    # static
    static_min_frames: int = 4
    static_min_score: float = 0.62
    static_margin: float = 0.10
    use_canned: bool = True

    # dynamic
    enable_swipes: bool = True
    enable_circles: bool = True
    enable_wave: bool = True
    swipe: SwipeConfig = field(default_factory=SwipeConfig)

    # continuous
    enable_pointer: bool = False
    enable_scroll: bool = False
    enable_zoom: bool = False
    pointer: PointerConfig = field(default_factory=PointerConfig)
    scroll_sensitivity: float = 1.0
    #: Frame aspect ratio, mirrored from TrackerConfig so the pointer can map
    #: aspect-corrected landmarks back onto the screen.
    aspect: float = 1.0

    # global
    cooldown: float = 0.45           # per-gesture repeat suppression
    global_cooldown: float = 0.12    # any-gesture repeat suppression
    history: int = 48


class GestureEngine:
    """Owns per-hand context and every recogniser."""

    def __init__(self, config: Optional[GestureEngineConfig] = None):
        self.config = config or GestureEngineConfig()
        cfg = self.config

        self._contexts: dict[str, RecognizerContext] = {}
        self._static: dict[str, StaticRecognizer] = {}
        self._dynamic: dict[str, DynamicRecognizer] = {}
        self._wave: dict[str, WaveRecognizer] = {}

        self._continuous = ContinuousRecognizer(
            pointer=PointerRecognizer(cfg.pointer, enabled=cfg.enable_pointer,
                                      aspect=cfg.aspect),
            scroll=ScrollRecognizer(cfg.scroll_sensitivity, enabled=cfg.enable_scroll),
            zoom=ZoomRecognizer(enabled=cfg.enable_zoom),
        )

        self._last_by_name: dict[str, float] = {}
        self._last_any = 0.0

        self.armed = not cfg.require_arm
        self.armed_until = 0.0
        self._arm_latch = False

    # -- helpers -----------------------------------------------------------
    def _key(self, hand: HandState, idx: int) -> str:
        return hand.handedness.value if hand.handedness.value != "Unknown" else f"hand{idx}"

    def _ctx_for(self, key: str) -> RecognizerContext:
        ctx = self._contexts.get(key)
        if ctx is None:
            cfg = self.config
            ctx = RecognizerContext(maxlen=cfg.history)
            ctx.states = deque(maxlen=cfg.history)
            self._contexts[key] = ctx
            self._static[key] = StaticRecognizer(
                min_frames=cfg.static_min_frames,
                min_score=cfg.static_min_score,
                margin=cfg.static_margin,
                use_canned=cfg.use_canned,
            )
            self._dynamic[key] = DynamicRecognizer(cfg.swipe, enable_circles=cfg.enable_circles)
            self._wave[key] = WaveRecognizer()
        return ctx

    # -- arming ------------------------------------------------------------
    def _update_arming(self, events: list[GestureEvent], now: float) -> list[GestureEvent]:
        cfg = self.config
        if not cfg.require_arm:
            self.armed = True
            return events

        passed: list[GestureEvent] = []

        # Expire an old window first, so a stale window can never pass events.
        if self.armed and now > self.armed_until:
            self.armed = False
            self.armed_until = 0.0
            if self._arm_latch:
                self._arm_latch = False
                passed.append(GestureEvent("arm_expired", 1.0, Handedness.UNKNOWN, now, "meta", {}))

        for ev in events:
            if ev.kind == "continuous":
                passed.append(ev)          # cursor/scroll are never gated
                continue

            if ev.name == cfg.arm_gesture and ev.kind == "static":
                self.armed = True
                self.armed_until = now + cfg.arm_window
                self._arm_latch = True
                passed.append(
                    GestureEvent(
                        "armed", ev.confidence, ev.hand, now, "meta",
                        {"window": cfg.arm_window},
                    )
                )
                continue

            if cfg.disarm_gesture and ev.name == cfg.disarm_gesture and self.armed:
                self.armed = False
                self.armed_until = 0.0
                self._arm_latch = False
                passed.append(GestureEvent("disarmed", ev.confidence, ev.hand, now, "meta", {}))
                continue

            if self.armed and now <= self.armed_until:
                passed.append(ev)

        return passed

    @property
    def arm_remaining(self) -> float:
        if not self.config.require_arm:
            return float("inf")
        return max(0.0, self.armed_until - time.perf_counter())

    # -- main --------------------------------------------------------------
    def process(self, hands: list[HandState], now: Optional[float] = None) -> list[GestureEvent]:
        now = time.perf_counter() if now is None else now
        cfg = self.config
        raw: list[GestureEvent] = []

        seen: set[str] = set()
        for i, hand in enumerate(hands):
            key = self._key(hand, i)
            seen.add(key)
            ctx = self._ctx_for(key)
            ctx.push(hand)

            raw.extend(self._static[key].update(ctx))
            if cfg.enable_swipes or cfg.enable_circles:
                raw.extend(self._dynamic[key].update(ctx))
            if cfg.enable_wave:
                raw.extend(self._wave[key].update(ctx))

        # Continuous gestures track the primary (first) hand only.
        if hands:
            primary = self._key(hands[0], 0)
            raw.extend(self._continuous.update(self._contexts[primary]))
        if cfg.enable_zoom and len(hands) >= 2:
            raw.extend(self._continuous.zoom.update_two(hands))

        # Hands that vanished this frame: let their stabilisers decay so a
        # gesture cannot stay "active" after the hand has left the frame.
        for key in self._contexts:
            if key not in seen:
                self._static[key].update(RecognizerContext())
        if not hands:
            self._continuous.reset()

        gated = self._update_arming(raw, now)
        return self._apply_cooldowns(gated, now)

    def _apply_cooldowns(self, events: list[GestureEvent], now: float) -> list[GestureEvent]:
        """Rate-limit discrete gestures.

        Two cooldowns with different jobs:

        * per-gesture -- stops one held pose from repeating an action;
        * global -- stops a burst of *different* gestures firing back to back
          (which usually means the recogniser is confused, not that the user
          wanted five commands at once).

        The global cooldown is keyed on the previous *frame*, so two hands
        gesturing simultaneously are not treated as a burst.
        """
        cfg = self.config
        out: list[GestureEvent] = []
        for ev in events:
            # Streaming gestures must never be rate-limited or the cursor stutters.
            if ev.kind in ("continuous", "meta"):
                out.append(ev)
                continue
            last = self._last_by_name.get(ev.name, 0.0)
            if now - last < cfg.cooldown:
                continue
            if self._last_any < now and (now - self._last_any) < cfg.global_cooldown:
                continue
            self._last_by_name[ev.name] = now
            out.append(ev)
        if out:
            self._last_any = now
        return out

    def scores_for(self, hand_key: str = "Right") -> dict:
        rec = self._static.get(hand_key)
        return rec.last_scores if rec else {}

    def any_scores(self) -> dict:
        for rec in self._static.values():
            if rec.last_scores:
                return rec.last_scores
        return {}

    def reset(self) -> None:
        for ctx in self._contexts.values():
            ctx.clear()
        for r in self._static.values():
            r.reset()
        for r in self._dynamic.values():
            r.reset()
        for r in self._wave.values():
            r.reset()
        self._continuous.reset()
        self._last_by_name.clear()
        self._last_any = 0.0
        self.armed = not self.config.require_arm
        self.armed_until = 0.0
